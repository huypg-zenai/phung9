# kopt-topkbf (class A, bitwise): DFlash verify without the fp32 logits copy and the `/ T` pass

Target: SGLang @1cf2b8c54 (`/opt/sglang-env/lib/python3.11/site-packages/sglang`), DFlash coder,
vocab 248320, verify M = bs x 8 rows, seeded sampling (temp 0.7, top_k 20). Works with or without the topk20 patch
(`opt/v4/topk20`) and the seed patch (`opt/v3/seed`), applied in any order.

## What
Stock verify chain per step (M = 512):

| step | where | traffic per logit |
|---|---|---|
| lm_head matmul -> bf16 logits | `logits_processor._get_logits` | write 2 B |
| `_copy_logits_to_buffer`: fp32 buffer `copy_` | `logits_processor` | read 2 B + write 4 B |
| `scaled_logits = logits / T` | `dflash_utils.build_dflash_verify_target_probs` | read 4 B + write 4 B |
| top-20 (torch.topk or topk20) | same | read 4 B |

With `SGL_TOPKBF=1`, on target-verify forwards of a DFLASH server:
- The lm_head matmul writes bf16 directly into a bf16 view over the first half of the same shared fp32 logits
  buffer the stock copy would fill. Under CUDA graphs that is the same static address and no new memory. In eager
  runs there is no buffer, so the matmul output tensor is kept as it is. The fp32 copy is skipped.
- The target-probs builder calls `topk_scaled_bf16`: the topk20 two-stage kernel reading bf16 and dividing by the
  row temperature in registers with an inline-PTX `div.rn.f32` (IEEE, no `.ftz`). Then it runs the same mask / softmax / top-p / scatter code as
  stock. No `scaled_logits` tensor is materialized on the sparse path.

Why not just keep the matmul's own bf16 output under CUDA graphs: each captured decode batch size would keep its
own `[bs*8, V]` bf16 tensor alive in the graph pool. The served config captures 51 sizes up to bs 256, which is
about 18 GB.

## Why bitwise
- bf16 -> fp32 is exact.
- torch's CUDA `fp32 / fp32-tensor` is an IEEE `div.rn` without flush-to-zero. The kernel divides with
  `tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;")`, the same instruction, so the divided values carry the stock
  bits, subnormals included. `SGL_TOPKBF_DIV=math` switches to `tl.math.div_rn` (only if its PTX also passes the
  check). `--check` greps the compiled stage-1/2 PTX: it must contain `div.rn.f32` and no `ftz`.
- Keys, ties, NaN and -0.0 follow torch's radix select order (topk20 design). Stage 2 re-loads and re-divides the
  winners, so values, indices and slot order equal `torch.topk(buffer / T, k)`.
- The downstream code is the stock code, fed the same top-k tensors.
- `torch.matmul(h, W.T, out=view)` is the same cuBLAS call as `torch.matmul(h, W.T)`: same shape, strides and
  dtype, with a 256 B aligned pointer. `--check` verifies this eager and in a CUDA graph.
- Triton's default fp32 `/` (`div.full.f32`) and `x * (1/T)` are NOT exact. The bench and `--check` show the
  bitwise test catches both.

## Switch, fallbacks, safety
- `SGL_TOPKBF=1` turns it on, read per call. Unset or 0: the hooks never import `kopt_topkbf` and every hooked
  function runs its stock code.
- The on/off is FROZEN AT CUDA-GRAPH CAPTURE: the lm_head hook is recorded into the decode graphs at server start.
  Changing `SGL_TOPKBF` later in the process has no effect on captured batch sizes; set it before launch.
- The builder dispatch fires on ANY bf16 `next_token_logits` (not only ours). That is harmless: the stock function
  would have produced the same values from the exact fp32 upcast, and with SGL_TOPKBF unset the logits are always
  fp32 so the dispatch never fires.
- `SGL_TOPKBF_CFG="CH,GS,HG,CAP,W1,W2,HOT"` (positional ints, empty = default) tunes the kernel. A bad value never
  crashes: it logs `[kopt-topkbf] fallback: bad SGL_TOPKBF_CFG=...` and uses the defaults
  (16384,64,32,32,16,8,1).
- Logs: `[kopt-topkbf] ON ...` once, at the first verify forward (CUDA-graph capture). Each fallback reason logs
  `[kopt-topkbf] fallback: <reason>` once.
- Static fallbacks, per LogitsProcessor (the stock lm_head and fp32 copy run unchanged):
  - server not DFLASH
  - fp32 lm_head
  - final_logit_softcapping
  - TP / DP-attention logits gather
  - LoRA, quantized or AMX lm_head
  - rl_on_policy_target
  - lm_head weight not CUDA bf16, or padded vocab
- Per-batch fallbacks, in the worker right after the target forward. Triggers: return_logprob, grammar
  (`has_grammar` or a `grammar_mask`), selector draft, custom logit processor, penalties (`penalizer.is_required`
  or `acc_linear_penalties`), logit_bias, and the dense target-probs cases (sampling without top_k, or a request
  with top_k >= vocab). The bf16 logits are upcast exactly, and every consumer runs the stock fp32 code.
  - Graph runs: the upcast is IN PLACE into the shared fp32 buffer (the bf16 view is its first half). Rows are
    converted in descending chunks `[ceil(r1/2), r1)`, which never overlap their source; row 0 goes through a
    one-row temporary. Result: the same tensor and bits the stock copy writes, no new `[M, V]` fp32 allocation
    (stock would not allocate either; a plain `.float()` would cost 2 GB at bs 256).
  - Eager runs (no shared buffer): `.float()`, as stock. This per-batch decision
  cannot live inside the captured graph, which is why it sits after the forward.
- Builder fallbacks:
  - Dense path (no sparse top-k, or max_top_k >= vocab): fp32 upcast, then the stock function computes
    `scaled_logits` lazily.
  - Shapes the kernel does not cover: rows <= 10 (torch uses a stable full sort there), k > 32, or a
    non-contiguous last dim. These use `torch.topk(logits / T, k)`: one promoting divide, one `[M, V]` fp32
    temporary (stock has the buffer + `scaled_logits`, i.e. also one temporary).
  - The dense remainder the guard cannot see uses `.float()` + stock (logged).
- Greedy path: `argmax` on bf16 gives the same index as on the fp32 copy (exact, order-preserving, first max).
  `--check` verifies this with forced ties.
- CUDA graphs: the lm_head hook only creates views and runs a matmul with `out=` at a static address. The builder
  runs eagerly after replay, as stock does, with no host sync beyond stock's own `.item()` (when max_top_k is None).

## Files and hooks (`# kopt topkbf:<tag>` markers)
- `kopt_topkbf.py` is copied to `srt/speculative/kopt_topkbf.py`. It holds the kernels, `topk_scaled_bf16`,
  `build_target_probs_bf16`, `lm_head_hook` and `verify_logits_guard`.
- `srt/layers/logits_processor.py`:
  - `LogitsProcessor._get_logits`: the `_compute_lm_head` call (`lm_head`)
  - `_copy_logits_to_buffer`: early return (`copy`)
- `srt/speculative/dflash_utils.py`, `build_dflash_verify_target_probs`: bf16 dispatch at the top (`dispatch`).
  topk20 edits the `torch.topk` line further down; the seed patch edits other functions.
- `srt/speculative/dflash_worker_v2.py`, `DFlashWorkerV2` verify: guard after
  `logits_output = target_out.logits_output` (`verify`). The seed hunk is ~40 lines below and untouched.
- `apply_topkbf.py`:
  - apply: idempotent. All files are planned and the new texts compiled in memory first; any problem (anchor
    drift, compile error, unverifiable stale hook) exits with nothing written. Then, in order: module installed,
    `<file>.kopt_topkbf.orig` backups, record `.kopt_topkbf_record.json` (sha256 before/after), files (atomic
    replace each).
  - Stale hook (marker present, block differs from this kit): re-applied from the backup when the file is exactly
    what the last apply wrote and the backup matches the recorded sha256; otherwise it fails loudly, rc 1.
  - `--revert`: planned first; verbatim hooks are removed as text, a verified stale file is restored from its backup.
    Exit 0 = every file equals its recorded original sha256; 7 = hooks removed but a file differs (another patch
    changed it after us); 8 = refused, stale hook without a verified backup, nothing written.
  - `--check`: see below
  - `--root DIR`, `--hidden H`, `--model DIR|repo` (or env `SGL_TOPKBF_MODEL`)

## What `apply_topkbf.py --check` proves
Its last line is `apply_topkbf check: PASS ...` or `FAIL`.
1. All 4 hooks and the module are present and compile. The patched modules import from this sglang.
2. GPU bitwise checks at M = 512, 384, 40 and 8 (bs 1 = torch-sort fallback). Logits are random peaked bf16 with
   forced ties at the 20th value and at the max. T cycles 0.7 / 0.3 / 1.3 / 1.0 per request. Every 17th row is a
   subnormal row: all -inf except 25 tiny values (+-k*1e-39, +0, -0); its top-k must stay bitwise (no FTZ).
   - The hooked `build_dflash_verify_target_probs(bf16)` equals the installed function on the fp32 copy (the
     current chain) for SGL_TOPK20=0 and, when topk20 is installed, SGL_TOPK20=1.
   - Two cases are covered: uniform top_k 20, and mixed top_k (20/5/1) with top_p 0.95.
   - The pure torch reference also matches.
   - The dense path matches stock (M=40).
3. `topk_scaled_bf16` equals `torch.topk(fp32 / T)`, values bits and indices. The BAD variants (approximate `/`
   and reciprocal multiply) must be caught. The compiled stage-1/2 PTX must contain `div.rn.f32` and no `ftz`
   (FAIL if the PTX cannot be read).
4. `argmax(bf16)` equals `argmax(fp32)` and both are the first max index.
5. The lm_head hook writes bf16 into the shared buffer, bitwise equal to the stock matmul, eager and as a replayed
   CUDA graph against a captured stock matmul (rows 512/384/40; H = `--hidden`, default 5120, and the model's
   `hidden_size` from `--model`'s config.json, which also sets V). The copy hook returns it untouched and resets
   its flag; a stale flag with fp32 logits takes the stock copy; the stock copy path is unchanged. Non-verify
   forwards stay stock. The guard's in-place upcast lands in the shared buffer (same pointer) with the stock bits.
6. The verify guard, 14 mocked cases: return_logprob, has_grammar, grammar_mask, selector, custom processor,
   penalties, acc_linear_penalties, logit_bias, top_k off, top_k >= vocab each upcast exactly; none, penalizer not
   required, top_k 20 and all-greedy leave the bf16 logits alone.

`--check --compile-only` runs only step 1, without imports, on a copy (`--root`).

## Evidence
K2 bench, `kt/k2_topk_bench.py`, run on an H100 (torch 2.12, triton 3.7). Log:
`results/kt/k2_topk_bench.log`. C_full is the chain this patch serves.

| M | stock chain: copy + divide + topk20 | C_full: bf16 + div_rn in topk | saving / verify step | bitwise |
|---|---|---|---|---|
| 512 | 1437.5 us | 639.1 us | 798 us (0.445x) | values + idx + probs, 2 seeds |
| 384 | 1113.2 us | 537.6 us | 576 us (0.483x) | same |

- In the 512-row bench data, 92% of rows have a tie inside the top-20 and 23% have 20th == 21st.
- Both sanity variants were caught.
- CPU dry runs, done in `$TMPDIR`, not kept:
  - apply / re-apply / check `--compile-only` / revert on copies of stock 1cf2b8c54 and of the seed-patched files.
  - Orders tested: topk20 then ours, ours then topk20, ours then the seed patch, and the production order
    topk20 -> ours -> seed. Every patch applied cleanly.
  - Stale hook with a verified backup: re-applied, then reverted to the original sha. Unverified stale hook:
    apply rc 1 and revert rc 8, nothing written. Foreign edit after apply: revert rc 7. Missing anchor: nothing
    written (no module, no record).
  - In-place upcast on a fake CPU shared buffer, rows 1/2/3/5/8/40/512/1023 (inf, nan, -0, subnormal included):
    same storage, bitwise equal to `.float()`. Bad `SGL_TOPKBF_CFG` values: defaults, no exception.
  - LIFO reverts restored the files to their original sha256.
  - The builder replica was bitwise equal to the stock function on CPU: bs 1/5/8, uniform, mixed + top-p, dense.

## Open risks
- Not yet run in a server. Next steps: `--check` on the H200 box, then an A/B serve with greedy and seeded-sampling
  output equality, and verify-step timing.
- The served chain also moves the lm_head write into the shared buffer. Its bitwise equality relies on cuBLAS
  choosing the same algorithm for `out=`; `--check` tests H = 5120 and the model's real hidden size when
  `--model` / `SGL_TOPKBF_MODEL` points to its config (local dir or cached HF repo).
- Revert order: topk20's `--revert` restores its whole-file backup. Reverting topk20 after ours when ours was
  applied first re-inserts our hooks. Revert in reverse order of apply (production: topk20 -> ours -> seed).
- If an upstream patch changes the sparse block of `build_dflash_verify_target_probs`, the replica in
  `build_target_probs_bf16` would diverge. `--check` compares against the installed function, so it would FAIL.
- The K2 timings were measured with `tl.math.div_rn`; the production kernel now uses the inline-PTX `div.rn.f32`
  (same instruction, operands passed as .b32 registers). Not yet compiled on a GPU here: `--check` calls the kernel
  directly (no silent torch fallback) and FAILs if it does not compile or its PTX shows `ftz`; re-time on the box.
