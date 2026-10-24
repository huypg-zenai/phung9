# kopt-normcfg (class A, bitwise)

What: the GDN gated RMSNorm (`qwen3_5` `self.norm` -> `layernorm_gated.RMSNorm` -> `_layer_norm_fwd` ->
`_layer_norm_fwd_1pass_kernel`) is launched with 32 rows / 8 warps per CTA instead of the stock 4 rows / 1 warp.
It is the same kernel, with the same BLOCK_N and PDL kwargs. The per-row reduction stays 8 values per thread
followed by a 16-lane butterfly.

Evidence: kt/k1_bench.py on H100 (`results/kt/k1_bench.log`), 48 layers in one graph. At B 64 the norm drops from
0.522 to 0.380 ms (-27%), and at B 48 from 0.438 to 0.315 ms. Both are bitwise on y and rstd. r4w4, r4w8 and r8w8
are NOT bitwise, so do not use them.

Where: `sglang/kernels/ops/attention/fla/layernorm_gated.py` (sglang @1cf2b8c54, sha256 3ce4895e...). There are two
blocks marked `# kopt normcfg begin/end`:
1. after `MAX_ROWS_PER_BLOCK = 4`: reads env at import time.
2. in `_layer_norm_fwd` after `rows_per_block = calc_rows_per_block(...)`: overrides (rows, warps) only when the
   launch is bf16, has a z gate, is RMS with norm_before_gate, has no bias, uses swish, has one group with
   N = 128, and the stock launch is 4 rows / 1 warp (M > ~4 x SMs). Everything else is untouched.

Env: `SGL_NORMCFG=1` enables it; `SGL_NORMCFG_ROWS` (32) and `SGL_NORMCFG_WARPS` (8) override the config.
When `SGL_NORMCFG` is unset, the stock path runs.

```
python apply_normcfg.py            # apply / refresh (idempotent); records original sha256 next to the file
python apply_normcfg.py --check    # present + compiles; GPU: bitwise stock vs patched at Ntok 512/384/40 (x48 rows) and raw M
python apply_normcfg.py --revert   # strip blocks, verify sha256 == original, write back
python apply_normcfg.py --file /path/to/layernorm_gated.py ...   # explicit target
```
Tested offline on a copy of the 1cf2b8c54 file: apply, re-apply (no change), --check (no GPU), and --revert
(sha 3ce4895e restored). The GPU bitwise --check has NOT been run yet.

Not covered: e2e greedy/seeded identity of the served model. The kernel is bitwise in the bench, but the shipped
patch still needs the usual e2e gate. Triton must match the bench version, because the layout comes from the
compiler.
