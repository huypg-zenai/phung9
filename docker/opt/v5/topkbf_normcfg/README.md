# HANDOVER kopt-topkbf + kopt-normcfg (kernel-optim -> pipeline-speed, 21/10): CLASS A (bitwise identical), coder only

## What you get (1x H200, production coder, stairs 120 s hold, accept unchanged 5.55-5.56)
| running | off (topk20 on) | on (+topkbf +normcfg) | off-repeat | change |
|---|---|---|---|---|
| 48 | 54.64 | 54.29 | 54.07 | inside noise (off vs off-repeat 0.57 ms) |
| 64 | 60.54 | 60.16 | 60.61 | -0.41 ms (-0.7%), noise 0.07 |
Small, but class A (ship on any gain). Microbench parts: verify top-k on bf16 logits -0.80 ms at M512 (-0.58 M384);
gated-norm launch shape 32 rows/8 warps -0.195 ms at B64. The e2e gain is below the sum: take -0.4 ms at 64 as the number.

- topkbf (SGL_TOPKBF=1): DFlash target-verify keeps the lm_head logits in bf16 (skips the fp32 copy) and the top-k
  reads bf16 and divides by T in-kernel (inline div.rn.f32, no FTZ) -> values/indices/probs bitwise = stock
  torch.topk(fp32 / T). Any batch with logprobs / penalties / grammar / logit_bias / custom processor -> stock upcast
  (in place, no new allocation). Rows <= 10 -> torch (one log line "fallback: top-k via torch (rows <= 10 ...)" = normal).
  Details: topkbf/README.md.
- normcfg (SGL_NORMCFG=1): gated RMSNorm (fla layernorm_gated _layer_norm_fwd) 32 rows / 8 warps instead of 4 / 1, only
  where stock chose 4 / 1 (verify M 24576 / 18432 / 1920); bitwise per row. Details: normcfg/README.md.

## Install (on top of topk20, before the seed shim)
  (cd topkbf && python apply_topkbf.py && python apply_topkbf.py --check)   # check: PASS (GPU self-test, needs the model
       hidden size: --model <dir|repo> or SGL_TOPKBF_MODEL)
  (cd normcfg && python apply_normcfg.py && python apply_normcfg.py --check) # normcfg: CHECK PASS
  run with SGL_TOPKBF=1 SGL_NORMCFG=1 (set BEFORE launch; on/off is frozen at CUDA-graph capture)
Boot log: `[kopt-topkbf] ON (SGL_TOPKBF=1) ...`, no `Traceback`.
REVERT ORDER MATTERS: seed shim --revert first, then normcfg, topkbf, topk20 (each revert refuses files another patch
changed after it: exit 7, nothing written). An old revert order left our box tree half-patched (seed shim then refused).

## Evidence (box 21/10 13:20-14:05 VN, /workspace/kb/KB2E_SUMMARY.txt, kernel-optim results/KB2E_SUMMARY.txt)
- --check on H200: topkbf PASS (M512 tie rows, subnormal rows, 4 temperatures, 14 guard cases, BAD variants caught),
  normcfg PASS (stock vs 32/8 bitwise at all M, 1-ulp caught), topk20 OK.
- Greedy conc-1: off vs on identical 8/8 (prefix median 10259 chars).
- Seeded conc-4: off vs on 0/4, on vs on2 4/4 = the same pattern as the stock control (WQD: off1 vs off2 0/4,
  on1 vs on2 4/4): seeded conc-4 is not reproducible across restarts in stock, so it is no evidence either way.
- No Traceback / CUDA error in any arm; revert rc 0, tree back to stock RECORD except the production patches.
- Independent review (opus) of topkbf: no output-changing path, no graph hazard; all 7 requested fixes applied and
  re-checked on GPU.
Not tested together with WQ / fusedq / mrope-nb / casc (no shared hook: topkbf = logits_processor + dflash files,
normcfg = layernorm_gated).
