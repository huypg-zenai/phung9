# fusedq - fused SiLU-and-mul + per-token FP8 quant for the coder MLP (class A, kernel-optim 21/10)

What: in every Qwen3.6 MLP the coder runs silu_and_mul (bf16 out) and then scaled_fp8_quant (per-token) on it before
down_proj. One fused CUDA kernel (silu_quant.cuh, built by SGLang's own JIT loader, --use_fast_math on sm90 like the
originals) writes the SAME fp8 bytes AND the SAME fp32 scales; down_proj gets them via a tiny cache in the patched
scaled_fp8_quant. Only the SiLU fusion is on. The Gemma add+RMSNorm+quant fusion in the same file is OFF unless
SGL_FUSEDQ_NORM=1 (it is 1 ulp off in rare rows vs the production flashinfer norm -> do NOT enable).

Install (image build, sglang env; SGLang @1cf2b8c54):
  python apply_fusedq.py                  # copies fusedq.py -> srt/layers/kopt_fusedq.py (+ 2 .cuh), appends a hook to
                                          # srt/models/qwen3_5.py (idempotent; marker '# >>> fusedq hook')
  SGL_FUSEDQ=1 python apply_fusedq.py --check   # copies current + hook + import probe + GPU selftest (JIT build,
                                          # bitwise vs silu_and_mul + per-token quant, M 1..2111) -> "check: PASS"
  python apply_fusedq.py --revert         # byte-identical restore of qwen3_5.py (truncation bug fixed 21/10)
Run-time switch: SGL_FUSEDQ=1 (unset/0 = original methods, nothing else changes).
Safety: per MLP module, the first eager call (SGLang's warm-up runs before each graph capture) runs the stock path
AND the fused kernel and keeps the fusion only if the bytes+scales are equal and down_proj consumed the cached pair;
otherwise that module stays stock for good (stderr '[fusedq] ...: stock path'). Only sm90, bf16, M < 2*SMs*8 rows
(2112 on H200), not under torch.compile. First use JIT-compiles once (cached like SGLang's other JIT kernels; nvcc
needed at that time, same as SGLang's own jit kernels).

Evidence (1x H200, production coder: AK FP8 Qwen-3.6-27B + DFlash 8, SGLang @1cf2b8c54, seed patch on; K5/K6 21/10):
- Unit: bitwise PASS (M 1..2111, zero rows, outliers, widths 17408 + 5120); 64 sites in a graph at M=512: -0.43 ms/step.
- Served identity: greedy conc-1 cold off vs on identical 8/8; seeded temp-0.7 conc-4 fresh servers off vs on 4/4.
  0 errors; log: silu fused 200 calls / 264 cache hits in the first window.
- Stairs, ms per verify step (off / off-repeat / on): 48 running 54.42 / 53.87 / 53.48; 64 running 60.19 / 59.88 / 59.43
  -> -0.6..-0.7 ms (~1%) vs an off-vs-off noise floor of 0.3-0.55 ms: SMALL, at the edge of noise, consistent with
  the unit timing. Expected end-to-end <=1% coder decode time. Identity is the strong part; the speed is marginal.
- Stacks with gsw_m6 (GEMM configs) and topk20 (verify top-k): different call sites.
Not tested: B200, tree/greedy verify (n/a), full pipeline batch.
