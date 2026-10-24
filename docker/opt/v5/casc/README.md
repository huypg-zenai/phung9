# casc - shared-prefix cascade attention for the coder's verify step (CLASS B, kernel-optim 21/10)

What: at 64 running, ~99% of verify rows share a long prompt prefix with other rows (system prompt + photo; mean ~6.8-7.3k
of ~9k KV tokens). Stock FA3 re-reads that prefix for every row (memory-bound, ~4.7 TB/s). The patch splits each
full-attention layer's verify attention into (1) ONE FA3 pass per shared-prefix group (all rows of the group as one long
query; compute-bound, num_splits=1), (2) a per-row FA3 pass over the private suffix, (3) merge_state_v2 of the two
partial softmaxes. Same math, different fp32 reduction order -> outputs can differ in rare tokens (class B).
A per-step builder (inside the CUDA graph, no host sync) finds the groups from req_to_token (longest common prefix probes).

Install (image build, sglang env; SGLang @1cf2b8c54):
  python apply_casc.py            # copies kopt_casc.py -> srt/layers/attention/kopt_casc.py + hook in flashattention_backend.py
  SGL_CASC=1 python apply_casc.py --check   # copy current + hook + import probes (SGL_CASC=1 rebinds 3 methods, =0 none)
  python apply_casc.py --revert   # remove hook + copy
Run-time switch: SGL_CASC=1 (unset/0 = stock FA3 path). Knobs: SGL_CASC_PRE_SPLITS (default 1 = REQUIRED; auto split
made the prefix pass 338 instead of 187 us/layer), SGL_CASC_MIN_P 1024, SGL_CASC_MIN_BS 16, SGL_CASC_MAXP 12288,
SGL_CASC_PROBES; SGL_CASC_STATS=N = debug only (host sync every N steps).
Safety: only the target's verify (TARGET_VERIFY, FA3, page size 1, no sliding window); batches < MIN_BS or rows with
prefix < MIN_P use the stock path inside the same kernel calls (P = 0 -> cascade == stock bitwise, tested).

Evidence (1x H200, production coder: AK FP8 Qwen-3.6-27B + DFlash 8, SGLang @1cf2b8c54, braw+a2; 21/10):
- GPU unit tests: empty prefix -> o 0 / lse -inf, merge == suffix bitwise; no sharing -> cascade == stock bitwise
  (num_splits 0 and 1); realistic layout P = 7731 for 64/64 rows, error vs fp32 2.79e-4 (stock 2.50e-4).
  Microbench 16 layers: -23% (long suffixes) .. -34% (production-like suffixes 0.3-3k).
- Served (C1): sharing stats 99.0-100% of rows share a prefix, mean P/row 6.8-7.3k, 1.9-3 shared segments/step.
  ms/step at 64 running (logstat, accept-normalised): off 59.11 / 59.35, on 56.69 / 56.59 -> -2.6 ms (-4.4%),
  tok/s 6031/5975 -> 6231/6277; accept length 5.53 both; errors 0.
  Greedy conc-1 off vs on: identical 8/8 (~10k chars each).
  Draft quality proxy (276 drafts per arm): syntax_ok off 0.996/0.996, on 0.986/0.993 (6 vs 2 failures of 552,
  -0.7 pp); export 1.000 all; length and similarity unchanged.
- CLASS B rule: >= 3% (met: 4.4% step) + valid drafts within 1 pp (met, borderline: -0.7 pp) + 64-photo A/B >= 45%
  (NOT YET - needs the full pipeline with the judge).
Not tested: B200, page size > 1, sliding-window layers (none in this model), combination with kopt_wq (different code,
expected to stack), full pipeline.
