# HANDOVER kopt-wq (kernel-optim -> pipeline-speed, 21/10): FP8 GEMM tile table, CLASS A (bitwise identical)

## What you get
Coder only. ms/step served on 1x H200 (accept 5.50-5.55 in every arm, so drafts unchanged):
| running | off | on | off-repeat | change |
|---|---|---|---|---|
| 40 | 50.42 | 40.89 | 50.72 | -19% |
| 48 | 54.08 | 45.07 | 54.37 | -17% |
| 64 | 60.29 | 57.97 | 60.51 | -4% |
Noise floor (off vs off-repeat) 0.3 ms. The biggest gain is in the mid range (bs 33-48: GEMM saves 8-15 ms per step),
which is where the batch sits whenever fewer than 64 are running (round tail, judge-bound phases).

## Install (image build) - built FROM SOURCE, no kopt-wq binary in the repo
docker/Dockerfile stage `kopt_wq_build` runs `bash build_wq.sh /opt/kopt_wq` (this dir): a python3.11 venv with torch
2.13.0+cu129 (= the coder env) + ninja, then `python apply_wq.py --build --prefix /opt/kopt_wq` = ONE extension with the
table's 7 configs, from kopt_wq_gemm.cuh + sgl_csrc/ (sgl-kernel 0.4.6.post1's cutlass_extensions headers, unchanged,
Apache-2.0) + CUTLASS @57e3cfb4 (downloaded, sha256-checked). nvcc = the base image's 12.8, explicit sm_90a gencode, so no
GPU is needed. The main stage copies kopt_wq_ext.so + manifest.json to /opt/kopt_wq and the kit copy wq/built/ (+ their
md5 in ../MANIFEST) that kopt_shim.sh checks. A failed build -> '!! kopt-wq: build FAILED', no .so, SGL_WQ=1 REFUSED at
coder start (stock path). Build log in the image: /opt/kopt_wq/BUILD.log; generated sources: /opt/kopt_wq/src/.
Downloads at build time: torch==2.13.0 from download.pytorch.org/whl/cu129 (pinned by version, like the vllm-env pip
lock) and the CUTLASS GitHub archive @57e3cfb4 (pinned by commit + sha256 in apply_wq.py). If either changes, the build
fails safe (WQ off, -4% coder speed), the image still builds.
Then:
  python apply_wq.py            # hook into sglang fp8_utils.py (idempotent, markers '# kopt wq')
  python apply_wq.py --check    # must print "check: PASS" (GPU probe: every table entry bitwise vs sgl_kernel)
  run the server with SGL_WQ=1  # unset/0 = hook does nothing
  python apply_wq.py --revert   # byte-identical revert (sha256 checked)
Boot log must show: `[kopt-wq] installed: 362 (N,K,M) entries over 6 shapes ...` and NO `[kopt-wq] fallback` / `drop`.

## Do not combine with
SGL_GSW (gsw / gsw2: same hook; kopt-wq replaces it and is a superset for M > 64) and SGL_FP8PIN (repro pin hook).
No shared hook with MROPE-NB, fusedq or casc (casc = attention only), but the combination is NOT tested together yet.

## Evidence
- Kernel identity: same production kernel code + flags, only tile/cluster/schedule differ; FP8 fast-accum keeps each
  element's K chain -> bitwise. Bench 648/648 bitwise; negative control caught (480/480 'not bitwise').
- apply --check on the box: probe 1811/1811 bitwise (362 entries, eager + graph replays, 1448 ext calls, 0 drops).
- Served e2e (results/WQ_E2E_SUMMARY.txt): greedy conc-1 off vs on identical 8/8; 0 errors/drops in all arms.
- Seeded sampling (results/WQD_SUMMARY.txt, fresh servers off/on/off/on): on vs on 4/4 identical; conc-1 seeded
  off = on = on; BUT off vs off (stock, no patch) is itself 0/4 at conc-4 and 0/1 at conc-1. So seeded output is NOT
  reproducible across server restarts even in stock production; the off-vs-on differences are that baseline noise,
  not the patch. (Greedy is reproducible and identical.)
- Table is H200-specific for SPEED (entries chosen by H200 timings); on other sm90 GPUs the probe still guarantees
  bitwise output, but the speed gain is unmeasured. Not for B200 (sm100: falls back to production).

## Files
apply_wq.py, kopt_wq.py, kopt_wq_gemm.cuh, kopt_wq_table.json, sgl_csrc/ (unchanged sgl-kernel headers for the build),
build_wq.sh (image-build step). No binary in this kit.
