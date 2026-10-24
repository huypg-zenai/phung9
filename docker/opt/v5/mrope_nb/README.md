# mrope_nb - remove the blocking mrope copy before every coder verify step (class A, kernel-optim K5/K7 21/10)

What: for image batches, ForwardBatch.compute_spec_mrope_positions (srt/model_executor/forward_batch_info.py) stacks the
per-request mrope deltas (CPU tensors) and copies them with a pageable `.to(device)`. That copy waits for ALL queued GPU
work, so the CPU builds the next verify inputs while the GPU sits idle (K5 trace: ~2.2 ms/step of late launches).
Patch: the same stacked tensor goes through pinned memory with non_blocking=True. Same values -> same positions.
(PyTorch's caching host allocator keeps the pinned block alive until the copy's stream event fires.)

Install (image build, sglang env; SGLang @1cf2b8c54):
  python apply_mrope.py            # edits srt/model_executor/forward_batch_info.py (idempotent, marker '# kopt mrope-nb')
  python apply_mrope.py --check    # exit 0 only if the patched line is present and the module imports
  python apply_mrope.py --revert   # restores the original line
Run-time switch: SGL_MROPE_NB=1 (unset/0 = the original line runs).

Evidence (1x H200, production coder: AK FP8 Qwen-3.6-27B + DFlash 8, SGLang @1cf2b8c54, braw+a2; K7 21/10):
- apply + --check rc 0. Errors 0 in all three servers.
- Identity: greedy conc-1 off vs on identical 8/8 (prefix ~10k chars); seeded temp-0.7 conc-4 fresh servers 4/4 identical.
- Stairs, ms per verify step (off / on / off-repeat): 48 running 54.38 / 53.41 / 53.92; 64 running 60.71 / 59.80 / 60.50
  -> -0.7..-1.0 ms/step (~1.3%), tok/s 5829 -> 5894 at 64; off-vs-off noise 0.2-0.46 ms.
- Profiled run with it on: GPU idle 0.56 ms/step (late launches 0.25 ms/step) vs ~2.2 ms before; the blocking
  cudaStreamSynchronize in compute_spec_mrope_positions is gone from the trace.
- Stacks with fusedq (different code), gsw, topk20, braw/a2.
Not tested: B200, text-only batches (path not taken there), full pipeline batch.
