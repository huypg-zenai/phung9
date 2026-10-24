"""Apply / check / revert the MROPE-NB patch on the sglang that `import sglang` resolves to (kernel-optim K5 finding, 21/10).
K5 (stack-attributed torch trace, 64 running): the coder's ~2.2 ms/step of GPU idle is late launches after a blocking
cudaStreamSynchronize in ForwardBatch.compute_spec_mrope_positions: for image batches the per-request mrope deltas
(CPU tensors) are stacked and copied with a pageable `.to(device)`, which waits for all queued GPU work, so the CPU
builds the verify inputs while the GPU sits idle.
Patch: the same stacked CPU tensor goes through pinned memory with non_blocking=True (same values -> same positions;
class A). Runtime switch SGL_MROPE_NB=1 (unset/0 = the original line runs).
  python apply_mrope.py            edit <sglang>/srt/model_executor/forward_batch_info.py (idempotent)
  python apply_mrope.py --check    exit 0 only if the patched line is present and the module imports
  python apply_mrope.py --revert   restore the original line
"""
import importlib.util
import subprocess
import sys

OLD = "            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)\n"
NEW = ("            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0)  # kopt mrope-nb\n"
       "            if os.environ.get(\"SGL_MROPE_NB\", \"0\") == \"1\":  # kopt mrope-nb\n"
       "                mrope_delta_tensor = mrope_delta_tensor.pin_memory().to(device=device, non_blocking=True)  # kopt mrope-nb\n"
       "            else:  # kopt mrope-nb\n"
       "                mrope_delta_tensor = mrope_delta_tensor.to(device=device)  # kopt mrope-nb\n")


IMP = "import os  # kopt mrope-nb\n"


def add_import(src):
    if IMP in src:
        return src
    i = src.index("\nimport ") + 1          # first top-level plain import (after the docstring / __future__)
    return src[:i] + IMP + src[i:]


def target():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_mrope: sglang not found")
    return list(spec.submodule_search_locations)[0] + "/srt/model_executor/forward_batch_info.py"


def main():
    p = target()
    src = open(p).read()
    if "--revert" in sys.argv:
        if NEW in src or IMP in src:
            open(p, "w").write(src.replace(NEW, OLD).replace(IMP, ""))
        print("apply_mrope: reverted")
        return 0
    if "--check" in sys.argv:
        ok = NEW in src and IMP in src
        r = subprocess.run([sys.executable, "-c", "import sglang.srt.model_executor.forward_batch_info"],
                           capture_output=True, text=True)
        print(f"apply_mrope check: patched {ok}, import rc {r.returncode} {r.stderr.strip()[-200:]}")
        return 0 if ok and r.returncode == 0 else 1
    if NEW not in src:
        if src.count(OLD) != 1:
            sys.exit(f"apply_mrope: anchor found {src.count(OLD)} times (need 1)")
        open(p, "w").write(add_import(src.replace(OLD, NEW)))
    print(f"apply_mrope: patched {p} (active only with SGL_MROPE_NB=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
