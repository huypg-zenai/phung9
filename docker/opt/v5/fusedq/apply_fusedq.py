"""Apply / check / revert the FUSEDQ patch (fused act-quant kernels, see fusedq.py) on the sglang that
`import sglang` resolves to (apply_b.py style).
  python apply_fusedq.py            copy fusedq.py -> <sglang>/srt/layers/kopt_fusedq.py (+ the two .cuh next to it)
                                    + append a hook to <sglang>/srt/models/qwen3_5.py (idempotent)
  python apply_fusedq.py --check    exit 0 only if the copies are current, the hook is present AND (subprocess)
                                    importing qwen3_5 installs the wrappers; with SGL_FUSEDQ=1 also the GPU selftest
                                    (JIT build + bitwise vs the production pair)
  python apply_fusedq.py --revert   remove hook + copy
Runtime switch: SGL_FUSEDQ=1 (unset/0 = the original methods are called, same stream, nothing else changes).
SGLang @1cf2b8c54: GemmaRMSNorm._forward_impl, Qwen2MoeMLP.forward, fp8_utils.scaled_fp8_quant.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN, END = "# >>> fusedq hook (apply_fusedq.py) >>>", "# <<< fusedq hook <<<"
HOOK = f"""
{BEGIN}
from sglang.srt.layers import kopt_fusedq as _kopt_fusedq
_kopt_fusedq.install()
{END}
"""


def sgl_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_fusedq: sglang not found on this interpreter's path")
    return list(spec.submodule_search_locations)[0]


CUH = ("silu_quant.cuh", "gemma_quant.cuh")


def paths(root):
    d = os.path.join(root, "srt", "layers")
    return os.path.join(d, "kopt_fusedq.py"), os.path.join(root, "srt", "models", "qwen3_5.py")


def strip_hook(src):
    while BEGIN in src:
        i = src.index(BEGIN)
        j = src.index(END, i) + len(END)
        i = src.rfind("\n", 0, i) if src.rfind("\n", 0, i) >= 0 else i
        src = src[:i] + src[j:]
    return src.rstrip("\n") + "\n"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def apply(root):
    dst, wk = paths(root)
    src = open(wk).read()
    for need in ("GemmaRMSNorm", "Qwen2MoeMLP"):
        if need not in src:
            sys.exit(f"apply_fusedq: anchor not found in qwen3_5.py: {need}")
    for f in ("fusedq.py",) + CUH:
        out = dst if f == "fusedq.py" else os.path.join(os.path.dirname(dst), f)
        with open(os.path.join(HERE, f), "rb") as a, open(out, "wb") as b:
            b.write(a.read())
    new = strip_hook(src) + HOOK
    if new != src:
        open(wk, "w").write(new)
    print(f"apply_fusedq: module -> {dst}\napply_fusedq: hook   -> {wk} (active only with SGL_FUSEDQ=1)")


def revert(root):
    dst, wk = paths(root)
    if os.path.exists(wk):
        src = open(wk).read()                      # read BEFORE open(wk, "w") truncates it (bug fixed 21/10)
        if BEGIN in src:
            open(wk, "w").write(strip_hook(src))
    for f in [dst] + [os.path.join(os.path.dirname(dst), c) for c in CUH]:
        if os.path.exists(f):
            os.remove(f)
    print("apply_fusedq: reverted")


PROBE = r"""
import os
os.environ["SGL_FUSEDQ"] = "1"
from sglang.srt.models import qwen3_5
from sglang.srt.layers import kopt_fusedq as K, layernorm as LN
from sglang.srt.layers.quantization import fp8_utils as FU
from sglang.srt.models import qwen2_moe as QM
assert getattr(FU, "_fusedq", False), "hook did not run"
assert FU.scaled_fp8_quant.__module__ == K.__name__
assert LN.GemmaRMSNorm._forward_impl.__module__ == K.__name__
assert QM.Qwen2MoeMLP.forward.__module__ == K.__name__
print("probe OK: scaled_fp8_quant / GemmaRMSNorm._forward_impl / Qwen2MoeMLP.forward wrapped")
"""


def check(root):
    dst, wk = paths(root)
    ok = sha(dst) == sha(os.path.join(HERE, "fusedq.py")) and all(
        sha(os.path.join(os.path.dirname(dst), c)) == sha(os.path.join(HERE, c)) for c in CUH)
    if not ok:
        print(f"check: module copy missing/stale: {dst}")
    if not os.path.exists(wk) or BEGIN not in open(wk).read():
        print(f"check: hook missing in {wk}")
        ok = False
    if ok:
        r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd="/")
        print(r.stdout.strip() or r.stderr.strip()[-1500:])
        ok = r.returncode == 0
    if ok and os.environ.get("SGL_FUSEDQ", "0") == "1":
        # GPU selftest (JIT-builds the kernel): bitwise vs silu_and_mul + per-token quant, M 1..2111 + timing
        r = subprocess.run([sys.executable, "-c", "import sys; from sglang.srt.layers import kopt_fusedq as K; "
                            "sys.exit(0 if K.test() else 1)"], capture_output=True, text=True, cwd="/")
        print("\n".join((r.stdout.strip() or r.stderr.strip()[-1500:]).splitlines()[-4:]))
        ok = r.returncode == 0
    print(f"check: {'PASS' if ok else 'FAIL'} (sglang at {root})")
    return ok


if __name__ == "__main__":
    root = sgl_root()
    if "--check" in sys.argv:
        sys.exit(0 if check(root) else 1)
    if "--revert" in sys.argv:
        revert(root)
    else:
        apply(root)
