"""Apply / check / revert the gsw patch (DEEPDIVE3 "GEMM switch": torch._scaled_mm fast-accum instead of sgl_kernel
fp8_scaled_mm for table (N,K,M), bitwise identical; derived from gdnk/gpad/apply_gpad.py) on the sglang that
`import sglang` resolves to (same contract as dd2/braw/apply_b.py and a2/apply_a2.py).

  python apply_gsw.py            apply (idempotent; refreshes the runtime copy, the table copy and the hook), exit 0
  python apply_gsw.py --check    exit 0 only if both copies match dd3/gsw/, the hook is present, AND (subprocess, on the
                                  GPU) after `import fp8_utils` the hook has rebound fp8_utils.fp8_scaled_mm to gsw_mm,
                                  and for EVERY table entry (N,K,M->P) apply_fp8_linear (the server's call path: bf16
                                  input -> per-token quant -> fp8_scaled_mm) gives output BITWISE equal to the stock
                                  path on random inputs (2 draws), eagerly and inside a captured CUDA graph (1/shape).
  python apply_gsw.py --revert   remove the hook and the copies

What it writes (nothing else; nothing in dd3/gsw/ at run time):
  1. <sglang>/srt/layers/quantization/gdnk_gsw.py            = copy of dd3/gsw/gsw_rt.py
  2. <sglang>/srt/layers/quantization/gdnk_gsw_table.json    = copy of dd3/gsw/gsw_table.json
  3. a hook appended to <sglang>/srt/layers/quantization/fp8_utils.py between "# >>> gdnk-gsw hook" and
     "# <<< gdnk-gsw hook" markers: at the end of that module it calls gdnk_gsw.install(), which rebinds the module
     global fp8_scaled_mm (used by apply_fp8_linear) to gdnk_gsw.gsw_mm.
Runtime switch: active once applied; SGL_GSW=0 turns it off. SGL_GSW_LOG=1 logs each (N,K,M) seen once.
Clash: touches fp8_utils.py (append only). Independent of the a1/a2/B patches (they touch GDN files only).
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN, END = "# >>> gdnk-gsw hook (apply_gsw.py) >>>", "# <<< gdnk-gsw hook <<<"
HOOK = f"""
{BEGIN}
import os as _gdnk_gsw_os
if _gdnk_gsw_os.environ.get("SGL_GSW", "1") != "0":
    from sglang.srt.layers.quantization import gdnk_gsw as _gdnk_gsw
    _gdnk_gsw.install()
{END}
"""


def sgl_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_gsw: sglang not found on this interpreter's path")
    return list(spec.submodule_search_locations)[0]


def paths(root):
    q = os.path.join(root, "srt", "layers", "quantization")
    return os.path.join(q, "gdnk_gsw.py"), os.path.join(q, "gdnk_gsw_table.json"), os.path.join(q, "fp8_utils.py")


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
    rt, tb, fu = paths(root)
    if not os.path.exists(fu):
        sys.exit(f"apply_gsw: {fu} missing (SGLang layout differs)")
    src = open(fu).read()
    for need in ("from sgl_kernel import fp8_scaled_mm", "output = fp8_scaled_mm(", "def apply_fp8_linear("):
        if need not in src:
            sys.exit(f"apply_gsw: anchor not found in fp8_utils.py: {need}")
    for s, d in (("gsw_rt.py", rt), ("gsw_table.json", tb)):
        with open(os.path.join(HERE, s), "rb") as f:
            b = f.read()
        with open(d, "wb") as f:
            f.write(b)
    new = strip_hook(src) + HOOK
    if new != src:
        with open(fu, "w") as f:
            f.write(new)
    print(f"apply_gsw: runtime -> {rt}\napply_gsw: table   -> {tb}\napply_gsw: hook    -> {fu} (active unless SGL_GSW=0)")


def revert(root):
    rt, tb, fu = paths(root)
    if os.path.exists(fu):
        src = open(fu).read()
        if BEGIN in src:
            with open(fu, "w") as f:
                f.write(strip_hook(src))
    for p in (rt, tb):
        if os.path.exists(p):
            os.remove(p)
    print("apply_gsw: reverted")


PROBE = r"""
import os, torch
os.environ["SGL_GSW"] = "1"
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization import gdnk_gsw as g
assert getattr(fp8_utils, "_gdnk_gsw_installed", False) and fp8_utils.fp8_scaled_mm is g.gsw_mm, "fp8_scaled_mm not rebound"
assert g.TABLE, "empty table"
torch.manual_seed(1234)
dev = "cuda"

def wq(N, K):
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    s = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / 448.0
    return (w / s).to(torch.float8_e4m3fn).t(), s          # (K, N) column-major view, like the server's layer.weight

def lin(x, w, s):
    return fp8_utils.apply_fp8_linear(input=x, weight=w, weight_scale=s, input_scale=None, bias=None,
                                      use_per_token_if_dynamic=True, compressed_tensor_quant=True)

def stock(x, w, s):
    fp8_utils.fp8_scaled_mm = g._orig
    try:
        return lin(x, w, s)
    finally:
        fp8_utils.fp8_scaled_mm = g.gsw_mm

n = bad = 0
for (N, K), mp in sorted(g.TABLE.items()):
    w, s = wq(N, K)
    for M in sorted(mp):
        P = "torch"
        for d in range(2):
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * (3 if d == 0 else 200)
            a, b = lin(x, w, s), stock(x, w, s)
            n += 1
            if a.shape != b.shape or not torch.equal(a, b):
                bad += 1
                print(f"MISMATCH N={N} K={K} M={M} P={P} draw {d}")
    # CUDA graph capture at the largest table M of this shape
    M = max(mp)
    xs = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            lin(xs, w, s)
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        yo = lin(xs, w, s)
    xs.copy_(torch.randn(M, K, device=dev, dtype=torch.bfloat16)); gr.replay(); torch.cuda.synchronize()
    n += 1
    if not torch.equal(yo, stock(xs, w, s)):
        bad += 1
        print(f"GRAPH MISMATCH N={N} K={K} M={M}")
print(f"probe: {n - bad}/{n} bitwise-equal (incl. {len(g.TABLE)} CUDA-graph replays), "
      f"{sum(len(v) for v in g.TABLE.values())} table entries")
assert bad == 0
"""


def check(root):
    rt, tb, fu = paths(root)
    ok = True
    for s, d in (("gsw_rt.py", rt), ("gsw_table.json", tb)):
        if sha(d) != sha(os.path.join(HERE, s)):
            print(f"check: copy missing/stale: {d}")
            ok = False
    if not os.path.exists(fu) or BEGIN not in open(fu).read():
        print(f"check: hook missing in {fu}")
        ok = False
    if ok:
        r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd="/")
        print(r.stdout.strip()[-3000:] or r.stderr.strip()[-1500:])
        if r.returncode != 0 and r.stdout.strip():
            print(r.stderr.strip()[-1500:])
        ok = r.returncode == 0
    print(f"check: {'PASS' if ok else 'FAIL'} (sglang at {root})")
    return ok


if __name__ == "__main__":
    root = sgl_root()
    if "--check" in sys.argv:
        sys.exit(0 if check(root) else 1)
    elif "--revert" in sys.argv:
        revert(root)
    else:
        apply(root)
