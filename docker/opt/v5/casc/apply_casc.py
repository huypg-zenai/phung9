"""Apply / check / revert the CASC patch (shared-prefix cascade attention for the FA3 TARGET_VERIFY forward, see
kopt_casc.py and DESIGN.md) on the sglang that `import sglang` resolves to (apply_fusedq.py style).
  python apply_casc.py            copy kopt_casc.py -> <sglang>/srt/layers/attention/kopt_casc.py
                                  + append a hook to <sglang>/srt/layers/attention/flashattention_backend.py (idempotent)
  python apply_casc.py --check    exit 0 only if the copy is current, the hook is present AND (subprocesses) importing
                                  the backend installs the wrappers with SGL_CASC=1 and leaves the class untouched without
  python apply_casc.py --revert   remove hook + copy
Runtime switch: SGL_CASC=1 (unset/0 = install() is never called; the backend class is the original one).
Knobs: SGL_CASC_PROBES, SGL_CASC_MAXP, SGL_CASC_MIN_P, SGL_CASC_MIN_BS (see kopt_casc.py).
SGLang @1cf2b8c54: FlashAttentionBackend.init_cuda_graph_state / init_forward_metadata_in_graph / forward_extend.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN, END = "# >>> kopt casc hook (apply_casc.py) >>>", "# <<< kopt casc hook <<<"
HOOK = f"""
{BEGIN}
if __import__("os").environ.get("SGL_CASC", "0") == "1":
    from sglang.srt.layers.attention import kopt_casc as _kopt_casc
    _kopt_casc.install(FlashAttentionBackend)
{END}
"""
ANCHORS = ("class FlashAttentionBackend(AttentionBackend):", "    def init_forward_metadata_in_graph(self, forward_batch",
           "    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):", "    def forward_extend(",
           "def merge_state_v2_wrapper(o, s_a, o_exp, s_b):", "create_trtllm_mha_kv_indices_triton")


def sgl_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_casc: sglang not found on this interpreter's path")
    return list(spec.submodule_search_locations)[0]


def paths(root):
    d = os.path.join(root, "srt", "layers", "attention")
    return os.path.join(d, "kopt_casc.py"), os.path.join(d, "flashattention_backend.py")


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
    dst, be = paths(root)
    src = open(be).read()
    tbl = open(os.path.join(root, "kernels", "ops", "kvcache", "trtllm_mha_page_table.py")).read()
    for need in ANCHORS:
        if need not in src + tbl:
            sys.exit(f"apply_casc: anchor not found: {need!r}")
    with open(os.path.join(HERE, "kopt_casc.py"), "rb") as a, open(dst, "wb") as b:
        b.write(a.read())
    new = strip_hook(src) + HOOK
    if new != src:
        open(be, "w").write(new)
    print(f"apply_casc: module -> {dst}\napply_casc: hook   -> {be} (active only with SGL_CASC=1)")


def revert(root):
    dst, be = paths(root)
    src = open(be).read() if os.path.exists(be) else ""
    if BEGIN in src:   # read BEFORE open(.., "w") truncates (apply_fusedq.py's revert has that bug)
        with open(be, "w") as f:
            f.write(strip_hook(src))
    if os.path.exists(dst):
        os.remove(dst)
    print("apply_casc: reverted")


PROBE = r"""
import os, sys
on = sys.argv[1] == "1"
if on:
    os.environ["SGL_CASC"] = "1"
else:
    os.environ.pop("SGL_CASC", None)
from sglang.srt.layers.attention import flashattention_backend as FB
C = FB.FlashAttentionBackend
mods = {f: getattr(C, f).__module__ for f in ("forward_extend", "init_forward_metadata_in_graph", "init_cuda_graph_state")}
if on:
    assert all(m.endswith("kopt_casc") for m in mods.values()), mods
else:
    assert not any(m.endswith("kopt_casc") for m in mods.values()) and not hasattr(C, "_kopt_casc_installed"), mods
print(f"probe SGL_CASC={sys.argv[1]} OK: {mods}")
"""


def check(root):
    dst, be = paths(root)
    ok = sha(dst) == sha(os.path.join(HERE, "kopt_casc.py"))
    if not ok:
        print(f"check: module copy missing/stale: {dst}")
    if not os.path.exists(be) or BEGIN not in open(be).read():
        print(f"check: hook missing in {be}")
        ok = False
    if ok:
        for flag in ("1", "0"):
            r = subprocess.run([sys.executable, "-c", PROBE, flag], capture_output=True, text=True, cwd="/")
            print(r.stdout.strip() or r.stderr.strip()[-1500:])
            ok = ok and r.returncode == 0
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
