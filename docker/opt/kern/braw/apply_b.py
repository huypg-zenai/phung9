"""Apply / check / revert the Track B patch (chunked GDN verify + fold commit) on the sglang that `import sglang`
resolves to (importlib.util.find_spec - works for site-packages, an editable install or an overlay on PYTHONPATH).

  python apply_b.py            apply (idempotent; re-running refreshes the kernel copy and the hook), exit 0
  python apply_b.py --check    exit 0 only if the kernel copy == b/kernels.py, the hook is present, AND the patched
                               modules import with the patch active (subprocess: gdn_backend import -> install() ->
                               TritonGDNKernel.target_verify and hybrid_linear_attn_backend.scatter_... replaced)
  python apply_b.py --revert   remove the hook and the kernel copy

What it writes (nothing else; nothing in b/ at run time):
  1. <sglang>/kernels/ops/attention/fla/gdnk_b.py  = copy of b/kernels.py (the patched sglang never imports from b/)
  2. a 6-line hook appended to <sglang>/srt/layers/attention/linear/gdn_backend.py, between "# >>> gdnk-B hook" and
     "# <<< gdnk-B hook" markers. At the END of that module both hooked modules are fully imported (gdn_backend imports
     hybrid_linear_attn_backend and kernels.gdn_triton at its top), so the hook calls gdnk_b.install(), which
     monkeypatches:
       a) sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.target_verify  (the default Triton
          GDN verify used by DFLASH target_verify via GDNKernelDispatcher) -> gdnk_b.gdn_b_verify: outputs for all
          draft tokens + a 0.26 MB fp32 stash per request row in intermediate_ssm, NO per-token state writes.
       b) sglang.srt.layers.attention.hybrid_linear_attn_backend.scatter_mamba_states_after_mtp_verify (called by
          HybridLinearAttnBackend.update_mamba_state_after_mtp_verify <- DFlashWorkerV2._update_target_mamba_state_
          after_verify) -> gdnk_b.gdn_b_fold_all_layers (temporal in place + extra_buffer track slots) + the stock
          conv-window scatter (fused_conv_window_scatter_with_mask), the same structure as SGLang's own
          commit_gdn_replayssm_fold_after_verify.
Runtime switch: the patch is ACTIVE by default once applied; SGL_GDN_B=0 turns it off (stock kernels, same process
layout). Tuning env: SGL_GDN_B_PREC / _PREC_S / _BV / _WARPS / _FOLD_BV / _FOLD_WARPS (see kernels.py; b_bench.py prints
the best values). Unsupported (raises loudly): tree drafts (EAGLE topk>1), --linear-attn-verify-backend flashinfer is
not hooked (then the fold is skipped automatically: it only runs after a B verify ran in the process).
Clash: touches gdn_backend.py (append only) - another patch appending to the same file is fine; one that rewrites
TritonGDNKernel.target_verify or the scatter name would be overridden by this one.
"""
import hashlib
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN, END = "# >>> gdnk-B hook (apply_b.py) >>>", "# <<< gdnk-B hook <<<"
HOOK = f"""
{BEGIN}
import os as _gdnk_b_os
if _gdnk_b_os.environ.get("SGL_GDN_B", "1") != "0":
    from sglang.kernels.ops.attention.fla import gdnk_b as _gdnk_b
    _gdnk_b.install()
{END}
"""


def sgl_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_b: sglang not found on this interpreter's path")
    return list(spec.submodule_search_locations)[0]


def paths(root):
    return (os.path.join(root, "kernels", "ops", "attention", "fla", "gdnk_b.py"),
            os.path.join(root, "srt", "layers", "attention", "linear", "gdn_backend.py"))


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
    kdst, gdn = paths(root)
    if not os.path.exists(gdn):
        sys.exit(f"apply_b: {gdn} missing (SGLang layout differs)")
    src = open(gdn).read()
    for need in ("from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel",
                 "from sglang.srt.layers.attention.hybrid_linear_attn_backend import"):
        if need not in src:
            sys.exit(f"apply_b: anchor import not found in gdn_backend.py: {need}")
    with open(os.path.join(HERE, "kernels.py"), "rb") as f:
        kb = f.read()
    with open(kdst, "wb") as f:
        f.write(kb)
    new = strip_hook(src) + HOOK
    if new != src:
        with open(gdn, "w") as f:
            f.write(new)
    print(f"apply_b: kernel -> {kdst}\napply_b: hook  -> {gdn} (active unless SGL_GDN_B=0)")


def revert(root):
    kdst, gdn = paths(root)
    if os.path.exists(gdn):
        src = open(gdn).read()
        if BEGIN in src:
            with open(gdn, "w") as f:
                f.write(strip_hook(src))
    if os.path.exists(kdst):
        os.remove(kdst)
    print("apply_b: reverted")


PROBE = r"""
import os
os.environ["SGL_GDN_B"] = "1"
import sglang.srt.layers.attention.linear.gdn_backend  # noqa: F401  (runs the hook)
from sglang.srt.layers.attention.linear.kernels import gdn_triton
import sglang.srt.layers.attention.hybrid_linear_attn_backend as hlab
from sglang.kernels.ops.attention.fla import gdnk_b
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_conv_window_scatter_with_mask  # noqa: F401
assert getattr(gdn_triton.TritonGDNKernel, "_gdnk_b", False), "target_verify not patched"
assert getattr(hlab, "_gdnk_b", False), "scatter not patched"
assert hlab.scatter_mamba_states_after_mtp_verify.__module__ == gdnk_b.__name__, "scatter name not replaced"
print("probe OK: target_verify + scatter patched, conv scatter importable")
"""


def check(root):
    kdst, gdn = paths(root)
    ok = True
    if sha(kdst) != sha(os.path.join(HERE, "kernels.py")):
        print(f"check: kernel copy missing/stale: {kdst}")
        ok = False
    if not os.path.exists(gdn) or BEGIN not in open(gdn).read():
        print(f"check: hook missing in {gdn}")
        ok = False
    if ok:
        r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd="/")
        print(r.stdout.strip() or r.stderr.strip()[-1500:])
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
