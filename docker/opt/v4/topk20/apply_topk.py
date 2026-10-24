"""Apply the TOPK20 patch (see topk20.py) to the sglang that `import sglang` resolves to, in place, idempotent
(apply_slim.py style).

  python apply_topk.py                        patch (exit 0; an already-applied edit is skipped)
  python apply_topk.py --check                exit 0 only if the edit + the module are present AND the patched modules
                                              import (server env); with SGL_TOPK20=1 and a GPU it also runs
                                              topk20.selftest() (bitwise == torch.topk on tie-heavy rows)
  python apply_topk.py --check --compile-only same, but only compile the patched files (no sglang deps needed)
  python apply_topk.py --revert               restore the *.topk20.orig backup and remove the module
  python apply_topk.py --root DIR             patch the package at DIR (the directory that contains srt/)

Runtime switch: SGL_TOPK20=1 routes the DFlash verify sparse top-k through topk20.topk_exact; unset/0 = the stock
`torch.topk(scaled_logits, k=max_top_k, dim=-1)` call, byte for byte (read per call, so no import of topk20 when off).
Only that one call site is replaced. dflash_utils.py:121 (_dflash_top_k_renorm_prob's torch.topk) is NOT on the
production path: it runs only when sgl_kernel's top_k_renorm_prob is missing (non-CUDA / failed import), and the
whole dense branch runs only when use_sparse_topk is off or max_top_k >= vocab. Compatible with dflash_seed.patch
(it touches other hunks of dflash_utils.py; the seed patch changes only the verify coins, not this call).

Files touched (relative to the sglang package dir):
  srt/speculative/topk20.py          NEW: copy of topk20.py
  srt/speculative/dflash_utils.py    build_dflash_verify_target_probs: the sparse torch.topk call
"""
import argparse
import importlib
import importlib.util
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_SRC = os.path.join(HERE, "topk20.py")
MODULE_DST = "srt/speculative/topk20.py"
BACKUP = ".topk20.orig"

DU = "srt/speculative/dflash_utils.py"

CALL_ANCHOR = "            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)\n"
CALL_TEXT = (
    "            # sgl-topk20:call  SGL_TOPK20=1 -> bitwise-exact Triton top-k (topk20.py); else the stock torch.topk\n"
    "            import os as _sgl_topk20_os\n"
    "\n"
    "            if _sgl_topk20_os.environ.get(\"SGL_TOPK20\", \"0\") == \"1\":\n"
    "                from sglang.srt.speculative.topk20 import topk_exact as _sgl_topk_exact\n"
    "\n"
    "                topk_logits, topk_indices = _sgl_topk_exact(scaled_logits, max_top_k)\n"
    "            else:\n"
    "                topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)\n"
)

# (file, tag, mode, anchor, text)   mode: replace | before
EDITS = [
    (DU, "call", "replace", CALL_ANCHOR, CALL_TEXT),
]
FILES = [DU]
MODULES = ["sglang.srt.speculative.topk20", "sglang.srt.speculative.dflash_utils"]


def find_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_topk: `import sglang` does not resolve (set PYTHONPATH or use --root)")
    return list(spec.submodule_search_locations)[0]


def _compile(path):
    compile(open(path).read(), path, "exec")


def marker(tag):
    return f"sgl-topk20:{tag}"


def patch_text(rel, src):
    out = []
    for f, tag, mode, anchor, text in EDITS:
        if f != rel:
            continue
        if marker(tag) in src:
            out.append((tag, "already"))
            continue
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"apply_topk: {rel} edit '{tag}': anchor found {n} times (SGLang version differs?)")
        src = src.replace(anchor, text if mode == "replace" else text + anchor)
        out.append((tag, "applied"))
    return src, out


def status(root):
    missing = []
    for f, tag, *_ in EDITS:
        p = os.path.join(root, f)
        if not os.path.exists(p) or marker(tag) not in open(p).read():
            missing.append(f"{f}:{tag}")
    dst = os.path.join(root, MODULE_DST)
    if not os.path.exists(dst) or open(dst).read() != open(MODULE_SRC).read():
        missing.append(MODULE_DST + " (missing or differs from topk20.py)")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    root = os.path.abspath(a.root) if a.root else find_root()
    print(f"apply_topk: sglang package at {root}")

    if a.check:
        miss = status(root)
        if miss:
            print("apply_topk --check: NOT applied / incomplete: " + ", ".join(miss))
            return 1
        for f in FILES + [MODULE_DST]:
            _compile(os.path.join(root, f))
        st = ""
        if not a.compile_only:
            sys.path.insert(0, os.path.dirname(root))
            for m in MODULES:
                try:
                    importlib.import_module(m)
                except Exception as e:  # noqa: BLE001
                    print(f"apply_topk --check: patched module {m} does not import: {type(e).__name__}: {e}")
                    return 2
            mod = importlib.import_module("sglang.srt.speculative.topk20")
            if os.path.dirname(os.path.abspath(mod.__file__)) != os.path.join(root, "srt", "speculative"):
                print(f"apply_topk --check: topk20 imported from {mod.__file__}, not from {root}")
                return 3
            if os.environ.get("SGL_TOPK20") == "1":
                import torch
                if torch.cuda.is_available():
                    if not mod.selftest():
                        print("apply_topk --check: SGL_TOPK20=1 but topk20.selftest() != torch.topk on this GPU")
                        return 4
                    st = ", GPU selftest bitwise OK"
                else:
                    st = ", no GPU: selftest skipped"
        print(f"apply_topk --check: OK ({len(EDITS)} edit + module present"
              f"{', compile only' if a.compile_only else ', patched modules import'}{st}; "
              f"SGL_TOPK20={os.environ.get('SGL_TOPK20', '0 (default off)')})")
        return 0

    if a.revert:
        for f in FILES:
            p = os.path.join(root, f)
            if os.path.exists(p + BACKUP):
                shutil.copyfile(p + BACKUP, p)
                os.remove(p + BACKUP)
                print(f"  restored {f}")
        dst = os.path.join(root, MODULE_DST)
        if os.path.exists(dst):
            os.remove(dst)
            print(f"  removed {MODULE_DST}")
        return 0

    for f in FILES:
        p = os.path.join(root, f)
        src = open(p).read()
        new, st = patch_text(f, src)
        if new != src:
            if not os.path.exists(p + BACKUP):
                shutil.copyfile(p, p + BACKUP)
            with open(p, "w") as fh:
                fh.write(new)
            _compile(p)
        print(f"  {f}: " + ", ".join(f"{t} {s}" for t, s in st))
    shutil.copyfile(MODULE_SRC, os.path.join(root, MODULE_DST))
    print(f"  {MODULE_DST}: installed from {MODULE_SRC}")
    print("apply_topk: done (active only with SGL_TOPK20=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
