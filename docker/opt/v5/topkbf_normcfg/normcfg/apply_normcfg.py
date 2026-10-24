"""kopt-normcfg (class A, bitwise): launch the gated RMSNorm of the GDN layers (FLA _layer_norm_fwd_1pass_kernel,
called by qwen3_5 self.norm -> layernorm_gated.RMSNorm -> _layer_norm_fwd) with 32 rows / 8 warps per CTA instead
of the stock 4 rows / 1 warp. Same kernel, same per-row reduction tree (k1_bench H100: bitwise, -27% norm time).

  python apply_normcfg.py            apply (idempotent: an existing block is replaced), records the original sha256
  python apply_normcfg.py --check    patch present + file compiles; with a GPU: bitwise test patched vs stock launch
  python apply_normcfg.py --revert   strip the blocks, verify sha256 == the original, write back
  --file PATH   layernorm_gated.py to patch (default: found from the sglang that `import sglang` resolves to)

Edits (two inserted line blocks between '# kopt normcfg begin' / '# kopt normcfg end', nothing else):
  1. after `MAX_ROWS_PER_BLOCK = 4`: read env at import. SGL_NORMCFG=1 enables; SGL_NORMCFG_ROWS (32) /
     SGL_NORMCFG_WARPS (8) override. Unset / 0 -> the code path is the stock one.
  2. in _layer_norm_fwd after `rows_per_block = calc_rows_per_block(M, x.device)`: override (rows, warps) ONLY for the
     measured case: bf16 x, z gate, RMS, norm_before_gate, no bias, swish, one group with N = group_size = 128
     (BLOCK_N 128), and stock launch 4 rows / 1 warp (= M > 4 * SMs; small M where stock picks 1-2 rows untouched).
     grid, PDL kwargs (USE_GDC / launch_pdl), BLOCK_N and the kernel are unchanged.
"""
import argparse
import hashlib
import importlib.util
import os
import py_compile
import sys

TAG_B, TAG_E = "# kopt normcfg begin", "# kopt normcfg end"
STOCK_SHA = "3ce4895e768aead4f12031b37fc0ee511d783b9ec476016c85b715c2dcf84988"   # sglang @1cf2b8c54 layernorm_gated.py
A1 = "MAX_ROWS_PER_BLOCK = 4\n"
B1 = f"""{TAG_B} (kernel-optim/normcfg/apply_normcfg.py; --revert removes it)
_KOPT_NORMCFG = None
if __import__("os").environ.get("SGL_NORMCFG", "0") == "1":
    _KOPT_NORMCFG = (int(__import__("os").environ.get("SGL_NORMCFG_ROWS", "32")),
                     int(__import__("os").environ.get("SGL_NORMCFG_WARPS", "8")))
{TAG_E}
"""
A2 = "    rows_per_block = calc_rows_per_block(M, x.device)\n"
B2 = f"""    {TAG_B}
    if (_KOPT_NORMCFG is not None and rows_per_block == 4 and num_warps == 1 and BLOCK_N == 128
            and group_size == 128 and ngroups == 1 and z is not None and bias is None and is_rms_norm
            and norm_before_gate and activation == "swish" and x.dtype == torch.bfloat16):
        rows_per_block, num_warps = _KOPT_NORMCFG
    {TAG_E}
"""


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def find_file():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("normcfg: sglang not found (pass --file)")
    root = list(spec.submodule_search_locations)[0]
    for rel in ("kernels/ops/attention/fla/layernorm_gated.py", "srt/layers/attention/fla/layernorm_gated.py"):
        if os.path.exists(os.path.join(root, rel)):
            return os.path.join(root, rel)
    sys.exit(f"normcfg: layernorm_gated.py not found under {root}")


def strip(src):
    out, skip = [], False
    for ln in src.splitlines(keepends=True):
        if TAG_B in ln:
            skip = True
        if not skip:
            out.append(ln)
        if TAG_E in ln:
            skip = False
    return "".join(out)


def apply(path):
    src = open(path).read()
    orig = strip(src)
    for a in (A1, A2):
        if orig.count(a) != 1:
            sys.exit(f"normcfg: anchor {a.strip()!r} found {orig.count(a)}x (need 1) in {path}")
    side = path + ".kopt_normcfg.sha256"
    if not os.path.exists(side):
        open(side, "w").write(sha(orig) + "\n")
    elif open(side).read().strip() != sha(orig):
        sys.exit(f"normcfg: {path} (blocks stripped) no longer matches the recorded original sha256 ({side})")
    if sha(orig) != STOCK_SHA:
        print(f"normcfg: note: original sha256 {sha(orig)[:16]} != sglang@1cf2b8c54 {STOCK_SHA[:16]} (anchors OK)")
    new = orig.replace(A1, A1 + B1).replace(A2, A2 + B2)
    if new != src:
        open(path, "w").write(new)
    py_compile.compile(path, doraise=True)
    print(f"normcfg: applied to {path} ({'unchanged' if new == src else 'written'}); enable with SGL_NORMCFG=1")


def revert(path):
    src = open(path).read()
    orig = strip(src)
    side = path + ".kopt_normcfg.sha256"
    want = open(side).read().strip() if os.path.exists(side) else STOCK_SHA
    if sha(orig) != want:
        sys.exit(f"normcfg: revert refused: stripped file sha256 {sha(orig)[:16]} != original {want[:16]}")
    if orig != src:
        open(path, "w").write(orig)
    if os.path.exists(side):
        os.remove(side)
    print(f"normcfg: reverted {path} (sha256 {want[:16]} verified)")


def check(path):
    src = open(path).read()
    ok = src.count(TAG_B) == 2 and src.count(TAG_E) == 2 and strip(src).replace(A1, A1 + B1).replace(A2, A2 + B2) == src
    py_compile.compile(path, doraise=True)
    print(f"normcfg: patch present and current: {ok}")
    if not ok:
        return 1
    import torch
    if not torch.cuda.is_available():
        print("normcfg: no GPU -> bitwise test skipped")
        return 0
    from sglang.kernels.ops.attention.fla import layernorm_gated as lg
    if os.path.abspath(lg.__file__) != os.path.abspath(path):
        print(f"normcfg: !! imported {lg.__file__} != patched {path}")
        return 1
    cfg = (int(os.environ.get("SGL_NORMCFG_ROWS", "32")), int(os.environ.get("SGL_NORMCFG_WARPS", "8")))
    dev, bad = torch.device("cuda"), 0
    torch.manual_seed(0)
    w = (1 + 0.1 * torch.randn(128, device=dev)).to(torch.bfloat16)
    sm = torch.cuda.get_device_properties(dev).multi_processor_count
    for ntok in (512, 384, 40):
        for M in (ntok * 48, ntok):          # verify rows (tokens x 48 v-heads), and the raw count
            x = (torch.randn(M, 128, device=dev) * 0.3).to(torch.bfloat16)
            z = torch.randn(M, 128, device=dev, dtype=torch.bfloat16)
            res = []
            for c in (None, cfg):
                lg._KOPT_NORMCFG = c
                y, _, r = lg._layer_norm_fwd(x, w, None, 1e-6, z=z, group_size=None, norm_before_gate=True,
                                             is_rms_norm=True, activation="swish")
                torch.cuda.synchronize()
                res.append((y.clone(), r.clone()))
            same = torch.equal(res[0][0].view(torch.int16), res[1][0].view(torch.int16)) and torch.equal(
                res[0][1].view(torch.int32), res[1][1].view(torch.int32))
            yb = res[0][0].clone()
            yb.view(torch.int16).view(-1)[0] += 1
            caught = not torch.equal(yb.view(torch.int16), res[1][0].view(torch.int16))
            applies = lg.calc_rows_per_block(M, dev) == 4
            bad += (not same) or (not caught)
            print(f"  M {M:6d}: override applies {applies} (SMs {sm}); stock vs rows {cfg[0]} warps {cfg[1]} "
                  f"bitwise {same}; 1-ulp caught {caught}")
    lg._KOPT_NORMCFG = cfg if os.environ.get("SGL_NORMCFG", "0") == "1" else None
    print(f"normcfg: CHECK {'PASS' if not bad else 'FAIL'}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file")
    a = ap.parse_args()
    f = a.file or find_file()
    if a.revert:
        revert(f)
    elif a.check:
        sys.exit(check(f))
    else:
        apply(f)
