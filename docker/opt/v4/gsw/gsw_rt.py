"""gsw_rt.py - "GEMM switch" for SGLang's channel-wise FP8 GEMM (DEEPDIVE3).

apply_fp8_linear (sglang/srt/layers/quantization/fp8_utils.py) calls the module-global `fp8_scaled_mm`
(sgl_kernel cutlass sm90, per-token x per-channel scales) for every Merlin W8A8-FP8 linear. For many row counts M its
dispatch picks a slow tile config (e.g. M=352: 26.5 ms of GEMM per verify step vs 14.9 ms). torch._scaled_mm with
rowwise scales and use_fast_accum=True (torch's own CUTLASS sm90 kernel) gives BITWISE identical output on these shapes
(gsw_bench.py checks every table entry on 4 draws; apply_gsw.py --check re-proves it through apply_fp8_linear, eagerly
and in a CUDA graph). install() rebinds fp8_utils.fp8_scaled_mm to gsw_mm, which switches only for (N,K,M) in the table.
Table: gdnk_gsw_table.json (installed copy) or gsw_table.json: {"table": {"N,K": [M, ...]}}.
Env: SGL_GSW=0 -> not installed; SGL_GSW_LOG=1 -> print each new (N,K,M) once (switch/stock) to stderr.
"""
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_FILE = os.environ.get("SGL_GSW_TABLE") or next(
    (p for p in (os.path.join(_HERE, "gdnk_gsw_table.json"), os.path.join(_HERE, "gsw_table.json")) if os.path.exists(p)),
    os.path.join(_HERE, "gdnk_gsw_table.json"))
_orig = None
TABLE = {}
_LOG = os.environ.get("SGL_GSW_LOG", "0") == "1"
_seen = set()


def load_table(path=TABLE_FILE):
    raw = json.load(open(path))
    return {tuple(int(v) for v in nk.split(",")): set(int(m) for m in ms) for nk, ms in raw["table"].items() if ms}


def gsw_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    if mat_a.dim() == 2 and bias is None and out_dtype == torch.bfloat16:
        M = mat_a.shape[0]
        key = (mat_b.shape[1], mat_b.shape[0])
        hit = M in TABLE.get(key, ()) and scales_a.numel() == M and scales_b.numel() == key[0]
        if _LOG and (key, M) not in _seen:
            _seen.add((key, M))
            print(f"[gsw] N={key[0]} K={key[1]} M={M} -> {'torch' if hit else 'stock'}", file=sys.stderr, flush=True)
        if hit:
            return torch._scaled_mm(mat_a, mat_b, scale_a=scales_a.reshape(M, 1), scale_b=scales_b.reshape(1, -1),
                                    out_dtype=out_dtype, use_fast_accum=True)
    return _orig(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)


def install():
    global _orig, TABLE
    from sglang.srt.layers.quantization import fp8_utils
    if getattr(fp8_utils, "_gdnk_gsw_installed", False):
        return
    if not hasattr(fp8_utils, "fp8_scaled_mm"):
        raise RuntimeError("gsw: fp8_utils has no fp8_scaled_mm (SGLang layout differs)")
    TABLE = load_table()
    _orig = fp8_utils.fp8_scaled_mm
    fp8_utils.fp8_scaled_mm = gsw_mm
    fp8_utils._gdnk_gsw_installed = True
    if os.environ.get("RANK", "0") in ("0", ""):
        print(f"[gsw] installed: {sum(len(v) for v in TABLE.values())} (N,K,M) entries over {len(TABLE)} shapes "
              f"from {TABLE_FILE}", file=sys.stderr, flush=True)
