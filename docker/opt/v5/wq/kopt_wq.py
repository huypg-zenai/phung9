"""kopt_wq.py - wave-quantisation-aware tile table for SGLang's channel-wise FP8 GEMM (kernel-optim WAVEQ, class A).

apply_fp8_linear (sglang/srt/layers/quantization/fp8_utils.py) calls the module-global `fp8_scaled_mm` (sgl_kernel
0.4.6.post1 cutlass sm90: for M > 128 always tile 128x128, cluster 2x1, pingpong FP8 fast-accum). At the coder's verify
sizes the N=5120 GEMMs make 160 tiles on 132 SMs (2 waves). install() rebinds fp8_utils.fp8_scaled_mm to wq_mm, which
for (N, K, M) in the table runs the SAME production kernel code (struct/caller/epilogue copied verbatim, same flags) with
another tile/cluster/schedule from ONE extension built from source (apply_wq.py --build -> $SGL_WQ_DIR/kopt_wq_ext.so, default
/opt/kopt_wq). FP8 fast-accum keeps the per-element K chain, the epilogue is per element -> bitwise-identical output
(WAVEQ bench: 648/648 + sweep; negative control caught).

Safety:
- every table entry is probed at its first EAGER use (SGLang's warm-up runs before each graph capture): output on the
  real inputs + 2 synthetic draws with the real weight must be bitwise equal to sgl_kernel.fp8_scaled_mm, else the
  entry is dropped for good ('[kopt-wq] drop ...'). The probe call returns the production result.
- an unprobed entry met inside a CUDA-graph capture -> production kernel (logged once).
- only bf16 out, no bias, fp8 e4m3 per-token x per-channel scales, row-major A / column-major B, sm90, not under
  torch.compile; everything else -> production.
- extension missing / wrong torch / wrong sgl_kernel -> '[kopt-wq] fallback: <reason>' once, nothing rebound.
Env: SGL_WQ=1 enables (hook in fp8_utils.py); SGL_WQ_DIR (extension dir); SGL_WQ_TABLE (table json);
SGL_WQ_LOG=1 (log every (N,K,M) once).
"""
import hashlib
import importlib.util
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WQ_DIR = os.environ.get("SGL_WQ_DIR", "/opt/kopt_wq")
TABLE_FILE = os.environ.get("SGL_WQ_TABLE") or next(
    (p for p in (os.path.join(_HERE, "kopt_wq_table.json"), os.path.join(_HERE, "wq_table.json")) if os.path.exists(p)),
    os.path.join(_HERE, "kopt_wq_table.json"))
SGL_KERNEL_VERSION = "0.4.6.post1"
_LOG_ALL = os.environ.get("SGL_WQ_LOG", "0") == "1"

UNPROBED, OK, DROP = 0, 1, 2
_prev = None      # what fp8_utils.fp8_scaled_mm was before install (normally sgl_kernel.fp8_scaled_mm)
_sgl = None       # sgl_kernel.fp8_scaled_mm itself (probe reference)
_EXT = None
ENT = {}          # (N, K, M) -> [ext index, state, config name]
ST = {"ext_calls": 0, "probes": 0, "drops": 0, "capture_unprobed": 0, "installed": False}
_logged = set()


def _log(msg, key=None):
    if key is not None:
        if key in _logged:
            return
        _logged.add(key)
    if os.environ.get("RANK", "0") in ("0", "") or key is None:
        print(f"[kopt-wq] {msg}", file=sys.stderr, flush=True)


def _compiling():
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


def _bits(t):
    return t.view(torch.int16)


def _probe(ent, key, a, b, sa, sb):
    """First eager use of a table entry: compare against production on the real inputs + 2 synthetic draws."""
    idx, _, name = ent
    N, K, M = key
    ST["probes"] += 1
    ref = _sgl(a, b, sa, sb, torch.bfloat16)
    ok = torch.equal(_bits(_EXT.mm(idx, a, b, sa, sb)), _bits(ref))
    if ok:
        g = torch.Generator(device=a.device)
        g.manual_seed(1000003 + M)
        for amp in (3.0, 400.0):
            x = (torch.randn(M, K, device=a.device, generator=g) * amp).clamp(-448, 448).to(torch.float8_e4m3fn)
            xs = (torch.rand(M, 1, device=a.device, generator=g) * 0.05 + 1e-4).contiguous()
            if not torch.equal(_bits(_EXT.mm(idx, x, b, xs, sb)), _bits(_sgl(x, b, xs, sb, torch.bfloat16))):
                ok = False
                break
    ent[1] = OK if ok else DROP
    if not ok:
        ST["drops"] += 1
        _log(f"drop N={N} K={K} M={M} cfg={name}: not bitwise-equal to sgl_kernel -> production for good")
    elif _LOG_ALL:
        _log(f"N={N} K={K} M={M} -> {name} (probe OK)", key=("ok",) + key)
    return ref


def wq_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    if (bias is None and out_dtype == torch.bfloat16 and mat_a.dim() == 2 and mat_b.dim() == 2
            and not _compiling()):
        M, N, K = mat_a.shape[0], mat_b.shape[1], mat_b.shape[0]
        ent = ENT.get((N, K, M))
        if ent is not None and ent[1] != DROP and (
                mat_a.dtype == torch.float8_e4m3fn and mat_b.dtype == torch.float8_e4m3fn
                and mat_a.stride(1) == 1 and mat_b.stride(0) == 1
                and scales_a.numel() == M and scales_b.numel() == N
                and scales_a.dtype == torch.float32 and scales_b.dtype == torch.float32
                and scales_a.is_contiguous() and scales_b.is_contiguous()):
            if ent[1] == OK:
                ST["ext_calls"] += 1
                if _LOG_ALL and not torch.cuda.is_current_stream_capturing():
                    _log(f"eager ext call N={N} K={K} M={M} (first, call #{ST['ext_calls']})", key=("eager", N, K, M))
                return _EXT.mm(ent[0], mat_a, mat_b, scales_a, scales_b)
            if torch.cuda.is_current_stream_capturing():
                ST["capture_unprobed"] += 1
                _log(f"N={N} K={K} M={M}: first seen inside a graph capture (unprobed) -> production",
                     key=("cap", N, K, M))
            else:
                return _probe(ent, (N, K, M), mat_a, mat_b, scales_a, scales_b)
        elif _LOG_ALL:
            _log(f"N={N} K={K} M={M} -> production", key=("prod", N, K, M))
    return _prev(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)


def load_table(path=TABLE_FILE):
    raw = json.load(open(path))
    return raw, {(tuple(int(v) for v in nk.split(",")), int(m)): cfg for nk, e in raw["table"].items()
                 for m, cfg in e.items()}


def _load_ext():
    so, man = os.path.join(WQ_DIR, "kopt_wq_ext.so"), os.path.join(WQ_DIR, "manifest.json")
    if not os.path.exists(so) or not os.path.exists(man):
        raise RuntimeError(f"extension not built ({so}); run apply_wq.py --build")
    m = json.load(open(man))
    if m.get("torch") != torch.__version__:
        raise RuntimeError(f"extension built for torch {m.get('torch')}, running {torch.__version__}")
    spec = importlib.util.spec_from_file_location("kopt_wq_ext", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = list(mod.names())
    if names != m.get("configs"):
        raise RuntimeError(f"extension configs {names} != manifest {m.get('configs')}")
    return mod, names, m


def install():
    """Called by the fp8_utils.py hook when SGL_WQ=1. Never raises: any problem -> production + one log line."""
    global _prev, _sgl, _EXT, ENT
    from sglang.srt.layers.quantization import fp8_utils
    if getattr(fp8_utils, "_kopt_wq_installed", False):
        return
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            raise RuntimeError("not an sm90 GPU")
        import sgl_kernel
        if getattr(sgl_kernel, "__version__", None) != SGL_KERNEL_VERSION:
            raise RuntimeError(f"sgl_kernel {getattr(sgl_kernel, '__version__', None)} != {SGL_KERNEL_VERSION} "
                               f"(table proven against {SGL_KERNEL_VERSION})")
        if not hasattr(fp8_utils, "fp8_scaled_mm"):
            raise RuntimeError("fp8_utils has no fp8_scaled_mm (SGLang layout differs)")
        raw, table = load_table()
        ext, names, man = _load_ext()
        tsha = hashlib.sha256(open(TABLE_FILE, "rb").read()).hexdigest()
        ent, missing = {}, set()
        for (nk, m), cfg in table.items():
            if cfg in names:
                ent[(nk[0], nk[1], m)] = [names.index(cfg), UNPROBED, cfg]
            else:
                missing.add(cfg)
        if missing:
            _log(f"table configs not in the extension (entries -> production): {sorted(missing)}")
        _sgl = sgl_kernel.fp8_scaled_mm
        _EXT, ENT = ext, ent
        _prev = fp8_utils.fp8_scaled_mm
        fp8_utils.fp8_scaled_mm = wq_mm
        fp8_utils._kopt_wq_installed = True
        ST["installed"] = True
        _log(f"installed: {len(ENT)} (N,K,M) entries over {len({k[:2] for k in ENT})} shapes, configs {names}, "
             f"table {TABLE_FILE} (sha {tsha[:12]}{'' if tsha == man.get('table_sha256') else ', != build table'}), "
             f"ext {WQ_DIR}")
    except Exception as e:  # noqa: BLE001 - any failure means: stay on production
        _log(f"fallback: {e}")
