"""kopt_topkbf.py - [kopt-topkbf] DFlash verify: keep the target logits in bf16 and fuse `/ temperature` into topk20.

Installed by apply_topkbf.py as <sglang>/srt/speculative/kopt_topkbf.py. Active only with SGL_TOPKBF=1 (read per call
by the hooks; unset/0 = the hooks never import this module and SGLang runs byte for byte as before). NOTE: under CUDA
graphs the lm_head choice is frozen at capture time (the env must not change after the server started).

Stock chain (DFlash seeded-sampling verify, M = bs * 8 rows, V = 248320):
  lm_head matmul -> bf16 logits                                   (logits_processor._get_logits)
  _copy_logits_to_buffer: fp32 buffer.copy_(bf16)                  (read 2 B, write 4 B per logit)
  build_dflash_verify_target_probs: scaled = logits / T[M,1]       (read 4 B, write 4 B)
  torch.topk(scaled, 20) (or topk20.topk_exact, bitwise the same)  (read 4 B)
  softmax(top-20) -> [top-p renorm] -> zeros(M, V).scatter_(idx, probs)
With SGL_TOPKBF=1 (target-verify forwards of a DFLASH server only):
  lm_head matmul writes bf16 straight into the first half of the SAME shared fp32 logits buffer the stock path would
  copy into (CUDA-graph runs; no extra memory, no per-graph allocation) or returns its own bf16 tensor (eager runs);
  the fp32 copy is skipped; the target-probs builder runs topk_scaled_bf16 = a topk20 variant that reads bf16 and
  divides by the row temperature in registers with IEEE round-to-nearest division (inline PTX div.rn.f32, no .ftz),
  then the same softmax / top-p / scatter code as stock.

WHY BITWISE: bf16 -> fp32 is exact, and torch's CUDA `fp32 / fp32-tensor` is an IEEE div.rn without flush-to-zero
  (nvcc default -prec-div=true -ftz=false; only a CPU-scalar divisor becomes a reciprocal multiply). So
  div.rn.f32(float(x_bf16), T) has exactly the bits of `buffer / T`, subnormals included. The keys are computed on
  those divided values (same order, ties, NaN/-0.0 handling as torch's radix select; see topk20.py) and stage 2
  re-loads and re-divides the winners (same op -> same bits), so values, indices and slot order equal
  torch.topk(buffer / T, k). Triton's plain fp32 '/' (div.full.f32) and x * (1/T) are NOT exact; the bench / --check
  prove the bitwise test catches them; --check also greps the kernels' PTX for 'ftz' and runs subnormal rows.
  The lm_head write into a dtype-view of the shared buffer is the same torch.matmul call with out= (same shape,
  strides, dtype, >= 256 B aligned pointer -> same cuBLAS call); --check verifies it bitwise, eager and in a graph.

FALLBACKS (each logged once as '[kopt-topkbf] fallback: <reason>'):
  static, per LogitsProcessor: server is not DFLASH, fp32 lm_head, final_logit_softcapping, TP / DP-attention logits
    gather, LoRA / quantized / AMX lm_head, rl_on_policy_target, non-bf16 or padded-vocab lm_head weight
    -> the stock lm_head + fp32 copy run unchanged.
  per verify batch (worker hook): return_logprob, grammar, selector draft, custom logit processor, penalties,
    logit_bias, dense target probs (top_k off / a request with top_k >= vocab) -> the bf16 logits are upcast to fp32
    right after the forward (exact: the same values the stock copy writes) and every consumer runs the stock fp32
    code. When the logits are the bf16 view of the shared buffer the upcast is done IN PLACE into that buffer
    (descending row chunks, no new MxV allocation, the result is exactly the stock buffer); eager logits: .float().
  target-probs builder: dense path not caught by the guard -> logits.float() then the stock function; topk shapes the
    kernel does not cover (rows <= 10 with V >= 100000 -> torch uses a stable full sort, k > 32, ...) or a kernel
    compile error -> torch.topk(logits / T, k) (bf16 / fp32 promotes to fp32 in one pass = the stock math).
  Greedy argmax on bf16 returns the same index as on the fp32 copy (exact upcast, order-preserving, first max).

Env: SGL_TOPKBF=1 on. SGL_TOPKBF_CFG="CH,GS,HG,CAP,W1,W2,HOT" kernel tuning (default = topk20's M7-tested default;
     a bad value logs a fallback and uses the default). SGL_TOPKBF_DIV=asm (default, inline div.rn.f32) | math
     (tl.math.div_rn).
"""
import os

import torch
import triton
import triton.language as tl

TAG = "[kopt-topkbf]"
ENV = "SGL_TOPKBF"

_logged = set()


def enabled():
    return os.environ.get(ENV, "0") == "1"


def log_once(key, msg):
    if key in _logged:
        return
    _logged.add(key)
    print(f"{TAG} {msg}", flush=True)


def fallback(reason):
    log_once("fb:" + reason, f"fallback: {reason}")


# ------------------------------------------------------------------------------------------------ config
DEFAULT_CFG = dict(CH=16384, GS=64, HG=32, CAP=32, W1=16, W2=8, HOT=1)
_FIELDS = ("CH", "GS", "HG", "CAP", "W1", "W2", "HOT")


def _pow2(v):
    return v > 0 and v & (v - 1) == 0


def cfg_problem(cfg):
    if not all(_pow2(cfg[n]) for n in ("CH", "GS", "HG", "CAP", "W1", "W2")):
        return "CH, GS, HG, CAP, W1, W2 must be powers of two"
    G = cfg["CH"] // cfg["GS"]
    if G < 1 or cfg["HG"] > G or cfg["CAP"] < 1:
        return "need HG <= CH/GS"
    if cfg["HOT"] not in (0, 1):
        return "HOT must be 0 or 1"
    return None


def parse_cfg(s):
    """Never raises: a bad SGL_TOPKBF_CFG logs a fallback and returns the default."""
    cfg = dict(DEFAULT_CFG)
    if not s:
        return cfg
    try:
        parts = s.split(",")
        if len(parts) > len(_FIELDS):
            raise ValueError(f"{len(parts)} fields, max {len(_FIELDS)}")
        for name, v in zip(_FIELDS, parts):
            if v.strip():
                cfg[name] = int(v)
        why = cfg_problem(cfg)
        if why:
            raise ValueError(why)
        return cfg
    except Exception as e:  # noqa: BLE001
        fallback(f"bad SGL_TOPKBF_CFG={s!r} ({e}) -> default {DEFAULT_CFG}")
        return dict(DEFAULT_CFG)


CFG = parse_cfg(os.environ.get("SGL_TOPKBF_CFG", ""))
DIV_MODE = 4 if os.environ.get("SGL_TOPKBF_DIV", "asm") == "math" else 1

# ------------------------------------------------------------------------------------------------ Triton kernels
KMIN = tl.constexpr(-2147483648)   # below every real key (-inf is -2139095041)
KMAX = tl.constexpr(2147483647)    # NaN key (torch radix select: every NaN is the largest)

if hasattr(tl.math, "div_rn"):
    _math_div_rn = tl.math.div_rn
else:  # pragma: no cover - older triton
    try:
        from triton.language.extra import libdevice as _ld
    except Exception:  # noqa: BLE001
        from triton.language.extra.cuda import libdevice as _ld
    _math_div_rn = _ld.div_rn


@triton.jit
def _div_rn_ieee(x, y):
    """Exactly PTX div.rn.f32 (IEEE round-to-nearest, subnormals kept: no .ftz). Operands passed as b32 bit patterns
    (PTX accepts .b32 registers for .f32 instructions), so no float register-class constraint is needed."""
    xi = x.to(tl.int32, bitcast=True)
    yi = y.to(tl.int32, bitcast=True)
    ri = tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", "=r,r,r", [xi, yi], dtype=tl.int32, is_pure=True,
                                   pack=1)
    return ri.to(tl.float32, bitcast=True)


@triton.jit
def _scale(x, t, DIV: tl.constexpr):
    """x fp32, t fp32 scalar. DIV: 0 none, 1 inline div.rn.f32 (production), 4 tl.math.div_rn (alternative),
    2 triton '/' (approx), 3 x*(1/t). 2 and 3 exist only as deliberately wrong variants for the sanity test."""
    if DIV == 1:
        tb = tl.zeros(x.shape, tl.float32) + t          # explicit broadcast (0 + t == t exactly)
        x = _div_rn_ieee(x, tb)
    elif DIV == 4:
        tb = tl.zeros(x.shape, tl.float32) + t
        x = _math_div_rn(x, tb)
    elif DIV == 2:
        x = x / t
    elif DIV == 3:
        x = x * (1.0 / t)
    return x


@triton.jit
def _keys(rowp, offs, V, t, DIV: tl.constexpr):
    """int32 key with torch's radix order of scale(float(x)); out of range -> KMIN."""
    inr = offs < V
    x = tl.load(rowp + offs, mask=inr, other=0.0).to(tl.float32)
    x = _scale(x, t, DIV)
    b = x.to(tl.int32, bitcast=True)
    mag = b & 0x7FFFFFFF
    s = b ^ ((b >> 31) & 0x7FFFFFFF)
    s = tl.where(mag > 0x7F800000, KMAX, s)
    return tl.where(inr, s, KMIN)


@triton.jit
def _excl2d(a):
    inc = tl.cumsum(a, axis=1)
    tot = tl.sum(a, axis=1)
    off = tl.cumsum(tot, axis=0) - tot
    return inc - a + off[:, None]


@triton.jit
def _kth(s, K: tl.constexpr):
    cur = tl.max(tl.max(s, axis=1), axis=0)
    cnt = tl.sum(tl.sum((s >= cur).to(tl.int32), axis=1), axis=0)
    while cnt < K:
        cur = tl.max(tl.max(tl.where(s < cur, s, KMIN), axis=1), axis=0)
        cnt = tl.sum(tl.sum((s >= cur).to(tl.int32), axis=1), axis=0)
    return cur


@triton.jit
def _write(s, idx, t, ps, pi, CAP: tl.constexpr):
    gt = s > t
    eq = s == t
    gti = gt.to(tl.int32)
    eqi = eq.to(tl.int32)
    c_gt = tl.sum(tl.sum(gti, axis=1), axis=0)
    c_eq = tl.sum(tl.sum(eqi, axis=1), axis=0)
    pe = c_gt + _excl2d(eqi)
    pos = tl.where(gt, _excl2d(gti), pe)
    w = gt | (eq & (pe < CAP))
    tl.store(ps + pos, s, mask=w)
    tl.store(pi + pos, idx, mask=w)
    n = tl.minimum(c_gt + c_eq, CAP)
    ar = tl.arange(0, CAP)
    tl.store(ps + ar, tl.full([CAP], KMIN, tl.int32), mask=ar >= n)


@triton.jit
def _stage1(x_ptr, stride_m, V, t_ptr, cs_ptr, ci_ptr, NCHP, K: tl.constexpr, CH: tl.constexpr, GS: tl.constexpr,
            HG: tl.constexpr, CAP: tl.constexpr, HOT: tl.constexpr, KDIV: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    G: tl.constexpr = CH // GS
    t = tl.load(t_ptr + row)
    rowp = x_ptr + row * stride_m
    g = tl.arange(0, G)
    j = tl.arange(0, GS)
    offs = ch * CH + g[:, None] * GS + j[None, :]
    s = _keys(rowp, offs, V, t, KDIV)
    m = tl.max(s, axis=1)
    cge = tl.sum((m[None, :] >= m[:, None]).to(tl.int32), axis=1)
    t_k = tl.max(tl.where(cge >= K, m, KMIN), axis=0)
    slot = (row * NCHP + ch) * CAP
    ps = cs_ptr + slot
    pi = ci_ptr + slot
    if HOT:
        hot = m >= t_k
        hoti = hot.to(tl.int32)
        n_hot = tl.sum(hoti, axis=0)
        ph = tl.cumsum(hoti, axis=0) - 1
        h = tl.arange(0, HG)
        sel = hot[None, :] & (ph[None, :] == h[:, None])
        hg = tl.sum(tl.where(sel, g[None, :], 0), axis=1)
        hoffs = ch * CH + hg[:, None] * GS + j[None, :]
        hoffs = tl.where((h < n_hot)[:, None], hoffs, V)
        sh = _keys(rowp, hoffs, V, t, KDIV)
        sh = tl.where((h < n_hot)[:, None], sh, KMIN)
        c_gt = tl.sum(tl.sum((sh > t_k).to(tl.int32), axis=1), axis=0)
        if (n_hot <= HG) & (c_gt <= CAP):
            _write(sh, hoffs, t_k, ps, pi, CAP)
        else:
            s2 = _keys(rowp, offs, V, t, KDIV)
            t2 = _kth(s2, K)
            _write(s2, offs, t2, ps, pi, CAP)
    else:
        c_gt = tl.sum(tl.sum((s > t_k).to(tl.int32), axis=1), axis=0)
        if c_gt > CAP:
            t_k = _kth(s, K)
        _write(s, offs, t_k, ps, pi, CAP)


@triton.jit
def _stage2(x_ptr, stride_m, t_ptr, cs_ptr, ci_ptr, val_ptr, idx_ptr, NCH, NCHP: tl.constexpr, CAP: tl.constexpr,
            K: tl.constexpr, VDIV: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    t = tl.load(t_ptr + row)
    c = tl.arange(0, NCHP)
    j = tl.arange(0, CAP)
    off = row * (NCHP * CAP) + c[:, None] * CAP + j[None, :]
    ok = (c[:, None] + 0 * j[None, :]) < NCH
    s = tl.load(cs_ptr + off, mask=ok, other=KMIN)
    ii = tl.load(ci_ptr + off, mask=ok, other=0)
    kth = _kth(s, K)
    gt = s > kth
    eq = s == kth
    gti = gt.to(tl.int32)
    n_gt = tl.sum(tl.sum(gti, axis=1), axis=0)
    pe = n_gt + _excl2d(eq.to(tl.int32))
    pos = tl.where(gt, _excl2d(gti), pe)
    w = gt | (eq & (pe < K))
    v = tl.load(x_ptr + row * stride_m + ii, mask=w, other=0.0).to(tl.float32)
    v = _scale(v, t, VDIV)
    tl.store(val_ptr + row * K + pos, v, mask=w)
    tl.store(idx_ptr + row * K + pos, ii.to(tl.int64), mask=w)


def kernel_eligible(x, t_row, k, cfg=None):
    """None when the Triton path covers this call, else the reason (the caller then runs the stock torch math)."""
    cfg = cfg or CFG
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.dim() == 2):
        return "logits not a 2-D CUDA tensor"
    if x.dtype not in (torch.bfloat16, torch.float32):
        return f"logits dtype {x.dtype}"
    M, V = x.shape
    G = cfg["CH"] // cfg["GS"]
    if not (1 <= k <= min(cfg["CAP"], cfg["HG"], G)):
        return f"top-k {k} outside the kernel range (max {min(cfg['CAP'], cfg['HG'], G)})"
    if x.stride(1) != 1:
        return "logits last dim not contiguous"
    if V < cfg["CH"] or V >= 2 ** 31 - cfg["CH"]:
        return f"vocab {V} outside the kernel range"
    if M <= 10 and V >= 100000:
        return "rows <= 10 (torch.topk uses a stable full sort there)"
    if t_row is None or t_row.dtype != torch.float32 or t_row.numel() != M or t_row.device != x.device:
        return "temperature not fp32 [rows, 1] on the logits device"
    return None


_LAST = {}   # last compiled (stage1, stage2) handles, for the PTX check in apply_topkbf.py --check


def topk_variant(x, t_row, k, div=None, key_div=True, cfg=None):
    """Launch the fused top-k: topk(scale(float(x), t_row[:, None]), k). t_row fp32 [M] contiguous.
    div None/1 = production (inline div.rn.f32), 4 = tl.math.div_rn; 2/3 are the broken sanity variants."""
    cfg = cfg or CFG
    div = DIV_MODE if div is None else div
    M, V = x.shape
    CH, CAP = cfg["CH"], cfg["CAP"]
    nch = triton.cdiv(V, CH)
    nchp = triton.next_power_of_2(nch)
    cs = torch.empty((M, nchp, CAP), dtype=torch.int32, device=x.device)
    ci = torch.empty((M, nchp, CAP), dtype=torch.int32, device=x.device)
    val = torch.empty((M, k), dtype=torch.float32, device=x.device)
    idx = torch.empty((M, k), dtype=torch.int64, device=x.device)
    with torch.cuda.device(x.device):
        k1 = _stage1[(M, nch)](x, x.stride(0), V, t_row, cs, ci, nchp, K=k, CH=CH, GS=cfg["GS"], HG=cfg["HG"],
                               CAP=CAP, HOT=bool(cfg["HOT"]), KDIV=(div if key_div else 0), num_warps=cfg["W1"])
        k2 = _stage2[(M,)](x, x.stride(0), t_row, cs, ci, val, idx, nch, NCHP=nchp, CAP=CAP, K=k, VDIV=div,
                           num_warps=cfg["W2"])
    _LAST[div] = (k1, k2)
    # torch.topk on the k pre-sort slots reproduces torch's own sort network on the same keys (see topk20.py)
    sv, perm = torch.topk(val, k, dim=-1)
    return sv, torch.gather(idx, 1, perm)


def _torch_topk(logits, temperature, k):
    """The stock math in one fp32 pass: bf16 / fp32-tensor promotes to fp32 (upcast then IEEE divide) == buffer / T."""
    t = temperature if temperature.dim() == 2 else temperature.reshape(-1, 1)
    return torch.topk(logits / t, k, dim=-1)


def topk_scaled_bf16(logits, temperature, k):
    """== torch.topk(logits.float() / temperature, k, dim=-1) bit for bit (values fp32, indices int64).
    logits: [M, V] bf16 (fp32 also accepted), temperature: fp32 [M, 1] (or [M]). No host sync; capture-safe."""
    k = int(k)
    t_row = temperature.reshape(-1) if isinstance(temperature, torch.Tensor) else None
    why = kernel_eligible(logits, t_row, k)
    if why is not None:
        fallback(f"top-k via torch ({why})")
        return _torch_topk(logits, temperature, k)
    if not t_row.is_contiguous():
        t_row = t_row.contiguous()
    try:
        return topk_variant(logits, t_row, k)
    except Exception as e:  # noqa: BLE001  (compile / launch error -> stock math, never a crash)
        fallback(f"top-k kernel error {type(e).__name__}: {str(e)[:200]} -> torch")
        return _torch_topk(logits, temperature, k)


# ------------------------------------------------------------------------------------- target probs (dflash_utils)
def build_target_probs_bf16(orig, *, next_token_logits, sampling_info, draft_token_num, bs, max_top_k=None,
                            uniform_top_k_value=None, use_sparse_topk=True):
    """build_dflash_verify_target_probs for bf16 verify logits. Sparse top-k path: same code as stock with the
    copy + divide + topk replaced by topk_scaled_bf16. Every other path: exact fp32 upcast, then the stock
    function `orig` (scaled_logits is computed there, lazily). The worker guard already upcasts (in place) the
    dense cases it can see, so the .float() here is the rare remainder."""
    import torch.nn.functional as F

    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    kw = dict(sampling_info=sampling_info, draft_token_num=draft_token_num, bs=bs, max_top_k=max_top_k,
              uniform_top_k_value=uniform_top_k_value, use_sparse_topk=use_sparse_topk)
    if not (use_sparse_topk and need_top_k):
        fallback("dense target-probs path (no sparse top-k) in the builder -> fp32 upcast + stock")
        return orig(next_token_logits=next_token_logits.float(), **kw)

    device = next_token_logits.device
    expanded_temperature = torch.repeat_interleave(sampling_info.temperatures, draft_token_num, dim=0)
    repeated_top_ks = torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0).to(dtype=torch.int64)
    vocab_size = int(next_token_logits.shape[-1])
    repeated_top_ks.clamp_(min=1, max=vocab_size)
    if max_top_k is None:
        max_top_k = int(repeated_top_ks.max().item())   # same host sync as stock
    else:
        max_top_k = int(max_top_k)
    if max_top_k < 1:
        max_top_k = 1
    elif max_top_k > vocab_size:
        max_top_k = vocab_size
    if not (0 < max_top_k < vocab_size):
        fallback("max_top_k >= vocab (dense path) in the builder -> fp32 upcast + stock")
        return orig(next_token_logits=next_token_logits.float(), **kw)

    topk_logits, topk_indices = topk_scaled_bf16(next_token_logits, expanded_temperature, max_top_k)
    if uniform_top_k_value is None or int(uniform_top_k_value) != max_top_k:
        ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[None, :]
        valid = ranks < repeated_top_ks.unsqueeze(1)
        topk_logits = topk_logits.masked_fill(~valid, float("-inf"))
    topk_probs = F.softmax(topk_logits, dim=-1)
    if need_top_p:
        from sglang.srt.speculative.dflash_utils import _dflash_top_p_renorm_prob

        repeated_top_ps = torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0)
        topk_probs = _dflash_top_p_renorm_prob(topk_probs, repeated_top_ps)
    target_probs = torch.zeros(next_token_logits.shape, dtype=topk_probs.dtype, device=device)
    target_probs.scatter_(1, topk_indices, topk_probs)
    return target_probs.view(bs, draft_token_num, -1).contiguous()


# ---------------------------------------------------------------------------------------- lm_head (logits_processor)
_SHARED_STORAGE = set()   # data_ptr of the shared fp32 logits buffers lm_head_hook wrote bf16 into


def _spec_is_dflash():
    try:
        from sglang.srt.runtime_context import get_spec

        alg = getattr(get_spec(), "speculative_algorithm", None)
    except Exception as e:  # noqa: BLE001
        return f"cannot read the speculative algorithm ({type(e).__name__})"
    name = getattr(alg, "name", alg)
    if name is None or str(name).upper() != "DFLASH":
        return f"speculative algorithm is {name!r}, not DFLASH"
    return None


def _static_reason(lp, lm_head):
    """None when this LogitsProcessor + lm_head may keep bf16 verify logits, else the reason."""
    why = _spec_is_dflash()
    if why:
        return why
    if getattr(lp, "use_fp32_lm_head", False):
        return "fp32 lm_head"
    if getattr(lp, "final_logit_softcapping", None):
        return "final_logit_softcapping"
    if getattr(lp, "do_tensor_parallel_all_gather", False) or getattr(lp, "do_tensor_parallel_all_gather_dp_attn",
                                                                       False):
        return "TP / DP-attention logits gather"
    if getattr(lp, "rl_on_policy_target", None) is not None:
        return "rl_on_policy_target"
    if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
        return "LoRA lm_head"
    w = getattr(lm_head, "weight", None)
    if not isinstance(w, torch.Tensor):
        return "lm_head has no dense weight"
    try:
        from sglang.srt.layers import logits_processor as _lpm

        if _lpm.should_apply_lm_head_quant_method(lm_head, getattr(lm_head, "quant_method", None)):
            return "quantized lm_head"
        if _lpm.use_intel_amx_backend(lm_head):
            return "AMX lm_head"
    except Exception as e:  # noqa: BLE001
        return f"cannot inspect the lm_head ({type(e).__name__}: {e})"
    if w.dtype != torch.bfloat16 or not w.is_cuda or w.dim() != 2:
        return f"lm_head weight {w.dtype} / {w.device} (need CUDA bf16)"
    if int(w.shape[0]) != int(getattr(lp, "vocab_size", -1)):
        return f"padded vocab (weight rows {w.shape[0]} != vocab {getattr(lp, 'vocab_size', None)})"
    return None


def lm_head_hook(lp, hidden_states, lm_head, embedding_bias, logits_metadata):
    """Called by the patched LogitsProcessor._get_logits instead of _compute_lm_head when SGL_TOPKBF=1.
    Returns None -> the stock path runs; else the bf16 logits [rows, V] (and arms the copy skip)."""
    lp._kopt_tbf_keep = False
    fm = getattr(logits_metadata, "forward_mode", None)
    if fm is None or not fm.is_target_verify():
        return None
    st = getattr(lp, "_kopt_tbf_static", None)
    if st is None or st[0] is not lm_head:
        st = (lm_head, _static_reason(lp, lm_head))
        lp._kopt_tbf_static = st
        if st[1] is None:
            log_once("ON", "ON (SGL_TOPKBF=1): DFlash target-verify logits stay bf16 (no fp32 copy); verify "
                           "top-k reads bf16 and divides by T in-kernel (div.rn.f32, bitwise = stock); "
                           f"cfg {CFG} div={'asm' if DIV_MODE == 1 else 'math'}")
    if st[1] is not None:
        fallback(st[1])
        return None
    if hidden_states.dim() != 2:
        fallback("hidden_states not 2-D")
        return None
    w = lm_head.weight
    h = hidden_states.to(w.dtype)                       # stock: torch.matmul(hidden_states.to(w.dtype), w.T)
    rows, V = int(h.shape[0]), int(w.shape[0])
    buf = getattr(logits_metadata, "next_token_logits_buffer", None)
    out = None
    if (buf is not None and buf.dtype == torch.float32 and buf.is_contiguous() and buf.device == h.device
            and tuple(buf.shape) == (rows, V)):
        # bf16 view over the first rows*V*2 bytes of the shared fp32 logits buffer: static address under CUDA
        # graphs, no new memory (keeping a graph-pool bf16 tensor per captured batch size would cost GBs).
        out = buf.view(-1).view(torch.bfloat16)[: rows * V].view(rows, V)
        _SHARED_STORAGE.add(buf.untyped_storage().data_ptr())
    if out is not None:
        logits = torch.matmul(h, w.T, out=out)
    else:
        logits = torch.matmul(h, w.T)
    lp._kopt_tbf_keep = True
    return logits


# ------------------------------------------------------------------------------------------ verify (worker v2)
def upcast_shared_inplace(x):
    """x: bf16 [rows, V] view at offset 0 of a shared fp32 logits buffer (registered by lm_head_hook). Rewrites the
    buffer's first rows*V fp32 values with float(x) IN PLACE -> exactly what the stock _copy_logits_to_buffer would
    have left there; returns that fp32 [rows, V] view, or None when x is not such a view.

    Byte layout: fp32 row r = bytes [4rV, 4(r+1)V) = bf16 rows [2r, 2r+2). Writing fp32 rows [r0, r1) clobbers bf16
    rows [2r0, 2r1). Chunks go DOWNWARDS with r0 = ceil(r1 / 2), so 2*r0 >= r1: the clobbered bf16 rows are >= r1,
    i.e. sources of chunks already converted (stream-ordered before), never this chunk's source rows [r0, r1) or any
    row below. One direct convert-copy per chunk (no temporary, ~log2(rows) launches). Row 0 (r1 = 1 -> r0 would be 1)
    overlaps itself (fp32 row 0 = bf16 rows 0..1), so it goes through a [1, V] fp32 temporary (read before write)."""
    if not (isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16 and x.dim() == 2 and x.is_contiguous()
            and x.storage_offset() == 0):
        return None
    st = x.untyped_storage()
    if st.data_ptr() not in _SHARED_STORAGE:
        return None
    rows, V = int(x.shape[0]), int(x.shape[1])
    if rows == 0 or st.nbytes() < rows * V * 4:
        return None
    f32 = torch.empty(0, dtype=torch.float32, device=x.device).set_(st, 0, (rows, V), (V, 1))
    r1 = rows
    while r1 > 1:
        r0 = (r1 + 1) // 2
        f32[r0:r1].copy_(x[r0:r1])       # bf16 -> fp32 convert-copy; disjoint (2*r0 >= r1)
        r1 = r0
    f32[0:1].copy_(x[0:1].float())       # row 0: temporary first, then write
    return f32


def verify_fp32_reason(worker, batch, sampling_info, vocab=None):
    """None when every consumer of this verify batch's logits is exact on bf16, else the reason."""
    if getattr(batch, "return_logprob", False):
        return "return_logprob"
    if getattr(batch, "has_grammar", False):
        return "grammar"
    if getattr(worker, "_selector_sample", None) is not None:
        return "selector draft"
    si = sampling_info
    if si is not None:
        if getattr(si, "has_custom_logit_processor", False):
            return "custom logit processor"
        pen = getattr(si, "penalizer_orchestrator", None)
        if pen is not None and getattr(pen, "is_required", False):
            return "penalties"
        if getattr(si, "acc_linear_penalties", None) is not None:
            return "penalties"
        if getattr(si, "logit_bias", None) is not None:
            return "logit_bias"
        if getattr(si, "grammar_mask", None) is not None:
            return "grammar"
        if not getattr(si, "is_all_greedy", False):
            # sampling verify: the dense target-probs path would upcast in the builder (fresh MxV fp32) -> do it
            # here, in place. draft_input = batch.spec_info carries the host-side max_top_k.
            if not getattr(si, "need_top_k_sampling", True):
                return "dense target probs (top_k off)"
            mk = getattr(getattr(batch, "spec_info", None), "max_top_k", None)
            if vocab is not None and isinstance(mk, int) and mk >= vocab:
                return "dense target probs (a request has top_k >= vocab)"
    return None


def verify_logits_guard(worker, batch, logits_output):
    """Patched into DFlashWorkerV2's verify right after the target forward. bf16 logits + a consumer that needs fp32
    -> exact upcast (same values as the stock fp32 copy): in place into the shared buffer when the logits are its
    bf16 view (graph runs), else .float() (eager runs). Never syncs."""
    x = logits_output.next_token_logits
    if x is None or x.dtype != torch.bfloat16:
        return
    why = verify_fp32_reason(worker, batch, getattr(batch, "sampling_info", None), vocab=int(x.shape[-1]))
    if why is not None:
        y = upcast_shared_inplace(x)
        fallback(f"{why} -> verify logits upcast to fp32 ({'in place, shared buffer' if y is not None else 'new'})")
        logits_output.next_token_logits = y if y is not None else x.float()
