"""Track B kernels: chunked tensor-core GDN target-verify (WY/UT form, the whole draft window = ONE chunk) that writes NO
per-token states, plus the fold that, once acceptance is known, writes the state after the accepted prefix.

Math: MATH.txt (fp64 reference + proof: ref.py).  Layout = SGLang's: state pool [slots, HV, V, K] (K contiguous),
intermediate_ssm [rows, TS, HV, V, K] (per layer) / [L, rows, TS, HV, V, K] (all layers), rows = request position.

  gdn_b_verify(...)           one launch per GDN layer, replaces fused_sigmoid_gating_delta_rule_update(target_verify).
                              grid (V/BV, N*HV); one CTA = (request, v-head, V-tile); reads S0 once, outputs o for all
                              tokens, and writes a small fp32 STASH (Vn = delta-corrected values, normalized k, gate g)
                              into the first 0.26 MB of the request's (otherwise unused) intermediate_ssm row.
  gdn_b_fold_all_layers(...)  one launch for all layers, replaces the intermediate_ssm part of
                              scatter_mamba_states_after_mtp_verify: S <- exp(g_t) S + vn_t k_t^T for t < accepted,
                              in place in the pool (read S0 once, write once) + the extra_buffer track slot.
  install()                   runtime monkeypatch for the installed SGLang (called by the hook apply_b.py adds).

Precision of the two state dots (S0 is bf16, exact in bf16/tf32), PREC:
  bf16x2  X = hi + lo (two bf16), 2 bf16 MMAs, ~2^-17 relative  (default: same MMA cost as one tf32 dot)
  bf16x3  3 bf16 MMAs, ~fp32;  bf16  1 MMA, 2^-9 (test only);  tf32  1 tf32 MMA (operand truncation ~2^-10);
  tf32x2  X split hi/lo in tf32 (S exact) 2 MMAs ~2^-20;  tf32x3 Triton's 3-pass;  ieee  fp32 FMA (slow reference).
PREC_S = precision of the small in-window dots (16x16 / 16x128): ieee default (fp32 FMA; tf32x3 spills registers), tf32.
A fp32 state pool (--mamba-ssm-dtype float32) forces PREC to tf32x3 (bf16 splits of the state would round it).
Env overrides (server): SGL_GDN_B_PREC, SGL_GDN_B_PREC_S, SGL_GDN_B_BV, SGL_GDN_B_WARPS, SGL_GDN_B_FOLD_BV,
SGL_GDN_B_FOLD_WARPS.
"""
import os

import torch
import triton
import triton.language as tl

TP = 16                     # padded window (tl.dot needs >= 16); draft windows up to 16 tokens
DEF = dict(prec=os.environ.get("SGL_GDN_B_PREC", "bf16x2"), prec_s=os.environ.get("SGL_GDN_B_PREC_S", "ieee"),
           BV=int(os.environ.get("SGL_GDN_B_BV", "64")), warps=int(os.environ.get("SGL_GDN_B_WARPS", "4")),
           fold_BV=int(os.environ.get("SGL_GDN_B_FOLD_BV", "32")),
           fold_warps=int(os.environ.get("SGL_GDN_B_FOLD_WARPS", "4")))
PRECS = ("raw", "bf16x2", "bf16x3", "bf16", "tf32", "tf32x2", "tf32x3", "ieee")


# ============================================================================================== device helpers
@triton.jit
def _dot_state(Sb, X, PREC: tl.constexpr):
    """Sb [BV, K] state tile (bf16 for the bf16* / tf32x2 modes), X [K, TP] fp32 -> Sb @ X [BV, TP] (fp32 acc)."""
    if PREC == "bf16x3":
        x1 = X.to(tl.bfloat16)
        r1 = X - x1.to(tl.float32)
        x2 = r1.to(tl.bfloat16)
        x3 = (r1 - x2.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(Sb, x1)
        acc = tl.dot(Sb, x2, acc=acc)
        acc = tl.dot(Sb, x3, acc=acc)
    elif PREC == "bf16x2":
        x1 = X.to(tl.bfloat16)
        x2 = (X - x1.to(tl.float32)).to(tl.bfloat16)
        acc = tl.dot(Sb, x1)
        acc = tl.dot(Sb, x2, acc=acc)
    elif PREC == "bf16":
        acc = tl.dot(Sb, X.to(tl.bfloat16))
    elif PREC == "tf32x2":
        Sf = Sb.to(tl.float32)
        xi = X.to(tl.int32, bitcast=True)
        hi = ((xi >> 13) << 13).to(tl.float32, bitcast=True)   # exact tf32 head (truncation), lo = exact remainder
        acc = tl.dot(Sf, hi, input_precision="tf32")
        acc = tl.dot(Sf, X - hi, acc=acc, input_precision="tf32")
    else:  # "tf32" | "tf32x3" | "ieee"
        acc = tl.dot(Sb.to(tl.float32), X, input_precision=PREC)
    return acc


# ============================================================================================== verify
@triton.jit
def gdn_b_verify_kernel(
    q, k, v, a, b, A_log, dt_bias, o,
    h_src, h_idx, stride_h_slot,
    stash, stash_idx, stride_stash_row,
    cu_seqlens, scale, softplus_beta, softplus_threshold,
    stride_qt, stride_kt, stride_vt, stride_at, stride_bt, stride_ot,
    TP: tl.constexpr, TS: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, PREC: tl.constexpr, PREC_A: tl.constexpr, PREC_O: tl.constexpr, PREC_T: tl.constexpr,
    NEED_A8: tl.constexpr, EARLY_S: tl.constexpr,
    WRITE_STASH: tl.constexpr, STATE_BF16: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    n_tok = eos - bos
    offs_t = tl.arange(0, TP)
    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    mask_t = offs_t < n_tok
    mask_v = offs_v < V
    tok = bos + offs_t
    if EARLY_S:   # dd2: issue the state-tile load first so its HBM latency overlaps the in-window algebra
        idx = tl.load(h_idx + i_n).to(tl.int64)
        m_s = mask_v[:, None] & (offs_k[None, :] < K) & (idx >= 0)
        S = tl.load(h_src + idx * stride_h_slot + i_hv * V * K + offs_v[:, None] * K + offs_k[None, :], mask=m_s,
                    other=0.0)

    # ---- inputs, gating, l2norm (same formulas / op order per element as the SGLang verify kernel)
    b_q = tl.load(q + tok[:, None] * stride_qt + i_h * K + offs_k[None, :], mask=mask_t[:, None], other=0.0
                  ).to(tl.float32)                                                       # [TP, K]
    b_k = tl.load(k + tok[:, None] * stride_kt + i_h * K + offs_k[None, :], mask=mask_t[:, None], other=0.0
                  ).to(tl.float32)                                                       # [TP, K]
    b_vT = tl.load(v + tok[None, :] * stride_vt + i_hv * V + offs_v[:, None],
                   mask=mask_t[None, :] & mask_v[:, None], other=0.0).to(tl.float32)     # [BV, TP]
    b_a = tl.load(a + tok * stride_at + i_hv, mask=mask_t, other=0.0).to(tl.float32)
    b_b = tl.load(b + tok * stride_bt + i_hv, mask=mask_t, other=0.0).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)
    x = b_a + b_dt
    bx = softplus_beta * x
    sp = tl.where(bx <= softplus_threshold, (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(bx)), x)
    g = tl.where(mask_t, -tl.exp(b_A) * sp, 0.0)
    beta = tl.where(mask_t, 1.0 / (1.0 + tl.exp(-b_b)), 0.0)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q, axis=1) + 1e-6)[:, None]
    b_q = b_q * scale
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k, axis=1) + 1e-6)[:, None]
    gc = tl.cumsum(g, axis=0)
    gam = tl.exp(gc)

    # ---- S0 tile [BV, K], read once
    if not EARLY_S:
        idx = tl.load(h_idx + i_n).to(tl.int64)
        m_s = mask_v[:, None] & (offs_k[None, :] < K) & (idx >= 0)
        S = tl.load(h_src + idx * stride_h_slot + i_hv * V * K + offs_v[:, None] * K + offs_k[None, :], mask=m_s,
                    other=0.0)
    if STATE_BF16:
        Sb = S.to(tl.bfloat16)
    else:
        Sb = S.to(tl.float32)

    # ---- in-window algebra (TP x TP)
    rows = offs_t[:, None]
    cols = offs_t[None, :]
    strict = rows > cols
    causal = rows >= cols
    diff = gc[:, None] - gc[None, :]
    KK = tl.dot(b_k, tl.trans(b_k), input_precision=PREC_A)
    Am = tl.where(strict, KK * tl.exp(tl.where(strict, diff, 0.0)) * beta[:, None], 0.0)
    eye = tl.where(rows == cols, 1.0, 0.0)
    N1 = -Am                                       # (I + A)^-1 = (I - A)(I + A^2)(I + A^4)[(I + A^8)]
    Tinv = eye + N1
    N2 = tl.dot(N1, N1, input_precision=PREC_T)
    Tinv = tl.dot(Tinv, eye + N2, input_precision=PREC_T)
    N4 = tl.dot(N2, N2, input_precision=PREC_T)
    Tinv = tl.dot(Tinv, eye + N4, input_precision=PREC_T)
    if NEED_A8:
        N8 = tl.dot(N4, N4, input_precision=PREC_T)
        Tinv = tl.dot(Tinv, eye + N8, input_precision=PREC_T)
    TinvT = tl.trans(Tinv)
    UT = tl.dot(b_vT * beta[None, :], TinvT, input_precision=PREC_A)                       # [BV, TP] = U^T
    WT = tl.dot(tl.trans(b_k * (beta * gam)[:, None]), TinvT, input_precision=PREC_A)      # [K, TP]  = W^T
    VnT = UT - _dot_state(Sb, WT, PREC)                                                    # [BV, TP] = Vn^T
    QK = tl.dot(b_q, tl.trans(b_k), input_precision=PREC_O)
    P = tl.where(causal, QK * tl.exp(tl.where(causal, diff, 0.0)), 0.0)
    OT = _dot_state(Sb, tl.trans(b_q * gam[:, None]), PREC) + tl.dot(VnT, tl.trans(P), input_precision=PREC_O)
    tl.store(o + tok[None, :] * stride_ot + i_hv * V + offs_v[:, None], OT.to(o.dtype.element_ty),
             mask=mask_t[None, :] & mask_v[:, None])

    # ---- stash for the fold (fp32): Vn [HV, TS, V] | kn [H, TS, K] | g [HV, TS]
    if WRITE_STASH:
        row = tl.load(stash_idx + i_n).to(tl.int64)
        base = stash + row * stride_stash_row
        m_ts = mask_t & (offs_t < TS)
        tl.store(base + i_hv * TS * V + offs_t[None, :] * V + offs_v[:, None], VnT,
                 mask=m_ts[None, :] & mask_v[:, None])
        if i_v == 0:
            if i_hv % (HV // H) == 0:
                tl.store(base + HV * TS * V + i_h * TS * K + offs_t[:, None] * K + offs_k[None, :], b_k,
                         mask=m_ts[:, None])
            tl.store(base + HV * TS * V + H * TS * K + i_hv * TS + offs_t, g, mask=m_ts)



# ============================================================================================== verify, raw-bf16 form (dd2)
@triton.jit
def gdn_b_verify_raw_kernel(
    q, k, v, a, b, A_log, dt_bias, o,
    h_src, h_idx, stride_h_slot,
    stash, stash_idx, stride_stash_row,
    cu_seqlens, scale, softplus_beta, softplus_threshold,
    stride_qt, stride_kt, stride_vt, stride_at, stride_bt, stride_ot,
    TP: tl.constexpr, TS: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BV: tl.constexpr, PREC_A: tl.constexpr, PREC_O: tl.constexpr, PREC_T: tl.constexpr,
    NEED_A8: tl.constexpr, EARLY_S: tl.constexpr, WRITE_STASH: tl.constexpr,
):
    # dd2 advisor idea 1: every 128-long contraction runs on the RAW bf16 operands (q, k, S0 are bf16 -> products
    # exact, fp32 accumulate); the l2norm factors rq, rk and all gating are applied afterwards on 16-wide tiles.
    #   X_k = S0 k_raw^T, X_q = S0 q_raw^T              [BV, TP]  (2 bf16 MMAs, replaces the bf16x2 W / gamQ dots)
    #   KK = rk rk^T * (k_raw k_raw^T), QK = rq rk^T * (q_raw k_raw^T)   (bf16 MMAs)
    #   R^T[:, t] = beta_t (v_t - gam_t rk_t X_k[:, t]);  Vn^T = R^T Tinv^T;  O^T = X_q * (gam rq) + Vn^T P^T
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    n_tok = eos - bos
    offs_t = tl.arange(0, TP)
    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    mask_t = offs_t < n_tok
    mask_v = offs_v < V
    tok = bos + offs_t
    if EARLY_S:
        idx = tl.load(h_idx + i_n).to(tl.int64)
        m_s = mask_v[:, None] & (offs_k[None, :] < K) & (idx >= 0)
        Sb = tl.load(h_src + idx * stride_h_slot + i_hv * V * K + offs_v[:, None] * K + offs_k[None, :], mask=m_s,
                     other=0.0)
    # raw operands, both orientations loaded directly (no tl.trans of big tiles)
    q_tk = tl.load(q + tok[:, None] * stride_qt + i_h * K + offs_k[None, :], mask=mask_t[:, None], other=0.0)  # [TP,K]
    k_tk = tl.load(k + tok[:, None] * stride_kt + i_h * K + offs_k[None, :], mask=mask_t[:, None], other=0.0)  # [TP,K]
    k_kt = tl.load(k + tok[None, :] * stride_kt + i_h * K + offs_k[:, None], mask=mask_t[None, :], other=0.0)  # [K,TP]
    q_kt = tl.load(q + tok[None, :] * stride_qt + i_h * K + offs_k[:, None], mask=mask_t[None, :], other=0.0)  # [K,TP]
    b_vT = tl.load(v + tok[None, :] * stride_vt + i_hv * V + offs_v[:, None],
                   mask=mask_t[None, :] & mask_v[:, None], other=0.0).to(tl.float32)     # [BV, TP]
    b_a = tl.load(a + tok * stride_at + i_hv, mask=mask_t, other=0.0).to(tl.float32)
    b_b = tl.load(b + tok * stride_bt + i_hv, mask=mask_t, other=0.0).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)
    x = b_a + b_dt
    bx = softplus_beta * x
    sp = tl.where(bx <= softplus_threshold, (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(bx)), x)
    g = tl.where(mask_t, -tl.exp(b_A) * sp, 0.0)
    beta = tl.where(mask_t, 1.0 / (1.0 + tl.exp(-b_b)), 0.0)
    qf = q_tk.to(tl.float32)
    kf = k_tk.to(tl.float32)
    rq = scale / tl.sqrt(tl.sum(qf * qf, axis=1) + 1e-6)              # [TP] (row factors)
    rk = 1.0 / tl.sqrt(tl.sum(kf * kf, axis=1) + 1e-6)
    gc = tl.cumsum(g, axis=0)
    gam = tl.exp(gc)
    if not EARLY_S:
        idx = tl.load(h_idx + i_n).to(tl.int64)
        m_s = mask_v[:, None] & (offs_k[None, :] < K) & (idx >= 0)
        Sb = tl.load(h_src + idx * stride_h_slot + i_hv * V * K + offs_v[:, None] * K + offs_k[None, :], mask=m_s,
                     other=0.0)
    Xk = tl.dot(Sb, k_kt)                                             # [BV, TP] fp32, exact products
    Xq = tl.dot(Sb, q_kt)
    KKr = tl.dot(k_tk, k_kt)                                          # [TP, TP]
    QKr = tl.dot(q_tk, k_kt)
    rows = offs_t[:, None]
    cols = offs_t[None, :]
    strict = rows > cols
    causal = rows >= cols
    diff = gc[:, None] - gc[None, :]

    KK = KKr * rk[:, None] * rk[None, :]
    Am = tl.where(strict, KK * tl.exp(tl.where(strict, diff, 0.0)) * beta[:, None], 0.0)
    eye = tl.where(rows == cols, 1.0, 0.0)
    N1 = -Am
    Tinv = eye + N1
    N2 = tl.dot(N1, N1, input_precision=PREC_T)
    Tinv = tl.dot(Tinv, eye + N2, input_precision=PREC_T)
    N4 = tl.dot(N2, N2, input_precision=PREC_T)
    Tinv = tl.dot(Tinv, eye + N4, input_precision=PREC_T)
    if NEED_A8:
        N8 = tl.dot(N4, N4, input_precision=PREC_T)
        Tinv = tl.dot(Tinv, eye + N8, input_precision=PREC_T)
    RT = (b_vT - Xk * (gam * rk)[None, :]) * beta[None, :]            # [BV, TP]
    VnT = tl.dot(RT, tl.trans(Tinv), input_precision=PREC_A)          # [BV, TP] = Vn^T
    QK = QKr * rq[:, None] * rk[None, :]
    P = tl.where(causal, QK * tl.exp(tl.where(causal, diff, 0.0)), 0.0)
    OT = Xq * (gam * rq)[None, :] + tl.dot(VnT, tl.trans(P), input_precision=PREC_O)
    tl.store(o + tok[None, :] * stride_ot + i_hv * V + offs_v[:, None], OT.to(o.dtype.element_ty),
             mask=mask_t[None, :] & mask_v[:, None])
    if WRITE_STASH:
        row = tl.load(stash_idx + i_n).to(tl.int64)
        base = stash + row * stride_stash_row
        m_ts = mask_t & (offs_t < TS)
        tl.store(base + i_hv * TS * V + offs_t[None, :] * V + offs_v[:, None], VnT,
                 mask=m_ts[None, :] & mask_v[:, None])
        if i_v == 0:
            if i_hv % (HV // H) == 0:
                tl.store(base + HV * TS * V + i_h * TS * K + offs_t[:, None] * K + offs_k[None, :], kf * rk[:, None],
                         mask=m_ts[:, None])
            tl.store(base + HV * TS * V + H * TS * K + i_hv * TS + offs_t, g, mask=m_ts)

# ============================================================================================== fold (commit)
@triton.jit
def gdn_b_fold_kernel(
    h, stash, state_idx, last_step, track_idx, track_step, stash_rows,
    stride_h_layer, stride_h_slot, stride_s_layer, stride_s_row,
    TS: tl.constexpr, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr,
    HAS_TRACK: tl.constexpr, HAS_ROWS: tl.constexpr, NULL_ID: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_lh = tl.program_id(2)
    i_l = (i_lh // HV).to(tl.int64)
    i_hv = i_lh % HV
    i_h = i_hv // (HV // H)
    idx = tl.load(state_idx + i_n).to(tl.int64)
    if idx <= NULL_ID:
        return
    n = tl.load(last_step + i_n).to(tl.int32) + 1          # accepted tokens (protocol: <= TS)
    if n <= 0:
        return
    if HAS_TRACK:
        tr_idx = tl.load(track_idx + i_n).to(tl.int64)
        tr_step = tl.load(track_step + i_n).to(tl.int32)
    else:
        tr_idx = NULL_ID
        tr_step = -1
    if HAS_ROWS:
        row = tl.load(stash_rows + i_n).to(tl.int64)
    else:
        row = i_n.to(tl.int64)

    offs_k = tl.arange(0, K)
    offs_v = i_v * BV + tl.arange(0, BV)
    mask_v = offs_v < V
    m_h = mask_v[:, None] & (offs_k[None, :] < K)
    hb = h + i_l * stride_h_layer + i_hv * V * K + offs_v[:, None] * K + offs_k[None, :]
    S = tl.load(hb + idx * stride_h_slot, mask=m_h, other=0.0).to(tl.float32)
    sb = stash + i_l * stride_s_layer + row * stride_s_row
    p_vn = sb + i_hv * TS * V + offs_v
    p_k = sb + HV * TS * V + i_h * TS * K + offs_k
    p_g = sb + HV * TS * V + H * TS * K + i_hv * TS
    for t in tl.static_range(TS):
        if t < n:
            vn = tl.load(p_vn + t * V, mask=mask_v, other=0.0)
            kk = tl.load(p_k + t * K)
            gg = tl.load(p_g + t)
            S = S * tl.exp(gg) + vn[:, None] * kk[None, :]
            if HAS_TRACK:
                if (t == tr_step) & (tr_idx > NULL_ID):
                    tl.store(hb + tr_idx * stride_h_slot, S.to(h.dtype.element_ty), mask=m_h)
    tl.store(hb + idx * stride_h_slot, S.to(h.dtype.element_ty), mask=m_h)


# ============================================================================================== host wrappers
def stash_f32(buf):
    """fp32 view of an intermediate_ssm-shaped buffer (any float dtype, last dim contiguous)."""
    assert buf.stride(-1) == 1, "stash buffer needs a contiguous last dim"
    return buf if buf.dtype == torch.float32 else buf.view(torch.float32)


def stash_floats(H, HV, K, V, TS):
    return HV * TS * V + H * TS * K + HV * TS


def _tok_view(x, D):
    """[..., Ntok, heads, D] -> (tensor, token stride); needs D contiguous and head stride D."""
    if x.stride(-1) != 1 or x.stride(-2) != D:
        x = x.contiguous()
    return x, x.stride(-3)


def gdn_b_verify(A_log, dt_bias, q, k, v, a, b, state, state_idx, cu_seqlens, stash, stash_rows, *, scale=None,
                 BV=None, num_warps=None, prec=None, prec_s=None, softplus_beta=1.0, softplus_threshold=20.0,
                 write_stash=True):
    """q, k [..., Ntok, H, K]; v [..., Ntok, HV, V]; a, b [..., Ntok, HV]; state [slots, HV, V, K] (slot stride free);
    state_idx [>=N] int (-1 = padding); cu_seqlens [N+1] int; stash = this layer's intermediate_ssm [rows, TS, HV, V,
    K] (any float dtype); stash_rows [>=N] int (row per request).  Returns o with v's shape (q's dtype).
    Linear chain, uniform window (Ntok = N*T, T <= min(16, TS)); no host sync (CUDA-graph safe)."""
    H, K = q.shape[-2], q.shape[-1]
    HV, V = v.shape[-2], v.shape[-1]
    Ntok = q.shape[-3]
    N = cu_seqlens.shape[0] - 1
    assert N > 0 and Ntok % N == 0, f"gdnk-B verify needs a uniform window (Ntok={Ntok}, N={N})"
    T = Ntok // N
    TS = stash.shape[1]
    assert T <= TP and T <= TS, f"window {T} > {min(TP, TS)}"
    assert stash_floats(H, HV, K, V, TS) * 4 <= stash[0].numel() * stash.element_size(), "stash row too small"
    assert state.stride(-1) == 1 and state.stride(-2) == K and state.stride(-3) == V * K
    BV = BV or DEF["BV"]
    num_warps = num_warps or DEF["warps"]
    prec = prec or DEF["prec"]
    prec_s = prec_s or DEF["prec_s"]
    state_bf16 = state.dtype == torch.bfloat16
    if not state_bf16 and (prec.startswith("bf16") or prec == "tf32x2"):
        prec = "tf32x3"
    # dd2: prec_s = 'A' or 'A+O' or 'A+O+T' (A: KK/UT/WT, O: QK and Vn P^T, T: the 16x16 inverse products)
    _ps = prec_s.split('+')
    pa = _ps[0]
    po = _ps[1] if len(_ps) > 1 else pa
    pt = _ps[2] if len(_ps) > 2 else 'ieee'
    q, sq = _tok_view(q, K)
    k, sk = _tok_view(k, K)
    v, sv = _tok_view(v, V)
    if a.stride(-1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1:
        b = b.contiguous()
    o = q.new_empty(v.shape)
    st = stash_f32(stash)
    if prec == "raw":
        assert state_bf16 and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16, "raw mode needs bf16 q/k/state"
        gdn_b_verify_raw_kernel[(triton.cdiv(V, BV), N * HV)](
            q, k, v, a, b, A_log, dt_bias, o,
            state, state_idx, state.stride(0),
            st, stash_rows, st.stride(0),
            cu_seqlens, float(K ** -0.5 if scale is None else scale), float(softplus_beta), float(softplus_threshold),
            sq, sk, sv, a.stride(-2), b.stride(-2), HV * V,
            TP=TP, TS=TS, H=H, HV=HV, K=K, V=V, BV=BV, PREC_A=pa, PREC_O=po, PREC_T=pt, NEED_A8=T > 8,
            EARLY_S=os.environ.get('DD2_EARLY_S', '0') == '1', WRITE_STASH=write_stash, num_warps=num_warps,
            num_stages=1)
        return o
    gdn_b_verify_kernel[(triton.cdiv(V, BV), N * HV)](
        q, k, v, a, b, A_log, dt_bias, o,
        state, state_idx, state.stride(0),
        st, stash_rows, st.stride(0),
        cu_seqlens, float(K ** -0.5 if scale is None else scale), float(softplus_beta), float(softplus_threshold),
        sq, sk, sv, a.stride(-2), b.stride(-2), HV * V,
        TP=TP, TS=TS, H=H, HV=HV, K=K, V=V, BV=BV, PREC=prec, PREC_A=pa, PREC_O=po, PREC_T=pt, NEED_A8=T > 8, EARLY_S=os.environ.get('DD2_EARLY_S', '0') == '1',
        WRITE_STASH=write_stash, STATE_BF16=state_bf16, num_warps=num_warps, num_stages=1)
    return o


def gdn_b_fold_all_layers(temporal, stash_all, state_idx, last_step, track_idx=None, track_step=None, *,
                          num_k_heads, stash_rows=None, BV=None, num_warps=None, null_id=-1):
    """temporal [L, slots, HV, V, K] (in place); stash_all [L, rows, TS, HV, V, K] (intermediate_ssm, written by
    gdn_b_verify); state_idx [>=B] (-1 = skip); last_step [B] = accepted - 1 (-1 = keep S0); track_idx/track_step
    [B] (extra_buffer: state after track_step+1 tokens -> slot track_idx; -1 = none). Row of request i = i unless
    stash_rows is given. One launch, grid (V/BV, B, L*HV)."""
    L, _, HV, V, K = temporal.shape
    assert temporal.stride(-1) == 1 and temporal.stride(-2) == K and temporal.stride(-3) == V * K
    TS = stash_all.shape[2]
    B = last_step.shape[0]
    if B == 0:
        return
    BV = BV or DEF["fold_BV"]
    num_warps = num_warps or DEF["fold_warps"]
    st = stash_f32(stash_all)
    has_track = track_idx is not None and track_step is not None
    gdn_b_fold_kernel[(triton.cdiv(V, BV), B, L * HV)](
        temporal, st, state_idx, last_step,
        track_idx if has_track else state_idx, track_step if has_track else last_step,
        stash_rows if stash_rows is not None else state_idx,
        temporal.stride(0), temporal.stride(1), st.stride(0), st.stride(1),
        TS=TS, H=num_k_heads, HV=HV, K=K, V=V, BV=BV, HAS_TRACK=has_track, HAS_ROWS=stash_rows is not None,
        NULL_ID=null_id, num_warps=num_warps, num_stages=1)


def read_stash(stash_layer_row, H, HV, K, V):
    """Torch view of one request's stash (tests): stash_layer_row = intermediate[row] [TS, HV, V, K] ->
    Vn [TS, HV, V], kn [TS, H, K], g [TS, HV] (fp32)."""
    TS = stash_layer_row.shape[0]
    f = stash_f32(stash_layer_row.reshape(-1))
    o1, o2 = HV * TS * V, HV * TS * V + H * TS * K
    vn = f[:o1].view(HV, TS, V).transpose(0, 1)
    kn = f[o1:o2].view(H, TS, K).transpose(0, 1)
    g = f[o2:o2 + HV * TS].view(HV, TS).transpose(0, 1)
    return vn, kn, g


# ============================================================================================== SGLang runtime patch
_STATE = {"installed": False, "H": None, "verify_calls": 0}


def install():
    """Monkeypatch the installed SGLang (idempotent). Hooks:
      1. sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.target_verify -> gdn_b_verify
         (linear chains only; tree drafts raise: DFLASH never builds trees).
      2. sglang.srt.layers.attention.hybrid_linear_attn_backend.scatter_mamba_states_after_mtp_verify (the name
         HybridLinearAttnBackend.update_mamba_state_after_mtp_verify calls = the DFLASH / spec commit) ->
         gdn_b_fold_all_layers on (temporal, intermediate_ssm) + the stock conv-window scatter
         (fused_conv_window_scatter_with_mask, exactly like SGLang's commit_gdn_replayssm_fold_after_verify).
         Falls back to the stock scatter until a B verify has run in this process."""
    if _STATE["installed"]:
        return
    import logging

    import sglang.srt.layers.attention.hybrid_linear_attn_backend as hlab
    from sglang.srt.layers.attention.linear.kernels import gdn_triton

    log = logging.getLogger("gdnk_b")
    cls = gdn_triton.TritonGDNKernel
    if not getattr(cls, "_gdnk_b", False):
        def target_verify(self, A_log, dt_bias, q, k, v, a, b, *, ssm_states, cache_indices, query_start_loc,
                          intermediate_states_buffer=None, intermediate_state_indices=None, cache_steps=None,
                          retrieve_parent_token=None, **kwargs):
            if retrieve_parent_token is not None or intermediate_states_buffer is None:
                raise RuntimeError("gdnk-B (SGL_GDN_B=1): tree verify / missing intermediate_ssm is not supported")
            _STATE["H"] = int(q.shape[-2])
            _STATE["verify_calls"] += 1
            return gdn_b_verify(A_log, dt_bias, q, k, v, a, b, ssm_states, cache_indices, query_start_loc,
                                intermediate_states_buffer, intermediate_state_indices)
        cls._gdnk_b_orig_target_verify = cls.target_verify
        cls.target_verify = target_verify
        cls._gdnk_b = True
    if not getattr(hlab, "_gdnk_b", False):
        orig = hlab.scatter_mamba_states_after_mtp_verify

        def scatter(mamba_caches, state_indices, last_correct_step_indices, mamba_track_indices=None,
                    mamba_steps_to_track=None):
            if _STATE["verify_calls"] == 0 or getattr(mamba_caches, "intermediate_ssm", None) is None:
                return orig(mamba_caches, state_indices, last_correct_step_indices, mamba_track_indices,
                            mamba_steps_to_track)
            from sglang.kernels.ops.mamba.mamba_state_scatter_triton import fused_conv_window_scatter_with_mask

            has_track = mamba_track_indices is not None and mamba_steps_to_track is not None
            gdn_b_fold_all_layers(mamba_caches.temporal, mamba_caches.intermediate_ssm, state_indices,
                                  last_correct_step_indices, mamba_track_indices if has_track else None,
                                  mamba_steps_to_track if has_track else None, num_k_heads=_STATE["H"])
            for conv_states, interm in zip(mamba_caches.conv, mamba_caches.intermediate_conv_window):
                fused_conv_window_scatter_with_mask(conv_states, interm, state_indices, last_correct_step_indices)
                if has_track:
                    fused_conv_window_scatter_with_mask(conv_states, interm, mamba_track_indices,
                                                        mamba_steps_to_track)
        hlab._gdnk_b_orig_scatter = orig
        hlab.scatter_mamba_states_after_mtp_verify = scatter
        hlab._gdnk_b = True
    _STATE["installed"] = True
    log.warning("gdnk-B installed: GDN target_verify -> chunked WY verify (prec %s/%s, BV %d, warps %d), commit -> "
                "fold (BV %d, warps %d)", DEF["prec"], DEF["prec_s"], DEF["BV"], DEF["warps"], DEF["fold_BV"],
                DEF["fold_warps"])
