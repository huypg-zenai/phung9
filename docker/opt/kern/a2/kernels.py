"""A2 (output-identical) kernels for the DFLASH GDN verify step: own Triton re-implementations, written from the math
of the GDN target-verify + causal conv1d update, that keep TODAY's per-element arithmetic order so the results can be
bitwise identical (proved or refuted on the GPU by a2_bench.py, never assumed).

  verify(...)        GDN target-verify (sigmoid gating + qk l2norm + delta rule, 8 per-draft states) with knobs that do
                     not touch the arithmetic: BV / num_warps (only splits that keep the K-axis reduction tree:
                     8 elements per thread x 16 lanes), num_stages, static unroll of the token loop, grid order,
                     cache hint on the intermediate-state stores, maxnreg.
  conv_update(...)   causal conv1d update for the verify step: one CTA = (request, block of channels) computes all T
                     tokens as one [T, BLOCK_C] tile, same per-element op order as today (acc = 0|bias, then
                     acc += x_j * w_j for the 4 taps oldest first, then silu), coalesced loads, and in the production
                     de-duplicated window layout writes every window element ONCE (today: 3 overlapping stores).
  fused_conv_verify(...)  conv1d computed inside the verify kernel (per token, rolling 3-input window in registers);
                     conv output rounded to the activation dtype exactly where today stores it, so q/k/v are the same
                     values today's verify loads. Writes the de-duplicated conv windows. Only valid when today's conv
                     does not write conv_state during verify (a2_bench.py detects that).

Arithmetic-order rules used (see ANALYSIS.txt): keep the [BK, BV] tile orientation (K on axis 0, reductions over axis 0),
the same expression shapes (so ptxas makes the same mul+add -> FMA contractions), tl.exp / division / sqrt as today
(Triton's approx lowerings), `zeros + load` for the initial state (turns -0.0 into +0.0 like today), and the same
1-D q/k l2norm. Shapes: K = V = 128, any HV multiple of H, any number of requests / tokens.
"""
try:
    import torch
except ImportError:  # offline_ir_check.py compiles the kernels without torch / GPU
    torch = None
import triton
import triton.language as tl

# ----------------------------------------------------------------------------------------------- config grids
# Verify: BV / nw must keep 2 lanes per warp on the V axis (BV // nw >= 2) so Triton's layout keeps 16 lanes x 8
# elements on K (the reduction tree of today). V0 = exact replica of today's launch (sanity: must be 100% bitwise and
# the same speed as today).
VERIFY_GRID = [
    dict(name="V0_bv32w1", BV=32, nw=1, ns=3, unroll=0, gfast=0, store=0),
    dict(name="bv32w1_u", BV=32, nw=1, ns=3, unroll=1, gfast=0, store=0),
    dict(name="bv32w1_u_g", BV=32, nw=1, ns=3, unroll=1, gfast=1, store=0),
    dict(name="bv32w1_ns1", BV=32, nw=1, ns=1, unroll=0, gfast=0, store=0),
    dict(name="bv16w1", BV=16, nw=1, ns=3, unroll=0, gfast=0, store=0),
    dict(name="bv16w1_u", BV=16, nw=1, ns=3, unroll=1, gfast=0, store=0),
    dict(name="bv16w1_u_ef", BV=16, nw=1, ns=3, unroll=1, gfast=0, store=2),
    dict(name="bv32w2_u", BV=32, nw=2, ns=3, unroll=1, gfast=0, store=0),
    dict(name="bv64w2_u", BV=64, nw=2, ns=3, unroll=1, gfast=0, store=0),
    dict(name="bv64w4_u", BV=64, nw=4, ns=3, unroll=1, gfast=0, store=0),
    dict(name="bv128w4_u", BV=128, nw=4, ns=3, unroll=1, gfast=0, store=0),
    dict(name="bv128w8_u", BV=128, nw=8, ns=3, unroll=1, gfast=0, store=0),
]
# offline sm_90 compile (offline_ir_check.py, triton 3.7.1) pruned: *_cs on BV=32 and maxnreg variants spill
# (stack 256..1176 B); BV=16 + unroll + cs/gfast change the K-reduce layout to [4,1]/[32,1] (not bitwise).
# Still kept as evidence: bv16w1_u (layout [4,1]/[32,1]), nw>=2 (cross-warp l2norm reduce) - expected NOT bitwise.
VERIFY_QUICK = ["V0_bv32w1", "bv32w1_u", "bv16w1", "bv16w1_u_ef"]

CONV_GRID = [
    dict(name="c128w4", BLOCK_C=128, nw=4),
    dict(name="c256w4", BLOCK_C=256, nw=4),
    dict(name="c256w8", BLOCK_C=256, nw=8),
    dict(name="c512w8", BLOCK_C=512, nw=8),
    dict(name="c64w2", BLOCK_C=64, nw=2),
    dict(name="c128w2", BLOCK_C=128, nw=2),
    dict(name="c1024w8", BLOCK_C=1024, nw=8),
]
CONV_QUICK = ["c128w4", "c256w4"]
# arithmetic variants of the conv (which one is TODAY's is detected by a2_bench.py: bitwise match on the outputs)
#   prod_f32 = 0: product in the activation dtype (bf16 x bf16 -> bf16, then fp32 add)   [vLLM-lineage kernel]
#   prod_f32 = 1: product in fp32 (ptxas may contract into an FMA)
#   silu = 0: acc / (1 + exp(-acc))      silu = 1: acc * sigmoid(acc)
CONV_ARITH = [dict(prod_f32=0, silu=0), dict(prod_f32=1, silu=0), dict(prod_f32=0, silu=1), dict(prod_f32=1, silu=1)]


LAST_COMPILED = {}   # last CompiledKernel per launcher (a2_bench.py reads registers / spills from it)


# ----------------------------------------------------------------------------------------------- shared helpers
@triton.jit
def _gdn_gate(b_a, b_dt, b_A_log, b_b, softplus_beta, softplus_threshold):
    # g = -exp(A_log) * softplus(a + dt_bias) (beta / threshold form), beta = sigmoid(b); same expression shapes as today
    x = b_a + b_dt
    beta_x = softplus_beta * x
    softplus_x = tl.where(beta_x <= softplus_threshold, (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)), x)
    b_g = -tl.exp(b_A_log) * softplus_x
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    return b_g, b_beta


@triton.jit
def _gdn_delta(b_h, b_q, b_k, b_v, b_g, b_beta, scale):
    # one token of the gated delta rule on a [BK, BV] fp32 state tile (K on axis 0)
    b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
    b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
    b_q = b_q * scale
    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[:, None], 0)
    b_v *= b_beta
    b_h += b_k[:, None] * b_v[None, :]
    b_o = tl.sum(b_h * b_q[:, None], 0)
    return b_h, b_o


@triton.jit
def _store_state(p, val, mask, STORE_POLICY: tl.constexpr):
    if STORE_POLICY == 1:
        tl.store(p, val, mask=mask, cache_modifier=".cs")
    elif STORE_POLICY == 2:
        tl.store(p, val, mask=mask, eviction_policy="evict_first")
    else:
        tl.store(p, val, mask=mask)


@triton.jit
def _conv_tap(acc, e, w, PROD_F32: tl.constexpr):
    if PROD_F32:
        acc += e.to(tl.float32) * w.to(tl.float32)
    else:
        acc += e * w
    return acc


@triton.jit
def _silu(acc, SILU_MODE: tl.constexpr):
    if SILU_MODE == 1:
        acc = acc * tl.sigmoid(acc)
    else:
        acc = acc / (1 + tl.exp(-acc))
    return acc


# ----------------------------------------------------------------------------------------------- GDN verify
@triton.jit
def a2_gdn_verify_kernel(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    q, k, v, b, o,
    h0, h0_idx, stride_h0,
    inter, inter_idx, stride_inter_row, stride_inter_step,
    cu_seqlens, scale,
    stride_a, stride_q, stride_k, stride_v, stride_b,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, TPR: tl.constexpr,
    UNROLL: tl.constexpr, GRID_NH_FAST: tl.constexpr, STORE_POLICY: tl.constexpr, WRITE_INTER: tl.constexpr,
):
    if GRID_NH_FAST:
        i_nh = tl.program_id(0)
        i_v = tl.program_id(1)
    else:
        i_v = tl.program_id(0)
        i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T_n = eos - bos

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_a = a + bos * stride_a + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v

    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_idx + i_n).to(tl.int64)
    if idx >= 0:
        p_h0 = h0 + idx * stride_h0 + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    row = tl.load(inter_idx + i_n).to(tl.int64)
    p_i = inter + row * stride_inter_row + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
    mask_i = mask_h & (row >= 0)

    if UNROLL:
        for t in tl.static_range(TPR):
            valid = t < T_n
            b_q = tl.load(p_q + t * stride_q, mask=mask_k & valid, other=0).to(tl.float32)
            b_k = tl.load(p_k + t * stride_k, mask=mask_k & valid, other=0).to(tl.float32)
            b_v = tl.load(p_v + t * stride_v, mask=mask_v & valid, other=0).to(tl.float32)
            b_b = tl.load(p_b + t * stride_b, mask=valid, other=0).to(tl.float32)
            b_a = tl.load(p_a + t * stride_a, mask=valid, other=0).to(tl.float32)
            b_g, b_beta = _gdn_gate(b_a, b_dt, b_A_log, b_b, softplus_beta, softplus_threshold)
            b_h, b_o = _gdn_delta(b_h, b_q, b_k, b_v, b_g, b_beta, scale)
            tl.store(p_o + t * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v & valid)
            if WRITE_INTER:
                _store_state(p_i + t * stride_inter_step, b_h.to(p_i.dtype.element_ty), mask_i & valid, STORE_POLICY)
    else:
        t = 0
        for _ in range(0, T_n):
            b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_b = tl.load(p_b).to(tl.float32)
            b_a = tl.load(p_a).to(tl.float32)
            b_g, b_beta = _gdn_gate(b_a, b_dt, b_A_log, b_b, softplus_beta, softplus_threshold)
            b_h, b_o = _gdn_delta(b_h, b_q, b_k, b_v, b_g, b_beta, scale)
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            if WRITE_INTER:
                _store_state(p_i + t * stride_inter_step, b_h.to(p_i.dtype.element_ty), mask_i, STORE_POLICY)
            t += 1
            p_q += stride_q
            p_k += stride_k
            p_v += stride_v
            p_b += stride_b
            p_a += stride_a
            p_o += HV * V


def _launch_opts(cfg):
    opts = dict(num_warps=cfg["nw"], num_stages=cfg.get("ns", 3))
    if cfg.get("maxnreg"):
        opts["maxnreg"] = cfg["maxnreg"]
    return opts


def verify(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, initial_state_source,
           initial_state_indices, cu_seqlens, intermediate_states_buffer, intermediate_state_indices, scale=None,
           cfg=None, tokens_per_req=None):
    """Drop-in for today's verify call (GDN, varlen, no tree, no ring, disable_state_update=True).
    q, k: [1, Ntok, H, K]; v: [1, Ntok, HV, V]; a, b: [Ntok, HV]; state pool [slots, HV, V, K] (any slot stride);
    intermediate buffer [rows, steps, HV, V, K] (inner dims contiguous). Returns o [1, Ntok, HV, V]."""
    cfg = cfg or VERIFY_GRID[0]
    _, Ntok, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.shape[0] - 1
    BK, BV = triton.next_power_of_2(K), cfg["BV"]
    assert BK == K and V % BV == 0 and BV // cfg["nw"] >= 2
    if scale is None:
        scale = K ** -0.5
    if tokens_per_req is None:
        tokens_per_req = max(1, Ntok // max(N, 1))
    o = q.new_empty(v.shape)
    inter = intermediate_states_buffer
    write_inter = inter is not None
    if write_inter:
        assert inter.stride(-1) == 1 and inter.stride(-2) == K and inter.stride(-3) == K * V
    nv = V // BV
    grid = (N * HV, nv) if cfg.get("gfast") else (nv, N * HV)
    stride_a = a.stride()[-2]
    LAST_COMPILED["verify"] = a2_gdn_verify_kernel[grid](
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        q, k, v, b, o,
        initial_state_source, initial_state_indices, initial_state_source.stride(0),
        inter if write_inter else o, intermediate_state_indices if write_inter else initial_state_indices,
        inter.stride(0) if write_inter else 0, inter.stride(1) if write_inter else 0,
        cu_seqlens, scale,
        stride_a, q.stride()[1], k.stride()[1], v.stride()[1], b.stride()[-2],
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, TPR=tokens_per_req,
        UNROLL=bool(cfg.get("unroll")), GRID_NH_FAST=bool(cfg.get("gfast")), STORE_POLICY=int(cfg.get("store", 0)),
        WRITE_INTER=write_inter, **_launch_opts(cfg))
    return o


# ----------------------------------------------------------------------------------------------- conv1d update
@triton.jit
def a2_conv_verify_kernel(
    x, stride_x_b, stride_x_c, stride_x_t,
    w, stride_w_c, stride_w_j, bias,
    cs, cs_idx, stride_cs_s, stride_cs_c, stride_cs_j,
    win, win_idx, stride_win_s, stride_win_t, stride_win_c, stride_win_j,
    out, stride_o_b, stride_o_c, stride_o_t,
    C,
    T: tl.constexpr, TP: tl.constexpr, TPW: tl.constexpr, W: tl.constexpr, BLOCK_C: tl.constexpr,
    HAS_BIAS: tl.constexpr, SILU: tl.constexpr, PROD_F32: tl.constexpr, SILU_MODE: tl.constexpr,
    WIN_MODE: tl.constexpr, WRITE_STATE: tl.constexpr, INPLACE: tl.constexpr,
):
    """One CTA = (block of BLOCK_C channels, request). E = [state_0 .. state_{W-2}, x_0 .. x_{T-1}] per channel;
    out_t = silu(acc), acc = (bias|0) + x(E[t])*w_0 + ... + x(E[t+W-1])*w_{W-1} (taps oldest first, today's order).
    WIN_MODE 1 = de-duplicated window buffer (window of step t = phys[c, t : t+W-1]): phys[c, p] = E[p+1], each
    element written once. WIN_MODE 2 = dense [rows, T, C, W-1]. WIN_MODE 0 = no window.
    Order: all reads of x / old state happen before the state write and before the (possibly in-place) out store."""
    i_c = tl.program_id(0)
    i_b = tl.program_id(1)
    oc = i_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mc = oc < C
    slot = tl.load(cs_idx + i_b).to(tl.int64)
    has = slot >= 0
    xb = x + i_b * stride_x_b
    csb = cs + slot * stride_cs_s
    ot = tl.arange(0, TP)
    mt = ot < T

    if HAS_BIAS:
        acc0 = tl.load(bias + oc, mask=mc, other=0.0).to(tl.float32)
    else:
        acc0 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc = tl.broadcast_to(acc0[None, :], (TP, BLOCK_C))
    for j in tl.static_range(W):
        r = ot + (j - (W - 1))                                     # index into x; r < 0 -> state[W-1+r]
        xin = (r >= 0) & (r < T)
        xv = tl.load(xb + r[:, None] * stride_x_t + oc[None, :] * stride_x_c, mask=xin[:, None] & mc[None, :],
                     other=0.0)
        sv = tl.load(csb + (r + (W - 1))[:, None] * stride_cs_j + oc[None, :] * stride_cs_c,
                     mask=(r < 0)[:, None] & mc[None, :] & has, other=0.0)
        e = tl.where((r >= 0)[:, None], xv, sv.to(xv.dtype))
        wj = tl.load(w + oc * stride_w_c + j * stride_w_j, mask=mc, other=0.0)
        acc = _conv_tap(acc, e, wj[None, :], PROD_F32)
    if SILU:
        acc = _silu(acc, SILU_MODE)

    if WIN_MODE != 0:
        row = tl.load(win_idx + i_b).to(tl.int64)
        wbase = win + row * stride_win_s
        okw = has & (row >= 0)
        if WIN_MODE == 1:
            op = tl.arange(0, TPW)
            rr = op + 1 - (W - 1)
            xv = tl.load(xb + rr[:, None] * stride_x_t + oc[None, :] * stride_x_c,
                         mask=((rr >= 0) & (rr < T))[:, None] & mc[None, :], other=0.0)
            sv = tl.load(csb + (rr + (W - 1))[:, None] * stride_cs_j + oc[None, :] * stride_cs_c,
                         mask=(rr < 0)[:, None] & mc[None, :] & has, other=0.0)
            e = tl.where((rr >= 0)[:, None], xv, sv.to(xv.dtype))
            tl.store(wbase + op[:, None] * stride_win_j + oc[None, :] * stride_win_c, e.to(win.dtype.element_ty),
                     mask=(op < T + W - 2)[:, None] & mc[None, :] & okw)
        else:
            for jj in tl.static_range(W - 1):
                rr = ot + 1 + jj - (W - 1)
                xv = tl.load(xb + rr[:, None] * stride_x_t + oc[None, :] * stride_x_c,
                             mask=((rr >= 0) & (rr < T))[:, None] & mc[None, :], other=0.0)
                sv = tl.load(csb + (rr + (W - 1))[:, None] * stride_cs_j + oc[None, :] * stride_cs_c,
                             mask=(rr < 0)[:, None] & mc[None, :] & has, other=0.0)
                e = tl.where((rr >= 0)[:, None], xv, sv.to(xv.dtype))
                tl.store(wbase + ot[:, None] * stride_win_t + oc[None, :] * stride_win_c + jj * stride_win_j,
                         e.to(win.dtype.element_ty), mask=mt[:, None] & mc[None, :] & okw)

    if WRITE_STATE:
        # new state = last W-1 entries of E (= E[T .. T+W-2]); every read of the old state happens before the barrier
        os_ = tl.arange(0, TPW)
        rr = os_ + T - (W - 1)
        xv = tl.load(xb + rr[:, None] * stride_x_t + oc[None, :] * stride_x_c,
                     mask=((rr >= 0) & (rr < T) & (os_ < W - 1))[:, None] & mc[None, :], other=0.0)
        sv = tl.load(csb + (rr + (W - 1))[:, None] * stride_cs_j + oc[None, :] * stride_cs_c,
                     mask=((rr < 0) & (os_ < W - 1))[:, None] & mc[None, :] & has, other=0.0)
        e = tl.where((rr >= 0)[:, None], xv, sv.to(xv.dtype))
        tl.debug_barrier()
        tl.store(csb + os_[:, None] * stride_cs_j + oc[None, :] * stride_cs_c, e.to(cs.dtype.element_ty),
                 mask=(os_ < W - 1)[:, None] & mc[None, :] & has)

    if INPLACE:
        tl.debug_barrier()
    tl.store(out + i_b * stride_o_b + ot[:, None] * stride_o_t + oc[None, :] * stride_o_c,
             acc.to(out.dtype.element_ty), mask=mt[:, None] & mc[None, :])


def window_mode(win):
    """1 = de-duplicated sliding-window view (step stride == window-column stride), 2 = dense, 0 = none."""
    if win is None:
        return 0
    return 1 if win.stride(1) == win.stride(3) else 2


def conv_update(x, conv_state, weight, bias, activation, conv_state_indices, intermediate_conv_window,
                intermediate_state_indices, cfg=None, arith=None, write_state=False, inplace=False):
    """Drop-in for today's verify conv call: x [B, C, T] (any strides; production = [B,T,C].transpose(1,2)),
    conv_state [slots, C, W-1], weight [C, W], window [rows, T, C, W-1] (dedup as_strided view or dense).
    Returns out with x's strides (empty_like) in x's dtype, or x itself when inplace=True (mirrors a kernel that
    overwrites x, like the vLLM-lineage one)."""
    cfg = cfg or CONV_GRID[0]
    arith = arith or CONV_ARITH[0]
    Bn, C, T = x.shape
    W = weight.shape[1]
    if x.dtype != conv_state.dtype:
        x = x.to(conv_state.dtype)
    out = x if inplace else torch.empty_like(x)
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    silu = activation in ("silu", "swish")
    wm = window_mode(intermediate_conv_window)
    win = intermediate_conv_window if wm else out
    widx = intermediate_state_indices if wm else conv_state_indices
    BLOCK_C = cfg["BLOCK_C"]
    grid = (triton.cdiv(C, BLOCK_C), Bn)
    LAST_COMPILED["conv"] = a2_conv_verify_kernel[grid](
        x, x.stride(0), x.stride(1), x.stride(2),
        weight, weight.stride(0), weight.stride(1), bias if bias is not None else weight,
        conv_state, conv_state_indices, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        win, widx, win.stride(0) if wm else 0, win.stride(1) if wm else 0, win.stride(2) if wm else 0,
        win.stride(3) if wm else 0,
        out, out.stride(0), out.stride(1), out.stride(2),
        C,
        T=T, TP=triton.next_power_of_2(T), TPW=triton.next_power_of_2(max(T + W - 2, W - 1)), W=W, BLOCK_C=BLOCK_C,
        HAS_BIAS=bias is not None, SILU=silu, PROD_F32=bool(arith["prod_f32"]), SILU_MODE=int(arith["silu"]),
        WIN_MODE=wm, WRITE_STATE=bool(write_state), INPLACE=bool(inplace), num_warps=cfg["nw"], num_stages=1)
    return out


# ----------------------------------------------------------------------------------------------- fused conv + verify
@triton.jit
def _conv_token(c0, c1, c2, xt, w0, w1, w2, w3, acc0, PROD_F32: tl.constexpr, SILU: tl.constexpr,
                SILU_MODE: tl.constexpr, OUT_DTYPE: tl.constexpr):
    acc = acc0
    acc = _conv_tap(acc, c0, w0, PROD_F32)
    acc = _conv_tap(acc, c1, w1, PROD_F32)
    acc = _conv_tap(acc, c2, w2, PROD_F32)
    acc = _conv_tap(acc, xt, w3, PROD_F32)
    if SILU:
        acc = _silu(acc, SILU_MODE)
    return acc.to(OUT_DTYPE).to(tl.float32)


@triton.jit
def a2_fused_conv_verify_kernel(
    A_log, a, dt_bias, softplus_beta, softplus_threshold,
    xr, stride_x_tok, w, stride_w_c, stride_w_j, bias,
    cs, stride_cs_s, stride_cs_c, stride_cs_j,
    win, win_idx, stride_win_s, stride_win_c, stride_win_j,
    b, o, h0, h0_idx, stride_h0,
    inter, inter_idx, stride_inter_row, stride_inter_step,
    cu_seqlens, scale, stride_a, stride_b,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    TPR: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, PROD_F32: tl.constexpr, SILU_MODE: tl.constexpr,
    WRITE_WIN: tl.constexpr, STORE_POLICY: tl.constexpr, OUT_DTYPE: tl.constexpr,
):
    """Width-4 conv only. xr = raw (pre-conv) mixed_qkv [Ntok, C], channels q | k | v. Every CTA recomputes the conv
    for the 2x128 q/k channels of its k-head and its BV v channels (4 MACs per element per token, negligible);
    windows of q/k channels are written by the CTA with i_v == 0 and the first v-head of the group, v windows by
    every CTA for its own channels (each element written once). conv_state is NOT written (see module doc)."""
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T_n = eos - bos
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    c_q = i_h * K + o_k
    c_k = H * K + i_h * K + o_k
    c_v = 2 * H * K + i_hv * V + o_v

    slot = tl.load(h0_idx + i_n).to(tl.int64)
    has = slot >= 0
    csb = cs + slot * stride_cs_s
    q0 = tl.load(csb + c_q * stride_cs_c + 0 * stride_cs_j, mask=mask_k & has, other=0.0)
    q1 = tl.load(csb + c_q * stride_cs_c + 1 * stride_cs_j, mask=mask_k & has, other=0.0)
    q2 = tl.load(csb + c_q * stride_cs_c + 2 * stride_cs_j, mask=mask_k & has, other=0.0)
    k0 = tl.load(csb + c_k * stride_cs_c + 0 * stride_cs_j, mask=mask_k & has, other=0.0)
    k1 = tl.load(csb + c_k * stride_cs_c + 1 * stride_cs_j, mask=mask_k & has, other=0.0)
    k2 = tl.load(csb + c_k * stride_cs_c + 2 * stride_cs_j, mask=mask_k & has, other=0.0)
    v0 = tl.load(csb + c_v * stride_cs_c + 0 * stride_cs_j, mask=mask_v & has, other=0.0)
    v1 = tl.load(csb + c_v * stride_cs_c + 1 * stride_cs_j, mask=mask_v & has, other=0.0)
    v2 = tl.load(csb + c_v * stride_cs_c + 2 * stride_cs_j, mask=mask_v & has, other=0.0)
    wq0 = tl.load(w + c_q * stride_w_c + 0 * stride_w_j, mask=mask_k, other=0.0)
    wq1 = tl.load(w + c_q * stride_w_c + 1 * stride_w_j, mask=mask_k, other=0.0)
    wq2 = tl.load(w + c_q * stride_w_c + 2 * stride_w_j, mask=mask_k, other=0.0)
    wq3 = tl.load(w + c_q * stride_w_c + 3 * stride_w_j, mask=mask_k, other=0.0)
    wk0 = tl.load(w + c_k * stride_w_c + 0 * stride_w_j, mask=mask_k, other=0.0)
    wk1 = tl.load(w + c_k * stride_w_c + 1 * stride_w_j, mask=mask_k, other=0.0)
    wk2 = tl.load(w + c_k * stride_w_c + 2 * stride_w_j, mask=mask_k, other=0.0)
    wk3 = tl.load(w + c_k * stride_w_c + 3 * stride_w_j, mask=mask_k, other=0.0)
    wv0 = tl.load(w + c_v * stride_w_c + 0 * stride_w_j, mask=mask_v, other=0.0)
    wv1 = tl.load(w + c_v * stride_w_c + 1 * stride_w_j, mask=mask_v, other=0.0)
    wv2 = tl.load(w + c_v * stride_w_c + 2 * stride_w_j, mask=mask_v, other=0.0)
    wv3 = tl.load(w + c_v * stride_w_c + 3 * stride_w_j, mask=mask_v, other=0.0)
    if HAS_BIAS:
        aq = tl.load(bias + c_q, mask=mask_k, other=0.0).to(tl.float32)
        ak = tl.load(bias + c_k, mask=mask_k, other=0.0).to(tl.float32)
        av = tl.load(bias + c_v, mask=mask_v, other=0.0).to(tl.float32)
    else:
        aq = tl.zeros((BK,), dtype=tl.float32)
        ak = tl.zeros((BK,), dtype=tl.float32)
        av = tl.zeros((BV,), dtype=tl.float32)

    row = tl.load(win_idx + i_n).to(tl.int64)
    wb = win + row * stride_win_s
    okw = has & (row >= 0)
    wqk = okw & (i_v == 0) & ((i_hv % (HV // H)) == 0)
    if WRITE_WIN:
        tl.store(wb + c_q * stride_win_c + 0 * stride_win_j, q1, mask=mask_k & wqk)
        tl.store(wb + c_q * stride_win_c + 1 * stride_win_j, q2, mask=mask_k & wqk)
        tl.store(wb + c_k * stride_win_c + 0 * stride_win_j, k1, mask=mask_k & wqk)
        tl.store(wb + c_k * stride_win_c + 1 * stride_win_j, k2, mask=mask_k & wqk)
        tl.store(wb + c_v * stride_win_c + 0 * stride_win_j, v1, mask=mask_v & okw)
        tl.store(wb + c_v * stride_win_c + 1 * stride_win_j, v2, mask=mask_v & okw)

    p_x = xr + bos * stride_x_tok
    p_b = b + bos * stride_b + i_hv
    p_a = a + bos * stride_a + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if slot >= 0:
        p_h0 = h0 + slot * stride_h0 + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    irow = tl.load(inter_idx + i_n).to(tl.int64)
    p_i = inter + irow * stride_inter_row + i_hv * K * V + o_v[None, :] * K + o_k[:, None]
    mask_i = mask_h & (irow >= 0)

    for t in tl.static_range(TPR):
        valid = t < T_n
        xq = tl.load(p_x + t * stride_x_tok + c_q, mask=mask_k & valid, other=0.0)
        xk = tl.load(p_x + t * stride_x_tok + c_k, mask=mask_k & valid, other=0.0)
        xv = tl.load(p_x + t * stride_x_tok + c_v, mask=mask_v & valid, other=0.0)
        b_q = _conv_token(q0, q1, q2, xq, wq0, wq1, wq2, wq3, aq, PROD_F32, SILU, SILU_MODE, OUT_DTYPE)
        b_k = _conv_token(k0, k1, k2, xk, wk0, wk1, wk2, wk3, ak, PROD_F32, SILU, SILU_MODE, OUT_DTYPE)
        b_v = _conv_token(v0, v1, v2, xv, wv0, wv1, wv2, wv3, av, PROD_F32, SILU, SILU_MODE, OUT_DTYPE)
        if WRITE_WIN:
            tl.store(wb + c_q * stride_win_c + (t + 2) * stride_win_j, xq, mask=mask_k & wqk & valid)
            tl.store(wb + c_k * stride_win_c + (t + 2) * stride_win_j, xk, mask=mask_k & wqk & valid)
            tl.store(wb + c_v * stride_win_c + (t + 2) * stride_win_j, xv, mask=mask_v & okw & valid)
        q0, q1, q2 = q1, q2, xq
        k0, k1, k2 = k1, k2, xk
        v0, v1, v2 = v1, v2, xv
        b_b = tl.load(p_b + t * stride_b, mask=valid, other=0).to(tl.float32)
        b_a = tl.load(p_a + t * stride_a, mask=valid, other=0).to(tl.float32)
        b_g, b_beta = _gdn_gate(b_a, b_dt, b_A_log, b_b, softplus_beta, softplus_threshold)
        b_h, b_o = _gdn_delta(b_h, b_q, b_k, b_v, b_g, b_beta, scale)
        tl.store(p_o + t * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v & valid)
        _store_state(p_i + t * stride_inter_step, b_h.to(p_i.dtype.element_ty), mask_i & valid, STORE_POLICY)


_TL_DT = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32} if torch else {}


def fused_conv_verify(A_log, a, dt_bias, softplus_beta, softplus_threshold, mixed_qkv_raw, conv_state, weight, bias,
                      activation, b, ssm_state, cache_indices, cu_seqlens, inter, inter_idx, win, win_idx, H, HV,
                      K, V, scale=None, cfg=None, arith=None, tokens_per_req=None):
    """mixed_qkv_raw [Ntok, C] (pre-conv), channel order q (H*K) | k (H*K) | v (HV*V); width-4 conv; dedup window.
    Returns o [1, Ntok, HV, V] (same as today's conv -> split -> verify)."""
    cfg = cfg or VERIFY_GRID[0]
    arith = arith or CONV_ARITH[0]
    assert weight.shape[1] == 4 and window_mode(win) == 1 and K == triton.next_power_of_2(K)
    Ntok = mixed_qkv_raw.shape[0]
    N = cu_seqlens.shape[0] - 1
    BV = cfg["BV"]
    if scale is None:
        scale = K ** -0.5
    if tokens_per_req is None:
        tokens_per_req = max(1, Ntok // max(N, 1))
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    o = mixed_qkv_raw.new_empty(1, Ntok, HV, V)
    assert inter.stride(-1) == 1 and inter.stride(-2) == K and inter.stride(-3) == K * V
    LAST_COMPILED["fused"] = a2_fused_conv_verify_kernel[(V // BV, N * HV)](
        A_log, a, dt_bias, softplus_beta, softplus_threshold,
        mixed_qkv_raw, mixed_qkv_raw.stride(0), weight, weight.stride(0), weight.stride(1),
        bias if bias is not None else weight,
        conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
        win, win_idx, win.stride(0), win.stride(2), win.stride(3),
        b, o, ssm_state, cache_indices, ssm_state.stride(0),
        inter, inter_idx, inter.stride(0), inter.stride(1),
        cu_seqlens, scale, a.stride()[-2], b.stride()[-2],
        H=H, HV=HV, K=K, V=V, BK=K, BV=BV, TPR=tokens_per_req, HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"), PROD_F32=bool(arith["prod_f32"]), SILU_MODE=int(arith["silu"]),
        WRITE_WIN=True, STORE_POLICY=int(cfg.get("store", 0)), OUT_DTYPE=_TL_DT[mixed_qkv_raw.dtype],
        **_launch_opts(cfg))
    return o
