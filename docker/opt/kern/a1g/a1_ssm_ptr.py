"""A1 (gdnk sprint, 18/10): zero-copy GDN target-verify commit for SGLang DFLASH + mamba extra_buffer(_lazy).

Own code (not copied from SGLang). Installed by apply_a1.py as sglang/srt/mem_cache/gdnk_a1.py and imported by the
patched memory_pool.py / gdn_backend.py / hybrid_linear_attn_backend.py (installed as sglang/srt/mem_cache/gdnk_a1.py);
the micro-benchmark a1_bench.py imports this very file, so the bench times and checks the kernels the server runs.
ON as soon as the patch is applied (the base arm simply does not apply it); SGLANG_A1_SSM_PTR=0 switches it off.

Today (per verify step, see TRACE.txt): the Triton verify writes h_t (t = 0..T-1, bf16) to
intermediate_ssm[layer, batch_row, t]; after acceptance a Triton scatter copies intermediate_ssm[:, b, n_b - 1]
(48 layers x 1.5 MiB per request = 4.8 GB read + 4.8 GB written at B=64, ~3.5 ms) into temporal[:, slot_b], plus the
conv window and the radix track copies.

A1:
  layout  temporal and intermediate_ssm become two views of ONE allocation per layer,
          big[l] = [S + R*T, HV, V, K]  (S pool slots first, then R scratch rows x T steps; same bytes as today),
          so a single "physical row" index addresses both. ssm_src[slot] (int64, device) = physical row that holds
          the slot's current SSM state; identity (= home row) unless the slot was committed by a verify.
  verify  h0 is read from big[l][ssm_src[slot]] (the Triton kernel already takes an index + a slot stride), the T
          states go to scratch row r = req_pool_idx (stable for the request's lifetime) instead of the batch row.
  commit  ssm_src[slot] = S + r*T + (n-1)  (8 bytes instead of 72 MiB per request); conv window and radix track
          copies stay real copies (same bytes as today, tiny).
  in-place  the next verify of the same request reads h0 from row r step j and rewrites row r steps 0..T-1. Safe:
          every (request, head, V-tile) program (1 warp) loads its h0 tile before any store, every store depends on
          that load through the K reduction, and no other program touches that tile.
  exits   every write to a home row resets the slot's pointer (clear_slots, copy_from dst, load_cpu_copy, commit
          track copy, extend/decode forwards: their slots are materialized and their track destinations reset);
          every read of a home row outside the verify materializes first (copy_from src, get_cpu_copy, extend/decode
          forwards). Only copies move data, so states and outputs are bitwise identical to today by construction.
"""
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001  (bare CPU box: py_compile only)
    triton = None
    tl = None

ENV = "SGLANG_A1_SSM_PTR"
_LOGGED = set()


def env_on():
    return os.environ.get(ENV, "1") != "0"


def _log(msg):
    if msg in _LOGGED:
        return
    _LOGGED.add(msg)
    rank = 0
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    except Exception:  # noqa: BLE001
        pass
    if rank == 0:
        print(f"[A1 ssm-ptr] {msg}", flush=True)


def _np2(n):
    p = 16
    while p < n:
        p *= 2
    return p


# ----------------------------------------------------------------------------------------------------- Triton kernels
if triton is not None:

    @triton.jit
    def _a1_verify_idx_kernel(ci_ptr, rpi_ptr, ssm_src, h0_ptr, rows_ptr, N, S, BLOCK: tl.constexpr):
        # h0[b] = ssm_src[slot_b] (or -1 for padded rows), rows[b] = req_pool_idx_b (int32 for the verify kernels)
        o = tl.arange(0, BLOCK)
        m = o < N
        ci = tl.load(ci_ptr + o, mask=m, other=-1).to(tl.int64)
        ok = (ci >= 0) & (ci < S)
        p = tl.load(ssm_src + tl.where(ok, ci, 0), mask=m, other=0)
        tl.store(h0_ptr + o, tl.where(ok, p, -1).to(tl.int32), mask=m)
        r = tl.load(rpi_ptr + o, mask=m, other=0)
        tl.store(rows_ptr + o, r.to(tl.int32), mask=m)

    @triton.jit
    def _a1_set_ptr_kernel(ssm_src, sidx_ptr, step_ptr, rows_ptr, err_ptr, N, S, R, T, BLOCK: tl.constexpr):
        # commit: ssm_src[slot_b] = S + r_b*T + step_b for valid rows (slot >= 0, 0 <= step < T)
        o = tl.arange(0, BLOCK)
        m = o < N
        s = tl.load(sidx_ptr + o, mask=m, other=-1).to(tl.int64)
        j = tl.load(step_ptr + o, mask=m, other=-1).to(tl.int64)
        r = tl.load(rows_ptr + o, mask=m, other=-1).to(tl.int64)
        ok = m & (s >= 0) & (s < S) & (j >= 0) & (j < T) & (r >= 0) & (r < R)
        # dd2 guard: a live slot with NO accepted step whose pointer sits in its own scratch row lost its state (this
        # step's verify rewrote that row). DFLASH never does this (commit_lens >= 1); count it so the host raises.
        bad = m & (s >= 0) & (s < S) & (j < 0) & (r >= 0) & (r < R)
        cur = tl.load(ssm_src + tl.where(bad, s, 0), mask=bad, other=0)
        lost = bad & (cur >= S + r * T) & (cur < S + r * T + T)
        tl.atomic_add(err_ptr, tl.sum(lost.to(tl.int32), 0))
        tl.store(ssm_src + tl.where(ok, s, 0), S + r * T + j, mask=ok)

    @triton.jit
    def _a1_conv_copy_kernel(dst_idx_ptr, step_ptr, rows_ptr, conv, cwin,
                             c_sl, c_ss, c_sd, c_sw, w_sl, w_sr, w_st, w_sd, w_sw,
                             S, R, T, D, W, BLOCK: tl.constexpr):
        # conv[l, dst_b] = conv_window_view[l, r_b, step_b]   (any strides: dedup as_strided view or dense)
        b = tl.program_id(0)
        lyr = tl.program_id(1).to(tl.int64)
        s = tl.load(dst_idx_ptr + b).to(tl.int64)
        j = tl.load(step_ptr + b).to(tl.int64)
        r = tl.load(rows_ptr + b).to(tl.int64)
        ok = (s >= 0) & (s < S) & (j >= 0) & (j < T) & (r >= 0) & (r < R)
        if ok:
            src = cwin + lyr * w_sl + r * w_sr + j * w_st
            dst = conv + lyr * c_sl + s * c_ss
            n = D * W
            for off in range(0, n, BLOCK):
                e = off + tl.arange(0, BLOCK)
                me = e < n
                d = e // W
                w = e - d * W
                v = tl.load(src + d * w_sd + w * w_sw, mask=me)
                tl.store(dst + d * c_sd + w * c_sw, v, mask=me)

    @triton.jit
    def _a1_track_ssm_kernel(big, ssm_src, trk_ptr, step_ptr, rows_ptr, S, R, T, stride_l, stride_row, E, CH,
                             BLOCK: tl.constexpr):
        # radix track: big[l, trk_b] = big[l, S + r_b*T + step_b] (= today's intermediate[b, step] copy); the track
        # slot's home row is now its truth -> ssm_src[trk_b] = trk_b
        b = tl.program_id(0)
        lyr = tl.program_id(1).to(tl.int64)
        c = tl.program_id(2).to(tl.int64)
        t = tl.load(trk_ptr + b).to(tl.int64)
        j = tl.load(step_ptr + b).to(tl.int64)
        r = tl.load(rows_ptr + b).to(tl.int64)
        ok = (t >= 0) & (t < S) & (j >= 0) & (j < T) & (r >= 0) & (r < R)
        if ok:
            base = big + lyr * stride_l
            src = base + (S + r * T + j) * stride_row
            dst = base + t * stride_row
            start = c * CH
            for off in range(0, CH, BLOCK):
                o = start + off + tl.arange(0, BLOCK)
                mo = (o < start + CH) & (o < E)
                tl.store(dst + o, tl.load(src + o, mask=mo), mask=mo)
            if (lyr == 0) & (c == 0):
                tl.store(ssm_src + t, t)

    @triton.jit
    def _a1_mat_kernel(big, ssm_src, idx_ptr, S, stride_l, stride_row, E, CH, BLOCK: tl.constexpr):
        # materialize: big[l, s] = big[l, ssm_src[s]] when the slot is deferred (pointer != home); no-op otherwise
        b = tl.program_id(0)
        lyr = tl.program_id(1).to(tl.int64)
        c = tl.program_id(2).to(tl.int64)
        s = tl.load(idx_ptr + b).to(tl.int64)
        s_ok = (s >= 0) & (s < S)
        p = tl.load(ssm_src + tl.where(s_ok, s, 0))
        if s_ok & (p != s):
            base = big + lyr * stride_l
            src = base + p * stride_row
            dst = base + s * stride_row
            start = c * CH
            for off in range(0, CH, BLOCK):
                o = start + off + tl.arange(0, BLOCK)
                mo = (o < start + CH) & (o < E)
                tl.store(dst + o, tl.load(src + o, mask=mo), mask=mo)


# ----------------------------------------------------------------------------------------------------------- state
class A1State:
    """One combined allocation per layer: big[l] = [S pool slots | R scratch rows x T steps] x state_shape."""

    NCH = 16          # chunks per (row, layer) for the state copies (track / materialize)
    BLOCK = 4096

    def __init__(self, num_layers, num_slots, num_rows, steps, state_shape, dtype, device):
        L, S, R, T = int(num_layers), int(num_slots), int(num_rows), int(steps)
        self.L, self.S, self.R, self.T = L, S, R, T
        self.state_shape = tuple(int(x) for x in state_shape)
        self.big = torch.zeros((L, S + R * T) + self.state_shape, dtype=dtype, device=device)
        self.temporal = self.big[:, :S]                                         # [L, S, HV, V, K]
        self.intermediate = self.big[:, S:].view((L, R, T) + self.state_shape)  # [L, R, T, HV, V, K]
        self.ssm_src = torch.arange(S + 1, dtype=torch.int64, device=device)    # index S = write sink
        self.err = torch.zeros(1, dtype=torch.int32, device=device)            # dd2 guard counter (device)
        self.err_host = torch.zeros(1, dtype=torch.int32, pin_memory=str(device).startswith("cuda"))
        E = 1
        for x in self.state_shape:
            E *= x
        self.E = E
        self.stride_l = self.big.stride(0)
        self.stride_row = self.big.stride(1)
        assert self.stride_row == E, (self.stride_row, E)
        self.CH = -(-E // self.NCH)

    def gib(self):
        return self.big.numel() * self.big.element_size() / 2**30

    def logical(self, slots):
        """[L, len(slots), ...] SSM state of the given slots as the rest of SGLang would see it (debug/bench)."""
        slots = torch.as_tensor(slots, device=self.big.device, dtype=torch.int64)
        return self.big[:, self.ssm_src[slots]]


def _as_idx(st, indices):
    t = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices)
    if t.dim() == 0:
        t = t.reshape(1)
    return t.reshape(-1).to(device=st.big.device, dtype=torch.int64, non_blocking=True)


def reset_ptr(st, idx):
    """ssm_src[s] = s for every valid s in idx (no host sync; invalid entries go to the sink)."""
    if idx.numel() == 0:
        return
    w = torch.where((idx >= 0) & (idx < st.S), idx, torch.full_like(idx, st.S))
    st.ssm_src.index_put_((w,), w)


def materialize(st, idx):
    """Copy each deferred slot's state to its home row, then point the slot home (no-op for home slots)."""
    n = idx.numel()
    if n == 0:
        return
    _a1_mat_kernel[(n, st.L, st.NCH)](st.big, st.ssm_src, idx, st.S, st.stride_l, st.stride_row, st.E, st.CH,
                                      BLOCK=st.BLOCK)
    reset_ptr(st, idx)


def verify_indices(st, cache_indices, req_pool_indices):
    """-> (h0 rows into big[l] (int32, -1 = padded), scratch rows = req_pool_idx (int32))."""
    n = cache_indices.shape[0]
    h0 = torch.empty(n, dtype=torch.int32, device=cache_indices.device)
    rows = torch.empty(n, dtype=torch.int32, device=cache_indices.device)
    if n:
        _a1_verify_idx_kernel[(1,)](cache_indices, req_pool_indices, st.ssm_src, h0, rows, n, st.S,
                                    BLOCK=_np2(n))
    return h0, rows


def _conv_copy(st, conv, cwin, dst_idx, steps, rows):
    n = dst_idx.shape[0]
    if n == 0:
        return
    D, W = int(conv.shape[2]), int(conv.shape[3])
    assert tuple(cwin.shape[3:]) == (D, W) and conv.dim() == 4 and cwin.dim() == 5, (conv.shape, cwin.shape)
    cs, ws = conv.stride(), cwin.stride()
    _a1_conv_copy_kernel[(n, int(conv.shape[0]))](
        dst_idx, steps, rows, conv, cwin, cs[0], cs[1], cs[2], cs[3], ws[0], ws[1], ws[2], ws[3], ws[4],
        int(conv.shape[1]), int(cwin.shape[1]), int(cwin.shape[2]), D, W, BLOCK=1024)


def commit_raw(st, convs, cwins, state_indices, last_steps, rows, track_indices=None, track_steps=None):
    """The A1 replacement of scatter_mamba_states_after_mtp_verify (rows = req_pool_indices used by the verify)."""
    n = last_steps.shape[0]
    if n == 0:
        return
    state_indices, rows = state_indices[:n], rows[:n]
    if int(st.err_host[0]) != 0 and os.environ.get("SGLANG_A1_GUARD", "1") != "0":   # previous commit's count (async)
        raise RuntimeError(f"A1 ssm-ptr: {int(st.err_host[0])} live slot(s) committed with last_step=-1 while their "
                           "state lived in their own scratch row -> state lost. This caller is not DFLASH-like; "
                           "set SGLANG_A1_SSM_PTR=0.")
    _a1_set_ptr_kernel[(1,)](st.ssm_src, state_indices, last_steps, rows, st.err, n, st.S, st.R, st.T,
                             BLOCK=_np2(n))
    st.err_host.copy_(st.err, non_blocking=True)
    for conv, cwin in zip(convs, cwins):
        _conv_copy(st, conv, cwin, state_indices, last_steps, rows)
    if track_indices is not None and track_steps is not None:
        trk, tst = track_indices[:n], track_steps[:n]
        _a1_track_ssm_kernel[(n, st.L, st.NCH)](st.big, st.ssm_src, trk, tst, rows, st.S, st.R, st.T,
                                                st.stride_l, st.stride_row, st.E, st.CH, BLOCK=st.BLOCK)
        for conv, cwin in zip(convs, cwins):
            _conv_copy(st, conv, cwin, trk, tst, rows)


# ------------------------------------------------------------------------------------ hooks: memory_pool.MambaPool
def want(loc):
    """Called with MambaPool.__init__'s locals() right before the temporal (SSM) pool allocation."""
    if not env_on() or loc.get("speculative_num_draft_tokens") is None:
        return False          # off, or a pool that never runs TARGET_VERIFY (draft heads, PD prefill)
    reasons = []
    if loc.get("enable_linear_replayssm") or loc.get("enable_linear_replayssm_spec"):
        reasons.append("ReplaySSM is on")
    topk = loc.get("speculative_eagle_topk")
    if topk is not None and int(topk) > 1:
        reasons.append("tree drafts (topk > 1)")
    dev = str(loc.get("device"))
    if not dev.startswith("cuda") or not torch.cuda.is_available():
        reasons.append(f"device {dev}")
    if loc.get("envelope_layout"):
        reasons.append("envelope layout")
    if triton is None:
        reasons.append("triton missing")
    if reasons:
        raise RuntimeError(f"A1 patch active ({ENV}!=0) but this mamba pool config is unsupported: {'; '.join(reasons)}")
    return True


def alloc_temporal(pool, loc):
    st = A1State(loc["num_mamba_layers"], int(loc["size"]) + 1, int(loc["spec_state_size"]) + 1,
                 loc["speculative_num_draft_tokens"], loc["temporal_state_shape"], loc["ssm_dtype"], loc["device"])
    pool._a1 = st
    _log(f"combined SSM pool + verify scratch: L={st.L} slots={st.S} rows={st.R} T={st.T} state={st.state_shape} "
         f"{st.gib():.2f} GiB (same bytes as temporal + intermediate_ssm)")
    return st.temporal


def has(pool):
    return getattr(pool, "_a1", None) is not None


def take_intermediate(pool):
    return pool._a1.intermediate


def on_reset(pool, indices):
    """The pool is about to (re)write these home rows: point the slots home (clear_slots / load_cpu_copy)."""
    st = getattr(pool, "_a1", None)
    if st is not None:
        reset_ptr(st, _as_idx(st, indices))


def materialize_pool(pool, indices):
    """The pool is about to read these home rows outside a verify (get_cpu_copy)."""
    st = getattr(pool, "_a1", None)
    if st is not None:
        materialize(st, _as_idx(st, indices))


def on_copy(pool, src_indices, dst_indices):
    """copy_from: the source must be at home, the destination home row is about to be overwritten."""
    st = getattr(pool, "_a1", None)
    if st is not None:
        materialize(st, _as_idx(st, src_indices))
        reset_ptr(st, _as_idx(st, dst_indices))


# --------------------------------------------------------------------------------------- hooks: attention backends
def _backend_state(backend):
    cached = getattr(backend, "_a1_cache", None)
    if cached is not None:
        return cached[0]
    rtp = getattr(backend, "req_to_token_pool", None)
    cand = getattr(getattr(rtp, "mamba_pool", None), "_a1", None)
    st = None
    if cand is not None:
        reasons = []
        disp = getattr(backend, "kernel_dispatcher", None)
        vk = getattr(disp, "verify_kernel", None)
        if disp is None:
            reasons.append(f"backend {type(backend).__name__} is not the GDN backend")
        elif type(vk).__name__ != "TritonGDNKernel":
            reasons.append(f"verify kernel {type(vk).__name__} (A1 needs the Triton verify)")
        if int(getattr(backend, "topk", 1) or 1) > 1:
            reasons.append("topk > 1")
        if getattr(getattr(type(backend), "forward_extend", None), "_gdnk_a2", False):
            reasons.append("a2 fused forward_extend is installed (it bypasses the A1 verify indices; unset SGL_A2_FUSED)")
        if type(rtp).__name__ != "HybridReqToTokenPool":
            reasons.append(f"req pool {type(rtp).__name__} (virtual mamba slot ids)")
        if not getattr(rtp, "enable_mamba_extra_buffer", False):
            reasons.append("needs --mamba-scheduler-strategy extra_buffer(_lazy): the radix cache must snapshot "
                           "track slots, never donate the live slot")
        if getattr(rtp, "mamba_ckpt_pool", None) is not None:
            reasons.append("int8 mamba checkpoint pool")
        if int(rtp.size) > cand.R - 1:
            reasons.append(f"req pool size {rtp.size} > scratch rows {cand.R - 1}")
        if reasons:
            raise RuntimeError(f"A1 patch active ({ENV}!=0) but unsupported (set {ENV}=0 or unpatch): {'; '.join(reasons)}")
        st = cand
        backend._a1_first_layer = min(rtp.mamba_map.keys())
        _log(f"active on {type(backend).__name__}: verify reads h0 via ssm_src, scratch rows = req_pool_idx, "
             f"commit = pointer + conv copy (first GDN layer {backend._a1_first_layer})")
    backend._a1_cache = (st,)
    return st


def verify_args(backend, layer, forward_batch, ssm_states, cache_indices, intermediate_state_indices):
    """gdn_backend.forward_extend (target verify) -> (h0 source, h0 indices, scratch rows)."""
    st = _backend_state(backend)
    if st is None:
        return ssm_states, cache_indices, intermediate_state_indices
    li = backend.req_to_token_pool.mamba_map[layer.layer_id]
    stash = getattr(backend, "_a1_fwd", None)
    if layer.layer_id == backend._a1_first_layer or stash is None or stash[0] is not forward_batch:
        n = cache_indices.shape[0]
        h0, rows = verify_indices(st, cache_indices, forward_batch.req_pool_indices[:n])
        stash = (forward_batch, h0, rows)
        backend._a1_fwd = stash
    return st.big[li], stash[1], stash[2]


def materialize_for_forward(backend, layer, forward_batch, cache_indices):
    """Non-verify GDN forward (extend / decode): bring this batch's slots home, reset its track destinations."""
    st = _backend_state(backend)
    if st is None or layer.layer_id != backend._a1_first_layer:
        return
    materialize(st, cache_indices.to(torch.int64))
    trk = getattr(getattr(backend, "forward_metadata", None), "mamba_track_indices", None)
    if trk is None:
        trk = getattr(forward_batch, "mamba_track_indices", None)
    if trk is not None:
        reset_ptr(st, trk.reshape(-1).to(torch.int64))


def commit(linear_backend, mamba_caches, state_indices, last_steps, track_indices, track_steps, req_pool_indices):
    """hybrid_linear_attn_backend.update_mamba_state_after_mtp_verify -> True if A1 handled the commit."""
    st = _backend_state(linear_backend)
    if st is None:
        return False
    if req_pool_indices is None:
        raise RuntimeError("A1 patch active: the verify commit caller must pass req_pool_indices (DFLASH does)")
    commit_raw(st, list(mamba_caches.conv), list(mamba_caches.intermediate_conv_window), state_indices,
               last_steps, req_pool_indices, track_indices, track_steps)
    return True


def disable_all_layer_track(backend):
    """The fused all-layers decode track kernel indexes [L, S] pools; A1's temporal view has a larger layer stride,
    so fall back to the per-layer track path (per-layer views are contiguous)."""
    return _backend_state(backend) is not None
