"""kopt_casc - shared-prefix ("cascade") attention for the FA3 TARGET_VERIFY forward of the DFlash coder.

Installed into SGLang @1cf2b8c54 by apply_casc.py as srt/layers/attention/kopt_casc.py plus a 4-line hook at the end
of flashattention_backend.py. Runtime switch SGL_CASC=1; unset/0 = install() is never called, the backend class is
untouched (byte-identical behaviour). Why: the verify requests of a batch often share a long prompt prefix (radix
cache); attending to it once per group instead of once per request cuts the KV reads.

What it does (only for the TARGET model's verify graphs, page_size 1, bf16 KV, topk<=1, full-attention layers):
  per step, inside the CUDA graph (init_forward_metadata_in_graph, no host sync):
    group the requests that share KV slots (radix cache) by probing req_to_token at fixed depths, take the longest
    common prefix (LCP) with the group leader -> per-request shared length P (0 = no sharing), segment = (group, P);
    build: perm (requests sorted by segment), prefix-pass varlen metadata (one q-segment per segment, KV = the
    segment's first member's page-table row, length P), suffix-pass metadata (row offset by P, length seq_len+N-P).
  per layer (forward_extend):
    q -> sorted order; FA3 non-causal on prefixes; FA3 causal on suffixes; merge_state_v2; scatter back.
Environment knobs (read once at import): SGL_CASC_PROBES (default 1024,2048,3072,4096,6144,8192), SGL_CASC_MAXP
(LCP window, default 12288), SGL_CASC_MIN_P (default 1024), SGL_CASC_MIN_BS (graphs with bs below this stay stock,
default 16).
compute_meta() is pure torch, device-agnostic and sync-free, so test_casc_cpu.py checks it on CPU.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)


def _ints(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]


PROBES = sorted(set(_ints("SGL_CASC_PROBES", "1024,2048,3072,4096,6144,8192")))
MAXP = int(os.environ.get("SGL_CASC_MAXP", "12288"))
MIN_P = int(os.environ.get("SGL_CASC_MIN_P", "1024"))
MIN_BS = int(os.environ.get("SGL_CASC_MIN_BS", "16"))
PRE_SPLITS = int(os.environ.get("SGL_CASC_PRE_SPLITS", "1"))  # prefix pass is compute-bound: auto split (0) = 338 vs 1 = 187 us/layer (casc_parts 21/10)
STATS = int(os.environ.get("SGL_CASC_STATS", "0"))   # debug only: every STATS steps read the last replay's grouping (host sync)


# ----------------------------------------------------------------------------------------------- metadata (pure torch)
def compute_meta(r2t, rpi, sl, n_tok, ar, pr, pidx, cols, min_p):
    """Grouping + cascade metadata. Static shapes, no host sync, no scatter (deterministic under
    torch.use_deterministic_algorithms), so it can be recorded in a CUDA graph.
    r2t [rows, ctx] int32 req_to_token; rpi [bs] req pool idx; sl [bs] committed seq lens (verify KV = sl + n_tok);
    ar = arange(bs) int64; pr [nP] probe depths (int64, ascending, <= ctx); pidx = arange(nP); cols = arange(W) int32.
    Returns dict of [bs] tensors: P (shared prefix len per request), leader, rep (segment representative = lowest batch
    index of the segment), cnt (members per segment, indexed by rep; 0 elsewhere), csl_pre (prefix len per rep slot),
    perm (sorted position -> batch index), csl_suf / vrow (suffix length / req_to_token flat row start, sorted order).
    """
    bs = rpi.shape[0]
    rpi = rpi.long()
    sl = sl.long()
    nP = pr.shape[0]
    W = cols.shape[0]
    # 1. group key: the slot at the deepest probe depth where some other request has the same slot. Radix-shared slots
    #    are identical along the whole path, so equal slot at depth d <=> common prefix >= d (live slots are unique).
    K = r2t[rpi[:, None], (pr - 1)[None, :]].long()                       # [bs, nP]
    valid = sl[:, None] >= pr[None, :]
    K = torch.where(valid, K, -1 - ar[:, None] * nP - pidx[None, :])      # unique negative where invalid
    partner = valid & ((K[:, None, :] == K[None, :, :]).sum(1) > 1)       # [bs, nP]
    depth = (partner.long() * (pidx + 1)[None, :]).amax(1)                # 0 = no partner at any probe
    kd = K.gather(1, (depth - 1).clamp(min=0)[:, None]).squeeze(1)
    gkey = torch.where(depth > 0, kd, -1 - ar)                            # ungrouped -> unique negative
    eqg = gkey[:, None] == gkey[None, :]
    leader = torch.where(eqg, ar[None, :], bs).amin(1)
    is_lead = leader == ar
    grouped = depth > 0
    # 2. LCP with the leader inside the window [0, W)
    A = r2t[:, :W].index_select(0, rpi)                                   # [bs, W]
    Lr = A.index_select(0, leader)
    cap = torch.minimum(sl, sl.index_select(0, leader)).clamp(max=W)
    bad = (A != Lr) | (cols[None, :] >= cap[:, None])
    lcp = torch.where(bad, cols[None, :], W).amin(1).long()
    lcp = torch.where(grouped & ~is_lead, lcp, 0)
    # leader's P = max LCP of its members (every member LCP <= leader's seq len, so the leader may read all of it)
    plead = torch.where(leader[:, None] == ar[None, :], lcp[:, None], 0).amax(0)
    P = torch.where(is_lead, plead, lcp)
    P = torch.where(P >= min_p, P, 0)
    # 3. segments = (group, P); P == 0 -> own segment with an empty prefix
    same = eqg & (P[:, None] == P[None, :]) & (P > 0)[:, None]
    rep = torch.minimum(torch.where(same, ar[None, :], bs).amin(1), ar)
    onehot = rep[:, None] == ar[None, :]                                  # [i, k]: request i in segment k
    cnt = onehot.long().sum(0)
    csl_pre = torch.where(onehot, P[:, None], 0).amax(0)
    # 4. sorted order: by rep, then batch index (segments contiguous, in rep order = prefix-pass q order)
    pos = (rep[None, :] < rep[:, None]).long().sum(1) + (
        (rep[None, :] == rep[:, None]) & (ar[None, :] < ar[:, None])).long().sum(1)
    perm = torch.where(pos[:, None] == ar[None, :], ar[:, None], 0).amax(0)
    csl_suf = (sl + n_tok - P).index_select(0, perm)
    vrow = (rpi * r2t.stride(0) + P).index_select(0, perm)
    return dict(P=P, leader=leader, rep=rep, cnt=cnt, csl_pre=csl_pre, perm=perm, csl_suf=csl_suf, vrow=vrow)


def fill_suffix_page_table(r2t, vrow, csl_suf, pt):
    """pt[s, j] = r2t.flat[vrow[s] + j] for j < csl_suf[s] (page_size 1). CUDA: the stock sync-free page-table kernel
    with req_to_token_stride=1 and 'virtual rows' vrow = rpi*ctx + P; CPU: reference loop (tests)."""
    if pt.is_cuda:
        from sglang.kernels.ops.kvcache.trtllm_mha_page_table import (
            create_trtllm_mha_kv_indices_triton, get_num_mha_kv_index_blocks)
        bs = pt.shape[0]
        create_trtllm_mha_kv_indices_triton[(bs, get_num_mha_kv_index_blocks(pt.shape[1], 1))](
            r2t, vrow, csl_suf, None, pt, None, 0, 1, pt.stride(0), PAGE_SIZE=1, HAS_SWA=False)
        return
    flat = r2t.reshape(-1)
    for s in range(pt.shape[0]):
        n = int(csl_suf[s])
        pt[s, :n] = flat[int(vrow[s]):int(vrow[s]) + n]


# --------------------------------------------------------------------------------------------------- backend glue
class CascState:
    """Static buffers (allocated in init_cuda_graph_state, outside any graph)."""

    def __init__(self, be, max_bs):
        dev, ctx = be.device, be.req_to_token.shape[1]
        self.n = int(be.speculative_num_draft_tokens)
        self.max_bs = max_bs
        W = min(MAXP, ctx)
        probes = [p for p in PROBES if 0 < p <= ctx]
        self.ar = torch.arange(max_bs, device=dev, dtype=torch.int64)
        self.cols = torch.arange(W, device=dev, dtype=torch.int32)
        self.pr = torch.tensor(probes, device=dev, dtype=torch.int64)
        self.pidx = torch.arange(len(probes), device=dev, dtype=torch.int64)
        self.cu_q_pre = torch.zeros(max_bs + 1, device=dev, dtype=torch.int32)
        self.csl_pre = torch.zeros(max_bs, device=dev, dtype=torch.int32)
        self.csl_suf = torch.zeros(max_bs, device=dev, dtype=torch.int32)
        self.perm = torch.zeros(max_bs, device=dev, dtype=torch.int64)
        self.vrow = torch.zeros(max_bs, device=dev, dtype=torch.int64)
        self.pt_suf = torch.zeros(max_bs, be.max_num_pages, device=dev, dtype=torch.int32)
        self.built_for = None
        logger.info("[kopt-casc] ON: max_bs=%d n_tok=%d probes=%s W=%d min_p=%d min_bs=%d pt_suf=%.1f MB",
                    max_bs, self.n, probes, W, MIN_P, MIN_BS, self.pt_suf.numel() * 4 / 2**20)

    def build(self, be, fb):
        self.built_for = None
        bs = fb.batch_size
        md = be.forward_metadata
        if (not fb.forward_mode.is_target_verify() or bs < MIN_BS or bs > self.max_bs
                or getattr(fb.spec_info, "ragged_verify_layout", None) is not None
                or md is None or md is not be.target_verify_metadata.get(bs)):
            return
        n = self.n
        m = compute_meta(be.req_to_token, fb.req_pool_indices[:bs], fb.seq_lens[:bs], n, self.ar[:bs], self.pr,
                         self.pidx, self.cols, MIN_P)
        self.cu_q_pre[1:bs + 1].copy_(torch.cumsum(m["cnt"] * n, 0))
        self.csl_pre[:bs].copy_(m["csl_pre"])
        self.csl_suf[:bs].copy_(m["csl_suf"])
        self.perm[:bs].copy_(m["perm"])
        self.vrow[:bs].copy_(m["vrow"])
        fill_suffix_page_table(be.req_to_token, self.vrow[:bs], self.csl_suf[:bs], self.pt_suf[:bs])
        self.built_for = md


def _eligible(be):
    return (not be.is_draft_runner and not be.skip_prefill and (be.topk or 0) <= 1 and be.page_size == 1
            and not be.use_mla and not be.use_sliding_window_kv_pool and be.fa_impl_ver == 3
            and be.kv_cache_dtype_str == "auto" and not be.is_encoder_decoder and not be.has_local_attention
            and be.attn_cp_size == 1 and not be.fa_skip_kv_cache and not be._unified_dense
            and not be.is_prefill_aware_swa and be.speculative_num_draft_tokens is not None)


def _layer_ok(be, st, q, layer, fb):
    if st.built_for is None or st.built_for is not be.forward_metadata or not fb.forward_mode.is_target_verify():
        return False
    from sglang.srt.layers.radix_attention import AttentionType
    swa = layer.sliding_window_size is not None and layer.sliding_window_size > -1
    if swa or layer.is_cross_attention or layer.attn_type != AttentionType.DECODER:
        return False
    bs = fb.batch_size
    out = getattr(fb, "_attn_output", None)
    return q.shape[0] == bs * st.n and (out is None or out.shape[0] == bs * st.n)


def _casc_forward(be, st, q, k, v, layer, fb, save_kv_cache):
    from sglang.srt.layers.attention import flashattention_backend as FB
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    md = be.forward_metadata
    if k is not None:
        assert v is not None
        if save_kv_cache:   # == stock bf16 non-CP non-MLA branch (flashattention_backend.py:1194-1254)
            be.token_to_kv_pool.set_kv_buffer(layer, KVWriteLoc(fb.out_cache_loc, md.swa_out_cache_loc), k, v,
                                              layer.k_scale, layer.v_scale)
    bs, n = fb.batch_size, st.n
    H, D, Dv, Hk = layer.tp_q_head_num, layer.head_dim, layer.v_head_dim, layer.tp_k_head_num
    perm = st.perm[:bs]
    qs = q.contiguous().view(bs, -1).index_select(0, perm).view(bs * n, H, D)
    kc, vc = be.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    kc = kc.view(-1, 1, Hk, D)
    vc = vc.view(-1, 1, layer.tp_v_head_num, Dv)
    common = dict(k_cache=kc, v_cache=vc, softmax_scale=layer.scaling, window_size=(-1, -1),
                  softcap=layer.logit_cap, return_softmax_lse=True, num_splits=be.num_splits, ver=be.fa_impl_ver)
    o1, l1, *_ = FB.flash_attn_with_kvcache(q=qs, page_table=md.page_table, cache_seqlens=st.csl_pre[:bs],
                                            cu_seqlens_q=st.cu_q_pre[:bs + 1], max_seqlen_q=bs * n, causal=False,
                                            **dict(common, num_splits=PRE_SPLITS))
    o2, l2, *_ = FB.flash_attn_with_kvcache(q=qs, page_table=st.pt_suf[:bs], cache_seqlens=st.csl_suf[:bs],
                                            cu_seqlens_q=md.cu_seqlens_q, max_seqlen_q=n, causal=True, **common)
    om, _ = FB.merge_state_v2_wrapper(o1, l1.T.contiguous(), o2, l2.T.contiguous())
    out = getattr(fb, "_attn_output", None)
    out = (out.view(bs * n, H, Dv) if out is not None else torch.empty_like(om))
    out.view(bs, -1).index_copy_(0, perm, om.view(bs, -1))
    return out.view(-1, H * Dv)


_SA = {"calls": 0, "steps": 0, "rows": 0, "shared": 0, "psum": 0, "segs": 0}


def _stats(st):
    """Debug: grouping of the last replayed verify step (built_for is set by the capture-time build; the buffers hold
    the values the graph wrote in its last replay). Logs cumulative sharing every 20 samples."""
    _SA["calls"] += 1
    bs = getattr(st, "last_bs", 0)
    if bs < MIN_BS or _SA["calls"] % STATS:
        return
    n = st.n
    cnt = (st.cu_q_pre[1:bs + 1] - st.cu_q_pre[:bs]).div(n, rounding_mode="floor").cpu()   # members per rep slot
    pre = st.csl_pre[:bs].cpu()
    sh = (pre > 0) & (cnt > 0)
    _SA["steps"] += 1
    _SA["rows"] += bs
    _SA["shared"] += int(cnt[sh].sum())
    _SA["psum"] += int((cnt * pre).sum())
    _SA["segs"] += int(sh.sum())
    if _SA["steps"] % 20 == 0:
        r = max(1, _SA["rows"])
        logger.info("[kopt-casc] stats: %d sampled steps, mean bs %.1f, rows sharing a prefix %.1f%%, mean P/row %.0f, "
                    "shared segments/step %.1f", _SA["steps"], r / _SA["steps"], 100 * _SA["shared"] / r,
                    _SA["psum"] / r, _SA["segs"] / _SA["steps"])


def install(cls):
    if getattr(cls, "_kopt_casc_installed", False):
        return
    orig_init, orig_ig, orig_fe = cls.init_cuda_graph_state, cls.init_forward_metadata_in_graph, cls.forward_extend

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        orig_init(self, max_bs, max_num_tokens)
        self._kopt_casc = CascState(self, max_bs) if _eligible(self) else None

    def init_forward_metadata_in_graph(self, forward_batch):
        orig_ig(self, forward_batch)
        st = getattr(self, "_kopt_casc", None)
        if st is not None:
            st.build(self, forward_batch)

    orig_og = getattr(cls, "init_forward_metadata_out_graph", None)

    def init_forward_metadata_out_graph(self, forward_batch, *a, **kw):
        # SGL_CASC_STATS=N (debug): eager, once per graph step; every N calls read the PREVIOUS replay's grouping
        st = getattr(self, "_kopt_casc", None)
        if st is not None and STATS > 0:
            _stats(st)
            fb = forward_batch
            st.last_bs = fb.batch_size if fb.forward_mode.is_target_verify() else 0
        return orig_og(self, forward_batch, *a, **kw)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, *args, **kwargs):
        st = getattr(self, "_kopt_casc", None)
        if (st is None or args or any(x is not None for x in kwargs.values())
                or not _layer_ok(self, st, q, layer, forward_batch)):
            return orig_fe(self, q, k, v, layer, forward_batch, save_kv_cache, *args, **kwargs)
        return _casc_forward(self, st, q, k, v, layer, forward_batch, save_kv_cache)

    for f in (init_cuda_graph_state, init_forward_metadata_in_graph, forward_extend, init_forward_metadata_out_graph):
        f.__module__ = __name__
    if STATS > 0:
        if orig_og is None:
            logger.warning("[kopt-casc] SGL_CASC_STATS: %s has no init_forward_metadata_out_graph, no stats", cls.__name__)
        else:
            cls.init_forward_metadata_out_graph = init_forward_metadata_out_graph
    cls.init_cuda_graph_state = init_cuda_graph_state
    cls.init_forward_metadata_in_graph = init_forward_metadata_in_graph
    cls.forward_extend = forward_extend
    cls._kopt_casc_installed = True
    logger.info("[kopt-casc] installed on %s", cls.__name__)
