"""topk20.py - bitwise-exact, CUDA-graph-capturable replacement for torch.topk(x, k, dim=-1) (largest, sorted) on
CUDA fp32 rows. Target call: SGLang DFlash verify sampling, dflash_utils.py build_dflash_verify_target_probs
`topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)`, scaled_logits [8*bs, 248320] fp32, k=20.

WHAT TORCH DOES (aten/src/ATen/native/cuda/TensorTopK.{cpp,cu}, rows > 10 or V < 100000):
  1. radix select of the k-th value on the uint32 "radix key" of each float:
       key = bits ^ (sign ? 0xffffffff : 0x80000000), and every NaN -> 0xffffffff (NaN = largest; -0.0 < +0.0).
  2. gather (single- and multi-block versions alike): output slots 0.. = every element with key > kth, in INDEX
     order, then the elements with key == kth, in INDEX order, until k slots are full.
  3. sorted=True: sortKeyValueInplace(values, indices, dim, largest) with stable=false; for k <= 32 that is the
     SmallBitonicSort network (not stable: the tie order it produces depends on the slot order from step 2).
  (rows <= 10 with V >= 100000 use a full stable sort instead -> we just call torch.topk there.)

WHAT WE DO (4 launches, no host sync, static shapes):
  stage 1 (Triton, grid M x ceil(V/CH)): each program loads one chunk of CH=8192 floats ONCE as [G=128, GS=64],
    converts to an order-preserving int32 key s (same order as torch's radix key, NaN -> max), takes the group
    maxima m[128], and t = the K-th largest of m (counted with multiplicity, via a 128x128 rank matrix). At least K
    chunk elements are >= t, so the chunk's top-K (by key desc, index asc) is inside {s > t} + the first {s == t} by
    index. Fast path: only the ~20 "hot" groups (m >= t) are re-read (L1/L2) and compacted; it writes
    {s > t} (all) + the first CAP-#{s>t} elements with s == t, in index order, to a CAP=32 slot list.
    Slow path (more than HG=32 hot groups or more than CAP elements > t, e.g. a sorted/adversarial or heavily tied
    chunk): exact chunk K-th key by a <=K-step loop, then the same write rule (now #{s > t} < K).
  stage 2 (Triton, grid M): reads the 32x32 candidates of the row, finds the exact row K-th key kth (<=K-step loop;
    the candidates contain the row's top-K, so their K-th key is the row's), and writes torch's PRE-SORT slot order:
    {s > kth} in index order, then the first K-#{s>kth} of {s == kth} in index order (candidate order is index
    order within each class, see the proof in the docstring of _stage2). Values are re-loaded from x (exact bits).
  stage 3: torch.topk(vals[M, K], K) - on our K slots its k-th value is the row kth, so its own gather reproduces
    our slot order unchanged ({> kth} then {== kth}, slot order), and it then runs exactly the sortKeyValueInplace
    call that the big torch.topk runs on the same keys in the same slots. That sorting network's swaps depend only
    on the keys, so the returned slot permutation applied to our indices gives torch's exact tie order (no
    re-implementation of torch's bitonic network). stage 4: torch.gather of the indices.
  => values, indices and order are bitwise equal to torch.topk whenever step 2 above is what torch does (the GPU
     bench checks it on heavy-tie data at every M; it is the documented behaviour of both gather kernels).

TIES: logits are bf16 matmul outputs (8-bit mantissa) divided by T in fp32, so equal logits stay equal: in the top-20
  of a real row ties are the norm (bench prints the measured rate: rows with a tie inside the top-20 and rows whose
  20th value equals the 21st). Downstream (softmax over the 20 values, flashinfer top-p renorm, scatter to the
  dense probs) only depends on the selected SET and the value sequence, except when a request's own top_k < 20
  masks by rank - which is why the order is reproduced exactly anyway.
NaN / -inf: -inf padding is an ordinary key (selected only when fewer than k finite values; ties by index like
  torch). NaN: same radix semantics as torch (all NaN equal, largest; values re-read so payload bits match).
  -0.0 vs +0.0: different keys (+0 > -0), as in torch's radix select.

Env: SGL_TOPK20_CFG="CH,GS,HG,CAP,W1,W2,HOT" overrides the default tuning (16384,64,32,32,16,8,1; M7-tested).
     SGL_TOPK20_DUMP=<dir> (debug): save the first rows of the first SGL_TOPK20_DUMP_N calls (outside graph capture)
     as <dir>/topk_in_<n>.pt for `bench_topk.py --dump`.
"""
import os

import torch
import triton
import triton.language as tl

KMIN = tl.constexpr(-2147483648)   # sentinel: below every real key (the smallest real key, -inf, is -2139095041)
KMAX = tl.constexpr(2147483647)    # NaN key (torch radix: every NaN -> 0xffffffff = largest)

DEFAULT_CFG = dict(CH=16384, GS=64, HG=32, CAP=32, W1=16, W2=8, HOT=1)  # M7-tested (21/10)
_FIELDS = ("CH", "GS", "HG", "CAP", "W1", "W2", "HOT")


def parse_cfg(s):
    cfg = dict(DEFAULT_CFG)
    if s:
        for name, v in zip(_FIELDS, s.split(",")):
            if v.strip():
                cfg[name] = int(v)
    return cfg


CFG = parse_cfg(os.environ.get("SGL_TOPK20_CFG", ""))


@triton.jit
def _keys(rowp, offs, V):
    """int32 key with the same order as torch's uint32 radix key (minus 2^31); out of range -> KMIN."""
    inr = offs < V
    x = tl.load(rowp + offs, mask=inr, other=0.0)
    b = x.to(tl.int32, bitcast=True)
    mag = b & 0x7FFFFFFF
    s = b ^ ((b >> 31) & 0x7FFFFFFF)
    s = tl.where(mag > 0x7F800000, KMAX, s)                 # NaN (any sign/payload), integer test: no fast-math
    return tl.where(inr, s, KMIN)


@triton.jit
def _excl2d(a):
    """exclusive prefix sum of int32 a[R, C] in row-major (= index) order."""
    inc = tl.cumsum(a, axis=1)
    tot = tl.sum(a, axis=1)
    off = tl.cumsum(tot, axis=0) - tot
    return inc - a + off[:, None]


@triton.jit
def _kth(s, K: tl.constexpr):
    """K-th largest key of s[R, C] counted with multiplicity: count(s >= v) >= K and count(s > v) < K."""
    cur = tl.max(tl.max(s, axis=1), axis=0)
    cnt = tl.sum(tl.sum((s >= cur).to(tl.int32), axis=1), axis=0)
    while cnt < K:
        cur = tl.max(tl.max(tl.where(s < cur, s, KMIN), axis=1), axis=0)
        cnt = tl.sum(tl.sum((s >= cur).to(tl.int32), axis=1), axis=0)
    return cur


@triton.jit
def _write(s, idx, t, ps, pi, CAP: tl.constexpr):
    """write {s > t} (all; caller guarantees <= CAP) then the first CAP-#{s>t} of {s == t}, index order; pad KMIN."""
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
def _stage1(x_ptr, stride_m, V, cs_ptr, ci_ptr, NCHP, K: tl.constexpr, CH: tl.constexpr, GS: tl.constexpr,
            HG: tl.constexpr, CAP: tl.constexpr, HOT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    G: tl.constexpr = CH // GS
    rowp = x_ptr + row * stride_m
    g = tl.arange(0, G)
    j = tl.arange(0, GS)
    offs = ch * CH + g[:, None] * GS + j[None, :]
    s = _keys(rowp, offs, V)
    m = tl.max(s, axis=1)                                                    # group maxima [G]
    cge = tl.sum((m[None, :] >= m[:, None]).to(tl.int32), axis=1)           # #{h: m_h >= m_g}
    t = tl.max(tl.where(cge >= K, m, KMIN), axis=0)                          # K-th largest group max
    slot = (row * NCHP + ch) * CAP
    ps = cs_ptr + slot
    pi = ci_ptr + slot
    if HOT:
        hot = m >= t                                                         # every element >= t is in a hot group
        hoti = hot.to(tl.int32)
        n_hot = tl.sum(hoti, axis=0)
        ph = tl.cumsum(hoti, axis=0) - 1
        h = tl.arange(0, HG)
        sel = hot[None, :] & (ph[None, :] == h[:, None])                     # [HG, G] one-hot compaction
        hg = tl.sum(tl.where(sel, g[None, :], 0), axis=1)                    # hot group ids, ascending
        hoffs = ch * CH + hg[:, None] * GS + j[None, :]
        hoffs = tl.where((h < n_hot)[:, None], hoffs, V)                     # unused rows: no memory traffic
        sh = _keys(rowp, hoffs, V)                                           # re-read ~20 groups (L1/L2 hit)
        sh = tl.where((h < n_hot)[:, None], sh, KMIN)
        c_gt = tl.sum(tl.sum((sh > t).to(tl.int32), axis=1), axis=0)
        if (n_hot <= HG) & (c_gt <= CAP):
            _write(sh, hoffs, t, ps, pi, CAP)
        else:
            s2 = _keys(rowp, offs, V)
            t2 = _kth(s2, K)
            _write(s2, offs, t2, ps, pi, CAP)
    else:
        c_gt = tl.sum(tl.sum((s > t).to(tl.int32), axis=1), axis=0)
        if c_gt > CAP:
            t = _kth(s, K)
        _write(s, offs, t, ps, pi, CAP)


@triton.jit
def _stage2(x_ptr, stride_m, cs_ptr, ci_ptr, val_ptr, idx_ptr, NCH, NCHP: tl.constexpr, CAP: tl.constexpr,
            K: tl.constexpr):
    """Row top-K in torch's pre-sort slot order.

    Candidates hold, per chunk c, {s > t_c} then {s == t_c} (index order, truncated), with t_c <= row kth.
    A row element with s > kth (>= t_c) sits in its chunk's {s > t_c} part; elements with s == kth sit all in
    the {s > t_c} part (kth > t_c) or all in the {s == t_c} part (kth == t_c). Chunks are in index order, so
    within each class (> kth, == kth) candidate order == index order, and the truncated tail of a chunk's
    {s == t_c} part can never be among the first K-#{s>kth} ties of the row (the chunk already kept >= that many
    earlier ones). So: exact kth over the candidates == row kth, and an exclusive scan gives torch's slots.
    """
    row = tl.program_id(0).to(tl.int64)
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
    v = tl.load(x_ptr + row * stride_m + ii, mask=w, other=0.0)
    tl.store(val_ptr + row * K + pos, v, mask=w)
    tl.store(idx_ptr + row * K + pos, ii.to(tl.int64), mask=w)


def eligible(x, k, cfg=None):
    """True when topk_exact runs the Triton path (else it returns torch.topk itself)."""
    cfg = cfg or CFG
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float32 and x.dim() == 2):
        return False
    M, V = x.shape
    G = cfg["CH"] // cfg["GS"]
    if not (1 <= k <= min(cfg["CAP"], cfg["HG"], G) and cfg["HG"] <= G):
        return False
    if not all(v > 0 and v & (v - 1) == 0 for v in (cfg["CH"], cfg["GS"], cfg["HG"], cfg["CAP"])):
        return False
    if x.stride(1) != 1 or V < cfg["CH"] or V >= 2 ** 31 - cfg["CH"]:
        return False
    if M <= 10 and V >= 100000:      # torch.topk uses a full stable sort there (different tie order)
        return False
    return M >= 1


_dump_n = [0]


def _maybe_dump(x):
    d = os.environ.get("SGL_TOPK20_DUMP")
    if not d or _dump_n[0] >= int(os.environ.get("SGL_TOPK20_DUMP_N", "8")):
        return
    if torch.cuda.is_current_stream_capturing():
        return
    os.makedirs(d, exist_ok=True)
    torch.save({"x": x[:16].detach().float().cpu()}, os.path.join(d, f"topk_in_{os.getpid()}_{_dump_n[0]}.pt"))
    _dump_n[0] += 1


def topk_exact(x, k, cfg=None):
    """== torch.topk(x, k, dim=-1) (largest=True, sorted=True) bit for bit; returns (values, indices)."""
    k = int(k)
    _maybe_dump(x)
    cfg = cfg or CFG
    if not eligible(x, k, cfg):
        return torch.topk(x, k, dim=-1)
    M, V = x.shape
    CH, CAP = cfg["CH"], cfg["CAP"]
    nch = triton.cdiv(V, CH)
    nchp = triton.next_power_of_2(nch)
    cs = torch.empty((M, nchp, CAP), dtype=torch.int32, device=x.device)
    ci = torch.empty((M, nchp, CAP), dtype=torch.int32, device=x.device)
    val = torch.empty((M, k), dtype=x.dtype, device=x.device)
    idx = torch.empty((M, k), dtype=torch.int64, device=x.device)
    with torch.cuda.device(x.device):
        _stage1[(M, nch)](x, x.stride(0), V, cs, ci, nchp, K=k, CH=CH, GS=cfg["GS"], HG=cfg["HG"], CAP=CAP,
                          HOT=bool(cfg["HOT"]), num_warps=cfg["W1"])
        _stage2[(M,)](x, x.stride(0), cs, ci, val, idx, nch, NCHP=nchp, CAP=CAP, K=k, num_warps=cfg["W2"])
    # torch.topk on the K pre-sort slots: its gather keeps our slot order, then the same in-place sort as the big
    # call (values bitwise; perm = slot permutation of that sort).
    sv, perm = torch.topk(val, k, dim=-1)
    return sv, torch.gather(idx, 1, perm)


def selftest(rows=64, V=248320, k=20, temp=0.7, seed=0):
    """Quick on-device check used by apply_topk.py --check (SGL_TOPK20=1): bf16-rounded logits / T and forced ties."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, V, device="cuda", generator=g) * 3).bfloat16().float() / temp
    v = torch.topk(x, k + 1, dim=-1).values
    x[:, 1000:1012] = v[:, k - 1:k]           # boundary ties
    x[:, 5000:5004] = v[:, :1]                # ties inside the top-k
    x[:, -3000:] = float("-inf")              # -inf tail
    rv, ri = torch.topk(x, k, dim=-1)
    ov, oi = topk_exact(x, k)
    return bool(torch.equal(rv.view(torch.int32), ov.view(torch.int32)) and torch.equal(ri, oi))
