"""fusedq.py - fused SiLU-and-mul + per-token FP8 quant for the coder's down_proj input (kernel-optim 21/10, advisor
K-R4 #4). Class A target: bytes AND scales identical to SGLang's silu_and_mul -> scaled_fp8_quant(per token).
  python fusedq.py gemma    step 2: fused Gemma add+RMSNorm + quant (gemma_quant.cuh) vs sgl_kernel.gemma_fused_add_rmsnorm
                            -> scaled_fp8_quant; bitwise on q, scales AND the in-place input/residual; graph timing
  python fusedq.py test     bitwise test vs the production pair (M 1..2111, real-ish + zero + outlier rows) + timing
                            of 64 down-proj sites in a CUDA graph (unfused pair vs fused) at M 384/512
Kernel: silu_quant.cuh, built through SGLang's own JIT loader with --use_fast_math on sm90 (same as the originals).
Valid only below the original quant's warp-kernel switch (M < 2 * SMs * 8); callers use the unfused path above it."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_MOD = {}


def module(dtype=torch.bfloat16):
    if dtype not in _MOD:
        from sglang.kernels.jit.utils import get_jit_cuda_arch, load_jit, make_cpp_args
        arch = get_jit_cuda_arch()
        fast = ["--use_fast_math"] if (arch.major, arch.minor) == (9, 0) else []
        args = make_cpp_args(dtype)
        _MOD[dtype] = load_jit("kopt_silu_mul_quant_fp8", "fast_math" if fast else "precise_math", *args,
                               cuda_files=[os.path.join(HERE, "silu_quant.cuh")],
                               cuda_wrappers=[("silu_mul_quant_fp8", f"silu_mul_quant_fp8<{args}>")],
                               extra_cuda_cflags=fast)
    return _MOD[dtype]


_GMOD = {}


def gmodule(dtype=torch.bfloat16):
    if dtype not in _GMOD:
        from sglang.kernels.jit.utils import get_jit_cuda_arch, load_jit, make_cpp_args
        arch = get_jit_cuda_arch()
        fast = ["--use_fast_math"] if (arch.major, arch.minor) == (9, 0) else []
        args = make_cpp_args(dtype)
        _GMOD[dtype] = load_jit("kopt_gemma_add_rmsnorm_quant_fp8", "fast_math" if fast else "precise_math", *args,
                                cuda_files=[os.path.join(HERE, "gemma_quant.cuh")],
                                cuda_wrappers=[("gemma_add_rmsnorm_quant_fp8", f"gemma_add_rmsnorm_quant_fp8<{args}>")],
                                extra_cuda_cflags=fast)
    return _GMOD[dtype]


def gemma_norm_quant(x, residual, weight, eps):
    """in place like gemma_fused_add_rmsnorm (x <- normed, residual <- x + residual) + returns (q, s) of the normed x."""
    M, D = x.shape
    q = torch.empty(M, D, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    gmodule(x.dtype).gemma_add_rmsnorm_quant_fp8(x, residual, weight, q, s, eps)
    return q, s


def gemma_reference(x, residual, weight, eps):
    from sgl_kernel import gemma_fused_add_rmsnorm
    from sglang.srt.layers.quantization import fp8_utils
    gemma_fused_add_rmsnorm(x, residual, weight, eps)
    return fp8_utils.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)


def bits(t):
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def test_gemma():
    D, eps = 5120, 1e-6
    lim = max_fused_tokens()
    ok = True
    g = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(D, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    for M in (1, 3, 8, 64, 255, 384, 512, 1024, lim - 1):
        for seed in (0, 1):
            x = inputs(M, D // 2, seed)
            r = (torch.randn(M, D, device="cuda", generator=g) * 4).to(torch.bfloat16)
            if M > 2:
                x[1] = 0
                r[1] = 0                              # all-zero row
            xa, ra, xb, rb = x.clone(), r.clone(), x.clone(), r.clone()
            qa = gemma_reference(xa, ra, w, eps)
            qb = gemma_norm_quant(xb, rb, w, eps)
            eq = same(qa, qb) and torch.equal(bits(xa), bits(xb)) and torch.equal(bits(ra), bits(rb))
            ok &= eq
            if not eq:
                print(f"  M={M} seed={seed}: MISMATCH x {(bits(xa) != bits(xb)).sum().item()} res "
                      f"{(bits(ra) != bits(rb)).sum().item()} q {(qa[0].view(torch.uint8) != qb[0].view(torch.uint8)).sum().item()}"
                      f" s {(qa[1].view(torch.int32) != qb[1].view(torch.int32)).sum().item()}")
    print(f"bitwise vs production gemma_fused_add_rmsnorm + per-token quant: {'PASS' if ok else 'FAIL'}")
    for M in (384, 512):
        xs = [inputs(M, D // 2, 20 + i) for i in range(4)]
        rs = [(torch.randn(M, D, device="cuda") * 4).to(torch.bfloat16) for _ in range(4)]

        def run(fn):
            def body():
                for i in range(176):
                    fn(xs[i % 4], rs[i % 4], w, eps)
            body()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                body()
            ts = []
            for _ in range(5):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(20):
                    gr.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) / 20)
            return sorted(ts)[2]
        tu, tf = run(gemma_reference), run(gemma_norm_quant)
        print(f"M={M}: 176 norm sites unfused {tu * 1000:.0f} us, fused {tf * 1000:.0f} us -> saves {(tu - tf) * 1000:.0f} us/step")
    return ok


def max_fused_tokens():
    return 2 * torch.cuda.get_device_properties(0).multi_processor_count * 8


def silu_mul_quant(x):
    """x [M, 2H] bf16 -> (q [M, H] fp8 e4m3, s [M, 1] fp32)."""
    M, H2 = x.shape
    q = torch.empty(M, H2 // 2, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    module(x.dtype).silu_mul_quant_fp8(x, q, s)
    return q, s


def reference(x):
    from sglang.srt.layers.activation import silu_and_mul
    from sglang.srt.layers.quantization import fp8_utils
    out = torch.empty(x.shape[0], x.shape[1] // 2, device=x.device, dtype=x.dtype)
    silu_and_mul(x, out)
    return fp8_utils.scaled_fp8_quant(out, None, use_per_token_if_dynamic=True)


def same(a, b):
    (qa, sa), (qb, sb) = a, b
    return torch.equal(qa.view(torch.uint8), qb.view(torch.uint8)) and torch.equal(sa.view(torch.int32), sb.view(torch.int32))


def inputs(M, H, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, 2 * H, device="cuda", generator=g) * torch.rand(M, 1, device="cuda", generator=g) * 6
    if M > 2:
        x[1] = 0                                   # all-zero row (scale 0 -> NaN path in the CTA quant)
        x[2, 7] = 3e3                              # outlier
        x[2, H + 7] = 50
    return x.to(torch.bfloat16)


def test():
    H = 17408
    lim = max_fused_tokens()
    ok = True
    for M in (1, 2, 3, 8, 64, 255, 384, 512, 520, 1024, lim - 1):
        for seed in (0, 1):
            x = inputs(M, H, seed)
            r, f = reference(x), silu_mul_quant(x)
            eq = same(r, f)
            ok &= eq
            if not eq:
                dq = (r[0].view(torch.uint8) != f[0].view(torch.uint8)).sum().item()
                ds = (r[1].view(torch.int32) != f[1].view(torch.int32)).sum().item()
                print(f"  M={M} seed={seed}: MISMATCH q bytes {dq}, scales {ds}")
    # also a non-17408 width (5120 rows exist elsewhere; the kernel is generic)
    for M in (64, 512):
        x = inputs(M, 5120, 3)
        ok &= same(reference(x), silu_mul_quant(x))
    print(f"bitwise vs production silu_and_mul + per-token quant: {'PASS' if ok else 'FAIL'} (M up to {lim - 1})")
    for M in (384, 512):
        xs = [inputs(M, H, 10 + i) for i in range(4)]   # rotate 4 inputs (L2-cold-ish)

        def run(fn):
            def body():
                for i in range(64):
                    fn(xs[i % 4])
            for _ in range(2):
                body()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                body()
            ts = []
            for _ in range(5):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                for _ in range(20):
                    gr.replay()
                e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1) / 20)
            return sorted(ts)[2]
        tu, tf = run(reference), run(silu_mul_quant)
        print(f"M={M}: 64 sites unfused {tu * 1000:.0f} us, fused {tf * 1000:.0f} us -> saves {(tu - tf) * 1000:.0f} us/step")
    return ok


# ============================================================================== runtime patch (SGL_FUSEDQ=1)
# The fused kernels hand their (q, s) to the NEXT per-token quant of the same tensor through a tiny cache, so the
# Linear/GEMM path stays exactly the production one (apply_fp8_linear -> fp8_scaled_mm with the same q and s bytes).
#  - GemmaRMSNorm with residual (decoder input/post-attention norms): fused add+norm+quant; x/residual written as before.
#  - Qwen2MoeMLP: gate_up -> fused silu+quant -> down_proj gets a placeholder bf16 tensor whose quant is the cached one
#    (a miss would quantize garbage -> it raises instead). The first eager call of every MLP module probes: stock
#    activation + quant vs fused bytes and a cache hit in down_proj; any miss keeps that module on the stock path.
#  - The norm fusion is OFF unless SGL_FUSEDQ_NORM=1 (not bitwise vs the production flashinfer JIT norm, K5 21/10).
# Only below the original quant's CTA/warp switch (M < 2 * SMs * 8); otherwise the stock path runs.
_ST = {"cache": [], "norm": 0, "silu": 0, "hits": 0, "logged": False, "lim": None}


def _on():
    return os.environ.get("SGL_FUSEDQ", "0") == "1"


def _key(t):
    return (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _put(t, qs):
    _ST["cache"].append((_key(t), t._version, t, qs))   # strong ref: the storage cannot be reused while cached
    del _ST["cache"][:-6]


def _get(t):
    k = _key(t)
    for key, ver, ref, qs in reversed(_ST["cache"]):
        if key == k and ver == t._version:
            _ST["hits"] += 1
            return qs
    return None


def _plain():
    """only where the stock kernels are the ones the bitwise test covers: sm90, no dynamo trace, no inference tensors."""
    if "sm90" not in _ST:
        from sglang.kernels.jit.utils import get_jit_cuda_arch
        a = get_jit_cuda_arch()
        _ST["sm90"] = (a.major, a.minor) == (9, 0)
    return (_ST["sm90"] and not torch.compiler.is_compiling()
            and not torch.is_inference_mode_enabled())


def _lim(dev):
    if _ST["lim"] is None:
        _ST["lim"] = 2 * torch.cuda.get_device_properties(dev).multi_processor_count * 8
    return _ST["lim"]


def _log():
    if not _ST["logged"] and _ST["norm"] + _ST["silu"] >= 200 and os.environ.get("SGL_FUSEDQ_LOG", "0") == "1":
        _ST["logged"] = True
        print(f"[fusedq] active: norm fused {_ST['norm']}, silu fused {_ST['silu']}, quant cache hits {_ST['hits']}",
              file=sys.stderr, flush=True)


def install():
    from sglang.srt.layers import layernorm as LN
    from sglang.srt.layers.quantization import fp8_utils as FU
    from sglang.srt.models import qwen2_moe as QM
    if getattr(FU, "_fusedq", False):
        return
    orig_q = FU.scaled_fp8_quant
    orig_norm = LN.GemmaRMSNorm._forward_impl
    orig_mlp = QM.Qwen2MoeMLP.forward

    def scaled_fp8_quant(input, scale=None, num_token_padding=None, use_per_token_if_dynamic=False):
        if scale is None and use_per_token_if_dynamic and not num_token_padding and _ST["cache"]:
            qs = _get(input)
            if qs is not None:
                return qs
        return orig_q(input, scale, num_token_padding=num_token_padding,
                      use_per_token_if_dynamic=use_per_token_if_dynamic)

    def norm_impl(self, x, residual=None, post_residual_addition=None):
        # OFF by default (K5 21/10: not bitwise vs production; flashinfer 0.6.17 JIT norm differs by 1 ulp in rare rows)
        if (_on() and os.environ.get("SGL_FUSEDQ_NORM", "0") == "1" and _plain() and residual is not None
                and post_residual_addition is None and x.dim() == 2
                and x.dtype == torch.bfloat16 and x.stride(-1) == 1 and residual.stride(-1) == 1
                and 0 < x.shape[0] < _lim(x.device) and x.shape[1] % 8 == 0):
            qs = gemma_norm_quant(x, residual, self.weight.data, self.variance_epsilon)
            _put(x, qs)
            _ST["norm"] += 1
            _log()
            return x, residual
        return orig_norm(self, x, residual, post_residual_addition)

    def mlp_forward(self, x):
        ok = getattr(self, "_fusedq_ok", None)        # None = not probed yet, False = stock path for good
        if not _on() or ok is False or self._enable_silu_fp4_quant_fusion or not _plain():
            return orig_mlp(self, x)
        gate_up, _ = self.gate_up_proj(x)
        if (isinstance(gate_up, tuple) or gate_up.dim() != 2 or gate_up.dtype != torch.bfloat16
                or not (0 < gate_up.shape[0] < _lim(gate_up.device)) or not gate_up.is_contiguous()
                or gate_up.shape[1] % 16 != 0):
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
            return x
        if ok is None:
            # first eager call of this module: run the stock activation, give its quant the fused (q, s); keep the
            # fusion only if the fused bytes equal the stock quant AND down_proj consumed the cached pair
            if torch.cuda.is_current_stream_capturing():
                x = self.act_fn(gate_up)
                x, _ = self.down_proj(x)
                return x
            act = self.act_fn(gate_up)
            q0, s0 = orig_q(act, None, use_per_token_if_dynamic=True)
            qs = silu_mul_quant(gate_up)
            same = torch.equal(q0.view(torch.uint8), qs[0].view(torch.uint8)) and torch.equal(s0, qs[1])
            _put(act, qs)
            h0 = _ST["hits"]
            out, _ = self.down_proj(act)
            self._fusedq_ok = bool(same and _ST["hits"] > h0)
            _ST["cache"].clear()
            if not self._fusedq_ok:
                print(f"[fusedq] {type(self).__name__}: stock path (bitwise {same}, cache hit {_ST['hits'] > h0})",
                      file=sys.stderr, flush=True)
            return out
        qs = silu_mul_quant(gate_up)
        ph = gate_up.new_empty(gate_up.shape[0], gate_up.shape[1] // 2)   # placeholder: never read
        _put(ph, qs)
        h0 = _ST["hits"]
        out, _ = self.down_proj(ph)
        if _ST["hits"] == h0:
            raise RuntimeError("fusedq: down_proj did not consume the fused quant (placeholder would be read)")
        _ST["silu"] += 1
        _log()
        return out

    FU.scaled_fp8_quant = scaled_fp8_quant
    LN.GemmaRMSNorm._forward_impl = norm_impl
    QM.Qwen2MoeMLP.forward = mlp_forward
    FU._fusedq = True
    if os.environ.get("RANK", "0") in ("0", ""):
        print(f"[fusedq] installed (active: {_on()})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(0 if ((a == ["test"] and test()) or (a == ["gemma"] and test_gemma())) else 1)
