"""A2 runtime hook (own code). apply_a2.py copies this file to sglang/kernels/ops/attention/fla/gdnk_a2_rt.py, kernels.py
to .../fla/gdnk_a2.py and the chosen configuration to .../fla/gdnk_a2_cfg.py, and appends a marked block to
sglang/srt/layers/attention/linear/gdn_backend.py that calls install(globals()) at the end of that module's import.

install() swaps in, only where a2_bench.py proved 100% bitwise identity and a speed-up:
  verify  gdn_triton.fused_sigmoid_gating_delta_rule_update -> wrapper: the DFLASH target-verify call shape (GDN, varlen,
          disable_state_update, intermediate buffer, no tree, no ReplaySSM ring) goes to gdnk_a2.verify with the chosen
          config; every other call (decode, tree, KDA, ring) goes to the original function unchanged.
  conv    gdn_backend.causal_conv1d_update -> wrapper: the verify call shape (3-D x, intermediate conv window, no tree, no
          num_accept_tokens / cache_seqlens / metadata) goes to gdnk_a2.conv_update; everything else to the original.
  fused   (opt-in: SGL_A2_FUSED=1 and the bench enabled it, and A1 not active = sglang.srt.mem_cache.gdnk_a1 not
          installed or SGLANG_A1_SSM_PTR=0) GDNAttnBackend.forward_extend target-verify
          branch -> one fused conv+verify kernel. Off by default.
Kill switch: SGL_A2=0 (nothing is swapped). Compatible with A1 (ON when installed unless SGLANG_A1_SSM_PTR=0): the verify wrapper accepts any
state-source tensor / slot stride / scratch rows, and loads h0 before any store of the same tile (A1's in-place rule).
"""
import inspect
import os
import sys

_STATE = {"installed": False, "active": []}


def _log(msg):
    if os.environ.get("RANK", "0") in ("0", ""):
        print(f"[gdnk-a2] {msg}", file=sys.stderr, flush=True)


def _cfg():
    from sglang.kernels.ops.attention.fla.gdnk_a2_cfg import CONFIG
    return CONFIG


def make_verify_wrapper(orig, cfg):
    from sglang.kernels.ops.attention.fla import gdnk_a2 as K2

    def fused_sigmoid_gating_delta_rule_update(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b,
                                               initial_state_source, initial_state_indices, scale=None,
                                               use_qk_l2norm_in_kernel=False, cu_seqlens=None, is_kda=False,
                                               lower_bound=None, disable_state_update=False,
                                               intermediate_states_buffer=None, intermediate_state_indices=None,
                                               cache_steps=None, retrieve_parent_token=None, cache_ring=False,
                                               **ring_kwargs):
        ok = (disable_state_update and intermediate_states_buffer is not None and intermediate_state_indices is not None
              and retrieve_parent_token is None and not cache_ring and not is_kda and lower_bound is None
              and cu_seqlens is not None and use_qk_l2norm_in_kernel and initial_state_source is not None
              and q.dim() == 4 and q.shape[0] == 1 and k.shape[-1] in (64, 128, 256)
              and v.shape[-1] % cfg["BV"] == 0 and intermediate_states_buffer.dim() == 5
              and intermediate_states_buffer.stride(-1) == 1)
        if not ok:
            return orig(A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=softplus_beta,
                        softplus_threshold=softplus_threshold, q=q, k=k, v=v, b=b,
                        initial_state_source=initial_state_source, initial_state_indices=initial_state_indices,
                        scale=scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens,
                        is_kda=is_kda, lower_bound=lower_bound, disable_state_update=disable_state_update,
                        intermediate_states_buffer=intermediate_states_buffer,
                        intermediate_state_indices=intermediate_state_indices, cache_steps=cache_steps,
                        retrieve_parent_token=retrieve_parent_token, cache_ring=cache_ring, **ring_kwargs)
        return K2.verify(A_log, a, dt_bias, softplus_beta, softplus_threshold, q, k, v, b, initial_state_source,
                         initial_state_indices, cu_seqlens, intermediate_states_buffer, intermediate_state_indices,
                         scale=scale, cfg=cfg, tokens_per_req=intermediate_states_buffer.shape[1])

    fused_sigmoid_gating_delta_rule_update._gdnk_a2 = True
    fused_sigmoid_gating_delta_rule_update._gdnk_a2_orig = orig
    return fused_sigmoid_gating_delta_rule_update


def make_conv_wrapper(orig, conv):
    from sglang.kernels.ops.attention.fla import gdnk_a2 as K2
    sig = inspect.signature(orig)
    cfg, arith = conv["best_cfg"], conv["arith"]
    write_state, inplace = bool(conv.get("write_state")), bool(conv.get("inplace"))

    def causal_conv1d_update(*args, **kwargs):
        try:
            ba = sig.bind(*args, **kwargs)
        except TypeError:
            return orig(*args, **kwargs)
        p = dict(ba.arguments)
        x, cs, w = p.get("x"), p.get("conv_state"), p.get("weight")
        win, widx, cidx = p.get("intermediate_conv_window"), p.get("intermediate_state_indices"), p.get("conv_state_indices")
        act = p.get("activation")
        ok = (x is not None and x.dim() == 3 and cs is not None and cs.dim() == 3 and w is not None and w.dim() == 2
              and w.shape[1] == cs.shape[2] + 1 and x.dtype == cs.dtype and win is not None and win.dim() == 4
              and widx is not None and cidx is not None and p.get("cache_seqlens") is None
              and p.get("num_accept_tokens") is None and p.get("retrieve_next_token") is None
              and p.get("retrieve_next_sibling") is None and p.get("retrieve_parent_token") is None
              and p.get("metadata") is None and p.get("pad_slot_id", -1) == -1
              and act in (None, True, False, "silu", "swish"))
        if not ok:
            return orig(*args, **kwargs)
        return K2.conv_update(x, cs, w, p.get("bias"), act, cidx, win, widx, cfg=cfg, arith=arith,
                              write_state=write_state, inplace=inplace)

    causal_conv1d_update._gdnk_a2 = True
    causal_conv1d_update._gdnk_a2_orig = orig
    return causal_conv1d_update


def make_fused_forward_extend(orig, fused, conv):
    """Target-verify branch of GDNAttnBackend.forward_extend as ONE kernel (conv1d + gating + l2norm + delta rule);
    falls back to the original method for anything that is not the plain DFLASH linear-chain verify."""
    import torch
    from sglang.kernels.ops.attention.fla import gdnk_a2 as K2
    cfg, arith = fused["best_cfg"], conv["arith"]

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        try:
            if not forward_batch.forward_mode.is_target_verify() or not isinstance(mixed_qkv, torch.Tensor):
                return orig(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
            md = self.forward_metadata
            if (md.retrieve_parent_token is not None or md.retrieve_next_token is not None
                    or md.retrieve_next_sibling is not None):
                return orig(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
            cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
            if (getattr(cache, "replayssm_rawv", None) is not None or getattr(cache, "replayssm_d", None) is not None
                    or getattr(cache, "intermediate_ssm", None) is None):
                return orig(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
            win = cache.intermediate_conv_window[0]
            w = layer.conv_weights
            if K2.window_mode(win) != 1 or w.shape[1] != 4 or mixed_qkv.dim() != 2:
                return orig(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
            T = forward_batch.spec_info.draft_token_num
            nb = mixed_qkv.shape[0] // T
            rows = self.verify_intermediate_state_indices[:nb]
            return K2.fused_conv_verify(
                layer.A_log, a, layer.dt_bias, 1.0, 20.0, mixed_qkv, cache.conv[0], w, layer.bias, layer.activation,
                b, cache.temporal, md.mamba_cache_indices, md.query_start_loc, cache.intermediate_ssm, rows, win, rows,
                layer.num_k_heads, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim, cfg=cfg, arith=arith,
                tokens_per_req=T)
        except AttributeError:
            return orig(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)

    forward_extend._gdnk_a2 = True
    return forward_extend


def _a1_active():
    """A1 (zero-copy commit) is ON whenever it is installed and SGLANG_A1_SSM_PTR is not "0" (unset = ON)."""
    if os.environ.get("SGLANG_A1_SSM_PTR", "") == "0":
        return False
    import importlib.util
    try:
        return importlib.util.find_spec("sglang.srt.mem_cache.gdnk_a1") is not None
    except (ImportError, ValueError):
        return False


def install(gb):
    """gb = globals() of sglang.srt.layers.attention.linear.gdn_backend (called at the end of its import)."""
    if _STATE["installed"]:
        return _STATE["active"]
    _STATE["installed"] = True
    if os.environ.get("SGL_A2", "1") == "0":
        _log("SGL_A2=0 -> nothing swapped")
        return []
    cfg = _cfg()
    if cfg.get("verify", {}).get("enabled"):
        gt = sys.modules.get("sglang.srt.layers.attention.linear.kernels.gdn_triton")
        if gt is None:
            import sglang.srt.layers.attention.linear.kernels.gdn_triton as gt
        fn = gt.fused_sigmoid_gating_delta_rule_update
        if not getattr(fn, "_gdnk_a2", False):
            gt.fused_sigmoid_gating_delta_rule_update = make_verify_wrapper(fn, cfg["verify"]["best_cfg"])
        _STATE["active"].append(f"verify:{cfg['verify']['best']}")
    if cfg.get("conv", {}).get("enabled"):
        fn = gb["causal_conv1d_update"]
        if not getattr(fn, "_gdnk_a2", False):
            gb["causal_conv1d_update"] = make_conv_wrapper(fn, cfg["conv"])
        _STATE["active"].append(f"conv:{cfg['conv']['best']}")
    if (cfg.get("fused", {}).get("enabled") and os.environ.get("SGL_A2_FUSED") == "1"
            and _a1_active()):
        _log("SGL_A2_FUSED=1 ignored: A1 (sglang/srt/mem_cache/gdnk_a1.py) is installed and SGLANG_A1_SSM_PTR != 0; "
             "the fused path would bypass A1's h0/scratch-row indices")
    elif (cfg.get("fused", {}).get("enabled") and os.environ.get("SGL_A2_FUSED") == "1"):
        cls = gb["GDNAttnBackend"]
        if not getattr(cls.forward_extend, "_gdnk_a2", False):
            cls.forward_extend = make_fused_forward_extend(cls.forward_extend, cfg["fused"], cfg["conv"])
        _STATE["active"].append(f"fused:{cfg['fused']['best']}")
    _log("active: " + (", ".join(_STATE["active"]) or "nothing (no bitwise + faster config in the bench result)")
         + f" | bench triton {cfg.get('triton')}")
    return _STATE["active"]
