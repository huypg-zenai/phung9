"""Apply the A2 kernels (output-identical GDN verify / conv1d update) to the sglang that `import sglang` resolves to.

  python apply_a2.py            install / refresh (idempotent), exit 0; exit 3 if the bench result enables nothing
  python apply_a2.py --check    exit 0 only if the patch is present, the copied modules import, gdn_backend.py compiles
                                (and imports, unless the env lacks server deps), and - when a GPU is visible - a small
                                bitwise self-test of the installed wrappers vs the original kernels passes
  python apply_a2.py --revert   remove the block and the copied files
  options: --config FILE (default: a2_bench_result.json next to this file, or $A2_CONFIG)

What it changes (all inside the sglang package dir, nothing outside, nothing in this directory):
  + sglang/kernels/ops/attention/fla/gdnk_a2.py      = kernels.py (own kernels)
  + sglang/kernels/ops/attention/fla/gdnk_a2_rt.py   = a2_runtime.py (wrappers + install)
  + sglang/kernels/ops/attention/fla/gdnk_a2_cfg.py  = the configs a2_bench.py chose (CONFIG dict)
  ~ sglang/srt/layers/attention/linear/gdn_backend.py: a block between '# >>> gdnk-a2 >>>' and '# <<< gdnk-a2 <<<'
    appended at the END of the file (calls gdnk_a2_rt.install(globals())). A1 also edits gdn_backend.py (anchored edits
    inside forward_extend); the a2 block is a pure append and both orders work.
Only configs that were 100% bitwise identical AND faster in the bench are enabled. The bench's Triton version must equal
the runtime one (bitwise identity depends on the compiler); override with A2_ALLOW_VERSION_MISMATCH=1.
"""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import pprint
import py_compile
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BEGIN, END = "# >>> gdnk-a2 >>>", "# <<< gdnk-a2 <<<"
BLOCK = f"""
{BEGIN}  (added by gdnk/a2/apply_a2.py; remove with apply_a2.py --revert)
try:
    from sglang.kernels.ops.attention.fla import gdnk_a2_rt as _gdnk_a2_rt
    _gdnk_a2_rt.install(globals())
except Exception as _gdnk_a2_e:  # loud, never silent: the arm must not run unpatched without saying so
    import sys as _gdnk_a2_sys
    print(f"[gdnk-a2] !! install failed: {{type(_gdnk_a2_e).__name__}}: {{_gdnk_a2_e}}", file=_gdnk_a2_sys.stderr)
    raise
{END}
"""


def pkg_dir():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_a2: sglang not found by importlib.util.find_spec")
    return list(spec.submodule_search_locations)[0]


def paths(pkg):
    fla = os.path.join(pkg, "kernels/ops/attention/fla")
    return dict(fla=fla, kern=os.path.join(fla, "gdnk_a2.py"), rt=os.path.join(fla, "gdnk_a2_rt.py"),
                cfg=os.path.join(fla, "gdnk_a2_cfg.py"),
                gb=os.path.join(pkg, "srt/layers/attention/linear/gdn_backend.py"))


def strip_block(src):
    if BEGIN not in src:
        return src
    a = src.index(BEGIN)
    a = src.rfind("\n", 0, a)
    b = src.index(END, a) + len(END)
    return src[:max(a, 0)] + src[b:]


def triton_version():
    try:
        return importlib.metadata.version("triton")
    except Exception:  # noqa: BLE001
        try:
            import triton
            return triton.__version__
        except Exception:  # noqa: BLE001
            return "?"


def build_config(res):
    cfg = {"triton": res.get("env", {}).get("triton"), "bench_env": res.get("env", {})}
    for sec in ("verify", "conv", "fused"):
        r = res.get(sec, {}) or {}
        cfg[sec] = dict(enabled=bool(r.get("enabled")), best=r.get("best"), best_cfg=r.get("best_cfg"))
        if sec == "conv":
            cfg[sec].update(arith=r.get("arith"), write_state=bool(r.get("write_state")), inplace=bool(r.get("inplace")))
            if not r.get("arith"):
                cfg[sec]["enabled"] = False
    if cfg["fused"]["enabled"] and not cfg["conv"].get("arith"):
        cfg["fused"]["enabled"] = False
    return cfg


def apply(args):
    cfg_path = args.config or os.environ.get("A2_CONFIG") or os.path.join(HERE, "a2_bench_result.json")
    if not os.path.exists(cfg_path):
        print(f"apply_a2: no bench result {cfg_path} -> run a2_bench.py first (nothing applied)", file=sys.stderr)
        return 3
    cfg = build_config(json.load(open(cfg_path)))
    tv = triton_version()
    if cfg["triton"] and tv != cfg["triton"] and os.environ.get("A2_ALLOW_VERSION_MISMATCH") != "1":
        print(f"apply_a2: bench ran with triton {cfg['triton']}, this env has {tv}: bitwise identity is not proven "
              f"here (set A2_ALLOW_VERSION_MISMATCH=1 to apply anyway)", file=sys.stderr)
        return 4
    on = [s for s in ("verify", "conv", "fused") if cfg[s]["enabled"]]
    if not [s for s in on if s != "fused"]:
        print(f"apply_a2: bench result {cfg_path} enables nothing (no config was 100% bitwise AND faster) -> not "
              f"applied", file=sys.stderr)
        return 3
    pkg = pkg_dir()
    p = paths(pkg)
    if not os.path.isdir(p["fla"]) or not os.path.exists(p["gb"]):
        print(f"apply_a2: unexpected sglang layout under {pkg}", file=sys.stderr)
        return 5
    shutil.copyfile(os.path.join(HERE, "kernels.py"), p["kern"])
    shutil.copyfile(os.path.join(HERE, "a2_runtime.py"), p["rt"])
    with open(p["cfg"], "w") as f:
        f.write("# written by gdnk/a2/apply_a2.py from " + os.path.basename(cfg_path) + "\nCONFIG = "
                + pprint.pformat(cfg, width=118) + "\n")
    src = open(p["gb"]).read()
    new = strip_block(src).rstrip("\n") + "\n" + BLOCK
    if new != src:
        with open(p["gb"], "w") as f:
            f.write(new)
    print(f"apply_a2: applied to {pkg}: enabled {on} (verify={cfg['verify']['best']}, conv={cfg['conv']['best']}, "
          f"fused={cfg['fused']['best']} [needs SGL_A2_FUSED=1]); triton {tv}")
    return 0


def revert():
    p = paths(pkg_dir())
    src = open(p["gb"]).read()
    if BEGIN in src:
        open(p["gb"], "w").write(strip_block(src).rstrip("\n") + "\n")
    for k in ("kern", "rt", "cfg"):
        if os.path.exists(p[k]):
            os.remove(p[k])
    print("apply_a2: reverted")
    return 0


SELFTEST = r'''
import sys, torch
from sglang.kernels.ops.attention.fla import gdnk_a2 as K2, gdnk_a2_rt as RT
from sglang.kernels.ops.attention.fla.gdnk_a2_cfg import CONFIG
torch.manual_seed(0)
dev, bf = "cuda", torch.bfloat16
B, T, H, HV, D, NS = 6, 8, 2, 6, 128, 9
bad = []
if CONFIG["verify"]["enabled"]:
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update as orig
    w = RT.make_verify_wrapper(orig, CONFIG["verify"]["best_cfg"])
    q = torch.randn(1, B*T, H, D, device=dev, dtype=bf); k = torch.randn_like(q); v = torch.randn(1, B*T, HV, D, device=dev, dtype=bf)
    a = torch.randn(B*T, HV, device=dev, dtype=bf); b = torch.randn_like(a)
    Al = torch.rand(HV, device=dev) * 2; dtb = torch.randn(HV, device=dev) * 0.1
    st = (torch.randn(NS, HV, D, D, device=dev) * 0.3).to(bf)
    idx = (torch.randperm(B, device=dev) + 1).to(torch.int32); rows = torch.arange(B, device=dev, dtype=torch.int32)
    cu = torch.arange(0, B*T+1, T, device=dev, dtype=torch.int32)
    outs = []
    for fn in (orig, w):
        it = torch.zeros(NS, T, HV, D, D, device=dev, dtype=bf)
        o = fn(A_log=Al, dt_bias=dtb, q=q, k=k, v=v, a=a, b=b, initial_state_source=st, initial_state_indices=idx,
               cu_seqlens=cu, use_qk_l2norm_in_kernel=True, softplus_beta=1.0, softplus_threshold=20.0, is_kda=False,
               disable_state_update=True, intermediate_states_buffer=it, intermediate_state_indices=rows, cache_steps=T)
        outs.append((o, it))
    torch.cuda.synchronize()
    if not (torch.equal(outs[0][0].view(torch.int16), outs[1][0].view(torch.int16))
            and torch.equal(outs[0][1].view(torch.int16), outs[1][1].view(torch.int16))):
        bad.append("verify")
if CONFIG["conv"]["enabled"]:
    import sglang.srt.layers.attention.linear.gdn_backend as gbm
    g = gbm.causal_conv1d_update
    orig = getattr(g, "_gdnk_a2_orig", g)
    w = RT.make_conv_wrapper(orig, CONFIG["conv"])
    C = 640
    res = []
    for fn in (orig, w):
        torch.manual_seed(1)
        xr = torch.randn(B*T, C, device=dev, dtype=bf)
        cs = torch.randn(NS, C, 3, device=dev, dtype=bf); wt = torch.randn(C, 4, device=dev, dtype=bf)
        ph = torch.zeros(NS, C, T + 2, device=dev, dtype=bf)
        win = ph.as_strided((NS, T, C, 3), (ph.stride(0), 1, ph.stride(1), 1))
        idx = (torch.randperm(B, device=dev) + 1).to(torch.int32); rows = torch.arange(B, device=dev, dtype=torch.int32)
        o = fn(xr.view(B, T, C).transpose(1, 2), cs, wt, None, "silu", conv_state_indices=idx,
               intermediate_conv_window=win, intermediate_state_indices=rows)
        res.append((o.clone(), ph, cs))
    torch.cuda.synchronize()
    if not all(torch.equal(x.view(torch.int16), y.view(torch.int16)) for x, y in zip(res[0], res[1])):
        bad.append("conv")
print("SELFTEST", "OK" if not bad else "FAILED " + ",".join(bad))
sys.exit(1 if bad else 0)
'''


def check():
    pkg = pkg_dir()
    p = paths(pkg)
    src = open(p["gb"]).read() if os.path.exists(p["gb"]) else ""
    missing = [k for k in ("kern", "rt", "cfg") if not os.path.exists(p[k])]
    if BEGIN not in src or END not in src or missing:
        print(f"apply_a2 --check: patch NOT present in {pkg} (block={'yes' if BEGIN in src else 'no'}, "
              f"missing={missing})", file=sys.stderr)
        return 1
    py_compile.compile(p["gb"], doraise=True)
    env = dict(os.environ)
    code = ("import importlib; m=importlib.import_module('sglang.kernels.ops.attention.fla.gdnk_a2'); "
            "importlib.import_module('sglang.kernels.ops.attention.fla.gdnk_a2_rt'); "
            "from sglang.kernels.ops.attention.fla.gdnk_a2_cfg import CONFIG; print('cfg', {k: CONFIG[k]['best'] "
            "for k in ('verify','conv','fused')})")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print("apply_a2 --check: copied modules do not import:\n" + r.stderr[-2000:], file=sys.stderr)
        return 1
    print("apply_a2 --check:", r.stdout.strip())
    r = subprocess.run([sys.executable, "-c", "import sglang.srt.layers.attention.linear.gdn_backend as g, "
                        "sglang.srt.layers.attention.linear.kernels.gdn_triton as t; "
                        "print('gdn_backend imported; conv wrapped:', getattr(g.causal_conv1d_update,'_gdnk_a2',False),"
                        " '| verify wrapped:', getattr(t.fused_sigmoid_gating_delta_rule_update,'_gdnk_a2',False))"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        err = r.stderr
        if "gdnk" in err or "gdnk_a2" in err:
            print("apply_a2 --check: gdn_backend import fails inside the a2 block:\n" + err[-2500:], file=sys.stderr)
            return 1
        print("apply_a2 --check: gdn_backend import fails for a reason outside the patch (env lacks server deps?): "
              + err.strip().splitlines()[-1][:300] + " -> import check skipped", file=sys.stderr)
    else:
        print("apply_a2 --check:", r.stdout.strip(), "|", (r.stderr.strip().splitlines() or [""])[-1][:200])
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        gpu = False
    if gpu:
        r = subprocess.run([sys.executable, "-c", SELFTEST], capture_output=True, text=True, env=env)
        print("apply_a2 --check:", (r.stdout.strip().splitlines() or ["(no output)"])[-1])
        if r.returncode != 0:
            print(r.stderr[-2500:], file=sys.stderr)
            return 1
    else:
        print("apply_a2 --check: no GPU visible -> bitwise self-test skipped")
    print("apply_a2 --check: OK (patch present, modules import)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--config", default="")
    a = ap.parse_args()
    sys.exit(check() if a.check else revert() if a.revert else apply(a))
