"""Apply / build / check / revert the kopt-wq patch (FP8 GEMM tile table, see kopt_wq.py) on the sglang that
`import sglang` resolves to (apply_gsw.py / apply_fusedq.py style). SGLang @1cf2b8c54 + sgl_kernel 0.4.6.post1, sm90.

  python apply_wq.py --build [--prefix /opt/kopt_wq] [--cutlass DIR] [--jobs 16]
        build ONE extension with only the table's configs -> PREFIX/kopt_wq_ext.so + manifest.json (needs nvcc, ninja,
        CUTLASS @57e3cfb47a2d (sgl-kernel's pin; downloaded + sha256-checked into PREFIX if --cutlass is not given)).
        ~2-4 min on 16+ cores. The served process only loads the .so (no compiler at boot).
  python apply_wq.py            apply (idempotent): copy kopt_wq.py + the table into <sglang>/srt/layers/quantization/
                                as kopt_wq.py / kopt_wq_table.json and append the hook (markers '# kopt wq') to
                                fp8_utils.py; the original fp8_utils.py sha256 is recorded in kopt_wq.orig.json.
  python apply_wq.py --check    exit 0 only if the copies are current, the hook is present, the extension matches the
                                table AND (subprocess, GPU, SGL_WQ=1) the hook rebinds fp8_utils.fp8_scaled_mm, and for
                                EVERY table entry apply_fp8_linear (bf16 -> per-token quant -> GEMM, the server's path)
                                is bitwise equal to the stock path: eager (probe call + 2 table calls) and in a captured
                                CUDA graph replayed on new data; the extension path must really be taken.
  python apply_wq.py --revert   remove the hook + copies; verifies fp8_utils.py is byte-identical (sha256) to the
                                recorded original.
Runtime switch: SGL_WQ=1 (unset/0 = the hook does nothing, original code path). SGL_WQ_DIR overrides /opt/kopt_wq.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HOOK = """
# kopt wq >>> (apply_wq.py; active only with SGL_WQ=1)
import os as _kopt_wq_os
if _kopt_wq_os.environ.get("SGL_WQ", "0") == "1":
    try:
        from sglang.srt.layers.quantization import kopt_wq as _kopt_wq
        _kopt_wq.install()
    except Exception as _kopt_wq_e:  # never break the server: production path
        import sys as _kopt_wq_sys
        print(f"[kopt-wq] fallback: {_kopt_wq_e!r}", file=_kopt_wq_sys.stderr, flush=True)
# kopt wq <<<
"""
CUTLASS_COMMIT = "57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
CUTLASS_SHA256 = "09237099a70f80bff1dc8bb80c843a674bb4fdcb46e43cc6993e711c5ca89bb5"
CUTLASS_URL = f"https://github.com/NVIDIA/cutlass/archive/{CUTLASS_COMMIT}.tar.gz"
DEF_PREFIX = os.environ.get("SGL_WQ_DIR", "/opt/kopt_wq")
# sgl-kernel SM90 build flags (CMakeLists.txt:121-152, :331 -use_fast_math, :243 sm_90a) - same as waveq_bench.py
FLAGS = [
    "-O3", "-DNDEBUG", "-gencode=arch=compute_90a,code=sm_90a",
    "-DCUTE_USE_PACKED_TUPLE=1", "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1", "-DCUTLASS_VERSIONS_GENERATED",
    "-DCUTLASS_TEST_LEVEL=0", "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1", "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "--expt-relaxed-constexpr", "--expt-extended-lambda", "-use_fast_math",
    "-Xcudafe=--diag_suppress=177", "-Xcudafe=--diag_suppress=2361",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]


def sgl_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_wq: sglang not found on this interpreter's path")
    return list(spec.submodule_search_locations)[0]


def paths(root):
    q = os.path.join(root, "srt", "layers", "quantization")
    return (os.path.join(q, "kopt_wq.py"), os.path.join(q, "kopt_wq_table.json"), os.path.join(q, "fp8_utils.py"),
            os.path.join(q, "kopt_wq.orig.json"))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def src_table():
    return os.path.join(HERE, "kopt_wq_table.json")


# ------------------------------------------------------------------------------------------------ apply / revert
def apply(root):
    rt, tb, fu, orig = paths(root)
    if not os.path.exists(fu):
        sys.exit(f"apply_wq: {fu} missing (SGLang layout differs)")
    src = open(fu).read()
    for need in ("from sgl_kernel import fp8_scaled_mm", "output = fp8_scaled_mm(", "def apply_fp8_linear("):
        if need not in src:
            sys.exit(f"apply_wq: anchor not found in fp8_utils.py: {need}")
    if HOOK not in src:
        if "# kopt wq >>>" in src:
            sys.exit("apply_wq: a different '# kopt wq' hook is present; run --revert with the old apply_wq.py first")
        json.dump({"sha256": sha(fu), "size": os.path.getsize(fu), "file": fu,
                   "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}, open(orig, "w"), indent=1)
    for s, d in (("kopt_wq.py", rt), ("kopt_wq_table.json", tb)):
        shutil.copyfile(os.path.join(HERE, s), d)
    if HOOK not in src:
        with open(fu, "a") as f:
            f.write(HOOK)
    print(f"apply_wq: runtime -> {rt}\napply_wq: table   -> {tb}\napply_wq: hook    -> {fu} (active only with SGL_WQ=1)")


def revert(root):
    rt, tb, fu, orig = paths(root)
    ok = True
    if os.path.exists(fu):
        src = open(fu).read()                       # read before rewriting (fusedq truncation lesson)
        if HOOK in src:
            with open(fu, "w") as f:
                f.write(src.replace(HOOK, "", 1))
        elif "# kopt wq >>>" in src:
            print("apply_wq: WARNING: a modified '# kopt wq' block is present; not touched")
            ok = False
    for p in (rt, tb):
        if os.path.exists(p):
            os.remove(p)
    if os.path.exists(orig):
        rec = json.load(open(orig))
        now = sha(fu)
        if now == rec["sha256"]:
            print(f"apply_wq: fp8_utils.py byte-identical to the original (sha256 {now[:16]}...)")
            os.remove(orig)
        else:
            print(f"apply_wq: WARNING fp8_utils.py sha256 {now} != recorded original {rec['sha256']} "
                  f"(another patch changed it after apply?); record kept at {orig}")
            ok = False
    else:
        print("apply_wq: no original record (was never applied here?)")
    print(f"apply_wq: reverted{'' if ok else ' WITH WARNINGS'}")
    return ok


# ------------------------------------------------------------------------------------------------ build
def get_cutlass(prefix, given):
    if given:
        if not os.path.isdir(os.path.join(given, "include", "cutlass")):
            sys.exit(f"apply_wq: no cutlass at {given}")
        return given
    d = os.path.join(prefix, f"cutlass-{CUTLASS_COMMIT}")
    if os.path.isdir(os.path.join(d, "include", "cutlass")):
        return d
    tgz = os.path.join(prefix, "cutlass.tgz")
    print(f"apply_wq: downloading {CUTLASS_URL}", flush=True)
    urllib.request.urlretrieve(CUTLASS_URL, tgz)
    if sha(tgz) != CUTLASS_SHA256:
        sys.exit(f"apply_wq: cutlass tarball sha256 {sha(tgz)} != {CUTLASS_SHA256}")
    with tarfile.open(tgz) as t:
        t.extractall(prefix)
    os.remove(tgz)
    return d


def build(prefix, cutlass, jobs):
    import torch
    from torch.utils.cpp_extension import load
    binp = os.path.join(sys.prefix, "bin")
    os.environ["PATH"] = binp + ":" + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["MAX_JOBS"] = str(jobs)
    os.makedirs(prefix, exist_ok=True)
    cutlass = get_cutlass(prefix, cutlass)
    tab = json.load(open(src_table()))
    used = sorted({c for e in tab["table"].values() for c in e.values()})
    if not used:
        sys.exit("apply_wq: table has no entries")
    srcd = os.path.join(prefix, "src")
    shutil.rmtree(srcd, ignore_errors=True)
    os.makedirs(srcd)
    shutil.copyfile(os.path.join(HERE, "kopt_wq_gemm.cuh"), os.path.join(srcd, "kopt_wq_gemm.cuh"))
    sources, decl, cases = [], [], []
    for i, n in enumerate(used):
        s, tm, tn, tk, cm, cn, ext = tab["configs"][n]
        p = os.path.join(srcd, f"cfg{i}_{n}.cu")
        with open(p, "w") as f:
            f.write(f"// generated by apply_wq.py --build: config {n}\n"
                    + ("#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED\n" if ext else "")
                    + '#include "kopt_wq_gemm.cuh"\n'
                    f"torch::Tensor kopt_wq_run_{i}(torch::Tensor const& a, torch::Tensor const& b, "
                    f"torch::Tensor const& sa, torch::Tensor const& sb) {{\n"
                    f"  return kopt_wq_mm<{s}, {tm}, {tn}, {tk}, {cm}, {cn}>(a, b, sa, sb);\n}}\n")
        sources.append(p)
        decl.append(f"torch::Tensor kopt_wq_run_{i}(torch::Tensor const&, torch::Tensor const&, torch::Tensor const&, "
                    f"torch::Tensor const&);")
        cases.append(f"    case {i}: return kopt_wq_run_{i}(a, b, sa, sb);")
    main = os.path.join(srcd, "kopt_wq_ext.cpp")
    with open(main, "w") as f:
        f.write("// generated by apply_wq.py --build\n#include <torch/extension.h>\n" + "\n".join(decl) + "\n"
                "static torch::Tensor mm(int64_t idx, torch::Tensor const& a, torch::Tensor const& b, "
                "torch::Tensor const& sa, torch::Tensor const& sb) {\n  switch (idx) {\n" + "\n".join(cases) +
                "\n    default: TORCH_CHECK(false, \"kopt_wq: bad config index \", idx);\n  }\n}\n"
                "static std::vector<std::string> names() { return {" + ", ".join(f'"{n}"' for n in used) + "}; }\n"
                "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n  m.def(\"mm\", &mm);\n  m.def(\"names\", &names);\n}\n")
    bdir = os.path.join(prefix, "build")
    os.makedirs(bdir, exist_ok=True)
    t = time.time()
    print(f"apply_wq: building {len(used)} configs {used} with {jobs} jobs ...", flush=True)
    mod = load(name="kopt_wq_ext", sources=[main] + sources, extra_cflags=["-O3"], extra_cuda_cflags=FLAGS,
               extra_include_paths=[os.path.join(HERE, "sgl_csrc"), srcd, os.path.join(cutlass, "include"),
                                    os.path.join(cutlass, "tools/util/include")],
               build_directory=bdir, verbose=False)
    assert list(mod.names()) == used
    shutil.copyfile(os.path.join(bdir, "kopt_wq_ext.so"), os.path.join(prefix, "kopt_wq_ext.so.tmp"))
    os.replace(os.path.join(prefix, "kopt_wq_ext.so.tmp"), os.path.join(prefix, "kopt_wq_ext.so"))
    json.dump({"configs": used, "table_sha256": sha(src_table()), "torch": torch.__version__,
               "cutlass": CUTLASS_COMMIT, "so_sha256": sha(os.path.join(prefix, "kopt_wq_ext.so")),
               "built": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())},
              open(os.path.join(prefix, "manifest.json"), "w"), indent=1)
    print(f"apply_wq: built {prefix}/kopt_wq_ext.so in {time.time() - t:.0f}s (configs {used})")


# ------------------------------------------------------------------------------------------------ check
PROBE = r"""
import os, sys, torch
os.environ["SGL_WQ"] = "1"
from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization import kopt_wq as W
assert getattr(fp8_utils, "_kopt_wq_installed", False) and fp8_utils.fp8_scaled_mm is W.wq_mm, \
    "fp8_scaled_mm not rebound (see the [kopt-wq] line above)"
assert W.ENT, "empty table"
import sgl_kernel
torch.manual_seed(4321)
dev = "cuda"

def wq(N, K):
    w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.05
    w[::3] *= 1e-2
    s = w.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / 448.0
    return (w / s).to(torch.float8_e4m3fn).t(), s.contiguous()     # (K, N) column-major view, like layer.weight

def lin(x, w, s):   # the server's path: Fp8LinearMethod.apply -> apply_fp8_linear (dynamic per-token quant)
    return fp8_utils.apply_fp8_linear(input=x, weight=w, weight_scale=s, input_scale=None, bias=None)

def stock(x, w, s):
    fp8_utils.fp8_scaled_mm = sgl_kernel.fp8_scaled_mm
    try:
        return lin(x, w, s)
    finally:
        fp8_utils.fp8_scaled_mm = W.wq_mm

def eq(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))

n = bad = graphs = 0
shapes = sorted({k[:2] for k in W.ENT})
for (N, K) in shapes:
    w, s = wq(N, K)
    for M in sorted(k[2] for k in W.ENT if k[:2] == (N, K)):
        ent = W.ENT[(N, K, M)]
        for d in range(3):        # d=0: first use = probe (returns production); d=1,2: table kernel
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16) * (2 if d != 2 else 300)
            if d == 2:
                x[:, ::5] = 0
            c0 = W.ST["ext_calls"]
            a, b = lin(x, w, s), stock(x, w, s)
            n += 1
            if not eq(a, b):
                bad += 1
                print(f"MISMATCH N={N} K={K} M={M} cfg={ent[2]} draw {d}")
            if d > 0 and ent[1] == W.OK and W.ST["ext_calls"] != c0 + 1:
                bad += 1
                print(f"EXT NOT USED N={N} K={K} M={M}")
        if ent[1] != W.OK:
            bad += 1
            print(f"ENTRY NOT OK after probe: N={N} K={K} M={M} cfg={ent[2]} state={ent[1]}")
        # CUDA graph: capture once, replay on new input data
        xs = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            lin(xs, w, s)
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        c0 = W.ST["ext_calls"]
        with torch.cuda.graph(gr):
            yo = lin(xs, w, s)
        used = W.ST["ext_calls"] == c0 + 1
        for r in range(2):
            xs.copy_(torch.randn(M, K, device=dev, dtype=torch.bfloat16) * (1 + 50 * r))
            gr.replay(); torch.cuda.synchronize()
            n += 1; graphs += 1
            if not eq(yo, stock(xs, w, s)) or not used:
                bad += 1
                print(f"GRAPH MISMATCH N={N} K={K} M={M} cfg={ent[2]} (ext captured: {used})")
        del gr, yo
    torch.cuda.empty_cache()
# a shape/M NOT in the table must stay on production
x = torch.randn(4096, 5120, device=dev, dtype=torch.bfloat16); w, s = wq(5120, 5120)
c0 = W.ST["ext_calls"]; n += 1
if not eq(lin(x, w, s), stock(x, w, s)) or W.ST["ext_calls"] != c0:
    bad += 1; print("NON-TABLE CALL CHANGED")
print(f"probe: {n - bad}/{n} bitwise-equal ({len(W.ENT)} table entries over {len(shapes)} shapes, {graphs} graph "
      f"replays, ext calls {W.ST['ext_calls']}, probes {W.ST['probes']}, drops {W.ST['drops']})")
sys.exit(0 if bad == 0 else 1)
"""


def check(root, prefix):
    rt, tb, fu, orig = paths(root)
    ok = True
    for s, d in (("kopt_wq.py", rt), ("kopt_wq_table.json", tb)):
        if sha(d) != sha(os.path.join(HERE, s)):
            print(f"check: copy missing/stale: {d}")
            ok = False
    if not os.path.exists(fu) or HOOK not in open(fu).read():
        print(f"check: hook missing in {fu}")
        ok = False
    man = os.path.join(prefix, "manifest.json")
    if not os.path.exists(man) or json.load(open(man)).get("table_sha256") != sha(src_table()):
        print(f"check: extension in {prefix} missing or built for another table (run --build)")
        ok = False
    if ok:
        env = dict(os.environ, SGL_WQ="1", SGL_WQ_DIR=prefix)
        r = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, cwd="/", env=env)
        out = [l for l in (r.stdout + "\n" + r.stderr).splitlines() if l.strip()
               and ("kopt-wq" in l or "probe:" in l or "MISMATCH" in l or "NOT" in l or "Error" in l
                    or "assert" in l.lower())]
        print("\n".join(out[-40:]) or (r.stderr.strip()[-2000:]))
        ok = r.returncode == 0
    print(f"check: {'PASS' if ok else 'FAIL'} (sglang at {root}, extension {prefix})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--prefix", default=DEF_PREFIX)
    ap.add_argument("--cutlass", default="")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    if a.build:
        build(a.prefix, a.cutlass, a.jobs)
    elif a.check:
        sys.exit(0 if check(sgl_root(), a.prefix) else 1)
    elif a.revert:
        sys.exit(0 if revert(sgl_root()) else 1)
    else:
        apply(sgl_root())
