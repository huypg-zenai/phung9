#!/usr/bin/env bash
# build_wq.sh [OUT=/opt/kopt_wq] - IMAGE-BUILD step (Dockerfile stage `kopt_wq_build`): builds the kopt-wq FP8 GEMM extension
# FROM THE SOURCES IN THIS DIRECTORY (kopt_wq_gemm.cuh + sgl_csrc/ + the 7 configs named in kopt_wq_table.json, via
# `apply_wq.py --build`) into OUT/kopt_wq_ext.so + OUT/manifest.json. No GPU is needed: apply_wq.py passes an explicit
# -gencode=arch=compute_90a,code=sm_90a (torch then adds no arch flags of its own). Toolchain = the one the extension runs
# with: torch $KOPT_WQ_TORCH+cu129 (= the coder env /opt/sglang-env that /opt/ls/setup_sglang_env.sh builds at container
# start; kopt_wq.py refuses a torch mismatch at run time), python3.11 (= that env's interpreter), nvcc of this base image
# (/usr/local/cuda-12.8, = the nvcc that sgl-kernel's flags were reproduced with), ninja 1.13.2, CUTLASS @57e3cfb4
# (= sgl-kernel 0.4.6.post1's pin; apply_wq.py downloads it and checks its sha256).
# NEVER fails the image build: any error -> one loud '!! kopt-wq' line + OUT/BUILD_FAILED + no .so; at coder start
# kopt_shim.sh then prints '[v5-kopt] wq REFUSED: kit file wq/built/kopt_wq_ext.so not present -> SGL_WQ=0' (stock path).
# Full build log: OUT/BUILD.log (kept in the image).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/opt/kopt_wq}
TORCH_V=${KOPT_WQ_TORCH:-2.13.0}
TORCH_IDX=${KOPT_WQ_TORCH_INDEX:-https://download.pytorch.org/whl/cu129}
JOBS=${KOPT_WQ_JOBS:-2}   # 2 parallel nvcc: CUTLASS sm90 templates need ~5 GB each and this stage runs beside the main one
VENV=${KOPT_WQ_VENV:-/opt/kopt_wq_venv}
export CUDA_HOME=${KOPT_WQ_CUDA_HOME:-/usr/local/cuda-12.8}
mkdir -p "$OUT"
rm -f "$OUT/kopt_wq_ext.so" "$OUT/manifest.json" "$OUT/BUILD_FAILED"

build() {  # (set -e does not work inside an if condition: every step has its own || return 1)
  echo "kopt-wq build: nvcc $("$CUDA_HOME/bin/nvcc" --version | tail -2 | head -1)"
  python3.11 -m venv "$VENV" || return 1
  "$VENV/bin/pip" install -q pip==26.2.1 || return 1
  "$VENV/bin/pip" install -q "torch==$TORCH_V" --index-url "$TORCH_IDX" || return 1
  "$VENV/bin/pip" install -q ninja==1.13.2 || return 1
  local tv; tv=$("$VENV/bin/python" -c 'import torch; print(torch.__version__)') || return 1
  echo "kopt-wq build: torch $tv, python $("$VENV/bin/python" -V 2>&1)"
  [ "$tv" = "$TORCH_V+cu129" ] || { echo "torch $tv != $TORCH_V+cu129"; return 1; }
  "$VENV/bin/python" "$HERE/apply_wq.py" --build --prefix "$OUT" --jobs "$JOBS" || return 1
  # the manifest must name exactly the table's configs and the table we ship
  "$VENV/bin/python" - "$OUT/manifest.json" "$HERE/kopt_wq_table.json" <<'PY'
import hashlib, json, sys
m, t = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
used = sorted({c for e in t["table"].values() for c in e.values()})
assert m["configs"] == used, (m["configs"], used)
assert m["table_sha256"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
print(f"kopt-wq build: manifest OK ({len(used)} configs, torch {m['torch']}, so sha256 {m['so_sha256'][:16]}...)")
PY
  [ $? = 0 ] || return 1
}

t0=$(date +%s)
if (build) > "$OUT/BUILD.log" 2>&1; then
  grep '^kopt-wq build\|^apply_wq: built' "$OUT/BUILD.log"
  echo "kopt-wq: extension built from source in $(( $(date +%s) - t0 )) s -> $OUT/kopt_wq_ext.so"
else
  tail -25 "$OUT/BUILD.log"
  echo "!! kopt-wq: build FAILED after $(( $(date +%s) - t0 )) s (log above, $OUT/BUILD.log) -> no extension;" \
       "SGL_WQ=1 will be REFUSED at coder start (stock FP8 GEMM path)"
  rm -f "$OUT/kopt_wq_ext.so" "$OUT/manifest.json"
  : > "$OUT/BUILD_FAILED"
fi
# keep the generated per-config sources (OUT/src) for review; drop the CUTLASS tree, the object files and the venv
rm -rf "$OUT/build" "$OUT"/cutlass-* "$OUT/cutlass.tgz" "$VENV"
exit 0
