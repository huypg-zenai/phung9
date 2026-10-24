#!/usr/bin/env bash
# /opt/v4/speed_shim.sh <spawn.py's vllm_bin args...>   (entry 2, 21/10)
# Coder start chain (the image sets ENV KERN_REAL_SHIM=/opt/v4/speed_shim.sh):
#   vllm_bin /opt/prefetch/shim_a2b.sh -> prefetch_shim.sh -> /opt/kern/shim_a2b.sh -> kern_shim.sh (GDN patches,
#   unchanged) -> THIS -> /opt/v3/coder_switch.sh (seed patch, unchanged) -> /opt/sgl/sgl_shim.sh "$@"
# Installs two CLASS-A (bitwise-identical output) coder speed patches IN PLACE into /opt/sglang-env, each FAIL CLOSED:
#   gsw     SGL_GSW=1     FP8 GEMM switch: torch._scaled_mm (fast accum) instead of sgl_kernel fp8_scaled_mm for the
#                         427 (N,K,M) entries of gsw/gsw_table.json (kernel-optim handover gsw_m6, M6 table).
#                         apply_gsw.py, then apply_gsw.py --check (GPU: every table entry bitwise == stock through
#                         apply_fp8_linear, eager + CUDA graph).
#   topk20  SGL_TOPK20=1  exact Triton top-20 for the DFlash verify (kernel-optim handover topk20, M7 cfg).
#                         apply_topk.py, then SGL_TOPK20=1 apply_topk.py --check (import of the patched modules +
#                         GPU selftest bitwise == torch.topk); the GPU selftest MUST have run ("bitwise OK" line).
# Guards (each patch separately): the switch must be 1; /opt/v4/MANIFEST must match (md5); every GPU nvidia-smi lists
# must be exactly one of V4_GPU_NAMES (default "NVIDIA H200": table and config were built and tested there only; all GPUs
# are checked, so CUDA vs nvidia-smi index order does not matter). Any guard, apply or check failure -> one '!!' line,
# the patch is reverted (best effort), its switch is exported as 0 (both are dormant at 0: gsw's hook installs nothing,
# topk20's call site is the stock torch.topk byte for byte), marker /tmp/v4_<name>_refused, and the coder starts exactly
# as in entry 1.
# This script NEVER stops the coder: it always ends in `exec $V4_REAL_SHIM "$@"` with the same arguments.
# Order: before coder_switch.sh on purpose: topk20 edits dflash_utils.py (one call, line ~999) and the seed patch edits
# other hunks of the same file; a topk20 --revert restores its pre-edit backup, and the seed patch is (re)applied after.
# Env: SGL_GSW, SGL_TOPK20 (0/1; unset = 0 here, the Dockerfile sets both to 1), V4_GPU_NAMES ('|'-separated exact
# nvidia-smi names), V4_CHECK_TIMEOUT (s per apply/check step, default 900), SGL_PYTHON (/opt/sglang-env/bin/python).
# Test hooks: V4_REAL_SHIM (default /opt/v3/coder_switch.sh), V4_DIR (default: this script's dir), V4_MARKER_DIR (/tmp),
# V4_NVSMI (default nvidia-smi), V4_LOCK (/tmp/v4_speed.lock).
set -u
D=${V4_DIR:-$(cd "$(dirname "$0")" && pwd)}
REAL=${V4_REAL_SHIM:-/opt/v3/coder_switch.sh}
PY=${SGL_PYTHON:-/opt/sglang-env/bin/python}
MK=${V4_MARKER_DIR:-/tmp}
LOCK=${V4_LOCK:-/tmp/v4_speed.lock}
TO=${V4_CHECK_TIMEOUT:-900}
NAMES=${V4_GPU_NAMES:-NVIDIA H200}
NVSMI=${V4_NVSMI:-nvidia-smi}
T="[v4-speed]"
say() { echo "$T $*"; }
loud() { echo "$T !! $*"; }
b01() {  # name value -> 0/1; anything else = '!!' + 0 (off = entry-1 behaviour)
  case $2 in 0 | 1) echo "$2" ;; *) loud "$1='$2' is not 0 or 1 -> treated as 0 (off)"; echo 0 ;; esac
}
GSW=$(b01 SGL_GSW "${SGL_GSW:-0}")
TOPK=$(b01 SGL_TOPK20 "${SGL_TOPK20:-0}")
t0=$(date +%s)
say "requested SGL_GSW=$GSW SGL_TOPK20=$TOPK (GPU names allowed: '$NAMES', CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
# ninja for the JIT kernels the GPU checks import: APPEND the env's bin dir (never prepend; the image PATH already has
# /opt/vllm-env/bin with ninja). Used for the check subprocesses only; the servers keep the inherited PATH.
CPATH=$PATH:$(dirname "$PY")

why_all=""
if [ "$GSW" = 1 ] || [ "$TOPK" = 1 ]; then
  # guard 1: the kit files are the reviewed ones
  if ! (cd "$D" && md5sum --quiet -c MANIFEST >/dev/null 2>&1); then
    why_all="$D differs from its MANIFEST (md5)"
  fi
  # guard 2: every GPU of the box is an allowed SKU
  if [ -z "$why_all" ]; then
    if ! command -v "$NVSMI" >/dev/null 2>&1; then
      why_all="$NVSMI not found (cannot verify the GPU model)"
    else
      nm=$("$NVSMI" --query-gpu=name --format=csv,noheader 2>/dev/null)
      [ -n "$nm" ] || why_all="nvidia-smi listed no GPU"
      while [ -z "$why_all" ] && IFS= read -r one; do
        one=$(echo "$one" | sed 's/^ *//; s/ *$//')
        ok=0; IFS='|' read -ra allowed <<< "$NAMES"
        for a in "${allowed[@]}"; do [ "$one" = "$a" ] && ok=1; done
        [ $ok = 1 ] || why_all="a GPU of this box is '$one', not one of '$NAMES' (tables/configs tested on H200 only)"
      done <<< "$nm"
    fi
  fi
fi

run() {  # name logfile cmd... -> rc (output kept in the log file and its tail echoed)
  local name=$1 log=$2; shift 2
  PATH=$CPATH timeout "$TO" "$@" > "$log" 2>&1; local rc=$?
  sed "s/^/$T $name: /" "$log" | tail -12
  [ $rc = 124 ] && loud "$name: timed out after $TO s"
  return $rc
}

refuse() {  # name why
  loud "$1 NOT ACTIVE: $2 -> coder starts with the stock path for it (= entry 1)"
  : > "$MK/v4_$1_refused" 2>/dev/null || true
}

# ---- gsw ----
GSW_ON=0
if [ "$GSW" = 1 ]; then
  if [ -n "$why_all" ]; then
    refuse gsw "$why_all"
  else
    (
      flock -w 600 9 || { loud "gsw: lock busy"; exit 1; }
      run gsw-apply "$MK/v4_gsw_apply.log" "$PY" "$D/gsw/apply_gsw.py" || exit 1
      run gsw-check "$MK/v4_gsw_check.log" "$PY" "$D/gsw/apply_gsw.py" --check || exit 1
      grep -q "check: PASS" "$MK/v4_gsw_check.log" || { loud "gsw: no 'check: PASS' line"; exit 1; }
    ) 9>"$LOCK"
    if [ $? = 0 ]; then
      GSW_ON=1
    else
      refuse gsw "apply_gsw.py apply/--check failed (see the lines above, $MK/v4_gsw_*.log)"
    fi
  fi
fi
if [ $GSW_ON = 0 ]; then
  # dormant: the fp8_utils hook installs nothing at SGL_GSW=0; also take the hook + copies out when possible
  if [ -f "$D/gsw/apply_gsw.py" ]; then
    PATH=$CPATH timeout 120 "$PY" "$D/gsw/apply_gsw.py" --revert > "$MK/v4_gsw_revert.log" 2>&1 \
      || say "gsw: --revert failed (rc $?), hook stays dormant with SGL_GSW=0"
  fi
fi
export SGL_GSW=$GSW_ON

# ---- topk20 ----
TOPK_ON=0
if [ "$TOPK" = 1 ]; then
  if [ -n "$why_all" ]; then
    refuse topk20 "$why_all"
  else
    (
      flock -w 600 9 || { loud "topk20: lock busy"; exit 1; }
      run topk20-apply "$MK/v4_topk20_apply.log" "$PY" "$D/topk20/apply_topk.py" || exit 1
      run topk20-check "$MK/v4_topk20_check.log" env SGL_TOPK20=1 "$PY" "$D/topk20/apply_topk.py" --check || exit 1
      grep -q "GPU selftest bitwise OK" "$MK/v4_topk20_check.log" || { loud "topk20: GPU selftest did not run/pass"; exit 1; }
    ) 9>"$LOCK"
    if [ $? = 0 ]; then
      TOPK_ON=1
    else
      refuse topk20 "apply_topk.py apply/--check failed (see the lines above, $MK/v4_topk20_*.log)"
      # Surgical revert: put the stock call line back (the kit's own CALL_TEXT -> CALL_ANCHOR) and remove topk20.py.
      # NOT apply_topk.py --revert: that restores its first backup, which on a restarted container predates the seed
      # patch's dflash_utils.py hunk -> half-applied seed tree -> coder unseeded (CPU selftest case c14b).
      PATH=$CPATH PYTHONDONTWRITEBYTECODE=1 timeout 120 "$PY" - "$D/topk20" > "$MK/v4_topk20_revert.log" 2>&1 <<'PYEOF' \
        && say "topk20: call site restored to the stock line" \
        || say "topk20: surgical revert failed (rc $?), the call site stays dormant (stock torch.topk) with SGL_TOPK20=0"
import os, sys
sys.path.insert(0, sys.argv[1])
import apply_topk as A
root = A.find_root()
p = os.path.join(root, A.DU)
src = open(p).read()
if src.count(A.CALL_TEXT) == 1:
    new = src.replace(A.CALL_TEXT, A.CALL_ANCHOR)
    compile(new, p, "exec")
    tmp = p + ".v4tmp"
    open(tmp, "w").write(new)
    os.replace(tmp, p)
    pyc = os.path.join(os.path.dirname(p), "__pycache__")
    for c in (os.listdir(pyc) if os.path.isdir(pyc) else []):
        if c.startswith("dflash_utils."):
            os.remove(os.path.join(pyc, c))
    print("reverted call site in", p)
elif A.marker("call") in src:
    sys.exit("marker present but the call text differs: left as is")
m = os.path.join(root, A.MODULE_DST)
if os.path.exists(m):
    os.remove(m)
PYEOF
    fi
  fi
fi
# off: no revert needed (SGL_TOPK20=0 = the stock torch.topk call byte for byte, topk20 is not even imported)
export SGL_TOPK20=$TOPK_ON

say "ACTIVE: SGL_GSW=$SGL_GSW SGL_TOPK20=$SGL_TOPK20 (requested $GSW/$TOPK) in $(( $(date +%s) - t0 )) s -> $REAL"
exec bash "$REAL" "$@"
