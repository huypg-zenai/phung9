#!/usr/bin/env bash
# /opt/v5/kopt_shim.sh <spawn.py's vllm_bin args...>   (entry 3 combo, 21/10)
# Installs the kernel-optim coder handovers (the kit dirs below, md5 of every file in MANIFEST)
# IN PLACE into /opt/sglang-env at coder start, each FAIL CLOSED, each behind its own 0/1 switch (only "1" = on):
#   item      switch        class  kit dir here            what it edits in <sglang>
#   wq        SGL_WQ        A      wq/                     srt/layers/quantization/fp8_utils.py (append) + 2 copies;
#                                                          ext built from source (apply_wq.py --build) in /opt/kopt_wq (= $SGL_WQ_DIR)
#   fusedq    SGL_FUSEDQ    A      fusedq/                 srt/models/qwen3_5.py (append) + srt/layers/kopt_fusedq.py + 2 .cuh
#   mrope_nb  SGL_MROPE_NB  A      mrope_nb/               srt/model_executor/forward_batch_info.py (1 line + import)
#   topkbf    SGL_TOPKBF    A      topkbf_normcfg/topkbf/  srt/layers/logits_processor.py (2 hooks), dflash_utils.py
#                                                          (dispatch), dflash_worker_v2.py (guard) + kopt_topkbf.py
#   normcfg   SGL_NORMCFG   A      topkbf_normcfg/normcfg/ kernels/ops/attention/fla/layernorm_gated.py (2 blocks)
#   casc      SGL_CASC      B      casc/                   srt/layers/attention/flashattention_backend.py (append) + copy
# Coder start chain (Dockerfile: KERN_REAL_SHIM=/opt/v4/speed_shim.sh, V4_REAL_SHIM=/opt/v5/kopt_shim.sh):
#   kern_shim.sh (GDN) -> /opt/v4/speed_shim.sh (gsw + topk20, entry 2, unchanged) -> THIS -> /opt/v3/coder_switch.sh
#   (seed patch) -> /opt/sgl/sgl_shim.sh "$@".   Apply order and why: COMBO.md "Apply order".
# Per requested item (fail closed, like v4): kit files present + listed in MANIFEST + MANIFEST md5 OK -> every GPU
# nvidia-smi lists is one of V5_GPU_NAMES (default "NVIDIA H200") -> item rules (below) -> apply -> --check (as its
# README says) must print ALL its PASS lines (incl. the GPU self-test line) and none of its forbidden lines -> switch
# exported 1. Any failure / timeout -> its --revert, marker $V5_MARKER_DIR/v5_<item>_refused, switch exported 0, and
# exactly one line '[v5-kopt] <item> REFUSED: <why> -> ...'. Every hook is dormant at switch 0 (checked by the tests).
# Item rules: wq replaces gsw (same fp8_scaled_mm hook, README "do not combine"): its check runs with SGL_GSW=0 and, if
#   it passes, SGL_GSW is exported 0 (the gsw hook stays in fp8_utils.py but installs nothing); if wq is refused, gsw
#   stays as v4 left it. wq is refused with SGL_FP8PIN=1, or when the built ext at $SGL_WQ_DIR differs from its kit copy wq/built.
#   fusedq: SGL_FUSEDQ_NORM is forced to 0 (README: 1 ulp off -> never on). casc (class B, image default OFF): needs
#   SGL_CASC_PRE_SPLITS=1 (README: required; anything else -> casc refused). topkbf: SGL_TOPKBF_MODEL defaults to the
#   coder model id (arg after 'serve') for its --check hidden/vocab sizes.
# Lines for the start gate (COMBO.md "Start-up lines"): per item exactly one of
#   '[v5-kopt] <item> ACTIVE (...)' | '[v5-kopt] <item> REFUSED: ...' | '[v5-kopt] <item> off (...)', then one
#   '[v5-kopt] summary: ...' line. This script NEVER stops the coder: it always ends in `exec bash $V5_REAL_SHIM "$@"`.
# Test hooks: V5_REAL_SHIM (default /opt/v3/coder_switch.sh), V5_DIR (this script's dir), V5_MARKER_DIR (/tmp),
# V5_NVSMI (nvidia-smi), V5_LOCK (/tmp/v5_kopt.lock), V5_CHECK_TIMEOUT (900 s per step), SGL_PYTHON (/opt/sglang-env/bin/python).
set -u
D=${V5_DIR:-$(cd "$(dirname "$0")" && pwd)}
REAL=${V5_REAL_SHIM:-/opt/v3/coder_switch.sh}
PY=${SGL_PYTHON:-/opt/sglang-env/bin/python}
MK=${V5_MARKER_DIR:-/tmp}
LOCK=${V5_LOCK:-/tmp/v5_kopt.lock}
TO=${V5_CHECK_TIMEOUT:-900}
NAMES=${V5_GPU_NAMES:-NVIDIA H200}
NVSMI=${V5_NVSMI:-nvidia-smi}
WQ_PREFIX=${SGL_WQ_DIR:-/opt/kopt_wq}
T="[v5-kopt]"
ITEMS="wq fusedq mrope_nb topkbf normcfg casc"
say() { echo "$T $*"; }
loud() { echo "$T !! $*"; }
b01() {  # name value -> 0/1; anything else = '!!' + 0 (off)
  case $2 in 0 | 1) echo "$2" ;; *) loud "$1='$2' is not 0 or 1 -> treated as 0 (off)" >&2; echo 0 ;; esac
}
knob() { case $1 in wq) echo SGL_WQ ;; fusedq) echo SGL_FUSEDQ ;; mrope_nb) echo SGL_MROPE_NB ;; topkbf) echo SGL_TOPKBF ;;
  normcfg) echo SGL_NORMCFG ;; casc) echo SGL_CASC ;; esac; }
script() { case $1 in wq) echo wq/apply_wq.py ;; fusedq) echo fusedq/apply_fusedq.py ;; mrope_nb) echo mrope_nb/apply_mrope.py ;;
  topkbf) echo topkbf_normcfg/topkbf/apply_topkbf.py ;; normcfg) echo topkbf_normcfg/normcfg/apply_normcfg.py ;;
  casc) echo casc/apply_casc.py ;; esac; }
files() { case $1 in
  wq) echo wq/apply_wq.py wq/kopt_wq.py wq/kopt_wq_table.json wq/built/kopt_wq_ext.so wq/built/manifest.json ;;
  fusedq) echo fusedq/apply_fusedq.py fusedq/fusedq.py fusedq/silu_quant.cuh fusedq/gemma_quant.cuh ;;
  mrope_nb) echo mrope_nb/apply_mrope.py ;;
  topkbf) echo topkbf_normcfg/topkbf/apply_topkbf.py topkbf_normcfg/topkbf/kopt_topkbf.py ;;
  normcfg) echo topkbf_normcfg/normcfg/apply_normcfg.py ;;
  casc) echo casc/apply_casc.py casc/kopt_casc.py ;; esac; }
# PASS lines (ALL must be in the --check output, extended regex, one per line) and forbidden lines (none may be)
need() { case $1 in
  wq) printf '%s\n' '^check: PASS' '^probe: ([0-9]+)/\1 bitwise-equal' '\[kopt-wq\] installed: 362 \(N,K,M\) entries' ;;
  fusedq) printf '%s\n' '^check: PASS' '^bitwise vs production silu_and_mul \+ per-token quant: PASS' ;;
  mrope_nb) printf '%s\n' '^apply_mrope check: patched True, import rc 0' ;;
  topkbf) printf '%s\n' '^apply_topkbf check: PASS \([0-9]+ hooks \+ module, patched modules import, GPU bitwise self-test OK' ;;
  normcfg) printf '%s\n' '^normcfg: CHECK PASS' ;;
  casc) printf '%s\n' '^check: PASS' '^probe SGL_CASC=1 OK' '^probe SGL_CASC=0 OK' ;; esac; }
forbid() { case $1 in wq) echo '\[kopt-wq\] (fallback|drop)' ;; esac; }

declare -A REQ ON
for it in $ITEMS; do k=$(knob "$it"); REQ[$it]=$(b01 "$k" "${!k:-0}"); ON[$it]=0; done
# fusedq's Gemma norm fusion is 1 ulp off in rare rows (README): never on, whatever the environment says
if [ "${SGL_FUSEDQ_NORM:-0}" != 0 ]; then loud "SGL_FUSEDQ_NORM='${SGL_FUSEDQ_NORM}' forced to 0 (fusedq README: do NOT enable)"; fi
export SGL_FUSEDQ_NORM=0
t0=$(date +%s)
any=0; for it in $ITEMS; do [ "${REQ[$it]}" = 1 ] && any=1; done
CPATH=$PATH:$(dirname "$PY")

gpu_guard() {  # -> prints a reason when the box is not allowed, nothing when OK
  if ! command -v "$NVSMI" >/dev/null 2>&1; then echo "$NVSMI not found (cannot verify the GPU model)"; return; fi
  local nm one a ok
  nm=$("$NVSMI" --query-gpu=name --format=csv,noheader 2>/dev/null)
  [ -n "$nm" ] || { echo "nvidia-smi listed no GPU"; return; }
  while IFS= read -r one; do
    one=$(echo "$one" | sed 's/^ *//; s/ *$//')
    ok=0; IFS='|' read -ra allowed <<< "$NAMES"
    for a in "${allowed[@]}"; do [ "$one" = "$a" ] && ok=1; done
    [ $ok = 1 ] || { echo "a GPU of this box is '$one', not one of '$NAMES'"; return; }
  done <<< "$nm"
}

kit_guard() {  # files... -> prints a reason when the kit is missing / not in MANIFEST / md5 differs
  local f
  for f in "$@"; do
    [ -f "$D/$f" ] || { echo "kit file $f not present"; return; }
    grep -q "  $f\$" "$D/MANIFEST" 2>/dev/null || { echo "kit file $f not listed in $D/MANIFEST"; return; }
  done
  (cd "$D" && md5sum --quiet -c MANIFEST >/dev/null 2>&1) || echo "$D differs from its MANIFEST (md5)"
}

rule_guard() {  # item -> prints a reason when an item rule forbids it
  case $1 in
    wq)
      [ "${SGL_FP8PIN:-0}" = 1 ] && { echo "SGL_FP8PIN=1 (same fp8_scaled_mm hook; wq README: do not combine)"; return; }
      local f; for f in kopt_wq_ext.so manifest.json; do
        cmp -s "$WQ_PREFIX/$f" "$D/wq/built/$f" || { echo "built ext $WQ_PREFIX/$f missing or != $D/wq/built/$f (both written at image build)"; return; }
      done ;;
    casc)
      [ "${SGL_CASC_PRE_SPLITS:-1}" = 1 ] || echo "SGL_CASC_PRE_SPLITS='${SGL_CASC_PRE_SPLITS}' (casc README: 1 is required)" ;;
  esac
}

run() {  # name logfile cmd... -> rc (output kept in the log file, its tail echoed)
  local name=$1 log=$2; shift 2
  PATH=$CPATH timeout "$TO" "$@" > "$log" 2>&1; local rc=$?
  sed "s/^/$T $name: /" "$log" | tail -8
  [ $rc = 124 ] && loud "$name: timed out after $TO s"
  return $rc
}

check_env() {  # item -> env assignments for its --check (README commands; earlier decisions already exported)
  case $1 in
    wq) echo "SGL_GSW=0 SGL_WQ_DIR=$WQ_PREFIX" ;;
    fusedq) echo "SGL_FUSEDQ=1 SGL_FUSEDQ_NORM=0" ;;
    topkbf) echo "SGL_TOPKBF_MODEL=$TOPKBF_MODEL" ;;
    casc) echo "SGL_CASC=1 SGL_CASC_PRE_SPLITS=1" ;;
  esac
}

one() {  # item -> 0 = active
  local it=$1 sc why s1 log re
  sc=$(script "$it")
  why=$(kit_guard $(files "$it")); [ -n "$why" ] || why=$GPU_WHY; [ -n "$why" ] || why=$(rule_guard "$it")
  if [ -z "$why" ]; then
    s1=$(date +%s)
    (
      flock -w 600 9 || { loud "$it: lock busy"; exit 1; }
      run "$it-apply" "$MK/v5_${it}_apply.log" "$PY" "$D/$sc" || exit 1
      log="$MK/v5_${it}_check.log"
      # shellcheck disable=SC2046
      run "$it-check" "$log" env $(check_env "$it") "$PY" "$D/$sc" --check || exit 1
      while IFS= read -r re; do
        grep -Eq "$re" "$log" || { loud "$it: PASS line missing in --check output: /$re/"; exit 1; }
      done < <(need "$it")
      re=$(forbid "$it")
      if [ -n "$re" ] && grep -Eq "$re" "$log"; then loud "$it: forbidden line in --check output: /$re/"; exit 1; fi
    ) 9>"$LOCK" && { say "$it ACTIVE ($(knob "$it")=1, apply + --check PASS in $(( $(date +%s) - s1 )) s)"; return 0; }
    why="$sc apply/--check failed (see the lines above, $MK/v5_${it}_*.log)"
    PATH=$CPATH timeout 120 "$PY" "$D/$sc" --revert > "$MK/v5_${it}_revert.log" 2>&1 \
      && say "$it: --revert done" || say "$it: --revert rc $? (its hook stays dormant at $(knob "$it")=0)"
  fi
  say "$it REFUSED: $why -> $(knob "$it")=0, coder starts with the stock path for it"
  : > "$MK/v5_${it}_refused" 2>/dev/null || true
  return 1
}

GPU_WHY=""; [ $any = 1 ] && GPU_WHY=$(gpu_guard)
TOPKBF_MODEL=${SGL_TOPKBF_MODEL:-}
[ -z "$TOPKBF_MODEL" ] && [ "${1:-}" = serve ] && TOPKBF_MODEL=${2:-}
[ $any = 1 ] && say "requested SGL_WQ=${REQ[wq]} SGL_FUSEDQ=${REQ[fusedq]} SGL_MROPE_NB=${REQ[mrope_nb]} SGL_TOPKBF=${REQ[topkbf]}" \
  "SGL_NORMCFG=${REQ[normcfg]} SGL_CASC=${REQ[casc]} (SGL_GSW=${SGL_GSW:-unset} from v4; GPU names allowed: '$NAMES')"
for it in $ITEMS; do
  k=$(knob "$it")
  if [ "${REQ[$it]}" = 1 ]; then
    one "$it" && ON[$it]=1
  else
    say "$it off ($k=0)"
  fi
  export "$k=${ON[$it]}"
  if [ "$it" = wq ] && [ "${ON[wq]}" = 1 ] && [ "${SGL_GSW:-0}" = 1 ]; then
    export SGL_GSW=0
    say "gsw replaced by wq: SGL_GSW exported 0 (the gsw hook in fp8_utils.py stays dormant)"
  fi
done
[ "${ON[casc]}" = 1 ] && export SGL_CASC_PRE_SPLITS=1
say "summary: SGL_WQ=$SGL_WQ SGL_FUSEDQ=$SGL_FUSEDQ SGL_MROPE_NB=$SGL_MROPE_NB SGL_TOPKBF=$SGL_TOPKBF SGL_NORMCFG=$SGL_NORMCFG" \
  "SGL_CASC=$SGL_CASC SGL_GSW=${SGL_GSW:-unset} (requested ${REQ[wq]}/${REQ[fusedq]}/${REQ[mrope_nb]}/${REQ[topkbf]}/${REQ[normcfg]}/${REQ[casc]})" \
  "in $(( $(date +%s) - t0 )) s -> $REAL"
exec bash "$REAL" "$@"
