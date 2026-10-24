#!/usr/bin/env bash
# kern_shim.sh <KERN> <spawn.py's vllm_bin args...>  (KERN knob of run_arm2.sh; mounted read-only at /opt/kern)
# The coder's vllm_bin in configuration.yaml points at /opt/kern/shim_<KERN>.sh -> here. Applies the GDN kernel patches
# IN PLACE to the container's SGLang (/opt/sglang-env; the container is fresh per run, the image is never changed),
# runs every patch's own --check, then execs the normal coder shim (/opt/sgl/sgl_shim.sh: plain or BURST variant).
# Any failure exits non-zero BEFORE SGLang starts, so the coder can never silently run unpatched.
#   a1a2  = A1 with guard (dd2/a1g) + A2 conv c64w2 + A2 verify replica V0_bv32w1 (output-identical)
#   b     = B-raw alone (dd2/braw, ieee+tf32x3+ieee BV128 w2 = the dd2 server-arm config; OUTPUT-CHANGING)
#   a2b   = A2 + B-raw (OUTPUT-CHANGING)
#   a1a2b = REFUSED: A1 + B together = broken output (both replace the commit; smoke 18/10 accept 1.0, garbage)
set -u
KERN=$1; shift
K=/opt/kern; PY=${SGL_PYTHON:-/opt/sglang-env/bin/python}; REAL=${KERN_REAL_SHIM:-/opt/sgl/sgl_shim.sh}
BENV() { export SGL_GDN_B=1 SGL_GDN_B_PREC=raw SGL_GDN_B_PREC_S=ieee+tf32x3+ieee SGL_GDN_B_BV=128 SGL_GDN_B_WARPS=2 \
                SGL_GDN_B_FOLD_BV=32 SGL_GDN_B_FOLD_WARPS=4; }
case $KERN in
  a1a2)  P="a1g a2" ;;
  b)     P="braw"; BENV ;;        # B-raw alone (= the dd2 braw server arm)
  a2b)   P="a2 braw"; BENV ;;     # A2 (conv + verify replica) + B-raw
  a1a2b) echo "[kern-shim] !! KERN=a1a2b is BROKEN (A1 and B both replace the GDN commit: smoke 18/10 = garbage text," \
              "accept length 1.0) - use a1a2, b or a2b"; exit 1 ;;
  *) echo "[kern-shim] unknown KERN '$KERN'"; exit 1 ;;
esac
export A2_CONFIG=$K/a2/a2_v.json
(
  flock 9
  for x in $P; do
    ap=$(ls $K/$x/apply_*.py | head -1)
    "$PY" "$ap" >&2 || { echo "[kern-shim] !! $ap failed"; exit 1; }
    "$PY" "$ap" --check >&2 || { echo "[kern-shim] !! $ap --check failed"; exit 1; }
    echo "[kern-shim] patch $x applied + checked"
  done
) 9>/tmp/kern_shim.lock || exit 1
echo "[kern-shim] KERN=$KERN ($P) ready -> $REAL"
exec bash "$REAL" "$@"
