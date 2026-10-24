#!/usr/bin/env bash
# /opt/v3/coder_switch.sh <spawn.py's vllm_bin args: serve <model> ... --sgl <SGLang args...>>   (pgh1v3, 20/10)
# The pgh1v3 image sets ENV KERN_REAL_SHIM=/opt/v3/coder_switch.sh, so the coder start chain is
#   vllm_bin /opt/prefetch/shim_a2b.sh -> prefetch_shim.sh (unchanged v2) -> /opt/kern/shim_a2b.sh -> kern_shim.sh
#   (unchanged v2: GDN patches, then exec $KERN_REAL_SHIM) -> THIS -> /opt/sgl/sgl_shim.sh "$@" [+ extra SGLang args]
# Switches (plain env, read here at every coder start; set with docker run -e or the Dockerfile ENV defaults):
#   SGLANG_DFLASH_SEEDED_SAMPLING=1  apply dflash_seed.patch (/opt/v3/seed/apply_seed_shim.sh --apply-only: idempotent,
#                                    all-or-nothing, import probe) and append '--sampling-backend pytorch' (the flashinfer
#                                    sampler asserts sampling_seed is None; = G2 arms b and c).
#   PGH_DETERMINISTIC_INFERENCE=1    append '--enable-deterministic-inference' (independent of the seed switch).
#   PGH_SEED_STRICT=1                a refused seed patch stops the coder (exit 1) instead of the fallback below.
# With SGLANG_DFLASH_SEEDED_SAMPLING=0 and PGH_DETERMINISTIC_INFERENCE=0 this is just `exec bash /opt/sgl/sgl_shim.sh "$@"`
# = the pgh1v2 chain (kern_shim's default KERN_REAL_SHIM), same args, same env.
# Seed patch refused (SGLang drift, half-applied tree, failed import probe) -> default = FALLBACK: one '!!' line, the
# patch is reverted when possible (else left dormant with SGLANG_DFLASH_SEEDED_SAMPLING=0 = stock behaviour), the
# pytorch-sampler arg is NOT added, marker /tmp/v3_seed_refused, and SGLang starts exactly as in pgh1v2.
# Why fallback, not fail-fast, for a contest run: a coder that does not start = the pod goes to REPLACE and the round
# gets no outputs at all (the replacement pods run the same image and would refuse the same way), while the fallback is
# the tested pgh1v2 behaviour whose only cost is today's audit re-run variance. GPU tests use PGH_SEED_STRICT=1.
# Test hooks: V3_REAL_SHIM (default /opt/sgl/sgl_shim.sh), V3_SEED_SHIM (default /opt/v3/seed/apply_seed_shim.sh),
# V3_MARKER (default /tmp/v3_seed_refused).
set -u
REAL=${V3_REAL_SHIM:-/opt/sgl/sgl_shim.sh}
SEEDSH=${V3_SEED_SHIM:-/opt/v3/seed/apply_seed_shim.sh}
T="[v3-switch]"
b01() {  # name value -> 0/1; anything else = '!!' + 0 (off = the pgh1v2 behaviour)
  case $2 in 0 | 1) echo "$2" ;; *) echo "$T !! $1='$2' is not 0 or 1 -> treated as 0 (off)" >&2; echo 0 ;; esac
}
SEED=$(b01 SGLANG_DFLASH_SEEDED_SAMPLING "${SGLANG_DFLASH_SEEDED_SAMPLING:-0}")
DET=$(b01 PGH_DETERMINISTIC_INFERENCE "${PGH_DETERMINISTIC_INFERENCE:-0}")
STRICT=$(b01 PGH_SEED_STRICT "${PGH_SEED_STRICT:-0}")
echo "$T SGLANG_DFLASH_SEEDED_SAMPLING=$SEED PGH_DETERMINISTIC_INFERENCE=$DET PGH_SEED_STRICT=$STRICT" \
     "(pipeline side: PGH_ADAPTIVE_K=${PGH_ADAPTIVE_K:-unset} PGH_ADAPTIVE_K_COEFFS='${PGH_ADAPTIVE_K_COEFFS:-}')"

if [ "$SEED" = 0 ] && [ "$DET" = 0 ]; then
  export SGLANG_DFLASH_SEEDED_SAMPLING=0
  exec bash "$REAL" "$@"
fi

# Extra SGLang args go at the end, i.e. after --sgl (spawn.py puts extra_args last; sgl_shim passes everything after
# --sgl to SGLang verbatim). Without --sgl they would be dropped silently -> refuse the switches instead.
has_sgl=0; sb=""; has_det=0; a=("$@")
for ((i = 0; i < ${#a[@]}; i++)); do
  case ${a[$i]} in
    --sgl) has_sgl=1 ;;
    --sampling-backend) sb=${a[$((i + 1))]:-} ;;
    --sampling-backend=*) sb=${a[$i]#*=} ;;
    --enable-deterministic-inference) has_det=1 ;;
  esac
done
if [ $has_sgl = 0 ]; then
  echo "$T !! no --sgl in the coder args: cannot pass SGLang flags -> seed + deterministic switches IGNORED (pgh1v2 start)"
  [ "$STRICT" = 1 ] && exit 1
  export SGLANG_DFLASH_SEEDED_SAMPLING=0
  exec bash "$REAL" "$@"
fi
DETX=(); [ "$DET" = 1 ] && [ $has_det = 0 ] && DETX=(--enable-deterministic-inference)

if [ "$SEED" = 1 ]; then
  SAMPX=(); why=""
  case $sb in
    "") SAMPX=(--sampling-backend pytorch) ;;
    pytorch) ;;
    *) why="the args already set --sampling-backend $sb (seeded sampling needs pytorch)" ;;
  esac
  if [ -z "$why" ]; then
    if SGLANG_DFLASH_SEEDED_SAMPLING=1 bash "$SEEDSH" --apply-only; then
      export SGLANG_DFLASH_SEEDED_SAMPLING=1
      echo "$T seed patch ON; extra SGLang args: ${SAMPX[*]} ${DETX[*]}"
      exec bash "$REAL" "$@" "${SAMPX[@]}" "${DETX[@]}"
    fi
    why="apply_seed_shim.sh refused (see its '!!' line above)"
  fi
  echo "$T !! SEED PATCH NOT APPLIED: $why -> coder starts UNSEEDED (= pgh1v2 drafts, NOT reproducible)"
  : > "${V3_MARKER:-/tmp/v3_seed_refused}" 2>/dev/null || true
  [ "$STRICT" = 1 ] && { echo "$T !! PGH_SEED_STRICT=1 -> coder start aborted"; exit 1; }
  if [ -z "${why##apply_seed_shim*}" ]; then
    # The shim checks every hunk before writing, so a refusal (drift) wrote nothing; only a failure AFTER the write
    # (py_compile / import probe) leaves a patched tree, which --revert then undoes.
    rv=$(bash "$SEEDSH" --revert 2>&1); rrc=$?
    case "$rrc:$rv" in
      0:*"already reverted"*) echo "$T SGLang tree is stock (the shim wrote nothing)" ;;
      0:*) echo "$T the patch had been written before the failure -> reverted to stock" ;;
      *"half-applied"*) echo "$T !! half-applied tree, revert refused; SGLANG_DFLASH_SEEDED_SAMPLING=0 keeps the patch code dormant" ;;
      *) echo "$T SGLang tree left as found (the shim wrote nothing: its target lines do not match)" ;;
    esac
  fi
fi
export SGLANG_DFLASH_SEEDED_SAMPLING=0
echo "$T extra SGLang args: ${DETX[*]:-none}"
exec bash "$REAL" "$@" "${DETX[@]}"
