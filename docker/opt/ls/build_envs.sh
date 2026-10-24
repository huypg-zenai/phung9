#!/usr/bin/env bash
# build_envs.sh (pgh1 leader-style, /opt/ls/build_envs.sh) - called by run.sh at CONTAINER START, before any server
# starts. Builds what the old pgh1 image built at docker-build time (owner 19/10: "switch pgh1 to leader style with retries"):
#   sgl   : /opt/sglang-env from /opt/ls/sglang-env.lock.txt (/opt/ls/setup_sglang_env.sh: same pins as the tested env)
#   kern  : the GDN kernel patches of the configured KERN applied in place into it (/opt/kern MANIFEST md5-checked first;
#           their GPU --check runs again at every coder start in /opt/kern/kern_shim.sh)
#   glm   : /opt/vllm-glm-env exactly as the leader builds it (its scripts/setup_glm_vllm_env.sh, GLM-4.6V-Flash @411bb4d7)
#   sched : the judge encoder-cache scheduler patch into that env (/opt/jf/make_sched.py, md5 of input and output checked)
# sgl+kern and glm+sched run in parallel. Each step is idempotent: a done-marker inside the env skips it on a restart of the
# same container; the env dir is removed before every new build attempt. Network steps retry (/opt/ls/retry.sh).
# Exit 0 = everything built and checked; anything else = run.sh must not start the servers.
# KERN is read from configuration.yaml (vllm_bin: /opt/prefetch/shim_<KERN>.sh) = the single switch (generator variable).
# LS_TEST_NO_GPU=1 (tests only, never set in the image): the leader GLM script's last step needs a GPU
#   (torch.zeros(1).cuda()); then accept the env if `import vllm, fastapi, transformers` works.
# Usage: build_envs.sh [all|sgl|kern|glm|sched ...]   (default all)
set -uo pipefail
. /opt/ls/retry.sh
CFG=${CONFIG_FILE:-/workspace/configuration.yaml}
SGL=/opt/sglang-env; GLM=/opt/vllm-glm-env
SP=$GLM/lib/python3.11/site-packages/vllm/config/scheduler.py
SCHED_MD5=0f758a80715c6e042af17b0ae36c887e          # patched scheduler.py of the tested image 404final:v2
GLM_MODEL=zai-org/GLM-4.6V-Flash; GLM_REV=411bb4d77144a3f03accbf4b780f5acb8b7cde4e
KERN=$(grep -oE 'vllm_bin: "/opt/prefetch/shim_[a-z0-9]+\.sh"' "$CFG" | sed -E 's/.*shim_([a-z0-9]+)\.sh"/\1/' | sort -u)
case $KERN in
  a1a2) PATS="a1g a2" ;; b) PATS="braw" ;; a2b) PATS="a2 braw" ;;
  *) ls_log "!! KERN '$KERN' from $CFG unknown (a1a2b refused: broken output)"; exit 1 ;;
esac
STEPS=${*:-all}; want() { [[ " $STEPS " == *" all "* || " $STEPS " == *" $1 "* ]]; }

step_sgl() {
  [ -f $SGL/.404_env_ok ] && { ls_log "sgl: env present (marker) - skip"; return 0; }
  rm -rf $SGL
  LOCK_FILE=/opt/ls/sglang-env.lock.txt bash /opt/ls/setup_sglang_env.sh || { ls_log "!! sgl: setup failed"; return 1; }
  $SGL/bin/python -c "import sglang, httpx, starlette, uvicorn; from sglang.srt.models.dflash import DFlash2DraftModel; print('[404-env] sglang env', sglang.__version__)" \
    || { ls_log "!! sgl: import check failed"; return 1; }
  touch $SGL/.404_env_ok
}
step_kern() {
  [ -f $SGL/.404_kern_$KERN ] && { ls_log "kern: $KERN already applied (marker) - skip"; return 0; }
  (cd /opt/kern && md5sum --quiet -c MANIFEST) || { ls_log "!! kern: /opt/kern differs from MANIFEST"; return 1; }
  local x ap
  for x in $PATS; do
    ap=$(ls /opt/kern/$x/apply_*.py | head -1)
    A2_CONFIG=/opt/kern/a2/a2_v.json $SGL/bin/python "$ap" || { ls_log "!! kern: $ap failed"; return 1; }
  done
  touch $SGL/.404_kern_$KERN; ls_log "kern: KERN=$KERN ($PATS) applied into $SGL (GPU --check runs at coder start)"
}
glm_clean() { rm -rf $GLM; }
glm_once() {
  rm -rf $GLM
  MODEL=$GLM_MODEL MODEL_REVISION=$GLM_REV bash /workspace/scripts/setup_glm_vllm_env.sh && return 0
  if [ "${LS_TEST_NO_GPU:-0}" = 1 ] && $GLM/bin/python -c "import vllm, fastapi, transformers; print('[404-env] glm env', vllm.__version__)"; then
    ls_log "glm: TEST MODE (no GPU): leader script's GPU step failed, install itself OK"; return 0; fi
  return 1
}
step_glm() {
  [ -f $GLM/.404_env_ok ] && { ls_log "glm: env present (marker) - skip"; return 0; }
  ls_retry "glm env (leader setup_glm_vllm_env.sh)" glm_clean glm_once || return 1
  [ -x $GLM/bin/vllm ] || { ls_log "!! glm: $GLM/bin/vllm missing"; return 1; }
  touch $GLM/.404_env_ok
}
step_sched() {
  [ -f $GLM/.404_sched_ok ] && { ls_log "sched: already patched (marker) - skip"; return 0; }
  (cd /opt/jf && md5sum --quiet -c JF_MD5) || { ls_log "!! sched: /opt/jf differs from JF_MD5"; return 1; }
  python3 /opt/jf/make_sched.py $SP /tmp/404_scheduler.py && cp /tmp/404_scheduler.py $SP && rm -f /tmp/404_scheduler.py \
    || { ls_log "!! sched: make_sched failed"; return 1; }
  [ "$(md5sum < $SP | cut -d' ' -f1)" = $SCHED_MD5 ] || { ls_log "!! sched: $SP md5 != $SCHED_MD5"; return 1; }
  touch $GLM/.404_sched_ok; ls_log "sched: $SP patched (md5 $SCHED_MD5)"
}

T0=$(date +%s)
ls_log "build_envs: steps '$STEPS', KERN=$KERN"
chain_sgl() { if want sgl; then step_sgl || return 1; fi; if want kern; then step_kern || return 1; fi; return 0; }
chain_glm() { if want glm; then step_glm || return 1; fi; if want sched; then step_sched || return 1; fi; return 0; }
chain_sgl > >(sed -u 's/^/[sgl] /') 2>&1 & P1=$!
chain_glm > >(sed -u 's/^/[glm] /') 2>&1 & P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
sleep 1   # let the sed prefixers flush
if [ $R1 = 0 ] && [ $R2 = 0 ]; then
  ls_log "build_envs: ALL OK in $(( $(date +%s) - T0 )) s"; exit 0
fi
ls_log "!! build_envs: FAILED (sgl+kern rc=$R1, glm+sched rc=$R2) after $(( $(date +%s) - T0 )) s"
exit 1
