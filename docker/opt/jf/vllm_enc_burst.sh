#!/usr/bin/env bash
# JUDGE_FAST_BURST_ENC=1: GLM_VLLM for the burst GLMs started by burst/sgl_shim.sh. That shim puts $(dirname $GLM_VLLM)
# first on PATH (flashinfer JIT needs the venv's ninja), so restore the venv bin first, then exec the real vllm.
export PATH=/opt/vllm-glm-env/bin:$PATH
export VLLM_ENC_CACHE_TOKENS=${JF_ENC_TOKENS:-1500000}
echo "[judgefast] burst enc: VLLM_ENC_CACHE_TOKENS=$VLLM_ENC_CACHE_TOKENS -> /opt/vllm-glm-env/bin/vllm $*" >&2
exec /opt/vllm-glm-env/bin/vllm "$@"
