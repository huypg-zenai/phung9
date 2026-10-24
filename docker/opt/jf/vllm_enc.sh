#!/usr/bin/env bash
# JUDGE_FAST "enc" vllm_bin for the leader-launched GLM judges (llm/spawn.py runs <vllm_bin> serve ...; env untouched).
# Sets the encoder cache budget read by the patched vllm/config/scheduler.py (mounted by run_arm2.sh JUDGE_FAST=1);
# with the variable unset that file behaves exactly like the original (budget = max_num_batched_tokens).
export VLLM_ENC_CACHE_TOKENS=${JF_ENC_TOKENS:-1500000}
echo "[judgefast] enc: VLLM_ENC_CACHE_TOKENS=$VLLM_ENC_CACHE_TOKENS -> /opt/vllm-glm-env/bin/vllm $*" >&2
exec /opt/vllm-glm-env/bin/vllm "$@"
