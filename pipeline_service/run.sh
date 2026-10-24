#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${CONFIG_PATH:-/workspace/configuration.yaml}"
export CONFIG_FILE

# Raise the open-file limit: the service runs hundreds of threads + Chromium sidecars; a 1024 soft limit
# breaks the validator ("Too many open files") and resets /status connections under load.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || ulimit -n 65536 2>/dev/null || true

# [sws] kernel-optim sws-llvm SwiftShader run-time switch (docker/opt/sws/sws_mode.sh; PGH_SWS / PGH_SWS_THREADS), before
# anything starts Chrome. Never fails the start; one "[sws]" line per Chrome dir. No file = image without sws (no-op).
if [ -f /opt/sws/sws_mode.sh ]; then bash /opt/sws/sws_mode.sh || true; fi


# Thread pools: the GLM vLLM's CPU-side image preprocessing spawns nproc-sized OpenMP pools; on a small-quota
# host (10 CPUs seen on RunPod) that oversubscription made the judge's cold path 1.9x slower, while OMP=1 costs
# nothing on 20-40 CPU quotas. Inherited by every vLLM server spawned from llm/spawn.py.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Preflight: network, GPU count, per-GPU bf16 TFLOPS + decode-shaped weight-stream GB/s + power limit.
# Writes /tmp/preflight_metrics.json (picked up by serve.py for the miner-diag header).
if python -m modules.metrics.preflight; then
    echo "=== PRE-FLIGHT OK ==="
else
    echo "=== PRE-FLIGHT FAILED — starting FastAPI in REPLACE mode ==="
fi


# FastAPI
echo "=== STAGE 2: FastAPI ==="
python serve.py &
SERVE_PID=$!


# 404 (leader-style pgh1): build the SGLang coder env + GDN kernel patches and the GLM vLLM judge env + scheduler patch
# here (network steps retried 5x, each step skipped when its done-marker exists). If anything still fails, no server is
# started: FastAPI restarts in REPLACE mode (/status=replace -> the validator drops this pod: warm-up fails loudly).
echo "=== STAGE 2.4: 404 env build (/opt/ls/build_envs.sh) ==="
if ! bash /opt/ls/build_envs.sh; then
    echo "=== 404 ENV BUILD FAILED - no servers started; FastAPI restarted in REPLACE mode ===" >&2
    touch /tmp/pod_replace
    kill $SERVE_PID 2>/dev/null || true; wait $SERVE_PID 2>/dev/null || true
    python serve.py
    exit 1
fi


# Dedicated GLM vLLM env (GLM-4.6V needs a newer transformers than the coder).
# Idempotent: only builds when the binary is missing. Non-fatal: on failure the
# judge/critic client is skipped and the coder still comes up.
GLM_VLLM_BIN="${GLM_VLLM_BIN:-/opt/vllm-glm-env/bin/vllm}"
if [ ! -x "$GLM_VLLM_BIN" ]; then
    echo "=== STAGE 2.5: building GLM vLLM env ($GLM_VLLM_BIN missing) ==="
    GLM_MODEL_INFO="$(GLM_VLLM_BIN="$GLM_VLLM_BIN" python - <<'PY'
import os, yaml
try:
    cfg = yaml.safe_load(open(os.environ["CONFIG_FILE"])) or {}
except Exception:
    raise SystemExit(0)
for spec in (cfg.get("llm_clients") or {}).values():
    if not isinstance(spec, dict):
        continue
    v = spec.get("vllm") or {}
    if str(v.get("vllm_bin") or "") == os.environ["GLM_VLLM_BIN"] and str(v.get("model") or "").strip():
        print(str(v["model"]).strip())
        print(str(v.get("revision") or "").strip())
        break
PY
)"
    GLM_MODEL="$(sed -n 1p <<<"$GLM_MODEL_INFO")"
    GLM_REVISION="$(sed -n 2p <<<"$GLM_MODEL_INFO")"
    [ -n "$GLM_MODEL" ] && echo "[run.sh] GLM env build target: $GLM_MODEL @ ${GLM_REVISION:-main}"
    MODEL="$GLM_MODEL" MODEL_REVISION="$GLM_REVISION" \
        bash "$SCRIPT_DIR/scripts/setup_glm_vllm_env.sh" \
        || echo "[run.sh] GLM env setup failed — judge/critic vLLM skipped, coder continues" >&2
else
    echo "[run.sh] GLM vLLM env present at $GLM_VLLM_BIN — skipping build"
fi


# Dedicated coder vLLM env. Same idea as the GLM env above: config-driven and
# idempotent, so it only builds when configuration.yaml points a client at
# $CODER_VLLM_BIN and the binary is missing. Non-fatal: failures are logged.
CODER_VLLM_BIN="${CODER_VLLM_BIN:-/opt/vllm-coder-env/bin/vllm}"
CODER_VLLM_VERSION="${CODER_VLLM_VERSION:-0.24.0}"
if [ ! -x "$CODER_VLLM_BIN" ]; then
    CODER_MODEL_INFO="$(GLM_VLLM_BIN="$CODER_VLLM_BIN" python - <<'PY'
import os, yaml
try:
    cfg = yaml.safe_load(open(os.environ["CONFIG_FILE"])) or {}
except Exception:
    raise SystemExit(0)
for spec in (cfg.get("llm_clients") or {}).values():
    if not isinstance(spec, dict):
        continue
    v = spec.get("vllm") or {}
    if str(v.get("vllm_bin") or "") == os.environ["GLM_VLLM_BIN"] and str(v.get("model") or "").strip():
        print(str(v["model"]).strip())
        print(str(v.get("revision") or "").strip())
        break
PY
)"
    CODER_MODEL="$(sed -n 1p <<<"$CODER_MODEL_INFO")"
    CODER_REVISION="$(sed -n 2p <<<"$CODER_MODEL_INFO")"
    if [ -n "$CODER_MODEL" ]; then
        echo "=== STAGE 2.6: building coder vLLM env $CODER_VLLM_VERSION ($CODER_VLLM_BIN missing) ==="
        echo "[run.sh] coder env build target: $CODER_MODEL @ ${CODER_REVISION:-main}"
        VENV="$(dirname "$(dirname "$CODER_VLLM_BIN")")" \
        VLLM_VERSION="$CODER_VLLM_VERSION" \
        MODEL="$CODER_MODEL" MODEL_REVISION="$CODER_REVISION" \
            bash "$SCRIPT_DIR/scripts/setup_glm_vllm_env.sh" \
            || echo "[run.sh] coder env setup failed — the coder cannot start without $CODER_VLLM_BIN; check logs" >&2
    fi
else
    echo "[run.sh] coder vLLM env present at $CODER_VLLM_BIN — skipping build"
fi


# vLLM spawn

echo "=== STAGE 3: vLLM spawn ==="
python -m llm.spawn || echo "[run.sh] vllm spawn returned non-zero — FastAPI continues for diagnostics" >&2

wait $SERVE_PID
