#!/usr/bin/env bash
# `vllm_bin` replacement for the coder client (OUR arm only) = best_coder/sgl_shim.sh + option A step 2 "burst".
# The leader's llm/spawn.py runs
#   <vllm_bin> serve <model> --port P --api-key K --max-model-len L --tensor-parallel-size 1 --data-parallel-size 2
#     --gpu-memory-utilization U ... --revision R --tokenizer-revision R  <extra_args>
# with CUDA_VISIBLE_DEVICES = the client's gpu_ids. Our extra_args start with "--sgl", everything after it is passed to
# SGLang verbatim, except "--burst-glm <util>" (taken out here):
#   1. per coder GPU start a GLM judge FIRST (the leader's exact GLM flags + --enable-sleep-mode, VLLM_SERVER_DEV_MODE=1,
#      ports 8031..), wait until ready, one warm request, then /sleep?level=1 (weights to CPU, KV freed: ~1.9 GB left)
#   2. start SGLang next to the sleeping GLM (the SGLang args must include the memory saver flags), wait until ready
#   3. one warm release/resume per SGLang server (the first release copies the weights to CPU: ~22 s, paid here, at
#      startup, which is not timed) + one tiny request
#   4. lb_proxy on the coder port with --glm = the GLM servers: it does the swaps (/burst/to_judge, /burst/to_coder)
# Order matters: vLLM sizes its KV cache from free memory at start, so GLM first (util <burst-glm>), asleep, then SGLang.
set -u
D=$(cd "$(dirname "$0")" && pwd)
[ "${1:-}" = serve ] && shift
MODEL=$1; shift
PORT=8001; KEY=local; REV=""; LEN=81920; UTIL=0.9; SGL=()
while [ $# -gt 0 ]; do
  case $1 in
    --port) PORT=$2; shift 2 ;;
    --api-key) KEY=$2; shift 2 ;;
    --revision) REV=$2; shift 2 ;;
    --max-model-len) LEN=$2; shift 2 ;;
    --gpu-memory-utilization) UTIL=$2; shift 2 ;;
    --sgl) shift; SGL=("$@"); break ;;
    *) shift ;;
  esac
done
BURST=""; A2=(); MIXS=""; MPO=""
for ((j=0; j<${#SGL[@]}; j++)); do
  if [ "${SGL[$j]}" = --burst-glm ]; then BURST=${SGL[$((j+1))]}; j=$((j+1))
  elif [ "${SGL[$j]}" = --mix-slot ]; then MIXS=${SGL[$((j+1))]}; j=$((j+1))   # [mix] R71 "<slot>/<n>" (pgh_mix.py)
  elif [ "${SGL[$j]}" = --mix-port-off ]; then MPO=${SGL[$((j+1))]}; j=$((j+1))   # [phung3 gsplit+mix] (pgh_gsplit.py)
  else A2+=("${SGL[$j]}"); fi
done
SGL=("${A2[@]}")
GLM_VLLM=${GLM_VLLM:-/opt/vllm-glm-env/bin/vllm}; GLM_REV=411bb4d77144a3f03accbf4b780f5acb8b7cde4e
PY=${SGL_PYTHON:-/opt/sglang-env/bin/python}
LOGD=${SGL_LOG_DIR:-/sgl_logs}; mkdir -p "$LOGD"
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
# [mix] R71: two coder shims on one pod (pgh_mix.py). Unset (= cand): OFF 0, the same ports/logs/files as before.
# Set: GLM/SGLang ports and log names shift by slot x GPUs-per-slot; then a barrier: wait (<= PGH_MIX_BARRIER_S) until
# every other slot's shim has also finished its patch chain (kern/v4/v5/seed apply scripts rewrite files in the
# shared SGLang env, which must not happen while this slot's SGLang imports them) or is dead (pid file from spawn.py).
OFF=0; POSTF=/tmp/shim_post.out
if [ -n "$MIXS" ]; then
  MS=${MIXS%/*}; MN=${MIXS#*/}
  case "$MIXS" in *[!0-9/]* | /* | */ | */*/* | [!0-9]*) echo "[sgl-shim] !! bad --mix-slot '$MIXS'"; exit 1 ;; */*) ;; *) echo "[sgl-shim] !! bad --mix-slot '$MIXS'"; exit 1 ;; esac
  OFF=$((MS * ${#GPUS[@]})); POSTF=/tmp/shim_post_mix$MS.out
  # [phung3] PGH_GPU_SPLIT under the mix: slots have different GPU counts (utevo 1, ak 2) -> pgh_gsplit passes the port
  # offset (= first coder GPU index of this slot) so the ports stay contiguous (:8031..8033); absent = the line above
  if [ -n "$MPO" ]; then
    case "$MPO" in *[!0-9]*) echo "[sgl-shim] !! bad --mix-port-off '$MPO'"; exit 1 ;; esac
    OFF=$MPO
  fi
  MIXD=${PGH_MIX_DIR:-/tmp/pgh_mix}; mkdir -p "$MIXD"; : > "$MIXD/patched_$MS"
  tb=$(date +%s); lim=${PGH_MIX_BARRIER_S:-1800}; last=0
  while :; do
    wait_for=""
    for ((s=0; s<MN; s++)); do
      [ "$s" = "$MS" ] && continue
      [ -f "$MIXD/patched_$s" ] && continue
      pp=$(cat "$MIXD/pid_$s" 2>/dev/null)
      [ -n "$pp" ] && kill -0 "$pp" 2>/dev/null && wait_for="$wait_for $s"
    done
    [ -z "$wait_for" ] && { echo "[sgl-shim] mix slot $MS/$MN: barrier passed after $(( $(date +%s) - tb )) s (port offset $OFF)"; break; }
    el=$(( $(date +%s) - tb ))
    [ "$el" -ge "$lim" ] && { echo "[sgl-shim] !! mix slot $MS/$MN: slot(s)$wait_for still patching after $lim s -> starting anyway"; break; }
    [ $((el - last)) -ge 60 ] && { echo "[sgl-shim] mix slot $MS/$MN: waiting for slot(s)$wait_for to finish patching ($el s)"; last=$el; }
    sleep 2
  done
fi
# JIT caches on the mounted HF dir (bro1 Dockerfile), so a restart does not recompile
E=/root/.cache/huggingface/.engine
export FLASHINFER_CACHE_DIR=$E/flashinfer TORCHINDUCTOR_CACHE_DIR=$E/inductor TRITON_CACHE_DIR=$E/triton SGLANG_CACHE_DIR=$E/sglang
export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
PIDS=()
trap 'kill ${PIDS[*]} 2>/dev/null; sleep 5; kill -9 ${PIDS[*]} 2>/dev/null' EXIT INT TERM
H=(-H "Authorization: Bearer $KEY")
t0=$(date +%s)
wait_ready() {  # port pid name logfile limit_s
  until curl -sf "${H[@]}" "127.0.0.1:$1/v1/models" >/dev/null; do
    kill -0 "$2" 2>/dev/null || { echo "[sgl-shim] $3 :$1 died"; tail -40 "$4"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt "$5" ] && { echo "[sgl-shim] $3 :$1 not ready after $5 s"; tail -30 "$4"; exit 1; }
    sleep 5
  done
  echo "[sgl-shim] $3 :$1 ready after $(( $(date +%s) - t0 )) s"
}
post() {  # url [json] -> exits on non-200 (or no answer in 1800 s: a cold first request may JIT)
  local code; code=$(curl -s --max-time 1800 -o "$POSTF" -w '%{http_code}' -X POST "${H[@]}" -H 'Content-Type: application/json' \
    ${2:+-d "$2"} "$1"); [ "$code" = 200 ] || { echo "[sgl-shim] POST $1 -> $code $(head -c 300 "$POSTF")"; exit 1; }
}
answer() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' "$POSTF"; }
sgl_idle() {  # port: wait until SGLang reports 0 running + waiting on 2 reads 0.5 s apart, +1 s (its release asserts a
  # fully idle scheduler and kills the server otherwise; right after /v1/models answers, SGLang's own warm-up still runs)
  local calm=0 t=$(date +%s) n
  while [ $(( $(date +%s) - t )) -lt 300 ]; do
    n=$(curl -s --max-time 10 "${H[@]}" "127.0.0.1:$1/get_load" | "$PY" -c 'import json,sys; print(sum(int(x.get("num_reqs",0))+int(x.get("num_waiting_reqs",0)) for x in json.load(sys.stdin)))' 2>/dev/null)
    if [ "$n" = 0 ]; then calm=$((calm+1)); [ $calm -ge 2 ] && { sleep 1; return 0; }; else calm=0; fi
    sleep 0.5
  done
  echo "[sgl-shim] SGLang :$1 not idle after 300 s (get_load: ${n:-unreadable})"; exit 1
}
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' '; }
GLMS=""
if [ -n "$BURST" ]; then
  for i in "${!GPUS[@]}"; do
    p=$((8031 + OFF + i)); GLMS="$GLMS${GLMS:+,}http://127.0.0.1:$p"
    # = the leader's spawn.py GLM command (judge-critic-instance) + --enable-sleep-mode; only the memory share differs
    # (KV cache size). flashinfer JIT needs the venv bin (ninja) + nvcc on PATH.
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} VLLM_SERVER_DEV_MODE=1 PATH=$(dirname "$GLM_VLLM"):/usr/local/cuda/bin:$PATH \
      CUDA_HOME=/usr/local/cuda "$GLM_VLLM" serve zai-org/GLM-4.6V-Flash --host 127.0.0.1 --port $p --api-key "$KEY" \
      --max-model-len 81920 --tensor-parallel-size 1 --gpu-memory-utilization "$BURST" --max_num_seqs 64 \
      --generation-config vllm --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
      --revision $GLM_REV --tokenizer-revision $GLM_REV --trust-remote-code --max-num-batched-tokens 16384 \
      --enable-sleep-mode > "$LOGD/glm_burst$((OFF + i)).log" 2>&1 &
    PIDS+=($!); GLMP[$i]=$!
    echo "[sgl-shim] GPU ${GPUS[$i]} -> burst GLM judge :$p (pid $!, mem $BURST, sleep mode)"
  done
  for i in "${!GPUS[@]}"; do
    p=$((8031 + OFF + i)); wait_ready $p "${GLMP[$i]}" "burst GLM" "$LOGD/glm_burst$((OFF + i)).log" 3000
    # self-check (untimed): the same greedy request before and after a sleep/wake must match; with prefix caching on
    # the 2nd request would hit stale KV if the cache were not reset on sleep (offload step 1 ran without it)
    Q='{"model":"zai-org/GLM-4.6V-Flash","messages":[{"role":"user","content":"Count from 1 to 20, comma separated."}],"max_tokens":64,"temperature":0}'
    post "http://127.0.0.1:$p/v1/chat/completions" "$Q"; o1=$(answer) && [ -n "$o1" ] && [ "$o1" != None ] || { echo "[sgl-shim] burst GLM :$p no answer text"; exit 1; }
    post "http://127.0.0.1:$p/sleep?level=1"; post "http://127.0.0.1:$p/wake_up"
    post "http://127.0.0.1:$p/v1/chat/completions" "$Q"; o2=$(answer) && [ -n "$o2" ] || { echo "[sgl-shim] burst GLM :$p no answer text after wake"; exit 1; }
    [ "$o1" = "$o2" ] || { echo "[sgl-shim] burst GLM :$p output changed after sleep/wake:"; echo "$o1"; echo "$o2"; exit 1; }
    echo "[sgl-shim] burst GLM :$p sleep/wake self-check OK (same greedy output)"
    post "http://127.0.0.1:$p/sleep?level=1"
  done
  echo "[sgl-shim] burst GLMs asleep; GPU mem: $(mem)"
fi
# SGLang's memory saver preloads a hook that needs libcudart.so.12; if the system has none (a CUDA-13 host, offload
# try 1 rc 127), use the env's pip nvidia libs (as test_offload.py does). No-op where libcudart.so.12 is installed.
if [ -n "$BURST" ] && ! ldconfig -p 2>/dev/null | grep libcudart.so.12 >/dev/null; then
  SP=$(ls -d "$(dirname "$(dirname "$PY")")"/lib/python3*/site-packages/nvidia 2>/dev/null | head -1)
  [ -n "$SP" ] && export LD_LIBRARY_PATH=$SP/cuda_runtime/lib:$SP/cublas/lib:$SP/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  echo "[sgl-shim] no system libcudart.so.12 -> LD_LIBRARY_PATH from ${SP:-<none found>}"
fi
t0=$(date +%s)   # SGLang's start limit counts from here, not from the GLM bring-up
UPS=""
for i in "${!GPUS[@]}"; do
  p=$((30001 + OFF + i)); UPS="$UPS${UPS:+,}http://127.0.0.1:$p"
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" -m sglang.launch_server --model-path "$MODEL" ${REV:+--revision "$REV"} \
    --served-model-name "$MODEL" --host 127.0.0.1 --port $p --api-key "$KEY" --context-length "$LEN" --tp-size 1 \
    --mem-fraction-static "$UTIL" "${SGL[@]}" > "$LOGD/sgl_gpu$((OFF + i)).log" 2>&1 &
  PIDS+=($!); SGLP[$i]=$!
  echo "[sgl-shim] GPU ${GPUS[$i]} -> SGLang :$p (pid $!) $MODEL@${REV:-default} ${SGL[*]}"
done
for i in "${!GPUS[@]}"; do
  wait_ready $((30001 + OFF + i)) "${SGLP[$i]}" SGLang "$LOGD/sgl_gpu$((OFF + i)).log" 2400
done
if [ -n "$BURST" ]; then
  for i in "${!GPUS[@]}"; do
    p=$((30001 + OFF + i)); ts=$(date +%s)
    # 18/10: /v1/models answers BEFORE SGLang's own warm-up thread (GET /model_info, then a generate) has run; get_load
    # read 0 twice in that gap and the release then hit the running warm-up -> AssertionError, server dead (404final:v2
    # smoke). SGLang logs "fired up and ready to roll" only after its warm-up request finished: wait for it (<= 600 s).
    until grep -q "fired up and ready to roll" "$LOGD/sgl_gpu$((OFF + i)).log" 2>/dev/null; do
      kill -0 "${SGLP[$i]}" 2>/dev/null || { echo "[sgl-shim] SGLang :$p died"; tail -40 "$LOGD/sgl_gpu$((OFF + i)).log"; exit 1; }
      [ $(( $(date +%s) - ts )) -gt 600 ] && { echo "[sgl-shim] SGLang :$p no 'ready to roll' after 600 s, going on"; break; }
      sleep 1
    done
    post "http://127.0.0.1:$p/v1/chat/completions" \
      "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"temperature\":0}"
    sgl_idle $p
    post "http://127.0.0.1:$p/release_memory_occupation" '{}'
    post "http://127.0.0.1:$p/resume_memory_occupation" '{}'
    post "http://127.0.0.1:$p/v1/chat/completions" \
      "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"temperature\":0}"
    echo "[sgl-shim] SGLang :$p warm release/resume/request OK in $(( $(date +%s) - ts )) s"
  done
  echo "[sgl-shim] GPU mem (coder awake, GLMs asleep): $(mem)"
fi
"$PY" "$D/lb_proxy.py" "$PORT" "$UPS" --host 0.0.0.0 --api-key "$KEY" ${GLMS:+--glm "$GLMS"} &
PIDS+=($!)
echo "[sgl-shim] lb_proxy :$PORT -> $UPS${GLMS:+ (burst GLMs $GLMS)} (pid $!)"
# background + `wait -n` (not a foreground child) so a TERM from spawn.py runs the trap at once; if any server or the
# proxy exits, stop everything (the pipeline then sees the coder down instead of a half-working pair)
wait -n
echo "[sgl-shim] a child exited (rc $?) -> stopping all"
exit 1
