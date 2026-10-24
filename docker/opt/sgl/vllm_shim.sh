#!/usr/bin/env bash
# Option C (owner 17/10): `vllm_bin` replacement for the coder client of the LEADER's own pipeline = the leader's coder
# (vLLM 0.24 + MTP, its exact command) + the burst (2 sleeping GLM judges on the coder GPUs, swapped in once drafting is
# done). No SGLang. The leader's llm/spawn.py runs
#   <vllm_bin> serve <model> --port P --api-key K --max-model-len L --tensor-parallel-size 1 --data-parallel-size 2
#     --gpu-memory-utilization U ... --revision R --tokenizer-revision R  <extra_args>
# with CUDA_VISIBLE_DEVICES = the client's gpu_ids. Our extra_args = the leader's MTP args + "--burst-glm <util>"
# [+ "--burst-layout dp|split"] (both taken out here); everything else goes to the real vLLM verbatim, plus
# --enable-sleep-mode (VLLM_SERVER_DEV_MODE=1 for /sleep, /wake_up, /is_sleeping):
#   1. per coder GPU start a GLM judge FIRST (= sgl_shim.sh: the leader's GLM flags + sleep mode, ports 8031..), wait,
#      greedy self-check across a sleep/wake, then /sleep?level=1 (weights to CPU, KV freed: ~1.9 GB left)
#   2. start the coder next to the sleeping GLMs:
#        dp    (default) = the leader's ONE server, DP2 inside vLLM, on :30001 (its sleep must reach both engines)
#        split = one DP1 server per GPU on :30001+i (fallback if DP2 sleep misbehaves; same model, flags, sampling)
#   3. per server (untimed, startup): warm request, greedy self-check across a sleep/wake (equal, else at least sane)
#   4. lb_proxy --engine vllm on the coder port with --glm = the GLM servers: it does the swaps
# Without --burst-glm this is a pure pass-through: exec the real vLLM with the leader's args unchanged.
set -u
D=$(cd "$(dirname "$0")" && pwd)
CODER_VLLM=${CODER_VLLM:-/opt/vllm-coder-env/bin/vllm}
ORIG=("$@")
[ "${1:-}" = serve ] && shift
MODEL=$1; shift
PORT=8001; KEY=local; BURST=""; LAYOUT=dp; DP=1; REST=()
while [ $# -gt 0 ]; do
  case $1 in
    --port) PORT=$2; shift 2 ;;
    --api-key) KEY=$2; REST+=("$1" "$2"); shift 2 ;;
    --data-parallel-size) DP=$2; REST+=("$1" "$2"); shift 2 ;;
    --burst-glm) BURST=$2; shift 2 ;;
    --burst-layout) LAYOUT=$2; shift 2 ;;
    *) REST+=("$1"); shift ;;
  esac
done
if [ -z "$BURST" ]; then
  exec "$CODER_VLLM" "${ORIG[@]}"
fi
GLM_VLLM=${GLM_VLLM:-/opt/vllm-glm-env/bin/vllm}; GLM_REV=411bb4d77144a3f03accbf4b780f5acb8b7cde4e
PY=${SHIM_PYTHON:-$(dirname "$CODER_VLLM")/python}
LOGD=${SGL_LOG_DIR:-/sgl_logs}; mkdir -p "$LOGD"
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
PIDS=()
trap 'kill ${PIDS[*]} 2>/dev/null; sleep 5; kill -9 ${PIDS[*]} 2>/dev/null' EXIT INT TERM
H=(-H "Authorization: Bearer $KEY")
OUTF=${TMPDIR:-/tmp}/vshim_post.$$
t0=$(date +%s)
wait_ready() {  # port pid name logfile limit_s
  until curl -sf "${H[@]}" "127.0.0.1:$1/v1/models" >/dev/null; do
    kill -0 "$2" 2>/dev/null || { echo "[vllm-shim] $3 :$1 died"; tail -40 "$4"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt "$5" ] && { echo "[vllm-shim] $3 :$1 not ready after $5 s"; tail -30 "$4"; exit 1; }
    sleep 5
  done
  echo "[vllm-shim] $3 :$1 ready after $(( $(date +%s) - t0 )) s"
}
post() {  # url [json] -> exits on non-200 (or no answer in 1800 s: a cold first request may JIT)
  local code; code=$(curl -s --max-time 1800 -o $OUTF -w '%{http_code}' -X POST "${H[@]}" -H 'Content-Type: application/json' \
    ${2:+-d "$2"} "$1"); [ "$code" = 200 ] || { echo "[vllm-shim] POST $1 -> $code $(head -c 300 $OUTF)"; exit 1; }
}
answer() { "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["choices"][0]["message"]["content"])' "$OUTF"; }
vllm_idle() {  # port: running + waiting == 0 on 2 reads 0.5 s apart (+1 s); never sleep a server under requests
  local calm=0 t=$(date +%s) n
  while [ $(( $(date +%s) - t )) -lt 300 ]; do
    n=$(curl -s --max-time 10 "127.0.0.1:$1/metrics" | awk '/^vllm:num_requests_(running|waiting)/{s+=$NF} END{if (NR) printf "%d", s}')
    if [ "$n" = 0 ]; then calm=$((calm+1)); [ $calm -ge 2 ] && { sleep 1; return 0; }; else calm=0; fi
    sleep 0.5
  done
  echo "[vllm-shim] vLLM :$1 not idle after 300 s (running+waiting: ${n:-unreadable})"; exit 1
}
mem() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' '; }
GLMS=""
for i in "${!GPUS[@]}"; do
  p=$((8031 + i)); GLMS="$GLMS${GLMS:+,}http://127.0.0.1:$p"
  # = the leader's spawn.py GLM command (judge-critic-instance) + --enable-sleep-mode; only the memory share differs
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} VLLM_SERVER_DEV_MODE=1 PATH=$(dirname "$GLM_VLLM"):/usr/local/cuda/bin:$PATH \
    CUDA_HOME=/usr/local/cuda "$GLM_VLLM" serve zai-org/GLM-4.6V-Flash --host 127.0.0.1 --port $p --api-key "$KEY" \
    --max-model-len 81920 --tensor-parallel-size 1 --gpu-memory-utilization "$BURST" --max_num_seqs 64 \
    --generation-config vllm --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192 \
    --revision $GLM_REV --tokenizer-revision $GLM_REV --trust-remote-code --max-num-batched-tokens 16384 \
    --enable-sleep-mode > "$LOGD/glm_burst$i.log" 2>&1 &
  PIDS+=($!); GLMP[$i]=$!
  echo "[vllm-shim] GPU ${GPUS[$i]} -> burst GLM judge :$p (pid $!, mem $BURST, sleep mode)"
done
Q='{"model":"zai-org/GLM-4.6V-Flash","messages":[{"role":"user","content":"Count from 1 to 20, comma separated."}],"max_tokens":64,"temperature":0}'
for i in "${!GPUS[@]}"; do
  p=$((8031 + i)); wait_ready $p "${GLMP[$i]}" "burst GLM" "$LOGD/glm_burst$i.log" 3000
  post "http://127.0.0.1:$p/v1/chat/completions" "$Q"; o1=$(answer) && [ -n "$o1" ] && [ "$o1" != None ] || { echo "[vllm-shim] burst GLM :$p no answer text"; exit 1; }
  post "http://127.0.0.1:$p/sleep?level=1"; post "http://127.0.0.1:$p/wake_up"
  post "http://127.0.0.1:$p/v1/chat/completions" "$Q"; o2=$(answer) && [ -n "$o2" ] || { echo "[vllm-shim] burst GLM :$p no answer text after wake"; exit 1; }
  [ "$o1" = "$o2" ] || { echo "[vllm-shim] burst GLM :$p output changed after sleep/wake:"; echo "$o1"; echo "$o2"; exit 1; }
  echo "[vllm-shim] burst GLM :$p sleep/wake self-check OK (same greedy output)"
  post "http://127.0.0.1:$p/sleep?level=1"
done
echo "[vllm-shim] burst GLMs asleep; GPU mem: $(mem)"
t0=$(date +%s)   # the coder's start limit counts from here, not from the GLM bring-up
UPS=""; CP=()
if [ "$LAYOUT" = split ]; then   # one DP1 server per GPU (drop --data-parallel-size)
  ARGS=(); skip=0
  for x in "${REST[@]}"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    [ "$x" = --data-parallel-size ] && { skip=1; continue; }
    ARGS+=("$x")
  done
  for i in "${!GPUS[@]}"; do
    p=$((30001 + i)); UPS="$UPS${UPS:+,}http://127.0.0.1:$p"
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} VLLM_SERVER_DEV_MODE=1 "$CODER_VLLM" serve "$MODEL" --host 127.0.0.1 --port $p \
      "${ARGS[@]}" --enable-sleep-mode > "$LOGD/coder_gpu$i.log" 2>&1 &
    PIDS+=($!); CP+=("$p:$!:$LOGD/coder_gpu$i.log")
    echo "[vllm-shim] GPU ${GPUS[$i]} -> coder vLLM :$p (pid $!, DP1, split layout)"
  done
else                              # the leader's own server: all GPUs, DP inside vLLM
  p=30001; UPS="http://127.0.0.1:$p"
  VLLM_SERVER_DEV_MODE=1 "$CODER_VLLM" serve "$MODEL" --host 127.0.0.1 --port $p "${REST[@]}" --enable-sleep-mode \
    > "$LOGD/coder.log" 2>&1 &
  PIDS+=($!); CP+=("$p:$!:$LOGD/coder.log")
  echo "[vllm-shim] GPUs ${CUDA_VISIBLE_DEVICES:-0} -> coder vLLM :$p (pid $!, DP $DP, the leader's server)"
fi
for c in "${CP[@]}"; do IFS=: read -r p pid lf <<< "$c"; wait_ready "$p" "$pid" "coder vLLM" "$lf" 2400; done
CQ="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 20, comma separated. Answer only.\"}],\"max_tokens\":96,\"temperature\":0}"
for c in "${CP[@]}"; do
  IFS=: read -r p pid lf <<< "$c"; ts=$(date +%s)
  post "http://127.0.0.1:$p/v1/chat/completions" "$CQ"            # cold path (JIT/graphs), not compared
  post "http://127.0.0.1:$p/v1/chat/completions" "$CQ"; o1=$(answer)
  vllm_idle $p
  post "http://127.0.0.1:$p/sleep?level=1"
  s=$(curl -s "${H[@]}" "127.0.0.1:$p/is_sleeping"); echo "[vllm-shim] coder :$p asleep: $s; GPU mem: $(mem)"
  post "http://127.0.0.1:$p/wake_up"
  post "http://127.0.0.1:$p/v1/chat/completions" "$CQ"; o2=$(answer)
  if [ "$o1" = "$o2" ]; then echo "[vllm-shim] coder :$p sleep/wake self-check OK (same greedy output) in $(( $(date +%s) - ts )) s"
  elif grep -q "1, 2, 3, 4, 5" <<< "$o2"; then echo "[vllm-shim] WARNING coder :$p greedy output differs after sleep/wake but is sane:"; echo "$o1"; echo "$o2"
  else echo "[vllm-shim] coder :$p output broken after sleep/wake:"; echo "$o1"; echo "$o2"; exit 1; fi
done
echo "[vllm-shim] GPU mem (coder awake, GLMs asleep): $(mem)"
"$PY" "$D/lb_proxy.py" "$PORT" "$UPS" --host 0.0.0.0 --api-key "$KEY" --glm "$GLMS" --engine vllm &
PIDS+=($!)
echo "[vllm-shim] lb_proxy :$PORT -> $UPS (burst GLMs $GLMS, engine vllm) (pid $!)"
# background + `wait -n` so a TERM from spawn.py runs the trap at once; if any server or the proxy exits, stop everything
wait -n
echo "[vllm-shim] a child exited (rc $?) -> stopping all"
exit 1
