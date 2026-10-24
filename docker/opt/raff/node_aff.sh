#!/bin/sh
# node_aff.sh - renderer.node_binary wrapper (RENDER_AFF=1, run_arm4.sh; ORTHO_REVIEW H1 fix (b)).
# Confines ONE render sidecar (node render_runner.mjs) and its Chrome/SwiftShader children (they inherit the mask) to
# the CPU list from aff_list.py (R = floor(cpu quota) - 14 CPUs). exec keeps the PID, so the renderer's process-group
# kill still works. Fails closed: any problem -> log to stderr (the renderer logs sidecar stderr) and exec plain node.
D=$(dirname "$0")
L=$(python3 -I "$D/aff_list.py" 2>/tmp/node_aff.$$.err)
if [ -n "$L" ] && command -v taskset >/dev/null 2>&1 && taskset -c "$L" true 2>>/tmp/node_aff.$$.err; then
  echo "[node_aff] pid $$ taskset -c $L ($(tail -n 1 /tmp/node_aff.$$.err 2>/dev/null))" >&2
  exec taskset -c "$L" node "$@"
fi
echo "[node_aff] pid $$ FALLBACK plain node: $(tail -n 2 /tmp/node_aff.$$.err 2>/dev/null | tr '\n' ' ')" >&2
exec node "$@"
