#!/usr/bin/env bash
# prefetch_shim.sh (leader-style pgh1: = kern/prefetch_shim.sh with 5 tries, waits 10/30/60/120 s) <KERN> serve <model> <spawn.py args...>   (pgh1 submission only; /opt/prefetch/prefetch_shim.sh)
# The coder shims start SGLang and the burst GLMs with HF_HUB_OFFLINE=1 (on our box the HF cache is pre-filled). In a
# fresh contest container the cache is empty, so this wrapper first downloads, at their pinned revisions: the coder
# (<model> --revision), the drafter (--speculative-draft-model-path / -revision; private repo -> needs HF_TOKEN in the
# environment) and the burst GLM (zai-org/GLM-4.6V-Flash@411bb4d7, = burst/sgl_shim.sh GLM_REV), 5 tries each; then it
# execs the tested chain /opt/kern/shim_<KERN>.sh -> /opt/sgl/sgl_shim.sh with the same arguments.
# A download by commit hash leaves no refs/main; SGLang (offline) loads the DFLASH draft config WITHOUT the revision
# (server_args handle_speculative_decoding) -> "couldn't connect" crash (start test 19/10 01:44 VN). So write refs/main =
# the pinned hash when it is missing (never overwrite an existing one).
# PREFETCH_DRY=1: print what would be fetched and stop (test).
set -u
K=$1; shift
a=("$@"); MODEL=""; REV=""; DM=""; DR=""
[ "${a[0]:-}" = serve ] && MODEL=${a[1]:-}
for ((i = 0; i < ${#a[@]}; i++)); do
  case ${a[$i]} in
    --revision) REV=${a[$((i + 1))]:-} ;;
    --speculative-draft-model-path) DM=${a[$((i + 1))]:-} ;;
    --speculative-draft-model-revision) DR=${a[$((i + 1))]:-} ;;
  esac
done
PY=${SGL_PYTHON:-/opt/sglang-env/bin/python}
for spec in "$MODEL@$REV" "$DM@$DR" "zai-org/GLM-4.6V-Flash@411bb4d77144a3f03accbf4b780f5acb8b7cde4e"; do
  repo=${spec%@*}; rev=${spec#*@}
  [ -n "$repo" ] || continue
  if [ "${PREFETCH_DRY:-0}" = 1 ]; then echo "[prefetch] would fetch $repo @ ${rev:-main}"; continue; fi
  ok=0
  for t in 1 2 3 4 5; do
    echo "[prefetch] $repo @ $rev: attempt $t/5"
    if HF_HUB_OFFLINE=0 "$PY" -c 'import sys, pathlib; from huggingface_hub import snapshot_download as s; p = pathlib.Path(s(sys.argv[1], revision=sys.argv[2] or None)); print("[prefetch]", p); r = p.parent.parent / "refs" / "main"
if sys.argv[2] and not r.exists(): r.parent.mkdir(parents=True, exist_ok=True); r.write_text(p.name); print("[prefetch] wrote", r, "=", p.name)' "$repo" "$rev"; then ok=1; break; fi
    [ $t -lt 5 ] || break
    w=$(echo 10 30 60 120 | cut -d' ' -f$t); echo "[prefetch] $repo: attempt $t/5 failed, retry in ${w}s"; sleep $w
  done
  [ $ok = 1 ] || { echo "[prefetch] !! could not fetch $repo @ $rev after 5 tries"; exit 1; }
done
[ "${PREFETCH_DRY:-0}" = 1 ] && exit 0
exec bash /opt/kern/shim_$K.sh "$@"
