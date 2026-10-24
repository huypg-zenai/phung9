# retry.sh (sourced; pgh1 leader-style, /opt/ls/retry.sh) - retries for every network step of the start-up env build.
#   ls_retry <label> <cleanup-function|-> <command...>
# Runs <command> up to LS_TRIES (5) times; waits LS_WAITS ("10 30 60 120" s, the last one repeats) between attempts and
# runs <cleanup-function> before each new attempt to clear partial state. Every attempt is logged. Returns 0 on the first
# success, 1 after the last failure (the caller then fails loudly). Safe under `set -e` (the command runs inside `if`).
ls_log() { echo "[404-env $(date -u +%T)] $*"; }
ls_retry() {
  local label=$1 clean=$2; shift 2
  local n=${LS_TRIES:-5} i w rc; local -a waits=(${LS_WAITS:-10 30 60 120})
  for ((i = 1; i <= n; i++)); do
    ls_log "$label: attempt $i/$n"
    if "$@"; then ls_log "$label: OK (attempt $i/$n)"; return 0; else rc=$?; fi
    ls_log "$label: attempt $i/$n FAILED (rc=$rc)" >&2
    [ $i -lt $n ] || break
    w=${waits[$((i - 1))]:-${waits[$((${#waits[@]} - 1))]}}
    ls_log "$label: retry in ${w}s"; sleep "$w"
    [ "$clean" = - ] || "$clean"
  done
  ls_log "!! $label: FAILED after $n attempts" >&2
  return 1
}
