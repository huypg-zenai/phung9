#!/usr/bin/env bash
# apply_seed_shim.sh [--apply-only | --check | --revert] [<coder shim args...>]
# Applies dflash_seed.patch (next to this file) IN PLACE to the container's SGLang, the same way
# /opt/kern/kern_shim.sh applies the GDN patches: the container is fresh per run, the image is never changed.
# PREPARED, NOT WIRED INTO ANY SUBMISSION.
#
# Wiring (when approved): mount this dir at /opt/seed and set KERN_REAL_SHIM=/opt/seed/apply_seed_shim.sh, so the
# chain becomes vllm_bin /opt/prefetch/shim_<K>.sh -> /opt/kern/shim_<K>.sh -> THIS -> /opt/sgl/sgl_shim.sh "$@".
# Default mode: apply (idempotent) -> verify -> export SGLANG_DFLASH_SEEDED_SAMPLING=1 -> exec $SEED_REAL_SHIM "$@".
#   --apply-only  apply + verify, then exit 0 (no exec)      --check   verify only (exit 1 if not applied)
#   --revert      undo the patch (tests)
# Rules: every hunk's old text must occur EXACTLY once (or its new text exactly once = already applied), else it
# refuses with '!!' and exits 1 BEFORE SGLang starts, writing nothing (all files are checked before any is written).
# A half-applied tree is refused too. One log line on success.
# Env: SGL_PYTHON (default /opt/sglang-env/bin/python), SEED_SGL_ROOT (override the sglang package dir, tests),
#      SEED_PATCH (default ./dflash_seed.patch), SEED_REAL_SHIM (default /opt/sgl/sgl_shim.sh),
#      SGLANG_DFLASH_SEEDED_SAMPLING (default 1 here; 0 = patch applied but dormant = stock behaviour),
#      SEED_SKIP_PROBE=1 (skip the import probe; for CPU-only tests without the GPU stack).
#      SEED_LOCK (default /tmp/seed_shim.lock).
set -u
D=$(cd "$(dirname "$0")" && pwd)
PATCH=${SEED_PATCH:-$D/dflash_seed.patch}
PY=${SGL_PYTHON:-/opt/sglang-env/bin/python}
REAL=${SEED_REAL_SHIM:-/opt/sgl/sgl_shim.sh}
MODE=run
case ${1:-} in --apply-only | --check | --revert) MODE=${1#--}; shift ;; esac
[ -f "$PATCH" ] || { echo "[seed-shim] !! patch file missing: $PATCH"; exit 1; }
export SGLANG_DFLASH_SEEDED_SAMPLING=${SGLANG_DFLASH_SEEDED_SAMPLING:-1}

if [ $MODE = run ] && [ "$SGLANG_DFLASH_SEEDED_SAMPLING" = 1 ]; then
  # Seeded sampling without --enable-deterministic-inference needs the pytorch sampler: the flashinfer sampler
  # asserts sampling_seed is None for top-k/top-p requests (sampler.py) -> the first request would crash the server.
  args=" $* "
  case $args in
    *" --enable-deterministic-inference "* | *" --sampling-backend pytorch "* | *" --sampling-backend=pytorch "*) ;;
    *) echo "[seed-shim] !! SGLANG_DFLASH_SEEDED_SAMPLING=1 needs '--sampling-backend pytorch' or" \
            "'--enable-deterministic-inference' in the SGLang args"; exit 1 ;;
  esac
fi

(
  flock -w 600 9 || { echo "!! [seed-shim] lock busy for 600 s: ${SEED_LOCK:-/tmp/seed_shim.lock}" >&2; exit 1; }
  "$PY" - "$MODE" "$PATCH" <<'PYEOF'
import importlib.util, os, re, subprocess, sys

mode, patch_path = sys.argv[1], sys.argv[2]
TAG = "[seed-shim]"

def die(msg):
    print(f"{TAG} !! {msg}")
    sys.exit(1)

root = os.environ.get("SEED_SGL_ROOT")
if not root:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        die("sglang not found on this interpreter's path")
    root = list(spec.submodule_search_locations)[0]

# Parse the unified diff (the commentary before the first 'diff --git' is skipped).
files, cur, hunk = [], None, None
for line in open(patch_path).read().split("\n"):
    if line.startswith("diff --git "):
        cur, hunk = {"path": None, "hunks": []}, None
        files.append(cur)
    elif cur is None:
        continue
    elif line.startswith("+++ "):
        p = line[4:].strip()
        i = p.find("sglang/")
        if i < 0:
            die(f"bad path in patch: {p}")
        cur["path"] = p[i + len("sglang/"):]
    elif line.startswith("--- "):
        continue
    elif line.startswith("@@"):
        hunk = {"old": [], "new": []}
        cur["hunks"].append(hunk)
    elif hunk is not None and line[:1] in (" ", "-", "+"):
        if line[0] in (" ", "-"):
            hunk["old"].append(line[1:])
        if line[0] in (" ", "+"):
            hunk["new"].append(line[1:])
    elif hunk is not None and line == "":
        hunk = None  # end of file / blank separator
if not files or any(not f["path"] or not f["hunks"] for f in files):
    die(f"could not parse {patch_path}")

n_hunks = sum(len(f["hunks"]) for f in files)
plan, states = [], []
for f in files:
    path = os.path.join(root, f["path"])
    if not os.path.exists(path):
        die(f"target missing: {path} (SGLang layout differs)")
    try:
        src = open(path).read()
    except OSError as exc:
        die(f"cannot read {path}: {exc}")
    new_src = src
    for k, h in enumerate(f["hunks"]):
        old, new = "\n".join(h["old"]) + "\n", "\n".join(h["new"]) + "\n"
        n_old, n_new = new_src.count(old), new_src.count(new)
        if mode == "revert":
            old, new, n_old, n_new = new, old, n_new, n_old
        if n_old == 1 and n_new == 0:
            states.append("todo")
            if mode != "check":
                new_src = new_src.replace(old, new, 1)
        elif n_new == 1 and n_old == 0:
            states.append("done")
        else:
            die(f"{f['path']} hunk {k + 1}: target lines do not match (old x{n_old}, new x{n_new}); "
                f"SGLang differs from @1cf2b8c54d81 - nothing written")
    plan.append((path, src, new_src))

if "todo" in states and "done" in states:
    die(f"half-applied tree ({states.count('done')}/{n_hunks} hunks present) - fix by hand, nothing written")
if mode == "check":
    if "todo" in states:
        print(f"{TAG} !! check: patch NOT applied at {root}")
        sys.exit(1)
else:
    written = 0
    for path, src, new_src in plan:
        if new_src != src:
            try:   # a write failure (read-only / full disk / permissions) -> one '!!' line, not a traceback
                tmp = path + ".seedtmp"
                with open(tmp, "w") as fh:
                    fh.write(new_src)
                os.replace(tmp, path)
                written += 1
                pyc = os.path.join(os.path.dirname(path), "__pycache__")
                if os.path.isdir(pyc):
                    stem = os.path.basename(path)[:-3] + "."
                    for c in os.listdir(pyc):
                        if c.startswith(stem):
                            os.remove(os.path.join(pyc, c))
            except OSError as exc:
                try:
                    os.remove(path + ".seedtmp")
                except OSError:
                    pass
                die(f"write failed ({mode}) at {path}: {exc}; files already written: {written} "
                    f"(re-run --revert / --apply-only to finish once the cause is fixed)")
    for path, _, _ in plan:
        r = subprocess.run([sys.executable, "-m", "py_compile", path], capture_output=True, text=True)
        if r.returncode != 0:
            die(f"py_compile failed after {mode}: {path}: {r.stderr.strip()[-300:]}")

if mode != "revert" and os.environ.get("SEED_SKIP_PROBE", "0") != "1":
    probe = ("import inspect, sglang.srt.speculative.dflash_utils as u, sglang.srt.speculative.dflash_worker_v2 as w, "
             "sglang.srt.sampling.sampling_batch_info as s\n"
             "assert hasattr(u, 'dflash_seeded_verify_coins'), 'helper missing'\n"
             "assert 'dflash_seeded_verify_coins(' in inspect.getsource(w.DFlashWorkerV2), 'worker not patched'\n"
             "assert 'SGLANG_DFLASH_SEEDED_SAMPLING' in inspect.getsource(s.SamplingBatchInfo), 'gate missing'\n")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/")
    if r.returncode != 0:
        die(f"import probe failed: {(r.stderr or r.stdout).strip()[-400:]}")

what = {"check": "verified", "revert": "reverted" if "todo" in states else "already reverted"}.get(
    mode, "applied" if "todo" in states else "already applied")
print(f"{TAG} dflash_seed.patch {what} ({n_hunks} hunks, {len(files)} files) at {root}; "
      f"SGLANG_DFLASH_SEEDED_SAMPLING={os.environ.get('SGLANG_DFLASH_SEEDED_SAMPLING')}")
PYEOF
) 9>"${SEED_LOCK:-/tmp/seed_shim.lock}" || exit 1
[ $MODE = run ] || exit 0
exec bash "$REAL" "$@"
