#!/bin/bash
# setup_sglang_env.sh (pgh1 leader-style, /opt/ls) = bro1's docker/setup_sglang_env.sh (md5 788b29e2, SGLang @1cf2b8c54d81
# + the same lock file) run at CONTAINER START by /opt/ls/build_envs.sh, with:
#  - every network step wrapped in ls_retry (5 attempts, waits 10/30/60/120 s, partial state cleared between attempts)
#  - SGLang fetched as ONE archive of the pinned commit (GitHub codeload tarball of the full SHA) instead of uv's git clone
#    (the clone failed twice with "gnutls_handshake() failed" in the clean build test, 18/10); the unpacked python/ tree is
#    checked against its md5 manifest, then committed into a throwaway local git repo (no network) so setuptools-scm's
#    file finder ships the same package-data files as a clone; version + commit id = those of the tested env (pretend env)
#  - Rust toolchain pinned to 1.99.1 (= the rustc that built the tested env's _grpc extension; bro1 used "stable")
#  - pip / uv / sglang-kernel / ninja pinned to the versions in the tested env (404final:v2 pip freeze); bro1 let them float
# Output: /opt/sglang-env, same packages + versions as the tested image (only the sglang line of `pip freeze` changes form:
# "sglang @ file://..." instead of "sglang @ git+...@1cf2b8c5...").
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/retry.sh"

VENV="${VENV:-/opt/sglang-env}"
PYBIN="${PYBIN:-python3.11}"
SGLANG_SHA=1cf2b8c54d81802abc15dcf23a29b9cc687bc01e          # = bro1 SGLANG_SHA 1cf2b8c54d81 (tested env direct_url)
SGLANG_VERSION=0.5.6.post3.dev9160+g1cf2b8c54                 # dist-info version of the tested env
SGLANG_NODE=g1cf2b8c54                                        # its _version.py commit_id
RUST_V=1.99.1
SGLANG_TREE_MD5=4a9e664bea51b8c18fb83243e9af1593              # md5 of `md5sum` of every file under python/ (4409 files)
SGLANG_URL="${SGLANG_URL:-https://codeload.github.com/sgl-project/sglang/tar.gz/$SGLANG_SHA}"
TORCH_BACKEND="${TORCH_BACKEND:-cu129}"
PIP_V=26.2.1; UV_V=0.13.1; SGLK_V=0.4.6.post1+cu129; NINJA_V=1.13.2
LOCK_FILE="${LOCK_FILE:-/opt/ls/sglang-env.lock.txt}"
WORK="${SGL_WORK:-/opt/ls-work/sglang}"
export CARGO_HOME="${CARGO_HOME:-/opt/rustup/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup/rustup}"
export PATH="$CARGO_HOME/bin:$PATH"
mkdir -p "$WORK"
[ -f "$LOCK_FILE" ] || { ls_log "!! lock file $LOCK_FILE missing"; exit 1; }

# 1. Rust toolchain (setuptools-rust is a build requirement of sglang's pyproject)
rust_clean() { rm -rf "$CARGO_HOME" "$RUSTUP_HOME" "$WORK/rustup-init.sh"; }
rust_get() {
  curl -fsSL --connect-timeout 30 --max-time 600 -o "$WORK/rustup-init.sh" https://sh.rustup.rs &&
    sh "$WORK/rustup-init.sh" -y --profile minimal --default-toolchain "$RUST_V" --no-modify-path
}
command -v cargo >/dev/null || ls_retry "rustup" rust_clean rust_get

ls_log "[sglang-env] $VENV | sglang @ $SGLANG_SHA ($SGLANG_VERSION) | torch=$TORCH_BACKEND | cargo $(cargo --version | awk '{print $2}')"
"$PYBIN" -m venv "$VENV"
ls_retry "pip+uv" - "$VENV/bin/pip" install "pip==$PIP_V" "uv==$UV_V"

# 2. SGLang source = one archive of the pinned commit
SRC="$WORK/src/sglang-$SGLANG_SHA"
sgl_clean() { rm -rf "$WORK/src" "$WORK/sglang.tgz" "$WORK/sglang.tgz.part"; }
sgl_get() {
  rm -rf "$WORK/src" "$WORK/sglang.tgz.part"
  curl -fSL --connect-timeout 30 --max-time 1200 -o "$WORK/sglang.tgz.part" "$SGLANG_URL" || return 1
  gzip -t "$WORK/sglang.tgz.part" || { ls_log "archive is not a complete gzip"; return 1; }
  ls_log "archive $(stat -c %s "$WORK/sglang.tgz.part") bytes sha256 $(sha256sum < "$WORK/sglang.tgz.part" | cut -d' ' -f1)"
  mkdir -p "$WORK/src" && tar -xzf "$WORK/sglang.tgz.part" -C "$WORK/src" || return 1
  local m; m=$(cd "$SRC" && find python -type f -print0 | LC_ALL=C sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1)
  [ "$m" = "$SGLANG_TREE_MD5" ] || { ls_log "python/ tree md5 $m != $SGLANG_TREE_MD5"; return 1; }
  (cd "$SRC" && git init -q && git add -A && git -c user.name=404 -c user.email=404@local commit -qm "sglang $SGLANG_SHA" \
     --no-gpg-sign) || { ls_log "local git commit failed"; return 1; }
  mv "$WORK/sglang.tgz.part" "$WORK/sglang.tgz"; touch "$WORK/src/.ok"
}
[ -f "$WORK/src/.ok" ] || ls_retry "sglang archive" sgl_clean sgl_get

# 3. install (same uv command as bro1, the git URL replaced by the unpacked archive; deps constrained by the lock file)
OVERRIDES="$WORK/overrides.txt"; echo "cuda-python<13" > "$OVERRIDES"
ls_log "[sglang-env] constraining transitive deps to $LOCK_FILE"
sgl_install() {
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$SGLANG_VERSION" SETUPTOOLS_SCM_PRETEND_METADATA_FOR_SGLANG="{node=\"$SGLANG_NODE\"}" \
    "$VENV/bin/uv" pip install --python "$VENV/bin/python" \
    "sglang[all] @ file://$SRC/python" \
    --torch-backend "$TORCH_BACKEND" --override "$OVERRIDES" --constraint "$LOCK_FILE"
}
ls_retry "uv install sglang[all]" - sgl_install

"$VENV/bin/pip" uninstall -y sgl-deep-gemm >/dev/null 2>&1 || true
ls_retry "sglang-kernel" - "$VENV/bin/pip" install --force-reinstall --no-deps "sglang-kernel==$SGLK_V" \
    --index-url https://docs.sglang.ai/whl/cu129/
ls_retry "ninja" - "$VENV/bin/pip" install "ninja==$NINJA_V"

"$VENV/bin/python" - "$SGLANG_VERSION" <<'PY'
import sys, importlib.metadata as md
import sglang, torch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.models.dflash import DFlash2DraftModel, CandidateSelector
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
assert SpeculativeAlgorithm.from_string("DFLASH").is_dflash()
assert md.version("sglang") == sys.argv[1], (md.version("sglang"), sys.argv[1])
print(f"[sglang-env] OK — sglang {md.version('sglang')} | torch {torch.__version__} | "
      f"DFlash2DraftModel + CandidateSelector + Qwen3_5ForConditionalGeneration present")
PY
rm -rf "$WORK/src" "$WORK/sglang.tgz"
ls_log "[sglang-env] done"
