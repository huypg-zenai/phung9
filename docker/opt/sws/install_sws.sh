#!/usr/bin/env bash
# install_sws.sh [LIB] - runs at image build, after npm ci: puts the sws-llvm libvk_swiftshader.so that the Dockerfile's
# "sws" stage just BUILT FROM SOURCE (build_sws.sh; default LIB /opt/sws/libvk_swiftshader.so) into every puppeteer Chrome
# under $PUPPETEER_CACHE_DIR (default /opt/puppeteer-cache), only if that Chrome is the one it was verified against:
# Chrome 147.0.7727.56 whose bundled lib has sha256 a2c809468d66c236... Anything else -> Chrome untouched + a line saying so.
# Checks first that the lib exists, matches the sha256 the build stage recorded (/opt/sws/SWS_SHA256) and loads here (ldd).
# Never fails the image build (exit 0): every failure is one "!! sws-llvm: NOT installed" line and Chrome keeps its own lib.
# The bundled lib is kept as libvk_swiftshader.so.bundled (sws_mode.sh switches at container start; undo = copy it back).
# SWS_THREADS=N (default 4; 0 = leave SwiftShader's default 16) writes [Processor] ThreadCount=N into SwiftShader.ini next
# to the lib (SwiftShader reads it from the LIBRARY's directory, not the cwd).
set -u
LIB=${1:-/opt/sws/libvk_swiftshader.so}; C=${PUPPETEER_CACHE_DIR:-/opt/puppeteer-cache}; WANT=a2c809468d66c236
SUMS=${SWS_SUMS:-$(dirname "$LIB")/SWS_SHA256}
no() { echo "!! sws-llvm: NOT installed ($*) -> Chrome's own SwiftShader is used"; exit 0; }
[ -f "$LIB" ] || no "the sws build stage produced no $LIB; see /opt/sws/build.log"
[ "$(sha256sum "$LIB" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$SUMS" 2>/dev/null)" ] || no "$LIB sha256 != the build stage's $SUMS"
lo=$(ldd "$LIB" 2>&1) && ! grep -q 'not found' <<< "$lo" || no "$LIB does not load here: $(grep -m1 'not found' <<< "$lo" | tr -s ' \t' ' ')"
grep -s 'sws build:' "$(dirname "$LIB")/BUILD_INFO.txt"
n=0
for d in "$C"/chrome/*/chrome-linux64; do
  [ -f "$d/libvk_swiftshader.so" ] || continue
  cur=$(sha256sum "$d/libvk_swiftshader.so.bundled" 2>/dev/null || sha256sum "$d/libvk_swiftshader.so"); cur=${cur:0:16}
  if [ "$cur" != "$WANT" ]; then echo "!! sws-llvm: NOT installed in $d (bundled lib sha $cur != $WANT: other Chrome build)"; continue; fi
  [ -f "$d/libvk_swiftshader.so.bundled" ] || cp "$d/libvk_swiftshader.so" "$d/libvk_swiftshader.so.bundled"
  cp "$LIB" "$d/libvk_swiftshader.so" && n=$((n + 1)) && echo "sws-llvm: installed in $d ($(sha256sum "$LIB" | cut -c1-16), built from source in this image)"
  th=${SWS_THREADS:-4}; if [ "$th" != 0 ]; then printf '[Processor]\nThreadCount=%s\n' "$th" > "$d/SwiftShader.ini"; echo "sws-llvm: ThreadCount=$th"; fi
done
[ $n -gt 0 ] || echo "!! sws-llvm: installed nowhere -> Chrome's own SwiftShader is used"
exit 0
