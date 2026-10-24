#!/usr/bin/env bash
# sws_mode.sh - RUN-TIME switch for kernel-optim's sws-llvm SwiftShader. The libs are BUILT FROM SOURCE in the Dockerfile's
# "sws" stage (build_sws.sh) and installed at image build by install_sws.sh. Called by run.sh before anything starts
# Chrome; a box A/B needs no rebuild:
#   PGH_SWS=1 (image default)  sws-llvm lib active (only where install_sws.sh installed it, i.e. Chrome 147.0.7727.56
#                              whose bundled lib sha256 = a2c809468d66c236...; it kept libvk_swiftshader.so.bundled)
#   PGH_SWS=0                  Chrome's bundled lib put back + SwiftShader.ini removed = exactly pgh17's renderer
#   PGH_SWS_THREADS=N          (with PGH_SWS=1; image default 4) [Processor] ThreadCount=N in SwiftShader.ini NEXT TO the
#                              lib (SwiftShader reads it from the library's dir); 0 = no ini (SwiftShader default 16)
#   PGH_SWS_LIB=llvm|cl4       (with PGH_SWS=1; image default llvm) llvm = /opt/sws/libvk_swiftshader.so | cl4 =
#                              /opt/sws/cl4/libvk_swiftshader.so (the llvm lib + Renderer.hpp MaxClusterCount 16->4,
#                              MaxBatchSize 128->1024; kernel-optim R8). Each lib is used only if its sha256 equals the one
#                              the build stage recorded next to it (SWS_SHA256) and ldd finds every library/symbol version
#                              it needs; cl4 failing -> '!!' + llvm; llvm failing -> '!!' + Chrome dir left unchanged.
#                              Anything else for PGH_SWS_LIB -> '!!' + llvm.
# PGH_SWS=1 with PGH_RENDER_PIN=0 -> '!!' + mode 0 (kernel-optim gate 22/10 on AVX-512 Zen 4: the lib WITHOUT the
#                              program pin was slower than Chrome's lib, 1.76 vs 2.08 drafts/s); PGH_SWS_NOPIN_OK=1 forces it.
# Anything else for PGH_SWS -> '!!' line + treated as 1. Never fails the container start (always exit 0); every outcome
# is one "[sws]" line naming the lib used (grep the pod log for it). Dirs without .bundled (= install refused) are left untouched.
# Test hooks: PUPPETEER_CACHE_DIR (default /opt/puppeteer-cache), SWS_LIB (default /opt/sws/libvk_swiftshader.so),
# SWS_SHA (expected sha256 of SWS_LIB, default the first field of SWS_SHA256 next to it), SWS_CL4_LIB (default
# /opt/sws/cl4/libvk_swiftshader.so), SWS_CL4_SHA (default from cl4/SWS_SHA256), SWS_LDD (default ldd).
set -u
C=${PUPPETEER_CACHE_DIR:-/opt/puppeteer-cache}
LIB=${SWS_LIB:-/opt/sws/libvk_swiftshader.so}
WANT=${SWS_SHA:-$(cut -d' ' -f1 "$(dirname "$LIB")/SWS_SHA256" 2>/dev/null)}
KLIB=${SWS_CL4_LIB:-/opt/sws/cl4/libvk_swiftshader.so}
KWANT=${SWS_CL4_SHA:-$(cut -d' ' -f1 "$(dirname "$KLIB")/SWS_SHA256" 2>/dev/null)}
mode=${PGH_SWS:-1}; th=${PGH_SWS_THREADS:-4}; sel=${PGH_SWS_LIB:-llvm}
case "$mode" in 0|1) ;; *) echo "[sws] !! PGH_SWS='$mode' is not 0/1 -> 1"; mode=1;; esac
case "$th" in ''|*[!0-9]*) echo "[sws] !! PGH_SWS_THREADS='$th' is not a number -> 4"; th=4;; esac
[ "$th" -gt 64 ] && { echo "[sws] !! PGH_SWS_THREADS=$th > 64 -> 4"; th=4; }
case "$sel" in llvm|cl4) ;; *) echo "[sws] !! PGH_SWS_LIB='$sel' is not llvm/cl4 -> llvm"; sel=llvm;; esac
if [ "$mode" = 1 ] && [ "${PGH_RENDER_PIN:-1}" = 0 ] && [ "${PGH_SWS_NOPIN_OK:-0}" != 1 ]; then
  echo "[sws] !! PGH_SWS=1 with PGH_RENDER_PIN=0: sws-llvm without the pin is slower than Chrome's lib -> mode 0 (PGH_SWS_NOPIN_OK=1 forces 1)"
  mode=0
fi
# cl4: check the built file once (sha, then loadability), before any Chrome dir is touched; on any doubt use llvm
if [ "$mode" = 1 ] && [ "$sel" = cl4 ]; then
  got=$(sha256sum "$KLIB" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$KWANT" ] || [ "$got" != "$KWANT" ]; then
    echo "[sws] !! PGH_SWS_LIB=cl4: $KLIB sha256 '${got:0:16}' != build stage's '${KWANT:0:16}' (not built?) -> llvm"; sel=llvm
  else
    lo=$(${SWS_LDD:-ldd} "$KLIB" 2>&1); lrc=$?
    if [ $lrc != 0 ] || grep -q 'not found' <<< "$lo"; then
      echo "[sws] !! PGH_SWS_LIB=cl4: $KLIB does not load here (ldd rc $lrc: $(grep -m1 'not found' <<< "$lo" | tr -s ' \t' ' ' | cut -c1-300)) -> llvm"; sel=llvm
    fi
  fi
fi
ini_() { if [ "$th" = 0 ]; then rm -f "$ini"; else printf '[Processor]\nThreadCount=%s\n' "$th" > "$ini"; fi; }
tc_() { [ "$th" = 0 ] && echo 'default (no ini)' || echo "$th"; }
n=0
for d in "$C"/chrome/*/chrome-linux64; do
  [ -f "$d/libvk_swiftshader.so.bundled" ] || continue
  n=$((n + 1)); cur="$d/libvk_swiftshader.so"; ini="$d/SwiftShader.ini"
  if [ "$mode" = 0 ]; then
    if cmp -s "$d/libvk_swiftshader.so.bundled" "$cur" || cp "$d/libvk_swiftshader.so.bundled" "$cur"; then
      rm -f "$ini" && echo "[sws] mode 0: Chrome's bundled lib in $d ($(sha256sum "$cur" | cut -c1-16)), no SwiftShader.ini" \
        || echo "[sws] !! mode 0: bundled lib in $d but SwiftShader.ini could not be removed"
    else
      echo "[sws] !! mode 0: could not restore the bundled lib in $d (read-only?) -> state unchanged: $(sha256sum "$cur" | cut -c1-16)"
    fi
    continue
  fi
  if [ "$sel" = cl4 ]; then
    # copied file verified after the copy; any failure -> fall through to the llvm path below (never a half state)
    if { cmp -s "$KLIB" "$cur" || cp "$KLIB" "$cur"; } 2>/dev/null && [ "$(sha256sum "$cur" | cut -d' ' -f1)" = "$KWANT" ]; then
      ini_; echo "[sws] mode 1: sws-llvm-cl4 in $d ($(sha256sum "$cur" | cut -c1-16)), ThreadCount $(tc_)"; continue
    fi
    echo "[sws] !! PGH_SWS_LIB=cl4: could not put sws-llvm-cl4 into $d -> llvm"
  fi
  got=$(sha256sum "$LIB" 2>/dev/null | cut -d' ' -f1)
  if [ -z "$WANT" ] || [ "$got" != "$WANT" ]; then
    echo "[sws] !! mode 1: $LIB sha256 '${got:0:16}' != build stage's '${WANT:0:16}' -> state unchanged in $d: $(sha256sum "$cur" | cut -c1-16)"
    continue
  fi
  if ! { cmp -s "$LIB" "$cur" || cp "$LIB" "$cur"; }; then
    echo "[sws] !! mode 1: could not copy sws-llvm into $d -> state unchanged: $(sha256sum "$cur" | cut -c1-16)"; continue
  fi
  ini_
  echo "[sws] mode 1: sws-llvm in $d ($(sha256sum "$cur" | cut -c1-16)), ThreadCount $(tc_)"
done
if [ $n = 0 ]; then
  if [ "$mode" = 1 ]; then echo "[sws] !! sws-llvm not installed in this image (build stage or install_sws.sh refused, see /opt/sws/build.log): Chrome's own SwiftShader, PGH_SWS ignored"
  else echo "[sws] mode 0: sws-llvm not installed in this image: Chrome's own SwiftShader"; fi
fi
exit 0
