#!/usr/bin/env bash
# build_sws.sh OUT_DIR [VARIANT ...] - builds kernel-optim's SwiftShader libs FROM SOURCE (run by the "sws" stage of
# docker/Dockerfile; no binary is shipped). Default VARIANTs: llvm cl4. SWS_JOBS = ninja jobs (default 16).
#   llvm -> OUT/libvk_swiftshader.so      Chrome 147's SwiftShader with the LLVM JIT (image default, PGH_SWS_LIB=llvm)
#   cl4  -> OUT/cl4/libvk_swiftshader.so  the same + src/Device/Renderer.hpp MaxClusterCount 16->4, MaxBatchSize 128->1024
#                                         (opt-in, PGH_SWS_LIB=cl4)
# Each OUT dir also gets SWS_SHA256 (the sha256 of THIS build; sws_mode.sh checks the lib against it at container start).
# OUT/BUILD_INFO.txt says, per lib, whether this build is byte-identical to the lib kernel-optim measured (REF_*), or
# identical except the compiler's version note (.comment section), or different.
# Needs Ubuntu 22.04 (= the glibc/libstdc++ of pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel), git, cmake, ninja-build,
# build-essential (gcc 11), python3, binutils. ~2-3 min per lib with 16 jobs.
# Fetches only the SwiftShader git repo itself: no submodule is initialised (LLVM 10 and the other deps are in-tree under
# third_party/), and CMake's commit-msg hook download is pre-empted (below); BUILD_INFO.txt records the submodule
# state and any download/fetch line CMake printed.
# Source: https://swiftshader.googlesource.com/SwiftShader (Apache 2.0, LICENSE-SwiftShader-Apache-2.0.txt) at the revision
# Chromium 147.0.7727.56 pins in its DEPS. git checks the tree against that full commit hash.
# What differs from SwiftShader's stock CMake build, and why:
#   1. Chromium's GN defines that CMake leaves out: SWIFTSHADER_LEGACY_PRECISION=true, SWIFTSHADER_HIGH_PRECISION_FILTERING,
#      SWIFTSHADER_ZERO_INITIALIZE_DEVICE_MEMORY (without them 0.8% of edge pixels differ from Chrome's own lib)
#   2. REACTOR_BACKEND=LLVM instead of Chrome's Subzero (the speed-up: the JIT'd raster code runs ~2x faster)
#   3. MulAdd() emitted as x * y + z like SubzeroReactor.cpp, not llvm.fmuladd (LLVM turns that into a hardware FMA on
#      AVX2 CPUs = different rounding). FMA() itself is unchanged.
#   4. cl4 only: Renderer.hpp MaxClusterCount 16 -> 4 and MaxBatchSize 128 -> 1024 (fewer, larger raster batches)
# Build paths are those of the measured builds (they end up in assert strings), so a build can be byte-identical.
# Links libstdc++ dynamically (a -static-libstdc++ build crashed Chrome).
set -euo pipefail
OUT=${1:?OUT_DIR}; shift; VARIANTS=${*:-llvm cl4}
REV=313545f85af72f954820e54f4110cda591a6cf7b
GIT=${SWS_GIT_DIR:-/w/sws/src}
REF_llvm=bb3e676767c277fc787856547a85a1561782c97b798f1741fecfb396859f5ffa   # measured llvm lib (kernel-optim R1)
REF_cl4=688cab9fe3fbf71cef7dbee09b4d0886b44f3660ef52ce58861dfac44ad61ca1    # measured cl4 lib (kernel-optim R8)
CODE_llvm=639de6f3bae73623dac8a5dd642294ef8090121d500e5ab9eb825b7e58742710  # same two files with .comment removed
CODE_cl4=2e644b02c6c4f973c4c6f9071e3072ca6bd2516d2d7c76612eac40376fcafb66
PIX_cl4=0c6f25c3c98a8aaa4689b2dede5481350861039e9514e5ebfa203dae307ce923   # this recipe's cl4 build (gcc 11.4.0-1ubuntu1~22.04.3),
# pixel-checked 23/10 against REF_cl4: all views of the 307 stable drafts of 316 bit-identical (the other 9 vary run to run with any lib)
mkdir -p "$OUT" "$(dirname "$GIT")"
[ -d "$GIT/.git" ] || git clone -q --filter=blob:none https://swiftshader.googlesource.com/SwiftShader "$GIT"
git -C "$GIT" checkout -q -f $REV
[ "$(git -C "$GIT" rev-parse HEAD)" = $REV ] || { echo "!! SwiftShader checkout is not $REV"; exit 1; }
echo "sws build: SwiftShader $REV, $(gcc --version | head -1), cmake $(cmake --version | head -1 | cut -d' ' -f3), ninja $(ninja --version)" | tee -a "$OUT/BUILD_INFO.txt"
D="-DSWIFTSHADER_LEGACY_PRECISION=true -DSWIFTSHADER_HIGH_PRECISION_FILTERING -DSWIFTSHADER_ZERO_INITIALIZE_DEVICE_MEMORY"
for v in $VARIANTS; do
  case $v in
    llvm) S=/w/ss/src_lmul; B=/w/ss/b_lmul; O=$OUT;;
    cl4)  S=/w/sws/t_cl4+bs1024/src; B=/w/sws/t_cl4+bs1024/b; O=$OUT/cl4;;
    *) echo "!! unknown variant $v (llvm|cl4)"; exit 1;;
  esac
  t0=$(date +%s); rm -rf "$S" "$B"; mkdir -p "$(dirname "$S")" "$O"; cp -a "$GIT" "$S"
  python3 - "$S" "$v" <<'PY'
import re, sys
root, v = sys.argv[1:3]
p = f"{root}/src/Reactor/LLVMReactor.cpp"; t = open(p).read()
pat = re.compile(r"(RValue<(Float4|SIMD::Float)> MulAdd\(RValue<\2> x, RValue<\2> y, RValue<\2> z\)\n\{\n).*?\n\}", re.S)
t, n = pat.subn(lambda m: m.group(1) + "\treturn x * y + z;  // kopt: no fmuladd, round like Subzero\n}", t)
assert n == 2, f"MulAdd anchors found {n}, expected 2"
open(p, "w").write(t)
if v == "cl4":
    p = f"{root}/src/Device/Renderer.hpp"; t = open(p).read()
    for old, new in (("static constexpr int MaxClusterCount = 16;", "static constexpr int MaxClusterCount = 4;  // kopt"),
                     ("static constexpr int MaxBatchSize = 128;", "static constexpr int MaxBatchSize = 1024;  // kopt")):
        assert t.count(old) == 1, (old, t.count(old))
        t = t.replace(old, new)
    open(p, "w").write(t)
PY
  # SwiftShader's CMakeLists downloads a gerrit commit-msg git hook when .git/hooks/commit-msg is missing (a developer
  # convenience, not part of the build): an empty placeholder keeps the build free of any download besides the clone
  mkdir -p "$S/.git/hooks" && : > "$S/.git/hooks/commit-msg"
  CC=gcc CXX=g++ cmake -S "$S" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DREACTOR_BACKEND=LLVM \
    -DSWIFTSHADER_BUILD_TESTS=OFF -DSWIFTSHADER_BUILD_BENCHMARKS=OFF -DSWIFTSHADER_WARNINGS_AS_ERRORS=OFF \
    -DCMAKE_C_FLAGS="$D" -DCMAKE_CXX_FLAGS="$D" > "$B.cmake.log" 2>&1 || { cat "$B.cmake.log"; exit 1; }
  grep -iE 'submodule|clon|fetch|download' "$B.cmake.log" | sed "s/^/sws build: $v cmake: /" | tee -a "$OUT/BUILD_INFO.txt" || true
  ninja -j "${SWS_JOBS:-16}" -C "$B" vk_swiftshader > "$B.ninja.log"
  sub=$(git -C "$S" submodule status 2>/dev/null || true); ns=$(grep -c . <<< "$sub" || true); ni=$(grep -vc '^-' <<< "$sub" || true)
  echo "sws build: $v submodules: $ns declared, $( [ -z "$sub" ] && echo 0 || echo $ni) initialised; ninja $(tail -1 "$B.ninja.log" | cut -c1-60)" | tee -a "$OUT/BUILD_INFO.txt"
  cp "$(find "$B" -name libvk_swiftshader.so -type f | head -1)" "$O/libvk_swiftshader.so"
  sha=$(sha256sum "$O/libvk_swiftshader.so" | cut -d' ' -f1)
  [ $v = llvm ] && dst=/opt/sws/libvk_swiftshader.so || dst=/opt/sws/$v/libvk_swiftshader.so
  echo "$sha  $dst" > "$O/SWS_SHA256"
  ref=REF_$v; code=CODE_$v
  objcopy --remove-section .comment "$O/libvk_swiftshader.so" "$B.nocomment.so"
  if [ "$sha" = "${!ref}" ]; then same="byte-identical to the measured lib ${!ref:0:16}"
  elif [ "$(sha256sum "$B.nocomment.so" | cut -d' ' -f1)" = "${!code}" ]; then
    same="identical to the measured lib ${!ref:0:16} except the compiler version note (.comment)"
  elif [ $v = cl4 ] && [ "$sha" = $PIX_cl4 ]; then
    same="= the pixel-checked build ${PIX_cl4:0:16} (vs the measured lib ${!ref:0:16}: 307/316 drafts bit-identical, the other 9 vary run to run with any lib)"
  else same="DIFFERS from the measured lib ${!ref:0:16} (pixel check needed)"; fi
  echo "sws build: $v sha256 $sha in $(( $(date +%s) - t0 )) s: $same" | tee -a "$OUT/BUILD_INFO.txt"
  rm -rf "$S" "$B"
done
