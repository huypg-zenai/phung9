# sws-llvm: SwiftShader for Chrome 147, LLVM JIT backend, built from source in the image

Nothing in this directory is a binary. The Dockerfile's `sws` stage (`FROM ubuntu:22.04@sha256:0a988793... AS sws`) runs
`build_sws.sh`, which builds two `libvk_swiftshader.so` files from **SwiftShader** source
(https://swiftshader.googlesource.com/SwiftShader, Apache License 2.0; a copy of the licence is in
`LICENSE-SwiftShader-Apache-2.0.txt`). The final image copies them to `/opt/sws/` with `COPY --from=sws`.

The source is the revision that Chromium 147.0.7727.56 pins in its DEPS, `313545f85af72f954820e54f4110cda591a6cf7b`.
git checks the checkout against that full commit hash. The build uses the LLVM JIT backend (`REACTOR_BACKEND=LLVM`) with
the LLVM copy in SwiftShader's `third_party/` (Apache 2.0 with LLVM exception). Chrome for Testing ships the same revision
built with the Subzero backend.

Changes compared with SwiftShader's stock CMake build. All of them are in `build_sws.sh`:
1. Chromium's GN defines that the CMake build leaves out: `SWIFTSHADER_LEGACY_PRECISION=true`,
   `SWIFTSHADER_HIGH_PRECISION_FILTERING`, `SWIFTSHADER_ZERO_INITIALIZE_DEVICE_MEMORY`.
2. In `src/Reactor/LLVMReactor.cpp`, `MulAdd()` for Float4 and SIMD::Float is emitted as `x * y + z`, as Subzero does.
   Stock LLVM emits `llvm.fmuladd`, which becomes a fused multiply-add on AVX2 CPUs and rounds differently.
3. Only for `cl4/libvk_swiftshader.so`: `src/Device/Renderer.hpp` `MaxClusterCount` 16 -> 4 and `MaxBatchSize` 128 -> 1024.

| Lib | Built to | Used when |
|---|---|---|
| sws-llvm | `/opt/sws/libvk_swiftshader.so` + `/opt/sws/SWS_SHA256` | `PGH_SWS=1`, `PGH_SWS_LIB=llvm` (image default) |
| sws-llvm-cl4 | `/opt/sws/cl4/libvk_swiftshader.so` + `/opt/sws/cl4/SWS_SHA256` | `PGH_SWS=1`, `PGH_SWS_LIB=cl4` (opt-in) |

`SWS_SHA256` holds the sha256 of the file this image build produced. `/opt/sws/BUILD_INFO.txt` records the compiler and
build time. It also says whether each build is byte-identical to the lib kernel-optim measured (llvm `bb3e6767...`,
cl4 `688cab9f...`), or identical except the compiler's version note in the `.comment` section.
The build uses the same source and build paths as those measured builds, so a build on Ubuntu 22.04 can match them.

| File | What it does |
|---|---|
| `build_sws.sh OUT [llvm] [cl4]` | The recipe above. It runs in the Dockerfile's `sws` stage. It needs Ubuntu 22.04 (glibc 2.35 and libstdc++ = the base of `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`), git, cmake, ninja-build, build-essential, python3 and binutils. ninja runs `SWS_JOBS` jobs (16 in the Dockerfile). It takes about 2-3 min per lib. |
| `install_sws.sh [LIB]` | Runs at image build. It checks that the built lib exists, matches `SWS_SHA256` and loads (`ldd`). It then copies the lib into `$PUPPETEER_CACHE_DIR/chrome/*/chrome-linux64/`, but only when that Chrome's bundled lib has sha256 `a2c809468d66c236...` (Chrome 147.0.7727.56). It keeps the bundled lib as `libvk_swiftshader.so.bundled` and writes `SwiftShader.ini` (`[Processor] ThreadCount=4`). If anything fails it prints `!! sws-llvm: NOT installed`, leaves Chrome's own lib in place and does not fail the build. |
| `sws_mode.sh` | Runs at container start; run.sh calls it. `PGH_SWS=0` restores the bundled lib. `PGH_SWS=1` (the default) uses sws-llvm with `PGH_SWS_THREADS` threads. `PGH_SWS_LIB=cl4` uses the cl4 lib, but only if it matches its `SWS_SHA256` and loads. Otherwise it prints one `!!` line and uses llvm. |

If the SwiftShader build fails (apt-get, clone or compile; for example, no network), the image still builds. The `sws` stage then
ships no lib, `install_sws.sh` prints `!! sws-llvm: NOT installed`, and every container start prints
`[sws] !! sws-llvm not installed`. Chrome then renders with its own bundled SwiftShader. That gives the same pixels at
about 1.6x the render CPU.
