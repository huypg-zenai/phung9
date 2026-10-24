# syntax=docker/dockerfile:1.7
# [sws-src] SwiftShader built FROM SOURCE (docker/opt/sws/build_sws.sh; README.md there): the revision Chrome 147.0.7727.56
# pins, LLVM JIT, two libs: llvm (default, PGH_SWS_LIB=llvm) and cl4 (opt-in). Ubuntu 22.04 (pinned by digest) = the
# glibc/libstdc++ of the final image's base. A failed SwiftShader build (apt-get, clone or compile; e.g. no network) does
# not fail the image: the stage then ships no lib, install_sws.sh prints "!! sws-llvm: NOT installed" and Chrome keeps its
# own SwiftShader. ninja runs at most SWS_JOBS=16 jobs (this stage runs next to the other build stages).
FROM ubuntu:22.04@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956 AS sws
RUN mkdir -p /out && { { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates git cmake ninja-build build-essential python3 binutils && rm -rf /var/lib/apt/lists/*; } > /out/apt.log 2>&1 \
    || echo "!! sws stage: apt-get FAILED (rc $?): no SwiftShader build, see /opt/sws/apt.log" | tee -a /out/build.log /out/BUILD_INFO.txt; }
COPY docker/opt/sws/build_sws.sh /w/build_sws.sh
RUN for v in llvm cl4; do SWS_JOBS=16 bash /w/build_sws.sh /out $v >> /out/build.log 2>&1 \
        || echo "!! sws build of $v FAILED (rc $?): see /opt/sws/build.log" | tee -a /out/build.log /out/BUILD_INFO.txt; done; \
    cat /out/BUILD_INFO.txt
# ---- [wq-src] kopt-wq FP8 GEMM tile table: the extension is built HERE, at image build, from the sources in
# docker/opt/v5/wq (kopt_wq_gemm.cuh + sgl_csrc/ + kopt_wq_table.json -> apply_wq.py --build; no kopt-wq binary in the repo).
# No GPU needed (explicit sm_90a gencode). Toolchain = the coder env's torch 2.13.0+cu129 + python3.11, this base image's
# nvcc 12.8, CUTLASS @57e3cfb4 (sha256-checked). A failed build never fails the image: build_wq.sh prints
# '!! kopt-wq: build FAILED' and leaves no .so, so SGL_WQ=1 is refused at coder start (stock FP8 GEMM path); the
# RUN below also catches a missing/crashing script, so this stage always ends with /opt/kopt_wq present.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS kopt_wq_build
COPY docker/opt/v5/wq/ /opt/v5/wq/
RUN mkdir -p /opt/kopt_wq && (bash /opt/v5/wq/build_wq.sh /opt/kopt_wq || { echo "!! kopt-wq: build step error -> no extension"; \
      rm -f /opt/kopt_wq/kopt_wq_ext.so /opt/kopt_wq/manifest.json; : > /opt/kopt_wq/BUILD_FAILED; })

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:/opt/vllm-env/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib64:$LD_LIBRARY_PATH \
    PUPPETEER_CACHE_DIR=/opt/puppeteer-cache \
    NODE_MAJOR=20 \
    RENDER_POOL_SIZE=32

# System deps + Node.js 20 + Chromium runtime libs
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget git build-essential bash gnupg \
        # headless Chromium runtime
        fonts-liberation fonts-noto-color-emoji \
        libnss3 libatk-bridge2.0-0 libx11-xcb1 libxcomposite1 \
        libxdamage1 libxrandr2 libgbm1 libasound2 \
        libpangocairo-1.0-0 libgtk-3-0 \
        libegl1 libgl1-mesa-dri libgles2 \
        xvfb; \
    # Node.js 20 LTS via NodeSource
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash -; \
    apt-get install -y --no-install-recommends nodejs; \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    speedtest-cli 

# Python deps
COPY docker/requirements.vllm.txt /tmp/requirements.vllm.txt
COPY docker/requirements.txt      /tmp/requirements.txt
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/vllm-env && \
    /opt/vllm-env/bin/pip install pip==26.2.1 && \
    /opt/vllm-env/bin/pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements.lock.txt

# Node deps 
# ---- 404 pgh1, leader style (19/10): only code is added here. /opt/ls/build_envs.sh (called by run.sh at container start,
# like the leader's own GLM env build) builds the SGLang coder env (lock file), the GDN kernel patches, the GLM vLLM judge
# env (the leader's script) and the judge scheduler patch, with retries; a failure keeps every server down (/status=replace).
ENV FLASHINFER_CACHE_DIR=/root/.cache/huggingface/.engine/flashinfer \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/huggingface/.engine/inductor \
    TRITON_CACHE_DIR=/root/.cache/huggingface/.engine/triton \
    SGLANG_CACHE_DIR=/root/.cache/huggingface/.engine/sglang
COPY docker/opt/ls/ /opt/ls/
COPY docker/opt/sgl/ /opt/sgl/
COPY docker/opt/kern/ /opt/kern/
COPY docker/opt/jf/ /opt/jf/
COPY docker/opt/prefetch/ /opt/prefetch/
COPY docker/opt/v3/ /opt/v3/
COPY docker/opt/raff/ /opt/raff/
COPY docker/opt/v4/ /opt/v4/
RUN chmod +x /opt/ls/*.sh /opt/sgl/*.sh /opt/kern/*.sh /opt/jf/*.sh /opt/prefetch/*.sh /opt/v3/*.sh /opt/v3/seed/*.sh /opt/raff/*.sh /opt/v4/*.sh
# entry 3 combo: kernel-optim coder handovers (wq, fusedq, mrope_nb, topkbf, normcfg, casc) + their installer
# /opt/v5/kopt_shim.sh. [wq-src] kopt-wq's extension comes from the kopt_wq_build stage above (built from source in this
# image build; kopt_wq.py checks its torch / configs / table). kopt_shim.sh expects the kit copy in /opt/v5/wq/built
# + its md5 in /opt/v5/MANIFEST and cmp's it with /opt/kopt_wq: both are written here from that build.
COPY docker/opt/v5/ /opt/v5/
COPY --from=kopt_wq_build /opt/kopt_wq/ /opt/kopt_wq/
RUN chmod +x /opt/v5/*.sh && cd /opt/v5 && if [ -f /opt/kopt_wq/kopt_wq_ext.so ] && [ -f /opt/kopt_wq/manifest.json ]; then \
      mkdir -p wq/built && cp /opt/kopt_wq/kopt_wq_ext.so /opt/kopt_wq/manifest.json wq/built/ && \
      md5sum wq/built/kopt_wq_ext.so wq/built/manifest.json >> MANIFEST && \
      echo "kopt-wq: built-from-source extension installed (sha256 $(sha256sum /opt/kopt_wq/kopt_wq_ext.so | cut -c1-16))"; \
    else echo "!! kopt-wq: no extension from the build stage (see /opt/kopt_wq/BUILD.log) -> SGL_WQ=1 will be REFUSED at coder start"; fi
WORKDIR /workspace
COPY docker/package.json docker/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund
# [sws] kernel-optim sws-llvm: Chrome 147's own SwiftShader revision with the
# LLVM JIT, BUILT FROM SOURCE in the "sws" stage at the top of this file (docker/opt/sws/build_sws.sh; README.md there).
# Same pixels, ~1/3 less render CPU. No binary is shipped. install_sws.sh checks the built lib (sha256 recorded
# by the build stage + ldd) and swaps it ONLY into Chrome 147.0.7727.56 with the verified bundled lib, keeps
# libvk_swiftshader.so.bundled, writes SwiftShader.ini ThreadCount=4; no lib / any failure -> "!! sws-llvm: NOT installed"
# and Chrome keeps its own lib (the image build never fails on it).
# Run time: PGH_SWS / PGH_SWS_THREADS / PGH_SWS_LIB below (sws_mode.sh, called by run.sh) switch it without a rebuild.
# [cl4] the opt-in sws-llvm-cl4 lib (/opt/sws/cl4/, PGH_SWS_LIB=cl4) is built in the same stage; it is copied in only
# at container start, after the same sha256 + ldd checks (failure: '!!' + llvm).
COPY docker/opt/sws /opt/sws
COPY --from=sws /out/ /opt/sws/
RUN bash /opt/sws/install_sws.sh /opt/sws/libvk_swiftshader.so

# Add the pipeline service
COPY pipeline_service     /workspace
COPY configuration.yaml   /workspace/configuration.yaml
RUN chmod +x /workspace/run.sh

EXPOSE 10006 8001 8002
CMD ["/workspace/run.sh"]
# run-time settings of the tested package (LOOP top-4 round robin, burst judges, uuid judge client)
ENV BURST_GLM_URLS="http://localhost:8031/v1,http://localhost:8032/v1"
ENV BURST_PROXY="http://localhost:8001"
ENV JUDGE_LB_MIGRATE="1"
ENV SAVE_DRAFTS_DIR=""
ENV SPEC_FINAL="0"
# [jgrid] judge grid seen by our S2 judge: off (= today) | small (1041 px, speed only) | contest (= the contest judge grid).
# Class C when not off; run the box acceptance checks A1-A10 first. Default off.
ENV JUDGE_GRID="off"
ENV SAVE_SEMIS_DIR=""
ENV SAVE_SEMIS_MIN_ROUND="1"
ENV JUDGE_STOP_ISSUES="0"
ENV JUDGE_KEEPALIVE2="0"
ENV RENDER_DEADLINE_S="0"
ENV RENDER_DEADLINE_MIN_LIVE="20"
# r42: one knob for the render-path timeouts (modules/renderer/render_timeouts.py). "0"/unset = old 60 s / 60 s / 90 s;
# 120 = render 120 s, Chrome protocol 150 s, Python HTTP 300 s (R43; output-changing only for renders that would have hit 60 s)
ENV PGH_RENDER_TIMEOUT_S="120"
# r42_keepalive: render sidecars keep idle sockets 65 s (> httpx 5 s), so the client never reuses a socket node closed
# (ReadError retries / dropped drafts). Unset or "0" = node default 5 s. render_runner.mjs reads it at sidecar start.
ENV PGH_RENDER_KEEPALIVE_MS="65000"
ENV ROUND_ROBIN_TOP4="1"
ENV ROUND_ROBIN_CUTOFF_S="2400"
ENV JUDGE_IMG_UUID="1"
# v2 RENDER_AFF: render sidecars pinned to floor(cpu quota) - RESERVE CPUs (refused below MIN -> plain node)
ENV RENDER_AFF_RESERVE="14"
ENV RENDER_AFF_MIN="8"
# v3 switches (pgh1v3_README.txt). Read at container start; override with docker run -e NAME=value, no rebuild.
# Coder chain (entry 2): kern_shim.sh -> /opt/v4/speed_shim.sh (gsw + topk20 speed patches, fail closed)
#   -> /opt/v3/coder_switch.sh -> /opt/sgl/sgl_shim.sh (seed patch + extra SGLang flags).
ENV KERN_REAL_SHIM="/opt/v4/speed_shim.sh"
# entry 2 speed patches (CLASS A: bitwise-identical output; /opt/v4/speed_shim.sh applies + GPU-checks them at coder
# start, H200 only; any failure -> that patch off + a '!!' line, the coder still starts). 0 = off (= entry 1).
# 1 = FP8 GEMM switch to torch._scaled_mm for the 427 table (N,K,M) shapes (kernel-optim gsw_m6)
# entry 3 combo: 0, because kopt-wq (SGL_WQ below) replaces it: same fp8_utils.fp8_scaled_mm hook, wq README "do not
# combine". Box arm for gsw instead of wq: -e SGL_GSW=1 -e SGL_WQ=0 (with both 1, wq wins when its check passes).
ENV SGL_GSW="0"
# 1 = exact Triton top-20 in the DFlash verify (kernel-optim topk20)
ENV SGL_TOPK20="1"
# entry 3 combo: speed_shim.sh's next link = /opt/v5/kopt_shim.sh (-> /opt/v3/coder_switch.sh): installs the
# kernel-optim handovers below at coder start, each fail closed (apply + its README --check incl. the GPU self-test,
# H200 only; any failure -> '[v5-kopt] <item> REFUSED' + that item reverted + the stock path, the coder still starts).
# All switches 0 = pure pass-through (reads/writes nothing). V4_REAL_SHIM=/opt/v3/coder_switch.sh skips it.
ENV V4_REAL_SHIM="/opt/v5/kopt_shim.sh"
# 1 = FP8 GEMM tile table (kopt-wq, CLASS A; its extension is compiled from source in stage kopt_wq_build).
# REPLACES gsw (SGL_GSW="0" above); if someone
# sets both to 1, wq wins when its check passes (SGL_GSW exported 0, gsw hook dormant), else gsw stays as /opt/v4 set it.
ENV SGL_WQ="1"
# 1 = fused SiLU-and-mul + per-token FP8 quant for down_proj (kernel-optim fusedq, CLASS A; SGL_FUSEDQ_NORM forced 0).
# Installed but OFF: gain -0.6 ms/step = 1-1.5x the noise floor, and its --check JIT-builds the kernel (ninja + nvcc)
# at coder start (time unmeasured on a loaded pod). Box arm: -e SGL_FUSEDQ=1 (time the "fusedq ACTIVE" line).
ENV SGL_FUSEDQ="0"
# 1 = pinned + non_blocking H2D copy of the mrope deltas in the verify prep (kernel-optim MROPE-NB, CLASS A)
ENV SGL_MROPE_NB="1"
# 1 = DFlash verify logits stay bf16, top-k divides by T in-kernel (kopt-topkbf, CLASS A; composes with topk20)
ENV SGL_TOPKBF="1"
# 1 = GDN gated RMSNorm launched 32 rows / 8 warps (kopt-normcfg, CLASS A)
ENV SGL_NORMCFG="1"
# 1 = shared-prefix cascade attention in the coder verify (kernel-optim casc, CLASS B: output can change in rare tokens).
# Installed but OFF; box arm: docker run -e SGL_CASC=1. SGL_CASC_PRE_SPLITS must stay 1 (README: required).
ENV SGL_CASC="0"
ENV SGL_CASC_PRE_SPLITS="1"
# 1 = dflash_seed.patch applied at coder start + --sampling-backend pytorch (output-changing; speed cost unmeasured)
ENV SGLANG_DFLASH_SEEDED_SAMPLING="1"
# 1 = --enable-deterministic-inference (output-changing; slower, unmeasured)
ENV PGH_DETERMINISTIC_INFERENCE="0"
# 1 = abort the coder start if the seed patch is refused (default: fall back to the unpatched start + a !! line)
ENV PGH_SEED_STRICT="0"
# coder K (ensemble_size), adaptive_k.py: fixed = PGH_K40 on every pod | adaptive = max(PGH_K40, rule(cores))
# | rule = the CPU-budget rule alone | off = the configured K 68 (= pgh1v2)
# [pgh16] THE K KNOB (as pgh15): mode fixed + PGH_K40 (4..128) = the coder draft count on every pod (pgh7/cand: rule, K 80).
# PGH_K40 80 is a PLACEHOLDER: the parent sets it with pgh16/set_k.sh after the box timing run.
ENV PGH_K_MODE="fixed"
# K for a 40.8-CPU pod (fixed: K everywhere; adaptive: the floor). Bad value -> !! + the build default
ENV PGH_K40="70"
# legacy switch: anything but 1 (e.g. 0) = off, overriding PGH_K_MODE; 1 = PGH_K_MODE decides
ENV PGH_ADAPTIVE_K="1"
# JSON object overriding any of the PROVISIONAL rule coefficients (adaptive_k.DEFAULT_COEFFS keys: target_s, kappa,
# overhead_s, coder_s_per_draft, render_core_s_per_draft, k_min, k_max, k_fallback, ...); "" = defaults
ENV PGH_ADAPTIVE_K_COEFFS='{"target_s":6000,"round_factor":4.47,"overhead_s":66.4,"coder_s_per_draft":10.33,"render_core_s_per_draft":13.48,"kappa":1.0,"render_sat_cores":32.0,"k_max":80,"k_fallback":80}'
# round-budget guard at batch boundaries (pgh_timeguard.py): live = cut K between batches when the round would
# overrun (output-changing when it fires) | shadow = log what live would do, change nothing | off = = pgh1v2
ENV PGH_TIMEGUARD="off"
# JSON object overriding the guard params (round_s, margin_s, batches_per_round, steps_pct, k_floor, idle_reset_s)
ENV PGH_TIMEGUARD_PARAMS=""
# [R96 pgh17d] 1 = a prompt cut off by the task deadline / batch budget ships its best draft so far (bracket
# leader > best rendered > lowest checked) instead of nothing (pgh_salvage.py; one [SALVAGE] line each). 0 = pgh17d.
# Image default ON (R96): it only touches prompts that would otherwise ship nothing (a sure loss).
ENV PGH_DEADLINE_SALVAGE="1"
# [R96 pgh17d] 1 = CPU-quota K prior + render-backlog brake (pgh_kcpu.py) | prior | brake | shadow (log only) | 0 = pgh17d
# Image default 0 = OFF (R96, owner 22/10: not used now; hooks skipped). `shadow` = log decisions only, K unchanged.
ENV PGH_K_CPU_AWARE="0"
# JSON object overriding pgh_kcpu.DEFAULT_PARAMS (e.g. {"target_s": 2000})
ENV PGH_KCPU_PARAMS=""
# 1 = A+ render fix in the js_checker serialize worker (output-changing: rejects non-texture map slots, bakes
# vertex-colour geometries); 0 = byte-identical to the unpatched worker. OFF until its duel test.
ENV PGH_APLUS="0"
# orient_gate: turn a side-best draft only if s_side - max(s_front, s_back) >= this; unset/"0" = always turn (= entry 1)
ENV PGH_ORIENT_MIN_MARGIN="0.15"
# [pgh16] coder prompt (modules/scene_coder/coder_prompt.py): es = expert-spork r56's full coder prompt verbatim (owner
# decision 22/10; system md5 7d9bd866, user md5 89e3576e) | pgh7 = byte-identical to pgh7 | ours = pgh7 + our texture
# section | oursneg = ours + the non-existent-name list (test variant)
ENV PGH_CODER_PROMPT="es"
# [pgh16] 1 = a draft rejected ONLY for non-existent THREE constants (UNKNOWN_THREE_API) gets them mapped to allowed ones
# (modules/js_checker/three_repair.py) and is kept if the same checker then passes it; 0 = off (= pgh7). OFF until its A/B.
ENV PGH_THREE_REPAIR="0"
# [R81] 1 = a draft rejected for ONE repairable error class (duplicate declaration, extra ')', undeclared names, const
# reassignment, NaN "Invalid bounds", texture > 4 MiB) gets a mechanical rewrite (modules/js_checker/mech_repair.py), kept only
# if the same checker then passes it; 0 = off (= pgh16).
ENV PGH_MECH_REPAIR="0"
# [R81] 1 = a draft whose ONLY checker failure is TIMEOUT_EXCEEDED is checked once more, unchanged; 0 = off (= pgh16).
ENV PGH_CHECK_TIMEOUT_RETRY="0"
# [pgh17 mix] TWO coders on one 4xH200 pod (pipeline_service/pgh_mix.py, pgh17/README.md): GPU 0 = UtevoLux (+ z-lab
# DFlash, = pgh16's coder), GPU 1 = AmberKite (+ AK-DFlash, = pgh7's coder), GPUs 2-3 = the GLM judges (unchanged). Each
# prompt's K drafts are interleaved between them (the FIRST name gets PGH_MIX_SPLIT of them); the normal dedup + judge
# bracket picks the final. "" (or 0/off) = exactly pgh16 (one UtevoLux coder on GPUs 0,1 DP2). Invalid -> '!!' + pgh16.
ENV PGH_MIX_CODERS="utevo:ak"
# share of K for the FIRST name (utevo), 0 < f < 1: 0.5 = owner's 50/50 (K80 -> 40 + 40; odd K: utevo gets the extra one).
# The UtevoLux GPU then does the same K/2 drafts as in pgh16's DP2, so the coder phase ~= pgh16's (AmberKite, ~1.7x
# faster per GPU, finishes first). 0.37 would balance the two GPUs (~0.74x pgh16's coder phase). A cut keeps the ratio.
ENV PGH_MIX_SPLIT="0.42"
# prompt per coder (modules/scene_coder/coder_prompt.py variants): UtevoLux needs ES's prompt (pgh7's -> made-up THREE
# names in 97% of drafts, utevo_why2), AmberKite was tuned on pgh7's. A name not listed uses PGH_CODER_PROMPT.
ENV PGH_MIX_PROMPTS="utevo=es,ak=pgh7"
# start-up: one coder up, the other not -> wait this long, then go READY on the one coder alone (it gets full K)
ENV PGH_MIX_WAIT_S="1200"
# sgl_shim.sh: max wait for the other coder's patch chain before this coder's SGLang starts (shared SGLang env)
ENV PGH_MIX_BARRIER_S="1800"
# [R84 jalign] judge-alignment switches (pipeline_service/pgh_jalign.py; opt_r3/judge/README.md). "" = off = pgh17.
# S1 early stop only in bracket rounds <= this (A2 #3a: 2); "" = configuration.yaml max_round (4).
ENV PGH_S1ES_MAX_ROUND=""
# 1 = full top-8 round robin (28 pairs of the eighth-block winners) for all but the last PGH_RR8_TAIL photos of a batch.
ENV PGH_RR8=""
ENV PGH_RR8_TAIL="4"
# s1pen = when ROUND_ROBIN_CUTOFF_S fires, pick the lowest mean S1 penalty of the top set (logs [RR_CUTOFF_FALLBACK]).
ENV PGH_RR_CUTOFF_FALLBACK="s1pen"
# [R87 pgh17c] the other switches of the merged tree (pgh17c/README.md). All OFF = pgh17 except PGH_TIE_NEUTRAL (hygiene)
# and PGH_RR_CUTOFF_FALLBACK=s1pen above (zero cost; acts only when the RR cutoff fires).
# [R86] 1 = after the pick, try turned copies of the final (PGH_ORIENT_FINAL_CANDS, code default y180,xm90,xp90); DINO
# margin gate + decisive judge win. 0 = off (= pgh17).
ENV PGH_ORIENT_FINAL="0"
# [R86] 1 = judge draws without a DINO signal / RR ties without the bracket winner -> DINO, then a sha256 coin (not seat A /
# lowest k = UtevoLux-leaning with the 0.5 interleave). 0 = pgh17.
ENV PGH_TIE_NEUTRAL="1"
# [R82] budget = equal coder tokens per photo (modules/metrics/budget_k.py) is a PGH_K_MODE value (above: fixed = off).
# [R82] 1 = finer time-guard cut steps [95,90,85,78,70,55,40,25]; 0 = pgh16/pgh17 steps.
ENV PGH_TIMEGUARD_FINE="0"
# runaway guard: 1 = a coder draft cut at max_tokens without a JS module is not re-asked
# (dropped as today's 'retries used up' ValueError). Anything but 1 = today's re-asks (byte-identical).
ENV RUNAWAY_GUARD="1"
# [sws] render speed (opt_r4/sws/README.md). Read at container start; docker run -e NAME=value, no rebuild.
# 1 = kernel-optim's sws-llvm SwiftShader (if installed above) | 0 = Chrome's bundled lib + no SwiftShader.ini (= pgh17)
ENV PGH_SWS="1"
# SwiftShader raster threads per Chrome (SwiftShader.ini next to the lib; with PGH_SWS=1 only); 0 = SwiftShader default 16
ENV PGH_SWS_THREADS="4"
# which sws lib with PGH_SWS=1 (both built from source in the "sws" stage): llvm = sws-llvm (gated, default) |
# cl4 = sws-llvm-cl4 (+ Renderer.hpp MaxClusterCount 4, MaxBatchSize 1024; less render CPU than llvm; opt-in only)
ENV PGH_SWS_LIB="llvm"
# object-page.html program pin (pipeline-speed's patch_object_page.py --pin snippet): 1 = on | 0 = off (= pgh17's page)
ENV PGH_RENDER_PIN="1"
# render slots "SIDECARS:PAGES:WORKERS" (pipeline_service/pgh_rslots.py) = renderer.sidecar_count : renderer.pool_size :
# actors.renderer.workers. "" = configuration.yaml (8:2:16, = pgh17). Proposal for a 40.8-CPU pod with sws: 12:2:24
ENV PGH_RENDER_SLOTS=""
# [gsplit] kernel-optim GPU split (pipeline_service/pgh_gsplit.py, kernel-optim analysis/gpusplit/RESULT.md): "1" = coder on
# GPUs 0,1,2 (dp 3, workers 192) + judge side (pipeline-speed); refused with '!!' while a judge still holds GPU 2. "0" = 2+2.
# [phung3] under PGH_MIX_CODERS the 3 coder GPUs are shared out per PGH_GPU_SPLIT_MIX (UtevoLux on 0, AmberKite DP2 on 1,2),
# judge on GPU 3, burst GLMs :8031-8033 in the tail. Refused -> one '!!' line and the huy3 layout (0 utevo, 1 ak, judges 2,3).
ENV PGH_GPU_SPLIT="1"
# [phung1] judge duel slots on the 1 judge GPU in the coder phase (24 -> 48; vLLM there ran mean 23, Waiting 0, KV<=11% at 24;
# 1-GPU bench 48 slots p95 19 s). Tail total stays 96 (burst extra = 24/GLM x 4 - 48 = 48).
ENV PGH_GPU_SPLIT_JUDGE_SLOTS="48"
# [phung3] coder GPUs per mix name in PGH_MIX_CODERS order, summing to 3 (GPUs 0,1,2 handed out in that order)
ENV PGH_GPU_SPLIT_MIX="utevo=2,ak=1"
