from __future__ import annotations

import asyncio
import gc
import time
import os

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai._constants import DEFAULT_CONNECTION_LIMITS as _OAI_LIMITS

from config.settings import LLMClientConfig, SettingsConf
from logger_config import logger
from llm.mm_resend import install_mm_resend  # judge image resend (tail/deep/gdnk/judge)
from modules.js_checker.module import JSCheckerModule
from modules.renderer.module import RendererModule
from pipeline.factory import build_pipeline
from pipeline.batch_stats import STATS
from pipeline.orchestrator import Pipeline
from pipeline.state import MinerState
from pipeline.task import PipelineTask
from llm import burst as _burst

_PLACEHOLDER_KEYS = {"", "placeholder", "your-key-here", "sk-or-...", "changeme"}
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


def _is_local_endpoint(base_url: str) -> bool:
    return any(h in base_url for h in _LOCAL_HOSTS)


def _resolve_api_key(name: str, cfg: LLMClientConfig) -> str | None:
    """Resolve api key for one client"""
    raw = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
    api_key = raw.strip().strip('"').strip("'")
    if not api_key:
        api_key = cfg.api_key.strip()

    if _is_local_endpoint(cfg.base_url):
        return api_key or "local"

    if api_key.lower() in _PLACEHOLDER_KEYS:
        env_hint = f" (env var {cfg.api_key_env})" if cfg.api_key_env else ""
        logger.warning(
            f"Skipping LLM client {name!r}{env_hint}: placeholder/empty API key "
        )
        return None
    return api_key


def _instrument_client(name: str, client: AsyncOpenAI) -> None:
    """Wrap `client.chat.completions.create` so every call (coder, judge, critic, probe) lands in STATS with its
    latency and usage. Measurement only: a failure to install the wrapper just logs, and the wrapper re-raises
    everything untouched (cancellations are not counted as errors)."""
    try:
        completions = client.chat.completions
        orig = completions.create

        async def create(*args, **kwargs):
            t0 = time.monotonic()
            ok = True
            usage = None
            record = True
            try:
                resp = await orig(*args, **kwargs)
                usage = getattr(resp, "usage", None)
                return resp
            except asyncio.CancelledError:
                record = False
                raise
            except Exception:
                ok = False
                raise
            finally:
                if record:
                    STATS.llm_call(name, time.monotonic() - t0, usage, ok)

        completions.create = create  # type: ignore[method-assign]
    except Exception as exc:
        logger.warning(f"[diag] could not instrument llm client {name}: {exc!r}")


class GenerationPipeline:
    """Top-level pipeline driver. Constructed once per app lifecycle."""

    def __init__(self, settings: SettingsConf, state: MinerState) -> None:
        self.settings = settings
        self.state = state

        # DET modules
        self.js_checker = JSCheckerModule(settings.js_checker)
        self.renderer = RendererModule(settings.renderer)

        self._clients: dict[str, AsyncOpenAI] = {}
        self._client_urls: dict[str, str] = {}
        self._http_client: httpx.AsyncClient | None = None
        self._pipeline: Pipeline | None = None

    # Lifecycle

    async def startup(self) -> None:
        logger.info("GenerationPipeline starting")

        _judge = self.settings.actors.judge  # judge image resend (tail/deep/gdnk/judge)
        _judge_names = {_judge.client, *getattr(_judge, "extra_clients", [])}
        for name, cfg in self.settings.llm_clients.items():
            if not cfg.enabled:
                continue
            api_key = _resolve_api_key(name, cfg)
            if api_key is None:
                continue
            self._clients[name] = AsyncOpenAI(
                base_url=cfg.base_url,
                api_key=api_key,
                # Per-endpoint read timeout (llm_clients.<name>.request_timeout_s). In-flight
                # requests are capped by the actor worker semaphores, so this measures service
                # time, not queue time; a hung vLLM request costs one candidate / one duel retry,
                # not the prompt or the batch. Connect timeout 120 s so judge bursts retry.
                timeout=httpx.Timeout(cfg.request_timeout_s, connect=120.0),
                max_retries=0,
                # [patch_loop] drop idle connections before the server's ~5 s keep-alive close (stale-connection race)
                # (openai 3.x runs on httpx2: build its own Limits type; openai's default pool sizes kept)
                # OPT-IN (env JUDGE_KEEPALIVE2=1): fewer failed first attempts = fewer re-samples, so verdicts change
                **({"http_client": DefaultAsyncHttpxClient(limits=type(_OAI_LIMITS)(
                    max_connections=1000, max_keepalive_connections=100, keepalive_expiry=2.0))}
                   if __import__("os").environ.get("JUDGE_KEEPALIVE2", "0") == "1" else {}),
            )
            logger.info(
                f"llm client ready | name={name} base_url={cfg.base_url} "
                f"local={_is_local_endpoint(cfg.base_url)} timeout={cfg.request_timeout_s:.0f}s"
            )
            self._client_urls[name] = cfg.base_url
            if name in _judge_names and install_mm_resend(self._clients[name], name=name):  # judge image resend (tail/deep/gdnk/judge)
                logger.info(f"llm client {name}: judge images sent once, then by uuid (JUDGE_IMG_UUID=0 disables)")
            _instrument_client(name, self._clients[name])
        actors = self.settings.actors
        _roles: dict[str, str] = {}
        for _c, _r in ((actors.planner.client, "planner"), (actors.critic.client, "critic"),
                       (actors.judge.client, "judge"), (actors.coder.client, "coder")):  # shared client names: coder/judge win
            if _c:
                _roles[_c] = _r
        STATS.set_roles(_roles)
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=50)
        self._http_client = httpx.AsyncClient(timeout=30.0, limits=limits)

        # [pgh1v3] PGH_K_MODE fixed|adaptive|rule: ensemble_size from PGH_K40 / the CPU quota (modules/metrics/
        # adaptive_k.py) before build_pipeline reads it. off / PGH_ADAPTIVE_K=0 (or any failure) = the configured K = pgh1v2.
        if os.environ.get("PGH_ADAPTIVE_K", "1").strip() == "1" and os.environ.get("PGH_K_MODE", "fixed").strip().lower() != "off":
            try:
                from modules.metrics import adaptive_k as _adaptive_k
                _adaptive_k.apply(self.settings, diag=self.state.diag, log=logger.info)
            except Exception as exc:  # never block start-up
                logger.warning(f"[ADAPTIVE_K] failed, keeping ensemble_size="
                               f"{self.settings.actors.coder.ensemble_size}: {exc!r}")
        await self.js_checker.startup()
        await self.renderer.startup()
        self._pipeline = build_pipeline(
            settings=self.settings,
            clients=self._clients,
            js_checker=self.js_checker,
            renderer=self.renderer,
            http_client=self._http_client,
        )
        if os.environ.get("PGH_MIX_CODERS", "").strip().lower() not in ("", "0", "off", "false", "no"):
            # [mix] R71 (pgh_mix.py): drafts split between 2 coder clients; invalid / failure -> the single coder
            try:
                import pgh_mix as _pgh_mix
                from modules.scene_coder.agent import SceneCoderAgent as _SCA
                from pipeline.state import ServiceState as _SS
                if _pgh_mix.init_from_env_and_cfg(__import__("config.settings", fromlist=["data_yaml"]).data_yaml, log=logger.warning) is not None:
                    _pgh_mix.READY = _pgh_mix.Readiness(_pgh_mix.SPEC, log=logger.info)
                    _pgh_mix.build_router(self._pipeline, self.settings, self._clients, _SCA, self.state, _SS.RUNNING,
                                          log=logger.info, warn=logger.warning)
                    for _c in _pgh_mix.SPEC.clients:
                        STATS.roles[_c] = "coder"
                    logger.info(_pgh_mix.SPEC.describe())
            except Exception as exc:  # never block start-up: the single coder stays
                logger.warning(f"[MIX] !! router setup failed, single coder kept: {exc!r}")
                try:
                    _pgh_mix.READY = None
                    _pgh_mix.SPEC = None
                except Exception:
                    pass
        await self._pipeline.start()
        actors = self.settings.actors
        _jname = actors.judge.client
        _jcfg = self.settings.llm_clients.get(_jname)

        def _burst_client(name: str, url: str) -> AsyncOpenAI:
            c = AsyncOpenAI(base_url=url, api_key=_resolve_api_key(_jname, _jcfg) or "local",
                            timeout=httpx.Timeout(_jcfg.request_timeout_s, connect=120.0), max_retries=0)
            if install_mm_resend(c, name=name):  # judge image resend (tail/deep/gdnk/judge) (burst GLM replica)
                logger.info(f"llm client {name}: burst judge images sent once, then by uuid")
            _instrument_client(name, c)
            self._clients[name] = c
            self._client_urls[name] = url
            STATS.roles[name] = "judge"
            return c

        if self._pipeline.judge is not None and _jcfg is not None:
            _burst.setup(self._pipeline, self._pipeline.judge.client, _burst_client)
        eb = self.settings.event_bus
        use_planner = self.settings.pipeline.use_planner
        logger.info(f"[Pipeline initialized — waiting for models]")
        if use_planner:
            logger.info(f"Planner: {actors.planner.client} | Model: {actors.planner.model}")
        else:
            logger.info("Planner: DISABLED (pipeline.use_planner=false — pure-image coder)")
        logger.info(
            f"Coder: {actors.coder.client} | Model: {actors.coder.model} | "
            f"multimodal={actors.coder.multimodal} | ensemble_size={actors.coder.ensemble_size} | "
            f"ensemble_temperature={actors.coder.ensemble_temperature}"
        )
        logger.info(f"Critic: {actors.critic.client} | Model: {actors.critic.model}")
        if actors.coder.ensemble_size > 1:
            logger.info(f"Judge: {actors.judge.client} | Model: {actors.judge.model}")
        else:
            logger.info("Judge: DISABLED (ensemble_size=1)")
        logger.info(f"Iter cap: {eb.max_iter} | Deadline: {eb.task_deadline_s:.0f}s | Threshold: {eb.score_threshold:.2f}")

    async def shutdown(self) -> None:
        logger.info("[Pipeline shutting down]")
        if self._pipeline is not None:
            await self._pipeline.stop()
        await self.renderer.shutdown()
        await self.js_checker.shutdown()
        if self._http_client is not None:
            await self._http_client.aclose()
        for name, client in list(self._clients.items()):
            try:
                await client.close()
            except Exception as exc:
                logger.warning(f"[Pipeline shutdown] Client {name} close failed: {exc}")

    def _cleanup_batch_memory(self, context: str) -> None:
        if self._pipeline is None:
            return
        sessions = self._pipeline.session_store.clear()
        statuses = len(self._pipeline.task_status)
        self._pipeline.task_status.clear()
        collected = gc.collect()
        logger.info(
            f"[{context} cleanup] sessions={sessions} "
            f"task_status={statuses} gc_collected={collected}"
        )

    async def run_batch(self, tasks: list[PipelineTask]) -> None:
        if self._pipeline is None:
            logger.error("[Batch failed] Run called before startup")
            return
        if tasks:
            batch_seed = tasks[0].seed
            if self._pipeline.planner is not None:
                self._pipeline.planner.seed = batch_seed
            self._pipeline.coder.seed = batch_seed
            self._pipeline.critic.seed = batch_seed
            if self._pipeline.judge is not None:
                self._pipeline.judge.seed = batch_seed
            planner_seed = self._pipeline.planner.seed if self._pipeline.planner is not None else "n/a"
            judge_seed = self._pipeline.judge.seed if self._pipeline.judge is not None else "n/a"
            logger.info(
                f"[Batch seed] Agents updated | seed={batch_seed} "
                f"(planner={planner_seed}, coder={self._pipeline.coder.seed}, "
                f"critic={self._pipeline.critic.seed}, judge={judge_seed})"
            )
            
        budget = self.settings.pipeline.batch_time_budget

        current: dict[str, PipelineTask] = {}   # [R96 pgh17d] the attempt in flight per stem (a retry is a new object)

        async def run_one(task: PipelineTask) -> None:
            while True:
                current[task.stem] = task
                result = await (await self._pipeline.submit(task))
                # Three attempts per prompt: a missing module is scored as a loss. A deadline
                # failure is not retried: the retry would start a full K-candidate run after
                # task_deadline_s and could only end at the batch budget.
                if not result.failed or task.attempt >= 2 or result.failure_stage == "deadline":
                    self.state.record_task(result)
                    if _burst.CTRL:
                        _burst.CTRL.task_done(task.stem)
                    return
                task = PipelineTask(
                    stem=task.stem, image_url=task.image_url, seed=task.seed,
                )
                task.attempt = result.attempt + 1
                logger.info(f"[Batch retry] {task.stem} | attempt={task.attempt}")

        _k_start = self.settings.actors.coder.ensemble_size
        try:   # [R96 pgh17d] PGH_K_CPU_AWARE: CPU-quota K prior (before the time guard, which then starts from it)
            import pgh_kcpu as _kcpu
            if _kcpu.ON:
                _k_start = _kcpu.KCPU.batch_begin(_k_start, n_tasks=len(tasks), log=logger.info)
        except Exception as exc:  # never block a batch
            logger.warning(f"[KCPU] batch_begin failed: {exc!r}")
        if os.environ.get("PGH_K_MODE", "").strip().lower() == "budget":   # [R82 kbudget] per-photo K reference: batch start
            try:
                from modules.metrics import budget_k as _bk
                _bk.ALLOC.batch_begin(self.settings.actors.coder.ensemble_size)
                logger.info(f"[KBUDGET] batch {_bk.ALLOC.batch} begin | ref photos={len(_bk.ALLOC.ref)} | "
                            f"params={_bk.ALLOC.p}{(' | ' + _bk.NOTE) if _bk.NOTE else ''}")
            except Exception as exc:
                logger.warning(f"[KBUDGET] batch_begin failed: {exc!r}")
        if os.environ.get("PGH_RR8"):   # [R84 jalign] batch order for the top-8 round robin (default off)
            try:
                import pgh_jalign as _ja
                _ja.batch_order([t.stem for t in tasks])
            except Exception as exc:  # never block a batch
                logger.warning(f"[JALIGN] batch_order failed: {exc!r}")
        if os.environ.get("PGH_SHADOW"):   # [opt_r5 shadow] batch order + slack deadline for the shadow picks (default off)
            try:
                import pgh_shadow as _shadow
                if _shadow.on():
                    _shadow.batch_begin([t.stem for t in tasks], budget)
            except Exception as exc:  # never block a batch
                logger.warning(f"[SHADOW] batch_begin failed: {exc!r}")
        # [pgh1v3] PGH_TIMEGUARD: batch-boundary round guard decides this batch's K before any coder starts
        if os.environ.get("PGH_TIMEGUARD", "0").strip().lower() not in ("", "0", "off"):
            try:
                import pgh_timeguard as _tg
                _tg.GUARD.batch_begin(_k_start, n_tasks=len(tasks), log=logger.info)
            except Exception as exc:  # never block a batch
                logger.warning(f"[TIMEGUARD] batch_begin failed: {exc!r}")
        if _burst.CTRL:
            _burst.CTRL.batch_start(len(tasks), [t.stem for t in tasks])
        await self._diag_snapshot("start")
        try:
            logger.info(
                f"[Batch starting] {len(tasks)} tasks | budget={budget:.0f}s "
                f"(effective {budget - 120:.0f}s) | task_deadline={self.settings.event_bus.task_deadline_s:.0f}s "
                f"| candidate_deadline={self.settings.pipeline.candidate_deadline_s:.0f}s"
            )
            await asyncio.wait_for(
                asyncio.gather(*(run_one(t) for t in tasks)),
                timeout=budget-120,
            )
            
        except asyncio.TimeoutError:
            logger.warning(f"[Batch deadline] hit {budget:.0f}s")
            try:   # [R96 pgh17d] PGH_DEADLINE_SALVAGE=1: unfinished prompts ship their best draft so far
                import pgh_salvage as _salv
                if _salv.ON:
                    for task in tasks:
                        recorded = self.state.tasks.get(task.stem)
                        cur = current.get(task.stem, task)
                        # only prompts CUT by a deadline (still running, or recorded as a task-deadline cut); a prompt
                        # that failed earlier for another reason stays failed; one [SALVAGE] line per task object
                        cut = recorded is None or (recorded.failed and recorded.failure_stage == "deadline")
                        if cut and not _salv.tried(cur, recorded):
                            if _salv.salvage(cur, "batch_deadline", log=logger.warning):
                                self.state.record_task(cur)
            except Exception as exc:
                logger.warning(f"[SALVAGE] batch hook failed: {exc!r}")
            for task in tasks:
                recorded = self.state.tasks.get(task.stem)
                if recorded is None or recorded.failed or not recorded.js_code:
                    task.failed = True
                    task.failure_reason = f"batch budget exceeded ({budget:.0f}s)"
                    task.failure_stage = "batch_deadline"
                    self.state.record_task(task)
            
        except asyncio.CancelledError:
            raise
        finally:
            if os.environ.get("PGH_TIMEGUARD", "0").strip().lower() not in ("", "0", "off"):  # [pgh1v3] PGH_TIMEGUARD: record this batch's wall
                try:
                    import pgh_timeguard as _tg
                    _tg.GUARD.batch_end(log=logger.info)
                except Exception as exc:
                    logger.warning(f"[TIMEGUARD] batch_end failed: {exc!r}")
            try:   # [R96 pgh17d] PGH_K_CPU_AWARE: brake summary (after the guard's batch_end, which stays first)
                import pgh_kcpu as _kcpu
                if _kcpu.ON:
                    _kcpu.KCPU.batch_end(log=logger.info)
            except Exception as exc:
                logger.warning(f"[KCPU] batch_end failed: {exc!r}")
            if os.environ.get("PGH_K_MODE", "").strip().lower() == "budget":   # [R82 kbudget] this batch joins the reference
                try:
                    from modules.metrics import budget_k as _bk
                    _bk.ALLOC.batch_end()
                except Exception as exc:
                    logger.warning(f"[KBUDGET] batch_end failed: {exc!r}")
            if os.environ.get("PGH_SHADOW"):   # [opt_r5 shadow] cancel unfinished shadow jobs (PGH_SHADOW_DRAIN_S, default 0 s; after the guard's batch_end)
                try:
                    import pgh_shadow as _shadow
                    await _shadow.batch_end()
                except Exception as exc:  # never fail a batch
                    logger.warning(f"[SHADOW] batch_end failed: {exc!r}")
            STATS.end()
            if _burst.CTRL:
                _burst.CTRL.batch_end()
            await self._diag_snapshot("end")
            self._cleanup_batch_memory("Batch")
            logger.info(
                f"[Batch done] {len(self.state.results)} ok, "
                f"{len(self.state.failed)} failed | diag: {' '.join(STATS.header_parts())}"
            )

    async def _diag_snapshot(self, when: str) -> None:
        """Batch-start / batch-end counters for the miner-diag header: local vLLM servers' /metrics + cgroup cpu.stat."""
        try:
            STATS.cgroup_snapshot(when)
            if self._http_client is None:
                return
            local = [(n, u) for n, u in self._client_urls.items() if n in self._clients and _is_local_endpoint(u)]
            if local:
                await asyncio.wait_for(
                    asyncio.gather(*(STATS.vllm_snapshot(n, u, when, self._http_client) for n, u in local),
                                   return_exceptions=True),
                    timeout=12.0,
                )
        except Exception as exc:
            logger.debug(f"[diag] snapshot {when} skipped: {exc!r}")
    
    async def run_coder_probe(self) -> float | None:
        """Measure the coder endpoint's aggregate output tok/s with `concurrency` text-only code requests.

        Returns tok/s (None when disabled / no client / every request failed). Records
        state.diag['probe_tps'] and friends. The batch wall-clock is coder tokens / tok/s, so this
        number predicts the audit fit; the caller decides REPLACE vs continue.
        """
        cfg = self.settings.pipeline.coder_probe
        if not cfg.enabled:
            return None
        coder = self.settings.actors.coder
        client = self._clients.get(coder.client or "")
        if client is None or not coder.model:
            logger.warning("[Coder probe] no coder client — skipped")
            return None
        prompt = (
            "Write a complete Three.js ES module `export default function generate(THREE)` that builds a detailed "
            "wooden park bench with slats, armrests and cast-iron legs, centered and fitted into a unit cube. "
            "Use only geometry and materials, no textures. Output only the code."
        )
        think = coder.enable_thinking if coder.enable_thinking is not None else bool(coder.reasoning_effort)

        async def one(i: int) -> int:
            resp = await client.chat.completions.create(
                model=coder.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=cfg.max_tokens,
                temperature=coder.ensemble_temperature,
                seed=10_000 + i,
                extra_body={"chat_template_kwargs": {"enable_thinking": think}},
            )
            usage = getattr(resp, "usage", None)
            return int(getattr(usage, "completion_tokens", 0) or 0)

        # Untimed warm-up burst first (same shape as the timed one): on a fresh pod the first concurrent burst still
        # pays FlashInfer/Triton JIT and CUDA-graph capture for the batch-48 shapes — a single small warm-up request
        # was not enough (fresh pod: 2481 tok/s after a 64-token warm-up vs 3146 with a warm JIT cache; cold 2116).
        t_w = time.monotonic()
        try:
            warm = await asyncio.wait_for(
                asyncio.gather(*(one(i) for i in range(cfg.concurrency)), return_exceptions=True),
                timeout=cfg.timeout_s,
            )
            w_toks = sum(r for r in warm if isinstance(r, int)); w_wall = max(time.monotonic() - t_w, 1e-3)
            logger.info(f"[Coder probe] warm-up burst: {w_toks} tokens in {w_wall:.1f}s = {w_toks / w_wall:.0f} tok/s (not scored)")
            self.state.diag["probe_tps_cold"] = round(w_toks / w_wall)
        except Exception as exc:
            logger.warning(f"[Coder probe] warm-up burst failed: {exc!r} — timing the probe anyway")
        t0 = time.monotonic()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(one(i) for i in range(cfg.concurrency)), return_exceptions=True),
                timeout=cfg.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.error(f"[Coder probe] timed out after {cfg.timeout_s:.0f}s")
            self.state.diag["probe_tps"] = 0.0
            return 0.0
        wall = max(time.monotonic() - t0, 1e-3)
        toks = sum(r for r in results if isinstance(r, int))
        errs = [r for r in results if not isinstance(r, int)]
        tps = toks / wall
        self.state.diag.update({
            "probe_tps": round(tps), "probe_tokens": toks, "probe_wall_s": round(wall, 1),
            "probe_conc": cfg.concurrency, "probe_errors": len(errs),
        })
        logger.info(
            f"[Coder probe] {cfg.concurrency} x {cfg.max_tokens} tok: {toks} output tokens in {wall:.1f}s "
            f"= {tps:.0f} tok/s | errors={len(errs)} | min_tps={cfg.min_tps:.0f}"
            + (f" | first error: {errs[0]!r}" if errs else "")
        )
        if errs and not toks:
            return None
        return tps

    async def run_warmup(self, task: PipelineTask) -> None:
        if self._pipeline is None:
            logger.error("[Warmup failed] Run called before startup")
            return
        try:
            logger.info(f"[Warmup task starting]")
            if _burst.CTRL:
                _burst.CTRL.batch_start(1, [task.stem])
            future = await self._pipeline.submit(task)
            result = await future
            if getattr(result, "failed", False):
                raise RuntimeError(
                    f"warmup produced no module: {getattr(result, 'failure_reason', 'unknown')}"
                )
            logger.info(f"[Warmup task completed]")
        except asyncio.CancelledError:
            logger.info("[Warmup task cancelled]")
            raise
        except Exception as exc:
            # Never fall through to READY on a failed warmup: the caller keeps the pod at
            # warming_up and retries, which beats advertising a pod that cannot generate.
            logger.exception(f"[Warmup failed] {exc}")
            raise
        finally:
            if _burst.CTRL:
                _burst.CTRL.batch_end()
                await _burst.CTRL.ensure_coder("warm-up end")   # untimed: batch 1 starts with the coder awake
            self._cleanup_batch_memory("Warmup")
