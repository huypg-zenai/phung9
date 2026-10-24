from __future__ import annotations

import asyncio
import base64
import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from logger_config import logger
from modules.base import BaseModule
from modules.renderer.grid import GRID_KEYS, GRID_VIEWS, compose_grid
from modules.renderer.settings import (
    JUDGE_GRID_BG, JUDGE_GRID_LIGHTING, JUDGE_GRID_TILE, RendererConfig, read_judge_grid,
)
from modules.renderer.render_timeouts import ENV_NAME as _RT_ENV, effective_timeouts as _rt_effective
from pipeline.batch_stats import STATS
from pipeline.task import PipelineTask

_RUNNER_JS = Path(__file__).parent / "render_service" / "render_runner.mjs"

_RETRYABLE_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)


@dataclass
class _Sidecar:
    idx: int
    port: int
    proc: asyncio.subprocess.Process
    client: httpx.AsyncClient
    log_tasks: list[asyncio.Task] = field(default_factory=list)
    browser_pid: int | None = None


@dataclass
class _Slot:
    idx: int
    sidecar: _Sidecar | None = None
    task: asyncio.Task | None = None


class _SpecResp:
    """[patch_loop] edit 10: the final views of a speculative combined request (render_views_spec), replayed through
    _render_and_fill's normal response handling as if the final request had just returned them."""
    status_code = 200
    text = ""

    def __init__(self, views: dict, lightings_applied: bool = False) -> None:
        self._views = views
        self._applied = lightings_applied   # [jgrid] F4: the speculative request's lightings_applied flag

    def json(self) -> dict:
        return {"views": self._views, "lightings_applied": True} if self._applied else {"views": self._views}


class RendererModule(BaseModule):
    _PING_INTERVAL_S = 5.0
    _PING_TIMEOUT_S = 5.0
    _PING_STRIKES = 3
    _SPAWN_BACKOFF_INITIAL_S = 1.0
    _SPAWN_BACKOFF_MAX_S = 30.0
    _TERM_GRACE_S = 5.0
    _HEALTHY_UPTIME_RESET_S = 60.0

    def __init__(self, config: RendererConfig) -> None:
        # [r42] PGH_RENDER_TIMEOUT_S (render_timeouts.py): unset/0 = the config as-is (same object, old behaviour)
        _rt, _rt_note = _rt_effective(config.render_timeout_ms, config.protocol_timeout_ms,
                                      config.request_timeout_s, os.environ.get(_RT_ENV))
        if _rt_note:
            (logger.warning if _rt_note.startswith("!!") else logger.info)(f"[RENDERER] {_rt_note}")
        if _rt != {"render_timeout_ms": config.render_timeout_ms, "protocol_timeout_ms": config.protocol_timeout_ms,
                   "request_timeout_s": config.request_timeout_s}:
            config = config.model_copy(update=_rt)
        self.config = config
        self._slots: list[_Slot] = []
        self._dispatch_counter = 0
        self._node_cwd: str | None = None
        self._shutting_down = False
        # [jgrid] JUDGE_GRID read once here (off | small | contest); unknown -> off + one warning
        self._jgrid, _warn = read_judge_grid()
        if _warn:
            logger.warning(f"[RENDERER] {_warn}")
        elif self._jgrid != "off":
            logger.info(f"[RENDERER] JUDGE_GRID={self._jgrid}: judge grid {JUDGE_GRID_TILE} px tiles"
                        + (f", {JUDGE_GRID_LIGHTING} light, bg {JUDGE_GRID_BG}" if self._jgrid == "contest" else ""))
        self._jgrid_code_warned = False
        self._jgrid_noapply = 0          # [jgrid] F4: contest grid views rendered by JS that ignored per-view lighting

    async def startup(self) -> None:
        if not _RUNNER_JS.exists():
            logger.warning(f"[RENDERER] runner missing at {_RUNNER_JS}, disabling")
            return

        self._node_cwd = self._resolve_node_cwd()
        count = max(1, self.config.sidecar_count)
        logger.info(
            f"[RENDERER] starting {count} sidecar(s) | node={self.config.node_binary} "
            f"runner={_RUNNER_JS} base_port={self.config.sidecar_port} cwd={self._node_cwd} "
            f"pool_size_per_sidecar={self.config.pool_size}"
        )

        self._slots = [_Slot(idx=i) for i in range(count)]
        for slot in self._slots:
            slot.task = asyncio.create_task(self._supervise(slot))

        deadline = time.monotonic() + self.config.startup_timeout_s + 30.0
        while time.monotonic() < deadline:
            ready = sum(1 for s in self._slots if s.sidecar is not None)
            if ready:
                logger.info(
                    f"[RENDERER] {ready}/{count} sidecar(s) ready "
                    f"(remaining come up in background), slots per sidecar={self.config.pool_size}"
                )
                return
            await asyncio.sleep(0.5)

        await self.shutdown()
        raise RuntimeError(
            f"[RENDERER] no sidecar became ready within {self.config.startup_timeout_s + 30.0:.0f}s"
        )

    async def _supervise(self, slot: _Slot) -> None:
        backoff = self._SPAWN_BACKOFF_INITIAL_S
        spawned = 0
        while not self._shutting_down:
            try:
                sc = await self._spawn_sidecar(slot.idx)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    f"[RENDERER] sidecar #{slot.idx} spawn failed: {exc} — retry in {backoff:.0f}s"
                )
                STATS.sidecar_spawn_fail()
                await asyncio.sleep(backoff + random.uniform(0.0, 0.5))
                backoff = min(backoff * 2.0, self._SPAWN_BACKOFF_MAX_S)
                continue

            if spawned:
                STATS.sidecar_restart()  # a re-spawn after a monitored death (miner-diag `render=.../restarts`)
            spawned += 1
            ready_at = time.monotonic()
            slot.sidecar = sc
            try:
                await self._monitor(sc)
            finally:
                slot.sidecar = None
                await self._teardown_sidecar(sc)

            if time.monotonic() - ready_at > self._HEALTHY_UPTIME_RESET_S:
                backoff = self._SPAWN_BACKOFF_INITIAL_S

    async def _spawn_sidecar(self, idx: int) -> _Sidecar:
        port = self.config.sidecar_port + idx
        static_port = self.config.static_port_base + idx
        env = {
            **os.environ,
            "PORT": str(port),
            "STATIC_PORT": str(static_port),
            "RENDER_POOL_SIZE": str(self.config.pool_size),
            "RENDER_TIMEOUT_MS": str(self.config.render_timeout_ms),
            "PROTOCOL_TIMEOUT_MS": str(self.config.protocol_timeout_ms),
            "PREWARM_LIGHTING": self.config.lighting.value,
        }
        proc = await asyncio.create_subprocess_exec(
            self.config.node_binary,
            str(_RUNNER_JS),
            env=env,
            cwd=self._node_cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        stdout_task = asyncio.create_task(
            self._pipe_logger(proc.stdout, f"#{idx}/stdout")
        )
        stderr_task = asyncio.create_task(
            self._pipe_logger(proc.stderr, f"#{idx}/stderr")
        )
        client = httpx.AsyncClient(timeout=self.config.request_timeout_s)
        try:
            browser_pid = await self._wait_ready_for(proc, client, port, idx)
        except BaseException:
            await self._kill_proc_group(proc, idx)
            for t in (stdout_task, stderr_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            try:
                await client.aclose()
            except Exception:
                pass
            raise
        logger.info(f"[RENDERER] sidecar #{idx} ready on port {port}")
        return _Sidecar(idx=idx, port=port, proc=proc, client=client,
                        log_tasks=[stdout_task, stderr_task],
                        browser_pid=browser_pid)

    async def _monitor(self, sc: _Sidecar) -> None:
        """Return when the sidecar is dead: process exited or pings starve."""
        ping_url = f"http://{self.config.sidecar_host}:{sc.port}/ping"
        strikes = 0
        while True:
            try:
                await asyncio.wait_for(sc.proc.wait(), timeout=self._PING_INTERVAL_S)
                logger.warning(
                    f"[RENDERER] sidecar #{sc.idx} process exited (rc={sc.proc.returncode})"
                )
                return
            except asyncio.TimeoutError:
                pass
            try:
                r = await sc.client.get(ping_url, timeout=self._PING_TIMEOUT_S)
                ok = r.status_code == 200
            except Exception:
                ok = False
            if ok:
                strikes = 0
                sc.browser_pid = self._parse_browser_pid(r) or sc.browser_pid
                continue
            strikes += 1
            if strikes >= self._PING_STRIKES:
                logger.warning(
                    f"[RENDERER] sidecar #{sc.idx} failed {strikes} consecutive pings — restarting"
                )
                return

    async def _kill_proc_group(
        self, proc: asyncio.subprocess.Process, idx: int, browser_pid: int | None = None
    ) -> None:
        try:
            proc.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._TERM_GRACE_S)
        except asyncio.TimeoutError:
            pass
        for pgid, label in ((proc.pid, "node"), (browser_pid, "browser")):
            if pgid is None:
                continue
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                logger.warning(f"[RENDERER] sidecar #{idx} killpg({label}) denied: {exc}")
        if proc.returncode is None:
            await proc.wait()

    async def _teardown_sidecar(self, sc: _Sidecar) -> None:
        """Idempotent: every step tolerates an already-dead process, an
        already-closed client and already-finished log pumps."""
        try:
            await sc.client.aclose()
        except Exception:
            pass
        await self._kill_proc_group(sc.proc, sc.idx, sc.browser_pid)
        # Log pumps last, so the sidecar's dying words reach the logs.
        for t in sc.log_tasks:
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    async def shutdown(self) -> None:
        self._shutting_down = True
        tasks = [s.task for s in self._slots if s.task is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for slot in self._slots:
            sc = slot.sidecar
            slot.sidecar = None
            slot.task = None
            if sc is not None:
                await self._teardown_sidecar(sc)
        self._slots = []

    def _next_sidecar(self, exclude: int | None = None) -> _Sidecar | None:
        live = [s.sidecar for s in self._slots if s.sidecar is not None]
        if not live:
            return None
        inflight = self.__dict__.setdefault("_inflight", {})   # [patch_loop] least-loaded dispatch
        start = self._dispatch_counter % len(live)
        order = live[start:] + live[:start]
        if exclude is not None and len(order) > 1:   # a retry never goes back to the sidecar that just failed
            order = [s for s in order if s.idx != exclude] or order
        sc = min(order, key=lambda s: inflight.get(s.idx, 0))
        self._dispatch_counter += 1
        return sc

    def _inflight_release(self, idx: int, failed: bool) -> None:
        # [patch_loop] a failed post (exception or HTTP 5xx) keeps counting as busy for 5 s, so a fast-failing
        # sidecar does not look idle and pull the traffic
        if failed:
            asyncio.get_running_loop().call_later(5.0, self._inflight_release, idx, False)
        else:
            self._inflight[idx] -= 1

    async def _post_with_retry(self, path: str, payload: dict) -> tuple[httpx.Response, int]:
        """POST to a live sidecar; on a dead-channel error retry once,
        preferring a different sidecar. Renders are idempotent, so with a
        single live sidecar retrying on the same one still beats failing.

        The try covers only the post itself — HTTP-status errors raised by
        callers after this returns are never retried.
        """
        sc = self._next_sidecar()
        if sc is None:
            raise RuntimeError("no sidecars available")
        try:
            self._inflight[sc.idx] = self._inflight.get(sc.idx, 0) + 1
            _failed = True
            try:
                resp = await sc.client.post(
                    f"http://{self.config.sidecar_host}:{sc.port}{path}", json=payload
                )
                _failed = resp.status_code >= 500
            except asyncio.CancelledError:
                _failed = False
                raise
            finally:
                self._inflight_release(sc.idx, _failed)
            return resp, sc.idx
        except (*_RETRYABLE_ERRORS, RuntimeError) as exc:
            first_idx = sc.idx
            retry_sc = self._next_sidecar(exclude=first_idx)
            if retry_sc is not None and retry_sc.idx == first_idx:
                alt = self._next_sidecar()
                if alt is not None and alt.idx != first_idx:
                    retry_sc = alt
            if retry_sc is None:
                raise RuntimeError("no sidecars available") from exc
            logger.warning(
                f"[RENDERER] {path} on sidecar #{first_idx} failed "
                f"({type(exc).__name__}: {exc}) — retrying on sidecar #{retry_sc.idx}"
            )
            self._inflight[retry_sc.idx] = self._inflight.get(retry_sc.idx, 0) + 1
            _failed = True
            try:
                resp = await retry_sc.client.post(
                    f"http://{self.config.sidecar_host}:{retry_sc.port}{path}", json=payload
                )
                _failed = resp.status_code >= 500
            except asyncio.CancelledError:
                _failed = False
                raise
            finally:
                self._inflight_release(retry_sc.idx, _failed)
            return resp, retry_sc.idx

    async def process(self, task: PipelineTask) -> PipelineTask:
        """Render the 2x2 grid (4 orbit views composed in Python)."""
        await self._render_and_fill(task, self._grid_specs())
        return task

    async def process_with_judge_views(
        self, task: PipelineTask
    ) -> tuple[dict[str, bytes], dict[str, bytes]]:
        """Grid + judge views in one sidecar request; fills the same task fields
        as ``process()`` and returns ``(white_views, gray_views)``.
        All-or-nothing: on failure both come back empty."""
        specs = self._grid_specs()
        specs += [{"name": n, "bg": self.config.judge_white_bg} for n in self._JUDGE_WHITE_VIEWS]
        specs.append({
            "name": self._JUDGE_GRAY_VIEW,
            "bg": self.config.judge_gray_bg,
            "key": self._JUDGE_GRAY_KEY,
        })
        rendered = await self._render_and_fill(task, specs)

        gray_png = rendered.pop(self._JUDGE_GRAY_KEY, None)
        gray = {self._JUDGE_GRAY_VIEW: gray_png} if gray_png else {}
        return rendered, gray

    async def render_views(
        self, task: PipelineTask, views: list[str], *, img_size: int
    ) -> dict[str, bytes]:
        """Render named preset views only, at ``img_size``, on the judge's white background (orientation probe).
        No grid, no mutation of the task; returns an empty dict on any failure (callers treat that as "no decision")."""
        if task.failed or (not task.js_code and not task.scene_json):
            return {}
        payload: dict = {
            "views": [{"name": v, "bg": self.config.judge_white_bg} for v in views],
            "options": {"imgSize": img_size, "lighting": self.config.lighting.value},
        }
        if task.scene_json is not None:
            payload["object"] = task.scene_json
        else:
            payload["source"] = task.js_code
        try:
            resp, sidecar_idx = await self._post_with_retry("/render/views", payload)
        except Exception as exc:  # a probe never fails its caller
            logger.warning(f"[RENDERER] '{task.stem}' probe FAIL (http) | {type(exc).__name__}: {exc}")
            return {}
        if resp.status_code != 200:
            logger.warning(f"[RENDERER] '{task.stem}' probe FAIL (status) sidecar=#{sidecar_idx} | HTTP {resp.status_code}")
            return {}
        rendered: dict[str, bytes] = {}
        for name, b64 in (resp.json().get("views") or {}).items():
            try:
                rendered[name] = base64.b64decode(b64)
            except Exception as exc:  # skip a single bad view, keep the rest
                logger.warning(f"[RENDERER] probe view {name!r} decode failed: {exc}")
        return rendered

    async def render_views_spec(
        self, task: PipelineTask, views: list[str], *, img_size: int
    ) -> dict[str, bytes]:
        """[patch_loop] edit 10 (SPEC_FINAL=1; called by orient_task inside the probe's render slot). ONE request = the
        probe views as render_views sends them (+ "size", key "probe_<view>": right/back/left/top_down are final view
        names too) + the final views exactly as process_with_judge_views builds them, at options.imgSize. Returns the
        probe views like render_views; the final views are kept on the task (single use, dropped by orient_task unless
        the decision is "no rotation") for _render_and_fill. Any failure -> the leader's own probe request."""
        task.__dict__.pop("_spec_final", None)
        if self.__dict__.get("_spec_broken") or task.failed or task.scene_json is None:
            return await self.render_views(task, views, img_size=img_size)
        finals = self._spec_final_specs()
        payload: dict = {
            "views": [{"name": v, "bg": self.config.judge_white_bg, "key": f"probe_{v}", "size": img_size}
                      for v in views] + finals,
            "options": {"imgSize": self.config.img_size, "lighting": self.config.lighting.value},
            "object": task.scene_json,
        }
        why = None
        try:
            resp, sidecar_idx = await self._post_with_retry("/render/views", payload)
            if resp.status_code != 200:
                why = f"HTTP {resp.status_code}"
            else:
                got = resp.json().get("views") or {}
                probe = {v: base64.b64decode(got[f"probe_{v}"]) for v in views}
                bad = [v for v, b in probe.items() if b[12:16] != b"IHDR"
                       or int.from_bytes(b[16:20], "big") != img_size or int.from_bytes(b[20:24], "big") != img_size]
                if bad:
                    self._spec_broken = True
                    why = f"probe views {bad} not {img_size} px (render_service JS without per-view sizes?) -> SPEC_FINAL off"
                    logger.error(f"[RENDERER] speculative final: {why}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - incl. a missing probe key or a bad base64 view
            why = f"{type(exc).__name__}: {exc}"
        if why is not None:
            logger.warning(f"[RENDERER] '{task.stem}' spec probe FAIL ({why}) -> plain probe")
            return await self.render_views(task, views, img_size=img_size)
        task._spec_final = (finals, task.scene_json, task.js_code,
                            {k: b for k, b in got.items() if not k.startswith("probe_")}, sidecar_idx,
                            resp.json().get("lightings_applied") is True)
        return probe

    def _spec_final_specs(self) -> list[dict]:
        """[patch_loop] edit 10: the view specs process_with_judge_views sends (kept in sync by _spec_take's check)."""
        specs = self._grid_specs()
        specs += [{"name": n, "bg": self.config.judge_white_bg} for n in self._JUDGE_WHITE_VIEWS]
        specs.append({
            "name": self._JUDGE_GRAY_VIEW,
            "bg": self.config.judge_gray_bg,
            "key": self._JUDGE_GRAY_KEY,
        })
        return specs

    def spec_ready(self, task: PipelineTask) -> bool:
        """[patch_loop] edit 10: process_with_judge_views(task) will replay a kept speculative final (no request)."""
        s = task.__dict__.get("_spec_final")
        return (s is not None and s[1] is not None and s[1] is task.scene_json and s[2] == task.js_code
                and s[0] == self._spec_final_specs())

    def _spec_take(self, task: PipelineTask, view_specs: list[dict]):
        """[patch_loop] edit 10: pop the kept speculative final; a (response, sidecar_idx) pair only when it was
        rendered from this very scene with exactly these view specs."""
        s = task.__dict__.pop("_spec_final", None)
        if s is None:
            return None
        if s[1] is None or s[1] is not task.scene_json or s[2] != task.js_code or s[0] != view_specs:
            logger.warning(f"[RENDERER] '{task.stem}' speculative final discarded (request differs)")
            return None
        logger.info(f"[RENDERER] '{task.stem}' speculative final reused (rendered with the probe)")
        return _SpecResp(s[3], s[5]), s[4]

    def _grid_specs(self) -> list[dict]:
        if self.__dict__.get("_jgrid") == "contest":   # [jgrid] per-view size/bg/lighting = contest /render/grid tiles
            return [
                {"theta": theta, "phi": phi, "key": key, "bg": JUDGE_GRID_BG, "size": JUDGE_GRID_TILE,
                 "lighting": JUDGE_GRID_LIGHTING}
                for (theta, phi), key in zip(GRID_VIEWS, GRID_KEYS)
            ]
        bg = self.config.bg_color
        return [
            {"theta": theta, "phi": phi, "key": key, **({"bg": bg} if bg else {})}
            for (theta, phi), key in zip(GRID_VIEWS, GRID_KEYS)
        ]

    # Per-angle views consumed by the multi-stage judge. The gray front view
    # rides in the same request under a distinct key (name collides with white).
    _JUDGE_WHITE_VIEWS = (
        "front_left", "front_right", "front_below", "front_above",
        "right", "back", "left", "top_down",
    )
    _JUDGE_GRAY_VIEW = "front_left"
    _JUDGE_GRAY_KEY = "front_left_gray"

    async def _render_and_fill(self, task: PipelineTask, view_specs: list[dict]) -> dict[str, bytes]:
        """One /render/views request: grid tiles composed into ``task.rendered_png``
        (missing/invalid tile is fatal), remaining views returned."""
        if task.failed or (not task.js_code and not task.scene_json):
            logger.debug(
                f"[RENDERER] '{task.stem}' skip "
                f"(failed={task.failed}, has_code={bool(task.js_code)}, "
                f"has_scene={bool(task.scene_json)})"
            )
            return {}

        use_object = task.scene_json is not None
        payload: dict = {
            "views": view_specs,
            "options": {"imgSize": self.config.img_size, "lighting": self.config.lighting.value},
        }
        if use_object:
            payload["object"] = task.scene_json
            size_note = "scene_json"
        else:
            payload["source"] = task.js_code
            size_note = f"js_code={len(task.js_code)} bytes"

        logger.info(
            f"[RENDERER] '{task.stem}' start | views={len(view_specs)} "
            f"| path={'object' if use_object else 'code'} | {size_note}"
        )

        t0 = time.monotonic()
        _spec = self._spec_take(task, view_specs)   # [patch_loop] edit 10: None unless SPEC_FINAL kept this final
        try:
            if _spec is not None:
                resp, sidecar_idx = _spec
            else:
                resp, sidecar_idx = await self._post_with_retry("/render/views", payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            task.render_errors = [f"{type(exc).__name__}: {exc}"]
            STATS.render_done(time.monotonic() - t0, False)
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (http) | {task.render_errors[0]}"
            )
            return {}

        task.render_ms = (time.monotonic() - t0) * 1000.0

        if resp.status_code != 200:
            detail = resp.text[:200] if resp.text else ""
            task.render_errors = [f"HTTP {resp.status_code}: {detail}"]
            STATS.render_done(task.render_ms / 1000.0, False)
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (status) sidecar=#{sidecar_idx} | "
                f"{task.render_errors[0]} | render={task.render_ms/1000:.1f}s"
            )
            return {}

        if (use_object and any("lighting" in s for s in view_specs)
                and resp.json().get("lightings_applied") is not True):   # [jgrid] F4: old JS draws neutral tiles
            self._jgrid_noapply += 1
            if self._jgrid_noapply == 1:
                logger.error(f"[RENDERER] JUDGE_GRID=contest: render_service did not apply per-view lighting "
                             f"('{task.stem}'): old JS? grid tiles are NOT contest-exact (counted in _jgrid_noapply)")

        rendered: dict[str, bytes] = {}
        for name, b64 in (resp.json().get("views") or {}).items():
            try:
                rendered[name] = base64.b64decode(b64)
            except Exception as exc:  # noqa: BLE001 - skip a single bad view, keep the rest
                logger.warning(f"[RENDERER] view {name!r} decode failed: {exc}")

        try:
            tiles = [rendered.pop(key) for key in GRID_KEYS]
            _jg = self.__dict__.get("_jgrid", "off")
            if _jg == "off":
                task.rendered_png = await asyncio.to_thread(
                    compose_grid, tiles, img_size=self.config.img_size, gap=self.config.grid_gap
                )
            else:   # [jgrid] 518 px tiles; larger tiles (small mode, or the code path ignoring "size") downscaled
                if _jg == "contest" and not use_object and not self._jgrid_code_warned:
                    self._jgrid_code_warned = True
                    logger.warning("[RENDERER] JUDGE_GRID=contest on the code render path: per-view size/lighting "
                                   "not supported there -> grey neutral tiles downscaled (grid not contest-exact)")
                task.rendered_png = await asyncio.to_thread(
                    compose_grid, tiles, img_size=min(JUDGE_GRID_TILE, self.config.img_size),
                    gap=self.config.grid_gap, downscale=True
                )
        except (KeyError, ValueError) as exc:
            task.render_errors = [f"grid tile invalid: {exc}"]
            STATS.render_done(task.render_ms / 1000.0, False)
            logger.warning(
                f"[RENDERER] '{task.stem}' FAIL (grid) sidecar=#{sidecar_idx} | "
                f"{task.render_errors[0]}"
            )
            return rendered

        STATS.render_done(task.render_ms / 1000.0, True)
        task.refinement_rendered_pngs.append(task.rendered_png)
        logger.info(
            f"[RENDERER] '{task.stem}' PASS sidecar=#{sidecar_idx} | "
            f"png={len(task.rendered_png)}B views={len(rendered)} "
            f"render={task.render_ms/1000:.1f}s"
        )
        return rendered

    def _resolve_node_cwd(self) -> str:
        preferred = os.environ.get("NODE_CWD")
        candidates = [preferred, "/workspace", str(_RUNNER_JS.parent.parent.parent.parent)]
        for c in candidates:
            if c and os.path.isdir(os.path.join(c, "node_modules")):
                return c
        return str(_RUNNER_JS.parent)

    @staticmethod
    def _parse_browser_pid(resp: httpx.Response) -> int | None:
        try:
            pid = resp.json().get("browserPid")
            return int(pid) if pid else None
        except Exception:
            return None

    async def _wait_ready_for(
        self,
        proc: asyncio.subprocess.Process,
        client: httpx.AsyncClient,
        port: int,
        idx: int,
    ) -> int | None:
        """Wait until the sidecar answers /ping with 200; returns the reported
        chromium pid (None when the sidecar predates the reporting)."""
        deadline = time.monotonic() + self.config.startup_timeout_s
        ping_url = f"http://{self.config.sidecar_host}:{port}/ping"
        last_err: str = ""
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                raise RuntimeError(
                    f"[RENDERER] sidecar #{idx} exited early: rc={proc.returncode}"
                )
            try:
                r = await client.get(ping_url, timeout=2.0)
                if r.status_code == 200:
                    return self._parse_browser_pid(r)
                last_err = f"status={r.status_code}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)
        raise RuntimeError(
            f"[RENDERER] sidecar #{idx} not ready after {self.config.startup_timeout_s}s "
            f"(last: {last_err})"
        )

    @staticmethod
    async def _pipe_logger(stream: asyncio.StreamReader | None, which: str) -> None:
        if stream is None:
            return
        try:
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    if "stderr" in which:
                        logger.warning(f"[renderer-sidecar {which}] {text}")
                    else:
                        logger.debug(f"[renderer-sidecar {which}] {text}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"[renderer-sidecar {which}] pipe error: {exc}")
