"""Option A step 2 "burst" (owner 17/10): in the judge-only tail of a batch, the 2 coder GPUs become 2 extra GLM judges.

Mounted as /workspace/llm/burst.py. Enabled by env BURST_GLM_URLS (comma list of the sleeping GLM servers that our
sgl_shim.sh starts on the coder GPUs, e.g. http://localhost:8031/v1,http://localhost:8032/v1); BURST_PROXY = the coder
lb_proxy (default http://localhost:8001), which does the actual GPU swap (/burst/to_judge, /burst/to_coder).

Life of a batch (hooks in stages.py and generation_pipeline.py; CTRL is None when disabled -> leader behaviour):
  batch_start(n)        new batch of n tasks; coder state is ensured between batches
  coder_begin(k, stem)  a task launched k coder calls (start_coders); coder_end(gen) when each returns
  task_done(stem)       a task is recorded for good (also covers tasks that failed before launching any coder)
  -> when no coder call is pending and every task has launched its coders or is done (debounce DEBOUNCE_S): to_judge =
     proxy /burst/to_judge (coder memory released, GLMs awake), then the 2 replicas join the judge balancer and the
     duel semaphore grows by SLOTS (48 slots cannot keep 4 GLMs busy)
  ensure_coder()        awaited before EVERY coder call: if the burst is up (a retry needs the coder), take the extra
                        replicas out of the balancer, wait until their in-flight requests finish, proxy /burst/to_coder
  batch_end()           background ensure_coder (run_warmup awaits it; after a batch the next batch's first coder call
                        waits for it if it is still running)
Review 17/10 fixes: the swap-in re-checks its conditions after the proxy answers (a coder call or batch end that came
during the swap reverts it at once); coder calls wait on the lock while a swap runs; the swap back is one shared,
cancel-shielded task; a failed swap back keeps state "judge" so the next caller retries; drain cap > judge timeout.
"""
from __future__ import annotations

import asyncio
import os
import time

import httpx

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

CTRL: "BurstController | None" = None
DEBOUNCE_S = float(os.environ.get("BURST_DEBOUNCE_S", "2"))
DRAIN_S = float(os.environ.get("BURST_DRAIN_S", "330"))   # > judge request_timeout_s (300): never sleep a GLM mid-request
SLOTS = int(os.environ.get("BURST_SLOTS", "48"))


class BurstController:
    def __init__(self, lb, idxs: list[int], proxy: str, sem: asyncio.Semaphore | None = None) -> None:
        self.lb, self.idxs, self.proxy, self.sem = lb, idxs, proxy.rstrip("/"), sem
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30))
        self.state = "coder"
        self.lock = asyncio.Lock()
        self.gen = 0               # batch generation: stale coder_end() from an older batch is ignored
        self.n: int | None = None  # tasks in the current batch (None = not in a batch: warm-up, probe)
        self.stems: set | None = None
        self.pending = 0
        self.launched: set = set()   # stems that launched their coders this batch
        self.done: set = set()       # stems recorded for good this batch
        self.armed = False
        self.t0 = time.time()
        self._timer: asyncio.TimerHandle | None = None
        self._sem_extra = 0
        self._shrink: asyncio.Task | None = None
        self._back: asyncio.Task | None = None   # the shared swap-back task
        self._tasks: set = set()                 # strong refs to background tasks
        self.events: list[dict] = []

    # ---- bookkeeping (sync, called from the pipeline's event loop) ----
    def _ev(self, what: str, **kw) -> None:
        e = {"t": round(time.time() - self.t0, 1), "gen": self.gen, "what": what, **kw}
        self.events.append(e)
        logger.info(f"[burst] {what} at batch t={e['t']}s {kw if kw else ''}")

    def batch_start(self, n: int, stems=None) -> None:
        self.stems = set(stems) if stems is not None else None   # tasks of THIS batch (a timed-out batch's leftovers
        self.gen += 1                                               # must not count as launched)
        self.n, self.pending, self.armed, self.t0 = n, 0, True, time.time()
        self.launched, self.done = set(), set()
        self._cancel_timer()
        self._ev("batch_start", n=n)

    def batch_end(self) -> None:
        self.armed = False
        self.n = None
        self._cancel_timer()
        self._ev("batch_end", state=self.state)
        if self.state != "coder":      # (a swap-in still running sees armed=False afterwards and reverts itself)
            self._spawn(self.ensure_coder("batch end"))

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    def coder_begin(self, k: int, stem=None) -> int:
        self.pending += k
        self.launched.add(stem if stem is not None else len(self.launched))
        self._cancel_timer()
        return self.gen

    def coder_end(self, gen: int) -> None:
        if gen != self.gen:
            return
        self.pending -= 1
        self._maybe_arm()

    def task_done(self, stem) -> None:
        self.done.add(stem)
        self._maybe_arm()

    def _maybe_arm(self) -> None:
        seen = self.launched | self.done
        if self.stems is not None:
            seen &= self.stems
        if (self.armed and self.n is not None and self.pending <= 0 and len(seen) >= self.n
                and self.state == "coder" and self._timer is None):
            gen = self.gen

            def fire():                 # clears its own handle synchronously (no window for a stray second timer)
                self._timer = None
                self._spawn(self._to_judge(gen))
            self._timer = asyncio.get_running_loop().call_later(DEBOUNCE_S, fire)

    def _still_wanted(self, gen: int) -> bool:
        return gen == self.gen and self.armed and self.pending <= 0

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    # ---- swaps ----
    async def _to_judge(self, gen: int) -> None:
        async with self.lock:
            if not self._still_wanted(gen) or self.state != "coder":
                return
            t = time.time()
            try:
                r = await self.http.post(self.proxy + "/burst/to_judge")
                js = r.json()
            except Exception as exc:  # noqa: BLE001
                js = {"ok": False, "error": repr(exc)}
                try:   # the swap may have happened anyway (timeout / reset after the proxy acted)
                    if (await self.http.get(self.proxy + "/burst/state", timeout=30)).json().get("state") == "judge":
                        self.state = "judge"
                        self.armed = False
                        self._ev("to_judge lost answer, proxy is in judge -> back", res=js)
                        await self._swap_back_locked("to_judge answer lost", time.time())
                        self._retry_back_if_idle()
                        return
                except Exception:  # noqa: BLE001
                    pass
            if not js.get("ok"):
                self.armed = False          # no second try this batch; the batch just runs like the leader's
                self._ev("to_judge FAILED", res=js)
                return
            self.state = "judge"            # GPUs are swapped now, whatever happens next
            if not self._still_wanted(gen):  # a coder call / batch end came during the swap: revert at once
                self._ev("to_judge reverted", s=round(time.time() - t, 2), pending=self.pending, armed=self.armed)
                await self._swap_back_locked("came during to_judge", time.time())
                self._retry_back_if_idle()
                return
            self.lb.set_active(self.idxs, True)
            self.state = "judge"
            self._grow()
            self._ev("to_judge", s=round(time.time() - t, 2), proxy=js, slots_added=self._sem_extra)

    def _retry_back_if_idle(self) -> None:
        """a swap back failed outside a batch: retry in the background (else the next batch's first coder call pays it)"""
        if self.state != "coder" and self.n is None:
            self._spawn(self._to_coder("retry after failed swap back"))

    async def ensure_coder(self, why: str = "coder call") -> None:
        if self.state == "coder" and not self.lock.locked():   # locked = a swap is running: wait for its outcome
            return
        if self._back is None or self._back.done():
            self._back = self._spawn(self._to_coder(why))
        # shielded: a cancelled caller (task deadline, batch timeout) must not cut a swap in half
        await asyncio.shield(self._back)

    async def _to_coder(self, why: str) -> None:
        async with self.lock:
            if self.state == "coder":
                return
            await self._swap_back_locked(why, time.time())

    async def _swap_back_locked(self, why: str, t: float) -> None:
        self.lb.set_active(self.idxs, False)   # no new judge requests go to the extra replicas
        while self.lb.inflight_of(self.idxs) and time.time() - t < DRAIN_S:
            await asyncio.sleep(0.05)
        left = self.lb.inflight_of(self.idxs)
        drain = round(time.time() - t, 2)
        try:
            r = await self.http.post(self.proxy + "/burst/to_coder")
            js = r.json()
        except Exception as exc:  # noqa: BLE001
            js = {"ok": False, "error": repr(exc)}
        if js.get("ok"):
            self.state = "coder"
            self._start_shrink()
        # on failure state stays "judge" (replicas stay out): the next ensure_coder retries, and the proxy's own
        # safety path resumes the coder on the next coder request anyway
        self._ev("to_coder" if js.get("ok") else "to_coder FAILED", why=why, drain_s=drain,
                 left_in_flight=left, s=round(time.time() - t, 2), proxy=js)

    # ---- duel slots ----
    def _grow(self) -> None:
        if self.sem is None or SLOTS <= 0:
            return
        if self._shrink is not None and not self._shrink.done():
            self._shrink.cancel()
        for _ in range(SLOTS - self._sem_extra):
            self.sem.release()
        self._sem_extra = SLOTS

    def _start_shrink(self) -> None:
        if self.sem is None or self._sem_extra <= 0:
            return
        if self._shrink is not None and not self._shrink.done():
            return                      # the running shrink loops until _sem_extra == 0 (two would take too much)

        async def shrink():
            while self._sem_extra > 0:
                await self.sem.acquire()   # takes back one extra slot as soon as a duel frees one
                self._sem_extra -= 1

        self._shrink = self._spawn(shrink())


def setup(pipeline, judge_lb, make_client) -> "BurstController | None":
    """Called once from GenerationPipeline.startup. make_client(name, base_url) -> instrumented AsyncOpenAI."""
    global CTRL
    urls = [u.strip() for u in os.environ.get("BURST_GLM_URLS", "").split(",") if u.strip()]
    if not urls:
        return None
    if not hasattr(judge_lb, "add_replicas"):
        logger.warning("[burst] judge client is not the burst balancer -> burst disabled")
        return None
    clients = [make_client(f"judge-burst-{i}", u) for i, u in enumerate(urls)]
    idxs = judge_lb.add_replicas(clients, active=False)
    sem = getattr(pipeline, "_sem", {}).get("judge")
    CTRL = BurstController(judge_lb, idxs, os.environ.get("BURST_PROXY", "http://localhost:8001"), sem)
    _mix = __import__("sys").modules.get("pgh_mix")
    if _mix is not None and _mix.active() and len(_mix.proxy_urls()) > 1:
        # [mix] R71: each coder has its own lb_proxy (+ its burst GLM): swap all of them together (pgh_mix.FanoutHTTP)
        CTRL.http, CTRL.proxy = _mix.FanoutHTTP(_mix.proxy_urls(), CTRL.http), ""
        logger.info(f"[burst] mix: one swap = every coder proxy {_mix.proxy_urls()} (all or none)")
    logger.info(f"[burst] enabled: extra judge replicas {urls} -> balancer idx {idxs}, proxy {CTRL.proxy}, "
                f"+{SLOTS} duel slots while up, debounce {DEBOUNCE_S}s")
    return CTRL
