"""[opt_r5 shadow] PGH_SHADOW: "shadow picks" for selection features that are OFF in the shipped run. Default OFF.

A separate A/B arm cannot see a +0.01-0.02 selection effect. Instead, ONE run ships its normal pick unchanged and, per
photo, logs what each listed feature WOULD have picked, so that later only the photos where the picks differ are judged
head to head offline (tools/shadow_duel_list.py).

Switches (ENV, read when used; parsing never raises):
  PGH_SHADOW              ""/unset/0/off = off (nothing below runs, the pipeline is byte-identical to pgh17d).
                          comma list of features, or "all":
                            orient_final  what PGH_ORIENT_FINAL=1 would do to the shipped final (orient_final.py's own
                                          maybe_turn_final on a COPY: rotated renders + margin gate + <= 1-2 duels)
                            s1es          what PGH_S1ES_MAX_ROUND=<PGH_SHADOW_S1ES_ROUND> would pick: the shipped bracket /
                                          round robin replayed; a duel is re-judged only where its early-stop flag differs
                                          and the shipped duel did (or may have) stopped early
                            rr8           what PGH_RR8=1 would pick: stages._round_robin_top8 replayed over the same
                                          leaves; the pairs the shipped bracket + top-4 RR already played are reused, the
                                          missing ones (<= 18 at 8 live eighth winners) are judged (stops early once one
                                          player leads beyond reach)
                          unknown names: one warning, dropped. A feature already ON in the shipped run is "skipped".
  PGH_SHADOW_MAX_DUELS    extra judge duels per photo, all features together (default 12)
  PGH_SHADOW_S1ES_ROUND   the max_round s1es emulates (default 2 = kit arm P17J)
  PGH_SHADOW_DIR          output dir (default $SAVE_DRAFTS_DIR/shadow if SAVE_DRAFTS_DIR is set, else /tmp/pgh_shadow)
  PGH_SHADOW_RESERVE_S    slack reserve, s (default 300): no extra work once the batch is within this of its budget
                          (batch_time_budget - 120) or, with PGH_TIMEGUARD on, of the guard's allowed wall for this batch
  PGH_SHADOW_JUDGE_FREE   a shadow duel starts only when >= this many judge slots are free and nobody waits (default 8)
  PGH_SHADOW_RENDER_FREE  same for render slots (orient_final; default 2); checker slots: 1
  PGH_SHADOW_PAR          in-flight shadow duels per photo (default 2); PGH_SHADOW_GLOBAL all photos (default 6)
  PGH_SHADOW_DRAIN_S      at batch end wait up to this long for running shadow jobs before cancelling them (default 0 =
                          never delay the batch end; tests use > 0)
The shipped pick is final before any shadow work starts: the job is a background task launched after _promote_winner,
the recorder only copies duel verdicts the bracket already computed, and every extra duel / render runs behind a gate
(slack + idle slots + caps). Shadow log lines carry the stem suffix "~sh" (replays) or the label prefix "SH ".
Output: <dir>/shadow.jsonl (one line per photo, see README), <dir>/shipped/<stem>.{js,png}, <dir>/<feature>/<stem>.{js,png}
for every feature whose alt code differs from the shipped code.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import sys
import time
import types

FEATURES = ("orient_final", "s1es", "rr8")
_ALIAS = {"of": "orient_final", "orient": "orient_final", "orientfinal": "orient_final", "orient_final": "orient_final",
          "s1es": "s1es", "s1es_max_round": "s1es", "rr8": "rr8"}
_OFF = ("", "0", "off", "false", "no")
_S1_FULL = 4   # multi_stage.S1_ANGLES: a shipped S1 with fewer angles may have stopped early
_WARNED: set = set()


def _logger():
    from logger_config import logger
    return logger


def _warn(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    try:
        _logger().warning(msg)
    except Exception:
        pass


def features(env=None) -> list:
    raw = str((os.environ if env is None else env).get("PGH_SHADOW", "") or "")
    if raw.strip().lower() in _OFF:
        return []
    if raw.strip().lower() == "all":
        return list(FEATURES)
    out = []
    for x in (p.strip().lower() for p in raw.split(",")):
        if not x:
            continue
        f = _ALIAS.get(x)
        if f is None:
            _warn(f"feat:{x}", f"[SHADOW] !! PGH_SHADOW: unknown feature {x!r} dropped (known: {', '.join(FEATURES)})")
        elif f not in out:
            out.append(f)
    return out


def on() -> bool:
    if not os.environ.get("PGH_SHADOW"):
        return False
    return bool(features())


def _num(name: str, default, lo=None, hi=None, cast=float):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        v = cast(str(raw).strip())
    except (TypeError, ValueError):
        _warn(name, f"[SHADOW] !! {name}={raw!r} invalid -> {default}")
        return default
    if v != v or (lo is not None and v < lo) or (hi is not None and v > hi):
        _warn(name, f"[SHADOW] !! {name}={raw!r} out of range -> {default}")
        return default
    return v


def max_duels() -> int:
    return _num("PGH_SHADOW_MAX_DUELS", 12, 0, 10000, int)


def s1es_round() -> int:
    return _num("PGH_SHADOW_S1ES_ROUND", 2, 1, 16, int)


def reserve_s() -> float:
    return _num("PGH_SHADOW_RESERVE_S", 300.0, 0.0)


def out_dir() -> str:
    d = os.environ.get("PGH_SHADOW_DIR", "").strip()
    if d:
        return d
    sd = os.environ.get("SAVE_DRAFTS_DIR", "").strip()
    return os.path.join(sd, "shadow") if sd else "/tmp/pgh_shadow"


class ShadowSkip(Exception):
    """Raised inside shadow work only (cap / no slack / cache miss); never reaches the shipped path."""


# ------------------------------------------------------------------ batch state + slack
BATCH: dict = {"idx": -1, "order": {}, "t0": None, "deadline": None}
_JOBS: set = set()
_GLOBAL: dict = {}


def batch_begin(stems, budget=None) -> None:
    """GenerationPipeline.run_batch, before any photo starts (only with PGH_SHADOW set)."""
    stems = list(stems or [])
    BATCH["idx"] += 1
    BATCH["order"] = {s: (i, len(stems)) for i, s in enumerate(stems)}
    BATCH["t0"] = time.time()
    try:
        BATCH["deadline"] = BATCH["t0"] + float(budget) - 120.0 - reserve_s() if budget is not None else None
    except (TypeError, ValueError):
        BATCH["deadline"] = None
    try:
        dl = "-" if BATCH["deadline"] is None else f"{BATCH['deadline'] - BATCH['t0']:.0f}s"
        _logger().info(f"[SHADOW] batch {BATCH['idx']} begin | features={features()} | max_duels={max_duels()} | "
                       f"dir={out_dir()} | slack deadline in {dl}")
    except Exception:
        pass


async def batch_end() -> None:
    """GenerationPipeline.run_batch finally: drain (PGH_SHADOW_DRAIN_S, default 0) then cancel the unfinished jobs; each
    job writes its own line ("cancelled: batch end") on the way out."""
    try:
        pend = [t for t in list(_JOBS) if not t.done()]
        drain = _num("PGH_SHADOW_DRAIN_S", 0.0, 0.0)
        if pend and drain > 0:
            await asyncio.wait(pend, timeout=drain)
        pend = [t for t in pend if not t.done()]
        for t in pend:
            t.cancel()
        if pend:
            await asyncio.gather(*pend, return_exceptions=True)
        _REC.clear()
        _logger().info(f"[SHADOW] batch {BATCH['idx']} end | cancelled {len(pend)} unfinished shadow job(s)")
    except asyncio.CancelledError:   # the batch itself is being cancelled: stop every job, keep the cancellation
        for t in list(_JOBS):
            t.cancel()
        raise
    except Exception as exc:  # noqa: BLE001 - never fail the batch
        try:
            _logger().warning(f"[SHADOW] batch_end: {type(exc).__name__}: {exc}")
        except Exception:
            pass


def slack() -> tuple:
    """(True, "") while extra shadow work cannot push the batch past its budget or the time guard's allowed wall."""
    now = time.time()
    d = BATCH.get("deadline")
    if d is not None and now > d:
        return False, f"batch budget (past the slack deadline by {now - d:.0f}s)"
    tg = sys.modules.get("pgh_timeguard")
    g = getattr(tg, "GUARD", None) if tg is not None else None
    try:
        if g is not None and getattr(g, "mode", "off") in ("live", "shadow") and getattr(g, "_t0", None) is not None:
            if g.mode == "live" and g.k_guard is not None and g.start_k is not None and g.k_guard < g.start_k:
                return False, f"timeguard cut active (K {g.k_guard} < {g.start_k})"
            p = g.p
            left = max(1, int(p["batches_per_round"]) - len(g.walls))
            allowed = (float(p["round_s"]) - float(p["margin_s"]) - sum(g.walls)) / left
            el = g.clock() - g._t0
            if el > allowed - reserve_s():
                return False, f"timeguard batch wall {el:.0f}s > allowed {allowed:.0f}s - reserve {reserve_s():.0f}s"
    except Exception as exc:  # noqa: BLE001 - an unreadable guard = no slack (safe side)
        return False, f"timeguard unreadable ({type(exc).__name__})"
    return True, ""


def _global_sem() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    s = _GLOBAL.get(id(loop))
    if s is None:
        _GLOBAL.clear()
        s = _GLOBAL[id(loop)] = asyncio.Semaphore(_num("PGH_SHADOW_GLOBAL", 6, 1, 10000, int))
    return s


class _Gate:
    """async with: global shadow limit, then wait until `sem` has >= free_min free slots and no waiter (re-checking the
    slack every poll), then hold one slot of `sem`. No slack -> ShadowSkip. Stateless per entry (re-entrant)."""

    def __init__(self, sem, job, free_min: int):
        self.sem, self.job, self.free_min = sem, job, free_min

    async def __aenter__(self):
        g = _global_sem()
        await g.acquire()
        try:
            poll = _num("PGH_SHADOW_POLL_S", 0.5, 0.001)
            while True:
                ok, why = slack()
                if not ok:
                    raise self.job.kill(f"no slack: {why}")
                if self.job.dead:
                    raise ShadowSkip(self.job.dead)
                if self.sem is None or (self.sem._value >= self.free_min and not getattr(self.sem, "_waiters", None)):
                    break
                await asyncio.sleep(poll)
            if self.sem is not None:
                await self.sem.acquire()
        except BaseException:
            g.release()
            raise
        return self

    async def __aexit__(self, *exc):
        if self.sem is not None:
            self.sem.release()
        _global_sem().release()
        return False


# ------------------------------------------------------------------ recorder (inside _resolve_bracket)
_REC: dict = {}


def _mode(stem: str, n: int) -> str:
    """Which selection _resolve_bracket runs for this photo (same conditions, same order)."""
    from pipeline import stages
    if stages._rr_top4_on() and n >= 4:
        if os.environ.get("PGH_RR8") and n >= 8:
            import pgh_jalign as _ja
            if _ja.rr8_for(stem, n):
                return "top8"
        return "top4"
    return "bracket"


class Recorder:
    """Copies every verdict the shipped bracket computes: (left k, right k) -> winner k, decided_by, S1 slim, round,
    early_stop flag (a pair is played at most once per photo). The wrapped duel returns exactly what the original returns."""

    def __init__(self, task, judge, leaf_tasks: dict):
        self.stem, self.judge, self.leaf_tasks, self.n = task.stem, judge, leaf_tasks, len(leaf_tasks)
        self.mode = _mode(task.stem, self.n)
        self.duels: dict = {}
        _REC[task.stem] = self

    def es(self, rnd: int):
        try:
            return bool(self.judge.early_stop_for_round(rnd))
        except Exception:
            return None

    def wrap(self, fn):
        async def duel(left, right, round_no, _rr=None):
            out = _rr if _rr is not None else {}
            w = await fn(left, right, round_no, out)
            try:
                self.duels[(left.k, right.k)] = {"w": w.k, "decided_by": str(out.get("decided_by") or ""),
                                                 "s1": out.get("s1"), "rnd": round_no, "es": self.es(round_no)}
            except Exception:
                pass
            return w
        return duel


def recorder(task, judge, leaf_tasks):
    try:
        return Recorder(task, judge, leaf_tasks)
    except Exception as exc:  # noqa: BLE001
        _warn("recorder", f"[SHADOW] recorder failed: {type(exc).__name__}: {exc}")
        return None


def _stopped(r: dict) -> bool:
    s1 = r.get("s1")
    if not s1:
        return True
    angles = s1.get("angles") or []
    return int(s1.get("n_total", len(angles)) or 0) < _S1_FULL


def _md5(s) -> str | None:
    return hashlib.md5(s.encode("utf-8")).hexdigest() if s else None


# ------------------------------------------------------------------ the per-photo job
def launch(task, factory, winner, *, judge, sem_judge, K: int) -> None:
    """multigen_first_iter, after _promote_winner: snapshot the shipped final and start the job in the background."""
    try:
        rec = _REC.pop(task.stem, None)
        job = _Job(task, factory, winner, judge, sem_judge, rec, K)
        t = asyncio.get_running_loop().create_task(job.run())
        _JOBS.add(t)
        t.add_done_callback(_JOBS.discard)
    except Exception as exc:  # noqa: BLE001 - never fail the prompt
        _warn("launch", f"[SHADOW] launch failed: {type(exc).__name__}: {exc}")


class _JudgeProxy:
    """judge for orient_final's own code: counts + caps the extra duels, label prefix 'SH '."""

    def __init__(self, job):
        self.job = job

    async def compare(self, **kw):
        self.job.take_duel()
        kw["match_label"] = "SH " + str(kw.get("match_label", ""))
        return await self.job.judge.compare(**kw)


class _RendererProxy:
    def __init__(self, job, r):
        self.job, self.r = job, r

    async def process_with_judge_views(self, probe):
        self.job.renders += 1
        self.job.fr["renders"] = self.job.fr.get("renders", 0) + 1
        return await self.r.process_with_judge_views(probe)


class _Job:
    def __init__(self, task, factory, winner, judge, sem_judge, rec, K):
        self.task, self.factory, self.judge, self.sem_judge, self.rec, self.K = task, factory, judge, sem_judge, rec, K
        self.stem = task.stem
        self.ship = copy.copy(winner)   # the shipped final as promoted (never touched again)
        self.ship_code = winner.js_code
        self.cap = max_duels()
        self.used = 0
        self.inflight = 0
        self.renders = 0
        self.misses = 0
        self.fresh: dict = {}   # (left k, right k, early_stop) -> verdict of a shadow duel
        self.dead = None
        self.fr: dict = {}
        self.par = asyncio.Semaphore(_num("PGH_SHADOW_PAR", 2, 1, 10000, int))
        pos = BATCH["order"].get(self.stem)
        self.line = {"stem": self.stem, "batch": BATCH["idx"], "pos": pos[0] if pos else None,
                     "n_batch": pos[1] if pos else None, "K": K, "n_positions": rec.n if rec else None,
                     "mode": rec.mode if rec else None, "shipped_k": winner.k, "shipped_md5": _md5(winner.js_code),
                     "shipped_js": f"shipped/{self.stem}.js", "features": {}}

    # ---- caps / kill
    def kill(self, why: str) -> ShadowSkip:
        if self.dead is None:
            self.dead = why
        return ShadowSkip(self.dead)

    def take_duel(self) -> None:
        if self.dead:
            raise ShadowSkip(self.dead)
        if self.used >= self.cap:
            raise self.kill(f"cap ({self.cap} duels/photo)")
        self.used += 1
        self.fr["duels"] = self.fr.get("duels", 0) + 1

    # ---- duels
    def _lookup(self, ka, kb, es):
        r = self.rec.duels.get((ka, kb))
        if r is not None and r["es"] is not None and (
                r["es"] == es or (es is False and r["es"] is True and not _stopped(r))):
            return r   # same flag, or S1 ran all angles under early stop (= the verdict without early stop)
        return self.fresh.get((ka, kb, es))

    async def _fresh(self, a, b, es, label):
        async with self.par:
            async with _Gate(self.sem_judge, self, _num("PGH_SHADOW_JUDGE_FREE", 8, 0, 10000, int)):
                r = self._lookup(a.k, b.k, es)   # another branch may have judged it meanwhile
                if r is not None:
                    return r
                self.take_duel()
                t = self.task
                v = await self.judge.compare(
                    task_id=self.stem, match_label=f"SH {label}", reference_bytes=t.image_bytes,
                    reference_mime=t.image_mime, render_a=a.rendered_png, render_b=b.rendered_png,
                    white_views_a=a.judge_white_views, white_views_b=b.judge_white_views,
                    gray_views_a=a.judge_gray_views, gray_views_b=b.judge_gray_views,
                    embeddings_a=a.judge_embeddings, embeddings_b=b.judge_embeddings, early_stop=bool(es))
        r = {"w": a.k if v.winner == "A" else b.k, "decided_by": str(getattr(v, "decided_by", "") or ""),
             "s1": (getattr(v, "detail", None) or {}).get("s1_slim"), "es": es, "fresh": True}
        self.fresh[(a.k, b.k, es)] = r
        return r

    def judge_fn(self, es_fn, strict: bool):
        async def jd(a, b, rnd, out=None):
            es = es_fn(rnd)
            r = self._lookup(a.k, b.k, es)
            if r is None:
                if strict:
                    self.misses += 1
                    raise ShadowSkip(f"cache miss k{a.k}-vs-k{b.k} R{rnd}")
                if self.dead:
                    raise ShadowSkip(self.dead)
                label = (out or {}).get("label") or f"R{rnd} k{a.k}-vs-k{b.k}"
                r = await self._fresh(a, b, es, label)
            if out is not None:
                out["decided_by"] = r["decided_by"]
                out["s1"] = r["s1"]
            return a if r["w"] == a.k else b
        return jd

    def winner_of_fn(self, jd):
        from pipeline.stages import _is_live
        leaves = self.rec.leaf_tasks

        async def wo(lo, hi):   # = _resolve_bracket._winner_of over the (resolved) leaf tasks
            if hi - lo <= 1:
                return await leaves[lo]
            mid = (lo + hi) // 2
            left, right = await asyncio.gather(wo(lo, mid), wo(mid, hi))
            if not _is_live(left):
                return right
            if not _is_live(right):
                return left
            return await jd(left, right, (hi - lo - 1).bit_length())
        return wo

    def _proxy_task(self):
        from pipeline.task import PipelineTask
        p = PipelineTask(stem=f"{self.stem}~sh", image_url=self.task.image_url, seed=self.task.seed)
        p.image_bytes, p.image_mime = self.task.image_bytes, self.task.image_mime
        p.started_at = time.time()
        return p

    async def replay(self, mode: str, es_fn, strict: bool):
        from pipeline import stages
        jd = self.judge_fn(es_fn, strict)
        wo = self.winner_of_fn(jd)
        if mode == "top8":
            return await stages._round_robin_top8(self._proxy_task(), self.rec.n, wo, jd)
        if mode == "top4":
            return await stages._round_robin_top4(self._proxy_task(), self.rec.n, wo, jd)
        return await wo(0, self.rec.n)

    # ---- run
    async def run(self):
        t0 = time.monotonic()
        feats = features()
        cancelled = False
        try:
            if self.rec is None or not self.rec.leaf_tasks:
                self.line["status"] = "no bracket (judge off or K <= 1)"
                for f in feats:
                    self.line["features"][f] = {"status": "skipped: no bracket"}
                return
            await self._self_check()
            for f in feats:
                self.fr = {"status": "running"}
                self.line["features"][f] = self.fr
                self.dead = None
                try:
                    await getattr(self, f"_f_{f}")()
                except ShadowSkip as exc:
                    self.fr["status"] = f"skipped: {self.dead or exc}"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.fr["status"] = f"error: {type(exc).__name__}: {exc}"[:300]
                if self.dead and not str(self.fr.get("status", "")).startswith(("skipped", "error")):
                    self.fr["status"] = f"skipped: {self.dead}"
                if not str(self.fr.get("status", "")).startswith(("same", "differs")):
                    for k in ("alt_k", "alt_md5", "alt_js", "_alt"):
                        self.fr.pop(k, None)
        except asyncio.CancelledError:
            cancelled = True
            for f in feats:
                fr = self.line["features"].setdefault(f, {})
                if fr.get("status") in (None, "running"):
                    fr["status"] = "cancelled: batch end"
                    for k in ("alt_k", "alt_md5", "alt_js", "_alt"):
                        fr.pop(k, None)
            raise
        except Exception as exc:  # noqa: BLE001
            self.line["error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self.line["extra_duels"] = self.used
            self.line["extra_renders"] = self.renders
            self.line["wall_s"] = round(time.monotonic() - t0, 2)
            self.line["cancelled"] = cancelled
            self._write()

    async def _self_check(self):
        """Replay the SHIPPED selection from the recorded verdicts only (0 duels): must give the shipped k."""
        self.misses = 0
        try:
            pick = await self.replay(self.rec.mode, self.rec.es, strict=True)
            k = getattr(pick, "k", None)
            self.line["self_check"] = "ok" if k == self.ship.k else f"MISMATCH replay k{k} != shipped k{self.ship.k}"
            if self.misses:   # an RR pair the shipped run never finished (cutoff / error): the replay fell back
                self.line["self_check"] = f"n/a ({self.misses} unplayed pair(s), shipped RR incomplete)"
        except ShadowSkip as exc:
            self.line["self_check"] = f"n/a ({exc})"
        if str(self.line["self_check"]).startswith("MISMATCH"):
            _warn(f"sc:{self.stem}", f"[SHADOW] !! {self.stem} self-check {self.line['self_check']}")

    def _set_alt(self, cand, **extra):
        code = getattr(cand, "js_code", None)
        self.fr.update(extra)
        self.fr["alt_k"] = getattr(cand, "k", None)
        self.fr["alt_md5"] = _md5(code)
        if code and code != self.ship_code:
            self.fr["status"] = "differs"
            self.fr["_alt"] = cand
        else:
            self.fr["status"] = "same"

    async def _f_orient_final(self):
        from pipeline import orient_final as _of
        if _of.enabled():
            self.fr["status"] = "skipped: already on (PGH_ORIENT_FINAL)"
            return
        f = self.factory
        if f is None:
            self.fr["status"] = "skipped: no factory"
            return
        final = copy.copy(self.ship)
        fac = types.SimpleNamespace(
            task=self._proxy_task(), js_checker=f.js_checker, renderer=_RendererProxy(self, f.renderer),
            embedder=f.embedder, ref_vec=f.ref_vec, checker_mode=f.checker_mode,
            sem_checker=_Gate(f.sem_checker, self, 1),
            sem_renderer=_Gate(f.sem_renderer, self, _num("PGH_SHADOW_RENDER_FREE", 2, 0, 10000, int)))
        rec = await _of.maybe_turn_final(fac, final, judge=_JudgeProxy(self),
                                         sem_judge=_Gate(self.sem_judge, self, _num("PGH_SHADOW_JUDGE_FREE", 8, 0, 10000, int)))
        if self.dead:
            raise ShadowSkip(self.dead)
        self._set_alt(final, of={k: rec.get(k) for k in ("applied", "reason", "best", "margin", "s_final") if k in rec})

    async def _f_s1es(self):
        n_alt = s1es_round()
        cfg = getattr(self.judge, "s1_early_stop", None)
        if cfg is None:
            self.fr["status"] = "skipped: judge has no s1_early_stop"
            return
        es_alt = lambda r: bool(cfg.enabled and r <= n_alt)   # noqa: E731 - = early_stop_for_round with max_round n_alt
        rounds = range(1, max(2, self.rec.n).bit_length() + 1)
        self.fr["max_round"] = n_alt
        if all(es_alt(r) == self.rec.es(r) for r in rounds):
            self.fr["status"] = f"skipped: shipped already uses max_round {n_alt}"
            return
        pick = await self.replay(self.rec.mode, es_alt, strict=False)
        if self.dead:
            raise ShadowSkip(self.dead)
        self._set_alt(pick)

    async def _f_rr8(self):
        from pipeline.stages import _is_live, _rr_is_draw
        if self.rec.mode == "top8" or os.environ.get("PGH_RR8", "").strip().lower() not in _OFF:
            self.fr["status"] = "skipped: already on (PGH_RR8)"
            return
        if self.rec.mode != "top4":
            self.fr["status"] = f"skipped: selection mode {self.rec.mode} (RR8 needs the top-4 round robin)"
            return
        n = self.rec.n
        if n < 8:
            self.fr["status"] = "skipped: < 8 positions"
            return
        pos = BATCH["order"].get(self.stem)
        if pos is not None:
            try:
                import pgh_jalign as _ja
                self.fr["tail"] = not (pos[0] < pos[1] - _ja.rr8_tail())   # PGH_RR8 itself keeps top-4 there
            except Exception:
                pass
        mid = n // 2
        qblocks = [(0, mid // 2), (mid // 2, mid), (mid, (mid + n) // 2), ((mid + n) // 2, n)]
        eblocks = []
        for lo, hi in qblocks:   # = _round_robin_top8's split
            m = (lo + hi) // 2
            eblocks += [(lo, m), (m, hi)]
        rnd_final = (n - 1).bit_length()
        es_f = self.rec.es(rnd_final)
        wo = self.winner_of_fn(self.judge_fn(self.rec.es, strict=True))
        eighth = list(await asyncio.gather(*(wo(lo, hi) for lo, hi in eblocks)))
        live = [i for i, e in enumerate(eighth) if _is_live(e)]
        pairs = [(i, j) for x, i in enumerate(live) for j in live[x + 1:]]
        need = [(i, j) for i, j in pairs if i // 2 != j // 2 and self._known(eighth[i].k, eighth[j].k, es_f) is None]
        self.fr["need"] = len(need)
        self.fr["live_eighths"] = len(live)
        total = len(need)
        while need and len(live) >= 3:
            lead = self._rr8_leader(eighth, live, pairs, need, _rr_is_draw)
            if lead is not None:
                self._set_alt(eighth[lead], decided=f"early ({total - len(need)} of {total} missing pairs played)")
                return
            wave = self._rr8_order(eighth, live, pairs, need, _rr_is_draw)[: _num("PGH_SHADOW_PAR", 2, 1, 10000, int)]
            res = await asyncio.gather(*(self._fresh(eighth[i], eighth[j], es_f, f"RR8 k{eighth[i].k}-vs-k{eighth[j].k}")
                                         for i, j in wave), return_exceptions=True)
            for r in res:
                if isinstance(r, BaseException):
                    if isinstance(r, asyncio.CancelledError):
                        raise r
                    raise r if isinstance(r, ShadowSkip) else ShadowSkip(f"duel error {type(r).__name__}: {r}"[:200])
            need = [p for p in need if self._known(eighth[p[0]].k, eighth[p[1]].k, es_f) is None]
        self.misses = 0
        pick = await self.replay("top8", self.rec.es, strict=True)   # every pair known: the feature's own code decides
        if self.misses:
            raise ShadowSkip(f"rr8 replay: {self.misses} pair(s) unknown")
        self._set_alt(pick, decided="full")

    def _known(self, ka, kb, es_f):
        """The verdict PGH_RR8 itself would use for this pair: the shipped bracket's own duel (any round) or a duel at
        the final's round (shipped top-4 RR extra or a shadow duel)."""
        r = self.rec.duels.get((ka, kb))
        if r is not None and (r["rnd"] != (self.rec.n - 1).bit_length() or r["es"] == es_f):
            return r
        return self._lookup(ka, kb, es_f)

    def _rr8_scores(self, eighth, live, pairs, need, is_draw):
        es_f = self.rec.es((self.rec.n - 1).bit_length())
        score = {i: 0 for i in live}
        rem = {i: 0 for i in live}
        nset = set(need)
        for i, j in pairs:
            if (i, j) in nset:
                rem[i] += 1
                rem[j] += 1
                continue
            r = self._known(eighth[i].k, eighth[j].k, es_f)
            if r is None:
                rem[i] += 1
                rem[j] += 1
                continue
            if not is_draw(r["decided_by"]):
                w = i if r["w"] == eighth[i].k else j
                score[w] += 1
                score[j if w == i else i] -= 1
        return score, rem

    def _rr8_leader(self, eighth, live, pairs, need, is_draw):
        """A player whose worst case beats every other player's best case: _rr_pick's unique maximum, whatever the
        missing results are."""
        score, rem = self._rr8_scores(eighth, live, pairs, need, is_draw)
        for i in live:
            if all(score[i] - rem[i] > score[j] + rem[j] for j in live if j != i):
                return i
        return None

    def _rr8_order(self, eighth, live, pairs, need, is_draw):
        score, _ = self._rr8_scores(eighth, live, pairs, need, is_draw)
        top = max(live, key=lambda i: (score[i], -i))
        return sorted(need, key=lambda p: (top not in p, -max(score[p[0]], score[p[1]]), p))

    # ---- output
    def _write(self):
        d = out_dir()
        try:
            os.makedirs(os.path.join(d, "shipped"), exist_ok=True)
            self._dump(os.path.join(d, "shipped", self.stem), self.ship_code, self.ship.rendered_png)
            for f, fr in self.line["features"].items():
                cand = fr.pop("_alt", None)
                if cand is not None and fr.get("status") == "differs":
                    os.makedirs(os.path.join(d, f), exist_ok=True)
                    self._dump(os.path.join(d, f, self.stem), cand.js_code, cand.rendered_png)
                    fr["alt_js"] = f"{f}/{self.stem}.js"
            with open(os.path.join(d, "shadow.jsonl"), "a") as fh:
                fh.write(json.dumps(self.line, default=str, sort_keys=True) + "\n")
        except Exception as exc:  # noqa: BLE001
            _warn(f"write:{d}", f"[SHADOW] !! cannot write {d}: {type(exc).__name__}: {exc}")
        try:
            fs = " ".join(f"{f}={str(fr.get('status'))[:60]}" for f, fr in self.line["features"].items())
            _logger().info(f"[SHADOW] {self.stem} shipped=k{self.ship.k} self_check={self.line.get('self_check')} {fs} "
                           f"extra_duels={self.used} extra_renders={self.renders} wall={self.line.get('wall_s')}s")
        except Exception:
            pass

    @staticmethod
    def _dump(base, code, png):
        if code:
            with open(base + ".js", "w") as fh:
                fh.write(code)
        if png and _num("PGH_SHADOW_SAVE_PNG", 1, 0, 1, int):
            with open(base + ".png", "wb") as fh:
                fh.write(png)
