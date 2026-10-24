from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager

import httpx

from config.settings import OrientationConfig
from logger_config import logger
from modules.scene_planner.schema import OSD
from pipeline.orientation import orient_task
from pipeline.task import Candidate, PipelineTask
from utils.http import download_image


from pipeline.batch_stats import STATS
from llm import burst as _burst
from modules.js_checker import three_repair as _three_repair   # [pgh16] PGH_THREE_REPAIR (default off)
from pipeline import orient_final as _orient_final   # [R86] PGH_ORIENT_FINAL (default off)
from modules.js_checker import mech_repair as _mech_repair   # [R81] PGH_MECH_REPAIR / PGH_CHECK_TIMEOUT_RETRY (default off)
try:   # [R96 pgh17d] PGH_DEADLINE_SALVAGE / PGH_K_CPU_AWARE (image: 1 / 0; unset = off; import failure = off)
    import pgh_salvage as _salv
except Exception:
    _salv = None
try:
    import pgh_kcpu as _kcpu
except Exception:
    _kcpu = None


class StageError(Exception):
    """Raised by a pipeline stage; carries stage name + original cause."""

    def __init__(self, stage: str, cause: Exception):
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause


@asynccontextmanager
async def stage_guard(
    task: PipelineTask,
    stage_name: str,
    sem: asyncio.Semaphore,
    status: dict[str, str],
):
    """Acquire semaphore, log start/done, wrap exceptions in StageError."""
    async with sem:
        status[task.stem] = stage_name
        t0 = time.time()
        try:
            yield
        except Exception as exc:
            raise StageError(stage_name, exc) from exc
        finally:
            dt = time.time() - t0


async def prepare_inputs_stage(
    task: PipelineTask,
    *,
    planner,
    use_planner: bool,
    http_client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    status: dict[str, str],
) -> None:
    """Always download the reference image. Optionally run the planner.

    When `use_planner=False` (or planner=None), `task.osd` stays None and the
    coder/critic must work in pure-image mode (`actors.coder.multimodal=true`).
    """
    async with stage_guard(task, "prepare", sem, status):
        task.image_bytes, task.image_mime = await download_image(
            task.image_url, http_client
        )
        if use_planner and planner is not None:
            osd = await planner.plan(
                task_id=task.stem,
                image_bytes=task.image_bytes,
                image_url=task.image_url,
                mime=task.image_mime,
            )
            task.osd = osd.model_dump_json(indent=2)
        else:
            task.osd = None


async def code_and_check(
    task: PipelineTask,
    *,
    coder,
    js_checker,
    sem_coder: asyncio.Semaphore,
    sem_checker: asyncio.Semaphore,
    coder_multimodal: bool,
    status: dict[str, str],
    last_report=None,
    render_from_object: bool = False,
) -> None:
    """Code → JS check → retry-on-js-error loop. Returns when js_valid is True."""
    checker_mode = "with_object" if render_from_object else "sanity"
    osd = OSD.model_validate_json(task.osd) if task.osd is not None else None
    send_image = (osd is None) or coder_multimodal
    attempt = 0
    while True:
        async with stage_guard(task, "coder", sem_coder, status):
            if task.iteration == 0 and attempt == 0:
                task.js_code = await coder.code(
                    task_id=task.stem,
                    osd=osd,
                    image_bytes=task.image_bytes if send_image else None,
                    image_mime=task.image_mime,
                )
            elif last_report is not None and attempt == 0:
                task.js_code = await coder.code_critic_repair(
                    task_id=task.stem,
                    osd=osd,
                    issues=last_report.issues,
                    overall_score=last_report.overall_score,
                    matching_aspects=list(
                        getattr(last_report, "matching_aspects", []) or []
                    ),
                    image_bytes=task.image_bytes if coder_multimodal else None,
                    image_mime=task.image_mime,
                    render_png=task.rendered_png if coder_multimodal else None,
                )
            else:
                task.js_code = await coder.code_repair(
                    task_id=task.stem,
                    osd=osd,
                    js_errors=list(task.js_errors or []),
                )

        _sd = __import__("os").environ.get("SAVE_DRAFTS_DIR")   # [patch_loop] edit 8 (opt-in): keep every draft's code
        if _sd and task.js_code:
            try:
                _p = __import__("pathlib").Path(_sd)
                _p.mkdir(parents=True, exist_ok=True)
                (_p / f"{task.stem}.it{task.iteration}.a{attempt}.js").write_text(task.js_code)
            except Exception:
                pass
        async with stage_guard(task, "js_checker", sem_checker, status):
            task.js_valid = None
            task.js_errors = []
            task.scene_json = None
            await js_checker.process(task, mode=checker_mode)

        if task.js_valid:
            return

        attempt += 1


async def renderer_stage(
    task: PipelineTask,
    *,
    renderer,
    sem: asyncio.Semaphore,
    status: dict[str, str],
) -> None:
    async with stage_guard(task, "renderer", sem, status):
        task.render_errors = []
        task.failed = False
        task.failure_reason = None

        await renderer.process(task)

        if task.failed or task.rendered_png is None:
            reason = task.failure_reason or (
                task.render_errors[0] if task.render_errors else "no png"
            )
            raise StageError("renderer", RuntimeError(reason))


async def critic_stage(
    task: PipelineTask,
    *,
    critic,
    sem: asyncio.Semaphore,
    status: dict[str, str],
):
    async with stage_guard(task, "critic", sem, status):
        if (
            task.image_bytes is None
            or task.js_code is None
            or task.rendered_png is None
        ):
            raise StageError("critic", RuntimeError("missing inputs"))

        osd = OSD.model_validate_json(task.osd) if task.osd is not None else None
        report = await critic.critique(
            task_id=task.stem,
            image_bytes=task.image_bytes,
            image_mime=task.image_mime,
            render_png=task.rendered_png,
            artifact_context={
                "kind": "coder_v1",
                "js_code": task.js_code,
                "osd": osd.model_dump() if osd is not None else None,
            },
        )
        return report


def _code_digest(js_code: str | None) -> str | None:
    """Stable content hash used to fold byte-identical candidate programs onto one."""
    if not js_code:
        return None
    return hashlib.sha256(js_code.encode("utf-8")).hexdigest()


def _is_live(cand: Candidate | None) -> bool:
    """A candidate still in contention: it produced code that rendered cleanly."""
    return (
        cand is not None
        and cand.drop_reason is None
        and cand.rendered_png is not None
    )


def _save_semi(stem: str, cand, task) -> None:   # [patch_loop] edit 13
    import json as _j
    from pathlib import Path as _P
    try:
        d = _P(__import__("os").environ["SAVE_SEMIS_DIR"]) / stem
        kd = d / f"k{cand.k}"
        if (kd / "done").exists():
            return
        for sub in ("white", "gray"):
            (kd / sub).mkdir(parents=True, exist_ok=True)
        if not (d / "photo.bin").exists():
            (d / "photo.bin").write_bytes(task.image_bytes or b"")
            (d / "photo.json").write_text(_j.dumps({"mime": task.image_mime}))
        (kd / "grid.png").write_bytes(cand.rendered_png or b"")
        for sub, views in (("white", cand.judge_white_views), ("gray", cand.judge_gray_views)):
            for name, b in (views or {}).items():
                (kd / sub / f"{name}.png").write_bytes(b)
        if cand.judge_embeddings:
            (kd / "emb.bin").write_bytes(cand.judge_embeddings)
        (kd / "code.js").write_text(cand.js_code or "")
        (kd / "done").write_text("")
    except Exception as exc:
        logger.warning(f"[SAVE_SEMIS] {stem} k{getattr(cand, 'k', '?')}: {exc}")


def _save_duel(stem: str, rec: dict) -> None:   # [patch_loop] edit 14
    try:
        d = __import__("pathlib").Path(__import__("os").environ["SAVE_SEMIS_DIR"]) / stem
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "duels.jsonl", "a") as f:
            f.write(__import__("json").dumps(rec, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"[SAVE_SEMIS] {stem} duel: {exc}")


def _rr_top4_on() -> bool:   # [patch_loop] edit 15
    return __import__("os").environ.get("ROUND_ROBIN_TOP4", "1").strip() != "0"


def _rr_is_draw(decided_by: str) -> bool:
    return "DINO tie-break" in decided_by or "draw -> A" in decided_by


def _rr_pick(top: list, res: dict, bracket_k):
    """[patch_loop] edit 15: res[(i, j)] = (winner position, draw) for every pair i < j of top. Returns (pick, score)."""
    score = {k: 0 for k in top}
    for (i, j), (w, draw) in sorted(res.items()):
        if not draw:
            score[w] += 1
            score[j if w == i else i] -= 1
    best = max(score.values())
    tied = sorted(k for k, s in score.items() if s == best)
    if len(tied) == 1:
        return tied[0], score
    if len(tied) == 2 and not res[(tied[0], tied[1])][1]:
        return res[(tied[0], tied[1])][0], score
    return (bracket_k if bracket_k in tied else tied[0]), score


# [R86] PGH_TIE_NEUTRAL=1 (default off = pgh17, additions only): pgh17's _rr_pick above breaks an RR tie the bracket
# winner is not in by the LOWEST k = a position (with PGH_MIX_SPLIT 0.5 even k = UtevoLux). The wrapper keeps every
# other outcome and replaces only that fallback by the best DINO best-view similarity, then a sha256(stem|k) coin.
_rr_pick_pgh17 = _rr_pick   # [R86]
_R86_TIE: dict = {}   # [R86] {"sims": {k: s}, "stem": str}, set right before the RR call (no await in between)


def _tie_neutral() -> bool:   # [R86]
    return __import__("os").environ.get("PGH_TIE_NEUTRAL", "").strip().lower() in ("1", "on", "true")


def _rr_pick(top: list, res: dict, bracket_k, sims: dict | None = None, stem: str | None = None):   # [R86] wrapper
    pick, score = _rr_pick_pgh17(top, res, bracket_k)
    if not _tie_neutral() or not score:
        return pick, score
    best = max(score.values())
    tied = sorted(k for k, s in score.items() if s == best)
    if len(tied) == 1 or bracket_k in tied or (len(tied) == 2 and not res[(tied[0], tied[1])][1]):
        return pick, score   # decided by results / the bracket's duel, not by position
    if sims is None:
        sims = _R86_TIE.get("sims") or {}
    return _neutral_tie_pick(tied, sims, stem if stem is not None else _R86_TIE.get("stem", "")), score


def _neutral_tie_pick(tied: list, sims: dict, stem: str):   # [R86]
    """Among tied keys: the highest best-view DINO similarity (the judge's own draw signal); missing/equal -> a
    sha256 coin over (stem, k). Never depends on the key order."""
    have = {k: sims.get(k) for k in tied if sims.get(k) is not None}
    if have:
        top = max(have.values())
        best = sorted(k for k, v in have.items() if v == top)
        if len(best) == 1:
            return best[0]
        tied = best
    return sorted(tied, key=lambda k: hashlib.sha256(f"{stem}|{k}".encode()).hexdigest())[0]


async def _round_robin_top4(task, n: int, winner_of, judge_duel):
    """[patch_loop] edit 15: the leader's bracket over positions [0, n) with the R5/R6 level replaced by the bracket's own
    semis + final plus the other pairs of the top set; returns the round-robin pick (the bracket winner on any fallback)."""
    env = __import__("os").environ
    try:
        cutoff = float(env.get("ROUND_ROBIN_CUTOFF_S", "2400") or 0)
    except ValueError:
        cutoff = 2400.0
    mid = n // 2   # the leader's _winner_of split: halves [0, mid) [mid, n), quarters split each half again
    blocks = [(0, mid // 2), (mid // 2, mid), (mid, (mid + n) // 2), ((mid + n) // 2, n)]
    rnd_half = ((mid - 1).bit_length(), (n - mid - 1).bit_length())
    rnd_final = (n - 1).bit_length()
    quarter = [asyncio.ensure_future(winner_of(lo, hi)) for lo, hi in blocks]
    res: dict = {}
    errors: dict = {}
    s1rec: dict = {}   # [R84 jalign] (left k, right k) -> S1 slim detail, for PGH_RR_CUTOFF_FALLBACK

    async def duel(a, b, rnd, extra):
        out = {"label": f"RR k{a.k}-vs-k{b.k}"} if extra else {}
        w = await judge_duel(a, b, rnd, out)
        res[(a.k, b.k)] = (w.k, _rr_is_draw(out.get("decided_by", "")))
        s1rec[(a.k, b.k)] = out.get("s1")
        return w

    async def half(h):   # = the leader's _winner_of(half): walkover rules, then the semi
        a, b = await asyncio.gather(quarter[2 * h], quarter[2 * h + 1])
        if not _is_live(a):
            return b
        if not _is_live(b):
            return a
        return await duel(a, b, rnd_half[h], False)

    halves = [asyncio.ensure_future(half(0)), asyncio.ensure_future(half(1))]

    async def cross(qa, qb):   # an RR-only pair: starts once it is known not to be the bracket's final
        a, b = await asyncio.shield(quarter[qa]), await asyncio.shield(quarter[qb])
        if not (_is_live(a) and _is_live(b)):
            return
        own = {halves[0]: a, halves[1]: b}
        rest = set(own)
        while rest:
            done, rest = await asyncio.wait(rest, return_when=asyncio.FIRST_COMPLETED)
            if any(t.result() is not own[t] for t in done):
                break
        else:
            return   # both won their half: the bracket's final, played below
        try:
            await duel(a, b, rnd_final, True)
        except Exception as exc:   # an RR-only duel never fails the prompt: the bracket winner is kept
            errors[(a.k, b.k)] = f"{type(exc).__name__}: {exc}"[:200]

    extras = [asyncio.ensure_future(cross(qa, qb)) for qa in (0, 1) for qb in (2, 3)]
    why = None
    try:   # the bracket exactly as the leader resolves its last level (errors here propagate as in the leader)
        left, right = await asyncio.gather(*halves)
        if not _is_live(left):
            bracket = right
        elif not _is_live(right):
            bracket = left
        else:
            bracket = await duel(left, right, rnd_final, False)
        t_b = time.monotonic()
        top = [q.result() for q in quarter if _is_live(q.result())]
        if len(top) >= 3:
            remaining = None if cutoff <= 0 else (task.started_at or time.time()) + cutoff - time.time()
            done, pending = await asyncio.wait(extras, timeout=None if remaining is None else max(0.0, remaining))
            if pending:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                why = f"cutoff {cutoff:.0f}s ({len(pending)} RR duels cancelled)"
            for t in done:
                if not t.cancelled() and t.exception() is not None:
                    errors[("?", id(t))] = f"{type(t.exception()).__name__}: {t.exception()}"[:200]
    finally:   # never leave an RR duel running (cancellation, a bracket error, or < 3 players = no RR pair at all)
        for t in extras:
            if not t.done():
                t.cancel()
            elif not t.cancelled():
                t.exception()   # retrieved (no "never retrieved" noise after a bracket error)
    keys = [c.k for c in top]
    pairs = [(i, j) for x, i in enumerate(keys) for j in keys[x + 1:]]
    if errors:
        why = "duel error " + "; ".join(f"{p}: {e}" for p, e in sorted(errors.items(), key=str))
    elif why is None and len(top) >= 3 and any(p not in res for p in pairs):
        why = f"missing {[p for p in pairs if p not in res]}"
    if len(top) < 3 or why is not None or not _is_live(bracket):
        pick, score = bracket, {}
        pick = _cutoff_fallback(task, top, s1rec, bracket, why, pick)   # [R84 jalign] off: returns pick unchanged
    else:
        if _tie_neutral():   # [R86] DINO sims for a neutral multi-way tie (off: never computed); no await until the call
            try:
                from pipeline.orient_final import _bvs
                _R86_TIE.update(sims={c.k: _bvs(c.judge_embeddings) for c in top}, stem=task.stem)
            except Exception:
                _R86_TIE.update(sims={}, stem=task.stem)
        pk, score = _rr_pick(keys, res, bracket.k)
        pick = next(c for c in top if c.k == pk)
    wins = {k: sum(1 for (w, d) in res.values() if w == k and not d) for k in keys}
    logger.info(
        f"[ROUND_ROBIN] {task.stem} top={keys} wins={wins} score={score} draws={sum(d for _, d in res.values())} "
        f"pick=k{getattr(pick, 'k', None)} bracket=k{getattr(bracket, 'k', None)} "
        f"changed={'yes' if pick is not bracket else 'no'} duels={len(res) + len(errors)} "
        f"rr_only={max(0, len(res) + len(errors) - (len(top) - 1))} wait_after_bracket={time.monotonic() - t_b:.1f}s"
        + (f" FALLBACK={why}" if why else "")
    )
    return pick


def _cutoff_fallback(task, top: list, s1rec: dict, bracket, why, pick):   # [R84 jalign] PGH_RR_CUTOFF_FALLBACK
    """Only when the RR cutoff fired (why starts with "cutoff"), the bracket is live, >= 3 players and the switch is
    "s1pen": the top-set player with the lowest mean S1 penalty over its finished top-set duels. Else `pick`."""
    if not (why and str(why).startswith("cutoff") and len(top) >= 3 and _is_live(bracket)):
        return pick
    if not __import__("os").environ.get("PGH_RR_CUTOFF_FALLBACK"):
        return pick
    import pgh_jalign as _ja
    if _ja.cutoff_fallback() != "s1pen":
        return pick
    try:
        pk, means = _ja.s1pen_pick_key([c.k for c in top], s1rec, bracket.k)
        new = next(c for c in top if c.k == pk)
    except Exception as exc:   # never fail the prompt over the fallback
        logger.warning(f"[RR_CUTOFF_FALLBACK] {task.stem} error {type(exc).__name__}: {exc} -> bracket")
        return pick
    logger.info(
        f"[RR_CUTOFF_FALLBACK] {task.stem} mode=s1pen pick=k{new.k} bracket=k{bracket.k} "
        f"changed={'yes' if new is not bracket else 'no'} means={ {k: round(v, 3) for k, v in sorted(means.items())} } "
        f"why={why}"
    )
    return new


async def _round_robin_top8(task, n: int, winner_of, judge_duel):
    """[R84 jalign] PGH_RR8: the same bracket as _round_robin_top4 / winner_of over positions [0, n) (same splits,
    rounds, seats and walkovers; the bracket winner is unchanged), but the round robin covers the 8 eighth-block
    winners: every pair the bracket does not play is an extra duel ("RR8 k<a>-vs-k<b>", left = lower block, round =
    the final's, so no S1 early stop), started as soon as it is known not to be a bracket duel. Pick = _rr_pick over
    all pairs among the live eighth winners (up to 28). Any extra-duel error, a missing pair or the cutoff -> the
    bracket winner (or PGH_RR_CUTOFF_FALLBACK on a cutoff), exactly as top-4."""
    env = __import__("os").environ
    try:
        cutoff = float(env.get("ROUND_ROBIN_CUTOFF_S", "2400") or 0)
    except ValueError:
        cutoff = 2400.0
    mid = n // 2
    qblocks = [(0, mid // 2), (mid // 2, mid), (mid, (mid + n) // 2), ((mid + n) // 2, n)]
    eblocks = []
    for lo, hi in qblocks:   # winner_of's own split of each quarter
        m = (lo + hi) // 2
        eblocks += [(lo, m), (m, hi)]
    rnd_q = [(hi - lo - 1).bit_length() for lo, hi in qblocks]
    rnd_half = ((mid - 1).bit_length(), (n - mid - 1).bit_length())
    rnd_final = (n - 1).bit_length()
    eighth = [asyncio.ensure_future(winner_of(lo, hi)) for lo, hi in eblocks]
    res: dict = {}
    errors: dict = {}
    s1rec: dict = {}

    async def duel(a, b, rnd, extra):
        out = {"label": f"RR8 k{a.k}-vs-k{b.k}"} if extra else {}
        w = await judge_duel(a, b, rnd, out)
        res[(a.k, b.k)] = (w.k, _rr_is_draw(out.get("decided_by", "")))
        s1rec[(a.k, b.k)] = out.get("s1")
        return w

    async def quarter(q):   # = winner_of(qblocks[q]): walkover rules, then the quarter duel
        a, b = await asyncio.gather(eighth[2 * q], eighth[2 * q + 1])
        if not _is_live(a):
            return b
        if not _is_live(b):
            return a
        return await duel(a, b, rnd_q[q], False)

    quarters = [asyncio.ensure_future(quarter(q)) for q in range(4)]

    async def half(h):
        a, b = await asyncio.gather(quarters[2 * h], quarters[2 * h + 1])
        if not _is_live(a):
            return b
        if not _is_live(b):
            return a
        return await duel(a, b, rnd_half[h], False)

    halves = [asyncio.ensure_future(half(0)), asyncio.ensure_future(half(1))]

    async def cross(i, j):   # eighth winners i < j from different quarters: an extra pair unless the bracket plays it
        a, b = await asyncio.shield(eighth[i]), await asyncio.shield(eighth[j])
        if not (_is_live(a) and _is_live(b)):
            return
        if i // 4 == j // 4:   # same half: the bracket plays it as that half's semi iff both win their quarter
            own = {quarters[i // 2]: a, quarters[j // 2]: b}
        else:   # different halves: the bracket's final iff both win their quarter and their half
            own = {quarters[i // 2]: a, halves[i // 4]: a, quarters[j // 2]: b, halves[j // 4]: b}
        rest = set(own)
        while rest:
            done, rest = await asyncio.wait(rest, return_when=asyncio.FIRST_COMPLETED)
            if any(t.result() is not own[t] for t in done):
                break
        else:
            return
        rr_only.add((a.k, b.k))
        try:
            await duel(a, b, rnd_final, True)
        except Exception as exc:   # an RR-only duel never fails the prompt: the bracket winner is kept
            errors[(a.k, b.k)] = f"{type(exc).__name__}: {exc}"[:200]

    rr_only: set = set()
    extras = [asyncio.ensure_future(cross(i, j)) for i in range(8) for j in range(i + 1, 8) if i // 2 != j // 2]
    why = None
    top: list = []
    t_b = time.monotonic()
    try:
        left, right = await asyncio.gather(*halves)
        if not _is_live(left):
            bracket = right
        elif not _is_live(right):
            bracket = left
        else:
            bracket = await duel(left, right, rnd_final, False)
        t_b = time.monotonic()
        top = [e.result() for e in eighth if _is_live(e.result())]
        if len(top) >= 3:
            remaining = None if cutoff <= 0 else (task.started_at or time.time()) + cutoff - time.time()
            done, pending = await asyncio.wait(extras, timeout=None if remaining is None else max(0.0, remaining))
            if pending:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                why = f"cutoff {cutoff:.0f}s ({len(pending)} RR duels cancelled)"
            for t in done:
                if not t.cancelled() and t.exception() is not None:
                    errors[("?", id(t))] = f"{type(t.exception()).__name__}: {t.exception()}"[:200]
    finally:
        for t in extras:
            if not t.done():
                t.cancel()
            elif not t.cancelled():
                t.exception()
    keys = [c.k for c in top]
    pairs = [(i, j) for x, i in enumerate(keys) for j in keys[x + 1:]]
    if errors:
        why = "duel error " + "; ".join(f"{p}: {e}" for p, e in sorted(errors.items(), key=str))
    elif why is None and len(top) >= 3 and any(p not in res for p in pairs):
        why = f"missing {[p for p in pairs if p not in res]}"
    if len(top) < 3 or why is not None or not _is_live(bracket):
        pick, score = bracket, {}
        pick = _cutoff_fallback(task, top, s1rec, bracket, why, pick)
    else:
        if _tie_neutral():   # [R87 pgh17c] R86's DINO sims for RR8 too (else a tie would read the last top-4 photo's sims)
            try:
                from pipeline.orient_final import _bvs
                _R86_TIE.update(sims={c.k: _bvs(c.judge_embeddings) for c in top}, stem=task.stem)
            except Exception:
                _R86_TIE.update(sims={}, stem=task.stem)
        pk, score = _rr_pick(keys, res, bracket.k)
        pick = next(c for c in top if c.k == pk)
    wins = {k: sum(1 for (w, d) in res.values() if w == k and not d and k in keys) for k in keys}
    logger.info(
        f"[ROUND_ROBIN] {task.stem} RR8 top={keys} wins={wins} score={score} draws={sum(d for _, d in res.values())} "
        f"pick=k{getattr(pick, 'k', None)} bracket=k{getattr(bracket, 'k', None)} "
        f"changed={'yes' if pick is not bracket else 'no'} duels={len(res) + len(errors)} "
        f"rr_only={sum(1 for p in res if p in rr_only)} "
        f"wait_after_bracket={time.monotonic() - t_b:.1f}s"
        + (f" FALLBACK={why}" if why else "")
    )
    return pick


async def _resolve_bracket(
    *,
    task: PipelineTask,
    leaf_tasks: dict[int, asyncio.Task],
    judge,
    sem_judge: asyncio.Semaphore,
) -> Candidate | None:
    """Streaming single-elimination over fixed candidate positions.

    Each position is resolved by its own leaf task (coder -> dedup -> render); a
    position that dropped is a walkover for its opponent, costing no judge call.
    A duel fires the moment both of its subtree winners are known, so judging
    overlaps rendering. Pairing is by position, independent of which candidates
    drop, so the outcome is reproducible under a fixed seed.
    """

    async def _judge_duel(left: Candidate, right: Candidate, round_no: int, _rr: dict | None = None) -> Candidate:
        """Compare two live candidates head-to-head and return the winner."""
        label = (_rr or {}).get("label") or f"R{round_no} k{left.k}-vs-k{right.k}"   # [patch_loop] edit 15: "RR ..." extras
        if __import__("os").environ.get("SAVE_SEMIS_DIR") and round_no >= int(__import__("os").environ.get("SAVE_SEMIS_MIN_ROUND") or 4):   # [patch_loop] edit 13: fire and forget (top 8; MIN_ROUND=1: every duel player)
            for _c in (left, right):
                asyncio.get_running_loop().run_in_executor(None, _save_semi, task.stem, _c, task)
        _t_q = time.monotonic()
        async with sem_judge:
            _t_s = time.monotonic()
            verdict = await judge.compare(
                task_id=task.stem,
                match_label=label,
                reference_bytes=task.image_bytes,
                reference_mime=task.image_mime,
                render_a=left.rendered_png,
                render_b=right.rendered_png,
                white_views_a=left.judge_white_views,
                white_views_b=right.judge_white_views,
                gray_views_a=left.judge_gray_views,
                gray_views_b=right.judge_gray_views,
                embeddings_a=left.judge_embeddings,
                embeddings_b=right.judge_embeddings,
                early_stop=judge.early_stop_for_round(round_no),
            )
        winner = left if verdict.winner == "A" else right
        if _salv is not None and _salv.ON:   # [R96 pgh17d] the bracket leader so far
            _salv.note_duel(task, round_no, winner, right if winner is left else left)
        if _rr is not None:   # [patch_loop] edit 15: the round robin needs the verdict's draw marker
            _rr["decided_by"] = str(getattr(verdict, "decided_by", "") or "")
            _rr["s1"] = (getattr(verdict, "detail", None) or {}).get("s1_slim")   # [R84 jalign] cutoff fallback input
        if __import__("os").environ.get("SAVE_SEMIS_DIR"):   # [patch_loop] edit 14: per-duel S1 detail, fire and forget
            asyncio.get_running_loop().run_in_executor(None, _save_duel, task.stem, {
                "label": label, "l": left.k, "r": right.k, "w": winner.k, "decided_by": str(getattr(verdict, "decided_by", "")),
                "rnd": round_no, **{k: (getattr(verdict, "detail", None) or {}).get(f"{k}_slim") for k in ("s1", "s2", "s3")}})
        _t_e = time.monotonic()
        STATS.duel(_t_e - _t_s, _t_s - _t_q)
        logger.info(
            f"[BRACKET {label}] {task.stem} -> k{winner.k} | "
            f"[JUDGE_TIMING] queue_wait={_t_s - _t_q:.1f}s compare={_t_e - _t_s:.1f}s in_flight={sem_judge._value}"
        )
        return winner

    async def _winner_of(lo: int, hi: int) -> Candidate | None:
        """Winner of candidate positions [lo, hi): one leaf, or the better half."""
        if hi - lo <= 1:
            return await leaf_tasks[lo]

        mid = (lo + hi) // 2
        left, right = await asyncio.gather(_winner_of(lo, mid), _winner_of(mid, hi))

        # A half with no live candidate is a walkover — the other side advances free.
        if not _is_live(left):
            return right
        if not _is_live(right):
            return left

        round_no = (hi - lo - 1).bit_length()  # R1 = first-round pair, then upward
        return await _judge_duel(left, right, round_no)

    if not leaf_tasks:
        return None
    if __import__("os").environ.get("PGH_SHADOW"):   # [opt_r5 shadow] default off: record the verdicts, change nothing
        import pgh_shadow as _shadow
        _sh_rec = _shadow.recorder(task, judge, leaf_tasks) if _shadow.on() else None
        if _sh_rec is not None:
            _judge_duel = _sh_rec.wrap(_judge_duel)
    if _rr_top4_on() and len(leaf_tasks) >= 4:   # [patch_loop] edit 15: top-4 round robin on the bracket's last 4
        if __import__("os").environ.get("PGH_RR8") and len(leaf_tasks) >= 8:   # [R84 jalign] top-8 RR, default off
            import pgh_jalign as _ja
            if _ja.rr8_for(task.stem, len(leaf_tasks)):
                return await _round_robin_top8(task, len(leaf_tasks), _winner_of, _judge_duel)
        return await _round_robin_top4(task, len(leaf_tasks), _winner_of, _judge_duel)
    return await _winner_of(0, len(leaf_tasks))


def _promote_winner(
    task: PipelineTask,
    winner: Candidate,
    candidates: list[Candidate],
    session_store,
) -> None:
    """Copy the winning candidate onto the task and free the losers' sessions."""
    task.winner_k = winner.k
    task.js_code = winner.js_code
    task.js_valid = winner.js_valid
    task.js_errors = list(winner.js_errors)
    task.scene_json = winner.scene_json
    task.rendered_png = winner.rendered_png

    # Rename the winner's coder session to the canonical actor so downstream
    # repair stages find it; evict the losers to free memory.
    session_store.rename_actor(task.stem, f"coder#k{winner.k}", "coder")
    for cand in candidates:
        if cand.k != winner.k:
            session_store.evict_actor(task.stem, f"coder#k{cand.k}")


class _CandidateFactory:
    """Generates, deduplicates, validates and renders one task's candidates.

    Owns the per-task config and stage dependencies so the orchestrator can fan
    out candidates without threading a dozen arguments through nested closures.
    Each bracket position is resolved by `leaf`: it awaits that position's coder,
    folds byte-identical programs onto their lowest-k leader (seed-deterministic,
    so the bracket is reproducible), then validates the leader through js-check,
    render and judge views. Only leaders render; duplicates and coder failures
    are walkovers.
    """

    def __init__(
        self,
        task: PipelineTask,
        *,
        coder,
        judge,
        embedder,
        js_checker,
        renderer,
        sem_coder: asyncio.Semaphore,
        sem_checker: asyncio.Semaphore,
        sem_renderer: asyncio.Semaphore,
        status: dict[str, str],
        osd: OSD | None,
        send_image: bool,
        checker_mode: str,
        ensemble_temperature: float,
        seed_offset: int = 0,
        orientation: OrientationConfig | None = None,
        candidate_deadline_s: float = 0.0,
    ) -> None:
        self.task = task
        self.seed_offset = seed_offset
        self.orientation = orientation or OrientationConfig()
        # Absolute monotonic-free deadline: task.started_at is time.time() set by the orchestrator.
        self.candidate_deadline_at = (
            (task.started_at or time.time()) + candidate_deadline_s if candidate_deadline_s > 0 else None
        )
        self.coder = coder
        self.judge = judge
        self.embedder = embedder
        self.js_checker = js_checker
        self.renderer = renderer
        self.sem_coder = sem_coder
        self.sem_checker = sem_checker
        self.sem_renderer = sem_renderer
        self.status = status
        self.osd = osd
        self.send_image = send_image
        self.checker_mode = checker_mode
        self.ensemble_temperature = ensemble_temperature
        self.ref_vec = None
        self._coder_tasks: dict[int, asyncio.Task] = {}

    def start_coders(self, k_count: int) -> dict[int, asyncio.Task]:
        """Launch one coder per candidate at once (sem_coder caps concurrency)."""
        self._burst_gen = _burst.CTRL.coder_begin(k_count, self.task.stem) if _burst.CTRL else 0
        self._coder_tasks = {
            k: asyncio.create_task(self._generate(k)) for k in range(k_count)
        }
        return self._coder_tasks

    def add_coders(self, k_from: int, k_to: int) -> None:   # [R82 kbudget] top-up drafts k_from..k_to-1 (same seeds as fixed)
        if k_to <= k_from:
            return
        if _burst.CTRL:
            _burst.CTRL.coder_begin(k_to - k_from, self.task.stem)
        for k in range(k_from, k_to):
            self._coder_tasks[k] = asyncio.create_task(self._generate(k))

    async def embed_reference(self) -> None:
        """Embed the reference image once, shared by every candidate's S2BV npz.

        Runs while the already-scheduled coders generate, so it adds no latency.
        """
        if self.embedder is not None and self.judge is not None and self.task.image_bytes:
            self.ref_vec = await self.embedder.embed_reference(self.task.image_bytes)

    async def leaf(self, k: int) -> Candidate | None:
        """Resolve bracket position k to a live candidate, or None for a walkover."""
        cand = await self._coder_tasks[k]
        digest = _code_digest(cand.js_code)
        if digest is None:
            return None  # coder failed -> walkover (drop_reason already set)

        # The dedup scan awaits every earlier coder, so candidate k is gated by the
        # slowest of 0..k-1 — the higher k, the longer the stall. Validation
        # (checker -> render -> judge views) does not depend on the scan result, so
        # it starts right away and overlaps it. The scan still covers all j < k, so
        # the outcome stays deterministic under a fixed seed; the only cost is
        # rendering a candidate that later turns out to be a duplicate.
        validate_task = asyncio.create_task(self._validate(cand))
        try:
            duplicate_of = None
            for j in range(k):
                if _code_digest((await self._coder_tasks[j]).js_code) == digest:
                    duplicate_of = j
                    break
        finally:
            await validate_task  # never leave the validation task dangling

        if duplicate_of is not None:
            cand.drop_reason = f"duplicate_of_k{duplicate_of}"
            return None  # the leader k=duplicate_of carries this code
        return cand if _is_live(cand) else None

    async def _generate(self, k: int) -> Candidate:
        try:
            return await self._generate_inner(k)
        finally:
            if _burst.CTRL:
                _burst.CTRL.coder_end(getattr(self, "_burst_gen", 0))

    async def _generate_inner(self, k: int) -> Candidate:
        task = self.task
        cand = Candidate(k=k, seed=task.seed + self.seed_offset + k)
        t0 = time.time()
        # Remaining wall-clock for this candidate, semaphore wait included: a candidate that
        # cannot even start before the deadline is a walkover, so the bracket never waits on
        # a wedged coder past candidate_deadline_s. None = unbounded (old behaviour).
        remaining = None
        if self.candidate_deadline_at is not None:
            remaining = max(0.0, self.candidate_deadline_at - time.time())
        try:
            async with asyncio.timeout(remaining):
                async with stage_guard(task, f"coder#k{k}", self.sem_coder, self.status):
                    if _kcpu is not None and _kcpu.ON and _kcpu.KCPU.brake(self, k, log=logger.info):   # [R96 pgh17d]
                        raise _kcpu.Braked("render backlog brake")
                    if _burst.CTRL:
                        await _burst.CTRL.ensure_coder()
                    cand.js_code = await self.coder.code(
                        task_id=task.stem,
                        osd=self.osd,
                        image_bytes=task.image_bytes if self.send_image else None,
                        image_mime=task.image_mime,
                        actor_override=f"coder#k{k}",
                        seed_override=cand.seed,
                        temperature_override=self.ensemble_temperature,
                    )
        except TimeoutError:
            cand.drop_reason = "coder:deadline"
            logger.warning(
                f"[CANDIDATE DEADLINE] {task.stem} k{k} dropped after {time.time() - t0:.0f}s "
                f"(candidate_deadline_s elapsed)"
            )
        except StageError as exc:
            cand.drop_reason = f"coder:{type(exc.cause).__name__}"
        except Exception as exc:
            cand.drop_reason = f"coder:{type(exc).__name__}"
        cand.elapsed_s = time.time() - t0
        return cand

    async def _try_three_repair(self, cand: Candidate, shadow: PipelineTask) -> PipelineTask:   # [pgh16]
        return await _try_three_repair_impl(self, cand, shadow)

    async def _validate(self, cand: Candidate) -> None:
        if cand.drop_reason is not None or cand.js_code is None:
            return
        task = self.task
        _sd = __import__("os").environ.get("SAVE_DRAFTS_DIR")   # [patch_loop] edit 8 (ensemble path): every raw draft
        if _sd:
            try:
                _p = __import__("pathlib").Path(_sd)
                _p.mkdir(parents=True, exist_ok=True)
                (_p / f"{task.stem}.k{cand.k}.js").write_text(cand.js_code)
            except Exception:
                pass
        shadow = PipelineTask(stem=f"{task.stem}#k{cand.k}", image_url=task.image_url)
        shadow.js_code = cand.js_code
        shadow.image_bytes = task.image_bytes
        shadow.image_mime = task.image_mime
        t0 = time.time()
        async with self.sem_checker:
            try:
                await self.js_checker.process(shadow, mode=self.checker_mode)
            except Exception as exc:
                cand.drop_reason = f"checker:{type(exc).__name__}"
                cand.elapsed_s += time.time() - t0
                return
        if not shadow.js_valid and _mech_repair.timeout_retry_enabled() and _mech_repair.timeout_only(shadow.js_errors):   # [R81]
            shadow = await _recheck_timeout_impl(self, cand, shadow)
        if not shadow.js_valid and _mech_repair.repair_enabled():   # [R81] opt-in; off = this line does nothing
            shadow = await _try_mech_repair_impl(self, cand, shadow)
        if not shadow.js_valid and _three_repair.repair_enabled():   # [pgh16] opt-in; off = this line does nothing
            shadow = await self._try_three_repair(cand, shadow)
        cand.js_valid = shadow.js_valid
        cand.js_errors = list(shadow.js_errors or [])
        cand.scene_json = shadow.scene_json
        if not shadow.js_valid:
            cand.drop_reason = "checker"
            cand.elapsed_s += time.time() - t0
            return
        if _render_late(self):   # [patch_loop] edit 11 (opt-in): walkover instead of a late render
            cand.drop_reason = "render:deadline"
            cand.elapsed_s += time.time() - t0
            logger.warning(f"[RENDER DEADLINE] {task.stem} k{cand.k} dropped before the probe")
            return
        if self.orientation.enabled:
            result = await orient_task(
                shadow, renderer=self.renderer, embedder=self.embedder, ref_vec=self.ref_vec,
                js_checker=self.js_checker, checker_mode=self.checker_mode, config=self.orientation,
                sem_render=self.sem_renderer, sem_check=self.sem_checker,
            )
            task.meta.setdefault("orientation", {})[cand.k] = result.model_dump(mode="json")
            if result.applied:
                cand.js_code = shadow.js_code
                cand.scene_json = shadow.scene_json
        judge_views_wanted = (
            self.judge is not None
            and getattr(self.renderer, "config", None) is not None
            and getattr(self.renderer.config, "judge_multiview", False)
            and hasattr(self.renderer, "process_with_judge_views")
        )
        white: dict = {}
        gray: dict = {}
        # [patch_loop] edit 10: a speculative final kept by the probe (SPEC_FINAL=1, no rotation) needs no render slot
        _spec_ok = (judge_views_wanted and getattr(self.renderer, "spec_ready", None) is not None
                    and self.renderer.spec_ready(shadow))
        async with (__import__("contextlib").nullcontext() if _spec_ok else self.sem_renderer):
            if not _spec_ok and _render_late(self):   # [patch_loop] edit 11 (opt-in), checked once the slot is ours
                cand.drop_reason = "render:deadline"
                cand.elapsed_s += time.time() - t0
                logger.warning(f"[RENDER DEADLINE] {task.stem} k{cand.k} dropped before the final render")
                return
            self.__dict__["_finals_admitted"] = self.__dict__.get("_finals_admitted", 0) + 1
            try:
                if judge_views_wanted:
                    white, gray = await self.renderer.process_with_judge_views(shadow)
                else:
                    await self.renderer.process(shadow)
            except Exception as exc:
                cand.drop_reason = f"renderer:{type(exc).__name__}"
                cand.elapsed_s += time.time() - t0
                return
        cand.rendered_png = shadow.rendered_png
        cand.render_errors = list(shadow.render_errors or [])
        if not shadow.rendered_png:
            cand.drop_reason = "renderer"
        cand.elapsed_s += time.time() - t0

        if cand.drop_reason is None and white:
            cand.judge_white_views, cand.judge_gray_views = white, gray
            if self.embedder is not None and self.ref_vec is not None:
                cand.judge_embeddings = await self.embedder.build_candidate_npz(
                    self.ref_vec, cand.judge_white_views
                )


async def _try_three_repair_impl(factory, cand: Candidate, shadow: PipelineTask) -> PipelineTask:
    """[pgh16] PGH_THREE_REPAIR=1: a draft rejected ONLY for non-existent THREE constants gets them mapped to allowed
    ones (modules/js_checker/three_repair.py) and is checked ONCE more by the same checker in the same mode. Returns the
    new, passing shadow task (cand.js_code is replaced) or the original failed shadow (cand unchanged)."""
    task = factory.task
    names = _three_repair.plan(shadow.js_errors)
    if not names:
        return shadow
    fixed, notes = _three_repair.rewrite(cand.js_code, names)
    if fixed is None:
        logger.info(f"[THREE_REPAIR] {task.stem} k{cand.k} SKIP {notes[0]}")
        return shadow
    retry = PipelineTask(stem=shadow.stem, image_url=task.image_url)
    retry.js_code = fixed
    retry.image_bytes = task.image_bytes
    retry.image_mime = task.image_mime
    try:
        async with factory.sem_checker:
            await factory.js_checker.process(retry, mode=factory.checker_mode)
    except Exception as exc:
        logger.warning(f"[THREE_REPAIR] {task.stem} k{cand.k} recheck error {type(exc).__name__}")
        return shadow
    if not retry.js_valid:
        logger.info(f"[THREE_REPAIR] {task.stem} k{cand.k} FAILED_RECHECK {', '.join(notes)} | {(retry.js_errors or ['?'])[0][:120]}")
        return shadow
    logger.info(f"[THREE_REPAIR] {task.stem} k{cand.k} REPAIRED {', '.join(notes)}")
    cand.js_code = fixed
    task.meta.setdefault("three_repair", {})[cand.k] = notes
    return retry


async def _recheck_once_impl(factory, cand: Candidate, shadow: PipelineTask, code: str, tag: str):
    """[R81] one more run of the SAME checker (same mode, same semaphore) on `code`; returns the new task or None on error."""
    task = factory.task
    retry = PipelineTask(stem=shadow.stem, image_url=task.image_url)
    retry.js_code = code
    retry.image_bytes = task.image_bytes
    retry.image_mime = task.image_mime
    try:
        async with factory.sem_checker:
            await factory.js_checker.process(retry, mode=factory.checker_mode)
    except Exception as exc:
        logger.warning(f"[{tag}] {task.stem} k{cand.k} recheck error {type(exc).__name__}")
        return None
    return retry


async def _recheck_timeout_impl(factory, cand: Candidate, shadow: PipelineTask) -> PipelineTask:
    """[R81] PGH_CHECK_TIMEOUT_RETRY=1: a draft whose ONLY failures are TIMEOUT_EXCEEDED (CPU contention) is checked once
    more, unchanged. Returns the passing re-check or the original failed shadow."""
    retry = await _recheck_once_impl(factory, cand, shadow, cand.js_code, "TIMEOUT_RETRY")
    ok = retry is not None and retry.js_valid
    logger.info(f"[TIMEOUT_RETRY] {factory.task.stem} k{cand.k} {'PASS' if ok else 'FAIL'}")
    if not ok:
        return shadow
    factory.task.meta.setdefault("timeout_retry", {})[cand.k] = True
    return retry


async def _try_mech_repair_impl(factory, cand: Candidate, shadow: PipelineTask) -> PipelineTask:
    """[R81] PGH_MECH_REPAIR=1: a draft rejected for ONE repairable error class (modules/js_checker/mech_repair.py:
    duplicate declaration, extra ')', undeclared names, const reassignment, NaN bounds, texture > 4 MiB) is rewritten and
    checked ONCE more by the same checker in the same mode. Returns the new, passing shadow task (cand.js_code is
    replaced) or the original failed shadow (cand unchanged)."""
    task = factory.task
    fixed, notes = await asyncio.to_thread(_mech_repair.repair, cand.js_code, shadow.js_errors)   # off the event loop
    if fixed is None:
        if notes and notes[0] != "skip:class":
            logger.info(f"[MECH_REPAIR] {task.stem} k{cand.k} SKIP {notes[0]}")
        return shadow
    retry = await _recheck_once_impl(factory, cand, shadow, fixed, "MECH_REPAIR")
    if retry is None:
        return shadow
    if not retry.js_valid:
        logger.info(f"[MECH_REPAIR] {task.stem} k{cand.k} FAILED_RECHECK {', '.join(notes)} | {(retry.js_errors or ['?'])[0][:120]}")
        return shadow
    logger.info(f"[MECH_REPAIR] {task.stem} k{cand.k} REPAIRED {', '.join(notes)}")
    cand.js_code = fixed
    task.meta.setdefault("mech_repair", {})[cand.k] = notes
    return retry


def _render_late(factory) -> bool:
    """[patch_loop] edit 11: past RENDER_DEADLINE_S (from the prompt's start) and the prompt already has
    RENDER_DEADLINE_MIN_LIVE finals admitted. Env unset/0 = never (the leader's behaviour)."""
    env = __import__("os").environ
    try:
        limit = float(env.get("RENDER_DEADLINE_S", "0") or 0)
        min_live = int(env.get("RENDER_DEADLINE_MIN_LIVE", "20") or 20)
    except ValueError:
        return False
    if limit <= 0 or time.time() < (factory.task.started_at or time.time()) + limit:
        return False
    return factory.__dict__.get("_finals_admitted", 0) >= min_live


async def multigen_first_iter(
    task: PipelineTask,
    *,
    coder,
    judge,
    embedder=None,
    js_checker,
    renderer,
    session_store,
    sem_coder: asyncio.Semaphore,
    sem_checker: asyncio.Semaphore,
    sem_renderer: asyncio.Semaphore,
    sem_judge: asyncio.Semaphore,
    coder_multimodal: bool,
    status: dict[str, str],
    ensemble_size: int,
    ensemble_temperature: float,
    render_from_object: bool = False,
    seed_offset: int = 0,
    orientation: OrientationConfig | None = None,
    candidate_deadline_s: float = 0.0,
) -> None:
    """K-of-N generation + judge bracket. Replaces code_and_check + renderer
    on iteration 0 when `coder.ensemble_size > 1`.

    Mutates `task` so the rest of the iteration loop (critic → code_critic_repair)
    can continue on the winner.
    """

    osd = OSD.model_validate_json(task.osd) if task.osd is not None else None
    send_image = (osd is None) or coder_multimodal
    K = ensemble_size
    # [pgh1v3] PGH_TIMEGUARD (pgh_timeguard.py): this batch's K from the batch-boundary round guard; the pool stays the
    # prefix range(K). Switch off, shadow mode or any failure = K unchanged = pgh1v2.
    if __import__("os").environ.get("PGH_TIMEGUARD", "0").strip().lower() not in ("", "0", "off"):
        try:
            import pgh_timeguard as _tg
            if _tg.enabled():
                K = max(1, min(K, int(_tg.k_cap(K, task=task))))
        except Exception as exc:  # never block a prompt
            logger.warning(f"[TIMEGUARD] hook failed, K={K} kept: {exc!r}")
    if _kcpu is not None and _kcpu.ON:   # [R96 pgh17d] PGH_K_CPU_AWARE prior: this batch's CPU-quota K cap
        try:
            K = _kcpu.KCPU.cap_k(K)
        except Exception as exc:  # never block a prompt
            logger.warning(f"[KCPU] hook failed, K={K} kept: {exc!r}")

    logger.info(
        f"[MULTIGEN] {task.stem} K={K} | temperature={ensemble_temperature} | "
        f"multimodal={send_image} | osd={'yes' if osd else 'no'}"
    )

    factory = _CandidateFactory(
        task,
        coder=coder,
        judge=judge,
        embedder=embedder,
        js_checker=js_checker,
        renderer=renderer,
        sem_coder=sem_coder,
        sem_checker=sem_checker,
        sem_renderer=sem_renderer,
        status=status,
        osd=osd,
        send_image=send_image,
        checker_mode="with_object" if render_from_object else "sanity",
        ensemble_temperature=ensemble_temperature,
        seed_offset=seed_offset,
        orientation=orientation,
        candidate_deadline_s=candidate_deadline_s,
    )
    if _salv is not None and _salv.ON:   # [R96 pgh17d] the deadline branch reads this prompt's drafts from the factory
        _salv.register(task, factory)
    if _kcpu is not None and _kcpu.ON:   # [R96 pgh17d] semaphore sizes for the brake
        _kcpu.KCPU.plan(factory)

    _kb, _kb_plan = None, None   # [R82 kbudget] PGH_K_MODE=budget: equal coder tokens per photo (budget_k.py); else = pgh16
    if __import__("os").environ.get("PGH_K_MODE", "").strip().lower() == "budget":
        try:
            from modules.metrics import budget_k as _kb
            _kb_plan = _kb.ALLOC.plan(K)
        except Exception as exc:  # never block a prompt
            logger.warning(f"[KBUDGET] plan failed, K={K} kept: {exc!r}")
            _kb, _kb_plan = None, None
    _kb_on, _kb_K, _kb_hold = bool(_kb_plan and _kb_plan.get("active")), K, None
    if _kb_on:
        K = _kb_plan["k_start"]   # first wave = K_lo drafts; the top-up below sets K = K_i before any leaf/bracket exists

    # Launch every coder at once, then embed the reference while they run; each
    # candidate then flows coder -> dedup -> render -> bracket on its own, with no
    # barrier, so rendering and judging overlap the slower coders still in flight.
    coder_tasks = factory.start_coders(K)
    if _kb_on and _burst.CTRL:   # [R82 kbudget] placeholder: the burst controller never sees 0 pending before the top-up
        _kb_hold = _burst.CTRL.coder_begin(1, task.stem)
    await factory.embed_reference()

    if _kb_on:   # [R82 kbudget] drafts k < n0 back (seed order = unbiased sample) -> K_i, launch k = K_lo..K_i-1
        K_i = K
        try:
            firsts = await asyncio.gather(*(coder_tasks[k] for k in range(_kb_plan["n0"])))
            rem = None
            if factory.candidate_deadline_at is not None and candidate_deadline_s > 0:
                rem = max(0.0, factory.candidate_deadline_at - time.time()) / candidate_deadline_s
            dec = _kb.ALLOC.decide(_kb_K, [len(c.js_code.encode("utf-8")) for c in firsts if c.js_code],
                                   stem=task.stem, remaining_frac=rem)
            K_i = max(K, int(dec["k"]))
            task.meta["kbudget"] = dec
            logger.info(f"[KBUDGET] {task.stem} K={_kb_K} -> K_i={K_i} | {dec}")
        except Exception as exc:  # never block a prompt: fall back to the batch K
            logger.warning(f"[KBUDGET] decide failed, K_i={_kb_K}: {exc!r}")
            K_i = max(K, _kb_K)
        except BaseException:      # cancelled (task deadline / batch timeout): no top-up, release the placeholder
            if _kb_hold is not None:
                _burst.CTRL.coder_end(_kb_hold)
            raise
        factory.add_coders(K, K_i)
        if _kb_hold is not None:
            _burst.CTRL.coder_end(_kb_hold)
        K = K_i

    leaf_tasks: dict[int, asyncio.Task] = {
        k: asyncio.create_task(factory.leaf(k)) for k in range(K)
    }

    if judge is None or K <= 1:
        leaves = await asyncio.gather(*leaf_tasks.values())
        winner = next((c for c in leaves if _is_live(c)), None)
    else:
        winner = await _resolve_bracket(
            task=task, leaf_tasks=leaf_tasks, judge=judge, sem_judge=sem_judge,
        )

    # Every leaf (hence every coder) has resolved by the time the bracket returns.
    candidates = [coder_tasks[k].result() for k in range(K)]
    task.candidates = list(candidates)
    if _kb is not None:   # [R82 kbudget] this photo's mean draft bytes join the reference at batch end
        try:
            _kb.ALLOC.observe(task.stem, [len(c.js_code.encode("utf-8")) for c in candidates if c.js_code])
        except Exception as exc:
            logger.warning(f"[KBUDGET] observe failed: {exc!r}")

    if not _is_live(winner):
        drops = [c.drop_reason for c in candidates]
        logger.warning(
            f"[MULTIGEN] {task.stem} K={K} | drops={drops} | ALL CANDIDATES FAILED"
        )
        raise StageError("multigen", RuntimeError("all candidates failed"))

    if _orient_final.enabled():   # [R86] PGH_ORIENT_FINAL=1: y180 / X +-90 copies of the final, margin gate + judge duel
        await _orient_final.maybe_turn_final(factory, winner, judge=judge, sem_judge=sem_judge)

    survivors = [c for c in candidates if _is_live(c)]
    unique = len({d for d in (_code_digest(c.js_code) for c in candidates) if d})
    # Rebuild deterministically by k (leaves resolve out of order now).
    task.multigen_pngs = [c.rendered_png for c in survivors]
    logger.info(
        f"[MULTIGEN] {task.stem} K={K} | unique={unique} | "
        f"survivors={[c.k for c in survivors]} | drops={[c.drop_reason for c in candidates]}"
    )

    _promote_winner(task, winner, candidates, session_store)
    logger.info(
        f"[MULTIGEN] {task.stem} winner=k{winner.k} | "
        f"js_bytes={len(winner.js_code.encode('utf-8')) if winner.js_code else 0}"
    )
    if __import__("os").environ.get("PGH_SHADOW"):   # [opt_r5 shadow] default off: alt picks in the background, after the pick
        import pgh_shadow as _shadow
        if _shadow.on():
            _shadow.launch(task, factory, winner, judge=judge, sem_judge=sem_judge, K=K)
