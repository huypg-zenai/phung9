"""[R86 orient_final] PGH_ORIENT_FINAL=1: extra orientation candidates for the photo's FINAL draft. Default OFF (= pgh17).

The per-draft probe (orientation.py) only turns side-best drafts and tips top-down-best ones; it never tries a 180-deg
back flip or a "face the camera" X tilt. R85 (opt_r3/eyes): UtevoLux lost a ceiling-fan photo (shot from below) 8/8 because
its level fan is edge-on in the judge's front views; ES's pipeline turned that object yaw 180 + 90 deg about X.

After the bracket / round robin has picked the final:
  1. each candidate in PGH_ORIENT_FINAL_CANDS (default y180,xm90,xp90) is written into the final's own generate() by
     the same rotate_runner.mjs (js_checker.rotate_source, which re-centres the bbox), re-checked by the same checker,
     rendered with the same judge views (renderer.process_with_judge_views) and embedded (embedder.build_candidate_npz):
     exactly the path a normal draft takes;
  2. margin gate (like PGH_ORIENT_MIN_MARGIN): m = best-view DINO similarity to the photo (the judge's S2BV signal,
     multi_stage.best_view_similarity) of the candidate minus that of the final; a candidate is kept only if
     m >= PGH_ORIENT_FINAL_MIN_MARGIN (default 0.05). Missing embeddings -> no turn;
  3. the best kept candidate duels the final with the in-pipeline judge (judge.compare, final = seat A). It replaces the
     final only on a DECISIVE win (not a draw / DINO tie-break / no-signal draw). PGH_ORIENT_FINAL_BOTH_SEATS=1 also
     plays the swapped seats and needs both.
Any failure keeps the final unchanged.

Names: y180 = rotate Y by PI (back flip); xm90 = rotate X by -PI/2 (the underside -Y turns toward the front camera +Z:
"shot from below"); xp90 = rotate X by +PI/2 (the top +Y faces the camera, = orientation.py's tip step).
"""
from __future__ import annotations

import asyncio
import os
import re
import time

from logger_config import logger
from pipeline.task import Candidate, PipelineTask

CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "y180": [("y", "Math.PI")],
    "xm90": [("x", "-Math.PI / 2")],
    "xp90": [("x", "Math.PI / 2")],
    "y180xm90": [("y", "Math.PI"), ("x", "-Math.PI / 2")],   # ES's 42c4 turn (yaw 180 then tilt); not in the default list
}
DEFAULT_CANDS = "y180,xm90,xp90"
DEFAULT_MARGIN = 0.05


def enabled() -> bool:
    return os.environ.get("PGH_ORIENT_FINAL", "").strip().lower() in ("1", "on", "true", "yes")


def cand_names() -> list[str]:
    """PGH_ORIENT_FINAL_CANDS: comma list of CANDIDATES keys; unknown names are dropped; empty/unset -> the default."""
    raw = os.environ.get("PGH_ORIENT_FINAL_CANDS", "").strip() or DEFAULT_CANDS
    out = []
    for n in (x.strip().lower() for x in raw.split(",")):
        if n in CANDIDATES and n not in out:
            out.append(n)
    return out


def min_margin() -> float:
    """PGH_ORIENT_FINAL_MIN_MARGIN in [-1, 1]; garbage/nan -> the default 0.05."""
    try:
        v = float(os.environ.get("PGH_ORIENT_FINAL_MIN_MARGIN", "") or DEFAULT_MARGIN)
    except (TypeError, ValueError):
        return DEFAULT_MARGIN
    return v if v == v and -1.0 <= v <= 1.0 else DEFAULT_MARGIN


def both_seats() -> bool:
    return os.environ.get("PGH_ORIENT_FINAL_BOTH_SEATS", "").strip() in ("1", "on", "true")


_ORIENTED = re.compile(r"\b__oriented\b")


def free_oriented_name(code: str) -> str:
    """rotate_runner.mjs writes `const __oriented = <expr>;` before every return. A final the probe already turned
    returns `__oriented`, so a second injection would read `const __oriented = __oriented` (TDZ ReferenceError, R22).
    Rename our own earlier identifier to a free `__oriented_<n>` first (word-boundary rename; no-op without it)."""
    if not code or not _ORIENTED.search(code):
        return code
    n = 1
    while f"__oriented_{n}" in code:
        n += 1
    return _ORIENTED.sub(f"__oriented_{n}", code)


def _bvs(npz: bytes | None) -> float | None:
    """Best-view DINO similarity of an npz (prompt + view_*), = the judge's draw tie-break signal."""
    if not npz:
        return None
    try:
        import io

        import numpy as np

        from modules.judge.multi_stage import best_view_similarity
        with np.load(io.BytesIO(npz)) as z:
            emb = {k: z[k] for k in z.files}
        v = best_view_similarity(emb)
        return float(v) if v is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[ORIENT_FINAL] embedding read failed: {type(exc).__name__}: {exc}")
        return None


def is_decisive(decided_by: str) -> bool:
    d = decided_by or ""
    return not ("DINO tie-break" in d or "draw ->" in d or "no DINO signal" in d)


async def _build(factory, final: Candidate, code: str, name: str) -> Candidate | None:
    task = factory.task
    rotated = await factory.js_checker.rotate_source(code, CANDIDATES[name])
    if not rotated:
        return None
    probe = PipelineTask(stem=f"{task.stem}~of_{name}", image_url=task.image_url)
    probe.js_code = rotated
    probe.image_bytes = task.image_bytes
    probe.image_mime = task.image_mime
    async with factory.sem_checker:
        await factory.js_checker.process(probe, mode=factory.checker_mode)
    if not probe.js_valid:
        logger.info(f"[ORIENT_FINAL] {task.stem} {name}: rotated code rejected {list(probe.js_errors or [])[:1]}")
        return None
    async with factory.sem_renderer:
        white, gray = await factory.renderer.process_with_judge_views(probe)
    if not white or not probe.rendered_png:
        return None
    emb = None
    if factory.embedder is not None and factory.ref_vec is not None:
        emb = await factory.embedder.build_candidate_npz(factory.ref_vec, white)
    return Candidate(k=final.k, seed=final.seed, js_code=rotated, js_valid=True, scene_json=probe.scene_json,
                     rendered_png=probe.rendered_png, judge_white_views=white, judge_gray_views=gray,
                     judge_embeddings=emb)


async def _duel(task, judge, sem_judge, a: Candidate, b: Candidate, label: str):
    async with sem_judge:
        v = await judge.compare(
            task_id=task.stem, match_label=label, reference_bytes=task.image_bytes, reference_mime=task.image_mime,
            render_a=a.rendered_png, render_b=b.rendered_png,
            white_views_a=a.judge_white_views, white_views_b=b.judge_white_views,
            gray_views_a=a.judge_gray_views, gray_views_b=b.judge_gray_views,
            embeddings_a=a.judge_embeddings, embeddings_b=b.judge_embeddings,
            early_stop=False,
        )
    return v.winner, str(getattr(v, "decided_by", "") or "")


async def maybe_turn_final(factory, final: Candidate, *, judge, sem_judge) -> dict:
    """Mutates `final` in place when a turned copy wins. Returns a small record (also put on task.meta)."""
    rec: dict = {"applied": False}
    task = factory.task
    t0 = time.monotonic()
    try:
        if judge is None or final is None or final.js_code is None or not final.judge_white_views:
            rec["reason"] = "no_judge_or_views"
            return rec
        s0 = _bvs(final.judge_embeddings)
        rec["s_final"] = s0
        if s0 is None:
            rec["reason"] = "no_final_embedding"
            return rec
        names = cand_names()
        code = free_oriented_name(final.js_code)
        built = await asyncio.gather(*(_build(factory, final, code, n) for n in names), return_exceptions=True)
        tau = min_margin()
        scored = []
        for n, c in zip(names, built):
            if isinstance(c, BaseException) or c is None:
                rec.setdefault("failed", []).append(n)
                continue
            s = _bvs(c.judge_embeddings)
            rec.setdefault("s", {})[n] = s
            if s is not None and s - s0 >= tau - 1e-9:
                scored.append((s - s0, n, c))
        rec["renders"] = len(names)
        if not scored:
            rec["reason"] = f"gate (tau={tau})"
            return rec
        m, name, cand = max(scored, key=lambda t: t[0])
        rec.update(best=name, margin=round(m, 5))
        w, how = await _duel(task, judge, sem_judge, final, cand, f"OF k{final.k}-vs-{name}")
        rec["duels"] = 1
        ok = w == "B" and is_decisive(how)
        if ok and both_seats():
            w2, how2 = await _duel(task, judge, sem_judge, cand, final, f"OF {name}-vs-k{final.k}")
            rec["duels"] = 2
            ok = w2 == "A" and is_decisive(how2)
            how = f"{how} || {how2}"
        rec["decided_by"] = how[:200]
        if not ok:
            rec["reason"] = "judge kept the final"
            return rec
        final.js_code, final.scene_json, final.rendered_png = cand.js_code, cand.scene_json, cand.rendered_png
        final.judge_white_views, final.judge_gray_views = cand.judge_white_views, cand.judge_gray_views
        final.judge_embeddings = cand.judge_embeddings
        rec["applied"] = True
        rec["reason"] = f"turned {name}"
        return rec
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - never fails the prompt; the final stays as it was
        rec["reason"] = f"error {type(exc).__name__}: {exc}"[:200]
        return rec
    finally:
        rec["wall_s"] = round(time.monotonic() - t0, 2)
        try:
            task.meta["orient_final"] = rec
        except Exception:
            pass
        logger.info(f"[ORIENT_FINAL] {task.stem} k{getattr(final, 'k', '?')} {rec}")
