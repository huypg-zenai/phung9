"""pgh_salvage.py - [opt_r5 salvage] PGH_DEADLINE_SALVAGE: ship the best draft found so far when a prompt is cut off.

Without it a prompt that hits the per-task deadline (orchestrator.run_one, `except TimeoutError`) or the batch budget
(generation_pipeline.run_batch, `except asyncio.TimeoutError`) ships NOTHING (state.results only lists tasks with
failed=False and js_code), and the validator scores the missing module as a loss (r56 pgh5 audit: the last 8 prompts of
batch 1). With it, the cut-off prompt ships, in this order:
  promoted  the multigen winner was already copied onto the task (cut during orient/critic/refinement): task.best_js_code
            or task.js_code with task.winner_k set
  leader    the current bracket leader: the live winner of the highest-round judge duel finished so far (ties: more duel
            wins, then lower k)
  rendered  a live draft (checker passed, rendered, not dropped): highest DINO best-view similarity to the photo when
            its judge embeddings exist, else lowest k
  checked   a draft that passed the JS checker but whose render had not finished (or was skipped by the opt-in render
            deadline): lowest k. The output format is just the module source, so unrendered code can be shipped.
  none      nothing usable: the task stays failed, exactly as without the switch.
Cost: in-memory only (no await, no I/O except np.load of the small npz blobs for the 'rendered' tier); it can never
extend the batch. One countable line per cut-off prompt: "[SALVAGE] <stem> where=<...> tier=<...> k=<...> ... took_ms=".

Switch PGH_DEADLINE_SALVAGE (read once at import = container start): 1/on/true = on; anything else = off. Off = no hook
does anything (no factory registration, no duel bookkeeping, deadline branches unchanged).
Hooks (all `if _salv is not None and _salv.ON`): stages.multigen_first_iter -> register(task, factory);
stages._resolve_bracket._judge_duel -> note_duel(...); orchestrator.run_one TimeoutError -> salvage(task, "task_deadline");
generation_pipeline.run_batch batch timeout -> salvage(task, "batch_deadline") only for a prompt still running at the
batch deadline (not recorded yet) or recorded as a task-deadline cut, and never twice for the same task object (tried()).
A prompt that failed earlier for another reason (StageError after 3 attempts, unexpected error) stays failed.
"""
from __future__ import annotations

import io
import os
import time

_TRUE = ("1", "on", "true", "yes")
ON = str(os.environ.get("PGH_DEADLINE_SALVAGE", "0") or "0").strip().lower() in _TRUE
COUNTS: dict[str, int] = {}      # tier -> salvages since start (also logged per prompt)
_FACTORY = "_pgh_salvage_factory"
_DUELS = "_pgh_salvage_duels"
_TRIED = "_pgh_salvage_tried"


def register(task, factory) -> None:
    """Remember the prompt's candidate factory (its coder tasks hold every draft produced so far)."""
    task.__dict__[_FACTORY] = factory
    task.__dict__[_DUELS] = []


def note_duel(task, round_no: int, winner, loser) -> None:
    """One finished judge duel of this prompt's bracket (or round robin)."""
    try:
        task.__dict__.setdefault(_DUELS, []).append((int(round_no), winner, loser))
    except Exception:
        pass


def tried(*tasks) -> bool:
    """True if salvage() already ran on any of these task objects (None entries ignored). The batch-deadline hook skips
    such a prompt, so a prompt cut at its task deadline gets exactly one countable [SALVAGE] line."""
    return any(t is not None and bool(getattr(t, "__dict__", {}).get(_TRIED)) for t in tasks)


def _drafts(task) -> list:
    """Every Candidate whose coder call finished (the coder tasks keep the objects that _validate mutates in place)."""
    factory = task.__dict__.get(_FACTORY)
    tasks = getattr(factory, "_coder_tasks", None) or {}
    out = []
    for k in sorted(tasks):
        t = tasks[k]
        try:
            if t.done() and not t.cancelled() and t.exception() is None:
                c = t.result()
                if c is not None and c.js_code:
                    out.append(c)
        except Exception:
            continue
    return out


def _live(c) -> bool:
    return c is not None and c.drop_reason is None and c.rendered_png is not None and bool(c.js_code)


def _dino(c) -> float | None:
    """Best-view cosine similarity to the photo from the candidate's S2BV npz (the bracket's tie-break signal)."""
    if not c.judge_embeddings:
        return None
    try:
        import numpy as np
        with np.load(io.BytesIO(c.judge_embeddings)) as z:
            if "prompt" not in z.files:
                return None
            p = z["prompt"]
            sims = [float(np.dot(z[n], p)) for n in z.files if n.startswith("view_")]
        return max(sims) if sims else None
    except Exception:
        return None


def pick(task) -> tuple[str | None, str, int | None, dict]:
    """(code, tier, k, info) of the best available output; code None = nothing usable (tier 'none')."""
    if task.best_js_code:
        return task.best_js_code, "promoted", task.winner_k, {}
    if task.winner_k is not None and task.js_code:
        return task.js_code, "promoted", task.winner_k, {}
    drafts = _drafts(task)
    info = {"drafts": len(drafts), "live": sum(1 for c in drafts if _live(c)),
            "checked": sum(1 for c in drafts if c.js_valid), "duels": len(task.__dict__.get(_DUELS) or [])}
    best = {}   # id(candidate) -> [max round won, wins, candidate]
    for rnd, w, _l in task.__dict__.get(_DUELS) or []:
        if not _live(w):
            continue
        b = best.setdefault(id(w), [0, 0, w])
        b[0] = max(b[0], rnd)
        b[1] += 1
    if best:
        rnd, wins, c = max(best.values(), key=lambda b: (b[0], b[1], -b[2].k))
        return c.js_code, "leader", c.k, dict(info, round=rnd, wins=wins)
    live = [c for c in drafts if _live(c)]
    if live:
        sims = {c.k: _dino(c) for c in live}
        scored = [c for c in live if sims[c.k] is not None]
        c = max(scored, key=lambda c: (sims[c.k], -c.k)) if scored else min(live, key=lambda c: c.k)
        return c.js_code, "rendered", c.k, dict(info, dino=None if sims[c.k] is None else round(sims[c.k], 4))
    checked = [c for c in drafts if c.js_valid and (c.drop_reason is None or c.drop_reason == "render:deadline")]
    if checked:
        c = min(checked, key=lambda c: (c.drop_reason is not None, c.k))
        return c.js_code, "checked", c.k, info
    return None, "none", None, info


def salvage(task, where: str, log=None, status: dict | None = None) -> bool:
    """Ship pick(task) on a cut-off prompt. True = the task now carries code and failed=False. Never raises."""
    t0 = time.perf_counter()
    try:
        task.__dict__[_TRIED] = where
    except Exception:
        pass
    try:
        code, tier, k, info = pick(task)
    except Exception as exc:  # never make a failure worse
        code, tier, k, info = None, "error", None, {"error": f"{type(exc).__name__}: {exc}"}
    COUNTS[tier] = COUNTS.get(tier, 0) + 1
    if code:
        reason = task.failure_reason
        task.js_code = code
        if task.winner_k is None:
            task.winner_k = k
        task.failed = False
        task.failure_reason = None
        task.failure_stage = None
        task.meta["salvage"] = dict(info, where=where, tier=tier, k=k, cut=reason)
        if status is not None:
            status[task.stem] = "salvaged"
    took = 1000.0 * (time.perf_counter() - t0)
    if log is not None:
        try:
            log(f"[SALVAGE] {task.stem} where={where} tier={tier} k={k} shipped={'yes' if code else 'no'} "
                f"{' '.join(f'{a}={b}' for a, b in info.items())} took_ms={took:.1f}")
        except Exception:
            pass
    return bool(code)
