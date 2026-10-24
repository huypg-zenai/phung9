"""[R84 jalign] Judge-alignment switches for pgh17. Every switch is ENV, default OFF (= pgh17 behaviour exactly).

  PGH_S1ES_MAX_ROUND   unset / "" / "0" / "off"        -> the config's actors.judge.s1_early_stop.max_round (4)
                       an integer 1..16 (e.g. 2)        -> S1 early stop only in bracket rounds <= that value
                       anything else                    -> one "!!" warning, config value
  PGH_RR8              unset / "" / "0" / "off"        -> top-4 round robin as pgh17 (ROUND_ROBIN_TOP4 rules)
                       "1" / "on"                       -> full top-8 round robin (all 28 pairs of the 8 eighth-block
                                                           winners) for every photo of a batch except the last
                                                           PGH_RR8_TAIL photos (batch order); those keep top-4
  PGH_RR8_TAIL         integer >= 0, default 4          -> how many photos at the end of each batch keep top-4
  PGH_RR_CUTOFF_FALLBACK  unset / "" / "0" / "off"     -> on a ROUND_ROBIN_CUTOFF_S cutoff the bracket winner (pgh17)
                       "s1pen"                          -> on a cutoff, the top-set player with the lowest mean S1
                                                           penalty over its finished top-set duels; logs one
                                                           "[RR_CUTOFF_FALLBACK]" line per firing
  (JUDGE_GRID=off|small|contest is the ported jgrid switch: modules/renderer/settings.py read_judge_grid.)

Parsing never raises; an invalid value always means "off" plus one warning line.
"""
from __future__ import annotations

import os

_OFF = ("", "0", "off", "false", "no")
_WARNED: set = set()


def _warn(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    try:
        from logger_config import logger
        logger.warning(msg)
    except Exception:
        pass


def s1es_max_round(config_value: int, env=None) -> int:
    """The S1 early-stop max_round to use: the config value unless PGH_S1ES_MAX_ROUND is a valid integer 1..16."""
    raw = (os.environ if env is None else env).get("PGH_S1ES_MAX_ROUND")
    if raw is None or raw.strip().lower() in _OFF:
        return config_value
    try:
        v = int(raw.strip())
    except ValueError:
        v = -1
    if not 1 <= v <= 16:
        _warn("s1es", f"[JALIGN] !! PGH_S1ES_MAX_ROUND={raw!r} invalid (want 1..16) -> config max_round={config_value}")
        return config_value
    return v


def rr8_on(env=None) -> bool:
    raw = (os.environ if env is None else env).get("PGH_RR8")
    if raw is None or raw.strip().lower() in _OFF:
        return False
    if raw.strip().lower() in ("1", "on", "true", "yes"):
        return True
    _warn("rr8", f"[JALIGN] !! PGH_RR8={raw!r} invalid (want 1/on or 0/off) -> off")
    return False


def rr8_tail(env=None) -> int:
    raw = (os.environ if env is None else env).get("PGH_RR8_TAIL")
    if raw is None or not raw.strip():
        return 4
    try:
        v = int(raw.strip())
    except ValueError:
        v = -1
    if v < 0:
        _warn("rr8tail", f"[JALIGN] !! PGH_RR8_TAIL={raw!r} invalid (want an integer >= 0) -> 4")
        return 4
    return v


def cutoff_fallback(env=None) -> str:
    """'' (pgh17: bracket winner) or 's1pen'."""
    raw = (os.environ if env is None else env).get("PGH_RR_CUTOFF_FALLBACK")
    if raw is None or raw.strip().lower() in _OFF:
        return ""
    if raw.strip().lower() == "s1pen":
        return "s1pen"
    _warn("cfb", f"[JALIGN] !! PGH_RR_CUTOFF_FALLBACK={raw!r} invalid (want s1pen or off) -> off")
    return ""


# ---- batch order for PGH_RR8 (filled by GenerationPipeline.run_batch; empty = every photo keeps top-4) ----
_ORDER: dict = {}


def batch_order(stems: list, env=None) -> None:
    """Record each photo's position in its batch. A no-op (registry stays empty) while PGH_RR8 is off."""
    _ORDER.clear()
    if not rr8_on(env):
        return
    n = len(stems)
    for i, s in enumerate(stems):
        _ORDER[s] = (i, n)


def rr8_for(stem: str, n_positions: int, env=None) -> bool:
    """True when this photo gets the top-8 round robin: PGH_RR8 on, >= 8 positions, and the photo is not one of the
    last PGH_RR8_TAIL photos of its batch. A photo outside the registry (no run_batch) keeps top-4."""
    if n_positions < 8 or not rr8_on(env):
        return False
    pos = _ORDER.get(stem)
    if pos is None:
        return False
    i, n = pos
    return i < n - rr8_tail(env)


def s1_means(top_keys, s1rec: dict) -> dict:
    """Mean S1 penalty per player over its finished duels inside the top set (per duel: the mean over its angles)."""
    keys = set(top_keys)
    acc: dict = {k: [] for k in keys}
    for (ka, kb), s1 in s1rec.items():
        if ka not in keys or kb not in keys or not s1:
            continue
        angles = [a for a in (s1.get("angles") or []) if "pen_a" in a and "pen_b" in a]
        if not angles:
            continue
        acc[ka].append(sum(float(a["pen_a"]) for a in angles) / len(angles))
        acc[kb].append(sum(float(a["pen_b"]) for a in angles) / len(angles))
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def s1pen_pick_key(top_keys, s1rec: dict, bracket_k):
    """(pick k, means): the lowest mean S1 penalty; ties -> the bracket winner if tied, else the lowest k.
    No S1 readings at all -> the bracket winner."""
    means = s1_means(top_keys, s1rec)
    if not means:
        return bracket_k, means
    best = min(means.values())
    tied = sorted(k for k, m in means.items() if m == best)
    return (bracket_k if bracket_k in tied else tied[0]), means
