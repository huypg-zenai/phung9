"""pgh_rslots.py - [sws] render-slot override from the environment (opt_r4/sws/README.md).

PGH_RENDER_SLOTS = "SIDECARS:PAGES:WORKERS" overrides, in the parsed configuration.yaml dict before validation:
  renderer.sidecar_count   (node render sidecars = Chrome browsers; ports sidecar_port + i, so max 16: 8003..8018 < 8020)
  renderer.pool_size       (Chromium pages per sidecar, 1..4)
  actors.renderer.workers  (the pipeline's render Semaphore, 1..128; usually SIDECARS x PAGES)
"" / unset / "0" / "off" = the dict is returned unchanged (= pgh17: 8:2:16). Any invalid value -> ONE '!!' line and the
dict unchanged (never fails the start). Pure function of (data, env); config/settings.py calls it only when the env is set.
"""
from __future__ import annotations

import copy
import os

OFF = ("", "0", "off", "false", "no")
LIMITS = (("sidecars", 1, 16), ("pages", 1, 4), ("workers", 1, 128))


def parse(raw: str):
    """-> (sidecars, pages, workers) or raises ValueError with the reason."""
    parts = raw.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"need SIDECARS:PAGES:WORKERS, got {raw!r}")
    out = []
    for text, (name, lo, hi) in zip(parts, LIMITS):
        if not text.strip().isdigit():
            raise ValueError(f"{name} {text!r} is not a positive integer")
        v = int(text)
        if not lo <= v <= hi:
            raise ValueError(f"{name} {v} outside {lo}..{hi}")
        out.append(v)
    return tuple(out)


def apply_cfg(data: dict, env=None, log=print) -> dict:
    env = os.environ if env is None else env
    raw = str(env.get("PGH_RENDER_SLOTS", "") or "")
    if raw.strip().lower() in OFF:
        return data
    try:
        s, p, w = parse(raw)
        if not isinstance(data.get("renderer"), dict) or not isinstance((data.get("actors") or {}).get("renderer"), dict):
            raise ValueError("configuration.yaml has no renderer / actors.renderer section")
    except ValueError as exc:
        log(f"[RSLOTS] !! PGH_RENDER_SLOTS ignored ({exc}): configuration.yaml render slots kept")
        return data
    out = copy.deepcopy(data)
    old = (out["renderer"].get("sidecar_count"), out["renderer"].get("pool_size"), out["actors"]["renderer"].get("workers"))
    out["renderer"]["sidecar_count"], out["renderer"]["pool_size"], out["actors"]["renderer"]["workers"] = s, p, w
    note = "" if w == s * p else f" (workers != sidecars x pages = {s * p})"
    log(f"[RSLOTS] render slots {old[0]}:{old[1]}:{old[2]} -> {s}:{p}:{w} (PGH_RENDER_SLOTS){note}")
    return out
