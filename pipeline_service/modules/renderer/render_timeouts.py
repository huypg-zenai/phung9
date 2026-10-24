"""[r42] PGH_RENDER_TIMEOUT_S: one env knob that raises every per-render timeout together (pure, no imports beyond stdlib).

Why: a draft whose render takes ~60 s under CPU load is dropped in one run and kept in another (the 60 s Chrome
protocol timeout / RENDER_TIMEOUT_MS fire on wall-clock time), so the candidate pool differs between reruns.
With the knob set to T seconds the three render-path timeouts become
  render_timeout_ms   = T * 1000            (render-pool/object-pool withTimeout around page.evaluate)
  protocol_timeout_ms = T * 1000 + 30000    (puppeteer protocolTimeout; > the render timeout, so the explicit
                                             withTimeout fires first and the slot is disposed cleanly)
  request_timeout_s   = max(config, 2*T + 60)  (httpx client; must outlast one queued render + its own render)
Unset / "" / "0" / "0.0" / "off" -> OFF: the config values are returned unchanged (old behaviour, byte-identical).
Bad value -> OFF plus a '!!' note. T is clamped to [MIN_S, MAX_S].
"""
from __future__ import annotations

import math

ENV_NAME = "PGH_RENDER_TIMEOUT_S"
MIN_S = 60.0   # [r43] never below the old 60 s
MAX_S = 600.0  # [r43] bound a hung-but-connected Chrome
PROTOCOL_EXTRA_MS = 30000
REQUEST_SLACK_S = 60.0


def parse_knob(raw) -> tuple[float | None, str]:
    """(T seconds or None = off, note). Never raises."""
    v = "" if raw is None else str(raw).strip().lower()
    if v in ("", "0", "off", "none"):
        return None, ""
    try:
        t = float(v)
    except ValueError:
        return None, f"!! {ENV_NAME}={raw!r} is not a number -> off (old timeouts)"
    if t == 0:
        return None, ""
    if not math.isfinite(t) or t <= 0:
        return None, f"!! {ENV_NAME}={raw!r} must be a finite number > 0 -> off (old timeouts)"
    if t < MIN_S:
        return MIN_S, f"!! {ENV_NAME}={raw!r} below {MIN_S:.0f}s -> clamped to {MIN_S:.0f}s"
    if t > MAX_S:
        return MAX_S, f"!! {ENV_NAME}={raw!r} above {MAX_S:.0f}s -> clamped to {MAX_S:.0f}s"
    return t, ""


def effective_timeouts(render_timeout_ms: int, protocol_timeout_ms: int, request_timeout_s: float,
                       raw) -> tuple[dict, str]:
    """Returns ({render_timeout_ms, protocol_timeout_ms, request_timeout_s}, log note). Knob off -> the inputs as-is."""
    t, note = parse_knob(raw)
    base = {"render_timeout_ms": render_timeout_ms, "protocol_timeout_ms": protocol_timeout_ms,
            "request_timeout_s": request_timeout_s}
    if t is None:
        return base, note
    out = {
        "render_timeout_ms": int(round(t * 1000)),
        "protocol_timeout_ms": int(round(t * 1000)) + PROTOCOL_EXTRA_MS,
        "request_timeout_s": max(float(request_timeout_s), 2.0 * t + REQUEST_SLACK_S),
    }
    msg = (f"{ENV_NAME}={t:g} -> render_timeout_ms {render_timeout_ms}->{out['render_timeout_ms']}, "
           f"protocol_timeout_ms {protocol_timeout_ms}->{out['protocol_timeout_ms']}, "
           f"request_timeout_s {request_timeout_s:g}->{out['request_timeout_s']:g}")
    return out, (note + " | " if note else "") + msg
