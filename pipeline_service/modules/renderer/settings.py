import os
from enum import Enum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

# rrggbb hex, optionally '#'-prefixed in config; normalized to the bare form
HexColor = Annotated[str, Field(pattern=r"^#?[0-9a-fA-F]{6}$"), AfterValidator(lambda v: v.removeprefix("#"))]


class LightingMode(str, Enum):
    FOLLOW = "follow"
    STUDIO = "studio"
    NEUTRAL = "neutral"


class RendererConfig(BaseModel):
    sidecar_host: str = "127.0.0.1"
    sidecar_port: int = 8003
    startup_timeout_s: float = 60.0
    request_timeout_s: float = 90.0
    node_binary: str = "node"
    img_size: int = 518
    grid_gap: int = 5
    bg_color: HexColor | None = None
    lighting: LightingMode = LightingMode.NEUTRAL
    sidecar_count: int = 4
    pool_size: int = 2
    judge_multiview: bool = True
    judge_white_bg: HexColor = "ffffff"
    judge_gray_bg: HexColor = "808080"
    render_timeout_ms: int = 60000
    protocol_timeout_ms: int = 60000
    static_port_base: int = 3000


# [jgrid] JUDGE_GRID switch: the 2x2 grid our in-pipeline judge sees (S2 artifact-compare + checklist verify only).
#   off (unset / "off" / unknown -> one warning): today's grid, byte-identical requests and output.
#   small:   same render request; the four 1024 px tiles are downscaled to 518 px in Python -> 1041 px grid.
#   contest: the contest judge's grid: 518 px tiles, studio light, grey 808080, 5 px black gaps (= :8017 /render/grid).
JUDGE_GRID_MODES = ("off", "small", "contest")
JUDGE_GRID_TILE = 518                 # contest render-service-js renderer.js DEFAULT_IMG_SIZE
JUDGE_GRID_BG = "808080"              # contest render-pool.js: null bgColor -> 0x808080
JUDGE_GRID_LIGHTING = "studio"        # contest renderer.js default lighting


def read_judge_grid(env=None) -> tuple[str, str | None]:
    """(mode, warning) from env JUDGE_GRID; called once at start-up by RendererModule.__init__."""
    raw = (os.environ if env is None else env).get("JUDGE_GRID")
    if raw is None:
        return "off", None
    v = raw.strip().lower()
    if v in JUDGE_GRID_MODES:
        return v, None
    return "off", f"JUDGE_GRID={raw!r} is not one of {JUDGE_GRID_MODES} -> off"
