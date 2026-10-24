"""[pgh16 R78] Which coder prompt (system + image-only user text) the pipeline sends. Switch: env PGH_CODER_PROMPT.

  es       (DEFAULT: unset, "" or "es") expert-spork r56's full coder prompt, verbatim (es56_prompt.json; owner decision
           22/10 01:55 VN: "this is opensource competition, we can do that, dont need our wording"). Prompt text only.
  pgh7     pgh7's own prompt, byte-identical to pgh7/cand (system + CODER_USER_TEMPLATE_IMAGE_ONLY).
  ours     pgh7's prompt + our texture section (texture_guide.py) appended to the system prompt; user text = pgh7's.
  oursneg  as ours, plus the list of non-existent names (test variant).
An unknown value falls back to es with a warning on stderr (never crash the service at import).
Read once when prompts.py is imported (image ENV; `docker run -e PGH_CODER_PROMPT=pgh7` switches without a rebuild).
The repair / critic / OSD user templates are pgh7's in every variant (pgh16 runs image-only, no critic loop).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from modules.scene_coder.texture_guide import texture_guide_text

CODER_PROMPT_ENV = "PGH_CODER_PROMPT"
VARIANTS = ("es", "pgh7", "ours", "oursneg")
DEFAULT_VARIANT = "es"
SECTION_SEP = "\n\n---\n\n"
ES_FILE = Path(__file__).with_name("es56_prompt.json")
ES_SYSTEM_MD5 = "7d9bd866bcaa38af5d823f4cbd2c07b8"
ES_USER_MD5 = "89e3576edc8c25acc96c004b3e069eb4"


def coder_prompt_variant(env: dict | None = None) -> str:
    v = str((os.environ if env is None else env).get(CODER_PROMPT_ENV, "") or "").strip().lower()
    if v == "":
        return DEFAULT_VARIANT
    if v not in VARIANTS:
        print(f"[pgh16] {CODER_PROMPT_ENV}={v!r} unknown (want {'|'.join(VARIANTS)}); using {DEFAULT_VARIANT}",
              file=sys.stderr)
        return DEFAULT_VARIANT
    return v


def load_es_prompt(path: str | os.PathLike | None = None) -> tuple[str, str]:
    """(system, user image-only) of expert-spork r56; raises if the shipped text is not the verified one."""
    d = json.loads(Path(path or ES_FILE).read_text(encoding="utf-8"))
    system, user = d["system"], d["user_image_only"]
    for name, text, want in (("system", system, ES_SYSTEM_MD5), ("user_image_only", user, ES_USER_MD5)):
        got = hashlib.md5(text.encode("utf-8")).hexdigest()
        if got != want:
            raise RuntimeError(f"es56_prompt.json {name} md5 {got} != {want}")
    return system, user


def select_coder_prompt(pgh7_system: str, pgh7_user: str, env: dict | None = None) -> tuple[str, str, str]:
    """(variant, system prompt, image-only user text)."""
    v = coder_prompt_variant(env)
    if v == "es":
        system, user = load_es_prompt()
        return v, system, user
    if v == "pgh7":
        return v, pgh7_system, pgh7_user
    return v, pgh7_system + SECTION_SEP + texture_guide_text(neg=(v == "oursneg")), pgh7_user
