"""[pgh16 R78] THREE-name repair for coder drafts. Switch: env PGH_THREE_REPAIR; "1" = on, unset / anything else = off
(default off: the pipeline is then byte-for-byte the pgh7 behaviour, this module is never called).

Our own code: a rewrite for pgh16 of our es_impr three_repair.py (R76, the module only; no expert-spork code), checked
against pgh16's own js_checker (modules/js_checker/validator: staticAnalysis.js error text, threeAllowlist.js targets).

Problem: UtevoLux often writes THREE constants that do not exist (THREE.RepeatWrap, THREE.LinearMipmapipmap,
THREE.UnsignedByteTypeFormat, THREE.LinearSpace, ...). The validator rejects the whole draft (UNKNOWN_THREE_API).

What it does (pure text, stdlib only, deterministic):
  1. plan(errors): the names come from the checker's own messages
     "UNKNOWN_THREE_API: THREE.<X> is not a recognized Three.js API at line <n>". If ANY failure is something else
     (another rule, a destructured access, a runtime error), nothing is done.
  2. rewrite(code, names): every `THREE.<X>` of those names becomes an allowed constant chosen by its context:
     `.wrapS/.wrapT/.wrapR =` -> a Wrapping, `.minFilter =` / `.magFilter =` -> a Filter, `colorSpace` -> a ColorSpace,
     argument 4 / 5 of `new THREE.DataTexture(` -> a Format / a Type; else a fixed name table; else the closest allowed
     constant (difflib, ratio >= 0.8). A use as constructor (`new THREE.X`), call (`THREE.X(`) or member chain
     (`THREE.X.y`) is never rewritten; one unmappable occurrence = no rewrite at all.
  3. the caller (pipeline/stages.py) runs the SAME checker once more on the rewritten code and keeps it only if it passes.
A draft that passes the checker is never touched, so the switch only adds drafts that would have been dropped.
"""
from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

REPAIR_ENV = "PGH_THREE_REPAIR"

# Target constants per family. Every one is in threeAllowlist.js THREE_ALLOWED and exported by three.js (tested);
# LuminanceFormat is allowlisted but no longer exists in three.js, so it is never a target.
FAMILY = {
    "wrap": ("RepeatWrapping", "ClampToEdgeWrapping", "MirroredRepeatWrapping"),
    "filter": ("NearestFilter", "LinearFilter", "NearestMipmapNearestFilter", "LinearMipmapNearestFilter",
               "NearestMipmapLinearFilter", "LinearMipmapLinearFilter"),
    "colorspace": ("SRGBColorSpace", "LinearSRGBColorSpace", "NoColorSpace"),
    "format": ("RGBAFormat", "RGBFormat", "RGFormat", "RedFormat"),
    "type": ("UnsignedByteType", "HalfFloatType", "FloatType"),
    "side": ("FrontSide", "BackSide", "DoubleSide"),
    "blending": ("NormalBlending", "AdditiveBlending", "SubtractiveBlending", "MultiplyBlending", "NoBlending"),
    "usage": ("StaticDrawUsage", "DynamicDrawUsage", "StreamDrawUsage"),
}
TARGETS = tuple(c for fam in FAMILY.values() for c in fam)

# Names seen without a telling context (utevo_diag/utevo_diag2 checker logs, UtevoLux) -> the constant meant.
KNOWN = {
    "RepeatWrap": "RepeatWrapping", "RepeatWrapper": "RepeatWrapping", "RepeatRepeat": "RepeatWrapping",
    "Repeat": "RepeatWrapping", "RepeatTexture": "RepeatWrapping", "ClampWrapping": "ClampToEdgeWrapping",
    "ClampToEdge": "ClampToEdgeWrapping", "MirroredRepeat": "MirroredRepeatWrapping",
    "MirrorRepeatWrapping": "MirroredRepeatWrapping",
    "LinearMipmapipmap": "LinearMipmapLinearFilter", "LinearMipmapipMap": "LinearMipmapLinearFilter",
    "LinearMipmap": "LinearMipmapLinearFilter", "LinearMipMapLinearFilter": "LinearMipmapLinearFilter",
    "LinearMipMapNearestFilter": "LinearMipmapNearestFilter", "MipmapNearestFilter": "LinearMipmapNearestFilter",
    "AnisotropicFilter": "LinearMipmapLinearFilter",
    "UnsignedByteTypeFormat": "UnsignedByteType", "UnsignedByte": "UnsignedByteType", "UNSIGNED_BYTE": "UnsignedByteType",
    "RGBA": "RGBAFormat", "RGBA8": "RGBAFormat", "RGBA8Format": "RGBAFormat", "RGB": "RGBFormat",
    "LinearSpace": "LinearSRGBColorSpace", "LinearSpaceSpace": "LinearSRGBColorSpace",
    "LinearSpaceColorSpace": "LinearSRGBColorSpace", "LinearColorSpace": "LinearSRGBColorSpace",
    "sRGBEncoding": "SRGBColorSpace", "SRGBSpace": "SRGBColorSpace", "LinearEncoding": "LinearSRGBColorSpace",
    "NoneColorSpace": "NoColorSpace", "NoColorColorSpace": "NoColorSpace", "VoidColorSpace": "NoColorSpace",
}

_MSG = re.compile(r"^UNKNOWN_THREE_API: THREE\.([A-Za-z_$][\w$]*) is not a recognized Three\.js API at line \d+$")
_LHS = (("wrap", re.compile(r"\.wrap[STR]\s*=\s*$")), ("min", re.compile(r"\.minFilter\s*=\s*$")),
        ("mag", re.compile(r"\.magFilter\s*=\s*$")), ("colorspace", re.compile(r"colorSpace\s*[:=]\s*$")))
_DATATEX = re.compile(r"new\s+THREE\.DataTexture\s*\(")


def repair_enabled(env=None) -> bool:
    return str((os.environ if env is None else env).get(REPAIR_ENV, "0")).strip() == "1"


def plan(js_errors) -> list[str] | None:
    """The distinct bad names, or None unless EVERY checker failure is a plain UNKNOWN_THREE_API message."""
    errs = [str(e).strip() for e in (js_errors or [])]
    if not errs:
        return None
    names: list[str] = []
    for e in errs:
        m = _MSG.match(e)
        if m is None:
            return None
        if m.group(1) not in names:
            names.append(m.group(1))
    return names


def _datatexture_arg(prefix: str) -> int | None:
    """0-based argument position at the end of `prefix` inside the innermost still-open `new THREE.DataTexture(`."""
    starts = [m.end() for m in _DATATEX.finditer(prefix)]
    if not starts:
        return None
    depth, arg = 0, 0
    for ch in prefix[starts[-1]:]:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                return None
            depth -= 1
        elif ch == "," and depth == 0:
            arg += 1
    return arg


def _pick(name: str, family: str) -> str:
    n = name.lower()
    if family == "wrap":
        return "MirroredRepeatWrapping" if "mirror" in n else "ClampToEdgeWrapping" if "clamp" in n else "RepeatWrapping"
    if family in ("min", "mag"):
        if family == "min" and ("mipmap" in n or "aniso" in n):
            base = "Nearest" if n.startswith("nearest") else "Linear"
            mip = "Nearest" if (n.endswith("nearestfilter") or "mipmapnearest" in n) else "Linear"
            return f"{base}Mipmap{mip}Filter"
        return "NearestFilter" if "nearest" in n else "LinearFilter"
    if family == "colorspace":
        if "linear" in n:
            return "LinearSRGBColorSpace"
        return "SRGBColorSpace" if "srgb" in n else "NoColorSpace"
    if family == "format":
        for pre, out in (("rgba", "RGBAFormat"), ("rgb", "RGBFormat"), ("rg", "RGFormat"), ("red", "RedFormat")):
            if n.startswith(pre):
                return out
        return "RGBAFormat"
    if family == "type":
        return "HalfFloatType" if "half" in n else "FloatType" if "float" in n else "UnsignedByteType"
    raise ValueError(family)


def target_for(name: str, prefix: str) -> str | None:
    """The allowed constant for one occurrence of THREE.<name>, given the source text before it."""
    stmt = re.split(r"[;{}\n]", prefix)[-1]
    for fam, rx in _LHS:
        if rx.search(stmt):
            return _pick(name, fam)
    arg = _datatexture_arg(prefix[-600:])
    if arg == 3:
        return _pick(name, "format")
    if arg == 4:
        return _pick(name, "type")
    if name in KNOWN:
        return KNOWN[name]
    near = difflib.get_close_matches(name, TARGETS, n=1, cutoff=0.8)
    return near[0] if near else None


def rewrite(code: str, names: list[str]) -> tuple[str | None, list[str]]:
    """(new code, ["Old->New", ...]) or (None, [reason]) - all occurrences are mapped or nothing is."""
    if not code or not names:
        return None, ["nothing to repair"]
    rx = re.compile(r"\bTHREE\.(" + "|".join(re.escape(x) for x in sorted(set(names), key=len, reverse=True)) + r")(?![\w$])")
    parts, notes, seen, last = [], [], set(), 0
    for m in rx.finditer(code):
        name = m.group(1)
        if re.search(r"\bnew\s+$", code[max(0, m.start() - 12):m.start()]) or code[m.end():m.end() + 1] == "." \
                or code[m.end():].lstrip()[:1] == "(":
            return None, [f"THREE.{name}: used as constructor/call/member, not a constant"]
        tgt = target_for(name, code[max(0, m.start() - 2000):m.start()])
        if tgt not in TARGETS:
            return None, [f"THREE.{name}: no mapping"]
        parts += [code[last:m.start()], "THREE." + tgt]
        last = m.end()
        seen.add(name)
        notes.append(f"{name}->{tgt}")
    if seen != set(names):
        return None, [f"not found in the code: {sorted(set(names) - seen)}"]
    parts.append(code[last:])
    return "".join(parts), notes


def allowlist_names(path: str | os.PathLike | None = None) -> set[str]:
    """THREE_ALLOWED of the shipped validator (tests use it to prove every target is allowed)."""
    p = Path(path) if path else Path(__file__).with_name("validator") / "threeAllowlist.js"
    body = re.search(r"export const THREE_ALLOWED = new Set\(\[(.*?)\]\);", p.read_text(), re.S).group(1)
    return set(re.findall(r"'([A-Za-z0-9_]+)'", re.sub(r"//[^\n]*", "", body)))
