"""[pgh16 R81] Mechanical repair of coder drafts that pgh16's js_checker rejects. Switch: env PGH_MECH_REPAIR; "1" = on,
unset / anything else = off (default off: pipeline/stages.py never calls this module, behaviour = pgh16 byte for byte).
Second switch: env PGH_CHECK_TIMEOUT_RETRY ("1" = on): a draft whose ONLY failures are TIMEOUT_EXCEEDED is checked once more.

Our own code (stdlib only, deterministic, pure text; no extra node run). Same flow as three_repair.py: the caller re-runs the
SAME checker once on the rewritten source and keeps it only on PASS. A passing draft is never touched.

Error classes (from the checker's own failure text) and the transform for each:
  dup      PARSE_ERROR "Identifier 'X' has already been declared. (L:C)": every later same-block re-declaration of a name
           (const/let/class/function, found by a brace/paren scanner over the comment/string-masked source) is renamed to
           X__r2 together with every reference after it up to the end of that block (member names `.X`, object keys `X:`
           untouched; shorthand `{ X }` becomes `{ X: X__r2 }`). `const X.y = v` (a member write) just loses the keyword.
  paren    PARSE_ERROR "Missing semicolon. (L:C)" where line L has one ')' too many and the char at C is ')': that ')' is
           deleted.
  ident    IDENTIFIER_NOT_ALLOWED "X at line N" (undeclared names): the ONE declared name equal to X up to case and
           underscores (roundedOutlinePoints ~ rounded_outline_points; X itself declared nowhere) replaces X; otherwise `let X = 0;` is declared at the top of generate(). A name used as a call,
           constructor, member base or index base (X( / new X / X. / X[) cannot be stubbed -> no rewrite.
  constlet EXECUTION_THREW "Assignment to constant variable.": every `const` declaration keyword becomes `let`
           (semantics-preserving for code that never reassigns).
  nanprune EXECUTION_THREW "Invalid bounds" (NaN/inf geometry reaching the draft's own bbox normalisation): just before that
           normalisation's `Box3().setFromObject(R)` statement, objects under R with non-finite position/scale or vertex
           positions are removed - but only when they hold <= 50% of R's vertices (else nothing is removed).
  texture  TEXTURE_DATA_EXCEEDED "<b> bytes (limit <l>)" with b <= 4*l: every `new THREE.DataTexture(` call goes through a
           helper that box-downsamples an RGBA Uint8Array texture of >= 256x256 (even sizes) to half width and height.
Anything else (other runtime errors, draw-call / heap limits, mixed unknown rules) -> no rewrite.
"""
from __future__ import annotations

import os
import re

REPAIR_ENV = "PGH_MECH_REPAIR"
TIMEOUT_ENV = "PGH_CHECK_TIMEOUT_RETRY"
MAX_CODE = 1_200_000          # larger sources are not touched (the checker's own limit is 1 MiB)
IDENT = r"[A-Za-z_$][\w$]*"
KEYWORDS = {
    "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do", "else", "export",
    "extends", "finally", "for", "function", "if", "import", "in", "instanceof", "let", "new", "return", "super",
    "switch", "this", "throw", "try", "typeof", "var", "void", "while", "with", "yield", "await", "of", "null", "true",
    "false", "undefined", "NaN", "Infinity", "static", "get", "set", "async",
}


def _on(env, name) -> bool:
    return str((os.environ if env is None else env).get(name, "0")).strip() == "1"


def repair_enabled(env=None) -> bool:
    return _on(env, REPAIR_ENV)


def timeout_retry_enabled(env=None) -> bool:
    return _on(env, TIMEOUT_ENV)


def timeout_only(js_errors) -> bool:
    errs = list(js_errors or [])
    return bool(errs) and all(str(e).split(":", 1)[0].strip() == "TIMEOUT_EXCEEDED" for e in errs)


# ------------------------------------------------------------------ masking (comments / strings / template text -> spaces)
def mask(code: str) -> str:
    """Same length as code; comment, string and template-literal TEXT replaced by spaces (newlines kept). Code inside
    `${...}` of a template stays visible. Regex literals are not recognised (rare in drafts)."""
    out = list(code)
    i, n = 0, len(code)
    tstack = []          # brace depth at each open `${` of a template
    depth = 0

    def blank(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    def template_from(i):        # i = index after the opening backtick (or after the closing } of ${...}); returns new i
        j = i
        while j < n:
            c = code[j]
            if c == "\\":
                j += 2; continue
            if c == "`":
                blank(i, j); return j + 1, False
            if c == "$" and j + 1 < n and code[j + 1] == "{":
                blank(i, j); return j + 2, True
            j += 1
        blank(i, n); return n, False

    while i < n:
        c = code[i]
        if c == "/" and i + 1 < n and code[i + 1] == "/":
            j = code.find("\n", i); j = n if j < 0 else j
            blank(i, j); i = j; continue
        if c == "/" and i + 1 < n and code[i + 1] == "*":
            j = code.find("*/", i + 2); j = n if j < 0 else j + 2
            blank(i, j); i = j; continue
        if c in "'\"":
            j = i + 1
            while j < n and code[j] != c and code[j] != "\n":
                j += 2 if code[j] == "\\" else 1
            blank(i + 1, min(j, n)); i = j + 1; continue
        if c == "`":
            i, opened = template_from(i + 1)
            if opened:
                tstack.append(depth); depth += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if tstack and depth == tstack[-1]:
                tstack.pop()
                i, opened = template_from(i + 1)
                if opened:
                    tstack.append(depth); depth += 1
                continue
        i += 1
    return "".join(out)


def _line_starts(code):
    starts = [0]
    for m in re.finditer("\n", code):
        starts.append(m.end())
    return starts


def _pos(code, line, col, starts=None):
    starts = starts or _line_starts(code)
    if line < 1 or line > len(starts):
        return None
    return starts[line - 1] + col


# ------------------------------------------------------------------ scope scanner (blocks + declarations)
def _scope_info(m: str, positions):
    """One pass over masked source m. Returns (blocks, info): blocks = [open, close, paren_depth_at_open] per `{` (index 0 =
    the whole file); info[pos] = (innermost block index, paren depth) for every pos in positions."""
    want = sorted(set(positions)); wi = 0
    blocks, stack, paren, info = [[-1, len(m), 0]], [0], 0, {}
    for i, c in enumerate(m):
        while wi < len(want) and want[wi] == i:
            info[i] = (stack[-1], paren); wi += 1
        if c == "(":
            paren += 1
        elif c == ")":
            paren -= 1
        elif c == "{":
            blocks.append([i, len(m), paren]); stack.append(len(blocks) - 1)
        elif c == "}":
            if len(stack) > 1:
                blocks[stack.pop()][1] = i
    while wi < len(want):
        info[want[wi]] = (stack[-1], paren); wi += 1
    return blocks, info


_STOP_PREV = set(",=+-*/%&|^?:<>(![{")
_CONT_NEXT = set(".,?:+-*/%&|^)]=<>")


def declarations(m: str):
    """(name, name_pos, kind_pos) for const/let/class/function declarations in masked source m. Multi-declarator
    const/let statements (`const a = 1, b = 2`) yield every simple identifier binding; destructuring patterns are skipped."""
    out = []
    for km in re.finditer(r"(?<![\w$.])(const|let|class|function)(?![\w$])", m):
        kind = km.group(1)
        j = km.end()
        if kind in ("class", "function"):
            nm = re.compile(r"\s*\*?\s*(" + IDENT + ")").match(m, j)
            if nm and nm.group(1) not in KEYWORDS:
                out.append((nm.group(1), nm.start(1), km.start()))
            continue
        # const / let: walk the declarator list
        n = len(m)
        expect_name = True
        depth = 0
        while j < n:
            if expect_name:
                nm = re.compile(r"\s*(" + IDENT + r"|[\[{])").match(m, j)
                if not nm:
                    break
                tok = nm.group(1)
                if tok in "[{":
                    j = nm.end(1) - 1     # skip the pattern by bracket matching below
                elif tok in KEYWORDS:
                    break
                else:
                    out.append((tok, nm.start(1), km.start())); j = nm.end(1)
                expect_name = False
                continue
            c = m[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth < 0:
                    break
            elif depth == 0 and c == ";":
                break
            elif depth == 0 and c == ",":
                expect_name = True
            elif depth == 0 and c == "\n":
                prev = m[:j].rstrip()[-1:] if m[:j].rstrip() else ""
                nxt = m[j:].lstrip()[:1]
                if prev not in _STOP_PREV and nxt not in _CONT_NEXT:
                    break
            elif depth == 0 and re.compile(r"(?<![\w$])(of|in)(?![\w$])").match(m, j) and not expect_name:
                break
            j += 1
    return out


def _ident_refs(m: str, name: str, a: int, b: int):
    """Positions of identifier tokens `name` in m[a:b] that are not member names (`.name`)."""
    pat = re.compile(r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])")
    res = []
    for r in pat.finditer(m, a, b):
        # `. name` with spaces / optional chaining `?.name`
        k = r.start() - 1
        while k >= 0 and m[k] in " \t":
            k -= 1
        if k >= 0 and m[k] == ".":
            continue
        res.append(r.start())
    return res


def _rename_edits(m: str, name: str, new: str, a: int, b: int):
    """{pos: (end, replacement)} renaming identifier tokens `name` in m[a:b] (object keys kept, shorthand expanded)."""
    edits = {}
    for p in _ident_refs(m, name, a, b):
        e = p + len(name)
        k = e
        while k < len(m) and m[k] in " \t\n":
            k += 1
        nxt = m[k] if k < len(m) else ""
        q = p - 1
        while q >= 0 and m[q] in " \t\n":
            q -= 1
        prv = m[q] if q >= 0 else ""
        if nxt == ":" and prv in "{,":
            continue                                       # object key `name: ...`
        if prv in "{," and nxt in ",}" and _open_bracket(m, p) == "{":
            edits[p] = (e, f"{name}: {new}")               # shorthand property `{ name }`
            continue
        edits[p] = (e, new)
    return edits


def _apply(code: str, edits) -> str:
    for p in sorted(edits, reverse=True):
        e, s = edits[p]
        code = code[:p] + s + code[e:]
    return code


def _rename(code: str, m: str, name: str, new: str, a: int, b: int) -> str:
    return _apply(code, _rename_edits(m, name, new, a, b))


def _open_bracket(m: str, pos: int) -> str:
    depth = {")": 0, "]": 0, "}": 0}
    pair = {"(": ")", "[": "]", "{": "}"}
    for k in range(pos - 1, -1, -1):
        c = m[k]
        if c in ")]}":
            depth[c] += 1
        elif c in "([{":
            if depth[pair[c]] == 0:
                return c
            depth[pair[c]] -= 1
    return ""


def fresh(code: str, name: str, taken=()) -> str:
    k = 2
    while f"{name}__r{k}" in taken or re.search(r"(?<![\w$])" + re.escape(f"{name}__r{k}") + r"(?![\w$])", code):
        k += 1
    return f"{name}__r{k}"


# ------------------------------------------------------------------ transforms
def fix_dup(code: str):
    m = mask(code)
    decls = declarations(m)
    blocks, info = _scope_info(m, [d[1] for d in decls] + [d[2] for d in decls])
    seen, dups = set(), []
    for name, npos, kpos in decls:
        blk, _ = info[npos]
        if info[kpos][1] != blocks[blk][2]:
            continue                       # for-head / parameter default etc.: its own scope
        if (blk, name) in seen:
            dups.append((npos, name, blk))
        else:
            seen.add((blk, name))
    if not dups:
        return None, []
    edits, notes, taken = {}, [], set()
    for npos, name, blk in sorted(dups):
        k = npos + len(name)
        while k < len(m) and m[k] in " \t":
            k += 1
        kpos = next(d[2] for d in decls if d[1] == npos)
        if k < len(m) and m[k] in ".[" and re.fullmatch(r"(const|let)\s+", m[kpos:npos]):
            edits[kpos] = (npos, "")                       # `const x.y = v` -> `x.y = v` (a member write, not a declaration)
            notes.append(f"dup:{name}:drop-keyword")
            continue     # a later re-declaration overrides the earlier one's edits from its position on
        new = fresh(code, name, taken); taken.add(new)
        edits.update(_rename_edits(m, name, new, npos, blocks[blk][1]))
        notes.append(f"dup:{name}->{new}")
    return _apply(code, edits), notes


def fix_paren(code: str, line: int, col: int):
    m = mask(code)
    starts = _line_starts(code)
    p = _pos(code, line, col, starts)
    if p is None or p >= len(m) or m[p] != ")":
        return None, []
    a = starts[line - 1]; b = starts[line] - 1 if line < len(starts) else len(m)
    seg = m[a:b]
    if seg.count("(") - seg.count(")") != -1:
        return None, []
    return code[:p] + code[p + 1:], [f"paren:-1@{line}:{col}"]


_GEN_OPEN = re.compile(r"function\s+generate\s*\([^)]*\)\s*\{")


def fix_ident(code: str, names):
    m = mask(code)
    declared = {d[0] for d in declarations(m)}
    for pm in re.finditer(r"\(([^()]*)\)\s*(?:=>|\{)", m):         # params of functions / arrows (rough)
        declared.update(x for x in re.findall(IDENT, pm.group(1)) if x not in KEYWORDS)
    stubs, notes = [], []
    for name in names:
        refs = _ident_refs(m, name, 0, len(m))
        if not refs:
            return None, [f"ident:{name}:no-ref"]
        norm = name.replace("_", "").lower()
        close = [] if name in declared else sorted(d for d in declared if d != name and d.replace("_", "").lower() == norm)
        if len(close) == 1:
            code = _rename(code, m, name, close[0], 0, len(m)); m = mask(code)
            notes.append(f"ident:{name}->{close[0]}")
            continue
        for p in refs:
            k = p + len(name)
            while k < len(m) and m[k] in " \t\n":
                k += 1
            q = p - 1
            while q >= 0 and m[q] in " \t":
                q -= 1
            if (k < len(m) and m[k] in "(.[`") or m[max(0, q - 2):q + 1] == "new":
                return None, [f"ident:{name}:object-use"]
        stubs.append(name)
    if stubs:
        g = _GEN_OPEN.search(m)
        if not g:
            return None, ["ident:no-generate"]
        code = code[:g.end()] + "\n  let " + ", ".join(f"{s} = 0" for s in stubs) + ";  // [mech_repair] undeclared" + code[g.end():]
        notes += [f"ident:{s}=0" for s in stubs]
    return code, notes


def fix_constlet(code: str):
    m = mask(code)
    pos = [r.start() for r in re.finditer(r"(?<![\w$.])const(?=\s)", m)]
    if not pos:
        return None, []
    for p in reversed(pos):
        code = code[:p] + "let  " + code[p + 5:]       # same length (keeps line:col of later errors)
    return code, [f"constlet:{len(pos)}"]


_PRUNE = ("{ const __mr_all = []; let __mr_n = 0, __mr_bad = 0; %s.traverse((o) => { "
          "const p = o.geometry && o.geometry.attributes && o.geometry.attributes.position; const a = p ? p.array : null; "
          "let bad = !(Number.isFinite(o.position.x) && Number.isFinite(o.position.y) && Number.isFinite(o.position.z) && "
          "Number.isFinite(o.scale.x) && Number.isFinite(o.scale.y) && Number.isFinite(o.scale.z)); "
          "if (!bad && a) { for (let i = 0; i < a.length; i++) { if (!Number.isFinite(a[i])) { bad = true; break; } } } "
          "const nv = a ? a.length / 3 : 0; __mr_n += nv; if (bad && o !== %s) { __mr_all.push(o); __mr_bad += nv; } }); "
          "if (__mr_bad <= 0.5 * __mr_n) { for (const o of __mr_all) { if (o.parent) o.parent.remove(o); } } }  // [mech_repair] nanprune\n")


def fix_nan(code: str):
    m = mask(code)
    t = re.search(r"throw\s+new\s+Error\(\s*'Invalid bounds'|throw\s+new\s+Error\(\s*\"Invalid bounds\"", code)
    if not t or "__mr_all" in code:
        return None, []
    sfo = None
    for r in re.compile(r"\.setFromObject\(\s*(" + IDENT + r")\s*\)").finditer(m, 0, t.start()):
        sfo = r
    if sfo is None or t.start() - sfo.start() > 1500:
        return None, ["nan:no-bbox-anchor"]
    ls = m.rfind("\n", 0, sfo.start()) + 1
    indent = re.match(r"[ \t]*", code[ls:]).group(0)
    root = sfo.group(1)
    return code[:ls] + indent + _PRUNE % (root, root) + code[ls:], [f"nanprune:{root}"]


_DS = ("const __mr_tex = (d, w, h, ...r) => { if (d instanceof Uint8Array && w >= 256 && h >= 256 && w %% 2 === 0 && "
       "h %% 2 === 0 && d.length === w * h * 4) { const W = w / 2, H = h / 2, o = new Uint8Array(W * H * 4); "
       "for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 4; c++) { "
       "const s = ((2 * y) * w + 2 * x) * 4 + c; o[(y * W + x) * 4 + c] = (d[s] + d[s + 4] + d[s + w * 4] + d[s + w * 4 + 4] + 2) >> 2; } "
       "return new THREE.DataTexture(o, W, H, ...r); } return new THREE.DataTexture(d, w, h, ...r); };  // [mech_repair] texture\n")


def fix_texture(code: str, used: int, limit: int):
    if used > 4 * limit or "__mr_tex" in code:
        return None, []
    m = mask(code)
    calls = [r for r in re.finditer(r"new\s+THREE\.DataTexture\s*\(", m)]
    g = _GEN_OPEN.search(m)
    if not calls or not g:
        return None, []
    for r in reversed(calls):
        code = code[:r.start()] + "__mr_tex(" + code[r.end():]
    code = code[:g.end()] + "\n  " + _DS % () + code[g.end():]
    return code, [f"texture:{len(calls)}"]


# ------------------------------------------------------------------ classification + driver
_RE_DUP = re.compile(r"^PARSE_ERROR: Identifier '(" + IDENT + r")' has already been declared\. \((\d+):(\d+)\)")
_RE_SEMI = re.compile(r"^PARSE_ERROR: Missing semicolon\. \((\d+):(\d+)\)")
_RE_IDENT = re.compile(r"^IDENTIFIER_NOT_ALLOWED: (" + IDENT + r") at line \d+$")
_RE_TEX = re.compile(r"^TEXTURE_DATA_EXCEEDED: (\d+) bytes \(limit (\d+)\)")


def classify(js_errors):
    """One repairable class for the whole failure list, or None. Every failure must belong to it."""
    errs = [str(e) for e in (js_errors or [])]
    if not errs:
        return None
    if all(_RE_IDENT.match(e) for e in errs):
        return "ident"
    if len(errs) != 1:
        return None
    e = errs[0]
    if _RE_DUP.match(e):
        return "dup"
    if _RE_SEMI.match(e):
        return "paren"
    if e.startswith("EXECUTION_THREW:") and "Assignment to constant variable" in e:
        return "constlet"
    if e.startswith("EXECUTION_THREW:") and "Invalid bounds" in e:
        return "nanprune"
    if _RE_TEX.match(e):
        return "texture"
    return None


def repair(code: str, js_errors):
    """Return (fixed_code | None, notes). None = not repairable / nothing changed (the caller keeps the draft as is)."""
    if not code or len(code) > MAX_CODE:
        return None, ["skip:size"]
    cls = classify(js_errors)
    if cls is None:
        return None, ["skip:class"]
    errs = [str(e) for e in js_errors]
    try:
        if cls == "ident":
            names = list(dict.fromkeys(_RE_IDENT.match(e).group(1) for e in errs))
            fixed, notes = fix_ident(code, names)
        elif cls == "dup":
            fixed, notes = fix_dup(code)
        elif cls == "paren":
            mm = _RE_SEMI.match(errs[0]); fixed, notes = fix_paren(code, int(mm.group(1)), int(mm.group(2)))
        elif cls == "constlet":
            fixed, notes = fix_constlet(code)
        elif cls == "nanprune":
            fixed, notes = fix_nan(code)
        else:
            mm = _RE_TEX.match(errs[0]); fixed, notes = fix_texture(code, int(mm.group(1)), int(mm.group(2)))
    except Exception as exc:                      # a transform bug never breaks the pipeline
        return None, [f"{cls}:error:{type(exc).__name__}"]
    if fixed is None or fixed == code:
        return None, notes or [f"{cls}:no-change"]
    return fixed, notes
