"""[pgh1v3] Coder draft count K (actors.coder.ensemble_size) - runtime switch PGH_K_MODE (owner 19/10: two versions).
MODES (PGH_K_MODE, read once at container start; image default "fixed"; K40 = PGH_K40, "the K for a 40.8-CPU pod"):
  fixed     K = K40 on every machine; no CPU detection needed for K (cores are still read once, for the log and the
            time guard's model). A smaller pod is just slower, a bigger one faster.
  adaptive  K = max(K40, rule(Q)) (rule <= k_max 80, so K = K40 when K40 >= 80): never below K40, more CPUs -> more
            drafts. Detection failure -> K40.
  rule      K = rule(Q) (the ad93c9f behaviour below: 10 cores -> 8, >= 54 -> 74; detection failure -> 26).
  off       K = configuration.yaml actors.coder.ensemble_size (68) = pgh1v2 (the start-up hook does not call this module).
  Anything else -> '!!' note + "fixed". Legacy: PGH_ADAPTIVE_K set to anything but "1" (e.g. 0) = off, overriding
  PGH_K_MODE (as in ad93c9f, where only exactly "1" was on); unset or "1" = PGH_K_MODE decides.
K40: PGH_K40 env (integer K40_MIN..K40_MAX = 4..128; [pgh16] cap 80 -> 128); unset/"" -> BUILD_K40 (the generator writes its --k40, default 61 =
  the rule at 40.8 cores, into this file and into the Dockerfile ENV); bad value -> '!!' note + BUILD_K40.
PGH_ADAPTIVE_K_FORCE=<n> still wins in every mode but off (tests / smoke runs).

THE RULE (mode "rule", and the rule(Q) part of "adaptive"):
GENERAL RULE (owner 19/10 07:13 VN "I want it general"), fitted by the time-guard study
(one closed-form fit over 343 simulated round cells, rms error 4.8%). No lookup table.
Installed by make_pgh1_v3.py as pipeline_service/modules/metrics/adaptive_k.py and called ONCE in
GenerationPipeline.startup (before build_pipeline reads K; warm-up is not timed), unless the mode is off. The
batch-boundary round guard (pgh_timeguard.py, PGH_TIMEGUARD) is the reproducible last net and uses per_batch_s() below.

Rule (Q = effective cores, read once):
    R(Q)         = min(usable_frac * kappa * Q - non_render_cores, render_sat_cores)     render cores usable
    per_draft(Q) = (C^n + (prompts_per_batch * c_r / R(Q))^n)^(1/n)    batch seconds per extra draft: smooth max (n) of the
                                                                      coder time C and the render time per draft
    K = clamp(floor((target_s / round_factor - overhead_s) / per_draft(Q)), k_min, k_max);   R(Q) <= r_min -> k_min
    batch time per_batch_s(K) = overhead_s + K * per_draft(Q);  round = round_factor * per_batch_s  (4 batches x 0.935)
  Same Q -> same K, always (no timing input) -> reproducible per pod size.
Coefficients (DEFAULT_COEFFS; JSON keys of PGH_ADAPTIVE_K_COEFFS; study values, PROVISIONAL until the plan B refit):
  target_s 6000            round target. The study RECOMMENDS 6,000 until kappa is calibrated (at 6,500 the guard fires in
                           24-34% of correctly modelled runs); 6,000 vs the owner's 6,500 is THE OWNER'S CALL.
  round_factor 3.74        round total / batch-0 time (4 batches x 0.935)
  overhead_s 92            A: fixed per-batch seconds (start + post-render bracket tail). FIT, +-40 s
  coder_s_per_draft 17.5   C: coder seconds per draft per batch of 32 prompts. FIT (measured ak68 17.6, fin68r 17.2)
  render_core_s_per_draft 31.8   c_r: render core-seconds per draft (speed8-equivalent cores). FIT, +-20%
  prompts_per_batch 32     smooth_n 4 (FIT)
  usable_frac 0.95         U, ASSUMED;  non_render_cores 7: N, MEASURED 6-7 (peaks ~11)
  kappa 1.32               speed8-equivalent cores per contest-pod core. EXTRAPOLATED, NOT calibrated (range 1.0-1.5)
  render_sat_cores 61.5    R_SAT, ASSUMED;  r_min 0.5 (R <= r_min -> k_min)
  k_min 4, k_max 80        k_max = largest K ever run (k80)
  k_fallback 26            K on detection failure = the rule at 26 cores with kappa 1.0 and target 6,000
K with the defaults (target 6,000): 10/16/20/26/32/33/40.8/54/57 cores -> 8/19/26/37/48/49/61/74/74, >= 57 -> 74
  (target 6,500: 8/21/29/41/52/54/66/80/80). The study's audit table (REPORT 4c) is matched within +-1 (32 cores: 47 there;
  6,500: 26 -> 40, 33 -> 53 there; the REPORT prints rounded coefficients); test_adaptive_k.py checks every core count. NOTE: a 40.8-core pod gets K 61 (rule); pgh1v2 (and off) runs K 68.

Env (read at container start; Dockerfile ENV defaults, docker run -e overrides; no rebuild):
  PGH_K_MODE / PGH_K40 / PGH_ADAPTIVE_K   see MODES above.
  PGH_ADAPTIVE_K_COEFFS     JSON object overriding any subset of DEFAULT_COEFFS (kappa and target_s included), e.g.
                            '{"target_s": 6500, "kappa": 1.0}'. Unknown key / bad value / inconsistent set -> ALL defaults
                            (src notes coeffs-error).
  PGH_ADAPTIVE_K_FORCE=<n>  use exactly K=n (tests / smoke runs).
Cores (read ONCE at start-up, logged): effective = min(cgroup quota, affinity CPUs, os.cpu_count()) of the known values.
  quota: cgroup v2 <root>/cpu.max ('max P' = unlimited | 'N P' -> N/P, e.g. 4080000 100000 = 40.8), else cgroup v1
         <root>/{cpu,cpu,cpuacct,cpuacct,cpu,.}/cpu.cfs_quota_us + cpu.cfs_period_us (-1 = unlimited).
  affinity: len(os.sched_getaffinity(0)) (cpuset limits).  cpu_count: os.cpu_count().
Detection failure = no cgroup CPU file at all, garbage, zero/negative values, unreadable file, or no CPU count. Then
  (mode rule) K = k_fallback (26), lowered to the rule's K at the visible CPUs when that is smaller (the quota can only be below the
  visible CPUs, so this never raises K). Unlimited quota ('max' / -1) is NOT a failure: cores = visible CPUs.
All paths are injectable (root, affinity, cpu_count, env) for the unit tests (remote8/kern/v3/test_adaptive_k.py).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

# timeguard/REPORT.txt 4a (PROVISIONAL until the plan B refit; see the docstring). Units: seconds, cores, drafts.
DEFAULT_COEFFS: dict = {
    "target_s": 6000.0, "round_factor": 3.74, "overhead_s": 92.0, "coder_s_per_draft": 17.5,
    "render_core_s_per_draft": 31.8, "prompts_per_batch": 32, "smooth_n": 4.0, "usable_frac": 0.95, "kappa": 1.32,
    "non_render_cores": 7.0, "render_sat_cores": 61.5, "r_min": 0.5, "k_min": 4, "k_max": 80, "k_fallback": 26}
INT_KEYS = ("k_min", "k_max", "k_fallback", "prompts_per_batch")
CGROUP_ROOT = "/sys/fs/cgroup"
LAST_DECISION: dict | None = None   # set by apply(); pgh_timeguard reuses its detected cores (CPU read once)
V1_DIRS = ("cpu", "cpu,cpuacct", "cpuacct,cpu", ".")
MODES = ("fixed", "adaptive", "rule", "off")
DEFAULT_MODE = "fixed"
BUILD_K40 = 70  # [build] K40 default; [pgh16] = the Dockerfile PGH_K40 (pgh16/set_k.sh sets both; 80 = placeholder)
K40_MIN, K40_MAX = 4, 128   # [pgh16] K40_MAX 80 -> 128 so PGH_K40 can exceed 80 (as pgh15)


class DetectError(RuntimeError):
    """The CPU quota could not be determined -> conservative fallback K."""


def k_mode(env=None) -> tuple[str, list]:
    """(mode, notes). Legacy PGH_ADAPTIVE_K set to anything but "1" -> off; else PGH_K_MODE (unset/"" -> DEFAULT_MODE,
    unknown -> '!!' note + DEFAULT_MODE). Must agree with the start-up hook's condition (make_pgh1_v3.py AK_COND)."""
    env = os.environ if env is None else env
    legacy = env.get("PGH_ADAPTIVE_K")
    if legacy is not None and str(legacy).strip() != "1":
        return "off", [f"legacy PGH_ADAPTIVE_K={legacy!r} (not '1') -> off"]
    raw = str(env.get("PGH_K_MODE", "") or "").strip().lower()
    if not raw:
        return DEFAULT_MODE, []
    if raw == "budget":   # [R82 kbudget] batch K = fixed K40; budget_k.py spreads it over the photos by draft length
        return "fixed", ["PGH_K_MODE=budget: batch K as fixed, per-photo K by modules/metrics/budget_k.py"]
    if raw not in MODES:
        return DEFAULT_MODE, [f"!! bad PGH_K_MODE {env.get('PGH_K_MODE')!r} (want {'|'.join(MODES)}) -> {DEFAULT_MODE}"]
    return raw, []


def switch_on(env=None) -> bool:
    """K is decided by this module (mode != off). False = the configured K (= pgh1v2)."""
    return k_mode(env)[0] != "off"


def read_k40(env=None, build: int | None = None) -> tuple[int, str, list]:
    """(K40, source env|build, notes). PGH_K40 must be an integer K40_MIN..K40_MAX; bad -> '!!' note + build default."""
    env = os.environ if env is None else env
    build = BUILD_K40 if build is None else int(build)
    raw = str(env.get("PGH_K40", "") or "").strip()
    if not raw:
        return build, "build", []
    try:
        x = _num(raw)
        if x != int(x) or not K40_MIN <= x <= K40_MAX:
            raise ValueError(f"not an integer in {K40_MIN}..{K40_MAX}")
        return int(x), "env", []
    except (ValueError, OverflowError) as exc:
        return build, "build", [f"!! bad PGH_K40 {raw!r} ({exc}) -> build default {build}"]


def _num(text: str) -> float:
    x = float(text)
    if not math.isfinite(x):
        raise ValueError(f"not finite: {text!r}")
    return x


def read_cpu_quota(root: str = CGROUP_ROOT) -> tuple[float | None, str]:
    """(cores allowed by the cgroup, source). None = unlimited. Raises DetectError when there is no cgroup CPU file,
    or it is unreadable / malformed / non-positive (never guesses)."""
    r = Path(root)
    v2 = r / "cpu.max"
    if v2.exists():
        try:
            parts = v2.read_text().split()
            if parts[0] == "max":
                return None, "cgroup2:max"
            q, per = _num(parts[0]), _num(parts[1])
        except (OSError, IndexError, ValueError) as exc:
            raise DetectError(f"cpu.max unreadable/malformed: {exc!r}") from exc
        if q <= 0 or per <= 0:
            raise DetectError(f"cpu.max bad values {q} {per}")
        return q / per, "cgroup2"
    for d in V1_DIRS:
        fq, fp = r / d / "cpu.cfs_quota_us", r / d / "cpu.cfs_period_us"
        if fq.exists():
            try:
                q = _num(fq.read_text().strip())
                if q == -1:
                    return None, "cgroup1:-1"
                per = _num(fp.read_text().strip())
            except (OSError, ValueError) as exc:
                raise DetectError(f"cgroup v1 cfs files unreadable/malformed: {exc!r}") from exc
            if q <= 0 or per <= 0:
                raise DetectError(f"cgroup v1 bad values {q} {per}")
            return q / per, "cgroup1"
    raise DetectError(f"no cgroup cpu.max or cpu.cfs_quota_us under {root}")


def _affinity() -> int | None:
    try:
        return len(os.sched_getaffinity(0)) or None
    except (AttributeError, OSError):
        return None


def detect(root: str = CGROUP_ROOT, affinity=_affinity, cpu_count=os.cpu_count) -> dict:
    """{cores, quota, quota_src, affinity, cpu_count, visible, ok, reason}. cores = min of the known values.
    ok=False (with cores = the visible CPUs or None) when the quota cannot be determined."""
    def safe(fn):
        try:
            v = fn()
            return int(v) if v and int(v) > 0 else None
        except Exception:
            return None
    aff, cnt = safe(affinity), safe(cpu_count)
    known = [x for x in (aff, cnt) if x]
    visible = min(known) if known else None
    out = dict(quota=None, quota_src=None, affinity=aff, cpu_count=cnt, visible=visible, cores=visible, ok=False, reason="")
    try:
        quota, src = read_cpu_quota(root)
    except DetectError as exc:
        out.update(quota_src="fail", reason=str(exc))
        return out
    out.update(quota=quota, quota_src=src)
    vals = [x for x in (quota, visible) if x is not None]
    if not vals:
        out["reason"] = "quota unlimited and no CPU count"
        return out
    out.update(cores=min(vals), ok=True)
    return out


def normalize_coeffs(raw, base: dict | None = None) -> dict:
    """JSON object text (or dict) overriding keys of `base` (DEFAULT_COEFFS) -> validated full dict. Raises ValueError."""
    c = dict(DEFAULT_COEFFS if base is None else base)
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    if not isinstance(raw, dict):
        raise ValueError(f"coefficients must be a JSON object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(DEFAULT_COEFFS))
    if unknown:
        raise ValueError(f"unknown coefficient(s) {unknown}; known: {sorted(DEFAULT_COEFFS)}")
    for key, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ValueError(f"{key}={v!r} is not a number")
        c[key] = _num(str(v))
    for key in INT_KEYS:
        if c[key] != int(c[key]):
            raise ValueError(f"{key}={c[key]} must be an integer")
        c[key] = int(c[key])
    if not 2 <= c["k_min"] <= c["k_max"]:
        raise ValueError(f"need 2 <= k_min <= k_max (got {c['k_min']}, {c['k_max']})")
    if not c["k_min"] <= c["k_fallback"] <= c["k_max"]:
        raise ValueError(f"need k_min <= k_fallback <= k_max (got {c['k_fallback']})")
    for key in ("overhead_s", "coder_s_per_draft", "render_core_s_per_draft", "non_render_cores", "r_min"):
        if c[key] < 0:
            raise ValueError(f"{key}={c[key]} < 0")
    for key in ("target_s", "round_factor", "prompts_per_batch", "smooth_n", "usable_frac", "kappa", "render_sat_cores"):
        if c[key] <= 0:
            raise ValueError(f"{key}={c[key]} must be > 0")
    if c["coder_s_per_draft"] == 0 and c["render_core_s_per_draft"] == 0:
        raise ValueError("coder_s_per_draft and render_core_s_per_draft are both 0 (K would be unbounded)")
    return c


def render_cores(cores: float, c: dict) -> float:
    """R(Q) = min(usable_frac * kappa * Q - non_render_cores, render_sat_cores)."""
    return min(c["usable_frac"] * c["kappa"] * cores - c["non_render_cores"], c["render_sat_cores"])


def per_draft_s(cores: float, c: dict) -> float | None:
    """Batch seconds per extra draft at `cores`; None = no render CPU (R(Q) <= r_min)."""
    r = render_cores(cores, c)
    if r <= c["r_min"]:
        return None
    render = c["prompts_per_batch"] * c["render_core_s_per_draft"] / r
    n = c["smooth_n"]
    return (c["coder_s_per_draft"] ** n + render ** n) ** (1.0 / n)


def per_batch_s(k: int, cores: float, c: dict) -> float | None:
    """Model batch time at K drafts: overhead_s + K * per_draft(Q) (the guard's step model)."""
    pd = per_draft_s(cores, c)
    return None if pd is None else c["overhead_s"] + k * pd


def formula_k(cores: float, c: dict) -> int:
    """K = clamp(floor((target_s / round_factor - overhead_s) / per_draft(Q)), k_min, k_max)."""
    pd, budget = per_draft_s(cores, c), c["target_s"] / c["round_factor"] - c["overhead_s"]
    if pd is None or budget <= 0:
        return c["k_min"]
    return int(max(c["k_min"], min(c["k_max"], math.floor(budget / pd + 1e-9))))


def predicted_round_s(k: int, cores: float, c: dict) -> float | None:
    pb = per_batch_s(k, cores, c)
    return None if pb is None else c["round_factor"] * pb


def _coeffs_text(c: dict) -> str:
    return json.dumps({k: (int(v) if isinstance(v, float) and v == int(v) else v) for k, v in c.items()},
                      separators=(",", ":"))


def decide(fixed_k: int, env=None, root: str = CGROUP_ROOT, affinity=_affinity, cpu_count=os.cpu_count) -> dict:
    """Pure decision, no side effects. src: single | off | force | fixed | adaptive-rule | adaptive-floor |
    adaptive-fallback | formula | fallback (the last two = mode rule, unchanged from ad93c9f)."""
    env = os.environ if env is None else env
    out = dict(k=int(fixed_k), fixed=int(fixed_k), src="off", mode="off", k40=None, k40_src="", rule_k=None, det=None,
               coeffs="", coeffs_src="", pred_s=None, reason="")
    if fixed_k <= 1:                                   # single-pass config: nothing to adapt
        out["src"] = "single"
        return out
    mode, notes = k_mode(env)
    out["mode"] = mode
    if mode == "off":
        out["reason"] = "; ".join(notes) or "PGH_K_MODE=off"
        return out
    raw = str(env.get("PGH_ADAPTIVE_K_COEFFS", "") or "").strip()
    co, out["coeffs_src"] = dict(DEFAULT_COEFFS), "default-PROVISIONAL"
    if raw:
        try:
            co, out["coeffs_src"] = normalize_coeffs(raw), "env"
        except (ValueError, TypeError) as exc:
            notes.append(f"coeffs-error ({type(exc).__name__}: {exc}) -> default coefficients")
    _gs = __import__("sys").modules.get("pgh_gsplit")   # [gsplit] 3 coder GPUs: coefficients fitted on 2 rescaled
    if _gs is not None and _gs.on(env):
        co, _gn = _gs.scale_adaptive_k(co, env)
        out["coeffs_src"] += "+gsplit"
        notes.append(_gn)
    out["coeffs"] = _coeffs_text(co)
    k40, out["k40_src"], n40 = read_k40(env)
    out["k40"] = k40
    if mode in ("fixed", "adaptive"):
        notes += n40
    force = str(env.get("PGH_ADAPTIVE_K_FORCE", "") or "").strip()
    if force:
        try:
            out.update(k=max(2, int(force)), src="force")
            out["reason"] = "; ".join(notes)
            return out
        except ValueError:
            notes.append(f"bad PGH_ADAPTIVE_K_FORCE {force!r} ignored")
    try:
        det = detect(root, affinity, cpu_count)
    except Exception as exc:  # never in practice (detect catches its own errors); fixed mode must not depend on it
        det = dict(quota=None, quota_src="fail", affinity=None, cpu_count=None, visible=None, cores=None, ok=False,
                   reason=f"detect raised {exc!r}")
    out["det"] = det
    if mode == "fixed":
        out.update(k=k40, src="fixed")
        if not det["ok"]:
            notes.append(f"cores unknown (info only, K does not depend on them): {det['reason']}")
    elif mode == "adaptive":
        if det["ok"]:
            rk = formula_k(det["cores"], co)
            out.update(k=max(k40, rk), rule_k=rk, src="adaptive-rule" if rk > k40 else "adaptive-floor")
        else:
            out.update(k=k40, src="adaptive-fallback")
            notes.append(f"detection failed: {det['reason']} -> K40")
    elif det["ok"]:                                    # mode rule = ad93c9f
        out.update(k=formula_k(det["cores"], co), src="formula")
    else:
        k = co["k_fallback"]
        if det["visible"]:
            k = min(k, formula_k(det["visible"], co))
        out.update(k=k, src="fallback")
        notes.append(f"detection failed: {det['reason']}")
    if det["ok"]:
        out["pred_s"] = predicted_round_s(out["k"], det["cores"], co)
    out["reason"] = "; ".join(notes)
    return out


def describe(d: dict) -> str:
    det = d.get("det") or {}
    pred = f", predicted round {d['pred_s']:.0f} s" if d.get("pred_s") is not None else ""
    rule = f", rule K={d['rule_k']}" if d.get("rule_k") is not None else ""
    bang = "!! " if "!!" in (d.get("reason") or "") else ""
    return (f"{bang}[ADAPTIVE_K] mode={d.get('mode')} K40={d.get('k40')}({d.get('k40_src')}) "
            f"cores={det.get('cores')} source={det.get('quota_src')} (quota={det.get('quota')}, "
            f"affinity={det.get('affinity')}, cpu_count={det.get('cpu_count')}) -> K={d['k']} (src={d['src']}{rule}, "
            f"configured K={d['fixed']}{pred}) coeffs[{d.get('coeffs_src')}]={d.get('coeffs')}"
            f"{'; ' + d['reason'] if d.get('reason') else ''}")


def apply(settings, diag: dict | None = None, env=None, root: str = CGROUP_ROOT, affinity=_affinity,
          cpu_count=os.cpu_count, log=print) -> dict:
    """Start-up hook: set settings.actors.coder.ensemble_size, record k for the miner-diag header, log one line."""
    coder = settings.actors.coder
    d = decide(int(coder.ensemble_size), env=env, root=root, affinity=affinity, cpu_count=cpu_count)
    coder.ensemble_size = int(d["k"])
    global LAST_DECISION
    LAST_DECISION = d
    if diag is not None:
        det = d.get("det") or {}
        diag["k"] = d["k"]
        diag["k_src"] = f"{d['src']}:{det.get('cores') if det.get('cores') is not None else '-'}"
    log(describe(d))
    return d
