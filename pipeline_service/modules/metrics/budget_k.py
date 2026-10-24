"""[R82 kbudget] PGH_K_MODE=budget: give every photo about the same coder TOKENS instead of the same number of drafts.

Why: the coder is the bottleneck and the K-curve slope (b ~ +0.17 per doubling of K) does not depend on how long a photo's
drafts are, so at a fixed token budget sum_i K_i*c_i the score sum_i b*log2(K_i) is maximised by K_i ~ 1/c_i (c_i = the
photo's mean tokens per draft). Draft length is unknown before generation, so the rule is ONLINE (no oracle):

  plan(K)     K = the batch's K (PGH_K40 after the time guard). Inactive (-> K drafts, exactly fixed mode) when this is the
              first batch with no reference yet (b0="fixed"), when fewer than min_ref photos are in the reference, or when
              K <= floor. Active: start K_lo = max(floor, round(lo*K)) drafts at once (<= K).
  decide()    once drafts k = 0..n0-1 of the photo are back (k order = seed order, so an unbiased sample; NOT the first n0
              to finish, which would be the shortest): c_hat = (sum of their JS bytes + shrink*c_ref) / (n_valid + shrink)
              K_i = clamp(round(lam * c_hat^-p), K_lo, K_hi),  K_hi = round(hi*K)
              lam solved on the reference WITH the clamps: mean_ref[clamp(lam*c^-p)*c] = K*mean_ref[c]
              ref = per-photo mean JS bytes of the photos of EARLIER batches (the pipeline's own drafts) -> no cross-photo
              barrier and no order bias; sum_i K_i*c_i ~= K*sum_i c_i when the batch looks like the reference.
              Extra drafts above K only while the photo has >= late_frac of its candidate deadline left.
  observe()   at the photo's end: its mean JS bytes join the reference at batch_end().
JS bytes are the cost proxy (advisor A1 ES-prompt coder drafts: corr 0.986 with completion tokens, 2.59 bytes/token).

PGH_KB_PARAMS = JSON overriding DEFAULT_PARAMS (bad -> defaults + a '!!' note). Pure functions + one module ALLOC; the
replay (opt_r2/kbudget/replay/replay.py) imports this file, so what was replayed is what ships.
"""
from __future__ import annotations

import json
import math
import os

DEFAULT_PARAMS: dict = {"p": 1.0, "lo": 0.5, "hi": 1.5, "n0": 8, "shrink": 4, "floor": 16, "b0": "fixed",
                        "prior": 0, "min_ref": 16, "late_frac": 0.5, "k_max": 256}


def normalize_params(raw) -> dict:
    p = dict(DEFAULT_PARAMS)
    if raw in (None, ""):
        return p
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"PGH_KB_PARAMS must be a JSON object, got {type(raw).__name__}")
    unknown = set(raw) - set(DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown PGH_KB_PARAMS keys {sorted(unknown)}")
    p.update(raw)
    for k in ("p", "lo", "hi", "late_frac", "prior"):
        p[k] = float(p[k])
    for k in ("n0", "shrink", "floor", "min_ref", "k_max"):
        p[k] = int(p[k])
    if not (0.0 <= p["p"] <= 2.0):
        raise ValueError(f"p={p['p']} not in [0, 2]")
    if not (0.0 < p["lo"] <= 1.0 <= p["hi"] <= 4.0):
        raise ValueError(f"need 0 < lo <= 1 <= hi <= 4, got lo={p['lo']} hi={p['hi']}")
    if p["n0"] < 1 or p["floor"] < 1 or p["shrink"] < 0 or p["min_ref"] < 1:
        raise ValueError("n0, floor, min_ref must be >= 1 and shrink >= 0")
    if p["b0"] not in ("fixed", "prior"):
        raise ValueError(f"b0={p['b0']!r} not fixed|prior")
    return p


def bounds(K: int, p: dict) -> tuple[int, int]:
    """(K_lo, K_hi) for a batch K. K_lo never above K and never below floor (unless K itself is below floor)."""
    k_lo = min(K, max(p["floor"], int(round(p["lo"] * K))))
    k_hi = max(K, min(p["k_max"], int(round(p["hi"] * K))))
    return k_lo, k_hi


def solve_lambda(costs: list[float], K: int, p: dict) -> float:
    """lam with sum_i clamp(lam*c_i^-p, K_lo, K_hi)*c_i = K*sum_i c_i (bisection; continuous, before rounding)."""
    k_lo, k_hi = bounds(K, p)
    pw = p["p"]
    target = K * sum(costs)

    def tot(lam):
        return sum(min(k_hi, max(k_lo, lam * c ** (-pw))) * c for c in costs)
    a, b = 1e-12, 1.0
    while tot(b) < target and b < 1e300:
        b *= 2.0
    if k_lo == k_hi or tot(b) < target:      # degenerate (K <= floor etc.): every photo K
        return K * (sum(costs) / len(costs)) ** pw
    for _ in range(200):
        m = math.sqrt(a * b)
        a, b = (m, b) if tot(m) < target else (a, m)
    return math.sqrt(a * b)


def alloc_one(K: int, c_hat: float, lam: float, p: dict) -> int:
    k_lo, k_hi = bounds(K, p)
    return int(min(k_hi, max(k_lo, round(lam * c_hat ** (-p["p"])))))


def solve_alloc(costs: list[float], K: int, pw: float, lo: float, hi: float, floor: int = 16, k_max: int = 256) -> list[int]:
    """Joint (oracle / test) allocation: K_i = clamp(lam*c_i^-pw) with sum K_i*c_i = K*sum c_i (bisection on lam), rounded."""
    p = dict(DEFAULT_PARAMS, p=pw, lo=lo, hi=hi, floor=floor, k_max=k_max)
    lam = solve_lambda(costs, K, p)
    return [alloc_one(K, c, lam, p) for c in costs]


class BudgetAllocator:
    def __init__(self, params: dict | None = None):
        self.p = normalize_params(params)
        self.ref: list[float] = []          # per-photo mean bytes, earlier batches
        self._pending: dict[str, float] = {}
        self.batch = -1
        self.decisions: list[dict] = []
        self._lam: dict[int, float] = {}      # K -> lam for the current reference (reset when the reference grows)

    # -- batch boundaries (GenerationPipeline.run_batch) ----------------------------------------------------------
    def batch_begin(self, K: int | None = None) -> None:
        self.batch += 1
        self._pending = {}

    def batch_end(self) -> None:
        self.ref.extend(self._pending.values())
        self._pending = {}
        self._lam = {}

    # -- per photo --------------------------------------------------------------------------------------------------
    def _reference(self) -> list[float] | None:
        if len(self.ref) >= self.p["min_ref"]:
            return self.ref
        if self.p["b0"] == "prior" and self.p["prior"] > 0:
            return [self.p["prior"]]
        return None

    def plan(self, K: int) -> dict:
        ref = self._reference()
        if ref is None or K <= self.p["floor"]:
            return {"active": False, "k_start": K, "n0": 0,
                    "why": "no reference yet (first batch)" if ref is None else f"K {K} <= floor {self.p['floor']}"}
        k_lo, k_hi = bounds(K, self.p)
        return {"active": True, "k_start": k_lo, "n0": min(self.p["n0"], k_lo), "k_lo": k_lo, "k_hi": k_hi}

    def decide(self, K: int, first_costs, stem=None, remaining_frac: float | None = None) -> dict:
        """first_costs: JS bytes of drafts k = 0..n0-1 that produced code (failures left out)."""
        ref = self._reference()
        plan = self.plan(K)
        if not plan["active"]:
            return {"k": K, "why": plan["why"]}
        xs = [float(x) for x in first_costs if x and x > 0]
        if len(xs) < min(3, plan["n0"]):
            d = {"k": K, "why": f"only {len(xs)} valid first drafts"}
        else:
            c_ref = sum(ref) / len(ref)
            c_hat = (sum(xs) + self.p["shrink"] * c_ref) / (len(xs) + self.p["shrink"])
            lam = self._lam.get(K)
            if lam is None:
                lam = self._lam[K] = solve_lambda(ref, K, self.p)
            k = alloc_one(K, c_hat, lam, self.p)
            why = "ok"
            if k > K and remaining_frac is not None and remaining_frac < self.p["late_frac"]:
                k, why = K, f"late ({remaining_frac:.2f} of the candidate deadline left): no extras above K"
            d = {"k": k, "c_hat": round(c_hat, 1), "c_ref": round(c_ref, 1), "n": len(xs), "why": why}
        d.update(stem=stem, batch=self.batch, K=K, k_lo=plan["k_lo"], k_hi=plan["k_hi"])
        self.decisions.append(d)
        return d

    def observe(self, stem, costs) -> None:
        xs = [float(x) for x in costs if x and x > 0]
        if xs and stem is not None:
            self._pending[stem] = sum(xs) / len(xs)


def _from_env() -> tuple[BudgetAllocator, str]:
    note = ""
    try:
        params = normalize_params(os.environ.get("PGH_KB_PARAMS", "") or "")
    except Exception as exc:  # noqa: BLE001
        params, note = dict(DEFAULT_PARAMS), f"!! bad PGH_KB_PARAMS ({exc}) -> defaults"
    return BudgetAllocator(params), note


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get("PGH_K_MODE", "") or "").strip().lower() == "budget"


ALLOC, NOTE = _from_env()
