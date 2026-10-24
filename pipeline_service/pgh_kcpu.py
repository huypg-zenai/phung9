"""pgh_kcpu.py - [opt_r5 salvage] PGH_K_CPU_AWARE: keep the CPU renderer from becoming the batch's bottleneck.

Why: r56 pgh5 audit (opt_r5/pgh5_audit.md). On a 68-core-quota pod at fixed K80 the renderer managed ~0.77 drafts/s
against ~2 drafts/s from the coder; the render backlog grew all batch and the 2,700 s task deadline killed the last 8
prompts of batch 1. The batch-boundary time guard (pgh_timeguard.py) cannot act inside batch 1.

Two parts (switch PGH_K_CPU_AWARE, read once at import = container start):
  1  / on    prior + brake, both applied
  prior      only the prior;  brake  only the brake
  shadow     both decide and log exactly what they would do; nothing changes (output = switch off)
  0 / off / "" / anything else  -> off (anything unknown also logs one '!!' line at the first batch)
PGH_KCPU_PARAMS = JSON object overriding any DEFAULT_PARAMS key (bad -> all defaults + a '!!' note).

PRIOR (before every batch, pure function of the detected CPU quota, so the same pod size always gets the same K):
  cores Q   = adaptive_k's effective cores (min of cgroup quota, affinity, cpu_count; read once at start-up)
  sws       = sws-llvm really installed in a Chrome dir (libvk_swiftshader.so differs from the .bundled copy) and
              PGH_SWS != 0; anything unknown -> Chrome's lib (the slower, conservative cost)
  rate(Q)   = min(max(0, usable_frac*Q - non_render_cores) / core_s, rate_sat)       renders per second
  T_coder   = coder_start_s + K_cfg * coder_s_per_k                                  the batch's coder time at K_cfg
  K_prior   = clamp(floor(T_coder * rate / (prompts * render_share)), k_min, K_cfg)
  i.e. the largest K whose render work fits in the time the coder needs anyway at the configured K. Never raises K.
  Coefficients (seconds, cores): core_s = pod core-seconds per rendered draft, calibrated on the r56 40.8-core pods
  (Chrome lib: backlogged plateau ~1.2 renders/s at 0.95*40.8-7 = 31.8 usable cores -> 26.5, rounded 27); sws =
  27 x 0.45 (kernel-optim gate: 6.10 / 13.56 core-s/draft) x 1.17 (UtevoLux vs AK render cost) = 14. rate_sat = the
  16-slot ceiling (Chrome lib: 1.2-1.4/s backlogged on 40.8-95.2 pods; sws: 2.3/s = RESULT.md's pessimistic 16-slot
  capacity). coder_start_s / coder_s_per_k from opt_r5/timing/RESULT.md (K72 -> coder end ~1,340 s).
BRAKE (within a batch, measured; the only part that can save a slow pod's batch 1):
  At every coder launch (the moment draft k of a photo gets its coder slot) with k >= ceil(keep_frac * K_photo):
    t        = seconds since batch start;  rendered = renders finished this batch (STATS); rate = rendered / seconds
               since the first render
    work     = drafts waiting for / in a render slot + render_share * (drafts waiting for / in a coder slot)
    proj_end = t + work / rate
    brake ON iff t >= t_min_s, rendered >= min_renders, (now - first render) >= min_window_s and proj_end > target_s.
  ON -> that draft is not generated (drop_reason "coder:Braked" = a walkover, like a coder deadline). Every photo keeps
  its first ceil(keep_frac*K) drafts (the same prefix property as the time guard's cut). It re-evaluates at every launch,
  so it stops braking once the projection is back under target_s. Why "k >= K/2" and not "photos that already have
  K/2 usable drafts": the coder semaphore is FIFO in photo order, so when a photo's draft k launches its earlier drafts
  are still being generated (usable ~0): the literal rule would never fire (README).
Logs: '[KCPU] ...' one line per batch (prior decision); '[KCPU_BRAKE] on|off ...' on every state change (rate-limited
to one per 20 s), '[KCPU_BRAKE] braked <stem> ...' once per braked photo, and a batch summary at batch end.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

DEFAULT_PARAMS: dict = {
    "usable_frac": 0.95, "non_render_cores": 7.0, "core_s_sws": 14.0, "core_s_chrome": 27.0,
    "rate_sat_sws": 2.3, "rate_sat_chrome": 1.4, "render_share": 0.93, "prompts": 32,
    "coder_start_s": 59.0, "coder_s_per_k": 17.8, "k_min": 8,
    "target_s": 2100.0, "t_min_s": 240.0, "keep_frac": 0.5, "min_renders": 48, "min_window_s": 90.0,
}
INT_KEYS = ("prompts", "k_min", "min_renders")
MODES = ("on", "prior", "brake", "shadow", "off")


def parse_mode(raw) -> tuple[str, str]:
    v = str(raw if raw is not None else "").strip().lower()
    if v in ("1", "on", "true", "live"):
        return "on", ""
    if v in ("prior", "brake", "shadow"):
        return v, ""
    if v in ("0", "off", "", "false"):
        return "off", ""
    return "off", f"!! PGH_K_CPU_AWARE={raw!r} is not 1/prior/brake/shadow/0 -> off"


def normalize_params(raw) -> dict:
    p = dict(DEFAULT_PARAMS)
    if isinstance(raw, str):
        raw = json.loads(raw) if raw.strip() else {}
    if not isinstance(raw, dict):
        raise ValueError("PGH_KCPU_PARAMS must be a JSON object")
    unknown = sorted(set(raw) - set(DEFAULT_PARAMS))
    if unknown:
        raise ValueError(f"unknown key(s) {unknown}")
    for k, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"{k}={v!r} must be a finite number >= 0")
        p[k] = int(v) if k in INT_KEYS else float(v)
    if not 0 < p["keep_frac"] <= 1 or p["prompts"] < 1 or p["render_share"] <= 0 or p["core_s_sws"] <= 0 \
            or p["core_s_chrome"] <= 0:
        raise ValueError("keep_frac in (0,1], prompts >= 1, render_share > 0, core_s_* > 0")
    return p


def sws_active(cache_dir: str | None = None, env=None) -> bool | None:
    """True = sws-llvm is the lib Chrome loads (sws_mode.sh swapped it in), False = Chrome's own lib, None = unknown."""
    env = os.environ if env is None else env
    if str(env.get("PGH_SWS", "1") or "1").strip() == "0":
        return False
    c = cache_dir or env.get("PUPPETEER_CACHE_DIR") or "/opt/puppeteer-cache"
    try:
        dirs = [d for d in glob.glob(os.path.join(c, "chrome", "*", "chrome-linux64"))
                if os.path.isfile(os.path.join(d, "libvk_swiftshader.so.bundled"))]
        if not dirs:
            return None
        return any(os.path.getsize(os.path.join(d, "libvk_swiftshader.so"))
                   != os.path.getsize(os.path.join(d, "libvk_swiftshader.so.bundled")) for d in dirs)
    except OSError:
        return None


def detect_cores() -> float | None:
    """adaptive_k's effective cores (read once at start-up; detected here if adaptive K did not run)."""
    try:
        try:
            from modules.metrics import adaptive_k as ak
        except ImportError:
            import adaptive_k as ak  # tests
        det = (getattr(ak, "LAST_DECISION", None) or {}).get("det") or ak.detect()
        return float(det["cores"]) if det.get("ok") and det.get("cores") else None
    except Exception:
        return None


def render_rate(cores: float, sws: bool, p: dict) -> float:
    usable = max(0.0, p["usable_frac"] * cores - p["non_render_cores"])
    core_s, sat = (p["core_s_sws"], p["rate_sat_sws"]) if sws else (p["core_s_chrome"], p["rate_sat_chrome"])
    return min(usable / core_s, sat)


def prior_k(k_cfg: int, cores: float | None, sws: bool | None, p: dict, n_tasks: int | None = None) -> tuple[int, dict]:
    """Pure: K for this batch from the CPU quota. cores None (detection failed) -> K_cfg unchanged."""
    info = dict(cores=cores, sws=sws, k_cfg=k_cfg)
    if cores is None or k_cfg <= 1:
        return k_cfg, dict(info, why="cores unknown" if cores is None else "K<=1")
    rate = render_rate(cores, bool(sws), p)
    n = n_tasks or p["prompts"]
    t_coder = p["coder_start_s"] + k_cfg * p["coder_s_per_k"]
    k_fit = math.floor(t_coder * rate / (n * p["render_share"]) + 1e-9)
    k = int(max(min(p["k_min"], k_cfg), min(k_cfg, k_fit)))
    return k, dict(info, rate=round(rate, 3), t_coder=round(t_coder), k_fit=k_fit,
                   render_s_at_k=round(k * n * p["render_share"] / rate) if rate > 0 else None)


def _sem_waiting(sem) -> int:
    w = getattr(sem, "_waiters", None)
    return sum(1 for f in w if not f.done()) if w else 0


class Braked(Exception):
    """Raised inside the coder's stage_guard: this draft is not generated (walkover 'coder:Braked')."""


class KCpu:
    def __init__(self, mode: str = "off", params: dict | None = None, clock=time.monotonic, rendered=None, note=""):
        self.mode, self.p, self.clock, self.note = mode, dict(params or DEFAULT_PARAMS), clock, note
        self._rendered = rendered          # callable -> renders finished since start-up (STATS by default)
        self._slots: dict[int, int] = {}   # id(semaphore) -> its size (captured idle at the first photo)
        self.k_batch = None
        self.batch = 0
        self.t0 = None
        self._reset_batch()

    # ------------------------------------------------------------------ switches
    @property
    def prior_on(self) -> bool:
        return self.mode in ("on", "prior", "shadow")

    @property
    def brake_on(self) -> bool:
        return self.mode in ("on", "brake", "shadow")

    @property
    def live(self) -> bool:
        return self.mode != "shadow"

    def _reset_batch(self):
        self.r0, self.t_first, self.state, self.last_log = None, None, False, -1e9
        self.braked, self.photos, self.on_since, self.on_s, self.last_eval = 0, {}, None, 0.0, None

    def _count_rendered(self) -> int:
        if self._rendered is not None:
            return int(self._rendered())
        from pipeline.batch_stats import STATS
        return int(STATS.render["n"])

    # ------------------------------------------------------------------ batch hooks
    def batch_begin(self, k_cfg: int, n_tasks: int | None = None, log=print, cores=None, sws=None) -> int:
        """Called before every batch (before the time guard). Returns the start K the batch may use."""
        self.batch += 1
        self._reset_batch()
        self.t0 = self.clock()
        try:
            self.r0 = self._count_rendered()
        except Exception:
            self.r0 = None
        if self.note and self.batch == 1:
            log(f"[KCPU] {self.note}")
        if not self.prior_on:
            self.k_batch = None
            return k_cfg
        cores = detect_cores() if cores is None else cores
        sws = sws_active() if sws is None else sws
        k, info = prior_k(k_cfg, cores, sws, self.p, n_tasks)
        self.k_batch = k
        what = "no cut" if k >= k_cfg else f"CUT K {k_cfg} -> {k}"
        if k < k_cfg and not self.live:
            what += f" (shadow: NOT applied, batch runs K {k_cfg})"
        log(f"[KCPU] mode={self.mode} batch={self.batch} {' '.join(f'{a}={b}' for a, b in info.items())} -> {what}")
        return k if self.live else k_cfg

    def batch_end(self, log=print) -> None:
        if self.t0 is None:
            return
        if self.brake_on:
            if self.on_since is not None:
                self.on_s += self.clock() - self.on_since
            log(f"[KCPU_BRAKE] batch={self.batch} summary: braked_drafts={self.braked} photos={len(self.photos)} "
                f"on_s={self.on_s:.0f} mode={self.mode}{' (shadow: nothing braked)' if not self.live else ''}")
        self.t0 = None

    def cap_k(self, k: int) -> int:
        if self.prior_on and self.live and self.k_batch is not None:
            return max(1, min(k, self.k_batch))
        return k

    def plan(self, factory) -> None:
        """First photo of the run: remember the semaphore sizes (they are idle at the first batch's start)."""
        for s in (getattr(factory, "sem_renderer", None), getattr(factory, "sem_coder", None)):
            if s is not None and id(s) not in self._slots:
                self._slots[id(s)] = int(getattr(s, "_value", 0))

    # ------------------------------------------------------------------ brake
    def _in_use(self, sem) -> int:
        size = max(self._slots.get(id(sem), 0), int(getattr(sem, "_value", 0)))
        return max(0, size - int(getattr(sem, "_value", 0))) + _sem_waiting(sem)

    def evaluate(self, factory, log=print) -> bool:
        """Brake state now (also logs transitions). Pure arithmetic on counters: microseconds."""
        now = self.clock()
        if self.t0 is None or not self.brake_on:
            return False
        t = now - self.t0
        try:
            rendered = self._count_rendered() - (self.r0 or 0)
        except Exception:
            return False
        if rendered > 0 and self.t_first is None:
            self.t_first = now
        p = self.p
        on, proj, rate, work = False, None, None, None
        if t >= p["t_min_s"] and rendered >= p["min_renders"] and self.t_first is not None \
                and now - self.t_first >= p["min_window_s"]:
            rate = rendered / max(now - self.t_first, 1e-6)
            work = self._in_use(factory.sem_renderer) + p["render_share"] * self._in_use(factory.sem_coder)
            proj = t + work / rate
            on = proj > p["target_s"]
        self.last_eval = dict(t=t, rendered=rendered, rate=rate, work=work, proj=proj, on=on)
        if on != self.state:
            if on:
                self.on_since = now
            elif self.on_since is not None:
                self.on_s += now - self.on_since
                self.on_since = None
            self.state = on
            if now - self.last_log >= 20.0 or on:
                self.last_log = now
                log(f"[KCPU_BRAKE] {'on' if on else 'off'} batch={self.batch} t={t:.0f}s rendered={rendered} "
                    f"rate={(rate or 0):.2f}/s work={(work or 0):.0f} proj_end={(proj or 0):.0f}s "
                    f"target={p['target_s']:.0f}s{' (shadow)' if not self.live else ''}")
        return on

    def brake(self, factory, k: int, log=print) -> bool:
        """At draft k's coder launch: True = do not generate it (live mode only). Never raises: it runs inside the
        coder's stage_guard, where an exception would drop the draft (R96: any error -> False, one '!!' line)."""
        try:
            return self._brake(factory, k, log=log)
        except Exception as exc:
            if not getattr(self, "_brake_err", False):
                self._brake_err = True
                try:
                    log(f"[KCPU_BRAKE] !! brake check failed, draft generated: {type(exc).__name__}: {exc}")
                except Exception:
                    pass
            return False

    def _brake(self, factory, k: int, log=print) -> bool:
        if not self.brake_on or self.t0 is None:
            return False
        k_photo = len(getattr(factory, "_coder_tasks", None) or {}) or 1
        if k < math.ceil(self.p["keep_frac"] * k_photo):
            return False
        if not self.evaluate(factory, log=log):
            return False
        stem = factory.task.stem
        first = stem not in self.photos
        self.photos[stem] = self.photos.get(stem, 0) + 1
        self.braked += 1
        if first:
            e = self.last_eval or {}
            log(f"[KCPU_BRAKE] braked {stem} from k{k} (K={k_photo}, keeps k<{math.ceil(self.p['keep_frac'] * k_photo)})"
                f" t={e.get('t', 0):.0f}s proj_end={(e.get('proj') or 0):.0f}s{' (shadow: generated anyway)' if not self.live else ''}")
        return self.live


def _from_env() -> KCpu:
    mode, note = parse_mode(os.environ.get("PGH_K_CPU_AWARE", "0"))
    try:
        p = normalize_params(os.environ.get("PGH_KCPU_PARAMS", "") or "")
    except (ValueError, TypeError) as exc:
        p = dict(DEFAULT_PARAMS)
        note = (note + "; " if note else "") + f"!! bad PGH_KCPU_PARAMS ({exc}) -> defaults"
    _gs = __import__("sys").modules.get("pgh_gsplit")   # [gsplit] config.settings (imported first) ran apply_cfg
    if _gs is not None and _gs.on():
        p, _gn = _gs.scale_kcpu(p)
        note = (note + "; " if note else "") + _gn
    return KCpu(mode, p, note=note)


KCPU = _from_env()
ON = KCPU.mode != "off"
