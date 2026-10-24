"""pgh_mix.py - ONE pod, TWO coders (R71, 21/10; ported onto pgh16 as pgh17 by R80, 22/10: + per-coder prompt and
sampling). Each prompt's K drafts are split between two coder servers; the pipeline's normal dedup + judge bracket picks
the final from the merged pool. Stdlib only at import (spawn.py and config/settings.py import it before anything else).

Switch PGH_MIX_CODERS (read at container start):
  ""/"0"/"off"/unset   OFF: every function below is the identity / not called -> behaviour == the base package (pgh16).
  "utevo:ak"           ON (pgh17 Dockerfile default): names = the `mix.name` of two llm_clients in configuration.yaml (the
                       FIRST name gets PGH_MIX_SPLIT of every prompt's drafts). Anything invalid -> one '!!' line, mix OFF.
PGH_MIX_SPLIT    fraction of the drafts for the FIRST name, 0 < f < 1 (default 0.5). Draft k goes to the first coder
                 iff ceil((k+1) f) > ceil(k f): an interleave, so EVERY prefix 0..n-1 holds ceil(n f) first-coder drafts
                 (the time guard's prefix cut keeps the ratio). Pure function of (k, f): reproducible.
PGH_MIX_WAIT_S   start-up: when one coder is up and the other is not, wait this long (default 1200 s) for it, then
                 go READY on the one coder alone (it then gets every draft = full K). Never "no output".
PGH_MIX_BARRIER_S  sgl_shim.sh: max wait (default 1800 s) for the other coder's patch chain before SGLang starts.
PGH_MIX_PROMPTS  [pgh17] coder prompt per mix name, "name=variant,..." (variants of modules/scene_coder/coder_prompt.py:
                 es | pgh7 | ours | oursneg). A mix name not listed sends the global PGH_CODER_PROMPT prompt (= pgh16's).
                 Unknown name / variant -> '!!' + mix OFF. Unset = DEFAULT_PROMPTS = the pgh17 image's "utevo=es,ak=pgh7".
`mix.sampling` (optional, per client, configuration.yaml) [pgh17]: overrides of that coder's sampling, keys SAMPLING_KEYS
                 (ensemble_temperature replaces the pipeline's per-draft temperature for that coder). None in pgh17's
                 config: pgh7's AmberKite settings == pgh16's UtevoLux settings (actors.coder, test-checked).

Config (configuration.yaml): a client joins the mix with a `mix:` block (ignored by pydantic and by spawn.py when OFF):
  coder-instance:   {..., mix: {name: ak, gpu_ids: "0", data_parallel_size: 1}}   # = pgh7's coder (GPUs 0,1 DP2) when OFF
  coder-instance-b: {enabled: false, ..., mix: {name: utevo, gpu_ids: "1", data_parallel_size: 1}}
ON: apply_cfg() enables every mix client, sets its gpu_ids / data_parallel_size from its mix block and inserts
"--mix-slot <i>/<n>" after "--sgl" (sgl_shim.sh: port/log offsets + patch barrier). actors.coder.client must be one of
the mix clients (it keeps the coder probe, time guard and K rule); the others get a SceneCoderAgent each.
"""
from __future__ import annotations

import copy
import math
import os
import re
import time
from fractions import Fraction

_OFF = ("", "0", "off", "false", "no")
DEFAULT_SPLIT = 0.5
DEFAULT_WAIT_S = 1200.0
DEAD_S = 60.0   # a coder that failed a request (connection/5xx) gets no new drafts for this long
PROMPT_VARIANTS = ("es", "pgh7", "ours", "oursneg")   # [pgh17] = modules/scene_coder/coder_prompt.VARIANTS (test-checked)
DEFAULT_PROMPTS = {"utevo": "es", "ak": "pgh7"}   # [pgh17] PGH_MIX_PROMPTS unset (explicit "" = every coder global)
SAMPLING_KEYS = {"temperature": float, "top_p": float, "top_k": int, "min_p": float, "presence_penalty": float,
                 "repetition_penalty": float, "max_tokens": int, "ensemble_temperature": float}


class MixSpec:
    def __init__(self, names, clients, split, wait_s, primary, prompts=None, sampling=None):
        self.names = list(names)          # mix names in PGH_MIX_CODERS order (slot i = position i)
        self.clients = list(clients)      # llm_clients names, same order
        self.split = split                # Fraction: share of the FIRST name
        self.wait_s = wait_s
        self.primary = primary            # actors.coder.client
        self.prompts = list(prompts or [None] * len(self.names))     # [pgh17] prompt variant per slot (None = global)
        self.sampling = list(sampling or [{} for _ in self.names])  # [pgh17] sampling overrides per slot
        self.notes: list[str] = []

    def describe(self) -> str:
        return (f"[MIX] ON coders={list(zip(self.names, self.clients))} split={float(self.split):.4g} to "
                f"{self.names[0]} (rest to {self.names[1]}) wait_s={self.wait_s:g} primary={self.primary} "
                f"prompts={dict(zip(self.names, [p or 'PGH_CODER_PROMPT' for p in self.prompts]))}"
                f"{' sampling=' + str(dict(zip(self.names, self.sampling))) if any(self.sampling) else ''}")


def _parse_prompts(raw, names):
    """'name=variant,...' -> ([variant|None per name], '') or (None, why). raw None (env unset) -> DEFAULT_PROMPTS."""
    if raw is None:
        return [DEFAULT_PROMPTS.get(n) for n in names], ""
    s = str(raw).strip()
    out = {}
    for item in [x.strip() for x in s.split(",") if x.strip()]:
        n, eq, v = item.partition("=")
        n, v = n.strip(), v.strip().lower()
        if not eq or n not in names:
            return None, f"PGH_MIX_PROMPTS item {item!r}: want <mix name>=<variant>, names {names}"
        if v not in PROMPT_VARIANTS or n in out:
            return None, f"PGH_MIX_PROMPTS item {item!r}: variant must be one of {PROMPT_VARIANTS}, each name once"
        out[n] = v
    return [out.get(n) for n in names], ""


def _parse_sampling(m):
    raw = m.get("sampling") if isinstance(m, dict) else None
    if raw in (None, {}):
        return {}, ""
    if not isinstance(raw, dict):
        return None, "mix.sampling must be a mapping"
    out = {}
    for k, v in raw.items():
        if k not in SAMPLING_KEYS or isinstance(v, bool):
            return None, f"mix.sampling key {k!r} not one of {sorted(SAMPLING_KEYS)}"
        try:
            out[k] = SAMPLING_KEYS[k](v)
        except (TypeError, ValueError):
            return None, f"mix.sampling {k}={v!r} is not a number"
        if not math.isfinite(out[k]):
            return None, f"mix.sampling {k}={v!r} is not finite"
    return out, ""


def _parse_split(raw):
    s = str(raw if raw is not None else "").strip()
    if not s:
        return Fraction(DEFAULT_SPLIT).limit_denominator(1000), ""
    try:
        f = Fraction(s).limit_denominator(1000)
    except (ValueError, ZeroDivisionError):
        return None, f"!! PGH_MIX_SPLIT={raw!r} is not a number"
    if not (0 < f < 1):
        return None, f"!! PGH_MIX_SPLIT={raw!r} must be strictly between 0 and 1"
    return f, ""


def parse(cfg: dict, env=None):
    """(MixSpec | None, note). None = OFF (note '' when the switch is off, '!! ...' when it is on but invalid)."""
    env = os.environ if env is None else env
    raw = str(env.get("PGH_MIX_CODERS", "") or "").strip()
    if raw.lower() in _OFF:
        return None, ""
    names = [x.strip() for x in re.split(r"[:,]", raw) if x.strip()]
    bad = lambda why: (None, f"!! PGH_MIX_CODERS={raw!r}: {why} -> mix OFF (single coder)")
    if len(names) != 2 or len(set(names)) != 2:
        return bad("need exactly 2 different names")
    split, note = _parse_split(env.get("PGH_MIX_SPLIT", ""))
    if split is None:
        return bad(note.lstrip("! "))
    try:
        wait_s = float(env.get("PGH_MIX_WAIT_S", "") or DEFAULT_WAIT_S)
        if not math.isfinite(wait_s) or wait_s < 0:
            raise ValueError
    except ValueError:
        return bad(f"PGH_MIX_WAIT_S={env.get('PGH_MIX_WAIT_S')!r} must be a number >= 0")
    llm = (cfg or {}).get("llm_clients") or {}
    by_name: dict[str, list[str]] = {}
    for cname, spec in llm.items():
        m = spec.get("mix") if isinstance(spec, dict) else None
        if isinstance(m, dict) and m.get("name"):
            by_name.setdefault(str(m["name"]), []).append(cname)
    clients = []
    for n in names:
        found = by_name.get(n, [])
        if len(found) != 1:
            return bad(f"name {n!r} matches {len(found)} llm_clients (want 1; known: {sorted(by_name)})")
        clients.append(found[0])
    gsets = []
    for c in clients:
        m, v = llm[c]["mix"], (llm[c].get("vllm") or {})
        ids = [x.strip() for x in str(m.get("gpu_ids", "")).split(",") if x.strip()]
        if not ids:
            return bad(f"{c}: mix.gpu_ids missing")
        try:
            dp = int(m.get("data_parallel_size", 1))
            tp = int(v.get("tensor_parallel_size", 1) or 1)
        except (TypeError, ValueError):
            return bad(f"{c}: bad data_parallel_size / tensor_parallel_size")
        if dp < 1 or tp * dp != len(ids):
            return bad(f"{c}: tp {tp} x dp {dp} != {len(ids)} gpu(s) {ids}")
        if not (v.get("model") or "").strip() or not isinstance(llm[c].get("base_url"), str):
            return bad(f"{c}: vllm.model / base_url missing")
        gsets.append(set(ids))
    if gsets[0] & gsets[1]:
        return bad(f"mix clients share GPU(s) {sorted(gsets[0] & gsets[1])}")
    for cname, spec in llm.items():   # other enabled local clients with explicit GPUs must not overlap
        if cname in clients or not isinstance(spec, dict) or spec.get("enabled", True) is False:
            continue
        ids = {x.strip() for x in str((spec.get("vllm") or {}).get("gpu_ids", "") or "").split(",") if x.strip()}
        clash = ids & (gsets[0] | gsets[1])
        if clash:
            return bad(f"{cname} already uses GPU(s) {sorted(clash)}")
    ports = [llm[c]["base_url"] for c in clients]
    if ports[0] == ports[1]:
        return bad("mix clients share a base_url")
    primary = (((cfg or {}).get("actors") or {}).get("coder") or {}).get("client")
    if primary not in clients:
        return bad(f"actors.coder.client={primary!r} is not one of the mix clients {clients}")
    prompts, why = _parse_prompts(env.get("PGH_MIX_PROMPTS"), names)   # [pgh17]
    if prompts is None:
        return bad(why)
    sampling = []
    for c in clients:
        sm, why = _parse_sampling(llm[c]["mix"])
        if sm is None:
            return bad(f"{c}: {why}")
        sampling.append(sm)
    return MixSpec(names, clients, split, wait_s, primary, prompts, sampling), note


def apply_cfg(cfg: dict, env=None, log=None) -> dict:
    """configuration dict -> the dict the pod runs. OFF/invalid: the SAME object, untouched (identity)."""
    spec, note = parse(cfg, env)
    if note and log:
        log(note)
    if spec is None:
        return cfg
    out = copy.deepcopy(cfg)
    n = len(spec.clients)
    for slot, c in enumerate(spec.clients):
        cl = out["llm_clients"][c]
        m = cl["mix"]
        cl["enabled"] = True
        v = cl.setdefault("vllm", {})
        v["gpu_ids"] = str(m["gpu_ids"])
        v["data_parallel_size"] = int(m.get("data_parallel_size", 1))
        ea = _strip_slot([str(x) for x in (v.get("extra_args") or [])])
        if "--sgl" in ea:
            i = ea.index("--sgl") + 1
            ea[i:i] = ["--mix-slot", f"{slot}/{n}"]
        v["extra_args"] = ea
    if log:
        log(spec.describe())
    return out


def _strip_slot(ea: list[str]) -> list[str]:
    out, skip = [], False
    for x in ea:
        if skip:
            skip = False
            continue
        if x == "--mix-slot":
            skip = True
            continue
        out.append(x)
    return out


# ---------------------------------------------------------------- K split (pure)
def first_gets(k: int, f: Fraction) -> bool:
    """Draft k -> the FIRST coder iff ceil((k+1) f) > ceil(k f). Any prefix of n drafts holds exactly ceil(n f)."""
    return math.ceil((k + 1) * f) > math.ceil(k * f)


def assign(k: int, f: Fraction) -> int:
    return 0 if first_gets(k, f) else 1


def counts(K: int, f: Fraction) -> tuple[int, int]:
    a = math.ceil(K * f)
    return a, K - a


_K_RE = re.compile(r"#k(\d+)$")


def k_of(actor: str | None):
    m = _K_RE.search(actor or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- run-time state (set by generation_pipeline)
SPEC: MixSpec | None = None


def init_from_env_and_cfg(cfg: dict, env=None, log=None) -> MixSpec | None:
    global SPEC
    SPEC, note = parse(cfg, env)
    if note and log:
        log(note)
    return SPEC


def active() -> bool:
    return SPEC is not None


class Readiness:
    """models_ready() for a mixed pod. Non-mix clients: all RUNNING (as before). Mix coders: both RUNNING -> ready;
    none -> not ready; one -> ready after wait_s (then 'committed': later a coder going down never blocks/aborts while
    the other is RUNNING; both down = not ready, exactly like the single-coder pod)."""

    def __init__(self, spec: MixSpec, clock=time.monotonic, log=print):
        self.spec, self.clock, self.log = spec, clock, log
        self.partial_since = None
        self.committed = False
        self._last = None

    def __call__(self, llm_status: dict, running) -> bool:
        others = [s for n, s in llm_status.items() if n not in self.spec.clients]
        if not all(s == running for s in others):
            return False
        up = [c for c in self.spec.clients if llm_status.get(c) == running]
        state = tuple(up)
        if state != self._last:
            self._last = state
            self.log(f"[MIX] coders up: {up or 'none'} of {self.spec.clients}")
        if len(up) == len(self.spec.clients):
            self.committed = True
            self.partial_since = None
            return True
        if not up:
            return False
        if self.committed:
            return True
        now = self.clock()
        if self.partial_since is None:
            self.partial_since = now
            self.log(f"[MIX] only {up} up; waiting up to {self.spec.wait_s:g}s for the other coder")
        if now - self.partial_since >= self.spec.wait_s:
            self.committed = True
            self.log(f"[MIX] !! FALLBACK: going ready on {up} alone after {self.spec.wait_s:g}s "
                     f"(it gets every draft = full K)")
            return True
        return False


READY: Readiness | None = None


# ---------------------------------------------------------------- the coder router
def _is_server_failure(exc) -> bool:
    try:
        import openai
    except Exception:  # pragma: no cover
        return False
    if isinstance(exc, openai.APITimeoutError):      # a slow request already cost request_timeout_s: no retry
        return False
    return isinstance(exc, (openai.APIConnectionError, openai.InternalServerError))


class MixCoder:
    """Stands in for the pipeline's SceneCoderAgent. code(actor_override='coder#k<k>') goes to coder assign(k, f);
    a coder that is not RUNNING (health checker) or that fails with a connection/5xx error is skipped for the other
    one (its session is dropped first: a session is bound to one client). Every other attribute = the primary agent."""

    def __init__(self, agents: list, names: list[str], clients: list[str], split: Fraction, status_fn,
                 session_store, running, log=print, warn=print, temps=None):
        self.__dict__.update(_agents=list(agents), _names=list(names), _clients=list(clients), _split=split,
                             _status_fn=status_fn, _store=session_store, _running=running, _log=log, _warn=warn,
                             _dead={}, sent=[0] * len(agents), rerouted=0, _clock=time.monotonic,
                             _temps=list(temps or [None] * len(agents)))   # [pgh17] per-coder ensemble temperature

    # seed etc.: run_batch sets coder.seed; keep every agent in step
    def __setattr__(self, name, value):
        if name in ("sent", "rerouted"):
            self.__dict__[name] = value
            return
        for a in self._agents:
            setattr(a, name, value)

    def __getattr__(self, name):
        return getattr(self.__dict__["_agents"][0], name)

    def _up(self, i: int) -> bool:
        c = self._clients[i]
        if c in self._dead:   # failed a request: out for DEAD_S, then the health checker decides again
            if self._clock() - self._dead[c] < DEAD_S:
                return False
            del self._dead[c]
            self._log(f"[MIX] {c}: re-admitted after {DEAD_S:g}s (health checker decides)")
        st = self._status_fn() or {}
        if not st:
            return True
        return st.get(self._clients[i]) == self._running

    def route(self, k):
        first = 0 if k is None else assign(k, self._split)
        order = [first, 1 - first]
        live = [i for i in order if self._up(i)]
        return live or order

    async def code(self, task_id, osd, image_bytes=None, image_mime="image/jpeg", *, actor_override=None,
                   seed_override=None, temperature_override=None):
        k = k_of(actor_override)
        order = self.route(k)
        last = None
        for j, i in enumerate(order[:2]):
            if j > 0:
                self._store.evict_actor(task_id, actor_override or "coder")
                self.__dict__["rerouted"] += 1
            try:
                t_i = temperature_override
                if t_i is not None and self._temps[i] is not None:
                    t_i = self._temps[i]   # [pgh17] mix.sampling.ensemble_temperature of this coder
                out = await self._agents[i].code(task_id, osd, image_bytes=image_bytes, image_mime=image_mime,
                                                 actor_override=actor_override, seed_override=seed_override,
                                                 temperature_override=t_i)
                self.sent[i] += 1
                return out
            except Exception as exc:
                if not _is_server_failure(exc):
                    raise
                last = exc
                if self._clients[i] not in self._dead:
                    self._dead[self._clients[i]] = self._clock()
                    self._warn(f"[MIX] !! {self._names[i]} ({self._clients[i]}) failed ({type(exc).__name__}) on "
                               f"{task_id} {actor_override}: its drafts go to the other coder")
        raise last


def build_router(pipeline, settings, clients: dict, agent_cls, state, running, log=print, warn=print):
    """Replace pipeline.coder by a MixCoder (called once after build_pipeline when SPEC is set)."""
    spec = SPEC
    primary = pipeline.coder
    agents, own = [], []   # own[i]: agents[i] really is coder i (not the primary standing in for a client-less coder)
    for c in spec.clients:
        if c == spec.primary:
            agents.append(primary)
            own.append(True)
            continue
        if c not in clients:
            warn(f"[MIX] !! client {c} has no AsyncOpenAI client -> its drafts go to the primary")
            agents.append(primary)
            own.append(False)
            continue
        s = settings.actors.coder.model_copy(update={"client": c, "model": settings.llm_clients[c].vllm.model})
        agents.append(agent_cls(clients[c], session_store=pipeline.session_store, settings=s))
        own.append(True)
    # [pgh17] per-coder prompt (system + image-only user text) and sampling, set on each agent BEFORE wrapping (the
    # router's setattr fans out to every agent). A stand-in primary keeps its own prompt/sampling.
    from modules.scene_coder import prompts as _P
    from modules.scene_coder.coder_prompt import select_coder_prompt
    import hashlib
    temps, plog = [], []
    for i, a in enumerate(agents):
        v = spec.prompts[i] if own[i] else None
        if v is not None:
            v2, sys_p, user_p = select_coder_prompt(_P.PGH7_CODER_SYSTEM_PROMPT, _P.PGH7_CODER_USER_TEMPLATE_IMAGE_ONLY,
                                                    env={"PGH_CODER_PROMPT": v})
            a.pgh_system_prompt, a.pgh_user_image_only, a.pgh_prompt_variant = sys_p, user_p, v2
        sys_now = getattr(a, "pgh_system_prompt", _P.CODER_SYSTEM_PROMPT)
        plog.append(f"{spec.names[i]}={getattr(a, 'pgh_prompt_variant', None) or _P.CODER_PROMPT_VARIANT}"
                    f"(sys md5 {hashlib.md5(sys_now.encode('utf-8')).hexdigest()[:8]})")
        sm = spec.sampling[i] if own[i] else {}
        for k, val in sm.items():
            if k != "ensemble_temperature":
                setattr(a, k, val)
        temps.append(sm.get("ensemble_temperature"))
    router = MixCoder(agents, spec.names, spec.clients, spec.split, lambda: state.llm_status,
                      pipeline.session_store, running, log=log, warn=warn, temps=temps)
    pipeline.coder = router
    a, b = counts(settings.actors.coder.ensemble_size, spec.split)
    log(f"[MIX] router: K={settings.actors.coder.ensemble_size} -> {spec.names[0]} {a} + {spec.names[1]} {b} "
        f"(interleaved; models {[getattr(x, 'model', '?') for x in agents]}; prompts {' '.join(plog)})")
    return router


# ---------------------------------------------------------------- burst: one controller, several coder proxies
class FanoutHTTP:
    """Duck-types the httpx.AsyncClient calls BurstController makes with self.proxy == '' (paths only).
    to_judge: all proxies or none (a partial success is reverted with to_coder). to_coder: every proxy; a proxy that
    refuses the connection (its shim is gone, nothing to resume) counts as ok. state: 'judge' if any proxy is."""

    def __init__(self, proxies: list[str], http):
        self.proxies = [p.rstrip("/") for p in proxies]
        self.http = http

    class _R:
        def __init__(self, js):
            self._js = js

        def json(self):
            return self._js

    async def _one(self, method, base, path, **kw):
        try:
            r = await getattr(self.http, method)(base + path, **kw)
            try:
                js = r.json()
            except Exception:  # noqa: BLE001
                js = {"ok": False, "error": f"HTTP {getattr(r, 'status_code', '?')} (no json)"}
            return js, None
        except Exception as exc:  # noqa: BLE001
            return None, exc

    async def post(self, path, **kw):
        import asyncio
        res = await asyncio.gather(*(self._one("post", p, path, **kw) for p in self.proxies))
        if path.endswith("/burst/to_judge"):
            ok = [p for p, (js, e) in zip(self.proxies, res) if js and js.get("ok")]
            if len(ok) == len(self.proxies):
                return self._R({"ok": True, "legs": [js for js, _ in res]})
            for p in ok:   # undo the legs that did swap
                await self._one("post", p, "/burst/to_coder")
            return self._R({"ok": False, "error": "to_judge failed on " + ", ".join(
                f"{p}: {(js or {}).get('error') or repr(e)}" for p, (js, e) in zip(self.proxies, res)
                if not (js and js.get("ok"))), "reverted": ok})
        if path.endswith("/burst/to_coder"):
            import httpx
            good = all((js and js.get("ok")) or isinstance(e, httpx.ConnectError) for js, e in res)
            return self._R({"ok": bool(good), "legs": [js if js else repr(e) for js, e in res]})
        return self._R({"ok": False, "error": f"FanoutHTTP: unknown path {path}"})

    async def get(self, path, **kw):
        import asyncio
        res = await asyncio.gather(*(self._one("get", p, path, **kw) for p in self.proxies))
        st = "judge" if any(js and js.get("state") == "judge" for js, _ in res) else "coder"
        return self._R({"state": st, "legs": [js if js else repr(e) for js, e in res]})


# ---------------------------------------------------------------- spawn.py helpers (pid files for the shim barrier)
def mix_dir(env=None) -> str:
    env = os.environ if env is None else env
    return str(env.get("PGH_MIX_DIR", "") or "/tmp/pgh_mix")


def spawn_clear(env=None) -> None:
    d = mix_dir(env)
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(d):
        if f.startswith(("pid_", "patched_")):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass


def spawn_note(spec: MixSpec, client: str, pid: int, env=None) -> None:
    if client in spec.clients:
        with open(os.path.join(mix_dir(env), f"pid_{spec.clients.index(client)}"), "w") as f:
            f.write(str(pid))


def proxy_urls() -> list[str]:
    """coder lb_proxy roots (base_url without /v1) of the mix clients, in slot order"""
    try:
        from config.settings import settings
        return [settings.llm_clients[c].base_url.rstrip("/").rsplit("/v1", 1)[0] for c in SPEC.clients]
    except Exception:  # noqa: BLE001
        return []
