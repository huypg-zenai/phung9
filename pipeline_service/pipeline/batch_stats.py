"""Per-batch three-axis accounting (coder / judge / render) for the `// miner-diag:` header.

The audit's regenerated modules are published, so the header of every module is the only channel through which an
audit pod reports back what happened. The preflight snapshot (GPU/CPU/network before the models load) tells us
whether a pod *started* slow; this module tells us *where a batch's time went*: LLM calls per client (count, errors,
tokens, latency percentiles, first/last completion), judge duels (compare time, queue wait), renders (latency, failures,
sidecar restarts), the judge tail after the coder's last output, the vLLM servers' own counters (token deltas,
preemptions, prefix-cache hit rate) and the container's cgroup CPU usage/throttling over the batch.

Measurement only: every hook is best-effort and never raises into the pipeline. One process-global `STATS`
instance; `MinerState.reset_for_batch()` resets it and `mark_complete()` closes it.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any

_VLLM_COUNTERS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:request_success_total",
)
_VLLM_GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
_METRIC_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|Inf)")


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, round((len(s) - 1) * p)))]


def _fmt(v: float | None, nd: int = 1) -> str:
    if v is None:
        return "?"
    return f"{v:.{nd}f}" if nd else f"{v:.0f}"


def _ktok(v: float | None) -> str:
    """Token counts as 12K / 2.1M so the header stays short."""
    if v is None:
        return "?"
    v = float(v)
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


def parse_vllm_metrics(text: str) -> dict[str, float]:
    """Sum every labelled series of the whitelisted vLLM metrics (a DP=2 server exposes one series per rank)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name not in _VLLM_COUNTERS and name not in _VLLM_GAUGES:
            continue
        try:
            out[name] = out.get(name, 0.0) + float(m.group(3))
        except ValueError:
            continue
    return out


def read_cgroup_cpu() -> dict[str, float] | None:
    """cgroup v2 cpu.stat (usage_usec, nr_throttled, throttled_usec) or the v1 equivalents, all in microseconds."""
    try:
        p = Path("/sys/fs/cgroup/cpu.stat")
        if p.exists():
            d = {}
            for line in p.read_text().splitlines():
                k, _, v = line.partition(" ")
                if v.strip().lstrip("-").isdigit():
                    d[k] = float(v)
            if "usage_usec" in d:
                return {"usage_usec": d["usage_usec"], "nr_throttled": d.get("nr_throttled", 0.0),
                        "throttled_usec": d.get("throttled_usec", 0.0), "nr_periods": d.get("nr_periods", 0.0)}
        st = Path("/sys/fs/cgroup/cpu/cpu.stat")
        if st.exists():
            d = {}
            for line in st.read_text().splitlines():
                k, _, v = line.partition(" ")
                if v.strip().isdigit():
                    d[k] = float(v)
            usage = None
            for cand in ("/sys/fs/cgroup/cpu/cpuacct.usage", "/sys/fs/cgroup/cpuacct/cpuacct.usage"):
                if Path(cand).exists():
                    usage = float(Path(cand).read_text().strip()) / 1e3
                    break
            return {"usage_usec": usage if usage is not None else 0.0, "nr_throttled": d.get("nr_throttled", 0.0),
                    "throttled_usec": d.get("throttled_time", 0.0) / 1e3, "nr_periods": d.get("nr_periods", 0.0)}
    except Exception:
        return None
    return None


class BatchStats:
    def __init__(self) -> None:
        self.roles: dict[str, str] = {}  # llm client name -> "coder" | "judge" | other actor
        self.gpu_tag = ""                # [gsplit] "c012/j3/b0123" (header token gpus=); "" = not split = no token
        self.vtags: dict[str, str] = {}  # [gsplit] vLLM metrics tag per client name (burst GLMs -> vb, vb2, ...)
        self.reset(None)

    # ---- lifecycle -------------------------------------------------------------------------------
    def reset(self, started_at: float | None) -> None:
        self.started_at = started_at
        self.ended_at: float | None = None
        self.llm: dict[str, dict[str, Any]] = {}
        self.duels: dict[str, Any] = {"n": 0, "lat": [], "wait": []}
        self.render: dict[str, Any] = {"n": 0, "fail": 0, "lat": [], "restarts": 0, "spawn_fail": 0}
        self.vllm: dict[str, dict[str, dict[str, float]]] = {}
        self.cgroup: dict[str, dict[str, float] | None] = {"start": None, "end": None}
        self.cgroup_t: dict[str, float] = {}

    def set_roles(self, roles: dict[str, str]) -> None:
        self.roles = {k: v for k, v in roles.items() if k}

    def end(self) -> None:
        if self.ended_at is None:
            self.ended_at = time.time()

    # ---- hooks (never raise) ---------------------------------------------------------------------
    def llm_call(self, name: str, dt_s: float, usage: Any = None, ok: bool = True) -> None:
        try:
            d = self.llm.setdefault(name, {"n": 0, "err": 0, "ptok": 0, "ctok": 0, "lat": [], "first": None, "last": None})
            d["n"] += 1
            if not ok:
                d["err"] += 1
            d["lat"].append(float(dt_s))
            if usage is not None:
                d["ptok"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                d["ctok"] += int(getattr(usage, "completion_tokens", 0) or 0)
            now = time.time()
            if d["first"] is None:
                d["first"] = now - dt_s
            d["last"] = now
        except Exception:
            pass

    def duel(self, compare_s: float, wait_s: float) -> None:
        try:
            self.duels["n"] += 1
            self.duels["lat"].append(float(compare_s))
            self.duels["wait"].append(float(wait_s))
        except Exception:
            pass

    def render_done(self, dt_s: float, ok: bool) -> None:
        try:
            self.render["n"] += 1
            if not ok:
                self.render["fail"] += 1
            self.render["lat"].append(float(dt_s))
        except Exception:
            pass

    def sidecar_restart(self) -> None:
        self.render["restarts"] += 1

    def sidecar_spawn_fail(self) -> None:
        self.render["spawn_fail"] += 1

    def cgroup_snapshot(self, when: str) -> None:
        try:
            self.cgroup[when] = read_cgroup_cpu()
            self.cgroup_t[when] = time.time()
        except Exception:
            pass

    async def vllm_snapshot(self, name: str, base_url: str, when: str, http_client: Any) -> None:
        """GET <server>/metrics (base_url minus the /v1 suffix) and keep the whitelisted counters."""
        try:
            url = re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/metrics"
            resp = await asyncio.wait_for(http_client.get(url), timeout=8.0)
            if resp.status_code != 200:
                return
            self.vllm.setdefault(name, {})[when] = parse_vllm_metrics(resp.text)
        except Exception:
            return

    # ---- header ----------------------------------------------------------------------------------
    def _llm_part(self, tag: str, d: dict[str, Any] | None) -> str:
        if not d or not d["n"]:
            return f"{tag}=0"
        t0 = self.started_at or d["first"] or 0.0
        span = max((d["last"] or 0.0) - (d["first"] or 0.0), 1e-3)
        tps = d["ctok"] / span if d["ctok"] else 0.0
        last = (d["last"] - t0) if d["last"] else None
        return (f"{tag}={d['n']}/{d['err']}/{_ktok(d['ptok'])}/{_ktok(d['ctok'])}/{_fmt(tps, 0)}/"
                f"{_fmt(_pct(d['lat'], 0.5))}/{_fmt(_pct(d['lat'], 0.95))}/{_fmt(last, 0)}")

    def _vllm_part(self, tag: str, snaps: dict[str, dict[str, float]] | None) -> str | None:
        if not snaps or "start" not in snaps or "end" not in snaps:
            return None
        a, b = snaps["start"], snaps["end"]
        delta = lambda k: (b.get(k, 0.0) - a.get(k, 0.0))  # noqa: E731
        q = delta("vllm:prefix_cache_queries_total")
        pch = f"{100.0 * delta('vllm:prefix_cache_hits_total') / q:.0f}%" if q > 0 else "?"
        return (f"{tag}={_ktok(delta('vllm:prompt_tokens_total'))}/{_ktok(delta('vllm:generation_tokens_total'))}/"
                f"{delta('vllm:num_preemptions_total'):.0f}/{pch}")

    def header_parts(self, now: float | None = None) -> list[str]:
        """Compact ASCII key=value tokens; every value is '?' or 0 when the axis produced nothing."""
        try:
            now = now or time.time()
            end = self.ended_at or now
            parts: list[str] = []
            if self.started_at:
                parts.append(f"end={end - self.started_at:.0f}")
            by_role: dict[str, dict[str, Any]] = {}
            for name, d in self.llm.items():  # two clients with one role (two judge servers) are merged
                role = self.roles.get(name, name)
                m = by_role.get(role)
                if m is None:
                    by_role[role] = dict(d, lat=list(d["lat"]))
                    continue
                for k in ("n", "err", "ptok", "ctok"):
                    m[k] += d[k]
                m["lat"] = m["lat"] + d["lat"]
                m["first"] = min(x for x in (m["first"], d["first"]) if x is not None) if (m["first"] or d["first"]) else None
                m["last"] = max(x for x in (m["last"], d["last"]) if x is not None) if (m["last"] or d["last"]) else None
            parts.append(self._llm_part("coder", by_role.get("coder")))
            parts.append(self._llm_part("judge", by_role.get("judge")))
            other = sum(d["n"] for r, d in by_role.items() if r not in ("coder", "judge"))
            if other:
                parts.append(f"llm_other={other}")
            du = self.duels
            parts.append(f"duels={du['n']}/{_fmt(_pct(du['lat'], 0.5))}/{_fmt(_pct(du['lat'], 0.95))}/{_fmt(_pct(du['wait'], 0.95))}")
            r = self.render
            parts.append(f"render={r['n']}/{r['fail']}/{_fmt(_pct(r['lat'], 0.5))}/{_fmt(_pct(r['lat'], 0.95))}/{r['restarts']}"
                         + (f"+{r['spawn_fail']}sf" if r["spawn_fail"] else ""))
            coder = by_role.get("coder")
            if coder and coder.get("last"):
                parts.append(f"tail={end - coder['last']:.0f}")
            seen: dict[str, int] = {}
            for name, snaps in sorted(self.vllm.items()):
                role = self.roles.get(name, name)
                tag = self.vtags.get(name) or {"coder": "vc", "judge": "vj"}.get(role, f"v_{role}"[:8])
                seen[tag] = seen.get(tag, 0) + 1
                if seen[tag] > 1:  # second server of the same role (two-judge layout): vj, vj2
                    tag = f"{tag}{seen[tag]}"
                p = self._vllm_part(tag, snaps)
                if p:
                    parts.append(p)
            if self.gpu_tag:   # [gsplit] GPU roles + duel calls served by the burst GLMs (they count in judge= too)
                nb = sum(d["n"] for n, d in self.llm.items() if self.vtags.get(n) == "vb")
                parts.append(f"gpus={self.gpu_tag} jb={nb}")
            a, b = self.cgroup.get("start"), self.cgroup.get("end")
            if a and b and "start" in self.cgroup_t and "end" in self.cgroup_t:
                wall = max(self.cgroup_t["end"] - self.cgroup_t["start"], 1e-3)
                cores = (b["usage_usec"] - a["usage_usec"]) / 1e6 / wall
                thr = (b["throttled_usec"] - a["throttled_usec"]) / 1e6 / wall
                parts.append(f"cpu={cores:.1f} thr={100.0 * thr:.0f}%")
            return parts
        except Exception:
            return ["stats=err"]


STATS = BatchStats()
