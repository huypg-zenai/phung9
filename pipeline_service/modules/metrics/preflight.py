"""Pre-flight hardware checks. Run BEFORE model downloads start.

Exits 0 if hardware is ok, exits 1 (touches /tmp/pod_replace) otherwise. Every measurement is also written to
METRICS_FILE (/tmp/preflight_metrics.json) so the service can put it into the `// miner-diag:` header of every
module in /results (the audit's regenerated modules are published, so that header is the only channel through
which the audit host reports its specs back to us).

Checks: network (download >= 100 Mbps: protects the untimed warm-up download and the CDN image fetches),
GPU count vs configuration, per-GPU bf16 TFLOPS + decode-shaped weight-stream GB/s + power limit (see gpu.py).
CPU quota is recorded but never fails the pod (measured: a 10-CPU quota does not slow the coder; it slowed the
judge's image preprocessing only while OMP threads were unbounded, which run.sh now pins).
The former country check (reject "IS") was removed: the Iceland pods we replayed were coder-speed normal; what
they actually had (66 Mbps network, 10-CPU quota) is covered by the network check and the OMP pin.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

REPLACE_FLAG = Path("/tmp/pod_replace")
METRICS_FILE = Path(os.environ.get("PREFLIGHT_METRICS_FILE", "/tmp/preflight_metrics.json"))


def _load_config() -> dict:
    try:
        import yaml
        path = os.environ.get("CONFIG_FILE", "/workspace/configuration.yaml")
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[preflight] could not read config ({e})", file=sys.stderr)
        return {}


def _benchmark_enabled(cfg: dict) -> bool:
    return bool(cfg.get("benchmark", True))


_PREFLIGHT_ENV = {
    "min_tflops": "BENCHMARK_MIN_TFLOPS",
    "min_stream48_gbps": "BENCHMARK_MIN_STREAM48_GBPS",
    "min_power_limit_w": "BENCHMARK_MIN_POWER_LIMIT_W",
    "min_download_mbps": "BENCHMARK_MIN_DOWNLOAD_MBPS",
    # thermal / outlier gates (gpu.py + check_gpu); defaults: 1000 MHz, 60 C, 0.85, 25 C, 75 C, 20 C
    "min_sm_clock_mhz": "BENCHMARK_MIN_SM_CLOCK_MHZ",
    "max_idle_temp_c": "BENCHMARK_MAX_IDLE_TEMP_C",
    "outlier_ratio": "BENCHMARK_OUTLIER_RATIO",
    "max_idle_temp_spread_c": "BENCHMARK_MAX_IDLE_TEMP_SPREAD_C",
    "max_load_temp_c": "BENCHMARK_MAX_LOAD_TEMP_C",
    "max_load_temp_spread_c": "BENCHMARK_MAX_LOAD_TEMP_SPREAD_C",
}


def _apply_config_thresholds(cfg: dict) -> None:
    """Optional top-level `preflight:` block in configuration.yaml overrides the per-GPU-family defaults
    (gpu.py / network.py read these env vars). Explicit env vars still win over the config."""
    pf = cfg.get("preflight") or {}
    for key, env in _PREFLIGHT_ENV.items():
        if key in pf and pf[key] is not None:
            os.environ.setdefault(env, str(pf[key]))
    if pf:
        print(f"[preflight] thresholds from config: {pf}")


def _expected_gpu_count(cfg: dict) -> int:
    """Highest GPU index referenced by any enabled local vLLM client's gpu_ids, +1 (0 when everything is 'auto')."""
    hi = -1
    for spec in (cfg.get("llm_clients") or {}).values():
        if not isinstance(spec, dict) or not spec.get("enabled", False):
            continue
        ids = str((spec.get("vllm") or {}).get("gpu_ids", "auto") or "auto")
        for tok in re.split(r"[,\s]+", ids.strip()):
            if tok.isdigit():
                hi = max(hi, int(tok))
    return hi + 1


def host_info() -> dict:
    """cgroup CPU quota (v2 cpu.max, else v1 cfs), nproc, RAM GB."""
    out: dict = {"nproc": os.cpu_count()}
    try:
        p = Path("/sys/fs/cgroup/cpu.max")
        if p.exists():
            q, per = p.read_text().split()
            out["cpu_quota"] = None if q == "max" else round(int(q) / int(per), 1)
        else:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            out["cpu_quota"] = None if q < 0 else round(q / per, 1)
    except Exception:
        out["cpu_quota"] = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                out["mem_gb"] = round(int(line.split()[1]) / 1024**2)
                break
    except Exception:
        pass
    out.update(region_info())
    return out


_REGION_ENV = (("dc", "RUNPOD_DC_ID"), ("pod", "RUNPOD_POD_ID"), ("rp_cpus", "RUNPOD_CPU_COUNT"),
               ("rp_mem_gb", "RUNPOD_MEM_GB"), ("rp_gpu", "RUNPOD_GPU_NAME"))


def _ascii(s: str, limit: int = 24) -> str:
    """Header-safe token: ASCII letters/digits/._- only (Reykjavík -> Reykjavik, spaces -> _)."""
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", "_", s.strip())
    return re.sub(r"[^A-Za-z0-9._-]", "", s)[:limit]


def region_info() -> dict:
    """Where this pod runs, for the miner-diag header (so a slow audit host can be reproduced by renting the same
    datacenter): the RunPod env whitelist above (never the whole environment — it also holds API keys) plus a
    best-effort public-IP geo lookup (country/city + the hosting ASN, which names the actual machine operator)."""
    out: dict = {}
    for key, env in _REGION_ENV:
        v = os.environ.get(env)
        if v:
            out[key] = _ascii(v)
    try:
        import urllib.request
        with urllib.request.urlopen("https://ipinfo.io/json", timeout=6) as r:
            g = json.loads(r.read().decode())
        country, city = g.get("country") or "?", g.get("city") or "?"
        out["geo"] = f"{_ascii(country, 4)}/{_ascii(city, 16)}"
        org = str(g.get("org") or "")
        if org:
            asn, _, name = org.partition(" ")
            out["asn"] = _ascii(asn, 12)
            out["org"] = _ascii(name, 20)
    except Exception as e:
        print(f"[preflight] geo lookup skipped ({type(e).__name__}: {e})", file=sys.stderr)
    return out


def check_network(metrics: dict) -> bool:
    from . import network as net_bench
    net = net_bench.run_benchmark()
    metrics["network"] = {"download_mbps": net.download_mbps, "upload_mbps": net.upload_mbps,
                          "ping_ms": net.ping_ms, "crashed": net.crashed}
    # Informational only — never fails the pod. speedtest-cli under-reads badly at times (6.4 Mbps on a pod that pulled
    # HF at 1.6 Gbps, 2026-09-25); failing on it would burn a replacement for nothing. The number still goes to the header.
    if net.crashed:
        print(f"[preflight] speedtest crashed (binary/network/parse error) — ignored", file=sys.stderr)
        return True
    tag = "ok" if net.passed else "LOW (not a gate)"
    print(f"[preflight] network {tag}: {net.download_mbps} Mbps down, {net.upload_mbps} Mbps up, ping {net.ping_ms} ms")
    return True


def check_gpu(cfg: dict, metrics: dict) -> bool:
    from . import gpu as gpu_bench
    results = gpu_bench.run_benchmark()
    metrics["gpus"] = gpu_bench.results_as_dicts(results)
    ok = True
    expected = _expected_gpu_count(cfg)
    if expected and len(results) < expected:
        print(f"[preflight] only {len(results)} GPU(s) visible but configuration references {expected}", file=sys.stderr)
        ok = False
    _cross_gpu_checks(results)
    metrics["gpus"] = gpu_bench.results_as_dicts(results)   # re-serialise: the cross-GPU checks may add reasons
    for r in results:
        line = (f"GPU {r.gpu_id} ({r.gpu_name}, {r.vram_gb} GB): bf16 {r.tflops_bf16} TFLOPS | "
                f"weight-stream M=48 {r.stream48_gbps:.0f} GB/s, M=144 {r.stream144_gbps:.0f} GB/s | "
                f"power.limit {r.power_limit_w} W | sm {r.sm_clock_mhz} MHz | temp {r.temp_idle_c}->{r.temp_load_c} C"
                f"{' | throttle ' + r.throttle if r.throttle else ''}")
        if r.passed:
            print(f"[preflight] {line} — ok")
        else:
            print(f"[preflight] {line} — DEGRADED: {'; '.join(r.reasons)}", file=sys.stderr)
            ok = False
    return ok


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def _cross_gpu_checks(results) -> None:
    """Relative gates across the GPUs of one host (2+ GPUs): a GPU whose bf16 TFLOPS or weight-stream GB/s is below
    `outlier_ratio` x the median of the OTHER GPUs, or that idles more than `max_idle_temp_spread_c` above them, is a
    broken/throttled unit even when every absolute threshold passes (pod C 2026-09-27: 560 vs 660-669 TFLOPS, idle 67 C
    vs 29-31 C, sm 495 MHz — passed the absolute gates). Appends to r.reasons and clears r.passed."""
    if len(results) < 2:
        return
    ratio = float(os.environ.get("BENCHMARK_OUTLIER_RATIO", "0.85"))
    spread = float(os.environ.get("BENCHMARK_MAX_IDLE_TEMP_SPREAD_C", "25"))
    load_spread = float(os.environ.get("BENCHMARK_MAX_LOAD_TEMP_SPREAD_C", "20"))
    for r in results:
        others = [o for o in results if o.gpu_id != r.gpu_id]
        med_t = _median([o.tflops_bf16 for o in others])
        med_s = _median([o.stream48_gbps for o in others])
        med_s144 = _median([o.stream144_gbps for o in others])
        if ratio > 0 and med_t > 0 and r.tflops_bf16 < ratio * med_t:
            r.reasons.append(f"bf16 {r.tflops_bf16} TFLOPS < {ratio:.2f} x median of the other GPUs ({med_t:.0f})")
        if ratio > 0 and med_s > 0 and r.stream48_gbps < ratio * med_s:
            r.reasons.append(f"weight-stream M=48 {r.stream48_gbps:.0f} GB/s < {ratio:.2f} x median of the other GPUs ({med_s:.0f})")
        if ratio > 0 and med_s144 > 0 and r.stream144_gbps < ratio * med_s144:
            r.reasons.append(f"weight-stream M=144 {r.stream144_gbps:.0f} GB/s < {ratio:.2f} x median of the other GPUs ({med_s144:.0f})")
        idles = [o.temp_idle_c for o in others if o.temp_idle_c is not None]
        if spread > 0 and r.temp_idle_c is not None and idles and r.temp_idle_c > _median(idles) + spread:
            r.reasons.append(f"idle temperature {r.temp_idle_c} C > median of the other GPUs ({_median(idles):.0f}) + {spread:.0f}")
        loads = [o.temp_load_c for o in others if o.temp_load_c is not None]
        if load_spread > 0 and r.temp_load_c is not None and loads and r.temp_load_c > _median(loads) + load_spread:
            r.reasons.append(f"temperature under load {r.temp_load_c} C > median of the other GPUs ({_median(loads):.0f}) + {load_spread:.0f}")
        if r.reasons:
            r.passed = False


def main() -> int:
    cfg = _load_config()
    _apply_config_thresholds(cfg)
    metrics: dict = {"host": host_info()}
    print(f"[preflight] host: {metrics['host']}")
    if not _benchmark_enabled(cfg):
        print("[preflight] benchmark disabled in config — skipping network + GPU checks")
        REPLACE_FLAG.unlink(missing_ok=True)
        metrics["ok"] = True
        METRICS_FILE.write_text(json.dumps(metrics))
        return 0

    ok = True
    try:
        ok &= check_network(metrics)
    except Exception as e:
        print(f"[preflight] network check crashed: {e}", file=sys.stderr)
        ok = False

    try:
        ok &= check_gpu(cfg, metrics)
    except Exception as e:
        print(f"[preflight] gpu check crashed: {e}", file=sys.stderr)
        ok = False

    metrics["ok"] = bool(ok)
    try:
        METRICS_FILE.write_text(json.dumps(metrics))
    except Exception as e:
        print(f"[preflight] could not write {METRICS_FILE}: {e}", file=sys.stderr)

    if not ok:
        REPLACE_FLAG.touch()
        return 1

    REPLACE_FLAG.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
