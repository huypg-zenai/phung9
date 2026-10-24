#!/usr/bin/env python3
"""aff_list.py - CPU list for the render fleet (RENDER_AFF=1, run_arm4.sh; ORTHO_REVIEW H1 fix (b)).

Prints a taskset CPU list "a,b,c,..." on stdout (exit 0) or a reason on stderr (exit 1 -> caller runs plain node).
  Q = floor(quota / period) from /sys/fs/cgroup/cpu.max (no quota -> number of allowed CPUs); if cpu.max does not
      exist: cgroup v1 cpu.cfs_quota_us / cpu.cfs_period_us, same dirs and rules as adaptive_k.py (-1 = no quota)
  R = Q - RENDER_AFF_RESERVE (default 14; 2 SGLang schedulers + tokenizers/detok, 2 judge engines + API servers,
      serve.py, proxies/metrics, margin)
  CPUs = R of the allowed CPUs (os.sched_getaffinity), one hardware thread per physical core first
      (topology/thread_siblings_list), the HIGHEST-numbered cores (away from CPU 0 housekeeping); siblings only if
      there are fewer than R physical cores.
Refuses (exit 1) if: cpu.max unreadable/odd, R < RENDER_AFF_MIN (default 8), or R >= allowed CPUs (nothing to confine).
"""
import math
import os
import sys

V1_DIRS = ("cpu", "cpu,cpuacct", "cpuacct,cpu", ".")  # = adaptive_k.V1_DIRS


def fail(msg):
    print(f"aff_list: {msg}", file=sys.stderr)
    sys.exit(1)


def cfs_v1(cg):
    """cgroup v1 quota as a (quota, period) pair for the cpu.max code: ("max", "-") for quota -1, else
    (str(floor(quota / period)), "1"). Same dirs/rules as adaptive_k.read_cpu_quota. None = no v1 quota file."""
    def num(path):
        x = float(open(path).read().strip())
        if not math.isfinite(x):
            raise ValueError(f"not finite: {x!r}")
        return x
    for d in V1_DIRS:
        fq, fp = f"{cg}/{d}/cpu.cfs_quota_us", f"{cg}/{d}/cpu.cfs_period_us"
        if os.path.exists(fq):
            try:
                q = num(fq)
                if q == -1:
                    return "max", "-"
                per = num(fp)
            except (OSError, ValueError) as e:
                fail(f"cannot read cgroup v1 {fq}: {e}")
            if q <= 0 or per <= 0:
                fail(f"odd cgroup v1 cfs '{q} {per}' in {cg}/{d}")
            return str(math.floor(q / per)), "1"
    return None


def main():
    reserve = int(os.environ.get("RENDER_AFF_RESERVE", "14"))
    rmin = int(os.environ.get("RENDER_AFF_MIN", "8"))
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        fail("empty affinity mask")
    cg = os.environ.get("AFF_CGROUP_ROOT", "/sys/fs/cgroup")  # override for tests only
    try:
        quota, period = open(f"{cg}/cpu.max").read().split()[:2]
    except Exception as e:  # noqa: BLE001
        v1 = None if os.path.exists(f"{cg}/cpu.max") else cfs_v1(cg)
        if v1 is None:
            fail(f"cannot read {cg}/cpu.max: {e}")
        quota, period = v1
    if quota == "max":
        q = len(allowed)
    else:
        try:
            q = int(quota) // int(period)
        except ValueError:
            fail(f"odd cpu.max '{quota} {period}'")
    r = q - reserve
    if r < rmin:
        fail(f"R = Q {q} - reserve {reserve} = {r} < min {rmin}")
    if r >= len(allowed):
        fail(f"R {r} >= {len(allowed)} allowed CPUs (nothing to confine)")
    first, rest, seen = [], [], set()
    for c in allowed:
        try:
            sib = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
        except Exception:  # noqa: BLE001  (no topology -> treat every CPU as its own core)
            sib = str(c)
        if sib in seen:
            rest.append(c)
        else:
            seen.add(sib)
            first.append(c)
    pick = first[-r:] if len(first) >= r else first + rest[-(r - len(first)):]
    if len(pick) != r:
        fail(f"picked {len(pick)} != R {r}")
    print(",".join(str(c) for c in sorted(pick)))
    print(f"aff_list: Q={q} reserve={reserve} R={r} allowed={len(allowed)} physical_cores={len(first)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
