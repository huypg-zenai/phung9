from __future__ import annotations

# pipeline-speed "option B" (owner 17/10): the leader's llm/balancer.py (jade-harrier-6 @cddc1970db1a) with LOAD-AWARE
# judge affinity. Mounted over /workspace/llm/balancer.py in our arm. Same model, same requests, same seeds: only WHICH
# GLM replica serves a request changes (math-neutral, like the leader's own replica choice).
#   Leader: a task's first request picks the replica with the fewest keys (32 stems -> 16/16) and the key stays there
#   for the process lifetime. Tasks differ in judge work (bracket size, image sizes), so one replica finishes its 16
#   early and idles while the other still has a queue (speed8: one judge GPU at 0 % for the last ~100-200 s of a batch).
#   Here: (1) first request -> the replica with the least work IN FLIGHT (ties: fewest keys, as before); (2) on every
#   later request, if the key's replica is clearly busier than the least-loaded one EVEN WITHOUT THIS TASK'S OWN requests,
#   the key MOVES there (one prefix-cache miss for that task: system prompt + reference image re-prefilled once), then
#   stays pinned again; at most one move per key per COOLDOWN_S. (v1 compared totals incl. the task's own requests: a
#   task alone on a replica ping-ponged, 62 moves for one task in the first 4 min of the 17/10 test.)
#   JUDGE_LB_MIGRATE=0 -> exactly the leader's behaviour (in-flight counting only, no decisions use it).
# Option A step 2 "burst" (owner 17/10): extra GLM replicas on the coder GPUs join for the judge-only tail of a batch:
#   add_replicas(clients, active=False) at startup, set_active(idxs, True/False) per batch (llm/burst.py). Inactive
#   replicas get no picks; a key pinned to a replica that went inactive is reassigned at its next request; while any
#   burst replica is active, keys move by the load rule above even with JUDGE_LB_MIGRATE=0 (else the fresh replicas
#   would get no traffic: every key of the batch is already pinned when the coder finishes).

import contextvars
import os
import time

from openai import AsyncOpenAI

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)

# Set by callers (JudgeAgent.compare) to pin every request of one task to one replica, so the task's shared
# prefix (system prompt + reference image) stays hot in a single engine's prefix cache. None = round-robin.
affinity_key: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_affinity_key", default=None)

LOAD_AWARE = os.environ.get("JUDGE_LB_MIGRATE", "1") != "0"
# move a key when in_flight(own) - in_flight(least) >= max(MIN_GAP, REL_GAP * in_flight(own))  (per unit weight; sim sweep 17/10: 2 / 0.3 best, +4.8 %)
MIN_GAP = int(os.environ.get("JUDGE_LB_MIN_GAP", "2"))
REL_GAP = float(os.environ.get("JUDGE_LB_REL_GAP", "0.3"))
# tie-break among equally loaded replicas: keys ACTIVE in the last ACTIVE_S seconds (the leader counts keys over the whole
# process lifetime, which in a 4-batch round would carry earlier batches' finished tasks into later batches' choices)
ACTIVE_S = float(os.environ.get("JUDGE_LB_ACTIVE_S", "300"))
COOLDOWN_S = float(os.environ.get("JUDGE_LB_COOLDOWN_S", "20"))


class _Completions:
    __slots__ = ("_lb", "_idx", "_key")

    def __init__(self, lb: "LoadBalancedClient", idx: int, key: str | None) -> None:
        self._lb = lb
        self._idx = idx
        self._key = key

    async def create(self, *args, **kwargs):
        lb, i, k = self._lb, self._idx, self._key
        lb._inflight[i] += 1
        if k is not None:
            lb._kin[k] = lb._kin.get(k, 0) + 1
        try:
            return await lb._clients[i].chat.completions.create(*args, **kwargs)
        finally:
            lb._inflight[i] -= 1
            if k is not None:
                lb._kin[k] -= 1

    def __getattr__(self, name):  # anything else (with_raw_response, ...) goes to the real client untouched
        return getattr(self._lb._clients[self._idx].chat.completions, name)


class _Chat:
    __slots__ = ("completions",)

    def __init__(self, lb: "LoadBalancedClient", idx: int, key: str | None) -> None:
        self.completions = _Completions(lb, idx, key)


class LoadBalancedClient:
    """Weighted round-robin over AsyncOpenAI clients serving the same model,
    with optional per-task affinity (see `affinity_key`), load-aware when LOAD_AWARE.
    """

    def __init__(self, clients: list[AsyncOpenAI], weights: list[int]) -> None:
        if len(clients) != len(weights):
            raise ValueError("clients and weights must have the same length")
        schedule: list[int] = []
        # Interleave by weight (e.g. weights [2,1] -> [a, b, a]) so bursts
        # don't land on a single replica.
        counters = [0] * len(clients)
        total = sum(max(1, w) for w in weights)
        for step in range(total):
            best = max(
                range(len(clients)),
                key=lambda i: max(1, weights[i]) / (counters[i] + 1),
            )
            counters[best] += 1
            schedule.append(best)
        self._schedule = schedule
        self._i = 0
        self._clients = list(clients)
        self._weights = [max(1, w) for w in weights]
        self._assign: dict[str, int] = {}
        self._keys = [0] * len(clients)
        self._inflight = [0] * len(clients)
        self._last: dict[str, float] = {}
        self._kin: dict[str, int] = {}      # in-flight requests per key
        self._moved: dict[str, float] = {}  # last move time per key
        self.moves = 0
        self._active = [True] * len(clients)
        self._burst: list[int] = []          # indices added by add_replicas
        self.reassigned = 0

    def _rebuild_schedule(self) -> None:
        act = [i for i in range(len(self._clients)) if self._active[i]] or [0]
        counters, schedule = {i: 0 for i in act}, []
        for _ in range(sum(self._weights[i] for i in act)):
            best = max(act, key=lambda i: self._weights[i] / (counters[i] + 1))
            counters[best] += 1
            schedule.append(best)
        self._schedule = schedule

    def add_replicas(self, clients: list, weights: list[int] | None = None, active: bool = False) -> list[int]:
        idxs = []
        for c, w in zip(clients, weights or [1] * len(clients)):
            idxs.append(len(self._clients))
            self._clients.append(c)
            self._weights.append(max(1, w))
            self._keys.append(0)
            self._inflight.append(0)
            self._active.append(active)
        self._burst += idxs
        self._rebuild_schedule()
        return idxs

    def set_active(self, idxs: list[int], flag: bool) -> None:
        for i in idxs:
            self._active[i] = flag
        self._rebuild_schedule()
        logger.info(f"[judge-lb] replicas {idxs} {'ACTIVE' if flag else 'inactive'} (in flight {self._inflight})")

    def inflight_of(self, idxs: list[int]) -> int:
        return sum(self._inflight[i] for i in idxs)

    def _migrating(self) -> bool:
        return LOAD_AWARE or any(self._active[i] for i in self._burst)

    def _load(self, i: int) -> float:
        return self._inflight[i] / self._weights[i]

    def _active_keys(self) -> list[int]:
        now, n = time.monotonic(), [0] * len(self._clients)
        for k, i in self._assign.items():
            if now - self._last.get(k, 0.0) < ACTIVE_S:
                n[i] += 1
        return n

    def _least(self) -> int:
        act = self._active_keys()
        live = [i for i in range(len(self._clients)) if self._active[i]] or [0]
        return min(live, key=lambda i: (self._load(i), act[i] / self._weights[i], i))

    def pick_index(self) -> int:
        key = affinity_key.get()
        if key is not None:
            idx = self._assign.get(key)
            if idx is not None and not self._active[idx]:   # its replica left the pool (burst over): re-pin
                self._keys[idx] -= 1
                idx = None
                self.reassigned += 1
            if idx is None:
                if self._migrating():
                    idx = self._least()
                else:
                    # leader: fewest keys per unit weight (32 stems -> 16/16); a key keeps its replica forever
                    live = [i for i in range(len(self._clients)) if self._active[i]] or [0]
                    idx = min(live, key=lambda i: (self._keys[i] / self._weights[i], i))
                self._assign[key] = idx
                self._keys[idx] += 1
            elif self._migrating() and sum(self._active) > 1:
                j = self._least()
                own = self._load(idx)
                others = (self._inflight[idx] - self._kin.get(key, 0)) / self._weights[idx]  # own replica minus this task
                now = time.monotonic()
                if (j != idx and others - self._load(j) >= max(MIN_GAP, REL_GAP * own)
                        and now - self._moved.get(key, -1e9) >= COOLDOWN_S):
                    self._moved[key] = now
                    self._keys[idx] -= 1
                    self._keys[j] += 1
                    self._assign[key] = j
                    self.moves += 1
                    logger.info(f"[judge-lb] task {key} moved replica {idx}->{j} (in flight "
                                f"{self._inflight[idx]} vs {self._inflight[j]}; moves so far {self.moves})")
                    idx = j
            self._last[key] = time.monotonic()
            return idx
        idx = self._schedule[self._i % len(self._schedule)]
        self._i += 1
        return idx if self._active[idx] else self._least()

    def pick(self):
        return _Chat(self, self.pick_index(), affinity_key.get())

    @property
    def chat(self):
        return self.pick()
