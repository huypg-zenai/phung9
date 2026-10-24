"""Send each judge image to a vLLM server once; later requests reference it by uuid only (same verdicts).

Why: every judge call carries the photo + 2 candidate views as base64 PNG data URLs (~1.5 MB JSON); the vLLM API server
base64-decodes, PNG-decodes, RGBA->RGB-converts and hashes every image on EVERY arrival (~60-80 ms CPU per call), even when
its processor cache already holds the processed tensor (MM cache hit 71-77%).
vLLM 0.23 (vllm/entrypoints/chat_utils.py _parse_chat_message_content_mm_part, vllm/multimodal/processing/inputs.py
get_mm_hashes, processor.py _get_cache_missing_items) accepts a user-provided "uuid" per media part:
  * {"type": "image_url", "image_url": {"url": URL}, "uuid": U}  -> the item is processed and cached under the key U
                                                                    (U replaces the content hash; no blake3 of the pixels)
  * {"type": "image_url", "image_url": None, "uuid": U}            -> no URL at all: the processor cache entry U is reused;
                                                                    on a cache miss the server answers HTTP 400
                                                                    "Cache miss for image at index i but data is not provided."
Same tensors, same placeholder tokens, same sampling -> same outputs. U is a content hash of the data URL (sha256, so a
different image can never alias); only data: URLs are touched (an http/file URL may change behind the same string).

Protocol per underlying AsyncOpenAI client (= one server; LoadBalancedClient wraps several, each gets its own state):
  1. images this client has not yet had ACCEPTED by the server go in full form (URL + uuid);
  2. images already accepted go uuid-only;
  3. if a request with any uuid-only part fails with an HTTP error status (cache miss after eviction / server restart /
     sleep-wake), forget those uuids and resend the SAME request in full form once (identical to what stock would send,
     plus uuids). Timeouts / connection errors are re-raised untouched (the caller's retry logic is unchanged).
A uuid is marked accepted only after a successful response (by then the server's processor cache holds it).
Client LRU of accepted uuids: MM_RESEND_LRU (default 128 images; the server's default 4 GiB processor cache holds ~150 GLM
1024 px images of 25.8 MB). A too-large LRU only costs an extra 400 + resend on eviction.

Use:  from llm.mm_resend import install_mm_resend; install_mm_resend(async_openai_client)
Env:  JUDGE_IMG_UUID=0 disables (install_mm_resend returns False); MM_RESEND_LRU=<n>; MM_RESEND_STATS=<path> appends one
      JSON line of counters at interpreter exit.
Pure stdlib + openai; no pipeline imports (safe to import from anywhere).
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import time
from collections import OrderedDict

_ALL_STATS: list[dict] = []


def _uuid_for(url: str) -> str:
    return "sha256-" + hashlib.sha256(url.encode()).hexdigest()


class _LRU:
    def __init__(self, cap: int) -> None:
        self.cap = max(1, cap)
        self._d: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, k: str) -> bool:
        return k in self._d

    def add(self, k: str) -> None:
        self._d[k] = None
        self._d.move_to_end(k)
        while len(self._d) > self.cap:
            self._d.popitem(last=False)

    def touch(self, k: str) -> None:
        if k in self._d:
            self._d.move_to_end(k)

    def discard(self, k: str) -> None:
        self._d.pop(k, None)


def _image_url_of(part: dict) -> str | None:
    """The data: URL of an OpenAI image_url part, else None (non-image parts, http/file URLs, parts that already carry
    a uuid are left exactly as the caller built them)."""
    if part.get("type") != "image_url" or "uuid" in part:
        return None
    iu = part.get("image_url")
    url = iu.get("url") if isinstance(iu, dict) else iu if isinstance(iu, str) else None
    if isinstance(url, str) and url.startswith("data:"):
        return url
    return None


def rewrite(messages, known: _LRU | None):
    """-> (full_messages, slim_messages, uuids, n_slim, bytes_slim). full = every data-URL image part as URL + uuid;
    slim = parts whose uuid is in `known` as uuid-only (None if no part is known). The caller's objects are never mutated
    (changed messages/parts are shallow copies; untouched messages are shared)."""
    full, slim, uuids, n_slim, bytes_slim = [], [], [], 0, 0
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            full.append(msg); slim.append(msg); continue
        fc, sc, changed = [], [], False
        for part in content:
            url = _image_url_of(part) if isinstance(part, dict) else None
            if url is None:
                fc.append(part); sc.append(part); continue
            u = _uuid_for(url)
            uuids.append(u)
            changed = True
            fpart = {**part, "uuid": u}
            fc.append(fpart)
            if known is not None and u in known:
                n_slim += 1
                bytes_slim += len(url)
                sc.append({"type": "image_url", "image_url": None, "uuid": u})
            else:
                sc.append(fpart)
        if changed:
            full.append({**msg, "content": fc}); slim.append({**msg, "content": sc})
        else:
            full.append(msg); slim.append(msg)
    return full, (slim if n_slim else None), uuids, n_slim, bytes_slim


def install_mm_resend(client, *, lru: int | None = None, name: str = "") -> bool:
    """Wrap client.chat.completions.create (an openai AsyncOpenAI client) in place. Idempotent. Returns True if active."""
    if os.environ.get("JUDGE_IMG_UUID", "1") == "0":
        return False
    import openai   # the caller's openai (pipeline env); only APIStatusError is used

    comp = client.chat.completions
    if getattr(comp, "_mm_resend_stats", None) is not None:
        return True
    orig = comp.create
    known = _LRU(lru or int(os.environ.get("MM_RESEND_LRU", "128")))
    st = {"name": name, "pid": os.getpid(), "calls": 0, "calls_with_images": 0, "image_parts": 0, "uuid_only_parts": 0,
          "full_parts": 0, "fallbacks": 0, "errors": 0, "bytes_not_sent": 0, "t0": time.time()}

    async def create(*args, **kwargs):
        st["calls"] += 1
        messages = kwargs.get("messages")
        if not isinstance(messages, list):
            return await orig(*args, **kwargs)
        full, slim, uuids, n_slim, bytes_slim = rewrite(messages, known)
        if not uuids:
            return await orig(*args, **kwargs)
        st["calls_with_images"] += 1
        st["image_parts"] += len(uuids)
        if slim is not None:
            try:
                resp = await orig(*args, **{**kwargs, "messages": slim})
            except openai.APIStatusError:
                # cache miss (evicted / restarted server) or any other HTTP error: forget and resend the full request
                st["fallbacks"] += 1
                for u in uuids:
                    known.discard(u)
            except BaseException:
                st["errors"] += 1   # timeout / connection / cancel: re-raised untouched, the caller retries as before
                raise
            else:
                st["uuid_only_parts"] += n_slim
                st["full_parts"] += len(uuids) - n_slim
                st["bytes_not_sent"] += bytes_slim
                for u in uuids:
                    known.add(u)
                return resp
        try:
            resp = await orig(*args, **{**kwargs, "messages": full})
        except BaseException:
            st["errors"] += 1
            raise
        st["full_parts"] += len(uuids)
        for u in uuids:
            known.add(u)
        return resp

    comp.create = create  # type: ignore[method-assign]
    comp._mm_resend_stats = st
    comp._mm_resend_known = known
    _ALL_STATS.append(st)
    return True


def stats_of(client) -> dict | None:
    return getattr(client.chat.completions, "_mm_resend_stats", None)


@atexit.register
def _dump_stats() -> None:
    path = os.environ.get("MM_RESEND_STATS")
    if not path or not _ALL_STATS:
        return
    try:
        with open(path, "a") as f:
            for s in _ALL_STATS:
                f.write(json.dumps({**s, "wall_s": round(time.time() - s["t0"], 1)}) + "\n")
    except OSError:
        pass
