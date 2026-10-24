"""Least-in-flight HTTP proxy: one OpenAI-style port in front of several single-GPU servers.

The leader pipeline talks to ONE coder URL (localhost:8001, DP2 inside vLLM). SGLang+DFlash was measured as one server per
GPU (1x H200 NVL, 64 in flight = 1.52x), so for the 2 coder GPUs we run 2 SGLang servers and put this proxy on the coder
port: each POST goes to the upstream with the fewest requests in flight (ties -> round robin), so 128 client workers
become ~64 per GPU, exactly the measured setting. Also reused in the judge phase to spread duels over several GLM servers.

    python lb_proxy.py <port> http://127.0.0.1:30001,http://127.0.0.1:30002 [--host 127.0.0.1]

GET /health -> 200 only if every upstream answers /health 200; GET /lb_stats -> per-upstream counters (json).
Other GETs go to the first upstream. Non-streaming only (the pipeline never streams).

Burst (option A step 2, owner 17/10; `--glm http://127.0.0.1:8031,...` = sleeping GLM judges on the same GPUs):
  POST /burst/to_judge : park the coder (new coder requests wait), wait until none in flight, SGLang
                         /release_memory_occupation on every upstream, then GLM /wake_up      -> {"ok", timings}
  POST /burst/to_coder : wait until the GLMs report no running/waiting request (vLLM /metrics, <= 330 s), GLM
                         /sleep?level=1, then SGLang /resume_memory_occupation, unpark           -> {"ok", timings}
                         (the caller drains its judge requests to the GLMs first)
  A failed swap undoes exactly what it did (per-upstream sets); if that fails too, "broken": /health -> 503.
  GET  /burst/state    : {"state": "coder"|"judge", counters}
  --engine vllm (option C, owner 17/10: burst in the LEADER's pipeline, vLLM 0.24 + MTP coder, no SGLang): the coder
  upstreams are vLLM servers started with --enable-sleep-mode (VLLM_SERVER_DEV_MODE=1): "release" = POST /sleep?level=1
  (weights to CPU, KV + prefix cache dropped), "resume" = POST /wake_up, idle = /metrics running + waiting == 0.
  While parked /health answers 200 itself (SGLang's /health runs a generation: never while its memory is released).
  SAFETY: a coder request that arrives while parked triggers to_coder itself (logged), then is served.
"""
import argparse
import asyncio
import itertools
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

ap = argparse.ArgumentParser()
ap.add_argument("port", type=int)
ap.add_argument("upstreams")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--api-key", default="")   # sent on the upstream /health probes (the pipeline's own probe has no key)
ap.add_argument("--glm", default="")       # burst: comma list of sleeping GLM servers on the coder GPUs
ap.add_argument("--drain-timeout", type=float, default=120.0)
ap.add_argument("--glm-drain-timeout", type=float, default=330.0)   # > judge request timeout (300 s)
ap.add_argument("--idle-poll", type=float, default=0.5)     # SGLang idle check: 2 zero reads this far apart ...
ap.add_argument("--idle-settle", type=float, default=1.0)   # ... then this long before the release
ap.add_argument("--engine", choices=["sglang", "vllm"], default="sglang")   # coder engine behind the upstreams
a = ap.parse_args()
VLLM = a.engine == "vllm"
REL_PATH, REL_BODY = ("/sleep?level=1", None) if VLLM else ("/release_memory_occupation", {})
RES_PATH, RES_BODY = ("/wake_up", None) if VLLM else ("/resume_memory_occupation", {})
GLM = [u.rstrip("/") for u in a.glm.split(",") if u.strip()]

UP = [u.rstrip("/") for u in a.upstreams.split(",") if u.strip()]
inflight = {u: 0 for u in UP}
sent = {u: 0 for u in UP}
errors = {u: 0 for u in UP}
tick = itertools.count()
AUTH = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
HOP = {"host", "content-length", "transfer-encoding", "connection", "content-encoding", "keep-alive"}
client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30),
                           limits=httpx.Limits(max_connections=2048, max_keepalive_connections=512))


# burst state: "coder" (SGLang has the GPUs) or "judge" (released; GLMs awake). `ready` is set iff state == coder.
# REL = upstreams whose memory is released, AWAKE = GLMs awake (so a partial failure undoes exactly what was done).
# broken = a swap AND its recovery failed: /health answers 503 so the pipeline sees it (never park silently forever).
B = {"state": "coder", "to_judge": 0, "to_coder": 0, "safety": 0, "last": {}, "broken": ""}
REL, AWAKE = set(), set()
ready = asyncio.Event()
ready.set()
lock = asyncio.Lock()
probing = 0                 # /health probes running on SGLang (each is a 1-token generation): drained before a release
models_cache = None         # last good GET /v1/models, answered from here while parked


def log(msg):
    print(f"[lb_proxy {time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def log_mem(tag):
    """GPU memory after a swap (best effort, runs after the swap returned: never delays it)."""
    try:
        pr = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader",
                                                  stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(pr.communicate(), 10)
        log(f"burst mem after {tag}: " + " | ".join(out.decode().split("\n")).strip(" |"))
    except Exception:  # noqa: BLE001
        pass


async def _post_each(urls, path, js=None, done=None, undo=False):
    """POST to every url; urls that answer 200 are added to (or, undo=True, removed from) the set `done`.
    Returns (seconds, [errors])."""
    async def one(u):
        try:
            r = await client.post(u + path, json=js, headers=AUTH, timeout=300)
            if r.status_code != 200:
                return f"{u}{path} -> {r.status_code} {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            return f"{u}{path} -> {type(e).__name__}: {e}"
        if done is not None:
            (done.discard if undo else done.add)(u)
        return None
    t = time.time()
    errs = [e for e in await asyncio.gather(*[one(u) for u in urls]) if e]
    return round(time.time() - t, 3), errs


async def _vllm_busy(urls):
    """running + waiting requests over vLLM servers, from /metrics (summed over every line = every DP engine; None if
    unreadable)."""
    n = 0.0
    for u in urls:
        try:
            r = await client.get(u + "/metrics", headers=AUTH, timeout=10)
            for ln in r.text.splitlines():
                if ln.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
                    n += float(ln.rsplit(" ", 1)[1])
        except Exception:  # noqa: BLE001
            return None
    return n


async def _glm_busy():
    """running + waiting requests over the awake GLMs (None if unreadable)."""
    return await _vllm_busy(list(AWAKE))


async def _sgl_busy(u):
    """running + waiting requests in one SGLang server (/get_load), None if unreadable. SGLang's release ASSERTS the
    scheduler is fully idle and kills the server otherwise (pass-2 start-up crash 17/10): never release on a guess."""
    try:
        r = await client.get(u + "/get_load", headers=AUTH, timeout=10)
        return sum(int(x.get("num_reqs", 0)) + int(x.get("num_waiting_reqs", 0)) for x in r.json())
    except Exception:  # noqa: BLE001
        return None


async def _wait_sgl_idle(limit):
    """True once every upstream reports 0 requests on 2 reads 0.5 s apart (+1 s settle for the scheduler loop).
    vLLM: never sleep a coder under running requests either (sleep drops its KV cache)."""
    t0, calm = time.time(), 0
    while time.time() - t0 < limit:
        busy = [await (_vllm_busy([u]) if VLLM else _sgl_busy(u)) for u in UP]
        calm = calm + 1 if all(b == 0 for b in busy) else 0
        if calm >= 2:
            await asyncio.sleep(a.idle_settle)
            return True
        await asyncio.sleep(a.idle_poll)
    return False


async def _recover_to_coder(why):
    """undo a partial swap: sleep the awake GLMs, resume the released upstreams. Sets broken on failure."""
    errs = []
    if AWAKE:
        errs += (await _post_each(list(AWAKE), "/sleep?level=1", done=AWAKE, undo=True))[1]
    if not AWAKE and REL:
        errs += (await _post_each(list(REL), RES_PATH, RES_BODY, done=REL, undo=True))[1]
    if AWAKE or REL:
        B["broken"] = f"{why}; recovery failed: {errs} (awake {sorted(AWAKE)}, released {sorted(REL)})"
        log("burst BROKEN: " + B["broken"])
        return False
    B["state"] = "coder"; B["broken"] = ""
    ready.set()
    return True


async def to_judge():
    async with lock:
        if B["state"] == "judge":
            return {"ok": True, "already": True}
        if B["broken"]:
            return {"ok": False, "error": "broken: " + B["broken"]}
        if not GLM:
            return {"ok": False, "error": "no --glm servers"}
        t0 = time.time()
        ready.clear()                                # park: new coder requests wait in proxy()
        while (sum(inflight.values()) or probing) and time.time() - t0 < a.drain_timeout:
            await asyncio.sleep(0.05)
        if sum(inflight.values()) or probing:
            ready.set()
            return {"ok": False, "error": f"coder still has {sum(inflight.values())} in flight after {a.drain_timeout} s"}
        tm = {"drain_s": round(time.time() - t0, 3)}
        if not await _wait_sgl_idle(60):
            ready.set()
            return {"ok": False, "error": f"{a.engine} coder not idle 60 s after the proxy drain: no release"}
        tm["idle_s"] = round(time.time() - t0 - tm["drain_s"], 3)
        tm["release_s"], errs = await _post_each(UP, REL_PATH, REL_BODY, done=REL)
        if errs:
            log(f"burst to_judge: release failed: {errs}; back to coder")
            ok = await _recover_to_coder(f"release failed: {errs}")
            return {"ok": False, "error": f"release: {errs}", "recovered": ok}
        tm["wake_s"], errs = await _post_each(GLM, "/wake_up", done=AWAKE)
        if errs:
            log(f"burst to_judge: GLM wake failed: {errs}; back to coder")
            ok = await _recover_to_coder(f"wake failed: {errs}")
            return {"ok": False, "error": f"wake: {errs}", "recovered": ok}
        B["state"] = "judge"; B["to_judge"] += 1; B["last"] = {"to_judge": tm}
        tm["total_s"] = round(time.time() - t0, 3)
        log(f"burst to_judge #{B['to_judge']}: {tm}")
        asyncio.get_running_loop().create_task(log_mem("to_judge"))
        return {"ok": True, **tm}


async def to_coder(reason="api"):
    async with lock:
        if B["state"] == "coder":
            if B["broken"] or REL or AWAKE or not ready.is_set():   # a failed recovery: retry it now
                ok = await _recover_to_coder("to_coder retry")
                log(f"burst to_coder ({reason}): recovery retry -> {'ok' if ok else 'still broken'}")
                return {"ok": ok, "recovered": ok, **({} if ok else {"error": B["broken"]})}
            return {"ok": True, "already": True}
        t0 = time.time(); tm = {}
        # never sleep a GLM under running requests (sleep drops its KV + prefix cache): wait until vLLM reports idle
        busy = await _glm_busy()
        while (busy is None or busy) and time.time() - t0 < a.glm_drain_timeout:   # None = /metrics unreadable
            await asyncio.sleep(0.1)
            busy = await _glm_busy()
        tm["glm_drain_s"] = round(time.time() - t0, 3)
        if busy is None or busy:
            log(f"burst to_coder: GLMs still busy ({busy}) after {a.glm_drain_timeout} s -> sleeping anyway")
        tm["sleep_s"], errs = await _post_each(list(AWAKE), "/sleep?level=1", done=AWAKE, undo=True)
        if errs:   # resuming next to an awake GLM would not fit
            B["broken"] = f"to_coder: GLM sleep failed: {errs}"
            log("burst BROKEN: " + B["broken"])
            return {"ok": False, "error": B["broken"]}
        tm["resume_s"], errs = await _post_each(list(REL), RES_PATH, RES_BODY, done=REL, undo=True)
        if errs:
            B["broken"] = f"to_coder: resume failed: {errs}"
            log("burst BROKEN: " + B["broken"])
            return {"ok": False, "error": B["broken"]}
        B["state"] = "coder"; B["to_coder"] += 1; B["last"]["to_coder"] = tm; B["broken"] = ""
        ready.set()
        tm["total_s"] = round(time.time() - t0, 3)
        log(f"burst to_coder #{B['to_coder']} ({reason}): {tm}")
        asyncio.get_running_loop().create_task(log_mem("to_coder"))
        return {"ok": True, **tm}


async def burst_to_judge(req):
    r = await to_judge()
    return JSONResponse(r, status_code=200 if r.get("ok") else 409)


async def burst_to_coder(req):
    r = await to_coder()
    return JSONResponse(r, status_code=200 if r.get("ok") else 409)


async def burst_state(req):
    return JSONResponse({**B, "inflight": inflight})


def pick():
    t = next(tick)
    # fewest in flight; among equals rotate by the tick so neither upstream is favoured
    return min(UP, key=lambda u: (inflight[u], (UP.index(u) - t) % len(UP)))


async def forward(u, req, body):
    path = req.url.path + (f"?{req.url.query}" if req.url.query else "")
    hdr = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
    r = await client.request(req.method, u + path, content=body, headers=hdr)
    return Response(r.content, status_code=r.status_code,
                    headers={k: v for k, v in r.headers.items() if k.lower() not in HOP})


async def health(req):
    global probing
    if B["broken"]:
        return Response("bad (burst swap failed: " + B["broken"][:300] + ")", status_code=503)
    if B["state"] != "coder" or not ready.is_set():
        return Response("ok (parked: coder memory released for the judge burst)")
    probing += 1            # counted before any await: to_judge waits for running probes before it releases
    try:
        bad = []
        for u in UP:
            try:
                r = await client.get(u + "/health", timeout=10, headers=AUTH)
                if r.status_code != 200:
                    bad.append(f"{u}:{r.status_code}")
            except Exception as e:  # noqa: BLE001
                bad.append(f"{u}:{type(e).__name__}")
    finally:
        probing -= 1
    return Response("bad " + " ".join(bad), status_code=503) if bad else Response("ok")


async def stats(req):
    return JSONResponse({"time": time.time(), "inflight": inflight, "sent": sent, "errors": errors, "burst": B})


async def proxy(req):
    body = await req.body()
    if req.method != "POST":
        global models_cache
        if req.url.path == "/v1/models" and not ready.is_set() and models_cache is not None:
            return Response(models_cache[0], media_type="application/json")   # parked: never touch a released SGLang
        r = await forward(UP[0], req, body)
        if req.url.path == "/v1/models" and r.status_code == 200:
            models_cache = (r.body,)
        return r
    while not ready.is_set():   # parked (or being parked): never send to a released SGLang
        if B["broken"] and not (await to_coder("safety retry")).get("ok"):
            return JSONResponse({"error": {"message": f"lb_proxy burst broken: {B['broken']}"}}, status_code=503)
        if B["state"] == "judge":
            B["safety"] += 1
            log(f"burst SAFETY: coder request {req.url.path} while parked -> to_coder")
            r = await to_coder("safety")
            if not r.get("ok"):
                return JSONResponse({"error": {"message": f"lb_proxy burst: {r.get('error')}"}}, status_code=503)
        else:
            try:
                await asyncio.wait_for(ready.wait(), 0.5)
            except asyncio.TimeoutError:
                pass
    u = pick()
    inflight[u] += 1
    sent[u] += 1
    try:
        return await forward(u, req, body)
    except Exception as e:  # noqa: BLE001
        errors[u] += 1
        return JSONResponse({"error": {"message": f"lb_proxy upstream {u}: {type(e).__name__}: {e}"}}, status_code=502)
    finally:
        inflight[u] -= 1


app = Starlette(routes=[Route("/health", health), Route("/lb_stats", stats),
                        Route("/burst/to_judge", burst_to_judge, methods=["POST"]),
                        Route("/burst/to_coder", burst_to_coder, methods=["POST"]),
                        Route("/burst/state", burst_state),
                        Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE"])])

if __name__ == "__main__":
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_keep_alive=600)
