"""Apply the burst hooks (option A step 2) to copies of the leader's pipeline files. Exact-string edits only, each must
match once, so a changed upstream file fails loudly instead of being patched wrongly.
    python3 patch_burst.py <leader pipeline_service dir> <out dir>
writes <out>/stages.py and <out>/generation_pipeline.py (mount over /workspace/pipeline/...), plus burst.py and
balancer_burst.py stay as they are (mount over /workspace/llm/burst.py and /workspace/llm/balancer.py).
With env BURST_GLM_URLS unset the patched files behave exactly like the leader's (burst.CTRL is None).
"""
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

EDITS = {
    "pipeline/stages.py": [
        ("from pipeline.batch_stats import STATS\n",
         "from pipeline.batch_stats import STATS\nfrom llm import burst as _burst\n"),
        ('''        """Launch one coder per candidate at once (sem_coder caps concurrency)."""
        self._coder_tasks = {''',
         '''        """Launch one coder per candidate at once (sem_coder caps concurrency)."""
        self._burst_gen = _burst.CTRL.coder_begin(k_count, self.task.stem) if _burst.CTRL else 0
        self._coder_tasks = {'''),
        ('''    async def _generate(self, k: int) -> Candidate:
        task = self.task''',
         '''    async def _generate(self, k: int) -> Candidate:
        try:
            return await self._generate_inner(k)
        finally:
            if _burst.CTRL:
                _burst.CTRL.coder_end(getattr(self, "_burst_gen", 0))

    async def _generate_inner(self, k: int) -> Candidate:
        task = self.task'''),
        ('''                async with stage_guard(task, f"coder#k{k}", self.sem_coder, self.status):
                    cand.js_code = await self.coder.code(''',
         '''                async with stage_guard(task, f"coder#k{k}", self.sem_coder, self.status):
                    if _burst.CTRL:
                        await _burst.CTRL.ensure_coder()
                    cand.js_code = await self.coder.code('''),
    ],
    "pipeline/generation_pipeline.py": [
        ("from pipeline.task import PipelineTask\n",
         "from pipeline.task import PipelineTask\nfrom llm import burst as _burst\n"),
        ('''        await self._pipeline.start()
        actors = self.settings.actors
''',
         '''        await self._pipeline.start()
        actors = self.settings.actors
        _jname = actors.judge.client
        _jcfg = self.settings.llm_clients.get(_jname)

        def _burst_client(name: str, url: str) -> AsyncOpenAI:
            c = AsyncOpenAI(base_url=url, api_key=_resolve_api_key(_jname, _jcfg) or "local",
                            timeout=httpx.Timeout(_jcfg.request_timeout_s, connect=120.0), max_retries=0)
            _instrument_client(name, c)
            self._clients[name] = c
            self._client_urls[name] = url
            STATS.roles[name] = "judge"
            return c

        if self._pipeline.judge is not None and _jcfg is not None:
            _burst.setup(self._pipeline, self._pipeline.judge.client, _burst_client)
'''),
        ('''        await self._diag_snapshot("start")
        try:
            logger.info(
                f"[Batch starting]''',
         '''        if _burst.CTRL:
            _burst.CTRL.batch_start(len(tasks), [t.stem for t in tasks])
        await self._diag_snapshot("start")
        try:
            logger.info(
                f"[Batch starting]'''),
        ('''        finally:
            STATS.end()
            await self._diag_snapshot("end")''',
         '''        finally:
            STATS.end()
            if _burst.CTRL:
                _burst.CTRL.batch_end()
            await self._diag_snapshot("end")'''),
        # a task recorded for good (incl. one that failed before launching any coder) counts as launched
        ('''                    self.state.record_task(result)
                    return
''',
         '''                    self.state.record_task(result)
                    if _burst.CTRL:
                        _burst.CTRL.task_done(task.stem)
                    return
'''),
        # warm-up (not timed) runs one full task: exercise the whole burst path once before batch 1
        ('''            logger.info(f"[Warmup task starting]")
            future = await self._pipeline.submit(task)''',
         '''            logger.info(f"[Warmup task starting]")
            if _burst.CTRL:
                _burst.CTRL.batch_start(1, [task.stem])
            future = await self._pipeline.submit(task)'''),
        ('''        finally:
            self._cleanup_batch_memory("Warmup")''',
         '''        finally:
            if _burst.CTRL:
                _burst.CTRL.batch_end()
                await _burst.CTRL.ensure_coder("warm-up end")   # untimed: batch 1 starts with the coder awake
            self._cleanup_batch_memory("Warmup")'''),
    ],
}

for rel, edits in EDITS.items():
    s = (src / rel).read_text()
    for a, b in edits:
        n = s.count(a)
        if n != 1:
            sys.exit(f"patch_burst: {rel}: expected 1 match, found {n}: {a[:70]!r}")
        s = s.replace(a, b)
    (out / Path(rel).name).write_text(s)
    print(f"patched {rel} ({len(edits)} edits) -> {out / Path(rel).name}")
