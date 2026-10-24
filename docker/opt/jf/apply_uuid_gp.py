"""apply_uuid_gp.py <generation_pipeline.py in> <out>: JUDGE_FAST uuid client (= tail/deep/gdnk/judge/apply_judge_uuid.py
edits) on a COPY of whichever pipeline/generation_pipeline.py the run would mount (image / burst / LOOP), plus, when the
file has the burst client factory, the same install on the burst GLM clients. Judge clients (actors.judge.client +
extra_clients, and the burst replicas) send each image once, then by uuid (llm/mm_resend.py; bitwise-equal replies per
jdd/uuid_equiv2). Exact-string anchors: an unknown file fails (exit 1) instead of patching the wrong place."""
import sys
TAG = "# judge image resend (tail/deep/gdnk/judge)"
EDITS = [
    ("from logger_config import logger\n",
     "from logger_config import logger\n"
     f"from llm.mm_resend import install_mm_resend  {TAG}\n"),
    ("        for name, cfg in self.settings.llm_clients.items():\n",
     f"        _judge = self.settings.actors.judge  {TAG}\n"
     "        _judge_names = {_judge.client, *getattr(_judge, \"extra_clients\", [])}\n"
     "        for name, cfg in self.settings.llm_clients.items():\n"),
    ("            _instrument_client(name, self._clients[name])\n",
     f"            if name in _judge_names and install_mm_resend(self._clients[name], name=name):  {TAG}\n"
     "                logger.info(f\"llm client {name}: judge images sent once, then by uuid (JUDGE_IMG_UUID=0 disables)\")\n"
     "            _instrument_client(name, self._clients[name])\n"),
]
BURST = ("            _instrument_client(name, c)\n",
         f"            if install_mm_resend(c, name=name):  {TAG} (burst GLM replica)\n"
         "                logger.info(f\"llm client {name}: burst judge images sent once, then by uuid\")\n"
         "            _instrument_client(name, c)\n")
src = open(sys.argv[1]).read()
if TAG in src:
    sys.exit("!! input already patched")
edits = EDITS + ([BURST] if "def _burst_client(" in src else [])
for old, new in edits:
    n = src.count(old)
    if n != 1:
        sys.exit(f"!! anchor found {n}x (want 1): {old.strip()!r}")
    src = src.replace(old, new)
open(sys.argv[2], "w").write(src)
print(f"apply_uuid_gp: {len(edits)} edits ({'with' if len(edits) == 4 else 'no'} burst replicas) -> {sys.argv[2]}")
