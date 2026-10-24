"""jf_config.py <configuration.yaml> [util]: JUDGE_FAST server side. Every llm_clients entry whose vllm.model is
GLM-4.6V-Flash and vllm_bin is /opt/vllm-glm-env/bin/vllm (= the leader-launched judges) gets
vllm_bin /opt/jf/vllm_enc.sh (enc cache 1.5M tokens) and gpu_memory_utilization <util> (0.81). Line edits only (the rest
of the file stays byte-identical); the result is re-parsed and checked. Exit 1 if no GLM judge was found."""
import re, sys
import yaml
path = sys.argv[1]; util = sys.argv[2] if len(sys.argv) > 2 else "0.81"
lines = open(path).read().split("\n")
top = None; client = None; blocks = {}
for i, l in enumerate(lines):
    if l.startswith("#"): continue
    if re.match(r"^\S.*:\s*$", l): top = l.split(":")[0]; client = None; continue
    if top == "llm_clients" and re.match(r"^  [A-Za-z0-9_.-]+:\s*$", l): client = l.strip()[:-1]; continue
    if top == "llm_clients" and client: blocks.setdefault(client, []).append(i)
done = []
for c, idx in blocks.items():
    body = [lines[i] for i in idx]
    if not any(re.match(r'^      model: "?zai-org/GLM-4\.6V-Flash"?\s*$', b) for b in body): continue
    if not any(re.match(r'^      vllm_bin: "?/opt/vllm-glm-env/bin/vllm"?\s*$', b) for b in body): continue
    nb = nu = 0
    for i in idx:
        if re.match(r'^      vllm_bin: ', lines[i]): lines[i] = '      vllm_bin: "/opt/jf/vllm_enc.sh"'; nb += 1
        elif re.match(r'^      gpu_memory_utilization: ', lines[i]): lines[i] = f"      gpu_memory_utilization: {util}"; nu += 1
    if nb != 1 or nu != 1: sys.exit(f"!! client {c}: vllm_bin x{nb}, gpu_memory_utilization x{nu} (want 1 each)")
    done.append(c)
if not done: sys.exit("!! no GLM judge client with /opt/vllm-glm-env/bin/vllm found")
txt = "\n".join(lines)
cfg = yaml.safe_load(txt)
for c in done:
    v = cfg["llm_clients"][c]["vllm"]
    assert v["vllm_bin"] == "/opt/jf/vllm_enc.sh" and float(v["gpu_memory_utilization"]) == float(util), c
open(path, "w").write(txt)
print(f"jf_config: {', '.join(done)} -> vllm_bin /opt/jf/vllm_enc.sh, gpu_memory_utilization {util}")
