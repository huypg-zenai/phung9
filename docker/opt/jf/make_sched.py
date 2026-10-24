"""make_sched.py <orig scheduler.py> <out>: the JUDGE_FAST "enc" patch of vLLM 0.23 vllm/config/scheduler.py
(= /workspace/gdnk/s8_out/judge/patch/sched_patched.py, the file the jdd/s8 judge arms measured): the encoder cache budget
comes from $VLLM_ENC_CACHE_TOKENS (default = max_num_batched_tokens = the original behaviour). md5 of input and output
are checked against the measured files, so a different vLLM fails closed."""
import hashlib, sys
ORIG_MD5, OUT_MD5 = "d4d98bb78838733ce7c12bd461a2f5e9", "0f758a80715c6e042af17b0ae36c887e"
OLD = "        self.encoder_cache_size = self.max_num_batched_tokens\n"
NEW = ("        self.encoder_cache_size = int(__import__('os').environ.get(\n"
       "            'VLLM_ENC_CACHE_TOKENS', self.max_num_batched_tokens))  # kernel1 patch\n")
src = open(sys.argv[1], "rb").read()
if hashlib.md5(src).hexdigest() != ORIG_MD5:
    sys.exit(f"!! {sys.argv[1]} md5 {hashlib.md5(src).hexdigest()} != {ORIG_MD5} (vLLM differs)")
s = src.decode()
if s.count(OLD) != 1:
    sys.exit("!! anchor not found exactly once")
out = s.replace(OLD, NEW).encode()
if hashlib.md5(out).hexdigest() != OUT_MD5:
    sys.exit(f"!! patched md5 {hashlib.md5(out).hexdigest()} != measured {OUT_MD5}")
open(sys.argv[2], "wb").write(out)
print(f"make_sched: {sys.argv[2]} md5 {OUT_MD5} (= measured sched_patched.py)")
