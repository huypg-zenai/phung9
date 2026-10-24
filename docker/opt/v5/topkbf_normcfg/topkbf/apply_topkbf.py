"""Apply the kopt-topkbf patch (see kopt_topkbf.py / README.md) to the sglang that `import sglang` resolves to, in
place, idempotent, coexisting with the topk20 patch (opt/v4/topk20/apply_topk.py) and the seed patch
(opt/v3/seed/dflash_seed.patch) in any order: no hook touches a line those patches match on.

  python apply_topkbf.py                         patch (exit 0; already-applied hooks are skipped)
  python apply_topkbf.py --check                 hooks + module present, patched files compile and import, then the
                                                 GPU bitwise self-test (needs a CUDA GPU); last line
                                                 'apply_topkbf check: PASS ...' (exit 0) or '... FAIL ...' (exit != 0)
  python apply_topkbf.py --check --compile-only  hooks + module present and compile (no sglang deps, no GPU)
  python apply_topkbf.py --revert                remove the hooks + module; verify each file's sha256 against the one
                                                 recorded before the first apply
  python apply_topkbf.py --root DIR              operate on the package at DIR (the directory that contains srt/)
  --hidden H (check)                             hidden size for the lm_head-into-buffer test (default 5120)

Runtime switch: SGL_TOPKBF=1 (read per call). Unset/0 = the hooks do nothing (the stock code runs, byte for byte).

Hooks (each marked '# kopt topkbf:<tag>'):
  srt/layers/logits_processor.py      LogitsProcessor._get_logits: the _compute_lm_head call      (tag lm_head)
                                      LogitsProcessor._copy_logits_to_buffer: early return        (tag copy)
  srt/speculative/dflash_utils.py     build_dflash_verify_target_probs: bf16 dispatch at the top  (tag dispatch)
  srt/speculative/dflash_worker_v2.py DFlashWorkerV2 verify: fp32 guard after the target forward  (tag verify)
  srt/speculative/kopt_topkbf.py      NEW: copy of kopt_topkbf.py
Record: srt/speculative/.kopt_topkbf_record.json (sha256 of each touched file before/after our edit).
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "apply_topkbf"
MODULE_SRC = os.path.join(HERE, "kopt_topkbf.py")
MODULE_DST = "srt/speculative/kopt_topkbf.py"
RECORD = "srt/speculative/.kopt_topkbf_record.json"

LP = "srt/layers/logits_processor.py"
DU = "srt/speculative/dflash_utils.py"
WV = "srt/speculative/dflash_worker_v2.py"

LMHEAD_ANCHOR = "        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)\n"
LMHEAD_TEXT = (
    "        # kopt topkbf:lm_head  SGL_TOPKBF=1 -> DFlash target-verify logits stay bf16 (kopt_topkbf.py); else stock\n"
    "        import os as _kopt_tbf_os\n"
    "\n"
    "        _kopt_tbf_logits = None\n"
    "        if _kopt_tbf_os.environ.get(\"SGL_TOPKBF\", \"0\") == \"1\":\n"
    "            from sglang.srt.speculative.kopt_topkbf import lm_head_hook as _kopt_tbf_hook\n"
    "\n"
    "            _kopt_tbf_logits = _kopt_tbf_hook(\n"
    "                self, hidden_states, lm_head, embedding_bias, logits_metadata\n"
    "            )\n"
    "        if _kopt_tbf_logits is not None:\n"
    "            logits = _kopt_tbf_logits\n"
    "        else:\n"
    "            logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)\n"
)

COPY_ANCHOR = (
    "    def _copy_logits_to_buffer(\n"
    "        self,\n"
    "        logits: torch.Tensor,\n"
    "        logits_metadata: LogitsMetadata,\n"
    "        use_buffer: bool = True,\n"
    "    ) -> torch.Tensor:\n"
)
COPY_TEXT = (
    "        # kopt topkbf:copy  bf16 verify logits from kopt_topkbf.lm_head_hook (SGL_TOPKBF=1): no fp32 copy\n"
    "        if getattr(self, \"_kopt_tbf_keep\", False):\n"
    "            try:\n"
    "                if logits.dtype == torch.bfloat16:\n"
    "                    return logits\n"
    "            finally:\n"
    "                self._kopt_tbf_keep = False\n"
)

DISPATCH_ANCHOR = (
    "    device = next_token_logits.device\n"
    "    need_top_k = bool(getattr(sampling_info, \"need_top_k_sampling\", True))\n"
)
DISPATCH_TEXT = (
    "    # kopt topkbf:dispatch  bf16 verify logits exist only with SGL_TOPKBF=1 (kopt_topkbf.py); fp32 = stock below\n"
    "    if next_token_logits.dtype == torch.bfloat16:\n"
    "        from sglang.srt.speculative.kopt_topkbf import (\n"
    "            build_target_probs_bf16 as _kopt_tbf_build,\n"
    "        )\n"
    "\n"
    "        return _kopt_tbf_build(\n"
    "            build_dflash_verify_target_probs,\n"
    "            next_token_logits=next_token_logits,\n"
    "            sampling_info=sampling_info,\n"
    "            draft_token_num=draft_token_num,\n"
    "            bs=bs,\n"
    "            max_top_k=max_top_k,\n"
    "            uniform_top_k_value=uniform_top_k_value,\n"
    "            use_sparse_topk=use_sparse_topk,\n"
    "        )\n"
)

VERIFY_ANCHOR = (
    "        logits_output = target_out.logits_output\n"
    "        can_run_cuda_graph = target_out.can_run_cuda_graph\n"
)
VERIFY_TEXT = (
    "        # kopt topkbf:verify  bf16 verify logits (SGL_TOPKBF=1) + a consumer that needs fp32 -> exact upcast\n"
    "        if (\n"
    "            logits_output.next_token_logits is not None\n"
    "            and logits_output.next_token_logits.dtype == torch.bfloat16\n"
    "        ):\n"
    "            from sglang.srt.speculative.kopt_topkbf import (\n"
    "                verify_logits_guard as _kopt_tbf_guard,\n"
    "            )\n"
    "\n"
    "            _kopt_tbf_guard(self, batch, logits_output)\n"
)

# (file, tag, mode, anchor, text)  mode: replace (text contains the anchor's code) | before | after
EDITS = [
    (LP, "lm_head", "replace", LMHEAD_ANCHOR, LMHEAD_TEXT),
    (LP, "copy", "after", COPY_ANCHOR, COPY_TEXT),
    (DU, "dispatch", "before", DISPATCH_ANCHOR, DISPATCH_TEXT),
    (WV, "verify", "after", VERIFY_ANCHOR, VERIFY_TEXT),
]
FILES = [LP, DU, WV]
MODULES = ["sglang.srt.speculative.kopt_topkbf", "sglang.srt.layers.logits_processor",
           "sglang.srt.speculative.dflash_utils", "sglang.srt.speculative.dflash_worker_v2"]


BACKUP = ".kopt_topkbf.orig"   # <file>.kopt_topkbf.orig = the file as it was before our first edit


def marker(tag):
    return f"# kopt topkbf:{tag}"


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def new_block(mode, anchor, text):
    if mode == "replace":
        return text
    return text + anchor if mode == "before" else anchor + text


def find_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit(f"{TAG}: `import sglang` does not resolve (set PYTHONPATH or use --root)")
    return list(spec.submodule_search_locations)[0]


def read(p):
    with open(p) as fh:
        return fh.read()


def write_atomic(p, s):
    tmp = p + ".kopt_tbf_tmp"
    with open(tmp, "w") as fh:
        fh.write(s)
    os.replace(tmp, p)
    if p.endswith(".py"):
        pyc = os.path.join(os.path.dirname(p), "__pycache__")
        stem = os.path.basename(p)[:-3] + "."
        if os.path.isdir(pyc):
            for c in os.listdir(pyc):
                if c.startswith(stem):
                    try:
                        os.remove(os.path.join(pyc, c))
                    except OSError:
                        pass


def compile_err(text, name):
    try:
        compile(text, name, "exec")
        return None
    except SyntaxError as e:
        return f"{name}: {e}"


def die(msg, rc=1):
    print(f"{TAG}: !! {msg}")
    raise SystemExit(rc)


def tags_of(rel):
    return [(t, m, a, x) for f, t, m, a, x in EDITS if f == rel]


def anchors_ok(rel, src):
    for t, m, a, x in tags_of(rel):
        n = src.count(a)
        if n != 1:
            return f"hook '{t}': anchor found {n} times (SGLang differs from @1cf2b8c54?)"
    return None


def apply_text(rel, src):
    for t, m, a, x in tags_of(rel):
        src = src.replace(a, new_block(m, a, x), 1)
    return src


def hooks_verbatim(rel, src):
    return all(src.count(new_block(m, a, x)) == 1 for t, m, a, x in tags_of(rel))


def any_marker(rel, src):
    return any(marker(t) in src for t, *_ in tags_of(rel))


def revert_text(rel, src):
    for t, m, a, x in tags_of(rel):
        src = src.replace(new_block(m, a, x), a, 1)
    return src


def load_record(root):
    p = os.path.join(root, RECORD)
    if os.path.exists(p):
        try:
            return json.loads(read(p))
        except ValueError:
            pass
    return {"files": {}}


def backup_of(root, rel, rec):
    """The recorded original text of rel when the backup exists and matches the recorded sha256, else None."""
    want = rec.get("files", {}).get(rel, {}).get("sha256_before")
    p = os.path.join(root, rel) + BACKUP
    if want and os.path.exists(p):
        b = read(p)
        if sha(b) == want:
            return b
    return None


def status(root):
    missing = []
    for rel in FILES:
        p = os.path.join(root, rel)
        src = read(p) if os.path.exists(p) else ""
        for t, m, a, x in tags_of(rel):
            if marker(t) not in src:
                missing.append(f"{rel}:{t}")
            elif src.count(new_block(m, a, x)) != 1:
                missing.append(f"{rel}:{t} (stale hook: marker present, block differs from this kit)")
    dst = os.path.join(root, MODULE_DST)
    if not os.path.exists(dst) or read(dst) != read(MODULE_SRC):
        missing.append(MODULE_DST + " (missing or differs from kopt_topkbf.py)")
    return missing


def do_apply(root):
    """Stage everything first (nothing written on any problem), then: module -> backups -> record -> files."""
    rec = load_record(root)
    plan = []    # (rel, path, base_text, new_text, how)
    for rel in FILES:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            die(f"target missing: {p} - nothing written")
        src = read(p)
        if any_marker(rel, src):
            if all(marker(t) in src for t, *_ in tags_of(rel)) and hooks_verbatim(rel, src):
                print(f"  {rel}: hooks already present")
                continue
            # stale hook (an older kit / hand edit): re-apply from the recorded original if the file is exactly
            # what our last apply wrote, else refuse loudly
            base = backup_of(root, rel, rec)
            if base is None or sha(src) != rec["files"][rel].get("sha256_after"):
                die(f"stale hook in {rel}: '# kopt topkbf' marker present but the block differs from this kit and "
                    f"the file is not exactly what the last apply wrote (or no verified backup) - nothing written; "
                    f"restore the original file by hand")
            how = "stale hook replaced (re-applied from the recorded original)"
        else:
            base, how = src, "hooks applied"
        why = anchors_ok(rel, base)
        if why:
            die(f"{rel} {why} - nothing written")
        new = apply_text(rel, base)
        err = compile_err(new, p)
        if err:
            die(f"patched text does not compile - nothing written: {err}")
        plan.append((rel, p, base, new, how))
    body = read(MODULE_SRC)
    err = compile_err(body, MODULE_SRC)
    if err:
        die(f"kopt_topkbf.py does not compile - nothing written: {err}")
    try:
        write_atomic(os.path.join(root, MODULE_DST), body)                    # 1. module (unused until hooks)
        for rel, p, base, new, how in plan:                                   # 2. backups of the originals
            b = p + BACKUP
            if not (os.path.exists(b) and read(b) == base):
                write_atomic(b, base)
            rec["files"][rel] = {"sha256_before": sha(base), "sha256_after": sha(new)}
        rec["module_sha256"] = sha(body)
        write_atomic(os.path.join(root, RECORD), json.dumps(rec, indent=1))   # 3. record
        for rel, p, base, new, how in plan:                                   # 4. files
            write_atomic(p, new)
            print(f"  {rel}: {how} ({', '.join(t for t, *_ in tags_of(rel))}); sha256 before {sha(base)[:12]}")
    except OSError as e:
        die(f"write failed: {e}; state is recorded in {RECORD} (+ {BACKUP} backups): re-run apply or --revert")
    print(f"  {MODULE_DST}: installed")
    print(f"{TAG}: done (active only with SGL_TOPKBF=1)")
    return 0


def do_revert(root):
    """Stage all reverts (nothing written if a file cannot be reverted), write, then verify sha256 vs the record.
    Exit 0 = every file back to its recorded original; 7 = hooks removed but a file differs from the recorded
    original (another patch edited it after us); 8 = refused (stale hook without a verified backup)."""
    rec = load_record(root)
    plan = []
    for rel in FILES:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            continue
        src = read(p)
        if not any_marker(rel, src):
            print(f"  {rel}: no hooks (nothing to revert)")
            continue
        if all(marker(t) in src for t, *_ in tags_of(rel)) and hooks_verbatim(rel, src):
            plan.append((rel, p, revert_text(rel, src), "hooks removed"))
            continue
        base = backup_of(root, rel, rec)
        if base is not None and sha(src) == rec["files"][rel].get("sha256_after"):
            plan.append((rel, p, base, "stale hook: original restored from the verified backup"))
            continue
        die(f"stale hook in {rel}: marker present but the block differs from this kit and no verified backup "
            f"matches - nothing written; restore the file by hand", rc=8)
    bad = 0
    for rel, p, new, how in plan:
        write_atomic(p, new)
        want = rec.get("files", {}).get(rel, {}).get("sha256_before")
        if want is None:
            print(f"  {rel}: {how} (no recorded sha256)")
        elif sha(new) == want:
            print(f"  {rel}: {how}, sha256 == original {want[:12]} OK")
        else:
            bad += 1
            print(f"  {rel}: {how}; !! sha256 {sha(new)[:12]} != recorded original {want[:12]} "
                  f"(another patch changed this file after kopt-topkbf was applied)")
    for rel in [MODULE_DST, RECORD] + [f + BACKUP for f in FILES]:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            os.remove(p)
            print(f"  removed {rel}")
    if bad:
        print(f"{TAG}: !! reverted, but {bad} file(s) differ from the recorded original (see above)")
        return 7
    print(f"{TAG}: reverted")
    return 0


# ------------------------------------------------------------------------------------------------ GPU self-test
def model_dims(model):
    """(hidden_size, vocab_size) from a local model dir or a cached HF repo id, else (None, None)."""
    if not model:
        return None, None
    cands = []
    if os.path.isdir(model):
        cands.append(os.path.join(model, "config.json"))
    else:
        try:
            from huggingface_hub import try_to_load_from_cache

            c = try_to_load_from_cache(model, "config.json")
            if isinstance(c, str):
                cands.append(c)
        except Exception:  # noqa: BLE001
            pass
        import glob

        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        cands += sorted(glob.glob(os.path.join(hub, "models--" + model.replace("/", "--"), "snapshots", "*",
                                               "config.json")))
    for c in cands:
        try:
            cfg = json.loads(read(c))
        except Exception:  # noqa: BLE001
            continue
        tc = cfg.get("text_config") or {}
        h = cfg.get("hidden_size") or tc.get("hidden_size")
        v = cfg.get("vocab_size") or tc.get("vocab_size")
        if h:
            return int(h), (int(v) if v else None)
    return None, None


def gpu_selftest(root, hiddens, V_model=None):
    """-> (ok, summary, fails). All comparisons bitwise (fp32 compared as int32 bit patterns)."""
    import types

    import torch

    du = importlib.import_module("sglang.srt.speculative.dflash_utils")
    lpm = importlib.import_module("sglang.srt.layers.logits_processor")
    km = importlib.import_module("sglang.srt.speculative.kopt_topkbf")
    fails, notes = [], []
    dev = "cuda"
    V = 248320
    K = 20
    topk20_on = os.path.exists(os.path.join(root, "srt/speculative/topk20.py")) and \
        "sgl-topk20:call" in read(os.path.join(root, DU))
    t20_modes = ["0", "1"] if topk20_on else ["0"]

    def bits(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype == torch.float32:
            return torch.equal(a.view(torch.int32), b.view(torch.int32))
        return torch.equal(a, b)

    def logits(M, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(M, V, device=dev, generator=g) * 2.0
        hot = torch.randint(0, V, (M, 40), device=dev, generator=g)
        boost = torch.rand(M, 40, device=dev, generator=g) * 14.0 + 3.0
        keep = torch.rand(M, 40, device=dev, generator=g) < torch.rand(M, 1, device=dev, generator=g)
        keep[:, 0] = True
        x.scatter_add_(1, hot, boost * keep)
        lb = x.to(torch.bfloat16)
        r = torch.arange(0, M, 3, device=dev)
        v = torch.topk(lb[r].float(), K + 1, dim=-1).values
        lb[r, 1000:1006] = v[:, K - 1:K].to(torch.bfloat16)      # ties at the k-th boundary
        lb[r, 9000:9003] = v[:, :1].to(torch.bfloat16)           # ties at the max (argmax tie test too)
        lb[r, 200000:200002] = v[:, :1].to(torch.bfloat16)
        # subnormal rows: all -inf except 25 tiny values (|x| ~ 1e-39 .. 2.5e-38, /T stays subnormal) + -0/+0
        sub = torch.arange(5, M, 17, device=dev)
        if sub.numel():
            lb[sub] = float("-inf")
            pos = torch.arange(25, device=dev) * 9001 + 77
            mag = torch.arange(1, 26, device=dev, dtype=torch.float32) * 1e-39
            sgn = torch.where(torch.arange(25, device=dev) % 4 == 3, -1.0, 1.0)
            vals = (mag * sgn).to(torch.bfloat16)
            vals[10] = 0.0
            vals[11] = -0.0
            lb[sub[:, None], pos[None, :]] = vals[None, :].expand(sub.numel(), -1)
        return lb, sub

    def sinfo(bs, mixed, top_p):
        cyc = torch.tensor([0.7, 0.3, 1.3, 1.0], device=dev, dtype=torch.float32)
        temps = cyc[torch.arange(bs, device=dev) % 4].reshape(bs, 1).contiguous()
        tks = torch.full((bs,), K, device=dev, dtype=torch.int32)
        if mixed:
            tks[1::3] = 5
            tks[2::5] = 1
        tps = torch.ones((bs,), device=dev, dtype=torch.float32)
        if top_p:
            tps[::2] = 0.95
        return types.SimpleNamespace(temperatures=temps, top_ks=tks, top_ps=tps, need_top_k_sampling=True,
                                     need_top_p_sampling=top_p)

    old_t20 = os.environ.get("SGL_TOPK20")
    try:
        for M in (512, 384, 40, 8):
            bs = M // 8
            lb, sub = logits(M, M)
            for name, mixed, top_p, uni in (("uniform20", False, False, K), ("mixed_topk+top_p", True, True, None)):
                si = sinfo(bs, mixed, top_p)
                kw = dict(sampling_info=si, draft_token_num=8, bs=bs, max_top_k=None, uniform_top_k_value=uni)
                hooked = du.build_dflash_verify_target_probs(next_token_logits=lb, **kw)
                for m20 in t20_modes:
                    os.environ["SGL_TOPK20"] = m20
                    cur = du.build_dflash_verify_target_probs(next_token_logits=lb.float(), **kw)
                    if not bits(hooked, cur):
                        fails.append(f"M={M} {name} SGL_TOPK20={m20}: hooked != current chain")
                if name == "uniform20":
                    te = torch.repeat_interleave(si.temperatures, 8, dim=0)
                    rv, ri = torch.topk(lb.float() / te, K, dim=-1)
                    pure = torch.zeros((M, V), device=dev).scatter_(1, ri, torch.softmax(rv, -1)).view(bs, 8, -1)
                    if not bits(hooked, pure):
                        fails.append(f"M={M} {name}: hooked != pure torch")
                    ov, oi = km.topk_scaled_bf16(lb, te, K)
                    if not (bits(ov, rv) and torch.equal(oi, ri)):
                        fails.append(f"M={M}: topk_scaled_bf16 != torch.topk(fp32 / T)")
                    if sub.numel() and M > 10:
                        sv = rv[sub]
                        nsub = int(((sv != 0) & (sv.abs() < 1.1754944e-38)).sum())
                        if not (bits(ov[sub], sv) and torch.equal(oi[sub], ri[sub])):
                            fails.append(f"M={M}: subnormal rows differ (flush-to-zero?)")
                        if nsub == 0:
                            fails.append(f"M={M}: subnormal rows carry no subnormal top-k value (bad test data)")
                        notes.append(f"M={M} subnormal top-k values {nsub}")
                    inner = (rv[:, 1:] == rv[:, :-1]).any(1).float().mean().item()
                    notes.append(f"M={M} tie-rows {inner:.2f}")
                    if M == 512:
                        caught = []
                        for div in (2, 3):
                            bv, bi = km.topk_variant(lb, te.reshape(-1).contiguous(), K, div=div)
                            caught.append(not (bits(bv, rv) and torch.equal(bi, ri)))
                        notes.append(f"BAD approx caught={caught[0]} recip caught={caught[1]}")
                        if not any(caught):
                            fails.append("sanity: neither BAD variant was caught by the bitwise test")
                        # PTX of the production kernels: IEEE div.rn.f32 present, no flush-to-zero anywhere
                        km.topk_variant(lb, te.reshape(-1).contiguous(), K)
                        handles = km._LAST.get(km.DIV_MODE)
                        for sname, hk in zip(("stage1", "stage2"), handles or (None, None)):
                            ptx = getattr(hk, "asm", {}).get("ptx") if hk is not None else None
                            if not ptx:
                                fails.append(f"PTX of {sname} not retrievable (cannot prove no .ftz)")
                            elif "ftz" in ptx:
                                fails.append(f"PTX of {sname} contains 'ftz'")
                            elif km.DIV_MODE == 1 and "div.rn.f32" not in ptx:
                                fails.append(f"PTX of {sname} has no div.rn.f32")
                        notes.append("PTX: div.rn.f32, no ftz" if handles else "PTX: n/a")
            if M == 40:   # dense path (need_top_k False) in the builder -> fp32 upcast + stock
                si = sinfo(bs, False, False)
                si.need_top_k_sampling = False
                kw = dict(sampling_info=si, draft_token_num=8, bs=bs, max_top_k=None, uniform_top_k_value=None)
                if not bits(du.build_dflash_verify_target_probs(next_token_logits=lb, **kw),
                            du.build_dflash_verify_target_probs(next_token_logits=lb.float(), **kw)):
                    fails.append("dense path: hooked != stock")
            a16, a32 = lb.argmax(-1), lb.float().argmax(-1)
            first = (lb.float() == lb.float().max(-1, keepdim=True).values).int().argmax(-1)
            if not (torch.equal(a16, a32) and torch.equal(a32, first)):
                fails.append(f"M={M}: argmax(bf16) != argmax(fp32) / first max")
            del lb
            torch.cuda.empty_cache()

        # lm_head into a bf16 view of the shared fp32 buffer (the hook), eager + CUDA graph; copy hook; guard
        Vl = V_model or V
        shared = torch.zeros((520, Vl), device=dev, dtype=torch.float32)
        lp = types.SimpleNamespace(use_fp32_lm_head=False, final_logit_softcapping=None,
                                   do_tensor_parallel_all_gather=False, do_tensor_parallel_all_gather_dp_attn=False,
                                   rl_on_policy_target=None, vocab_size=Vl)
        fm = types.SimpleNamespace(is_target_verify=lambda: True)
        real_spec = km._spec_is_dflash
        km._spec_is_dflash = lambda: None          # no server runtime context in this process
        try:
            for hidden in hiddens:
                g = torch.Generator(device=dev).manual_seed(7)
                W = (torch.randn(Vl, hidden, device=dev, generator=g) * 0.02).to(torch.bfloat16)
                lm_head = types.SimpleNamespace(weight=W)
                for rows in (512, 384, 40):
                    h = torch.randn(rows, hidden, device=dev, generator=g).to(torch.bfloat16)
                    ref = torch.matmul(h, W.T)
                    meta = types.SimpleNamespace(forward_mode=fm, next_token_logits_buffer=shared[:rows])
                    out = km.lm_head_hook(lp, h, lm_head, None, meta)
                    if out is None or out.dtype != torch.bfloat16 or out.data_ptr() != shared.data_ptr():
                        fails.append(f"H={hidden} rows={rows}: lm_head hook did not write bf16 into the shared buffer")
                        continue
                    if not bits(out, ref):
                        fails.append(f"H={hidden} rows={rows}: matmul(out=buffer view) != matmul (eager)")
                    kept = lpm.LogitsProcessor._copy_logits_to_buffer(lp, out, meta)
                    if kept is not out or getattr(lp, "_kopt_tbf_keep", True):
                        fails.append(f"rows={rows}: copy hook did not return the bf16 logits / reset the flag")
                    # in-place upcast by the guard (return_logprob batch) == stock buffer contents
                    lo = types.SimpleNamespace(next_token_logits=out)
                    km.verify_logits_guard(types.SimpleNamespace(), types.SimpleNamespace(
                        return_logprob=True, has_grammar=False, sampling_info=None), lo)
                    y = lo.next_token_logits
                    if not (y.dtype == torch.float32 and y.data_ptr() == shared.data_ptr() and bits(y, ref.float())):
                        fails.append(f"H={hidden} rows={rows}: in-place upcast into the shared buffer != stock copy")
                    lp._kopt_tbf_keep = True              # stale flag + fp32 logits -> stock copy, flag reset
                    stock = lpm.LogitsProcessor._copy_logits_to_buffer(lp, ref.float(), meta)
                    if stock.dtype != torch.float32 or not bits(stock, ref.float()) or lp._kopt_tbf_keep:
                        fails.append(f"rows={rows}: stale flag + fp32 logits did not take the stock copy path")
                    stock = lpm.LogitsProcessor._copy_logits_to_buffer(lp, ref, meta)   # flag off -> stock copy
                    if stock.dtype != torch.float32 or not bits(stock, ref.float()):
                        fails.append(f"rows={rows}: stock copy path changed")
                    # CUDA graph: hook matmul vs stock matmul, both captured, replayed on new hidden states
                    hs = h.clone()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        km.lm_head_hook(lp, hs, lm_head, None, meta)
                        torch.matmul(hs, W.T)
                    torch.cuda.current_stream().wait_stream(s)
                    g_hook, g_stock = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g_hook):
                        gout = km.lm_head_hook(lp, hs, lm_head, None, meta)
                    with torch.cuda.graph(g_stock):
                        sout = torch.matmul(hs, W.T)
                    hs.copy_(torch.randn(rows, hidden, device=dev, generator=g).to(torch.bfloat16))
                    g_hook.replay()
                    g_stock.replay()
                    torch.cuda.synchronize()
                    if not bits(gout, sout):
                        fails.append(f"H={hidden} rows={rows}: graph replay (out=buffer view) != graph replay (stock)")
                    if not bits(gout, torch.matmul(hs, W.T)):
                        notes.append(f"H={hidden} rows={rows}: graph matmul != eager matmul (stock property)")
                    del g_hook, g_stock, sout
                    lp._kopt_tbf_keep = False
                notes.append(f"lm_head H={hidden} V={Vl} OK" if not any(f"H={hidden}" in f for f in fails) else
                             f"lm_head H={hidden} FAILED")
                del W
                torch.cuda.empty_cache()
            meta_d = types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_target_verify=lambda: False),
                                           next_token_logits_buffer=shared[:8])
            if km.lm_head_hook(lp, torch.zeros(8, hiddens[0], device=dev, dtype=torch.bfloat16),
                               types.SimpleNamespace(weight=torch.zeros(Vl, hiddens[0], device=dev,
                                                                        dtype=torch.bfloat16)), None,
                               meta_d) is not None:
                fails.append("lm_head hook: non-verify forward was not left to the stock path")
        finally:
            km._spec_is_dflash = real_spec
        # verify guard: every reason upcasts exactly; no reason leaves bf16
        x = torch.randn(16, V, device=dev).to(torch.bfloat16)

        def si_mock(**over):
            d = dict(has_custom_logit_processor=False, penalizer_orchestrator=None, acc_linear_penalties=None,
                     logit_bias=None, grammar_mask=None, is_all_greedy=False, need_top_k_sampling=True)
            d.update(over)
            return types.SimpleNamespace(**d)

        cases = [
            ("none", {}, {}, {}, False),
            ("return_logprob", {"return_logprob": True}, {}, {}, True),
            ("has_grammar", {"has_grammar": True}, {}, {}, True),
            ("selector", {}, {}, {"_selector_sample": (1, 2)}, True),
            ("custom", {}, {"has_custom_logit_processor": True}, {}, True),
            ("penalties", {}, {"penalizer_orchestrator": types.SimpleNamespace(is_required=True)}, {}, True),
            ("penalizer not required", {}, {"penalizer_orchestrator": types.SimpleNamespace(is_required=False)},
             {}, False),
            ("acc_linear_penalties", {}, {"acc_linear_penalties": torch.zeros(1)}, {}, True),
            ("logit_bias", {}, {"logit_bias": torch.zeros(1)}, {}, True),
            ("grammar_mask", {}, {"grammar_mask": object()}, {}, True),
            ("top_k off", {}, {"need_top_k_sampling": False}, {}, True),
            ("top_k >= vocab", {"spec_info": types.SimpleNamespace(max_top_k=1 << 30)}, {}, {}, True),
            ("top_k 20", {"spec_info": types.SimpleNamespace(max_top_k=20)}, {}, {}, False),
            ("all greedy + top_k off", {}, {"is_all_greedy": True, "need_top_k_sampling": False}, {}, False),
        ]
        for cname, b_over, s_over, w_over, want_up in cases:
            b = dict(return_logprob=False, has_grammar=False, sampling_info=si_mock(**s_over))
            b.update(b_over)
            lo = types.SimpleNamespace(next_token_logits=x)
            km.verify_logits_guard(types.SimpleNamespace(**w_over), types.SimpleNamespace(**b), lo)
            y = lo.next_token_logits
            if want_up and not (y.dtype == torch.float32 and bits(y, x.float())):
                fails.append(f"verify guard '{cname}': not upcast exactly")
            if not want_up and y is not x:
                fails.append(f"verify guard '{cname}': upcast without a reason")
        notes.append(f"guard cases {len(cases)}")
    finally:
        if old_t20 is None:
            os.environ.pop("SGL_TOPK20", None)
        else:
            os.environ["SGL_TOPK20"] = old_t20
    summary = (f"topk20 {'installed (SGL_TOPK20 0/1 tested)' if topk20_on else 'not installed'}; "
               + "; ".join(notes))
    return (not fails), summary, fails


def do_check(root, compile_only, hidden, model):
    miss = status(root)
    if miss:
        print(f"{TAG} check: FAIL - not applied / incomplete: " + ", ".join(miss))
        return 1
    for f in FILES + [MODULE_DST]:
        p = os.path.join(root, f)
        err = compile_err(read(p), p)
        if err:
            print(f"{TAG} check: FAIL - does not compile: {err}")
            return 1
    if compile_only:
        print(f"{TAG} check: PASS (compile only: {len(EDITS)} hooks + module present and compile)")
        return 0
    probe = ("import importlib, os, sys\n"
             f"for m in {MODULES!r}:\n"
             "    importlib.import_module(m)\n"
             "k = sys.modules['sglang.srt.speculative.kopt_topkbf']\n"
             f"assert os.path.dirname(os.path.abspath(k.__file__)) == {os.path.join(root, 'srt', 'speculative')!r}, "
             "k.__file__\n")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, cwd="/", env=env)
    if r.returncode != 0:
        print(f"{TAG} check: FAIL - patched modules do not import: {r.stderr.strip()[-400:]}")
        return 2
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"{TAG} check: FAIL - torch not importable ({e})")
        return 3
    if not torch.cuda.is_available():
        print(f"{TAG} check: FAIL - no CUDA GPU for the bitwise self-test (use --compile-only for a static check)")
        return 4
    mh, mv = model_dims(model)
    hiddens = [hidden] + ([mh] if mh and mh != hidden else [])
    print(f"{TAG}: lm_head test hidden sizes {hiddens} (model {model or '-'}: hidden {mh}, vocab {mv})")
    sys.path.insert(0, os.path.dirname(root))
    try:
        ok, summary, fails = gpu_selftest(root, hiddens, V_model=mv)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"{TAG} check: FAIL - GPU self-test crashed: {type(e).__name__}: {e}")
        return 5
    for f in fails:
        print(f"  !! {f}")
    if not ok:
        print(f"{TAG} check: FAIL - {len(fails)} bitwise mismatch(es); {summary}")
        return 6
    print(f"{TAG} check: PASS ({len(EDITS)} hooks + module, patched modules import, GPU bitwise self-test OK on "
          f"{torch.cuda.get_device_name()}; {summary})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--model", default=os.environ.get("SGL_TOPKBF_MODEL", ""),
                    help="model dir or HF repo id (cached) to read hidden_size / vocab_size from for --check")
    a = ap.parse_args()
    root = os.path.abspath(a.root) if a.root else find_root()
    print(f"{TAG}: sglang package at {root}")
    if a.check:
        return do_check(root, a.compile_only, a.hidden, a.model)
    if a.revert:
        return do_revert(root)
    return do_apply(root)


if __name__ == "__main__":
    sys.exit(main())
