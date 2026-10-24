"""Apply the A1 patch (zero-copy GDN verify commit, see REPORT.txt / a1_ssm_ptr.py) to the sglang package that
`import sglang` resolves to (importlib.util.find_spec, so a PYTHONPATH overlay copy works too), in place, idempotent.

  python apply_a1.py                 patch (exit 0; already-patched edits are skipped)
  python apply_a1.py --check         exit 0 only if every edit + the module are present AND the patched modules import
  python apply_a1.py --check --compile-only    same, but only py_compile the patched files (no sglang deps needed)
  python apply_a1.py --revert        restore the *.gdnk_a1.orig backups and remove the module
  python apply_a1.py --root DIR      patch the package at DIR (= the directory that contains srt/ and kernels/)
  python apply_a1.py --root SRC --export OUT    write patched copies under OUT (SRC untouched) - used to make a1.diff

Files touched (relative to the sglang package dir):
  srt/mem_cache/gdnk_a1.py                              NEW: copy of a1_ssm_ptr.py (own code; no sglang imports)
  srt/mem_cache/memory_pool.py                          combined allocation; pointer reset/materialize in clear_slots,
                                                        copy_from, get_cpu_copy, load_cpu_copy
  srt/layers/attention/linear/gdn_backend.py            verify reads h0 via the pointer + scratch rows = req_pool_idx;
                                                        extend/decode materialize at the first GDN layer
  srt/layers/attention/hybrid_linear_attn_backend.py    commit = pointer + conv copy (+ track); fused all-layer decode
                                                        track disabled
The switch SGLANG_A1_SSM_PTR=0 turns the patched code back into today's behaviour (no rebuild needed).
Nothing is written inside this directory at run time; the patched sglang imports only sglang.srt.mem_cache.gdnk_a1.
"""
import argparse
import importlib
import importlib.util
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_SRC = os.path.join(HERE, "a1_ssm_ptr.py")
MODULE_DST = "srt/mem_cache/gdnk_a1.py"
BACKUP = ".gdnk_a1.orig"
IMP = "from sglang.srt.mem_cache import gdnk_a1 as _A1  # gdnk-a1:import\n"

MP = "srt/mem_cache/memory_pool.py"
GB = "srt/layers/attention/linear/gdn_backend.py"
HB = "srt/layers/attention/hybrid_linear_attn_backend.py"

# (file, tag, mode, anchor, text)   mode: after = insert text after anchor, before = before it, replace = replace it
EDITS = [
    (MP, "import", "after", "from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator\n", IMP),
    (MP, "temporal", "replace",
     "                temporal_state = torch.zeros(\n"
     "                    size=(num_mamba_layers, size + 1) + temporal_state_shape,\n",
     "                # gdnk-a1:temporal  (A1: SSM pool + verify scratch share one allocation)\n"
     "                temporal_state = _A1.alloc_temporal(self, locals()) if _A1.want(locals()) else torch.zeros(\n"
     "                    size=(num_mamba_layers, size + 1) + temporal_state_shape,\n"),
    (MP, "intermediate", "replace",
     "                    intermediate_ssm_state_cache = torch.zeros(\n",
     "                    # gdnk-a1:intermediate\n"
     "                    intermediate_ssm_state_cache = _A1.take_intermediate(self) if _A1.has(self) else torch.zeros(\n"),
    (MP, "clear", "after",
     '        """Zero out mamba state at the given pool indices. Must run on forward stream."""\n',
     "        _A1.on_reset(self, indices)  # gdnk-a1:clear\n"),
    (MP, "copy", "after",
     "        (the copied checkpoint has no pending ring entries).\n        \"\"\"\n",
     "        _A1.on_copy(self, src_indices, dst_indices)  # gdnk-a1:copy\n"),
    (MP, "cpucopy", "replace",
     "    def get_cpu_copy(self, indices):\n        current_platform.synchronize()\n",
     "    def get_cpu_copy(self, indices):\n"
     "        _A1.materialize_pool(self, indices)  # gdnk-a1:cpucopy\n"
     "        current_platform.synchronize()\n"),
    (MP, "cpuload", "after", "    def load_cpu_copy(self, mamba_cache_cpu, indices):\n",
     "        _A1.on_reset(self, indices)  # gdnk-a1:cpuload\n"),
    (GB, "import", "after", "from sglang.srt.mem_cache.memory_pool import MambaPool\n", IMP),
    (GB, "decode", "after", "        cache_indices = self.forward_metadata.mamba_cache_indices\n",
     "        _A1.materialize_for_forward(self, layer, forward_batch, cache_indices)  # gdnk-a1:decode\n"),
    (GB, "verify", "after", "            intermediate_state_indices = self.verify_intermediate_state_indices\n",
     "            _a1_ssm, _a1_idx, intermediate_state_indices = _A1.verify_args(  # gdnk-a1:verify\n"
     "                self, layer, forward_batch, ssm_states, cache_indices, intermediate_state_indices\n"
     "            )\n"),
    (GB, "extend", "after", "            has_initial_states = forward_batch.extend_prefix_lens > 0\n",
     "            _A1.materialize_for_forward(self, layer, forward_batch, cache_indices)  # gdnk-a1:extend\n"),
    (GB, "kwargs", "replace",
     "                    ssm_states=ssm_states,\n"
     "                    cache_indices=cache_indices,\n"
     "                    query_start_loc=query_start_loc,\n"
     "                    intermediate_states_buffer=intermediate_state_cache,\n",
     "                    ssm_states=_a1_ssm,  # gdnk-a1:kwargs\n"
     "                    cache_indices=_a1_idx,\n"
     "                    query_start_loc=query_start_loc,\n"
     "                    intermediate_states_buffer=intermediate_state_cache,\n"),
    (HB, "import", "after", "from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool\n", IMP),
    (HB, "rows", "replace", "        del req_pool_indices\n",
     "        _a1_rows = req_pool_indices  # gdnk-a1:rows\n        del req_pool_indices\n"),
    (HB, "commit", "before",
     "        scatter_mamba_states_after_mtp_verify(\n            mamba_caches,\n            state_indices_tensor,\n",
     "        if _A1.commit(  # gdnk-a1:commit\n"
     "            self.linear_attn_backend, mamba_caches, state_indices_tensor, last_correct_step_indices,\n"
     "            mamba_track_indices, mamba_steps_to_track, _a1_rows,\n"
     "        ):\n"
     "            return\n"),
    (HB, "trackpools", "after",
     "        of the last mamba layer, for the fused all-layers track launch. None if\n"
     "        the pool shape is not the expected layout.\"\"\"\n",
     "        if _A1.disable_all_layer_track(self):  # gdnk-a1:trackpools\n            return None\n"),
]
FILES = [MP, GB, HB]
MODULES = ["sglang.srt.mem_cache.gdnk_a1", "sglang.srt.mem_cache.memory_pool",
           "sglang.srt.layers.attention.hybrid_linear_attn_backend", "sglang.srt.layers.attention.linear.gdn_backend"]


def find_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("apply_a1: `import sglang` does not resolve (set PYTHONPATH or use --root)")
    return list(spec.submodule_search_locations)[0]


def _compile(path):
    compile(open(path).read(), path, "exec")      # syntax check without writing .pyc files


def marker(tag):
    return f"gdnk-a1:{tag}"


def patch_text(rel, src):
    """-> (patched text, list of (tag, status))"""
    out = []
    for f, tag, mode, anchor, text in EDITS:
        if f != rel:
            continue
        if marker(tag) in src:
            out.append((tag, "already"))
            continue
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"apply_a1: {rel} edit '{tag}': anchor found {n} times (SGLang version differs?)")
        if mode == "after":
            src = src.replace(anchor, anchor + text)
        elif mode == "before":
            src = src.replace(anchor, text + anchor)
        else:
            src = src.replace(anchor, text)
        out.append((tag, "applied"))
    return src, out


def status(root):
    missing = []
    for f, tag, *_ in EDITS:
        p = os.path.join(root, f)
        if not os.path.exists(p) or marker(tag) not in open(p).read():
            missing.append(f"{f}:{tag}")
    dst = os.path.join(root, MODULE_DST)
    if not os.path.exists(dst) or open(dst).read() != open(MODULE_SRC).read():
        missing.append(MODULE_DST + " (missing or differs from a1_ssm_ptr.py)")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--export")
    a = ap.parse_args()
    root = os.path.abspath(a.root) if a.root else find_root()
    print(f"apply_a1: sglang package at {root}")

    if a.check:
        miss = status(root)
        if miss:
            print("apply_a1 --check: NOT applied / incomplete: " + ", ".join(miss))
            return 1
        for f in FILES + [MODULE_DST]:
            _compile(os.path.join(root, f))
        if not a.compile_only:
            sys.path.insert(0, os.path.dirname(root))
            for m in MODULES:
                try:
                    importlib.import_module(m)
                except Exception as e:  # noqa: BLE001
                    print(f"apply_a1 --check: patched module {m} does not import: {type(e).__name__}: {e}")
                    return 2
            mod = importlib.import_module("sglang.srt.mem_cache.gdnk_a1")
            if os.path.dirname(os.path.abspath(mod.__file__)) != os.path.join(root, "srt", "mem_cache"):
                print(f"apply_a1 --check: gdnk_a1 imported from {mod.__file__}, not from {root}")
                return 3
        print(f"apply_a1 --check: OK ({len(EDITS)} edits + module present"
              f"{', compile only' if a.compile_only else ', patched modules import'}; "
              f"switch SGLANG_A1_SSM_PTR={os.environ.get('SGLANG_A1_SSM_PTR', '1 (default on)')})")
        return 0

    if a.revert:
        for f in FILES:
            p = os.path.join(root, f)
            if os.path.exists(p + BACKUP):
                shutil.copyfile(p + BACKUP, p)
                os.remove(p + BACKUP)
                print(f"  restored {f}")
        dst = os.path.join(root, MODULE_DST)
        if os.path.exists(dst):
            os.remove(dst)
            print(f"  removed {MODULE_DST}")
        return 0

    out_root = os.path.abspath(a.export) if a.export else root
    for f in FILES:
        p = os.path.join(root, f)
        src = open(p).read()
        new, st = patch_text(f, src)
        q = os.path.join(out_root, f)
        os.makedirs(os.path.dirname(q), exist_ok=True)
        if new != src or a.export:
            if not a.export and not os.path.exists(p + BACKUP):
                shutil.copyfile(p, p + BACKUP)
            with open(q, "w") as fh:
                fh.write(new)
            _compile(q)
        print(f"  {f}: " + ", ".join(f"{t} {s}" for t, s in st))
    q = os.path.join(out_root, MODULE_DST)
    os.makedirs(os.path.dirname(q), exist_ok=True)
    shutil.copyfile(MODULE_SRC, q)
    print(f"  {MODULE_DST}: installed from {MODULE_SRC}")
    print("apply_a1: done" + (f" (exported to {out_root})" if a.export else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
