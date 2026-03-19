#!/usr/bin/env python3
"""
Ablation testing for GLM-ASR Triton kernel optimizations on H200 MIG 3g.71gb.

For each test:
1. Records exact code changes (diff from baseline)
2. Clears Triton cache
3. Runs benchmark_student.py
4. Captures individual run times, mean, stddev, accuracy
5. Restores original code

Results → ablation_results.json (structured) + ablation_results.md (readable)

Usage:
    conda activate mls && cd hw1-asr && python3 ablation_test.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import difflib

import torch


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_gpu_info():
    if not torch.cuda.is_available():
        return {"gpu": "none"}
    props = torch.cuda.get_device_properties(0)
    smem = getattr(props, 'shared_memory_per_block_optin',
                   getattr(props, 'max_shared_memory_per_block',
                           props.shared_memory_per_block))
    return {
        "gpu": torch.cuda.get_device_name(0),
        "sms": props.multi_processor_count,
        "vram_gb": round(props.total_memory / (1024**3), 1),
        "smem_kb": round(smem / 1024, 1),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "triton": None,  # filled below
    }


def get_diff(original, modified, filename):
    """Return unified diff between original and modified file content."""
    if original == modified:
        return "(no changes)"
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=1,
    )
    return "".join(diff).strip()


def run_benchmark():
    """Run benchmark_student.py and parse all results."""
    result = subprocess.run(
        [sys.executable, "benchmark_student.py", "glm_asr_triton_template"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "HF_HOME": os.environ.get("HF_HOME", "/home/s2884198/.cache/huggingface")}
    )
    output = result.stdout + result.stderr

    # Parse mean time and stddev
    time_ms = std_ms = None
    for line in output.split('\n'):
        if line.startswith('Time:'):
            m = re.search(r'([\d.]+)\s*ms\s*\(\+/-\s*([\d.]+)\s*ms\)', line)
            if m:
                time_ms, std_ms = float(m.group(1)), float(m.group(2))
            else:
                m = re.search(r'([\d.]+)\s*ms', line)
                if m:
                    time_ms = float(m.group(1))
                    std_ms = 0.0

    # Parse accuracy
    accuracy = None
    for line in output.split('\n'):
        if 'Accuracy:' in line:
            m = re.search(r'([\d.]+)\s*%', line)
            if m:
                accuracy = float(m.group(1))

    # Parse individual run times
    individual_runs = []
    for line in output.split('\n'):
        m = re.match(r'\s*Run\s+\d+:\s*([\d.]+)\s*ms', line)
        if m:
            individual_runs.append(float(m.group(1)))

    # Parse tokens
    tokens = None
    for line in output.split('\n'):
        if 'Tokens:' in line:
            m = re.search(r'(\d+)', line.split('Tokens:')[1])
            if m:
                tokens = int(m.group(1))

    return {
        "time_ms": time_ms,
        "std_ms": std_ms,
        "accuracy": accuracy,
        "individual_runs": individual_runs,
        "tokens": tokens,
        "raw_output": output[-1500:],  # last 1500 chars for debugging
    }


def clear_triton_cache():
    for d in [os.path.expanduser("~/.triton/cache"), "/tmp/triton_cache"]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)


def apply_changes(files_original, changes_spec):
    """Apply changes and return dict of diffs."""
    diffs = {}

    for filepath, replacements in changes_spec.items():
        content = files_original[filepath]
        for old, new in replacements:
            content = content.replace(old, new, 1)
        with open(filepath, 'w') as f:
            f.write(content)
        diff = get_diff(files_original[filepath], content, os.path.basename(filepath))
        if diff != "(no changes)":
            diffs[os.path.basename(filepath)] = diff

    return diffs


def restore_all(files_original):
    for filepath, content in files_original.items():
        with open(filepath, 'w') as f:
            f.write(content)


# ── File paths ───────────────────────────────────────────────────────────────

INIT = "glm_asr_triton_template/__init__.py"
LAYERS = "glm_asr_triton_template/layers.py"
ROPE = "glm_asr_triton_template/rope.py"
ATTN = "glm_asr_triton_template/attention.py"


# ── Test Definitions ─────────────────────────────────────────────────────────
# Each test: { name, desc, changes: { filepath: [(old, new), ...] } }

TESTS = [
    # ── Reference ──
    {
        "name": "baseline_optimized",
        "desc": "Current optimized config (all optimizations enabled)",
        "changes": {},
    },

    # ── Precision ──
    {
        "name": "precision_fp32",
        "desc": "Full fp32 pipeline (disable half-precision)",
        "changes": {
            INIT: [('layers.Linear.BF16 = True', 'layers.Linear.BF16 = False')],
        },
    },

    # ── Fusion ──
    {
        "name": "fusion_off_mlp",
        "desc": "Disable fused SwiGLU and LinearGELU kernels",
        "changes": {
            INIT: [
                ('layers.MLP.FUSED = True', 'layers.MLP.FUSED = False'),
                ('layers.EncoderMLP.FUSED = True', 'layers.EncoderMLP.FUSED = False'),
            ],
        },
    },
    {
        "name": "fusion_off_rope",
        "desc": "Disable fused Q+K RoPE (separate kernel launches)",
        "changes": {
            ROPE: [('if q.is_cuda:', 'if False and q.is_cuda:')],
        },
    },

    # ── Backend ──
    {
        "name": "backend_triton_matmul",
        "desc": "Use Triton matmul instead of cuBLAS for Linear layers",
        "changes": {
            INIT: [('layers.Linear.BACKEND = "torch"', 'layers.Linear.BACKEND = "triton"')],
        },
    },

    # ── SDPA fallback ──
    {
        "name": "sdpa_fallback_off",
        "desc": "Disable SDPA fallback (always custom flash attention)",
        "changes": {
            ATTN: [('if q.is_cuda and seq_q <= 4:', 'if False:')],
        },
    },
    {
        "name": "sdpa_threshold_1",
        "desc": "SDPA fallback only for seq_q <= 1",
        "changes": {
            ATTN: [('seq_q <= 4:', 'seq_q <= 1:')],
        },
    },
    {
        "name": "sdpa_threshold_8",
        "desc": "SDPA fallback for seq_q <= 8",
        "changes": {
            ATTN: [('seq_q <= 4:', 'seq_q <= 8:')],
        },
    },
    {
        "name": "sdpa_threshold_16",
        "desc": "SDPA fallback for seq_q <= 16",
        "changes": {
            ATTN: [('seq_q <= 4:', 'seq_q <= 16:')],
        },
    },

    # ── Attention tiles: encoder (head_dim=64), H200 baseline is (128,128,2,8) ──
    {
        "name": "attn_enc_64x64_s1_w4",
        "desc": "Encoder attention 64x64, nstages=1, nwarps=4 (consumer config on H200)",
        "changes": {
            LAYERS: [('64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages',
                       '64:  (64, 64, 1, 4),  # ABLATION: consumer-sized tiles')],
        },
    },
    {
        "name": "attn_enc_128x64_s2_w8",
        "desc": "Encoder attention 128x64, nstages=2, nwarps=8",
        "changes": {
            LAYERS: [('64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages',
                       '64:  (128, 64, 2, 8),  # ABLATION: asymmetric tiles')],
        },
    },
    {
        "name": "attn_enc_64x32_s1_w4",
        "desc": "Encoder attention 64x32, nstages=1, nwarps=4 (small tiles)",
        "changes": {
            LAYERS: [('64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages',
                       '64:  (64, 32, 1, 4),  # ABLATION: small tiles')],
        },
    },

    # ── Attention tiles: decoder (head_dim=128), H200 baseline is (128,64,2,8) ──
    {
        "name": "attn_dec_64x64_s2_w8",
        "desc": "Decoder attention 64x64, nstages=2, nwarps=8",
        "changes": {
            LAYERS: [('128: (128, 64, 2, 8),',
                       '128: (64, 64, 2, 8),  # ABLATION')],
        },
    },
    {
        "name": "attn_dec_32x32_s1_w4",
        "desc": "Decoder attention 32x32, nstages=1, nwarps=4 (consumer config)",
        "changes": {
            LAYERS: [('128: (128, 64, 2, 8),',
                       '128: (32, 32, 1, 4),  # ABLATION')],
        },
    },
    {
        "name": "attn_dec_64x32_s2_w8",
        "desc": "Decoder attention 64x32, nstages=2, nwarps=8",
        "changes": {
            LAYERS: [('128: (128, 64, 2, 8),',
                       '128: (64, 32, 2, 8),  # ABLATION')],
        },
    },

    # ── num_stages ablation (encoder attn, 128x128 tiles, nwarps=8) ──
    {
        "name": "enc_nstages_1",
        "desc": "Encoder attention nstages=1 (disable double-buffering)",
        "changes": {
            LAYERS: [('64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages',
                       '64:  (128, 128, 1, 8),  # ABLATION: nstages=1')],
        },
    },
    {
        "name": "enc_nstages_3",
        "desc": "Encoder attention nstages=3 (triple-buffering)",
        "changes": {
            LAYERS: [('64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages',
                       '64:  (128, 128, 3, 8),  # ABLATION: nstages=3')],
        },
    },

    # ── num_warps ablation (encoder attn, 128x128 tiles, nstages=2) ──
    {
        "name": "enc_nwarps_4",
        "desc": "Encoder attention nwarps=4 (128 threads/block)",
        "changes": {
            LAYERS: [('64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages',
                       '64:  (128, 128, 2, 4),  # ABLATION: nwarps=4')],
        },
    },
    {
        "name": "enc_nwarps_16",
        "desc": "Encoder attention nwarps=16 (512 threads/block)",
        "changes": {
            LAYERS: [('64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages',
                       '64:  (128, 128, 2, 16),  # ABLATION: nwarps=16')],
        },
    },

    # ── Matmul tile sizes (H200 baseline: 128x128x64) ──
    {
        "name": "matmul_64x64x32",
        "desc": "Matmul tiles 64x64x32 (consumer config on H200)",
        "changes": {
            LAYERS: [('"matmul_tiles": (128, 128, 64),',
                       '"matmul_tiles": (64, 64, 32),  # ABLATION')],
        },
    },
    {
        "name": "matmul_128x64x32",
        "desc": "Matmul tiles 128x64x32 (asymmetric, smaller TILE_K)",
        "changes": {
            LAYERS: [('"matmul_tiles": (128, 128, 64),',
                       '"matmul_tiles": (128, 64, 32),  # ABLATION')],
        },
    },
    {
        "name": "matmul_128x128x32",
        "desc": "Matmul tiles 128x128x32 (same spatial, smaller TILE_K)",
        "changes": {
            LAYERS: [('"matmul_tiles": (128, 128, 64),',
                       '"matmul_tiles": (128, 128, 32),  # ABLATION')],
        },
    },
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GLM-ASR ABLATION TESTING — H200 MIG 3g.71gb")
    print("=" * 70)

    gpu_info = get_gpu_info()
    try:
        import triton
        gpu_info["triton"] = triton.__version__
    except:
        pass

    print(f"GPU: {gpu_info.get('gpu')}")
    print(f"SMs: {gpu_info.get('sms')}, VRAM: {gpu_info.get('vram_gb')}GB, "
          f"Shared mem: {gpu_info.get('smem_kb')}KB")
    print(f"PyTorch: {gpu_info.get('pytorch')}, CUDA: {gpu_info.get('cuda')}, "
          f"Triton: {gpu_info.get('triton')}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tests to run: {len(TESTS)}")
    print()

    # Read originals
    files_original = {}
    for fp in [INIT, LAYERS, ROPE, ATTN]:
        with open(fp, 'r') as f:
            files_original[fp] = f.read()

    results = []

    for i, test in enumerate(TESTS):
        name = test["name"]
        desc = test["desc"]
        changes = test.get("changes", {})

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(TESTS)}] {name}")
        print(f"  {desc}")
        print(f"{'='*70}")

        # Restore originals first
        restore_all(files_original)

        # Apply changes and record diffs
        diffs = apply_changes(files_original, changes)

        if diffs:
            print(f"  Code changes:")
            for fname, diff in diffs.items():
                for line in diff.split('\n'):
                    if line.startswith('+') and not line.startswith('+++'):
                        print(f"    {line}")
                    elif line.startswith('-') and not line.startswith('---'):
                        print(f"    {line}")
        else:
            print(f"  Code changes: (none — baseline)")

        # Clear cache and run
        clear_triton_cache()
        print(f"  Running benchmark...")

        try:
            bench = run_benchmark()
            time_ms = bench["time_ms"]
            std_ms = bench["std_ms"]
            accuracy = bench["accuracy"]
            runs = bench["individual_runs"]

            if time_ms is not None:
                print(f"  Result: {time_ms:.1f}ms (+/- {std_ms:.1f}ms)")
                print(f"  Individual runs: {', '.join(f'{r:.1f}' for r in runs)}ms")
                print(f"  Accuracy: {accuracy}%")
                status = "PASS" if accuracy == 100.0 else "FAIL"
            else:
                print(f"  FAILED TO PARSE OUTPUT")
                print(f"  Last 500 chars: {bench['raw_output'][-500:]}")
                status = "ERROR"

            result = {
                "name": name,
                "desc": desc,
                "time_ms": time_ms,
                "std_ms": std_ms,
                "accuracy": accuracy,
                "individual_runs": runs,
                "tokens": bench.get("tokens"),
                "status": status,
                "code_diffs": diffs,
            }

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            result = {
                "name": name,
                "desc": desc,
                "time_ms": None,
                "std_ms": None,
                "accuracy": None,
                "individual_runs": [],
                "tokens": None,
                "status": "ERROR",
                "error": str(e),
                "code_diffs": diffs,
            }

        results.append(result)

    # Restore originals
    restore_all(files_original)
    print("\n\nOriginal files restored.")

    # ── Save JSON ──
    output = {
        "gpu_info": gpu_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_tests": len(TESTS),
        "results": results,
    }
    with open("ablation_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    # ── Generate Markdown ──
    baseline_ms = None
    for r in results:
        if r["name"] == "baseline_optimized" and r["time_ms"]:
            baseline_ms = r["time_ms"]

    md = []
    md.append("# Ablation Test Results\n")
    md.append(f"**GPU:** {gpu_info.get('gpu')}")
    md.append(f"**SMs:** {gpu_info.get('sms')}, **VRAM:** {gpu_info.get('vram_gb')}GB, "
              f"**Shared mem:** {gpu_info.get('smem_kb')}KB")
    md.append(f"**Stack:** PyTorch {gpu_info.get('pytorch')}, CUDA {gpu_info.get('cuda')}, "
              f"Triton {gpu_info.get('triton')}")
    md.append(f"**Date:** {output['timestamp']}")
    md.append(f"**Baseline:** {baseline_ms}ms" if baseline_ms else "")
    md.append("")

    md.append("## Summary Table\n")
    md.append(f"| # | Test | Time (ms) | Std | Delta | Accuracy | Status |")
    md.append(f"|---|------|-----------|-----|-------|----------|--------|")

    for i, r in enumerate(results):
        t = f"{r['time_ms']:.1f}" if r['time_ms'] else "ERR"
        s = f"{r['std_ms']:.1f}" if r['std_ms'] else "-"
        if r['time_ms'] and baseline_ms:
            delta = r['time_ms'] - baseline_ms
            d = f"{delta:+.1f}"
        else:
            d = "-"
        a = f"{r['accuracy']:.0f}%" if r['accuracy'] is not None else "?"
        md.append(f"| {i+1} | `{r['name']}` | {t} | {s} | {d} | {a} | {r.get('status','')} |")

    md.append("")

    # Detailed results per test
    md.append("## Detailed Results\n")
    for i, r in enumerate(results):
        md.append(f"### {i+1}. `{r['name']}`\n")
        md.append(f"**Description:** {r['desc']}\n")

        if r.get("code_diffs"):
            md.append("**Code changes:**")
            md.append("```diff")
            for fname, diff in r["code_diffs"].items():
                md.append(diff)
            md.append("```")
        else:
            md.append("**Code changes:** (none — baseline config)")

        md.append("")
        if r["time_ms"]:
            md.append(f"- **Mean:** {r['time_ms']:.1f}ms (+/- {r['std_ms']:.1f}ms)")
            if r["individual_runs"]:
                runs_str = ", ".join(f"{x:.1f}" for x in r["individual_runs"])
                md.append(f"- **Individual runs:** {runs_str}ms")
            if baseline_ms and r["time_ms"]:
                delta = r["time_ms"] - baseline_ms
                md.append(f"- **Delta from baseline:** {delta:+.1f}ms")
            md.append(f"- **Accuracy:** {r['accuracy']}%")
            md.append(f"- **Tokens:** {r.get('tokens', '?')}")
        else:
            md.append(f"- **Status:** ERROR")
            if r.get("error"):
                md.append(f"- **Error:** {r['error']}")

        md.append("")

    with open("ablation_results.md", 'w') as f:
        f.write("\n".join(md))

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'#':<3} {'Test':<30} {'Time':>8} {'Std':>6} {'Delta':>8} {'Acc':>6}")
    print("-" * 65)
    for i, r in enumerate(results):
        t = f"{r['time_ms']:.1f}" if r['time_ms'] else "ERR"
        s = f"{r['std_ms']:.1f}" if r['std_ms'] else "-"
        if r['time_ms'] and baseline_ms:
            d = f"{r['time_ms'] - baseline_ms:+.1f}"
        else:
            d = "-"
        a = f"{r['accuracy']:.0f}%" if r['accuracy'] is not None else "?"
        print(f"{i+1:<3} {r['name']:<30} {t:>8} {s:>6} {d:>8} {a:>6}")

    print(f"\nSaved: ablation_results.json, ablation_results.md")


if __name__ == "__main__":
    main()
