# Benchmark Results: GLM-ASR Triton Kernel Implementation

Full benchmark results from the Edinburgh teaching cluster.

---

## Environment

| Parameter | Value |
|-----------|-------|
| **Date** | Mon 16 Mar 21:20 GMT 2026 |
| **Hostname** | saxa.inf.ed.ac.uk |
| **User** | s2884198 |
| **GPU** | NVIDIA H200 MIG 3g.71gb |
| **GPU Architecture** | Hopper (sm_9.0) |
| **VRAM** | 69.8 GB |
| **SMs** | 60 |
| **MIG Mode** | Enabled (3g.71gb partition of full 143GB H200) |
| **NVIDIA Driver** | 580.126.09 |
| **CUDA Version** | 13.0 |
| **Python** | 3.11.15 |
| **PyTorch** | 2.10.0+cu130 |
| **Triton** | 3.6.0 |
| **NumPy** | 2.4.3 |
| **Transformers** | 5.3.0 |
| **Conda Env** | mls |
| **SLURM Command** | `srun -p Teaching -w saxa --gres gpu:3g.71gb:1 --mem=32G` |
| **Branch** | ankush (commit 288ad9c) |
| **Implementation** | glm_asr_triton_template |
| **Generate Function** | _generate_v8b (KV-cached, O(n)) |
| **Test Audio** | test_audio.wav (3.50s, 16kHz) |
| **Expected Output** | CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS |

---

## Student Benchmark (5 runs)

Each run: 2 warmup iterations + 5 timed iterations.

### Run 1

```
Benchmarking (5 runs)...
  Run 1: 202.5ms (13 tokens)
  Run 2: 202.1ms (13 tokens)
  Run 3: 206.5ms (13 tokens)
  Run 4: 202.8ms (13 tokens)
  Run 5: 203.2ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 203.4ms (+/- 1.6ms)
Tokens: 13
Speed: 15.65ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Run 2

```
Benchmarking (5 runs)...
  Run 1: 202.9ms (13 tokens)
  Run 2: 203.0ms (13 tokens)
  Run 3: 203.0ms (13 tokens)
  Run 4: 203.7ms (13 tokens)
  Run 5: 207.1ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 204.0ms (+/- 1.6ms)
Tokens: 13
Speed: 15.69ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Run 3

```
Benchmarking (5 runs)...
  Run 1: 202.6ms (13 tokens)
  Run 2: 207.5ms (13 tokens)
  Run 3: 202.8ms (13 tokens)
  Run 4: 202.3ms (13 tokens)
  Run 5: 201.8ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 203.4ms (+/- 2.1ms)
Tokens: 13
Speed: 15.64ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Run 4

```
Benchmarking (5 runs)...
  Run 1: 208.8ms (13 tokens)
  Run 2: 212.7ms (13 tokens)
  Run 3: 208.6ms (13 tokens)
  Run 4: 208.3ms (13 tokens)
  Run 5: 208.5ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 209.4ms (+/- 1.7ms)
Tokens: 13
Speed: 16.11ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Run 5

```
Benchmarking (5 runs)...
  Run 1: 203.0ms (13 tokens)
  Run 2: 202.9ms (13 tokens)
  Run 3: 203.3ms (13 tokens)
  Run 4: 206.7ms (13 tokens)
  Run 5: 203.4ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 203.9ms (+/- 1.4ms)
Tokens: 13
Speed: 15.68ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Student Benchmark Summary

| Run | Time (ms) | Stddev (ms) | Speed (ms/tok) | Tokens | Accuracy | Status |
|-----|-----------|-------------|----------------|--------|----------|--------|
| 1 | 203.4 | 1.6 | 15.65 | 13 | 100% | PASS |
| 2 | 204.0 | 1.6 | 15.69 | 13 | 100% | PASS |
| 3 | 203.4 | 2.1 | 15.64 | 13 | 100% | PASS |
| 4 | 209.4 | 1.7 | 16.11 | 13 | 100% | PASS |
| 5 | 203.9 | 1.4 | 15.68 | 13 | 100% | PASS |
| **Average** | **204.8** | **1.7** | **15.75** | **13** | **100%** | **PASS** |

All 5 runs: 100% transcription accuracy, all PASS.

---

## Detailed Benchmark (5 runs)

Per-component profiling with 50 generated tokens (stock generate path, no KV cache).

> Note: this section is the original 2026-03-16 detailed benchmark record and
> includes first-use warmup effects. For the corrected warmup-controlled H200
> component benchmark used in the report, see [benchmarks_detailed.md](./benchmarks_detailed.md).

### Run 1

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           405.86ms       26.1%
Multi-modal Projector                     8.04ms        0.5%
Decoder (Prefill)                       294.75ms       19.0%
Decoder (50 decode steps)               844.94ms       54.4%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)        1553.58ms
```

Individual decoder layers:
- Layer 0: 1.40ms (+/- 1.15ms)
- Layer 1: 0.49ms (+/- 0.03ms)
- Layer 2: 0.48ms (+/- 0.02ms)
- Layer 3: 0.48ms (+/- 0.02ms)
- Layer 4: 0.47ms (+/- 0.01ms)

### Run 2

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           313.94ms       24.7%
Multi-modal Projector                     7.70ms        0.6%
Decoder (Prefill)                       291.28ms       23.0%
Decoder (50 decode steps)               655.99ms       51.7%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)        1268.91ms
```

Individual decoder layers:
- Layer 0: 0.67ms (+/- 0.11ms)
- Layer 1: 0.52ms (+/- 0.03ms)
- Layer 2: 0.51ms (+/- 0.03ms)
- Layer 3: 0.51ms (+/- 0.02ms)
- Layer 4: 0.49ms (+/- 0.02ms)

### Run 3

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           318.72ms       25.4%
Multi-modal Projector                     6.95ms        0.6%
Decoder (Prefill)                       290.75ms       23.2%
Decoder (50 decode steps)               639.13ms       50.9%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)        1255.55ms
```

Individual decoder layers:
- Layer 0: 0.66ms (+/- 0.10ms)
- Layer 1: 0.49ms (+/- 0.02ms)
- Layer 2: 0.50ms (+/- 0.02ms)
- Layer 3: 0.49ms (+/- 0.02ms)
- Layer 4: 0.50ms (+/- 0.02ms)

### Run 4

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           313.11ms       25.3%
Multi-modal Projector                     6.91ms        0.6%
Decoder (Prefill)                       292.63ms       23.6%
Decoder (50 decode steps)               625.00ms       50.5%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)        1237.66ms
```

Individual decoder layers:
- Layer 0: 0.65ms (+/- 0.12ms)
- Layer 1: 0.50ms (+/- 0.03ms)
- Layer 2: 0.49ms (+/- 0.03ms)
- Layer 3: 0.48ms (+/- 0.01ms)
- Layer 4: 0.48ms (+/- 0.02ms)

### Run 5

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           313.55ms       25.0%
Multi-modal Projector                     7.25ms        0.6%
Decoder (Prefill)                       303.87ms       24.2%
Decoder (50 decode steps)               628.56ms       50.2%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)        1253.23ms
```

Individual decoder layers:
- Layer 0: 0.65ms (+/- 0.11ms)
- Layer 1: 0.48ms (+/- 0.01ms)
- Layer 2: 0.47ms (+/- 0.02ms)
- Layer 3: 0.47ms (+/- 0.01ms)
- Layer 4: 0.49ms (+/- 0.03ms)

### Detailed Benchmark Summary

| Run | Encoder (ms) | Projector (ms) | Prefill (ms) | Decode 50 (ms) | Total (ms) |
|-----|-------------|---------------|-------------|---------------|-----------|
| 1 | 405.86 | 8.04 | 294.75 | 844.94 | 1553.58 |
| 2 | 313.94 | 7.70 | 291.28 | 655.99 | 1268.91 |
| 3 | 318.72 | 6.95 | 290.75 | 639.13 | 1255.55 |
| 4 | 313.11 | 6.91 | 292.63 | 625.00 | 1237.66 |
| 5 | 313.55 | 7.25 | 303.87 | 628.56 | 1253.23 |
| **Avg (runs 2-5)** | **314.83** | **7.20** | **294.63** | **637.17** | **1253.84** |

Note: Run 1 is slower due to Triton kernel compilation (first invocation). Runs 2-5 are
representative of steady-state performance. The high stddev reported per-component is also
due to the first warmup iteration including compilation time.

### Component Breakdown (Runs 2-5 average)

| Component | Time (ms) | % of Total |
|-----------|-----------|-----------|
| Audio Encoder (32 layers) | 314.83 | 25.1% |
| Multi-modal Projector | 7.20 | 0.6% |
| Decoder Prefill (28 layers) | 294.63 | 23.5% |
| Decoder Decode (50 steps) | 637.17 | 50.8% |
| **TOTAL** | **1253.84** | **100%** |

Average decode step: ~12.74ms (runs 2-5 average).
Average decoder layer: ~0.49ms per layer per step.

---

## Baseline Benchmark (glm_asr_triton_example)

```
Benchmarking (5 runs)...
  Run 1: 464.5ms (13 tokens)
  Run 2: 464.8ms (13 tokens)
  Run 3: 463.6ms (13 tokens)
  Run 4: 463.9ms (13 tokens)
  Run 5: 463.9ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 464.1ms (+/- 0.4ms)
Tokens: 13
Speed: 35.70ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

---

## Comparison: Our Template vs Baseline

| Metric | Our Template | Baseline (example) | Improvement |
|--------|-------------|-------------------|-------------|
| **Time** | **204.8ms** | **464.1ms** | **55.9% faster** |
| **Speed** | 15.75 ms/tok | 35.70 ms/tok | 2.27x |
| **Tokens** | 13 | 13 | — |
| **Accuracy** | 100% | 100% | — |
| **Generate Function** | _generate_v8b (KV cache) | generate (stock O(n²)) | — |

---

## Cross-GPU Comparison

| GPU | SMs | VRAM | Our Time | Baseline | Speedup |
|-----|-----|------|----------|----------|---------|
| RTX 5090 (full) | 170 | 32 GB | 98.5ms | 261.3ms | 62.3% |
| H200 MIG 3g.71gb | 60 | 70 GB | 204.8ms | 464.1ms | 55.9% |
| H200 MIG 1g.18gb | 16 | 16 GB | 309.7ms | — | — |

Performance scales roughly with SM count. The H200 MIG 3g.71gb has ~35% of the
RTX 5090's SM count and achieves ~48% of its throughput, suggesting memory bandwidth
(HBM3e) partially compensates for fewer SMs.

---

## Notes

- All student benchmark runs use 2 warmup + 5 timed iterations
- Detailed benchmark uses stock `generate()` path (no KV cache, 50 tokens) for profiling
- Student benchmark uses `_generate_v8b` (KV-cached, 13 tokens for test audio)
- GPUProfile detected Hopper (sm_90) and applied datacenter tile configs:
  - Attention: 128×128 (encoder), 128×64 (decoder), nstages=2, nwarps=8
  - Matmul: 128×128×64
  - RoPE: nstages=2, nwarps=8
- Run 4 of student benchmark showed higher latency (~209ms) — likely MIG partition contention
- Raw output saved to `benchmark_raw_output.txt`
