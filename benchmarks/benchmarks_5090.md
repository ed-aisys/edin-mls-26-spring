# Benchmark Results: GLM-ASR Triton Kernel Implementation (RTX 5090)

Full benchmark results from a RunPod instance with RTX 5090.

---

## Environment

| Parameter | Value |
|-----------|-------|
| **Date** | Mon 17 Mar 11:46 UTC 2026 |
| **Hostname** | 5448230d8a1e (RunPod) |
| **User** | root |
| **GPU** | NVIDIA GeForce RTX 5090 |
| **GPU Architecture** | Blackwell (sm_12.0) |
| **VRAM** | 31.4 GB |
| **SMs** | 170 |
| **MIG Mode** | N/A (consumer GPU, no MIG support) |
| **NVIDIA Driver** | 580.126.20 |
| **CUDA Version** | 13.0 |
| **Python** | 3.12.3 |
| **PyTorch** | 2.10.0+cu130 |
| **Triton** | 3.6.0 |
| **NumPy** | 2.3.5 |
| **Transformers** | 5.3.0 |
| **Conda Env** | N/A (system Python) |
| **Branch** | ankush (commit 25b1fd9) |
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
  Run 1: 100.2ms (13 tokens)
  Run 2: 100.1ms (13 tokens)
  Run 3: 101.0ms (13 tokens)
  Run 4: 100.4ms (13 tokens)
  Run 5: 100.2ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 100.4ms (+/- 0.3ms)
Tokens: 13
Speed: 7.72ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Run 2

```
Benchmarking (5 runs)...
  Run 1: 100.7ms (13 tokens)
  Run 2: 100.4ms (13 tokens)
  Run 3: 100.5ms (13 tokens)
  Run 4: 100.5ms (13 tokens)
  Run 5: 100.3ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 100.5ms (+/- 0.1ms)
Tokens: 13
Speed: 7.73ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Run 3

```
Benchmarking (5 runs)...
  Run 1: 101.0ms (13 tokens)
  Run 2: 100.5ms (13 tokens)
  Run 3: 100.5ms (13 tokens)
  Run 4: 100.4ms (13 tokens)
  Run 5: 100.2ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 100.5ms (+/- 0.3ms)
Tokens: 13
Speed: 7.73ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Run 4

```
Benchmarking (5 runs)...
  Run 1: 101.4ms (13 tokens)
  Run 2: 100.3ms (13 tokens)
  Run 3: 100.3ms (13 tokens)
  Run 4: 100.1ms (13 tokens)
  Run 5: 100.0ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 100.4ms (+/- 0.5ms)
Tokens: 13
Speed: 7.72ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Run 5

```
Benchmarking (5 runs)...
  Run 1: 100.2ms (13 tokens)
  Run 2: 100.3ms (13 tokens)
  Run 3: 100.5ms (13 tokens)
  Run 4: 100.6ms (13 tokens)
  Run 5: 100.4ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 100.4ms (+/- 0.1ms)
Tokens: 13
Speed: 7.72ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

### Student Benchmark Summary

| Run | Time (ms) | Stddev (ms) | Speed (ms/tok) | Tokens | Accuracy | Status |
|-----|-----------|-------------|----------------|--------|----------|--------|
| 1 | 100.4 | 0.3 | 7.72 | 13 | 100% | PASS |
| 2 | 100.5 | 0.1 | 7.73 | 13 | 100% | PASS |
| 3 | 100.5 | 0.3 | 7.73 | 13 | 100% | PASS |
| 4 | 100.4 | 0.5 | 7.72 | 13 | 100% | PASS |
| 5 | 100.4 | 0.1 | 7.72 | 13 | 100% | PASS |
| **Average** | **100.4** | **0.3** | **7.72** | **13** | **100%** | **PASS** |

All 5 runs: 100% transcription accuracy, all PASS.

---

## Detailed Benchmark (5 runs)

Per-component profiling with 50 generated tokens (stock generate path, no KV cache).

### Run 1

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           201.72ms       29.1%
Multi-modal Projector                     3.89ms        0.6%
Decoder (Prefill)                       191.89ms       27.7%
Decoder (50 decode steps)               295.34ms       42.6%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)         692.84ms
```

Individual decoder layers:
- Layer 0: 0.36ms (+/- 0.07ms)
- Layer 1: 0.26ms (+/- 0.02ms)
- Layer 2: 0.25ms (+/- 0.01ms)
- Layer 3: 0.25ms (+/- 0.02ms)
- Layer 4: 0.24ms (+/- 0.01ms)

### Run 2

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           196.63ms       28.7%
Multi-modal Projector                     4.03ms        0.6%
Decoder (Prefill)                       191.37ms       27.9%
Decoder (50 decode steps)               292.81ms       42.8%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)         684.84ms
```

Individual decoder layers:
- Layer 0: 0.35ms (+/- 0.08ms)
- Layer 1: 0.25ms (+/- 0.02ms)
- Layer 2: 0.25ms (+/- 0.01ms)
- Layer 3: 0.24ms (+/- 0.01ms)
- Layer 4: 0.24ms (+/- 0.01ms)

### Run 3

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           194.19ms       28.5%
Multi-modal Projector                     3.83ms        0.6%
Decoder (Prefill)                       191.99ms       28.2%
Decoder (50 decode steps)               290.82ms       42.7%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)         680.84ms
```

Individual decoder layers:
- Layer 0: 0.35ms (+/- 0.07ms)
- Layer 1: 0.25ms (+/- 0.02ms)
- Layer 2: 0.24ms (+/- 0.02ms)
- Layer 3: 0.24ms (+/- 0.02ms)
- Layer 4: 0.24ms (+/- 0.01ms)

### Run 4

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           197.00ms       28.6%
Multi-modal Projector                     3.82ms        0.6%
Decoder (Prefill)                       192.43ms       28.0%
Decoder (50 decode steps)               295.10ms       42.9%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)         688.35ms
```

Individual decoder layers:
- Layer 0: 0.36ms (+/- 0.07ms)
- Layer 1: 0.25ms (+/- 0.02ms)
- Layer 2: 0.24ms (+/- 0.01ms)
- Layer 3: 0.24ms (+/- 0.02ms)
- Layer 4: 0.25ms (+/- 0.02ms)

### Run 5

```
======================================================================
PERFORMANCE SUMMARY
======================================================================

Component                              Time (ms)   % of Total
------------------------------------------------------------
Audio Encoder                           197.64ms       28.9%
Multi-modal Projector                     3.78ms        0.6%
Decoder (Prefill)                       185.87ms       27.2%
Decoder (50 decode steps)               295.93ms       43.3%
------------------------------------------------------------
TOTAL (estimated for 50 tokens)         683.22ms
```

Individual decoder layers:
- Layer 0: 0.35ms (+/- 0.07ms)
- Layer 1: 0.25ms (+/- 0.02ms)
- Layer 2: 0.24ms (+/- 0.02ms)
- Layer 3: 0.24ms (+/- 0.01ms)
- Layer 4: 0.24ms (+/- 0.01ms)

### Detailed Benchmark Summary

| Run | Encoder (ms) | Projector (ms) | Prefill (ms) | Decode 50 (ms) | Total (ms) |
|-----|-------------|---------------|-------------|---------------|-----------|
| 1 | 201.72 | 3.89 | 191.89 | 295.34 | 692.84 |
| 2 | 196.63 | 4.03 | 191.37 | 292.81 | 684.84 |
| 3 | 194.19 | 3.83 | 191.99 | 290.82 | 680.84 |
| 4 | 197.00 | 3.82 | 192.43 | 295.10 | 688.35 |
| 5 | 197.64 | 3.78 | 185.87 | 295.93 | 683.22 |
| **Average** | **197.44** | **3.87** | **190.71** | **294.00** | **686.02** |

Note: Run 1 includes Triton kernel compilation in the profiled warmup, resulting in slightly
higher encoder time. All runs are otherwise consistent.

### Component Breakdown (Average)

| Component | Time (ms) | % of Total |
|-----------|-----------|-----------|
| Audio Encoder (32 layers) | 197.44 | 28.8% |
| Multi-modal Projector | 3.87 | 0.6% |
| Decoder Prefill (28 layers) | 190.71 | 27.8% |
| Decoder Decode (50 steps) | 294.00 | 42.8% |
| **TOTAL** | **686.02** | **100%** |

Average decode step: ~5.88ms (all runs average).
Average decoder layer: ~0.25ms per layer per step.

---

## Baseline Benchmark (glm_asr_triton_example)

```
Benchmarking (5 runs)...
  Run 1: 262.2ms (13 tokens)
  Run 2: 262.1ms (13 tokens)
  Run 3: 262.3ms (13 tokens)
  Run 4: 262.2ms (13 tokens)
  Run 5: 262.2ms (13 tokens)

======================================================================
RESULTS
======================================================================
Time: 262.2ms (+/- 0.1ms)
Tokens: 13
Speed: 20.17ms/token

Transcription: Concord returned to its place amidst the tents.

Accuracy: 100.0%
Status: PASS
```

---

## Comparison: Our Template vs Baseline

| Metric | Our Template | Baseline (example) | Improvement |
|--------|-------------|-------------------|-------------|
| **Time** | **100.4ms** | **262.2ms** | **61.7% faster** |
| **Speed** | 7.72 ms/tok | 20.17 ms/tok | 2.61x |
| **Tokens** | 13 | 13 | — |
| **Accuracy** | 100% | 100% | — |
| **Generate Function** | _generate_v8b (KV cache) | generate (stock O(n²)) | — |

---

## Cross-GPU Comparison

| GPU | SMs | VRAM | Our Time | Baseline | Speedup |
|-----|-----|------|----------|----------|---------|
| RTX 5090 (full) | 170 | 32 GB | 100.4ms | 262.2ms | 61.7% |
| H200 MIG 3g.71gb | 60 | 70 GB | 204.8ms | 464.1ms | 55.9% |
| H200 MIG 1g.18gb | 16 | 16 GB | 309.7ms | — | — |

Performance scales roughly with SM count. The RTX 5090 has ~2.8x the SMs of the
H200 MIG 3g.71gb partition and achieves ~2.0x the throughput, with the difference
attributable to Blackwell's higher per-SM performance and GDDR7 bandwidth vs HBM3e.

---

## Notes

- All student benchmark runs use 2 warmup + 5 timed iterations
- Detailed benchmark uses stock `generate()` path (no KV cache, 50 tokens) for profiling
- Student benchmark uses `_generate_v8b` (KV-cached, 13 tokens for test audio)
- GPUProfile detected Blackwell (sm_120) and applied Blackwell tile configs:
  - Attention: 128×128 (encoder), 128×64 (decoder), nstages=2, nwarps=8
  - Matmul: 128×128×64
  - RoPE: nstages=2, nwarps=8
- Results are remarkably consistent across runs (stddev ~0.1-0.5ms) — expected for dedicated GPU (no MIG partitioning)
- Triton cache was cleared before Run 1 of student benchmark; compilation happens during warmup
- cuBLAS 13.1 pip package was uninstalled to avoid conflict with system cuBLAS 13.0
