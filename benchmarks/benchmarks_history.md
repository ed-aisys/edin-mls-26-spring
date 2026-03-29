# RTX 5090 Development Benchmark History

This file is the canonical source for the original RTX 5090 development chain
used by Appendix Table `tab:progression` in `report/report_no_abstract.tex`.

## Source Status

No standalone raw RTX 5090 console log for the original `261.3 -> 98.5` chain is
archived in this repo. The numbers below were extracted from the pre-cleanup
branch state primarily from:

- `docs/design_choices.md`
- `docs/exhaustive_optimization_list.md`

That makes this file `history-backed`, not raw-log-backed. It exists so the
appendix can keep the original step-by-step benchmark chain after the verbose
historical docs are removed.

## Why This Exists Separately From `benchmarks_5090.md`

There are two different RTX 5090 benchmark records in this repo:

- `benchmarks_5090.md`: later confirmation rerun used for the report headline
  and cross-GPU table (`100.4 ms` vs `262.2 ms`)
- this file: original development benchmark chain used for the appendix
  progression (`261.3 ms` baseline to `98.5 ms` final)

These should not be mixed, because the appendix deltas are only meaningful when
kept on one coherent benchmark session.

## Original Development Progression

| # | Change | Time | Delta |
|---|--------|------|-------|
| 0 | Baseline | 261.3ms | -- |
| 1 | Triton kernels + cuBLAS + TF32 | 209.8ms | -51.5ms |
| 2 | bf16 weights + flash attention | 136.4ms | -73.4ms |
| 3 | Fused Q+K RoPE pair kernel | 124.6ms | -11.8ms |
| 4 | bf16 RMSNorm output kernel | 120.7ms | -3.9ms |
| 5 | bf16 LayerNorm output | 121.1ms | -0.7ms |
| 6 | `generate_v8b` with KV cache | 113.5ms | -7.6ms |
| 7 | SDPA fallback for `seq_q <= 4` | 110.0ms | -3.5ms |
| 8 | fp16 cuBLAS HGEMM | 109.6ms | -0.4ms |
| 9 | Remove `Linear._forward_torch` `.float()` | 102.1ms | -7.5ms |
| 10 | Remove SiLU/GELU fp32 cast | 98.4ms | -3.7ms |
| 11 | Remove norm fp32 cast | 98.1ms | -0.3ms |
| 12 | fp16 embeddings + fused MLP | 98.5ms | +0.4ms |

## Historical Rejected Optimizations

| Optimization | Impact | Historical Status |
|-------------|--------|-------------------|
| SwiGLU grid swizzling (`GROUP_SIZE_M=8`) | `+18 ms` | Rejected |
| `@triton.autotune` for GELU / SiLU | `+0.7 ms` | Rejected |
| Flash attention `num_stages=2` on RTX 5090 | `Crash` | Rejected |
| PyTorch SDPA for all attention | `+6 ms` | Rejected |
| SDPA `enable_gqa=True` | `+13 ms` | Rejected |
| Runtime warmup autotune | `+3.1 ms` | Rejected |

## Branch Attribution For Rejected Rows

The appendix rejected-optimization rows are still `history-backed`, but a sweep
across branches helps pin down which historical branch introduced or motivated each test:

| Optimization | Source | Notes |
|-------------|---------------------------|-------|
| SwiGLU grid swizzling (`GROUP_SIZE_M=8`) | person4 | Preserved on cleanup branch in `docs/design_choices.md` as `+18 ms` regression after adding swizzling. |
| `@triton.autotune` for GELU / SiLU | person2 | Preserved on cleanup branch in `docs/design_choices.md` as `+0.7 ms` tuning overhead. |
| Flash attention `num_stages=2` on RTX 5090 | historically associated with the Person4 tuning branch | Preserved on cleanup branch in `docs/design_choices.md` as “kernel won't launch” / consumer shared-memory limit. |
| PyTorch SDPA for all attention | historical local test, preserved on cleanup branch | Preserved on cleanup branch in `docs/design_choices.md` as `+6 ms` for encoder/prefill. |
| SDPA `enable_gqa=True` | historical local test, preserved on cleanup branch | Preserved on cleanup branch in `docs/design_choices.md` as `+13 ms`. |
| Runtime warmup autotune | person1 | Preserved on cleanup branch in `docs/design_choices.md` as `101.6 ms` vs `98.5 ms` (`+3.1 ms`). |

## Sweep Result

- person3: no additional rejected-optimization benchmark evidence was
  found beyond general benchmark utilities and implementation files.
- person2: useful for attributing the lightweight `@triton.autotune`
  rejection .
- person4: useful for attributing the swizzling regression, and for tying the aggressive attention tuning context to the `num_stages=2` rejection on consumer GPUs.

## Report Use

Use this file only for:

- Appendix Table `tab:progression`
- Appendix Table `tab:rejected`
- narrative discussion of the original RTX 5090 development chain

Do not use this file for:

- the headline RTX 5090 number
- the cross-GPU comparison table

Those belong to `benchmarks_5090.md`.
