# Ablation Test Results

**GPU:** NVIDIA H200 MIG 3g.71gb
**SMs:** 60, **VRAM:** 69.8GB, **Shared mem:** 227.0KB
**Stack:** PyTorch 2.10.0+cu130, CUDA 13.0, Triton 3.6.0
**Date:** 2026-03-19 03:00:48
**Baseline:** 205.2ms

## Summary Table

| # | Test | Time (ms) | Std | Delta | Accuracy | Status |
|---|------|-----------|-----|-------|----------|--------|
| 1 | `baseline_optimized` | 205.2 | 0.8 | +0.0 | 100% | PASS |
| 2 | `precision_fp32` | 206.7 | 1.4 | +1.5 | 100% | PASS |
| 3 | `fusion_off_mlp` | 204.5 | 2.7 | -0.7 | 100% | PASS |
| 4 | `fusion_off_rope` | 234.1 | 1.6 | +28.9 | 100% | PASS |
| 5 | `backend_triton_matmul` | 206.2 | 0.5 | +1.0 | 100% | PASS |
| 6 | `sdpa_fallback_off` | 208.6 | 1.7 | +3.4 | 100% | PASS |
| 7 | `sdpa_threshold_1` | 203.8 | 0.7 | -1.4 | 100% | PASS |
| 8 | `sdpa_threshold_8` | 204.8 | 2.3 | -0.4 | 100% | PASS |
| 9 | `sdpa_threshold_16` | 205.7 | 2.6 | +0.5 | 100% | PASS |
| 10 | `attn_enc_64x64_s1_w4` | 206.1 | 3.3 | +0.9 | 100% | PASS |
| 11 | `attn_enc_128x64_s2_w8` | 207.0 | 0.7 | +1.8 | 100% | PASS |
| 12 | `attn_enc_64x32_s1_w4` | 209.5 | 0.6 | +4.3 | 100% | PASS |
| 13 | `attn_dec_64x64_s2_w8` | 205.8 | 0.3 | +0.6 | 100% | PASS |
| 14 | `attn_dec_32x32_s1_w4` | 212.8 | 1.6 | +7.6 | 100% | PASS |
| 15 | `attn_dec_64x32_s2_w8` | 205.7 | 1.6 | +0.5 | 100% | PASS |
| 16 | `enc_nstages_1` | 217.4 | 4.8 | +12.2 | 100% | PASS |
| 17 | `enc_nstages_3` | 207.6 | 2.4 | +2.4 | 100% | PASS |
| 18 | `enc_nwarps_4` | 217.2 | 0.5 | +12.0 | 100% | PASS |
| 19 | `enc_nwarps_16` | 209.5 | 0.7 | +4.3 | 100% | PASS |
| 20 | `matmul_64x64x32` | 207.2 | 1.8 | +2.0 | 100% | PASS |
| 21 | `matmul_128x64x32` | 204.8 | 0.6 | -0.4 | 100% | PASS |
| 22 | `matmul_128x128x32` | 208.1 | 1.6 | +2.9 | 100% | PASS |

## Detailed Results

### 1. `baseline_optimized`

**Description:** Current optimized config (all optimizations enabled)

**Code changes:** (none — baseline config)

- **Mean:** 205.2ms (+/- 0.8ms)
- **Individual runs:** 206.4, 204.8, 204.5ms
- **Delta from baseline:** +0.0ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 2. `precision_fp32`

**Description:** Full fp32 pipeline (disable half-precision)

**Code changes:**
```diff
--- a/__init__.py
+++ b/__init__.py
@@ -27,3 +27,3 @@
 layers.Linear.BACKEND = "torch"
-layers.Linear.BF16 = True
+layers.Linear.BF16 = False
 layers.MLP.FUSED = True
```

- **Mean:** 206.7ms (+/- 1.4ms)
- **Individual runs:** 206.4, 205.2, 208.6ms
- **Delta from baseline:** +1.5ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 3. `fusion_off_mlp`

**Description:** Disable fused SwiGLU and LinearGELU kernels

**Code changes:**
```diff
--- a/__init__.py
+++ b/__init__.py
@@ -28,4 +28,4 @@
 layers.Linear.BF16 = True
-layers.MLP.FUSED = True
-layers.EncoderMLP.FUSED = True
+layers.MLP.FUSED = False
+layers.EncoderMLP.FUSED = False
```

- **Mean:** 204.5ms (+/- 2.7ms)
- **Individual runs:** 208.3, 202.6, 202.7ms
- **Delta from baseline:** -0.7ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 4. `fusion_off_rope`

**Description:** Disable fused Q+K RoPE (separate kernel launches)

**Code changes:**
```diff
--- a/rope.py
+++ b/rope.py
@@ -328,3 +328,3 @@
     # CUDA fast path: single fused kernel for both Q and K (from meave branch)
-    if q.is_cuda:
+    if False and q.is_cuda:
         total_qh = batch * num_q_heads
```

- **Mean:** 234.1ms (+/- 1.6ms)
- **Individual runs:** 233.2, 236.3, 232.7ms
- **Delta from baseline:** +28.9ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 5. `backend_triton_matmul`

**Description:** Use Triton matmul instead of cuBLAS for Linear layers

**Code changes:**
```diff
--- a/__init__.py
+++ b/__init__.py
@@ -26,3 +26,3 @@
 
-layers.Linear.BACKEND = "torch"
+layers.Linear.BACKEND = "triton"
 layers.Linear.BF16 = True
```

- **Mean:** 206.2ms (+/- 0.5ms)
- **Individual runs:** 206.9, 206.0, 205.8ms
- **Delta from baseline:** +1.0ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 6. `sdpa_fallback_off`

**Description:** Disable SDPA fallback (always custom flash attention)

**Code changes:**
```diff
--- a/attention.py
+++ b/attention.py
@@ -264,3 +264,3 @@
 
-    if q.is_cuda and seq_q <= 4:
+    if False:
         # For very short queries (KV-cached decode), use PyTorch SDPA
```

- **Mean:** 208.6ms (+/- 1.7ms)
- **Individual runs:** 207.8, 210.9, 207.0ms
- **Delta from baseline:** +3.4ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 7. `sdpa_threshold_1`

**Description:** SDPA fallback only for seq_q <= 1

**Code changes:**
```diff
--- a/attention.py
+++ b/attention.py
@@ -264,3 +264,3 @@
 
-    if q.is_cuda and seq_q <= 4:
+    if q.is_cuda and seq_q <= 1:
         # For very short queries (KV-cached decode), use PyTorch SDPA
```

- **Mean:** 203.8ms (+/- 0.7ms)
- **Individual runs:** 204.7, 203.2, 203.5ms
- **Delta from baseline:** -1.4ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 8. `sdpa_threshold_8`

**Description:** SDPA fallback for seq_q <= 8

**Code changes:**
```diff
--- a/attention.py
+++ b/attention.py
@@ -264,3 +264,3 @@
 
-    if q.is_cuda and seq_q <= 4:
+    if q.is_cuda and seq_q <= 8:
         # For very short queries (KV-cached decode), use PyTorch SDPA
```

- **Mean:** 204.8ms (+/- 2.3ms)
- **Individual runs:** 208.0, 203.1, 203.2ms
- **Delta from baseline:** -0.4ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 9. `sdpa_threshold_16`

**Description:** SDPA fallback for seq_q <= 16

**Code changes:**
```diff
--- a/attention.py
+++ b/attention.py
@@ -264,3 +264,3 @@
 
-    if q.is_cuda and seq_q <= 4:
+    if q.is_cuda and seq_q <= 16:
         # For very short queries (KV-cached decode), use PyTorch SDPA
```

- **Mean:** 205.7ms (+/- 2.6ms)
- **Individual runs:** 209.4, 204.1, 203.7ms
- **Delta from baseline:** +0.5ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 10. `attn_enc_64x64_s1_w4`

**Description:** Encoder attention 64x64, nstages=1, nwarps=4 (consumer config on H200)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -48,3 +48,3 @@
         "attn_tiles": {
-            64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
+            64:  (64, 64, 1, 4),  # ABLATION: consumer-sized tiles
             128: (128, 64, 2, 8),
```

- **Mean:** 206.1ms (+/- 3.3ms)
- **Individual runs:** 210.8, 203.7, 203.9ms
- **Delta from baseline:** +0.9ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 11. `attn_enc_128x64_s2_w8`

**Description:** Encoder attention 128x64, nstages=2, nwarps=8

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -48,3 +48,3 @@
         "attn_tiles": {
-            64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
+            64:  (128, 64, 2, 8),  # ABLATION: asymmetric tiles
             128: (128, 64, 2, 8),
```

- **Mean:** 207.0ms (+/- 0.7ms)
- **Individual runs:** 207.9, 206.7, 206.3ms
- **Delta from baseline:** +1.8ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 12. `attn_enc_64x32_s1_w4`

**Description:** Encoder attention 64x32, nstages=1, nwarps=4 (small tiles)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -48,3 +48,3 @@
         "attn_tiles": {
-            64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
+            64:  (64, 32, 1, 4),  # ABLATION: small tiles
             128: (128, 64, 2, 8),
```

- **Mean:** 209.5ms (+/- 0.6ms)
- **Individual runs:** 210.2, 209.4, 208.8ms
- **Delta from baseline:** +4.3ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 13. `attn_dec_64x64_s2_w8`

**Description:** Decoder attention 64x64, nstages=2, nwarps=8

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -49,3 +49,3 @@
             64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
-            128: (128, 64, 2, 8),
+            128: (64, 64, 2, 8),  # ABLATION
         },
```

- **Mean:** 205.8ms (+/- 0.3ms)
- **Individual runs:** 206.3, 205.7, 205.5ms
- **Delta from baseline:** +0.6ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 14. `attn_dec_32x32_s1_w4`

**Description:** Decoder attention 32x32, nstages=1, nwarps=4 (consumer config)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -49,3 +49,3 @@
             64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
-            128: (128, 64, 2, 8),
+            128: (32, 32, 1, 4),  # ABLATION
         },
```

- **Mean:** 212.8ms (+/- 1.6ms)
- **Individual runs:** 212.3, 211.2, 215.0ms
- **Delta from baseline:** +7.6ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 15. `attn_dec_64x32_s2_w8`

**Description:** Decoder attention 64x32, nstages=2, nwarps=8

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -49,3 +49,3 @@
             64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
-            128: (128, 64, 2, 8),
+            128: (64, 32, 2, 8),  # ABLATION
         },
```

- **Mean:** 205.7ms (+/- 1.6ms)
- **Individual runs:** 205.5, 203.9, 207.7ms
- **Delta from baseline:** +0.5ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 16. `enc_nstages_1`

**Description:** Encoder attention nstages=1 (disable double-buffering)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -48,3 +48,3 @@
         "attn_tiles": {
-            64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
+            64:  (128, 128, 1, 8),  # ABLATION: nstages=1
             128: (128, 64, 2, 8),
```

- **Mean:** 217.4ms (+/- 4.8ms)
- **Individual runs:** 224.1, 214.3, 213.7ms
- **Delta from baseline:** +12.2ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 17. `enc_nstages_3`

**Description:** Encoder attention nstages=3 (triple-buffering)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -48,3 +48,3 @@
         "attn_tiles": {
-            64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
+            64:  (128, 128, 3, 8),  # ABLATION: nstages=3
             128: (128, 64, 2, 8),
```

- **Mean:** 207.6ms (+/- 2.4ms)
- **Individual runs:** 210.9, 205.5, 206.4ms
- **Delta from baseline:** +2.4ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 18. `enc_nwarps_4`

**Description:** Encoder attention nwarps=4 (128 threads/block)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -48,3 +48,3 @@
         "attn_tiles": {
-            64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
+            64:  (128, 128, 2, 4),  # ABLATION: nwarps=4
             128: (128, 64, 2, 8),
```

- **Mean:** 217.2ms (+/- 0.5ms)
- **Individual runs:** 218.0, 216.9, 216.8ms
- **Delta from baseline:** +12.0ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 19. `enc_nwarps_16`

**Description:** Encoder attention nwarps=16 (512 threads/block)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -48,3 +48,3 @@
         "attn_tiles": {
-            64:  (128, 128, 2, 8),  # Large tiles + 2 pipeline stages
+            64:  (128, 128, 2, 16),  # ABLATION: nwarps=16
             128: (128, 64, 2, 8),
```

- **Mean:** 209.5ms (+/- 0.7ms)
- **Individual runs:** 210.5, 209.1, 208.9ms
- **Delta from baseline:** +4.3ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 20. `matmul_64x64x32`

**Description:** Matmul tiles 64x64x32 (consumer config on H200)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -51,3 +51,3 @@
         },
-        "matmul_tiles": (128, 128, 64),
+        "matmul_tiles": (64, 64, 32),  # ABLATION
         "rope_nstages": 2,
```

- **Mean:** 207.2ms (+/- 1.8ms)
- **Individual runs:** 206.5, 205.4, 209.6ms
- **Delta from baseline:** +2.0ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 21. `matmul_128x64x32`

**Description:** Matmul tiles 128x64x32 (asymmetric, smaller TILE_K)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -51,3 +51,3 @@
         },
-        "matmul_tiles": (128, 128, 64),
+        "matmul_tiles": (128, 64, 32),  # ABLATION
         "rope_nstages": 2,
```

- **Mean:** 204.8ms (+/- 0.6ms)
- **Individual runs:** 205.6, 204.7, 204.1ms
- **Delta from baseline:** -0.4ms
- **Accuracy:** 100.0%
- **Tokens:** 13

### 22. `matmul_128x128x32`

**Description:** Matmul tiles 128x128x32 (same spatial, smaller TILE_K)

**Code changes:**
```diff
--- a/layers.py
+++ b/layers.py
@@ -51,3 +51,3 @@
         },
-        "matmul_tiles": (128, 128, 64),
+        "matmul_tiles": (128, 128, 32),  # ABLATION
         "rope_nstages": 2,
```

- **Mean:** 208.1ms (+/- 1.6ms)
- **Individual runs:** 207.7, 206.4, 210.3ms
- **Delta from baseline:** +2.9ms
- **Accuracy:** 100.0%
- **Tokens:** 13
