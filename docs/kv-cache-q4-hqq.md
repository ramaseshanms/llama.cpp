# Q4_HQQ KV Cache Quantization

**Branch:** `feature/q4_hqq`
**Date:** 2026-03-06

---

## Overview

This document describes the changes required to support `Q4_HQQ` as a KV cache quantization type in llama.cpp, enabling the `--cache-type-k q4_hqq` and `--cache-type-v q4_hqq` CLI flags.

Q4_HQQ uses Half-Quadratic Quantization (HQQ) to encode weights at 4 bits per value with a block size of 32 elements and a 20-byte block struct (scale + zero-point in FP16, plus 16 bytes of packed nibbles), yielding **5.0 bits-per-weight** including metadata overhead.

---

## Changes Required

### 1. `common/arg.cpp` — Register as allowed KV cache type

**File:** `common/arg.cpp`

The static `kv_cache_types` vector at line ~391 enumerates every type accepted by `--cache-type-k` / `--cache-type-v`. Adding `GGML_TYPE_Q4_HQQ` here is the single gate that allows the CLI flags to accept `q4_hqq` as a value.

```diff
 const std::vector<ggml_type> kv_cache_types = {
     GGML_TYPE_F32,
     GGML_TYPE_F16,
     GGML_TYPE_BF16,
     GGML_TYPE_Q8_0,
     GGML_TYPE_Q4_0,
     GGML_TYPE_Q4_1,
     GGML_TYPE_IQ4_NL,
     GGML_TYPE_Q5_0,
     GGML_TYPE_Q5_1,
+    GGML_TYPE_Q4_HQQ,
 };
```

`kv_cache_type_from_str()` iterates this list and matches against `ggml_type_name(type)` which returns `"q4_hqq"` for `GGML_TYPE_Q4_HQQ`, so no further changes are needed in the parsing functions.

### 2. `ggml/src/ggml-cpu/ops.cpp` — Allow Q4_HQQ in tensor operation dispatchers

Several functions in `ops.cpp` dispatch tensor operations through `switch (src->type)` blocks. Any quantized type not listed in these switches falls to `default: GGML_ABORT("fatal error")`, which would crash the process if Q4_HQQ tensors pass through these paths during KV cache usage (e.g., with Flash Attention's internal ADD, ACC, SET, or GET_ROWS operations on cache tensors).

`GGML_TYPE_Q4_HQQ` must be added alongside the existing quantized types in the following functions:

| Function | Operation | What it does for quantized types |
|---|---|---|
| `ggml_compute_forward_add` | `GGML_OP_ADD` | Routes to `ggml_compute_forward_add_q_f32` |
| `ggml_compute_forward_add1` | `GGML_OP_ADD1` | Routes to `ggml_compute_forward_add1_q_f32` |
| `ggml_compute_forward_acc` | `GGML_OP_ACC` | Routes to `ggml_compute_forward_acc_f32` |
| `ggml_compute_forward_out_prod` | `GGML_OP_OUT_PROD` | Routes to `ggml_compute_forward_out_prod_q_f32` |
| `ggml_compute_forward_set` | `GGML_OP_SET` | Routes to `ggml_compute_forward_set_q_f32` |
| `ggml_compute_forward_get_rows` | `GGML_OP_GET_ROWS` | Routes to `ggml_compute_forward_get_rows_q` |
| `ggml_compute_forward_clamp` | `GGML_OP_CLAMP` | Routes to `GGML_ABORT` (not supported for quantized types) |

For all functions above, `GGML_TYPE_Q4_HQQ` is added as a fall-through case alongside the existing quantized type cases (e.g., `GGML_TYPE_Q4_0`, `GGML_TYPE_Q4_1`, `GGML_TYPE_MXFP4`, etc.).

---

## Why These Changes Are Sufficient

The underlying Q4_HQQ quantize/dequantize pipeline was already fully wired before this change:

| Component | File | Status |
|---|---|---|
| Block struct (`block_q4_hqq`, 20 bytes) | `ggml/src/ggml-common.h` | Already present |
| Type traits registration (blck_size, to_float, from_float_ref) | `ggml/src/ggml.c` | Already present |
| CPU kernel registration (from_float, vec_dot, vec_dot_type) | `ggml/src/ggml-cpu/ggml-cpu.c` | Already present |
| Quantize function (`quantize_q4_hqq`) | `ggml/src/ggml-quants.c` | Already present |
| Dequantize function (`dequantize_row_q4_hqq`) | `ggml/src/ggml-quants.c` | Already present |
| Vec-dot kernel (`ggml_vec_dot_q4_hqq_q8_0`) | `ggml/src/ggml-cpu/quants.c` | Already present |
| Chunk-quantize dispatch | `ggml/src/ggml.c` | Already present |
| Row-data validation | `ggml/src/ggml-quants.c` | Already present |
| LLAMA_FTYPE enum entry | `include/llama.h` | Already present |
| Ftype ↔ ggml_type mapping | `src/llama-quant.cpp` | Already present |

The KV cache constructor (`llama_kv_cache::llama_kv_cache`) accepts any `ggml_type` for K and V tensors and creates them with `ggml_new_tensor_3d`. The `llama_context` setup in `src/llama-context.cpp` validates block-size divisibility with `n_embd_head_k % blck_size != 0`; for Q4_HQQ with `QK4_HQQ = 32` and `n_embd_head_k = 128`, `128 % 32 == 0`, so this check passes.

The only missing pieces were:
1. CLI exposure (the `kv_cache_types` whitelist in `common/arg.cpp`), and
2. Defensive coverage of tensor operation dispatch switches in `ops.cpp`.

---

## Memory Usage Comparison

At a 2048-token context with a 3B Llama model (28 layers, 8 KV heads, 128 head-dim):

| KV cache type | Bytes/element | K+V total (28 layers) |
|---|---|---|
| F16 (default) | 2.0 | 448 MiB |
| Q8_0 | 1.0625 | ~238 MiB |
| Q4_0 | 0.5625 | ~126 MiB |
| **Q4_HQQ** | **0.625** | **~140 MiB** |
| Q5_0 | 0.6875 | ~154 MiB |

Q4_HQQ achieves approximately **3.2× compression** versus F16. Compared to Q4_0 it uses slightly more memory (extra 2 bytes for a separate zero-point field vs the Q4_0 single scale), but provides better numerical accuracy via the explicit affine zero-point, reducing outlier distortion in activations that appear in Key and Value projections.

**Block structure:**
```
block_q4_hqq = { scale: FP16, zero: FP16, qs[16]: uint8 }  // 20 bytes per 32 elements
```
vs
```
block_q4_0   = { delta: FP16, qs[16]: uint8 }               // 18 bytes per 32 elements
```

### Expected Output Quality

Q4_HQQ should produce output quality close to Q8_0 for most prompts, because:
- The explicit zero-point (vs Q4_0's symmetric-around-midpoint approach) better handles asymmetric activation distributions, which are common in attention K/V projections.
- The HQQ calibration-free quantization minimizes block-level reconstruction error.

Compared to F16 KV cache, Q4_HQQ will show a slight but perceptible quality degradation at long contexts where small errors accumulate. At short-to-medium context lengths (< 4096 tokens) the difference is typically below perceptible thresholds for instruction-following tasks.

---

## Usage

```bash
# Use Q4_HQQ for both K and V caches
./build/bin/llama-cli -m model.gguf -c 4096 -ctk q4_hqq -ctv q4_hqq -p "Hello"

# Note: quantized V cache requires Flash Attention (-fa)
./build/bin/llama-cli -m model.gguf -c 4096 -ctk q4_hqq -ctv q4_hqq -fa -p "Hello"

# Use Q4_HQQ only for K cache (no Flash Attention needed)
./build/bin/llama-cli -m model.gguf -c 4096 -ctk q4_hqq -p "Hello"

# Benchmark vs F16
./build/bin/llama-bench -ctk f16 -ctv f16 -c 4096
./build/bin/llama-bench -ctk q4_hqq -c 4096
./build/bin/llama-bench -ctk q4_hqq -ctv q4_hqq -fa -c 4096
```

> **Note:** Quantized V cache (`-ctv q4_hqq`) requires Flash Attention (`-fa`). This is a pre-existing constraint in `src/llama-context.cpp` enforced for all quantized V cache types:
> ```c
> if (!cparams.flash_attn && ggml_is_quantized(params.type_v)) {
>     throw std::runtime_error("quantized V cache requires Flash Attention");
> }
> ```

---

## Constraints

- **Flash Attention required for quantized V cache.** The standard attention path does not support quantized V tensors. Use `-fa` when setting `-ctv q4_hqq`.
- **Block size alignment.** `n_embd_head_k` must be divisible by `QK4_HQQ = 32`. Standard head sizes (64, 128, 256) all satisfy this.
- **No sgemm fast path.** Q4_HQQ is not registered in `ggml/src/ggml-cpu/llamafile/sgemm.cpp`. The system automatically falls back to the `vec_dot` path, which is correct but may be slightly slower for large batch sizes.
- **KV cache defragmentation.** Defrag operations use `ggml_cpy` internally, which handles arbitrary quantized types generically and works correctly with Q4_HQQ.

---

## Testing

```bash
# Verify the flag is accepted
./build/bin/llama-cli --help | grep -A5 "cache-type-k"

# Run inference with Q4_HQQ K cache
./build/bin/llama-cli -m model.gguf -c 2048 -ctk q4_hqq -p "The capital of France is" -n 10

# Run inference with Q4_HQQ K+V cache (Flash Attention required)
./build/bin/llama-cli -m model.gguf -c 2048 -ctk q4_hqq -ctv q4_hqq -fa -p "The capital of France is" -n 10

# Compare memory usage
watch -n1 free -h
```
