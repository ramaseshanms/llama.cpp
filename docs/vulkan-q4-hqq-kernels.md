# Vulkan Q4_HQQ GPU Kernels

## Overview

This document describes the Vulkan GPU acceleration kernels added for the
Q4_HQQ quantization format (GGML types `GGML_TYPE_Q4_HQQ` and
`GGML_TYPE_Q4_HQQ_128`). These kernels enable inference of HQQ-quantized
4-bit models on Vulkan-capable GPUs.

## Quantization Format

### Q4_HQQ (g32, 5.0 bpw)

**Block layout** (20 bytes total):

| Field      | Type        | Bytes | Description                                |
|------------|-------------|-------|--------------------------------------------|
| `scale`    | `float16_t` | 2     | Affine scale: `s = 15 / (max - min)`       |
| `zero`     | `uint8_t`   | 1     | Integer zero-point: `z = round(-min * s)`  |
| `_pad`     | `uint8_t`   | 1     | Always 0 (alignment padding)               |
| `qs[16]`   | `uint8_t`   | 16    | 4-bit nibble-packed weights (split-half)   |

### Q4_HQQ_128 (g128, 4.25 bpw)

Same layout as Q4_HQQ, but with `qs[64]` (64 bytes) for 128 weights per
block. Total block size: 68 bytes.

### Dequantization Formula

```
w = (q - zero) / scale
```

where `q` is a 4-bit integer in `[0, 15]`, `zero` is the stored integer
zero-point, and `scale` is the stored FP16 scale factor.

### Nibble Packing (Split-Half Layout)

Identical to Q4_0 and Q4_1:
- `qs[j] & 0xF` = element `j` (low half of block)
- `qs[j] >> 4`  = element `j + QUANT_K/2` (high half of block)

This split-half convention is important: it means the element ordering in
memory is interleaved across the two halves, which perfectly matches the
vectorized B-vector loading pattern in `mul_mat_vec.comp`.

## GPU Kernel Architecture

### Reused Infrastructure

Q4_HQQ uses the standard `mul_mat_vec.comp` kernel path (the same as Q4_0,
Q4_1, Q5_0, Q5_1, Q8_0) rather than the specialized K-quant or integer
dot-product paths. This is controlled by:

- `DATA_A_QUANT_LEGACY` defined in `types.glsl` → activates the legacy
  `mul_mat_vec.comp` branch
- `QUANT_AUXF = 2` signals that `dm.y != 0` (like Q4_1), enabling the
  `v = v * dm.x + dm.y` reconstruction in the kernel

### Why Q4_HQQ Cannot Use the Integer Dot-Product (mmvq) Path

The integer dot-product path (`mul_mat_vecq.comp`, `mul_mmq.comp`) computes:
```
result += d * (sum(q * y_i) - offset * sum(y_i))
```
This requires a multiplicative scale `d` and an additive offset. Q4_HQQ uses
`(q - zero) / scale`, which can be rewritten as:
```
result += (1/scale) * sum(q * y_i) - (zero/scale) * sum(y_i)
```
While mathematically equivalent, this requires passing `1/scale` as the
scale parameter and `zero/scale` as the offset — which is not compatible
with how the integer dot-product block cache is structured in `mul_mmq.comp`.
Q4_HQQ is therefore explicitly excluded from the mmvq path.

### get_dm() Implementation

The `get_dm()` function returns `(dm.x, dm.y)` such that:
```
w = q * dm.x + dm.y = q / scale + (-zero / scale)
```

In `dequant_funcs.glsl`:
```glsl
vec2 get_dm(uint ib, uint a_offset) {
    const float inv_scale = 1.0f / float(data_a[a_offset + ib].scale);
    const float zero_f    = float(data_a[a_offset + ib].zero);
    return vec2(inv_scale, -zero_f * inv_scale);
}
```

## New Files Added

### Shader Source Files

| File | Description |
|------|-------------|
| `vulkan-shaders/dequant_q4_hqq.comp`     | Standalone dequant shader for Q4_HQQ g32 blocks |
| `vulkan-shaders/dequant_q4_hqq_128.comp` | Standalone dequant shader for Q4_HQQ_128 g128 blocks |

### Modified Files

| File | Changes |
|------|---------|
| `vulkan-shaders/types.glsl` | Added `block_q4_hqq`, `block_q4_hqq_128`, and their `_packed16` variants; defined `QUANT_K`, `QUANT_R`, `QUANT_AUXF`, `A_TYPE`, `DATA_A_QUANT_LEGACY` for both types |
| `vulkan-shaders/dequant_funcs.glsl` | Added `dequantize()`, `dequantize4()`, and `get_dm()` for both Q4_HQQ and Q4_HQQ_128; used by `mul_mat_vec.comp` and `get_rows_quant.comp` |
| `vulkan-shaders/dequant_funcs_cm2.glsl` | Added `decodeBufQ4_HQQ`, `dequantFuncQ4_HQQ`, `decodeBufQ4_HQQ_128`, `dequantFuncQ4_HQQ_128`; added `#define dequantFuncA` mappings; used by `mul_mm_cm2.comp` and `flash_attn_cm2.comp` |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | Registered `q4_hqq` and `q4_hqq_128` in `type_names`; updated `is_legacy_quant()`; excluded both types from `mul_mat_vecq.comp` (vec path) and `mul_mmq.comp` (matrix path) |
| `ggml-vulkan.cpp` | Added pipeline creation for all Q4_HQQ variants: `mul_mat_vec_f32_f32`, `mul_mat_vec_f16_f32`, `mul_mat_vec_id_f32`, `dequant`, `get_rows`, `get_rows_f32`; added Q4_HQQ to all `switch` statements in pipeline-selection and `supports_op` functions |

## Generated Shader Variants

The build system auto-generates the following SPIR-V shaders for Q4_HQQ:

- `mul_mat_vec_q4_hqq_f32_f32{,_subgroup,_subgroup_no_shmem}`
- `mul_mat_vec_q4_hqq_f16_f32{,_subgroup,_subgroup_no_shmem}`
- `mul_mat_vec_id_q4_hqq_f32{,_subgroup,_subgroup_no_shmem}`
- `mul_mat_vec_q4_hqq_128_*` (same variants for g128)
- `get_rows_q4_hqq{,_f32}`
- `get_rows_q4_hqq_128{,_f32}`
- `dequant_q4_hqq`
- `dequant_q4_hqq_128`
- `matmul_q4_hqq_*` (coopmat2 matrix-matrix multiply variants)
- `flash_attn_f32_f16_q4_hqq{,_128}_cm2{,_f16acc}` (coopmat2 flash attention)

## Building

```bash
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-cli -j$(nproc)
```

Requires Vulkan SDK >= 1.3.221 (for `VkPhysicalDevicePipelineRobustnessFeaturesEXT`
and other features used by the Vulkan backend). The LunarG Vulkan SDK
provides `glslc` for GLSL-to-SPIR-V compilation.

## Running GPU Inference

```bash
# Run with all layers on GPU (replace N with number of transformer layers)
./build/bin/llama-cli -m model-q4_hqq.gguf -ngl 99 -p "Your prompt here"

# Verify GPU is being used
./build/bin/llama-cli -m model-q4_hqq.gguf -ngl 99 -p "Test" -v 2>&1 | grep "ggml_vulkan"
```

## Quantizing Models

Use the Phase 3 format (INT8 zero-point storage, which is the default):

```bash
./build/bin/llama-quantize model-f16.gguf model-q4_hqq.gguf Q4_HQQ
./build/bin/llama-quantize model-f16.gguf model-q4_hqq_128.gguf Q4_HQQ_128
```

**Important:** Models quantized before Phase 3 (before commit `44ebed103`)
used FP16 zero-point storage and are incompatible with the current code.
Re-quantize those models from their F16 source.

## Performance Notes

- Q4_HQQ (g32) is 5.0 bpw — slightly higher than Q4_0 (4.5 bpw) due to
  per-32 scale+zero overhead, but uses HQQ optimization for better quality
- Q4_HQQ_128 (g128) is 4.25 bpw — close to Q4_0 quality overhead, with
  HQQ optimization over 128-element groups
- The Vulkan `mul_mat_vec` path handles token generation (batch size = 1)
- The Vulkan `mul_mm_cm2` path handles prefill with cooperative matrix ops
  (requires GPU with coopmat2 support, e.g. NVIDIA Ada/Hopper)
- The `flash_attn_cm2` path handles flash attention for long context

## Dequant Shader Details

### dequant_q4_hqq.comp

- 256 threads/workgroup, 4 blocks/workgroup (64 threads/block)
- Each thread handles 8 bytes (16 elements) of a Q4_HQQ block
- Output to FP16 scratch buffer for matrix-matrix multiply path

### dequant_q4_hqq_128.comp

- 256 threads/workgroup, 2 blocks/workgroup (128 active threads/block)
- Each thread handles 1 byte (2 elements) of a Q4_HQQ_128 block
- Threads with `tid >= 64` return early (64 qs bytes per block)
