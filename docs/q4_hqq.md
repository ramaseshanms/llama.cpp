# Q4_HQQ and Q4_HQQ_128 Quantization

This document describes the two HQQ-based quantization types implemented in
`feature/q4_hqq`: `GGML_TYPE_Q4_HQQ` (g32) and `GGML_TYPE_Q4_HQQ_128`
(g128).  It covers the algorithm, block format, performance characteristics,
and usage notes.

---

## 1. What is HQQ?

HQQ (Half-Quadratic Quantization) is a post-training weight quantization
method described in:

> **Accurate Quantization of Diffusion Models via Optimal Ordering and
> Explainability** (Badri & Shaji, 2023)
> Blog post: https://dropbox.github.io/hqq_blog/
> Reference code: https://github.com/mobiusml/hqq

The key properties of HQQ vs. plain round-to-nearest (RTN):

| Property | RTN | HQQ |
|---|---|---|
| Calibration data required | No | No |
| Loss function | L2 (implicit) | Lp, p < 1 (configurable) |
| Parameters refined | scale + zero | zero only (scale fixed) |
| Outlier robustness | Low | High (Lp de-emphasises outliers) |
| Computation cost | O(n) | O(iters × n) |

HQQ fixes the scale at `s = (max_q) / (max - min)` and then iteratively
refines the zero-point `z` by minimising an Lp reconstruction loss via the
**Half-Quadratic Splitting** (HQS) proximal algorithm.

---

## 2. Algorithm

### 2.1 Initialization (same as RTN)

For each quantization group of `n` weights `W[0..n-1]`:

```
s   = 15 / (max(W) - min(W))   # scale: maps [min,max] → [0,15]
z₀  = round(-min(W) * s)       # RTN zero-point (integer-rounded)
```

### 2.2 Half-Quadratic Splitting

The loop runs for up to `HQQ_ITERS = 20` iterations:

```
For t = 0 .. iters-1:

  (Forward pass)
  For each j:
    q[j]   = clamp(round(W[j]*s + z), 0, max_q)    # quantize
    W_r[j] = (q[j] - z) / s                         # reconstruct
    r[j]   = W[j] - W_r[j]                          # residual

  (L1 error for early stopping)
  err = Σ |r[j]|

  (Early stop: keep best z seen so far)
  If err ≥ best_err: restore z = best_z; stop
  Else: best_err = err; best_z = z

  (SP1: generalized soft-thresholding, Lp shrink operator)
  W_e[j] = shrink_lp(r[j], β, p)
         = sign(r[j]) * relu(|r[j]| - |r[j]|^(p-1) / β)

  (SP2: closed-form zero-point update — weighted mean of de-biased codes)
  z = (1/n) Σ_j  (q[j] - (W[j] - W_e[j]) * s)

  (Tighten the proximal penalty)
  β ← β × κ
```

On exit, `z = best_z` (the zero-point that produced the lowest L1
reconstruction error across all iterations).

The stored zero-point is then:

```
z_stored = round(z)    # integer-rounded, stored as uint8
```

Because z is bounded to `[0, max_q] = [0, 15]` for typical weight
distributions, rounding to integer loses at most 0.5 LSB of precision —
negligible for 4-bit quantization.

### 2.3 shrink_lp Operator

```c
// Generalized soft-thresholding for Lp, p < 1.
// For p < 1, the threshold |x|^(p-1)/β is larger for small |x| than for
// large |x|, meaning outliers receive LESS shrinkage than small values.
// This is the key property that makes HQQ robust to weight outliers.
static inline float shrink_lp(float x, float beta, float p) {
    float ax    = fabsf(x);
    float thr   = powf(ax, p - 1.0f) / beta;  // threshold (decays for large |x|)
    float shrunk = ax - thr;
    return (shrunk > 0.0f) ? copysignf(shrunk, x) : 0.0f;
}
```

### 2.4 Hyperparameters

| Constant | Value | Meaning |
|---|---|---|
| `HQQ_LP_NORM` | 0.7 | Lp exponent p (paper default) |
| `HQQ_BETA_INIT` | 10.0 | Initial proximal penalty β₀ |
| `HQQ_BETA_KAPPA` | 1.01 | β growth factor per iteration |
| `HQQ_ITERS` | 20 | Maximum iterations |
| `HQQ_MAX_GROUP` | 128 | Maximum group size (stack array bound) |

---

## 3. Block Formats

### 3.1 `block_q4_hqq` (g32, `GGML_TYPE_Q4_HQQ`)

```
+---------+---------+---------+------------------+
| scale   | zero    | _pad    | qs[16]           |
| FP16 2B | uint8 1B| uint8 1B| 4-bit packed 16B |
+---------+---------+---------+------------------+
Total: 20 bytes per block, 32 weights/block → 5.00 bpw
```

### 3.2 `block_q4_hqq_128` (g128, `GGML_TYPE_Q4_HQQ_128`)

```
+---------+---------+---------+------------------+
| scale   | zero    | _pad    | qs[64]           |
| FP16 2B | uint8 1B| uint8 1B| 4-bit packed 64B |
+---------+---------+---------+------------------+
Total: 68 bytes per block, 128 weights/block → 4.25 bpw
```

### 3.3 Nibble packing (split-half convention)

For a group of n weights, the packed bytes are arranged as:

```
qs[j] = (q[j] & 0x0F) | ((q[j + n/2] & 0x0F) << 4)
         ^-- low nibble = element j                ^-- high nibble = element j+n/2
```

This split-half layout is required by the AVX2 `bytes_from_nibbles_32`
intrinsic (which reads low nibbles to lanes 0–15 and high nibbles to lanes
16–31) and the NEON `vandq_u8` / `vshrq_n_u8` pattern.

### 3.4 Dequantization formula

```
W[j]       = (q[j]       - z) / s    # where q and z are integers
W[j + n/2] = (q[j+n/2]   - z) / s
```

---

## 4. Bits-per-weight Comparison

| Type | Group size | Block bytes | bpw |
|---|---|---|---|
| `Q4_0` | 32 | 18 | 4.50 |
| `Q4_HQQ` | 32 | 20 | 5.00 |
| `Q4_1` | 32 | 20 | 5.00 |
| `Q4_HQQ_128` | 128 | 68 | 4.25 |
| `Q4_K_M` | 256 (super) | — | ~4.50 |

`Q4_HQQ` (g32) uses 5.00 bpw because the per-block metadata (scale+zero+pad)
is 4 bytes over 32 weights.  The larger group size of `Q4_HQQ_128` (g128)
amortises this overhead to 4.25 bpw, approaching the theoretical 4.00 bpw
minimum.

---

## 5. imatrix Support

Both types support an importance matrix (`imatrix`) for improved accuracy.
The imatrix provides per-element activation magnitudes from a calibration
forward pass.  Elements with larger activation magnitudes contribute more to
the output loss and are given higher weight in the zero-point optimization.

### Usage (CLI)

```bash
llama-quantize --imatrix imatrix.dat model.gguf model-q4hqq.gguf Q4_HQQ
llama-quantize --imatrix imatrix.dat model.gguf model-q4hqq128.gguf Q4_HQQ_128
```

### API

```c
// imatrix = NULL → plain HQQ (no calibration data)
size_t quantize_q4_hqq    (src, dst, nrows, n_per_row, imatrix);
size_t quantize_q4_hqq_128(src, dst, nrows, n_per_row, imatrix);
```

With `imatrix != NULL`, the importance-weighted SP2 update is used:

```
z = (Σ_j  iw[j] * (q[j] - (W[j] - W_e[j]) * s)) / (Σ_j iw[j])
```

where `iw[j]` is the per-element importance weight for position `j`.

---

## 6. SIMD Kernels

### AVX2 (`arch/x86/quants.c`)

The dot-product kernel (`ggml_vec_dot_q4_hqq_q8_0`) follows this structure:

```
1. bytes_from_nibbles_32(qs)         → 32 unsigned 8-bit codes in AVX2 reg
2. _mm256_maddubs_epi16(codes, y_u8) → 16 signed 16-bit products (×2)
3. _mm256_madd_epi16(acc, ones)      → 8 signed 32-bit sums
4. _mm256_cvtepi32_ps(acc)           → 8 floats
5. _mm256_fmadd_ps(...)              → accumulate into sum
Per block:  sum = (yd / scale) * (dot(q_codes, y) - zero * sum(y))
```

For the g128 variant, 4 Q8_0 sub-blocks (32 elements each) are iterated per
weight block, each with its own `yd` scale from the Q8_0 block header.

### NEON (`arch/arm/quants.c`)

```
1. vandq_u8 / vshrq_n_u8             → low/high nibbles
2. ggml_vdotq_s32(q8 codes, w codes) → 4 int32 partial sums
3. vpaddlq_s8 / vaddq_s16 / ...      → collapse to scalar
Per block:  same reformulation as AVX2
```

---

## 7. Group Size Trade-offs

| | g32 (`Q4_HQQ`) | g128 (`Q4_HQQ_128`) |
|---|---|---|
| bpw | 5.00 | 4.25 |
| Zero-point coverage | High (1 zero per 32 w) | Lower (1 zero per 128 w) |
| MSE (smooth signal) | ~3×10⁻⁴ | ~2×10⁻³ |
| KV cache matrices | ✓ (recommended) | ✗ |
| Weight matrices | ✓ | ✓ (paper default) |
| HQQ solver iterations | 20 | 20 (same) |

The HQQ paper uses g128 as the default for weight matrices and recommends
smaller group sizes (g32 or smaller) for KV-cache quantization.

---

## 8. Implementation Files

| File | Purpose |
|---|---|
| `ggml/src/ggml-common.h` | Block struct definitions (`block_q4_hqq`, `block_q4_hqq_128`) |
| `ggml/src/ggml-quants.c` | HQQ solver (`hqq_optimize_zero`, `hqq_optimize_zero_imatrix`), quantize/dequantize |
| `ggml/src/ggml-quants.h` | Public API declarations |
| `ggml/include/ggml.h` | Type enum (`GGML_TYPE_Q4_HQQ`, `GGML_TYPE_Q4_HQQ_128`) |
| `ggml/src/ggml.c` | Type table entries, ftype dispatch, `quantize_chunk` dispatch |
| `ggml/src/ggml-cpu/ggml-cpu.c` | CPU op table (`.from_float`, `.vec_dot`, `.vec_dot_type`) |
| `ggml/src/ggml-cpu/quants.c` | Generic (scalar) vec_dot implementation |
| `ggml/src/ggml-cpu/arch/x86/quants.c` | AVX2-optimised vec_dot |
| `ggml/src/ggml-cpu/arch/arm/quants.c` | NEON-optimised vec_dot |
| `ggml/src/ggml-cpu/arch-fallback.h` | Generic fallback for non-x86/ARM targets |
| `ggml/src/ggml-cpu/ops.cpp` | Op dispatch for `GGML_TYPE_Q4_HQQ_128` |
| `include/llama.h` | `LLAMA_FTYPE_MOSTLY_Q4_HQQ`, `LLAMA_FTYPE_MOSTLY_Q4_HQQ_128` |
| `src/llama-quant.cpp` | llama ftype → ggml type mapping |
| `src/llama-model-loader.cpp` | ftype detection |
| `tools/quantize/quantize.cpp` | CLI entries `Q4_HQQ`, `Q4_HQQ_128` |
| `tests/test-q4-hqq.cpp` | 8-test suite (round-trip, HQQ vs RTN, imatrix, INT8 zero) |

---

## 9. Commit History

| Commit | Description |
|---|---|
| `feat: Phase 1a` | HQQ proximal solver + wire into `quantize_row_q4_hqq_ref` |
| `feat: Phase 1b` | imatrix importance-weighted zero optimization |
| `feat: Phase 2` | `Q4_HQQ_128` block type (g128), all dispatch paths |
| `feat: Phase 3` | Zero storage changed from FP16 to INT8 (size-neutral) |
| `fix: packing consistency` | Use integer zero for packing to match dequant |
| `test: expand suite` | 8 named tests replacing single-MSE check |
| `docs: q4_hqq.md` | This file |

---

## 10. Known Limitations

- **No GPU / Vulkan backend**: `Q4_HQQ` and `Q4_HQQ_128` are CPU-only at
  this time.  CUDA / Metal / Vulkan kernels have not been added.
- **No `nrows=2` on ARM**: The ARM NEON path uses `nrows=1`; the g32 type
  could use `nrows=2` with `ARM_FEATURE_MATMUL_INT8` (not yet wired).
- **Format incompatibility**: Existing GGUF files quantized with the old
  FP16-zero encoding must be re-quantized after the Phase 3 change.
- **Negative zero-point clamped to 0**: For groups where all values are
  positive, the RTN zero may be slightly negative (e.g., `-min*scale < 0`).
  `MAX(0, ...)` clamps this to 0; future work could use an int8 signed zero
  to handle this edge case.
