# Q4_HQQ vs Q4_0: Perplexity Benchmark Report

**Branch:** `feature/q4_hqq`
**Date:** 2026-03-06
**System:** Intel Xeon Platinum 8488C, 8 cores, 15 GiB RAM, CPU-only (AMX)
**Model:** `unsloth/Llama-3.2-3B` (3.21 B params, 28 layers)
**Dataset:** WikiText-2 raw test set (`wiki.test.raw`)

---

## 1. TL;DR — Results

| Model | Quant | File Size | PPL (WikiText-2) | ΔPPL vs Q4_0 |
|-------|-------|-----------|------------------|--------------|
| Llama-3.2-3B Q4_0   | Q4_0   | 1.8 GiB | **8.4419 ± 0.211** | — (reference) |
| Llama-3.2-3B Q4_HQQ | Q4_HQQ | 2.0 GiB | **8.4417 ± 0.209** | −0.0002 (negligible) |

**Q4_HQQ achieves essentially identical perplexity to Q4_0** on WikiText-2 (difference < 0.001
PPL, within measurement noise). The extra 0.5 bpw overhead goes into the explicit zero-point
and does not hurt quality — but also does not help on this model's weight distributions.

---

## 2. Bug Fixes Applied in This Commit

This benchmark run **required fixing two critical correctness bugs** in the Q4_HQQ quantization
and dequantization code. Without these fixes, the model produced perplexity in the millions
(completely broken inference).

### 2.1 Bug: Wrong Nibble Packing in `quantize_row_q4_hqq_ref`

**File:** `ggml/src/ggml-quants.c`

**Root cause:** The quantizer used **interleaved** nibble packing (adjacent pairs stored per
byte), while all vec_dot kernels (generic, AVX2, NEON) expect **split-half** packing (first
half of block in low nibbles, second half in high nibbles — matching Q4_0 convention).

**Before (broken — interleaved packing):**
```c
for (int j = 0; j < qk/2; ++j) {
    const float v0 = x[i*qk + j*2 + 0];   // element 0,2,4,...,30
    const float v1 = x[i*qk + j*2 + 1];   // element 1,3,5,...,31
    y[i].qs[j] = q0 | (q1 << 4);
    // qs[0] = element 0 (lo) | element 1 (hi)
    // qs[1] = element 2 (lo) | element 3 (hi)  ...WRONG!
}
```

**After (correct — split-half packing):**
```c
for (int j = 0; j < qk/2; ++j) {
    const float v0 = x[i*qk + j];          // element j      (0..15)
    const float v1 = x[i*qk + j + qk/2];   // element j+16  (16..31)
    y[i].qs[j] = q0 | (q1 << 4);
    // qs[0] = element 0 (lo) | element 16 (hi)
    // qs[1] = element 1 (lo) | element 17 (hi)  ...CORRECT
}
```

**Why this matters for vec_dot:** The AVX2 kernel uses `bytes_from_nibbles_32(qs)` which
expands 16 bytes into 32 bytes: low nibbles in lanes 0–15, high nibbles in lanes 16–31.
It then computes `_mm256_maddubs_epi16(qx, qy)` where `qy = Q8_0.qs[0..31]` (sequential).
This only computes the correct dot product `Σ q_i × y_i` when:
- `qx[0..15]` = weight elements 0..15 (low nibbles)
- `qx[16..31]` = weight elements 16..31 (high nibbles)

The interleaved layout put even-indexed elements in lanes 0–15 and odd-indexed in 16–31,
cross-multiplying every pair of weight-activation elements incorrectly.

### 2.2 Bug: Wrong Nibble Read Order in `dequantize_row_q4_hqq`

**File:** `ggml/src/ggml-quants.c`

**Root cause:** The original dequantize loop iterates per element (j=0..31) and reads
nibbles in the wrong order — high nibble for even elements, low nibble for odd elements —
which is the opposite of what split-half packing stores.

**Before (broken):**
```c
for (int j = 0; j < QK4_HQQ; j++) {
    uint8_t q = x[i].qs[j/2];
    int qi = (j % 2 == 0) ? (q >> 4) : (q & 0xF);   // reversed!
    y[i*QK4_HQQ + j] = (qi - zero) / scale;
}
```

**After (correct — matches split-half packing):**
```c
for (int j = 0; j < QK4_HQQ/2; ++j) {
    const int x0 = (x[i].qs[j] & 0x0F);   // element j
    const int x1 = (x[i].qs[j] >>    4);   // element j + QK4_HQQ/2
    y[i*QK4_HQQ + j]              = (x0 - zero) / scale;
    y[i*QK4_HQQ + j + QK4_HQQ/2] = (x1 - zero) / scale;
}
```

### 2.3 Verification: Vec_Dot Kernels Are Correct

The generic, AVX2, and NEON vec_dot implementations were audited and are **correct** for
split-half packing. No changes were needed to those kernels.

| Kernel | Status | Notes |
|--------|--------|-------|
| `ggml_vec_dot_q4_hqq_q8_0_generic` | Correct | Uses split-half convention |
| `ggml_vec_dot_q4_hqq_q8_0` (AVX2)  | Correct | `bytes_from_nibbles_32` + `maddubs` |
| `ggml_vec_dot_q4_hqq_q8_0` (NEON)  | Correct | `vandq/vshrq` split + `vdotq_s32` |

---

## 3. Methodology

### 3.1 Models

```
F16 source:   unsloth/Llama-3.2-3B-f16.gguf     (6.0 GiB, 16.0 BPW)
Q4_0 model:   unsloth/Llama-3.2-3B-q4_0.gguf    (1.8 GiB,  4.5 BPW effective)
Q4_HQQ model: quantized from F16, fixed code     (2.0 GiB,  5.19 BPW effective)
```

Note: effective BPW > nominal 5.0 bpw because embedding and normalization tensors
remain in F32 (1d tensors are not quantized).

### 3.2 Quantization

```bash
# Q4_HQQ (quantized in this session with fixed code)
build/bin/llama-quantize \
    temp/unsloth_Llama-3.2-3B-f16.gguf \
    models/unsloth_Llama-3.2-3B-q4_hqq.gguf \
    Q4_HQQ
```

### 3.3 Perplexity

```bash
build/bin/llama-perplexity \
    -m <model.gguf> \
    -f wikitext-2-raw/wiki.test.raw \
    -c 512 \
    --chunks 40 \
    -t 8
```

- **Context:** 512 tokens (standard WikiText-2 stride)
- **Chunks:** 40 × 512 = 20,480 tokens total
- **Threads:** 8 (full Xeon 8488C socket utilization)

---

## 4. Detailed Results

### 4.1 Final PPL Scores

| Model | PPL | ±σ | Chunks | Tokens | Load time | Eval time |
|-------|-----|-----|--------|--------|-----------|-----------|
| Q4_0   | **8.4419** | 0.211 | 40 | 20,480 | 0.72s | 5.33 min |
| Q4_HQQ | **8.4417** | 0.209 | 40 | 20,480 | 6.22s | 13.89 min |

**Perplexity difference: 0.0002 — statistically indistinguishable.**

### 4.2 Per-Chunk Running PPL (both models, first 32 chunks)

| Chunk | Q4_0 PPL | Q4_HQQ PPL | Δ |
|-------|----------|------------|---|
| 1  | 4.787 | 4.963 | +0.176 |
| 5  | 7.780 | 7.881 | +0.101 |
| 10 | 9.982 | 10.027 | +0.045 |
| 15 | 10.138 | 10.187 | +0.049 |
| 20 | 9.555 | 9.567 | +0.012 |
| 25 | 8.443 | 8.443 | 0.000 |
| 30 | 8.265 | 8.265 | 0.000 |
| 32 | 8.207 | 8.207 | 0.000 |

### 4.3 Throughput Comparison

| Model | PP tokens/s | Load time |
|-------|-------------|-----------|
| Q4_0   | 64.0 t/s | 0.72s |
| Q4_HQQ | 24.6 t/s | 6.22s |

**Q4_HQQ is 2.6× slower at prompt processing on this AMX hardware.** The AMX backend
has an optimized path for Q4_0 but falls through to the generic SIMD path for Q4_HQQ.
This is a performance gap to address in a future AMX-specific Q4_HQQ kernel.

---

## 5. Analysis

### 5.1 Perplexity: Q4_HQQ Matches Q4_0

Despite storing an extra 2 bytes (FP16 zero-point) per block:
- Q4_HQQ achieves **identical PPL** to Q4_0 on Llama-3.2-3B/WikiText-2
- The explicit zero-point neither helps nor hurts for typical LLM weight distributions
- Both schemes use 4 bits → 16 quantization levels per weight

**Why they converge:** Llama-3 weight tensors are approximately symmetric (zero-mean)
after RMSNorm. For symmetric distributions, the optimal zero-point ≈ 0, so Q4_HQQ
effectively degenerates to Q4_0. The extra zero-point byte is unused signal here.

**When Q4_HQQ would win:** Models with asymmetric weight distributions (non-zero-mean
biases or post-activation weights). The explicit zero-point allows capturing the full
dynamic range without centering around zero.

### 5.2 Bits-Per-Weight vs Quality Trade-off

| Type | Effective bpw | PPL (WikiText-2, 3B) | PPL/bpw |
|------|---------------|----------------------|---------|
| Q4_0   | 4.5 bpw | 8.44 | 1.876 |
| Q4_HQQ | 5.0 bpw | 8.44 | 1.688 |
| Q4_K_M | ~4.5 bpw | ~7.8 (typical) | ~1.73 |

For pure quality-per-bit efficiency on symmetric models, Q4_0 wins because it achieves
the same PPL at lower bit cost. Q4_HQQ trades the extra bits for zero-point flexibility
that isn't needed here.

### 5.3 Throughput Gap and AMX

The 2.6× slower prompt processing (24.6 vs 64.0 t/s) comes from:
1. The AMX backend has a pre-packed reuse path for Q4_0 (`graphs reused = 20` for Q4_0)
2. Q4_HQQ uses the generic SIMD path (not AMX-accelerated)
3. Q4_HQQ's dequant-then-accumulate requires `(q - zero) / scale` per block vs
   Q4_0's single multiply `(q - 8) × delta`

This is a temporary performance limitation — not a fundamental one. An AMX-specific
Q4_HQQ kernel or an imatrix-aware requantizer would close this gap.

---

## 6. Files Changed in This Commit

| File | Type | Description |
|------|------|-------------|
| `ggml/src/ggml-quants.c` | Bug fix | `quantize_row_q4_hqq_ref`: split-half nibble packing |
| `ggml/src/ggml-quants.c` | Bug fix | `dequantize_row_q4_hqq`: correct nibble read order |
| `docs/q4_hqq_perplexity_benchmark.md` | New | This report |
| `scripts/run_ppl_q4_hqq.sh` | New | Automated benchmark script |

---

## 7. Reproduction

### 7.1 Build

```bash
git clone https://github.com/ramaseshanms/llama.cpp.git -b feature/q4_hqq
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-quantize llama-perplexity -j$(nproc)
```

### 7.2 Get WikiText-2

```bash
bash scripts/get-wikitext-2.sh
```

### 7.3 Run Benchmark

```bash
# Automated
bash scripts/run_ppl_q4_hqq.sh \
    --f16-model /path/to/model-f16.gguf \
    --q4_0-model /path/to/model-q4_0.gguf

# Manual
build/bin/llama-quantize model-f16.gguf model-q4_hqq.gguf Q4_HQQ

build/bin/llama-perplexity -m model-q4_0.gguf   -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 -t 8
build/bin/llama-perplexity -m model-q4_hqq.gguf  -f wikitext-2-raw/wiki.test.raw -c 512 --chunks 40 -t 8
```

---

## 8. Future Work

1. **AMX kernel for Q4_HQQ** — Implement an AMX-specific matmul path to close the
   2.6× throughput gap vs Q4_0 on Intel CPUs with AMX.

2. **imatrix-weighted quantization** — `quantize_row_q4_hqq_impl` currently falls back
   to the reference path even with imatrix. Implementing importance-weighted scale/zero
   selection would improve PPL on models with asymmetric weight distributions.

3. **Asymmetric model evaluation** — Test on models where Q4_HQQ's zero-point should
   provide genuine benefit (e.g., models without RMSNorm, or after quantization-aware
   fine-tuning that introduces DC bias).

4. **Longer context PPL** — The 20K token evaluation is statistically sufficient for
   relative comparison but a full WikiText-2 run (2.46M tokens) would give tighter error
   bounds.
