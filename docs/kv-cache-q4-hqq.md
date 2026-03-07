# Q4_HQQ KV Cache Quantization

**Branch:** `feature/kv-cache-q4-hqq` (merged into `feature/q4_hqq`)
**Last updated:** 2026-03-07

---

## Overview

This document covers using `Q4_HQQ` and `Q4_HQQ_128` as KV cache quantization
types via the `--cache-type-k` / `--cache-type-v` CLI flags.

| Type | Group size | Block struct | bpw | Use case |
|------|-----------|-------------|-----|---------|
| `q4_hqq` | 32 | `{f16 scale, f16 zero, u8[16]}` = 20 B | 5.00 | default HQQ; same footprint as Q4_1 |
| `q4_hqq_128` | 128 | `{f16 scale, f16 zero, u8[64]}` = 68 B | 4.25 | paper default; lower bpw, larger alignment req. |

Both types are calibration-free: the HQQ half-quadratic proximal solver
finds an optimal affine mapping `x ≈ (q - zero) / scale` per block without
needing a calibration dataset.

---

## Quick-start

```bash
# K cache only (no -fa required)
llama-cli -m model.gguf -c 4096 -ctk q4_hqq -p "Hello"

# K+V cache — Flash Attention is REQUIRED for quantised V
llama-cli -m model.gguf -c 4096 -ctk q4_hqq -ctv q4_hqq -fa -p "Hello"

# 128-element group variant (slightly lower bpw)
llama-cli -m model.gguf -c 4096 -ctk q4_hqq_128 -p "Hello"
llama-cli -m model.gguf -c 4096 -ctk q4_hqq_128 -ctv q4_hqq_128 -fa -p "Hello"
```

> **Warning** — output quality caveat: Q4_HQQ uses an affine zero-point whose
> DC bias accumulates across attention heads.  On models with symmetric
> activation distributions (Llama-3 family) this causes noticeable coherence
> degradation even at short context lengths.  See the
> [Quality Analysis](#quality-analysis) section before using in production.

---

## Changes Made (branch `feature/kv-cache-q4-hqq`)

### Commit 1 — `src/llama-context.cpp`

**Bug: block-size validation skipped when `-fa` was explicit**

`llama_init_from_model()` checked that `n_embd_head_{k,v} % blck_size == 0`
only when `flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO`.  When the user
passed `-fa` explicitly (`LLAMA_FLASH_ATTN_TYPE_ENABLED`) the guard was
silently skipped, meaning a misaligned type (e.g. `q4_hqq_128` on a 64-dim
head) would pass validation and crash at runtime.

**Fix:** condition changed from `== AUTO` to `!= DISABLED` so both AUTO and
ENABLED paths are protected.

```diff
-    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO && ggml_is_quantized(params.type_k)) {
+    // Guard both AUTO and ENABLED; only skip when FA is explicitly off.
+    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
```

**Bug: V-cache error message printed `n_embd_head_k` instead of `n_embd_head_v`**

```diff
-            LLAMA_LOG_ERROR("... does not divide n_embd_head_k=%u\n", ..., model->hparams.n_embd_head_v);
+            LLAMA_LOG_ERROR("... does not divide n_embd_head_v=%u\n", ..., model->hparams.n_embd_head_v);
```

**New: quality-degradation warning for HQQ KV types**

A `LLAMA_LOG_WARN` is emitted at context creation time when `type_k` or
`type_v` is `Q4_HQQ` or `Q4_HQQ_128`, informing the user of the potential
quality issue without hard-blocking experimental use.

---

### Commit 2 — `common/arg.cpp`

`GGML_TYPE_Q4_HQQ_128` added to the `kv_cache_types` vector so that
`-ctk q4_hqq_128` and `-ctv q4_hqq_128` are accepted by the CLI.

`Q4_HQQ` was already in the list (added in commit `b786774e9`); this commit
completes the HQQ pair.

`kv_cache_type_from_str()` works via `ggml_type_name()` → `"q4_hqq_128"` so
no changes to parsing logic are needed.

---

### Commit 3 — `tools/llama-bench/llama-bench.cpp`

`llama-bench` has its own static `ggml_type_from_name()` function with
hardcoded string comparisons.  This function is independent of `common/arg.cpp`
and must be updated manually.

`"q4_hqq_128"` added, enabling benchmark comparisons like:

```bash
./build/bin/llama-bench -m model.gguf -ctk f16
./build/bin/llama-bench -m model.gguf -ctk q4_hqq
./build/bin/llama-bench -m model.gguf -ctk q4_hqq_128
./build/bin/llama-bench -m model.gguf -ctk q4_hqq -ctv q4_hqq -fa 1
./build/bin/llama-bench -m model.gguf -ctk q4_hqq_128 -ctv q4_hqq_128 -fa 1
```

---

## Architecture — Why These Changes Are Sufficient

The Q4_HQQ quantize/dequantize pipeline was already fully implemented on the
`feature/q4_hqq` branch.  The table below confirms that every layer required
for KV cache use was already wired before this sub-branch was created.

| Component | File | Status |
|-----------|------|--------|
| Block struct `block_q4_hqq` (20 B, g32) | `ggml/src/ggml-common.h` | ✅ present |
| Block struct `block_q4_hqq_128` (68 B, g128) | `ggml/src/ggml-common.h` | ✅ present |
| Type traits (blck_size, type_size, to_float, from_float_ref) | `ggml/src/ggml.c` | ✅ present |
| CPU kernel dispatch (from_float, vec_dot, vec_dot_type) | `ggml/src/ggml-cpu/ggml-cpu.c` | ✅ present |
| AVX2 / NEON SIMD dot-product | `ggml/src/ggml-cpu/quants.c` | ✅ present |
| `quantize_q4_hqq` / `quantize_q4_hqq_128` | `ggml/src/ggml-quants.c` | ✅ present |
| `dequantize_row_q4_hqq` / `_128` | `ggml/src/ggml-quants.c` | ✅ present |
| Chunk-quantize dispatch | `ggml/src/ggml.c` | ✅ present |
| `ggml_compute_forward_{add,add1,acc,out_prod,set,get_rows,clamp}` cases | `ggml/src/ggml-cpu/ops.cpp` | ✅ present |
| `GGML_FTYPE_MOSTLY_Q4_HQQ` enum + ftype↔type mapping | `include/llama.h`, `src/llama-quant.cpp` | ✅ present |
| KV cache tensor allocation (`ggml_new_tensor_3d` with any ggml_type) | `src/llama-kv-cache.cpp` | ✅ present (generic) |
| Block-size alignment validation at context init | `src/llama-context.cpp` | ✅ present (this branch) |

---

## Memory Usage

Measured on Intel Xeon Platinum 8488C, Llama-3.2-3B, `c=4096`
(28 layers, 8 GQA KV heads, 128-dim head).

### Per-context memory (MiB)

| KV type | K only | K+V (with -fa) | vs F16 K+V |
|---------|--------|----------------|-----------|
| `f16` | 448 | 896 | baseline |
| `bf16` | 448 | 896 | 1.00× |
| `q8_0` | 238 | 476 | 1.88× savings |
| `q4_0` | 126 | 252 | 3.56× savings |
| `q4_1` | 140 | 280 | 3.20× savings |
| **`q4_hqq`** | **140** | **280** | **3.20× savings** |
| **`q4_hqq_128`** | **119** | **238** | **3.76× savings** |
| `q5_0` | 154 | 308 | 2.91× savings |
| `q5_1` | 168 | 336 | 2.67× savings |

`Q4_HQQ` (g32) has the **same footprint as Q4_1** — both are 20 bytes per 32
elements (5.0 bpw).  The difference is the quantisation algorithm: Q4_1 uses
a simple min-max affine mapping, Q4_HQQ uses the half-quadratic proximal solver
which minimises reconstruction error more precisely.

`Q4_HQQ_128` (g128) achieves **4.25 bpw** — the lowest of any 4-bit KV type
listed here — because metadata overhead (scale + zero-point) is amortised
over 128 elements instead of 32.

### Multi-context scaling (Llama-3.2-3B, K+V combined)

| Context tokens | F16 | Q4_HQQ | Q4_HQQ_128 |
|---------------|-----|--------|------------|
| 512 | 112 MiB | 35 MiB | 30 MiB |
| 2,048 | 448 MiB | 140 MiB | 119 MiB |
| 8,192 | 1,792 MiB | 560 MiB | 476 MiB |
| 32,768 | 7,168 MiB | 2,240 MiB | 1,904 MiB |
| 131,072 | ~28 GiB | ~8.8 GiB | ~7.5 GiB |

---

## Block Structure

### Q4_HQQ (group size 32)

```c
// ggml/src/ggml-common.h
#define QK4_HQQ 32

typedef struct {
    ggml_fp16_t scale; // FP16 — affine scale:   x ≈ (q - zero) / scale
    ggml_fp16_t zero;  // FP16 — affine zero-pt (stored as integer zero in INT8 form)
    uint8_t qs[QK4_HQQ/2]; // 4-bit packed nibbles, split-half layout
} block_q4_hqq;
// static_assert(sizeof(block_q4_hqq) == 20);
```

**Dequantize formula:**  `x[i] = ((q[i] - zero) * scale)` where `q[i] ∈ [0, 15]`.

### Q4_HQQ_128 (group size 128)

```c
#define QK4_HQQ_128 128

typedef struct {
    ggml_fp16_t scale;
    ggml_fp16_t zero;
    uint8_t qs[QK4_HQQ_128/2]; // 64 bytes of packed nibbles
} block_q4_hqq_128;
// static_assert(sizeof(block_q4_hqq_128) == 68);
```

Same dequantize formula; only the group size differs.

---

## Quality Analysis

> Based on empirical measurements documented in
> `docs/kv-cache-q4-hqq-compression-report.md`.

### Root cause of quality degradation

The HQQ affine form `x ≈ (q - zero) / scale` uses an explicit zero-point.
When activation distributions are **symmetric around zero** (as they are in
Llama-3 K/V projections after layer normalisation), an ideal quantiser would
have `zero ≈ 7` (the midpoint of `[0,15]`) and `scale ≈ range/16`.

The HQQ proximal solver minimises reconstruction error on the *training
distribution*, but at inference time the K/V activations can deviate from
that distribution.  The per-block zero-point introduces a DC offset that
**accumulates across all attention heads and all layers** during the
softmax-weighted V accumulation.  This compounding bias causes output tokens
to become repetitive or incoherent even at 6-token context lengths on Llama-3.

### Measured throughput impact (PP, c=256, AMX backend)

| KV type | PP tokens/s | Relative |
|---------|------------|---------|
| F16 | 105.9 | baseline |
| Q8_0 | 99.1 | −6.4% |
| Q4_0 | 95.3 | −10% |
| Q4_HQQ | 90.0 | −15% |

The ~15% PP overhead vs F16 comes from the division in the HQQ dequantize
formula (`(q - zero) / scale`) vs the multiply in Q4_0 (`q * delta`).
Token-generation (TG) is not memory-bandwidth-bound at short context on CPU,
so TG throughput is within 1% across all types.

### Recommendation

| Scenario | Recommended KV type |
|----------|-------------------|
| Best quality, memory not constrained | `f16` or `bf16` |
| Good quality + memory saving | `q8_0` |
| Aggressive memory saving, quality acceptable | `q4_0` |
| Maximum memory saving (experimental) | `q4_hqq_128` |
| Production Llama-3 | **avoid Q4_HQQ / Q4_HQQ_128** |

Q4_HQQ is better suited as a **model weight quantisation** format (not KV
cache) where the HQQ solver's calibration-free accuracy advantage over Q4_0
outweighs the dequantize cost.

---

## Constraints

| Constraint | Detail |
|-----------|--------|
| Flash Attention for V cache | `-fa` required when `-ctv q4_hqq` or `-ctv q4_hqq_128` is set. Hard error at context init if violated. |
| Block-size alignment (K) | `n_embd_head_k % QK4_HQQ == 0`. For g32: head sizes 32, 64, 96, 128, 256 all pass. For g128: head sizes 128, 256 pass; 64 does **not**. |
| Block-size alignment (V) | Same rule applied to `n_embd_head_v`. |
| No sgemm fast path | Q4_HQQ is not in `ggml-cpu/llamafile/sgemm.cpp`; falls back to `vec_dot`. Correct but potentially slower for large batch sizes. |
| KV defrag | Uses `ggml_cpy` internally — handles any quantised type generically. No special casing needed. |
| Saving/loading KV cache | KV cache is saved/loaded as raw bytes. Quantised types are preserved correctly on reload. |

---

## Testing

```bash
# Build
cmake -B build -DLLAMA_NATIVE=ON && cmake --build build -j$(nproc)

# Verify flags appear in help
./build/bin/llama-cli --help | grep -A6 "cache-type-k"

# Smoke test: K cache only (no -fa needed)
./build/bin/llama-cli -m model.gguf -c 2048 -ctk q4_hqq    -p "The capital of France is" -n 10
./build/bin/llama-cli -m model.gguf -c 2048 -ctk q4_hqq_128 -p "The capital of France is" -n 10

# Smoke test: K+V cache (Flash Attention required)
./build/bin/llama-cli -m model.gguf -c 2048 -ctk q4_hqq     -ctv q4_hqq     -fa -p "Hello" -n 10
./build/bin/llama-cli -m model.gguf -c 2048 -ctk q4_hqq_128 -ctv q4_hqq_128 -fa -p "Hello" -n 10

# Verify quality-degradation warning appears in stderr
./build/bin/llama-cli -m model.gguf -ctk q4_hqq -p "test" -n 1 2>&1 | grep -i "HQQ affine"

# Benchmark: compare all relevant KV types
./build/bin/llama-bench -m model.gguf -ctk f16        -c 2048
./build/bin/llama-bench -m model.gguf -ctk q8_0       -c 2048
./build/bin/llama-bench -m model.gguf -ctk q4_0       -c 2048
./build/bin/llama-bench -m model.gguf -ctk q4_hqq     -c 2048
./build/bin/llama-bench -m model.gguf -ctk q4_hqq_128 -c 2048
./build/bin/llama-bench -m model.gguf -ctk q4_hqq     -ctv q4_hqq     -fa 1 -c 2048
./build/bin/llama-bench -m model.gguf -ctk q4_hqq_128 -ctv q4_hqq_128 -fa 1 -c 2048

# Verify alignment guard fires for misaligned head size
# (Q4_HQQ_128 requires n_embd_head % 128 == 0; models with 64-dim heads should fail)
./build/bin/llama-cli -m model-64d-head.gguf -ctk q4_hqq_128 -p "test" -n 1 2>&1 | grep -i "does not divide"
```

---

## Related Files

| File | Purpose |
|------|---------|
| `docs/kv-cache-q4-hqq.md` | This document |
| `docs/kv-cache-q4-hqq-compression-report.md` | Empirical measurement report with full tables |
| `docs/q4_hqq.md` | Q4_HQQ weight quantisation overview |
| `docs/q4_hqq_simd_optimization.md` | AVX2/NEON kernel design notes |
| `ggml/src/ggml-common.h` | `block_q4_hqq`, `block_q4_hqq_128` struct definitions |
| `ggml/src/ggml-quants.c` | Quantize/dequantize implementations |
| `ggml/src/ggml-cpu/quants.c` | SIMD dot-product kernels |
| `ggml/src/ggml-cpu/ops.cpp` | Tensor op dispatch (add, get_rows, out_prod, …) |
| `common/arg.cpp` | `kv_cache_types` whitelist for CLI flags |
| `tools/llama-bench/llama-bench.cpp` | Benchmark type parser |
| `src/llama-context.cpp` | Context init: validation, warnings, KV cache setup |
