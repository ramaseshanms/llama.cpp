/**
 * bench_q4_hqq_dot.c  —  Standalone micro-benchmark
 *
 * Compares scalar vs AVX2 SIMD implementations of the
 * ggml_vec_dot_q4_hqq_q8_0 kernel.
 *
 * Compile (from repo root):
 *   gcc -O3 -mavx2 -mfma -march=native \
 *       tests/bench_q4_hqq_dot.c -o /tmp/bench_q4_hqq_dot -lm && \
 *   /tmp/bench_q4_hqq_dot
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Minimal FP16 support (IEEE 754 half, software conversion)            */
/* ------------------------------------------------------------------ */
typedef uint16_t fp16_t;

static inline float fp16_to_fp32(fp16_t h) {
    /* IEEE 754 binary16 -> binary32 */
    uint32_t sign     = (uint32_t)(h >> 15) << 31;
    uint32_t exponent = (uint32_t)((h >> 10) & 0x1F);
    uint32_t mantissa = (uint32_t)(h & 0x3FF);
    uint32_t f32;
    if (exponent == 0) {
        /* subnormal */
        if (mantissa == 0) { f32 = sign; }
        else {
            exponent = 127 - 14;
            while (!(mantissa & (1 << 10))) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;
            f32 = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        /* inf / nan */
        f32 = sign | 0x7F800000 | (mantissa << 13);
    } else {
        f32 = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &f32, 4);
    return result;
}

static inline fp16_t fp32_to_fp16(float f) {
    uint32_t f32;
    memcpy(&f32, &f, 4);
    uint32_t sign     = f32 >> 31;
    int32_t  exp      = (int32_t)((f32 >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = f32 & 0x7FFFFF;
    if (exp <= 0) { return (fp16_t)(sign << 15); }
    if (exp >= 31) { return (fp16_t)((sign << 15) | 0x7C00); }
    return (fp16_t)((sign << 15) | ((uint32_t)exp << 10) | (mantissa >> 13));
}

/* ------------------------------------------------------------------ */
/* Block types (mirrors ggml-common.h exactly)                          */
/* ------------------------------------------------------------------ */
#define QK4_HQQ 32
#define QK8_0   32

typedef struct { fp16_t scale; fp16_t zero; uint8_t qs[QK4_HQQ/2]; } block_q4_hqq;
typedef struct { fp16_t d;                  int8_t  qs[QK8_0];     } block_q8_0;

/* ------------------------------------------------------------------ */
/* AVX2 helpers                                                         */
/* ------------------------------------------------------------------ */
#if defined(__AVX2__)
#include <immintrin.h>

static inline float hsum_float_8(const __m256 x) {
    __m128 hi  = _mm256_extractf128_ps(x, 1);
    __m128 lo  = _mm256_castps256_ps128(x);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    return _mm_cvtss_f32(sum);
}

/* unpack 16 packed nibble bytes -> 32 uint8 each in [0,15]
 * low 128  = low nibbles  (qs[j] & 0xF) = q0, paired with y[0..15]
 * high 128 = high nibbles (qs[j] >> 4)  = q1, paired with y[16..31]
 * Matches bytes_from_nibbles_32 in arch/x86/quants.c. */
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    /* insertf128(castsi256(tmp), srli4, 1):
     *   low 128  = tmp          -> low nibbles  q0
     *   high 128 = srli(tmp,4)  -> high nibbles q1 */
    const __m256i bytes = _mm256_insertf128_si256(
        _mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    return _mm256_and_si256(_mm256_set1_epi8(0xF), bytes);
}
#endif  /* __AVX2__ */

/* ------------------------------------------------------------------ */
/* Scalar kernel  (ggml_vec_dot_q4_hqq_q8_0_generic)                   */
/* ------------------------------------------------------------------ */
static void __attribute__((noinline))
dot_scalar(int n, float * s, const block_q4_hqq * x, const block_q8_0 * y) {
    const int nb = n / QK8_0;
    float sumf = 0.0f;
    for (int ib = 0; ib < nb; ++ib) {
        const float scale  = fp16_to_fp32(x[ib].scale);
        const float zero   = fp16_to_fp32(x[ib].zero);
        const float factor = fp16_to_fp32(y[ib].d) / scale;
        int sumi_qy = 0, sumi_y = 0;
        for (int j = 0; j < 16; ++j) {
            const int q0 = x[ib].qs[j] & 0xF;
            const int q1 = x[ib].qs[j] >> 4;
            sumi_qy += q0 * (int)y[ib].qs[j]      + q1 * (int)y[ib].qs[j + 16];
            sumi_y  += (int)y[ib].qs[j] + (int)y[ib].qs[j + 16];
        }
        sumf += factor * ((float)sumi_qy - zero * (float)sumi_y);
    }
    *s = sumf;
}

/* ------------------------------------------------------------------ */
/* AVX2 SIMD kernel  (ggml_vec_dot_q4_hqq_q8_0)                        */
/* ------------------------------------------------------------------ */
static void __attribute__((noinline))
dot_avx2(int n, float * s, const block_q4_hqq * x, const block_q8_0 * y) {
    const int nb = n / QK8_0;
    int ib = 0;
    float sumf = 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (; ib < nb; ++ib) {
        const float scale  = fp16_to_fp32(x[ib].scale);
        const float zero   = fp16_to_fp32(x[ib].zero);
        const float factor = fp16_to_fp32(y[ib].d) / scale;

        /* Unpack nibbles -> 32 uint8 in [0,15] */
        const __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        /* Load 32 signed int8 activations */
        const __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        /* sum(q*y):  maddubs(u8,s8)->s16, madd(s16,1)->s32, cvt->f32 */
        const __m256i dot16 = _mm256_maddubs_epi16(qx, qy);
        const __m256 vdot   = _mm256_cvtepi32_ps(_mm256_madd_epi16(dot16, ones16));

        /* sum(y): sign-extend s8->s16 (lo+hi), element-wise add, madd->s32 */
        const __m128i qy_lo = _mm256_castsi256_si128(qy);
        const __m128i qy_hi = _mm256_extracti128_si256(qy, 1);
        const __m256i sy16  = _mm256_add_epi16(_mm256_cvtepi8_epi16(qy_lo),
                                                _mm256_cvtepi8_epi16(qy_hi));
        const __m256 vsum_y = _mm256_cvtepi32_ps(_mm256_madd_epi16(sy16, ones16));

        /* acc += factor * (sum(q*y) - zero * sum(y)) */
        acc = _mm256_fmadd_ps(_mm256_set1_ps(factor),
                              _mm256_sub_ps(vdot, _mm256_mul_ps(_mm256_set1_ps(zero), vsum_y)),
                              acc);
    }
    sumf = hsum_float_8(acc);
#endif

    /* scalar tail / non-AVX2 path */
    for (; ib < nb; ++ib) {
        const float scale  = fp16_to_fp32(x[ib].scale);
        const float zero   = fp16_to_fp32(x[ib].zero);
        const float factor = fp16_to_fp32(y[ib].d) / scale;
        int sumi_qy = 0, sumi_y = 0;
        for (int j = 0; j < 16; ++j) {
            const int q0 = x[ib].qs[j] & 0xF;
            const int q1 = x[ib].qs[j] >> 4;
            sumi_qy += q0 * (int)y[ib].qs[j]      + q1 * (int)y[ib].qs[j + 16];
            sumi_y  += (int)y[ib].qs[j] + (int)y[ib].qs[j + 16];
        }
        sumf += factor * ((float)sumi_qy - zero * (float)sumi_y);
    }
    *s = sumf;
}

/* ------------------------------------------------------------------ */
/* RNG + data generation                                               */
/* ------------------------------------------------------------------ */
static uint32_t lcg(uint32_t * s) { return (*s = *s * 1664525u + 1013904223u); }

static void fill_x(block_q4_hqq * b, int nb, uint32_t * rng) {
    for (int i = 0; i < nb; ++i) {
        b[i].scale = fp32_to_fp16(0.5f + (lcg(rng) & 0xFF) / 170.0f);
        b[i].zero  = fp32_to_fp16((float)(lcg(rng) & 0xF));
        for (int j = 0; j < QK4_HQQ/2; ++j)
            b[i].qs[j] = (uint8_t)((lcg(rng) & 0xF) | ((lcg(rng) & 0xF) << 4));
    }
}

static void fill_y(block_q8_0 * b, int nb, uint32_t * rng) {
    for (int i = 0; i < nb; ++i) {
        b[i].d = fp32_to_fp16(0.005f + (lcg(rng) & 0xFF) / 25500.0f);
        for (int j = 0; j < QK8_0; ++j)
            b[i].qs[j] = (int8_t)(lcg(rng));
    }
}

/* ------------------------------------------------------------------ */
/* Timer                                                                */
/* ------------------------------------------------------------------ */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ------------------------------------------------------------------ */
/* Main                                                                 */
/* ------------------------------------------------------------------ */
#define N_ELEMENTS    (1 << 14)   /* 16384 elements = 512 blocks       */
#define BENCH_WARMUP  500
#define BENCH_ITERS   5000
#define N_TRIALS      8

int main(void) {
    const int nb = N_ELEMENTS / QK8_0;

    block_q4_hqq * bx = (block_q4_hqq *)calloc(nb, sizeof(block_q4_hqq));
    block_q8_0   * by = (block_q8_0   *)calloc(nb, sizeof(block_q8_0));
    if (!bx || !by) { fputs("OOM\n", stderr); return 1; }

    uint32_t rng = 0xCAFEBABE;
    fill_x(bx, nb, &rng);
    fill_y(by, nb, &rng);

    /* --- header ---------------------------------------------------- */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Q4_HQQ dot product kernel benchmark                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  n = %d elements  |  %d blocks  |  warmup=%d  iters=%d\n",
           N_ELEMENTS, nb, BENCH_WARMUP, BENCH_ITERS);
#if defined(__AVX2__)
    printf("  ISA : AVX2 + FMA  (256-bit, 8 float32 lanes)\n");
#else
    printf("  ISA : scalar only  (AVX2 unavailable at compile time)\n");
#endif
    printf("  sizeof(block_q4_hqq) = %zu  |  sizeof(block_q8_0) = %zu\n",
           sizeof(block_q4_hqq), sizeof(block_q8_0));

    /* --- correctness ----------------------------------------------- */
    {
        float rs = 0.0f, rv = 0.0f;
        dot_scalar(N_ELEMENTS, &rs, bx, by);
        dot_avx2  (N_ELEMENTS, &rv, bx, by);
        float rel = fabsf(rv - rs) / (fabsf(rs) + 1e-8f);
        printf("\n  Correctness: scalar=%.7g  avx2=%.7g  rel_err=%.2e  [%s]\n",
               rs, rv, rel, rel < 2e-5f ? "PASS" : "FAIL");
    }

    /* --- single-run timing ----------------------------------------- */
    {
        float dummy = 0.0f;

        /* warmup */
        for (int i = 0; i < BENCH_WARMUP; ++i) dot_scalar(N_ELEMENTS, &dummy, bx, by);
        double t0 = now_sec();
        for (int i = 0; i < BENCH_ITERS; ++i) dot_scalar(N_ELEMENTS, &dummy, bx, by);
        double scalar_s = now_sec() - t0;

        for (int i = 0; i < BENCH_WARMUP; ++i) dot_avx2(N_ELEMENTS, &dummy, bx, by);
        t0 = now_sec();
        for (int i = 0; i < BENCH_ITERS; ++i) dot_avx2(N_ELEMENTS, &dummy, bx, by);
        double avx2_s = now_sec() - t0;

        double bytes_per_call =
            (double)nb * (sizeof(block_q4_hqq) + sizeof(block_q8_0));

        printf("\n");
        printf("  ┌─────────────────────┬────────────┬────────────┬───────────┐\n");
        printf("  │ Kernel              │  ns/call   │ M calls/s  │  GB/s     │\n");
        printf("  ├─────────────────────┼────────────┼────────────┼───────────┤\n");
        printf("  │ scalar (generic)    │ %10.2f │ %10.3f │ %9.3f │\n",
               scalar_s / BENCH_ITERS * 1e9,
               BENCH_ITERS / scalar_s / 1e6,
               bytes_per_call * BENCH_ITERS / scalar_s / 1e9);
        printf("  │ AVX2  (simd)        │ %10.2f │ %10.3f │ %9.3f │\n",
               avx2_s / BENCH_ITERS * 1e9,
               BENCH_ITERS / avx2_s / 1e6,
               bytes_per_call * BENCH_ITERS / avx2_s / 1e9);
        printf("  └─────────────────────┴────────────┴────────────┴───────────┘\n");
        printf("\n  Speedup (scalar → AVX2): %.2fx\n", scalar_s / avx2_s);
    }

    /* --- multi-trial sweep ----------------------------------------- */
    printf("\n  --- %d-trial sweep (different random inputs) ---\n", N_TRIALS);
    double scalar_sum = 0.0, avx2_sum = 0.0;
    printf("  %-6s  %-12s  %-12s  %-10s  %-8s\n",
           "Trial", "scalar(ns)", "avx2(ns)", "speedup", "rel_err");
    printf("  %s\n", "──────────────────────────────────────────────────────");

    for (int t = 0; t < N_TRIALS; ++t) {
        fill_x(bx, nb, &rng);
        fill_y(by, nb, &rng);

        float rs = 0.0f, rv = 0.0f;
        double t0, scalar_ns, avx2_ns;

        /* scalar */
        for (int i = 0; i < BENCH_WARMUP; ++i) dot_scalar(N_ELEMENTS, &rs, bx, by);
        t0 = now_sec();
        for (int i = 0; i < BENCH_ITERS; ++i) dot_scalar(N_ELEMENTS, &rs, bx, by);
        scalar_ns = (now_sec() - t0) / BENCH_ITERS * 1e9;

        /* avx2 */
        for (int i = 0; i < BENCH_WARMUP; ++i) dot_avx2(N_ELEMENTS, &rv, bx, by);
        t0 = now_sec();
        for (int i = 0; i < BENCH_ITERS; ++i) dot_avx2(N_ELEMENTS, &rv, bx, by);
        avx2_ns = (now_sec() - t0) / BENCH_ITERS * 1e9;

        float rel = fabsf(rv - rs) / (fabsf(rs) + 1e-8f);
        printf("  %-6d  %-12.2f  %-12.2f  %-10.2fx  %.2e\n",
               t + 1, scalar_ns, avx2_ns, scalar_ns / avx2_ns, rel);

        scalar_sum += scalar_ns;
        avx2_sum   += avx2_ns;
    }

    printf("  %s\n", "──────────────────────────────────────────────────────");
    printf("  %-6s  %-12.2f  %-12.2f  %-10.2fx\n", "avg",
           scalar_sum / N_TRIALS, avx2_sum / N_TRIALS,
           scalar_sum / avx2_sum);
    printf("\n");

    free(bx);
    free(by);
    return 0;
}
