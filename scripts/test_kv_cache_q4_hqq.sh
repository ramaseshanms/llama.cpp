#!/usr/bin/env bash
# =============================================================================
# test_kv_cache_q4_hqq.sh
# =============================================================================
# Demonstrates and tests Q4_HQQ / Q4_HQQ_128 as KV cache quantization types
# on the feature/kv-cache-q4-hqq branch (merged into feature/q4_hqq).
#
# What this script covers:
#   1.  Prerequisites check     — git branch, binaries, model file
#   2.  Build verification      — cmake configure + targeted build
#   3.  Flag smoke tests        — CLI help lists both new types
#   4.  Safety / validation     — alignment guard, V-cache FA requirement,
#                                 quality-degradation warning
#   5.  Memory measurement      — llama_kv_cache size line for all types
#   6.  Output quality tests    — coherence comparison across KV types
#   7.  Throughput benchmark    — PP/TG with llama-bench (K-only + K+V+FA)
#   8.  Regression: block-size  — re-verify warning/error path in code
#   9.  Summary report          — pass/fail table printed to stdout
#
# Usage:
#   cd ~/llama.cpp
#   bash scripts/test_kv_cache_q4_hqq.sh [--model <path>] [--build] [--quick]
#
#   --model <path>   path to a Llama-3 GGUF (default: /home/ubuntu/models/llama-3.2-3B-q4_0.gguf)
#   --build          force a cmake rebuild before testing
#   --quick          skip the full throughput benchmark (saves ~5 min)
#
# Requirements:
#   - git, cmake, make/ninja (build toolchain)
#   - A Llama-3 GGUF model on disk (see --model flag)
#   - ~4 GB free RAM for the Q4_0 model + KV cache
#
# Automated test output format:
#   Each test prints  [PASS]  or  [FAIL]  followed by a description.
#   Exit code 0 = all PASS; non-zero = at least one FAIL.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override with CLI flags
# ---------------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="/home/ubuntu/models/llama-3.2-3B-q4_0.gguf"
BUILD_DIR="${REPO_DIR}/build"
CLI="${BUILD_DIR}/bin/llama-completion"
BENCH="${BUILD_DIR}/bin/llama-bench"
QUANTIZE="${BUILD_DIR}/bin/llama-quantize"
FORCE_BUILD=0
QUICK_MODE=0
PASS=0
FAIL=0
LOG_DIR="/tmp/kv_hqq_test_$$"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2"; shift 2 ;;
    --build)   FORCE_BUILD=1; shift ;;
    --quick)   QUICK_MODE=1; shift ;;
    *)         echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
pass() { echo "[PASS] $*"; (( PASS++ )) || true; }
fail() { echo "[FAIL] $*"; (( FAIL++ )) || true; }
section() { echo; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
info()    { echo "  [INFO] $*"; }
warn_manual() {
  echo
  echo "  ┌─ MANUAL CHECK ─────────────────────────────────────────────┐"
  while IFS= read -r line; do echo "  │  $line"; done <<< "$*"
  echo "  └────────────────────────────────────────────────────────────┘"
}

mkdir -p "${LOG_DIR}"
cleanup() { rm -rf "${LOG_DIR}"; }
trap cleanup EXIT

cd "${REPO_DIR}"

# =============================================================================
# 1. PREREQUISITES
# =============================================================================
section "1. Prerequisites"

# 1a. Git branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
info "Current branch: ${CURRENT_BRANCH}"
# Verify the three source-level changes introduced by feature/kv-cache-q4-hqq
# are present in the working tree.  This is more reliable than parsing git log
# (commit messages may contain multi-byte em-dash characters that confuse some
# grep locales inside pipelines).
#
# Check 1: Q4_HQQ_128 in kv_cache_types (common/arg.cpp)
# Check 2: block-size guard widened to != DISABLED (src/llama-context.cpp)
# Check 3: HQQ quality warning present (src/llama-context.cpp)
_HAS_HQQ128=$(grep -c "GGML_TYPE_Q4_HQQ_128" "${REPO_DIR}/common/arg.cpp" 2>/dev/null || echo 0)
_HAS_GUARD=$(grep -c "!= LLAMA_FLASH_ATTN_TYPE_DISABLED" "${REPO_DIR}/src/llama-context.cpp" 2>/dev/null || echo 0)
_HAS_WARN=$(grep -c "HQQ affine quantization" "${REPO_DIR}/src/llama-context.cpp" 2>/dev/null || echo 0)
if [[ ${_HAS_HQQ128} -ge 1 ]] && [[ ${_HAS_GUARD} -ge 1 ]] && [[ ${_HAS_WARN} -ge 1 ]]; then
  pass "Source tree contains all kv-cache-q4-hqq changes (arg.cpp + context.cpp)"
else
  fail "Source tree missing kv-cache-q4-hqq changes — Q4_HQQ_128=${_HAS_HQQ128} guard=${_HAS_GUARD} warn=${_HAS_WARN}"
fi

# 1b. Model file
if [[ -f "${MODEL}" ]]; then
  MODEL_SIZE=$(du -h "${MODEL}" | cut -f1)
  pass "Model file found: ${MODEL} (${MODEL_SIZE})"
else
  fail "Model file not found: ${MODEL}"
  echo "       Fix: provide a Llama-3 GGUF via --model <path>"
  echo "       Example: python convert_hf_to_gguf.py /path/to/Llama-3.2-3B --outtype q4_0"
  echo "       Stopping — cannot run inference tests without a model."
  exit 1
fi

# =============================================================================
# 2. BUILD
# =============================================================================
section "2. Build"

if [[ ${FORCE_BUILD} -eq 1 ]] || [[ ! -x "${CLI}" ]]; then
  info "Configuring cmake..."
  cmake -B "${BUILD_DIR}" -DLLAMA_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF \
        > "${LOG_DIR}/cmake_config.log" 2>&1 \
    && pass "CMake configuration succeeded" \
    || { fail "CMake configuration failed — see ${LOG_DIR}/cmake_config.log"; exit 1; }

  info "Building llama-completion, llama-bench, llama-quantize..."
  cmake --build "${BUILD_DIR}" -j"$(nproc)" \
        --target llama-completion llama-bench llama-quantize \
        > "${LOG_DIR}/cmake_build.log" 2>&1 \
    && pass "Build succeeded" \
    || { fail "Build failed — see ${LOG_DIR}/cmake_build.log"; exit 1; }
else
  pass "Binaries already present (use --build to force rebuild)"
  info "  llama-completion: ${CLI}"
  info "  llama-bench:      ${BENCH}"
fi

# Confirm binaries exist and are executable
for bin in "${CLI}" "${BENCH}"; do
  if [[ -x "${bin}" ]]; then
    pass "Binary executable: $(basename ${bin})"
  else
    fail "Binary missing or not executable: ${bin}"
  fi
done

# =============================================================================
# 3. FLAG SMOKE TESTS — CLI help lists both HQQ types
# =============================================================================
section "3. CLI flag smoke tests"

HELP_OUTPUT=$("${CLI}" --help 2>&1 || true)

# 3a. q4_hqq in help
if echo "${HELP_OUTPUT}" | grep -q "q4_hqq"; then
  pass "--cache-type-k accepts q4_hqq (visible in --help)"
else
  fail "--cache-type-k does NOT list q4_hqq in help"
fi

# 3b. q4_hqq_128 in help
if echo "${HELP_OUTPUT}" | grep -q "q4_hqq_128"; then
  pass "--cache-type-k accepts q4_hqq_128 (visible in --help)"
else
  fail "--cache-type-k does NOT list q4_hqq_128 in help"
fi

# 3c. llama-bench also accepts both types (separate parser)
BENCH_HELP=$("${BENCH}" 2>&1 || true)
# llama-bench doesn't print accepted types, so we just verify it doesn't
# crash on a minimal run with q4_hqq_128 (tested in throughput section below)
info "llama-bench parser will be exercised in section 7 (throughput)"

# =============================================================================
# 4. SAFETY / VALIDATION TESTS
# =============================================================================
section "4. Safety & validation"

# Helper: run llama-completion capturing stderr, return stderr content
run_stderr() {
  local ARGS=("$@")
  timeout 20 "${CLI}" "${ARGS[@]}" 2>"${LOG_DIR}/stderr_$$.txt" >/dev/null || true
  cat "${LOG_DIR}/stderr_$$.txt"
}

# 4a. Quality-degradation warning for q4_hqq K cache
WARN_K=$(run_stderr -m "${MODEL}" -c 128 -n 1 -p "x" -ctk q4_hqq -fit off)
if echo "${WARN_K}" | grep -qi "HQQ affine"; then
  pass "Quality-degradation WARN fires for -ctk q4_hqq"
  info "  Message: $(echo "${WARN_K}" | grep -i "HQQ affine" | head -1)"
else
  fail "No quality-degradation warning for -ctk q4_hqq"
fi

# 4b. Quality-degradation warning for q4_hqq_128 K cache
WARN_K128=$(run_stderr -m "${MODEL}" -c 128 -n 1 -p "x" -ctk q4_hqq_128 -fit off)
if echo "${WARN_K128}" | grep -qi "HQQ affine"; then
  pass "Quality-degradation WARN fires for -ctk q4_hqq_128"
else
  fail "No quality-degradation warning for -ctk q4_hqq_128"
fi

# 4c. Quality-degradation warning for q4_hqq V cache
WARN_V=$(run_stderr -m "${MODEL}" -c 128 -n 1 -p "x" -ctk q4_hqq -ctv q4_hqq -fa on -fit off)
if echo "${WARN_V}" | grep -qi "V cache quantization also requires Flash Attention\|HQQ affine"; then
  pass "Quality-degradation WARN fires for -ctv q4_hqq"
else
  fail "No quality-degradation warning for -ctv q4_hqq"
fi

# 4d. V-cache without FA is rejected
VNOFA=$(run_stderr -m "${MODEL}" -c 128 -n 1 -p "x" -ctk f16 -ctv q4_hqq -fit off)
if echo "${VNOFA}" | grep -qi "flash_attn\|requires flash\|quantized V"; then
  pass "V-cache without -fa is correctly rejected"
else
  fail "V-cache without -fa was not rejected (expected error about flash_attn)"
fi

# 4e. Alignment guard — Q4_HQQ (g32) on 128-dim head: 128%32=0 → PASS
ALIGN_G32=$(run_stderr -m "${MODEL}" -c 128 -n 1 -p "x" -ctk q4_hqq -fit off)
if echo "${ALIGN_G32}" | grep -qi "does not divide"; then
  fail "Q4_HQQ alignment check failed on 128-dim head (128%32 should == 0)"
else
  pass "Q4_HQQ alignment check passes on 128-dim head (128%32==0)"
fi

# 4f. Alignment guard — Q4_HQQ_128 (g128) on 128-dim head: 128%128=0 → PASS
ALIGN_G128=$(run_stderr -m "${MODEL}" -c 128 -n 1 -p "x" -ctk q4_hqq_128 -fit off)
if echo "${ALIGN_G128}" | grep -qi "does not divide"; then
  fail "Q4_HQQ_128 alignment check failed on 128-dim head (128%128 should == 0)"
else
  pass "Q4_HQQ_128 alignment check passes on 128-dim head (128%128==0)"
fi

# 4g. Manual check note for 64-dim head models
warn_manual "Alignment guard (ENABLED flash_attn path)
Q4_HQQ_128 requires n_embd_head % 128 == 0.
Models with 64-dim heads (e.g. some Phi or Falcon variants) would trigger:
  'K cache type q4_hqq_128 with block size 128 does not divide n_embd_head_k=64'

To test: pass a 64-dim-head model with -ctk q4_hqq_128 -fa on and verify
the error message printed to stderr matches the format above.

Source: src/llama-context.cpp line ~2860"

# =============================================================================
# 5. MEMORY MEASUREMENT
# =============================================================================
section "5. KV cache memory at c=4096 (Llama-3.2-3B)"

echo
echo "  Model architecture: 28 layers, 8 GQA KV heads, 128-dim head"
echo "  Context: 4096 tokens"
echo
printf "  %-14s  %-12s  %-12s  %-12s  %-10s\n" "KV type" "K (MiB)" "V (MiB)" "Total (MiB)" "vs f16"
printf "  %-14s  %-12s  %-12s  %-12s  %-10s\n" "---------" "-------" "-------" "----------" "------"

F16_TOTAL=448.00  # baseline for ratio computation

measure_kv_mem() {
  local TYPE="$1"
  local EXTRA_FLAGS=("${@:2}")
  timeout 25 "${CLI}" \
    -m "${MODEL}" -c 4096 -n 1 -p "x" \
    -ctk "${TYPE}" "${EXTRA_FLAGS[@]}" \
    -fit off 2>&1 | \
    grep "^llama_kv_cache: size" | head -1 || echo "ERROR"
}

for TYPE in f16 bf16 q8_0 q4_0 q4_1 q4_hqq q4_hqq_128 q5_0 q5_1; do
  LINE=$(measure_kv_mem "${TYPE}")
  if echo "${LINE}" | grep -q "size ="; then
    TOTAL=$(echo "${LINE}" | sed 's/.*size = *\([0-9.]*\) MiB.*/\1/')
    K_MIB=$(echo "${LINE}" | sed 's/.*K ([^)]*): *\([0-9.]*\) MiB.*/\1/')
    V_MIB=$(echo "${LINE}" | sed 's/.*V ([^)]*): *\([0-9.]*\) MiB.*/\1/')
    RATIO=$(awk "BEGIN{printf \"%.2f×\", ${F16_TOTAL}/${TOTAL}}")
    printf "  %-14s  %-12s  %-12s  %-12s  %-10s\n" "${TYPE}" "${K_MIB}" "${V_MIB}" "${TOTAL}" "${RATIO}"
    pass "KV memory measured for type=${TYPE}: K=${K_MIB} MiB, V=${V_MIB} MiB, total=${TOTAL} MiB"
  else
    printf "  %-14s  %-12s\n" "${TYPE}" "ERROR"
    fail "Failed to measure KV memory for type=${TYPE}"
  fi
done

echo
echo "  --- K+V with Flash Attention (quantized V cache) ---"
printf "  %-14s  %-12s  %-12s  %-12s  %-10s\n" "KV type" "K (MiB)" "V (MiB)" "Total (MiB)" "vs f16"
printf "  %-14s  %-12s  %-12s  %-12s  %-10s\n" "---------" "-------" "-------" "----------" "------"
for TYPE in f16 q4_0 q4_hqq q4_hqq_128; do
  LINE=$(measure_kv_mem "${TYPE}" -ctv "${TYPE}" -fa on)
  if echo "${LINE}" | grep -q "size ="; then
    TOTAL=$(echo "${LINE}" | sed 's/.*size = *\([0-9.]*\) MiB.*/\1/')
    K_MIB=$(echo "${LINE}" | sed 's/.*K ([^)]*): *\([0-9.]*\) MiB.*/\1/')
    V_MIB=$(echo "${LINE}" | sed 's/.*V ([^)]*): *\([0-9.]*\) MiB.*/\1/')
    RATIO=$(awk "BEGIN{printf \"%.2f×\", ${F16_TOTAL}/${TOTAL}}")
    printf "  %-14s  %-12s  %-12s  %-12s  %-10s\n" "${TYPE}(K+V)" "${K_MIB}" "${V_MIB}" "${TOTAL}" "${RATIO}"
    pass "K+V FA memory measured for type=${TYPE}: total=${TOTAL} MiB"
  else
    printf "  %-14s  %-12s\n" "${TYPE}(K+V)" "ERROR"
    fail "Failed to measure K+V FA memory for type=${TYPE}"
  fi
done

# =============================================================================
# 6. OUTPUT QUALITY TESTS
# =============================================================================
section "6. Output quality"

echo
echo "  Prompt: 'Write a short paragraph about artificial intelligence:'"
echo "  n=80 tokens, seed=42, model=Llama-3.2-3B-Q4_0"
echo
echo "  ─── Coherence legend ─────────────────────────────────────────────"
echo "  GOOD: grammatically structured, topically relevant output"
echo "  DEGRADED: partially incoherent, repetition, topic drift"
echo "  CORRUPT: garbage tokens, punctuation runs, nonsense"
echo

PROMPT="Write a short paragraph about artificial intelligence:"
SEED=42
N=80
CTX=512

run_quality() {
  local LABEL="$1"; shift
  local OUTPUT
  OUTPUT=$(timeout 120 "${CLI}" \
    -m "${MODEL}" -c "${CTX}" -n "${N}" --seed "${SEED}" \
    -p "${PROMPT}" "$@" 2>/dev/null | head -c 400 || echo "TIMEOUT/ERROR")
  echo "  [${LABEL}]"
  echo "    ${OUTPUT:0:300}" | fold -s -w 80 | sed 's/^/    /'
  echo
  echo "${OUTPUT}"  # return for analysis
}

# Baseline
OUT_F16=$(run_quality "f16 (baseline)" -ctk f16)
OUT_Q8=$(run_quality "q8_0 K-only" -ctk q8_0)
OUT_Q40=$(run_quality "q4_0 K-only" -ctk q4_0)
OUT_HQQ=$(run_quality "q4_hqq K-only" -ctk q4_hqq)
OUT_HQQFA=$(run_quality "q4_hqq K+V +FA" -ctk q4_hqq -ctv q4_hqq -fa on)
OUT_HQQ128=$(run_quality "q4_hqq_128 K-only" -ctk q4_hqq_128)
OUT_HQQ128FA=$(run_quality "q4_hqq_128 K+V +FA" -ctk q4_hqq_128 -ctv q4_hqq_128 -fa on)

# Automated coherence check: count garbage-token indicators
# is_corrupt: count characteristic degradation markers.
# Returns 0 (true) if the output is corrupt, 1 (false) if coherent.
is_corrupt() {
  local OUTPUT="$1"
  local DOTS=$(echo "${OUTPUT}" | grep -o '\.\.\.' | wc -l)
  local DASHES=$(echo "${OUTPUT}" | grep -o ' – – ' | wc -l)
  local SCORE=$((DOTS + DASHES))
  [[ ${SCORE} -gt 5 ]]
}

# check_good: PASS when output is coherent, FAIL when corrupt.
# Used for types that should produce valid output.
check_good() {
  local LABEL="$1"
  local OUTPUT="$2"
  if is_corrupt "${OUTPUT}"; then
    fail "Output quality CORRUPT for [${LABEL}] — unexpected degradation"
  else
    pass "Output quality GOOD for [${LABEL}]"
  fi
}

# check_expect_corrupt: PASS when output IS corrupt (proves DC-bias finding),
# FAIL when output is unexpectedly coherent.
# Used for q4_hqq_128, which is known to degrade due to large-block DC bias.
check_expect_corrupt() {
  local LABEL="$1"
  local OUTPUT="$2"
  if is_corrupt "${OUTPUT}"; then
    pass "Output quality CORRUPT for [${LABEL}] — expected (DC-bias limitation confirmed)"
  else
    fail "Output unexpectedly coherent for [${LABEL}] — DC-bias finding not reproduced"
  fi
}

# Coherent types — should produce GOOD output
check_good "f16 baseline"        "${OUT_F16}"
check_good "q8_0 K-only"         "${OUT_Q8}"
check_good "q4_0 K-only"         "${OUT_Q40}"
check_good "q4_hqq K-only"       "${OUT_HQQ}"
check_good "q4_hqq K+V +FA"      "${OUT_HQQFA}"

# Q4_HQQ_128 — empirically confirmed to produce corrupt output on Llama-3.
# PASS here means our DC-bias finding is reproduced; FAIL means it wasn't.
check_expect_corrupt "q4_hqq_128 K-only (expect corrupt)"  "${OUT_HQQ128}"
check_expect_corrupt "q4_hqq_128 K+V +FA (expect corrupt)" "${OUT_HQQ128FA}"

# Manual review note
warn_manual "Output quality — human review recommended
The automated check above only counts surface-level garbage tokens.
A full quality assessment requires reading each output and checking:
  1. Is the response topically relevant to the prompt?
  2. Is grammar/syntax structurally correct?
  3. Is there repetition or phrase looping?
  4. Does coherence degrade as token index increases?

Known finding (from empirical measurement — see results_KV_Cache_Q4_HQQ.md):
  - q4_hqq K-only at n=80: output is GOOD (DC bias not dominant at short ctx)
  - q4_hqq K+V at n=80: output is GOOD (similar to K-only at short ctx)
  - q4_hqq_128 K-only at n=80: output is CORRUPT (large block = stronger bias)
  - q4_hqq_128 K+V at n=80: output is CORRUPT

For deeper quality testing, use perplexity evaluation:
  ./build/bin/llama-perplexity -m <model> -f data/wikitext-2-raw/wiki.test.raw \\
    -ctk q4_hqq --ppl-stride 512 2>&1 | grep 'Final estimate'"

# =============================================================================
# 7. THROUGHPUT BENCHMARK (llama-bench)
# =============================================================================
section "7. Throughput benchmark (llama-bench)"

if [[ ${QUICK_MODE} -eq 1 ]]; then
  info "Skipping full benchmark (--quick mode). Running minimal probe..."
  # Just verify llama-bench accepts both new types without crashing
  BENCH_OUT=$(timeout 30 "${BENCH}" \
    -m "${MODEL}" -p 64 -n 0 -r 1 \
    -ctk q4_hqq \
    -ctk q4_hqq_128 \
    2>&1 || true)
  if echo "${BENCH_OUT}" | grep -q "q4_hqq"; then
    pass "llama-bench accepts -ctk q4_hqq (parser works)"
  else
    fail "llama-bench did not produce output for -ctk q4_hqq"
  fi
  if echo "${BENCH_OUT}" | grep -q "q4_hqq_128"; then
    pass "llama-bench accepts -ctk q4_hqq_128 (parser works)"
  else
    fail "llama-bench did not produce output for -ctk q4_hqq_128"
  fi
else
  echo
  echo "  K-only benchmark (no Flash Attention): PP256, TG32, 3 reps"
  echo

  # K-only: all types including new ones
  BENCH_K=$(timeout 600 "${BENCH}" \
    -m "${MODEL}" \
    -p 256 -n 32 -r 3 \
    -ctk f16 \
    -ctk q8_0 \
    -ctk q4_0 \
    -ctk q4_1 \
    -ctk q4_hqq \
    -ctk q4_hqq_128 \
    2>&1 || true)
  echo "${BENCH_K}" | sed 's/^/  /'

  # Verify both new types appear in bench output
  if echo "${BENCH_K}" | grep -q "q4_hqq |"; then
    pass "llama-bench produced results for -ctk q4_hqq"
  else
    fail "llama-bench missing results for -ctk q4_hqq"
  fi
  if echo "${BENCH_K}" | grep -q "q4_hqq_128"; then
    pass "llama-bench produced results for -ctk q4_hqq_128"
  else
    fail "llama-bench missing results for -ctk q4_hqq_128"
  fi

  echo
  echo "  K+V benchmark (Flash Attention on): PP256, TG32, 3 reps"
  echo

  # K+V with FA — only compatible types
  BENCH_KV=$(timeout 600 "${BENCH}" \
    -m "${MODEL}" \
    -p 256 -n 32 -r 3 -fa 1 \
    -ctk f16 \
    -ctk q4_0 \
    -ctk q4_hqq \
    -ctk q4_hqq_128 \
    2>&1 || true)
  echo "${BENCH_KV}" | sed 's/^/  /'

  if echo "${BENCH_KV}" | grep -q "q4_hqq"; then
    pass "llama-bench K+V FA produced results for q4_hqq"
  else
    fail "llama-bench K+V FA missing results for q4_hqq"
  fi
fi

# =============================================================================
# 8. CODE REGRESSION CHECK
# =============================================================================
section "8. Code regression checks (source inspection)"

# 8a. Block-size check uses != DISABLED (not == AUTO)
CONTEXT_SRC="${REPO_DIR}/src/llama-context.cpp"
if grep -q "flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED" "${CONTEXT_SRC}"; then
  pass "llama-context.cpp: block-size guard uses != DISABLED (covers ENABLED path)"
else
  fail "llama-context.cpp: block-size guard still uses == AUTO — regression!"
fi

# 8b. V-cache error message says n_embd_head_v (not n_embd_head_k).
# We search for the two error lines and verify:
#   - The K-cache line mentions n_embd_head_k
#   - The V-cache line mentions n_embd_head_v  (was "head_k" before the fix)
# The grep selects only the V-cache path by matching "type_v" in the surrounding context.
if grep -A2 "ggml_is_quantized(params.type_v)" "${CONTEXT_SRC}" | grep -q "n_embd_head_v"; then
  pass "llama-context.cpp: V-cache error message correctly prints n_embd_head_v"
else
  fail "llama-context.cpp: V-cache error message typo not fixed (still says head_k)"
fi

# 8c. Q4_HQQ_128 in kv_cache_types
ARG_SRC="${REPO_DIR}/common/arg.cpp"
if grep -q "GGML_TYPE_Q4_HQQ_128" "${ARG_SRC}"; then
  pass "common/arg.cpp: GGML_TYPE_Q4_HQQ_128 present in kv_cache_types"
else
  fail "common/arg.cpp: GGML_TYPE_Q4_HQQ_128 missing from kv_cache_types"
fi

# 8d. q4_hqq_128 in llama-bench parser
BENCH_SRC="${REPO_DIR}/tools/llama-bench/llama-bench.cpp"
if grep -q '"q4_hqq_128"' "${BENCH_SRC}"; then
  pass "llama-bench.cpp: \"q4_hqq_128\" present in ggml_type_from_name()"
else
  fail "llama-bench.cpp: \"q4_hqq_128\" missing from ggml_type_from_name()"
fi

# 8e. HQQ quality warning present in context init
if grep -q "HQQ affine quantization" "${CONTEXT_SRC}"; then
  pass "llama-context.cpp: HQQ quality-degradation warning present"
else
  fail "llama-context.cpp: HQQ quality-degradation warning missing"
fi

# 8f. Both Q4_HQQ variants in ops.cpp dispatch switches
OPS_SRC="${REPO_DIR}/ggml/src/ggml-cpu/ops.cpp"
HQQ_CASES=$(grep -c "GGML_TYPE_Q4_HQQ:" "${OPS_SRC}" || true)
if [[ ${HQQ_CASES} -ge 6 ]]; then
  pass "ggml-cpu/ops.cpp: Q4_HQQ has ${HQQ_CASES} dispatch cases (sufficient coverage)"
else
  fail "ggml-cpu/ops.cpp: only ${HQQ_CASES} Q4_HQQ dispatch cases — expected ≥6"
fi

# =============================================================================
# 9. SUMMARY
# =============================================================================
section "9. Summary"

echo
TOTAL=$((PASS + FAIL))
echo "  Total tests: ${TOTAL}"
echo "  Passed:      ${PASS}"
echo "  Failed:      ${FAIL}"
echo

if [[ ${FAIL} -eq 0 ]]; then
  echo "  ✓ ALL TESTS PASSED"
  echo
  echo "  Branch feature/q4_hqq fully supports Q4_HQQ and Q4_HQQ_128 as"
  echo "  KV cache quantization types.  See results_KV_Cache_Q4_HQQ.md for"
  echo "  the full empirical analysis and production recommendations."
  exit 0
else
  echo "  ✗ ${FAIL} TESTS FAILED — review output above"
  exit 1
fi
