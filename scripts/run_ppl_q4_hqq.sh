#!/usr/bin/env bash
# run_ppl_q4_hqq.sh — Perplexity benchmark: Q4_HQQ vs Q4_0
#
# Usage:
#   bash scripts/run_ppl_q4_hqq.sh [OPTIONS]
#
# Options:
#   --build-dir DIR      Path to llama.cpp build dir (default: ./build)
#   --models-dir DIR     Directory containing GGUF model files (default: ./models)
#   --f16-model FILE     F16 GGUF model path (required for Q4_HQQ quantization)
#   --q4_0-model FILE    Pre-quantized Q4_0 GGUF model path
#   --wiki-file FILE     Path to wiki.test.raw (default: wikitext-2-raw/wiki.test.raw)
#   --chunks N           Number of 512-token chunks to evaluate (default: 40)
#   --threads N          CPU threads (default: nproc)
#   --ctx N              Context window size (default: 512)
#   --skip-quantize      Skip Q4_HQQ quantization step (use existing model)
#   --skip-download      Skip wikitext-2 download
#   --output-dir DIR     Directory to write results JSON/Markdown (default: .)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BUILD_DIR="${BUILD_DIR:-./build}"
MODELS_DIR="${MODELS_DIR:-/home/ubuntu/tether_qvac_assessment/models/gguf}"
F16_MODEL="${F16_MODEL:-$MODELS_DIR/unsloth_Llama-3.2-3B-f16.gguf}"
Q4_0_MODEL="${Q4_0_MODEL:-$MODELS_DIR/unsloth_Llama-3.2-3B-q4_0.gguf}"
Q4_HQQ_MODEL="${Q4_HQQ_MODEL:-$MODELS_DIR/unsloth_Llama-3.2-3B-q4_hqq.gguf}"
WIKI_FILE="${WIKI_FILE:-wikitext-2-raw/wiki.test.raw}"
CHUNKS="${CHUNKS:-40}"
THREADS="${THREADS:-$(nproc)}"
CTX="${CTX:-512}"
SKIP_QUANTIZE=0
SKIP_DOWNLOAD=0
OUTPUT_DIR="${OUTPUT_DIR:-.}"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)    BUILD_DIR="$2";    shift 2 ;;
        --models-dir)   MODELS_DIR="$2";   shift 2 ;;
        --f16-model)    F16_MODEL="$2";    shift 2 ;;
        --q4_0-model)   Q4_0_MODEL="$2";   shift 2 ;;
        --wiki-file)    WIKI_FILE="$2";    shift 2 ;;
        --chunks)       CHUNKS="$2";       shift 2 ;;
        --threads)      THREADS="$2";      shift 2 ;;
        --ctx)          CTX="$2";          shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";   shift 2 ;;
        --skip-quantize) SKIP_QUANTIZE=1; shift ;;
        --skip-download) SKIP_DOWNLOAD=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BIN="$BUILD_DIR/bin"
QUANTIZE="$BIN/llama-quantize"
PPL="$BIN/llama-perplexity"

# ── Helpers ────────────────────────────────────────────────────────────────────
log() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }
die() { printf "ERROR: %s\n" "$*" >&2; exit 1; }

check_binary() {
    [[ -x "$1" ]] || die "Binary not found: $1 — run 'cmake --build build' first"
}

# ── Validate binaries ──────────────────────────────────────────────────────────
check_binary "$QUANTIZE"
check_binary "$PPL"

# ── Download WikiText-2 ────────────────────────────────────────────────────────
if [[ "$SKIP_DOWNLOAD" -eq 0 && ! -f "$WIKI_FILE" ]]; then
    log "Downloading WikiText-2 raw test set..."
    bash "$(dirname "$0")/get-wikitext-2.sh"
fi

[[ -f "$WIKI_FILE" ]] || die "Wiki test file not found: $WIKI_FILE — run with --wiki-file or remove --skip-download"

# ── Quantize F16 → Q4_HQQ ─────────────────────────────────────────────────────
if [[ "$SKIP_QUANTIZE" -eq 0 ]]; then
    if [[ -f "$Q4_HQQ_MODEL" ]]; then
        log "Q4_HQQ model already exists: $Q4_HQQ_MODEL (skipping quantization)"
    else
        [[ -f "$F16_MODEL" ]] || die "F16 model not found: $F16_MODEL"
        log "Quantizing F16 → Q4_HQQ: $F16_MODEL → $Q4_HQQ_MODEL"
        "$QUANTIZE" "$F16_MODEL" "$Q4_HQQ_MODEL" Q4_HQQ
        log "Quantization complete."
    fi
else
    log "Skipping quantization (--skip-quantize)"
    [[ -f "$Q4_HQQ_MODEL" ]] || die "Q4_HQQ model not found: $Q4_HQQ_MODEL"
fi

# ── Run perplexity ─────────────────────────────────────────────────────────────
run_ppl() {
    local label="$1" model="$2"
    [[ -f "$model" ]] || die "Model not found: $model"
    log "Running perplexity [$label]: model=$model chunks=$CHUNKS ctx=$CTX threads=$THREADS"
    "$PPL" \
        -m "$model" \
        -f "$WIKI_FILE" \
        -c "$CTX" \
        --chunks "$CHUNKS" \
        -t "$THREADS" \
        2>&1
}

RESULT_Q4_0=""
RESULT_Q4_HQQ=""

if [[ -f "$Q4_0_MODEL" ]]; then
    log "=== Q4_0 Perplexity ==="
    RESULT_Q4_0=$(run_ppl "Q4_0" "$Q4_0_MODEL")
    echo "$RESULT_Q4_0"
    PPL_Q4_0=$(echo "$RESULT_Q4_0" | grep -oP "PPL = \K[0-9.]+" | tail -1)
else
    log "Q4_0 model not found ($Q4_0_MODEL) — skipping Q4_0 benchmark"
    PPL_Q4_0="N/A"
fi

log "=== Q4_HQQ Perplexity ==="
RESULT_Q4_HQQ=$(run_ppl "Q4_HQQ" "$Q4_HQQ_MODEL")
echo "$RESULT_Q4_HQQ"
PPL_Q4_HQQ=$(echo "$RESULT_Q4_HQQ" | grep -oP "PPL = \K[0-9.]+" | tail -1)

# ── Print summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " PERPLEXITY COMPARISON SUMMARY"
echo " Model:    Llama-3.2-3B"
echo " Dataset:  WikiText-2 raw test"
echo " Chunks:   $CHUNKS × $CTX tokens = $((CHUNKS * CTX)) total tokens"
echo " Threads:  $THREADS"
echo "------------------------------------------------------------"
printf " %-12s  PPL = %s\n" "Q4_0"   "${PPL_Q4_0:-N/A}"
printf " %-12s  PPL = %s\n" "Q4_HQQ" "${PPL_Q4_HQQ:-N/A}"
echo "============================================================"

# ── Write JSON results ─────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/ppl_results_q4_hqq_vs_q4_0.json"
cat > "$RESULTS_FILE" <<EOF
{
  "benchmark": "perplexity",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "model": "Llama-3.2-3B",
  "dataset": "wikitext-2-raw",
  "context": $CTX,
  "chunks": $CHUNKS,
  "tokens_evaluated": $((CHUNKS * CTX)),
  "threads": $THREADS,
  "results": {
    "Q4_0":   ${PPL_Q4_0:-null},
    "Q4_HQQ": ${PPL_Q4_HQQ:-null}
  }
}
EOF
log "Results written to $RESULTS_FILE"
