#!/bin/bash
# =============================================================================
# FineWeb-Edu Download & Tokenize Pipeline
# =============================================================================
#
# Usage:
#   SUBSET=sample-100BT bash scripts/dataset/pipeline_fineweb.sh
#
# Available datasets:
#   - sample-10BT   (~25GB raw, ~20GB tokenized)
#   - sample-100BT  (~250GB raw, ~200GB tokenized)
#   - sample-350BT  (~900GB raw, ~700GB tokenized)
#
# Modes:
#   PARALLEL=1 (default) - Parallel mode: download and tokenize simultaneously, saves time and space
#   PARALLEL=0           - Serial mode: download all data first, then tokenize
#
# Resume after interruption:
#   Simply re-run the same command, the script will automatically:
#   1. Skip already downloaded chunks (checks .jsonl files)
#   2. Skip already tokenized chunks (checks .bin files)
#   3. Continue from checkpoint
#
# Environment variables:
#   SUBSET      - Dataset subset (default: sample-10BT)
#   PARALLEL    - 1=parallel, 0=serial (default: 1)
#   CHUNK_SIZE  - Samples per chunk (default: 1000000)
#   WORKERS     - Tokenization parallelism (default: CPU core count)
#
# Examples:
#   # Download and tokenize 100BT dataset
#   SUBSET=sample-100BT bash scripts/dataset/pipeline_fineweb.sh
#
#   # Use serial mode (more stable, for debugging)
#   SUBSET=sample-10BT PARALLEL=0 bash scripts/dataset/pipeline_fineweb.sh
#
#   # Resume after interruption (just re-run)
#   SUBSET=sample-100BT bash scripts/dataset/pipeline_fineweb.sh
# =============================================================================

set -euo pipefail

# Cleanup function to kill background processes on exit
cleanup() {
    if [ -n "${DOWNLOAD_PID:-}" ]; then
        echo ""
        echo "Stopping download process..."
        kill $DOWNLOAD_PID 2>/dev/null || true
        wait $DOWNLOAD_PID 2>/dev/null || true
    fi
    exit 1
}

# Trap Ctrl+C and other termination signals
trap cleanup SIGINT SIGTERM

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/../config_local.sh"

# Configuration
SUBSET="${SUBSET:-sample-10BT}"
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
SPLIT="${SPLIT:-train}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"
WORKERS="${WORKERS:-$(nproc)}"
PARALLEL="${PARALLEL:-1}"  # Set to 1 for parallel download+tokenize

DOWNLOAD_DIR="$DATA_ROOT/fineweb-edu/$SUBSET/$SPLIT"
OUTPUT_DIR="$DATA_ROOT/fineweb-edu-tokenized/$SUBSET"

echo "=== FineWeb-Edu Pipeline ==="
echo "Subset: $SUBSET"
echo "Chunk size: $CHUNK_SIZE"
echo "Parallel mode: $PARALLEL"
echo "HF_ENDPOINT: ${HF_ENDPOINT:-not set}"
echo ""

mkdir -p "$DOWNLOAD_DIR" "$OUTPUT_DIR"

if [ "$PARALLEL" = "1" ]; then
    # Parallel mode: download and tokenize simultaneously
    echo "=== Parallel Mode: Download + Tokenize ==="

    python3 "$PROJECT_ROOT/scripts/dataset/download_fineweb.py" \
        --dataset-name "$DATASET_NAME" \
        --subset "$SUBSET" \
        --split "$SPLIT" \
        --output-dir "$DATA_ROOT/fineweb-edu" \
        --chunk-size "$CHUNK_SIZE" \
        --tokenized-dir "$OUTPUT_DIR" \
        --resume &
    DOWNLOAD_PID=$!

    # Wait for any chunk to appear
    echo "Waiting for first chunk..."
    while ! ls "$DOWNLOAD_DIR"/chunk_*.jsonl >/dev/null 2>&1; do
        sleep 5
        # Check if download failed or finished
        if ! kill -0 $DOWNLOAD_PID 2>/dev/null; then
            # Download finished - check if there are chunks to process
            if ls "$DOWNLOAD_DIR"/chunk_*.jsonl >/dev/null 2>&1; then
                break
            fi
            echo "Download process exited with no new chunks"
            wait $DOWNLOAD_PID
            exit $?
        fi
    done

    echo "First chunk ready, starting tokenizer..."

    # Tokenize chunks as they appear
    processed_chunks=""
    while true; do
        # Find new chunks
        for chunk_file in "$DOWNLOAD_DIR"/chunk_*.jsonl; do
            [ -f "$chunk_file" ] || continue

            chunk_name=$(basename "$chunk_file" .jsonl)

            # Skip already processed (in this session)
            if echo "$processed_chunks" | grep -q "|$chunk_name|"; then
                continue
            fi

            # Skip already tokenized (from previous run)
            if [ -f "$OUTPUT_DIR/${chunk_name}_text_document.bin" ] && \
               [ -f "$OUTPUT_DIR/${chunk_name}_text_document.idx" ]; then
                echo "Skipping (already tokenized): $chunk_name"
                processed_chunks="${processed_chunks}|${chunk_name}|"
                rm -f "$chunk_file"
                continue
            fi

            output_prefix="$OUTPUT_DIR/$chunk_name"

            if python3 "$PROJECT_ROOT/Megatron-LM/tools/preprocess_data.py" \
                --input "$chunk_file" \
                --output-prefix "$output_prefix" \
                --tokenizer-type HuggingFaceTokenizer \
                --tokenizer-model "${TOKENIZER:-EleutherAI/pythia-12b}" \
                --workers "$WORKERS" \
                --log-interval 1000000 \
                --append-eod >/dev/null 2>&1; then
                # Only mark as processed and delete if tokenization succeeded
                processed_chunks="${processed_chunks}|${chunk_name}|"
                rm -f "$chunk_file"
                echo "[Tokenized] $chunk_name"
            else
                echo "  ERROR: Failed to tokenize $chunk_name, will retry"
            fi
        done

        # Check if download finished
        if ! kill -0 $DOWNLOAD_PID 2>/dev/null; then
            # Process any remaining chunks
            remaining=$(find "$DOWNLOAD_DIR" -name "chunk_*.jsonl" ! -name "*.tmp" 2>/dev/null | wc -l)
            [ "$remaining" -eq 0 ] && break
        fi

        sleep 3
    done

    # Check download exit code
    if wait $DOWNLOAD_PID; then
        echo "Download completed successfully"
    else
        echo "ERROR: Download process failed"
        exit 1
    fi

else
    # Sequential mode (original behavior)
    echo "=== Step 1: Downloading ==="
    CHUNK_SIZE=$CHUNK_SIZE SUBSET=$SUBSET RESUME=1 bash "$SCRIPT_DIR/download_fineweb.sh"

    echo ""
    echo "=== Step 2: Tokenizing ==="
    DELETE_AFTER=1 SUBSET=$SUBSET WORKERS=$WORKERS RESUME=1 bash "$SCRIPT_DIR/tokenize_fineweb.sh"
fi

# Summary
echo ""
echo "=== Complete ==="
echo "Tokenized data: $OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    total_size=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    file_count=$(find "$OUTPUT_DIR" -name "*.bin" | wc -l)
    echo "Total size: $total_size ($file_count files)"
fi
