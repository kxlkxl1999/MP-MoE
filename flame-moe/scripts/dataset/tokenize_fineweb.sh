#!/bin/bash
# Tokenize FineWeb-Edu dataset for Megatron-LM.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/../config_local.sh"

# Configuration
SUBSET="${SUBSET:-sample-10BT}"
SPLIT="${SPLIT:-train}"
INPUT_DIR="${INPUT_DIR:-$DATA_ROOT/fineweb-edu/$SUBSET/$SPLIT}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/fineweb-edu-tokenized/$SUBSET}"
TOKENIZER="${TOKENIZER:-EleutherAI/pythia-12b}"
WORKERS="${WORKERS:-$(nproc)}"
MEGATRON_PATH="${MEGATRON_PATH:-$PROJECT_ROOT/Megatron-LM}"
DELETE_AFTER="${DELETE_AFTER:-0}"  # Set to 1 to delete JSONL after tokenizing
RESUME="${RESUME:-0}"

# Check input
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    echo "Run: bash scripts/dataset/download_fineweb.sh"
    exit 1
fi

DELETE_FLAG=""
if [ "$DELETE_AFTER" = "1" ]; then
    DELETE_FLAG="--delete-after-tokenize"
fi

RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume"
fi

python3 "$PROJECT_ROOT/scripts/dataset/tokenize_fineweb.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --tokenizer "$TOKENIZER" \
    --workers "$WORKERS" \
    --megatron-path "$MEGATRON_PATH" \
    --append-eod \
    $DELETE_FLAG \
    $RESUME_FLAG
