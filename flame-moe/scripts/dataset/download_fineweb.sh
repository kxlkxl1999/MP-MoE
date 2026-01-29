#!/bin/bash
# Download FineWeb-Edu dataset from Hugging Face.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/../config_local.sh"

# Configuration
DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
SUBSET="${SUBSET:-sample-10BT}"  # Options: sample-10BT, sample-100BT, sample-350BT
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/fineweb-edu}"
CHUNK_SIZE="${CHUNK_SIZE:-1000000}"  # 1M examples per chunk (0 = single file)
RESUME="${RESUME:-0}"

mkdir -p "$OUTPUT_DIR"

RESUME_FLAG=""
if [ "$RESUME" = "1" ]; then
    RESUME_FLAG="--resume"
fi

python3 "$PROJECT_ROOT/scripts/dataset/download_fineweb.py" \
    --dataset-name "$DATASET_NAME" \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    --output-dir "$OUTPUT_DIR" \
    --chunk-size "$CHUNK_SIZE" \
    $RESUME_FLAG
