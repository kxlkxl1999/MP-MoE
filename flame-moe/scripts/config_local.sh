#!/bin/bash
# Local configuration for FLAME-MoE training

# HuggingFace mirror (for faster downloads in China)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Resolve repository root, even when sourced from another directory
_CONFIG_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_DEFAULT_ROOT="$(cd -- "$_CONFIG_DIR/.." && pwd -P)"
if command -v git >/dev/null 2>&1; then
    _GIT_ROOT="$(git -C "$_DEFAULT_ROOT" rev-parse --show-toplevel 2>/dev/null)" || _GIT_ROOT=""
else
    _GIT_ROOT=""
fi

export PROJECT_ROOT="${PROJECT_ROOT:-${_GIT_ROOT:-$_DEFAULT_ROOT}}"
export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset}"
export WEIGHTS_ROOT="${WEIGHTS_ROOT:-$PROJECT_ROOT/weights}"
export LOGS_ROOT="${LOGS_ROOT:-$PROJECT_ROOT/logs}"

# Dataset and tokenizer
export LOCAL_DATASET="${LOCAL_DATASET:-$DATA_ROOT/fineweb-edu-tokenized/sample-100BT}"
export TOKENIZER="${TOKENIZER:-EleutherAI/pythia-12b}"

# Create directories
mkdir -p "$DATA_ROOT" "$WEIGHTS_ROOT" "$LOGS_ROOT"

unset _CONFIG_DIR _DEFAULT_ROOT _GIT_ROOT
