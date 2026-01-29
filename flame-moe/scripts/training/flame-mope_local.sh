#!/bin/bash
# Local training script for FLAME-MoE

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/../config_local.sh"

# Job configuration
export JOB_NAME="${JOB_NAME:-flame-moe-local}"
export JOB_ID="${JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

# Emulate SLURM environment expected by shared configs
export SLURM_JOB_NAME="${SLURM_JOB_NAME:-$JOB_NAME}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-$JOB_ID}"
export SLURM_NNODES="${SLURM_NNODES:-1}"
export SLURM_NODEID="${SLURM_NODEID:-0}"
if [ -z "${SLURM_GPUS_ON_NODE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        if SLURM_GPUS_ON_NODE=$(nvidia-smi --list-gpus | grep -c '^GPU '); then
            [ "$SLURM_GPUS_ON_NODE" -ge 1 ] || SLURM_GPUS_ON_NODE=1
        else
            SLURM_GPUS_ON_NODE=1
        fi
    else
        SLURM_GPUS_ON_NODE=1
    fi
    export SLURM_GPUS_ON_NODE
fi

# Compute world size and adjust expert parallelism for the available GPUs
WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
if (( WORLD_SIZE < 1 )); then
    WORLD_SIZE=1
fi
export WORLD_SIZE
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"

: "${EXPERT_MODEL_PARALLEL_SIZE:=1}"
if (( EXPERT_MODEL_PARALLEL_SIZE < 1 )); then
    EXPERT_MODEL_PARALLEL_SIZE=1
fi
adjusted_expert_mp=$EXPERT_MODEL_PARALLEL_SIZE
if (( adjusted_expert_mp > WORLD_SIZE )); then
    adjusted_expert_mp=$WORLD_SIZE
fi
while (( adjusted_expert_mp > 1 )) && (( WORLD_SIZE % adjusted_expert_mp != 0 )); do
    ((adjusted_expert_mp--))
done
if (( adjusted_expert_mp < 1 )); then
    adjusted_expert_mp=1
fi
if (( adjusted_expert_mp != EXPERT_MODEL_PARALLEL_SIZE )); then
    echo "Adjusting EXPERT_MODEL_PARALLEL_SIZE from $EXPERT_MODEL_PARALLEL_SIZE to $adjusted_expert_mp for world size $WORLD_SIZE."
fi
export EXPERT_MODEL_PARALLEL_SIZE=$adjusted_expert_mp

# Distributed training (single node: localhost:8000)
export RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
export RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:8000}"

# WandB (optional)
export WANDB_PROJECT="${WANDB_PROJECT:-}"
if [ -n "$WANDB_PROJECT" ]; then
    export WANDB_NAME="${WANDB_NAME:-$JOB_ID}"
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$JOB_NAME}"
else
    export WANDB_DISABLED="${WANDB_DISABLED:-true}"
fi

# Paths
export TRAIN_DATASET="${TRAIN_DATASET:-$LOCAL_DATASET}"
export TRAIN_WEIGHTS="${TRAIN_WEIGHTS:-$WEIGHTS_ROOT/$JOB_NAME/$JOB_ID}"
LOG_FILE="$LOGS_ROOT/$JOB_NAME/$JOB_ID.log"

mkdir -p "$TRAIN_WEIGHTS" "$(dirname "$LOG_FILE")"

# Check dataset
if [ ! -d "$TRAIN_DATASET" ] || [ -z "$(find "$TRAIN_DATASET" -name "*.bin" -print -quit)" ]; then
    echo "Error: Dataset not found: $TRAIN_DATASET"
    echo "Run: bash scripts/dataset/tokenize_fineweb.sh"
    exit 1
fi

# Prefer local tokenizer assets if present
if [[ "${TOKENIZER:-}" == "EleutherAI/pythia-12b" ]]; then
    for candidate in "$TRAIN_DATASET/tokenizer.json" \
                     "$TRAIN_DATASET/tokenizer.model" \
                     "$TRAIN_DATASET/tokenizer_config.json"; do
        if [ -f "$candidate" ]; then
            export TOKENIZER="$candidate"
            break
        fi
    done
fi
export TOKENIZER="${TOKENIZER:-EleutherAI/pythia-12b}"

# Environment
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
if [[ "${HF_HUB_OFFLINE:-}" == "1" || "${TRANSFORMERS_OFFLINE:-}" == "1" ]]; then
    if [ ! -e "$TOKENIZER" ]; then
        echo "Warning: Offline mode requested but tokenizer not found locally: $TOKENIZER" >&2
    fi
fi

# Load configs
source "$PROJECT_ROOT/configs/model/flame-mope.sh"
source "$PROJECT_ROOT/configs/train/flame-moe.sh"
# Override tokenizer argument for local/offline runs
MODEL_ARGS=("${MODEL_ARGS[@]}" --tokenizer-model "$TOKENIZER")

# Data arguments
DATA_ARGS=(
    --seq-length 2048
    --data-path $(find "$TRAIN_DATASET" -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 90,5,5
)

# Save arguments
SAVE_ARGS=(
    --log-interval 5
    --log-throughput
    --save "$TRAIN_WEIGHTS"
    --save-interval "${SAVE_INTERVAL:-1000}"
    --load "$TRAIN_WEIGHTS"
    --eval-interval "${EVAL_INTERVAL:-1000}"
    --tensorboard-dir "$TRAIN_WEIGHTS"
)

# WandB (if configured)
if [ -n "$WANDB_PROJECT" ]; then
    SAVE_ARGS+=(
        --wandb-project "$WANDB_PROJECT"
        --wandb-exp-name "$JOB_ID"
        --wandb-save-dir "$TRAIN_WEIGHTS"
    )
fi

# --- Distributed environment sanity setup ---
# export CUDA_VISIBLE_DEVICES=0,1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# Start training
cd "$PROJECT_ROOT/Megatron-LM" || exit 1
torchrun  "${TORCH_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${INFRA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${SAVE_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

exit ${PIPESTATUS[0]}
