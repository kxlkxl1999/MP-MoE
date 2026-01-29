#!/bin/bash
# Local training launcher for FLAME-MoE-290M
# Usage: bash scripts/release/flame-moe-290m_local.sh

# Model configuration for FLAME-MoE-290M-1.3B
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export NUM_LAYERS=9
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=5472
export MOE_FFN_HIDDEN_SIZE=704
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=4
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8
export TRAIN_ITERS=5473
export SAVE_INTERVAL=540
export EVAL_INTERVAL=540

# Activation Checkpointing
export ENABLE_RECOMPUTE=1
export RECOMPUTE_GRANULARITY="selective"


# Job name for this run
export JOB_NAME="flame-moe-290m"
export JOB_ID="$(date +%Y%m%d_%H%M%S)"


# Launch local training
bash scripts/training/flame-moe_local.sh
