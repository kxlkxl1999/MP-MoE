#!/bin/bash
# Local training launcher for FLAME-MoE-38M
# Usage: bash scripts/release/flame-moe-38m_local.sh

export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com

# Model configuration for FLAME-MoE-38M
export NUM_LAYERS=9
export HIDDEN_SIZE=256
export FFN_HIDDEN_SIZE=1368
export MOE_FFN_HIDDEN_SIZE=176
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=16
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8
export TRAIN_ITERS=2121
export SAVE_INTERVAL=212
export EVAL_INTERVAL=212

# Memory optimization: Enable activation checkpointing
export ENABLE_RECOMPUTE=1
export RECOMPUTE_GRANULARITY="full"
export RECOMPUTE_METHOD="uniform"
export RECOMPUTE_NUM_LAYERS=1

# Job name for this run
export JOB_NAME="flame-moe-38m"

# Launch local training
bash scripts/training/flame-moe_local.sh
