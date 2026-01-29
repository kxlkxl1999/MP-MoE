#!/bin/bash
# Local training launcher for FLAME-MoE-290M
# Usage: bash scripts/release/flame-moe-290m_local.sh

# Model configuration for FLAME-MoE-290M-1.3B
export WANDB_MODE=disabled
export HF_ENDPOINT=https://hf-mirror.com
export NUM_LAYERS=15
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=5472
export MOE_FFN_HIDDEN_SIZE=704
export MOE_LAYER_FREQ="[0]*1+[1]*14"
export MICRO_BATCH_SIZE=4
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8
export TRAIN_ITERS=5685
export SAVE_INTERVAL=568
export EVAL_INTERVAL=568

# Job name for this run
export JOB_NAME="flame-mope-419m-10-29"



# Launch local training
bash scripts/training/flame-mope_local.sh
