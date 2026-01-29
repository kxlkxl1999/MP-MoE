#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Model path
SSD_WEIGHTS=weights/flame-mope-98m-11-26/20251126_150856
JOBID=localtest
ITER=2424

# Basic hyperparameters
export NUM_LAYERS=9
export HIDDEN_SIZE=512
export FFN_HIDDEN_SIZE=2736
export MOE_FFN_HIDDEN_SIZE=352
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=12
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8
export TRAIN_ITERS=2424
export SAVE_INTERVAL=242
export EVAL_INTERVAL=242

export USE_LOCAL_DATA=1
export LOCAL_DATA_DIR="$(pwd)/../data"

# Environment import paths
export PYTHONPATH=$(pwd)/Megatron-LM:$(pwd)/lm-evaluation-harness:$PYTHONPATH

mkdir -p logs/evaluate/$JOBID
echo $ITER > $SSD_WEIGHTS/latest_checkpointed_iteration.txt

# Few-shot settings and task definitions
num_fewshots=(0 0 10)
fewshot_tasks=("openbookqa" "winogrande" "hellaswag")

# Model parameters definition
source configs/model/flame-mope.sh
echo "MODEL_ARGS = ${MODEL_ARGS[@]}"

# Serial task execution
for i in "${!num_fewshots[@]}"; do
    echo "Evaluating ${num_fewshots[$i]} shot for ${fewshot_tasks[$i]}"

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 --master_port=$((12345 + i)) -m lm_eval \
        "${MODEL_ARGS[@]}" \
        --bf16 \
        --seq-length 2048 \
        --micro-batch-size 32 \
        --num_fewshot ${num_fewshots[$i]} \
        --batch_size 16 \
        --max-tokens-to-oom 10000000 \
        --seed 42 \
        --load $SSD_WEIGHTS \
        --model megatron_lm \
        --tasks "${fewshot_tasks[$i]}" \
        --output_path results/flame-moe/$JOBID \
        > logs/evaluate/$JOBID/${fewshot_tasks[$i]}-${num_fewshots[$i]}shot_MOPE_11_27.log 2>&1
done

echo "✅ All evaluations completed."
