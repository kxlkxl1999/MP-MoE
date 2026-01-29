#!/bin/bash
# Usage: ./scripts/eval_benchmark.sh <weights_path> <model_name> [iteration]
# Example: ./scripts/eval_benchmark.sh /path/to/weights/model1 model1 2424
#          ./scripts/eval_benchmark.sh /path/to/weights/model2 model2 1000

# set -e  # Disabled to continue even if a task fails

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <weights_path> <model_name> [iteration]"
  echo "Example: $0 /path/to/weights model1 2424"
  exit 1
fi

SSD_WEIGHTS="$1"
MODEL_NAME="$2"
ITER="${3:-2424}"

# ====== Environment Variables ======
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
unset HF_DATASETS_TRUST_REMOTE_CODE

export NUM_LAYERS=9
export HIDDEN_SIZE=512
export FFN_HIDDEN_SIZE=2736
export MOE_FFN_HIDDEN_SIZE=352
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=32
export BATCH_SIZE="${BATCH_SIZE:-$MICRO_BATCH_SIZE}"
# Per-task override (e.g., to avoid OOM on MMLU)
export MMLU_MICRO_BATCH_SIZE="${MMLU_MICRO_BATCH_SIZE:-16}"
export MMLU_BATCH_SIZE="${MMLU_BATCH_SIZE:-$MMLU_MICRO_BATCH_SIZE}"
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=8
export TRAIN_ITERS=$ITER
export SAVE_INTERVAL=242
export EVAL_INTERVAL=242

export USE_LOCAL_DATA=1
export LOCAL_DATA_DIR="$(pwd)/../data"
export PYTHONPATH=$(pwd)/Megatron-LM:$(pwd)/lm-evaluation-harness:$PYTHONPATH

JOBID="$MODEL_NAME"
RESULT_DIR="results/flame-mope/$JOBID"
LOG_DIR="logs/evaluate/$JOBID"

mkdir -p "$LOG_DIR" "$RESULT_DIR"
echo $ITER > "$SSD_WEIGHTS/latest_checkpointed_iteration.txt"

source configs/model/flame-mope.sh

# ====== Task Config ======
# Note: BBH and GSM8K require generate_until, now implemented in megatron_lm.py
# TEST MODE: Test with a single BBH subtask
TEST_MODE=0
if [[ $TEST_MODE -eq 1 ]]; then
  declare -A TASK_CONFIG=(
    ["bbh_fewshot"]="3:gen"  # BBH for testing
  )
  TEST_LIMIT="--limit 10"  # Only test 10 samples
  GEN_KWARGS="temperature=0.0,max_gen_toks=64,do_sample=false"  # Shorter, deterministic generation for direct-answer BBH
else
  declare -A TASK_CONFIG=(
    ["mmlu"]="5"
    ["boolq"]="0"
    ["hellaswag"]="0"
    ["bbh_fewshot"]="3:gen"        # BBH without CoT (direct answer)
    ["arc_easy"]="0"
    ["arc_challenge"]="0"
    # ["gsm8k"]="3:gen"
  )
  TEST_LIMIT=""
  GEN_KWARGS="temperature=0.0,max_gen_toks=64,do_sample=false"
fi

BASE_PORT=12345
BASE_COMMON_ARGS=(
  --bf16
  --seq-length 2048
  --max-tokens-to-oom 10000000
  --seed 42
  --load "$SSD_WEIGHTS"
  --model megatron_lm
  --output_path "$RESULT_DIR"
  --dist-ckpt-strictness log_unexpected
  $TEST_LIMIT
)

run_task() {
  local task="$1"
  local config="$2"
  local idx="$3"

  local nshot="${config%%:*}"
  local is_gen=0
  [[ "$config" == *":gen"* ]] && is_gen=1
  local micro_bs="$MICRO_BATCH_SIZE"
  local batch_bs="$BATCH_SIZE"
  if [[ "$task" == "mmlu" ]]; then
    micro_bs="$MMLU_MICRO_BATCH_SIZE"
    batch_bs="$MMLU_BATCH_SIZE"
  fi
  local TASK_COMMON_ARGS=(
    --micro-batch-size "$micro_bs"
    --batch_size "$batch_bs"
    "${BASE_COMMON_ARGS[@]}"
  )

  local logfile="$LOG_DIR/${task}-${nshot}shot.log"

  echo ">> Evaluating ${task} with ${nshot}-shot"

  if [[ $is_gen -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 --master_port=$((BASE_PORT + idx)) -m lm_eval \
        "${MODEL_ARGS[@]}" \
        "${TASK_COMMON_ARGS[@]}" \
        --num_fewshot "${nshot}" \
        --tasks "${task}" \
        --gen_kwargs "$GEN_KWARGS" \
        > "${logfile}" 2>&1
  else
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 --master_port=$((BASE_PORT + idx)) -m lm_eval \
        "${MODEL_ARGS[@]}" \
        "${TASK_COMMON_ARGS[@]}" \
        --num_fewshot "${nshot}" \
        --tasks "${task}" \
        > "${logfile}" 2>&1
  fi
}

# ====== Run All Tasks ======
idx=0
for task in "${!TASK_CONFIG[@]}"; do
  run_task "$task" "${TASK_CONFIG[$task]}" "$idx"
  idx=$((idx + 1))
done

echo "[DONE] $MODEL_NAME evaluation completed, results saved in $RESULT_DIR"

# ====== Extract Results ======
echo ""
echo "====== $MODEL_NAME Results ======"
echo ""

printf "| %-12s | %-10s | %-10s | %-12s | %-10s | %-12s | %-15s | %-15s | %-10s |\n" \
  "Model" "MMLU(5)" "Boolq(0)" "HellaSwag(0)" "BBH(3)" "ARC-Easy(A)" "ARC-Easy(AN)" "ARC-Chal(A)" "ARC-Chal(AN)" "Avg"
printf "|%s|\n" "--------------|------------|------------|--------------|------------|------------|---------------|---------------|----------|"

echo "Check $RESULT_DIR for JSON result files"
