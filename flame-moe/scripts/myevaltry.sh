#!/bin/bash
# set -euo pipefail

# ====== Environment variables (keep your original settings) ======
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# export HF_DATASETS_TRUST_REMOTE_CODE=true
unset HF_DATASETS_TRUST_REMOTE_CODE

SSD_WEIGHTS=weights/flame-mope-98m-12-3-noshuffle/20251203_013450
JOBID=localtest
ITER=2424

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

export PYTHONPATH=$(pwd)/Megatron-LM:$(pwd)/lm-evaluation-harness:$PYTHONPATH

mkdir -p "logs/evaluate/$JOBID" "results/flame-mope/$JOBID"
echo $ITER > "$SSD_WEIGHTS/latest_checkpointed_iteration.txt"

# ====== Model parameters (keep your original) ======
source configs/model/flame-mope.sh
echo "MODEL_ARGS = ${MODEL_ARGS[@]}"

# ====== Task list (add/remove freely) ======
# Multiple choice/cloze tasks (no --gen needed by default)
TASKS=(
  # "piqa"          # common 0-shot
  "arc_easy"      # common 5 or 25-shot
  "arc_challenge" # common 5 or 25-shot
  "boolq"         # 0-shot works
  "rte"           # 0 or 8-shot
  "copa"          # common 0/8-shot
  "wsc"           # 0-shot
  "winogrande"    # 0/5-shot
  "hellaswag"     # common 10-shot
  "openbookqa"    # 0/10-shot
  "mmlu"          # aka hendrycksTest, common 5-shot
)

# Few-shot override (uses DEFAULT_FEWSHOT if not specified)
DEFAULT_FEWSHOT=0
declare -A FEWSHOT=(
  ["arc_easy"]=25
  ["arc_challenge"]=25
  ["hellaswag"]=10
  ["openbookqa"]=0
  ["mmlu"]=5
  ["winogrande"]=0
)

# ====== (Optional) Generation tasks, disabled by default ======
ENABLE_GEN=0   # Set to 1 to test GSM8K/TruthfulQA-Gen etc.
GEN_TASKS=(
  "truthfulqa_gen"  # generative
  "gsm8k"           # requires chain-of-thought, typically 8/0-shot; recommend small batch validation first
)
# Generation task few-shot override
declare -A GEN_FEWSHOT=(
  ["truthfulqa_gen"]=0
  ["gsm8k"]=0
)

# (Generation) common decoding parameters
GEN_ARGS=(
  --gen
  --temperature 0.0
  --max_gen_toks 512
  --do_sample false
)

# ====== Run parameters ======
# Whether to submit multiple tasks in parallel (each task uses all 8 GPUs; parallel may cause contention; serial is more stable)
PARALLEL=0
BASE_PORT=12345

# Common inference hyperparameters
COMMON_ARGS=(
  --bf16
  --seq-length 2048
  --micro-batch-size 32
  --batch_size 16
  --max-tokens-to-oom 10000000
  --seed 42
  --load "$SSD_WEIGHTS"
  --model megatron_lm
  --output_path "results/flame-mope/$JOBID"
)

run_one_task () {
  local task="$1"
  local nshot="$2"
  local idx="$3"         # for master_port
  local logfile="logs/evaluate/$JOBID/${task}-${nshot}shot-12-3.log"

  echo "▶ Evaluating ${task} with ${nshot}-shot"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 --master_port=$((BASE_PORT + idx)) -m lm_eval \
      "${MODEL_ARGS[@]}" \
      "${COMMON_ARGS[@]}" \
      --num_fewshot "${nshot}" \
      --tasks "${task}" \
      > "${logfile}" 2>&1
}

run_one_gen_task () {
  local task="$1"
  local nshot="$2"
  local idx="$3"
  local logfile="logs/evaluate/$JOBID/${task}-${nshot}shot-GEN.log"

  echo "▶ Evaluating (GEN) ${task} with ${nshot}-shot"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --nproc_per_node=4 --master_port=$((BASE_PORT + idx)) -m lm_eval \
      "${MODEL_ARGS[@]}" \
      "${COMMON_ARGS[@]}" \
      --num_fewshot "${nshot}" \
      --tasks "${task}" \
      "${GEN_ARGS[@]}" \
      > "${logfile}" 2>&1
}

# ====== Serial/Parallel execution ======
idx=0
pids=()

# Multiple choice/cloze tasks
for task in "${TASKS[@]}"; do
  nshot="${FEWSHOT[$task]:-$DEFAULT_FEWSHOT}"
  if [[ "$PARALLEL" -eq 1 ]]; then
    run_one_task "$task" "$nshot" "$idx" &
    pids+=($!)
  else
    run_one_task "$task" "$nshot" "$idx"
  fi
  idx=$((idx + 1))
done

# Generation tasks (optional)
if [[ "$ENABLE_GEN" -eq 1 ]]; then
  for task in "${GEN_TASKS[@]}"; do
    nshot="${GEN_FEWSHOT[$task]:-$DEFAULT_FEWSHOT}"
    if [[ "$PARALLEL" -eq 1 ]]; then
      run_one_gen_task "$task" "$nshot" "$idx" &
      pids+=($!)
    else
      run_one_gen_task "$task" "$nshot" "$idx"
    fi
    idx=$((idx + 1))
  done
fi

# Wait for parallel tasks
if [[ "${#pids[@]}" -gt 0 ]]; then
  echo "⏳ Waiting for ${#pids[@]} parallel jobs..."
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi

echo "✅ All evaluations completed."
