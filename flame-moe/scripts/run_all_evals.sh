#!/bin/bash
# Batch evaluation for multiple models
# Usage: ./scripts/run_all_evals.sh

# ====== Configure your models here ======
declare -A MODELS=(
  # ["model1"]="/root/autodl-tmp/FLAME-MoE/weights/flame-mope-98m/20260119_140549:2424"   # Format: weights_path:iteration
  ["model2"]="/root/autodl-tmp/FLAME-MoE/weights/flame-moe-98m-11-25/20260118_191952:2424"
)

# ====== Run Evaluation ======
for name in "${!MODELS[@]}"; do
  config="${MODELS[$name]}"
  weights="${config%%:*}"
  iter="${config##*:}"

  echo "========================================"
  echo "Evaluating: $name"
  echo "Weights: $weights"
  echo "Iteration: $iter"
  echo "========================================"

  bash scripts/eval_benchmark.sh "$weights" "$name" "$iter"
done

echo ""
echo "Run scripts/summarize_evals.sh to print the results summary."
