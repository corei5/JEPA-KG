#!/usr/bin/env bash
# Run Hetionet JEPA fine-tuning with stp.py
# Usage: bash run_hetionet_stp.sh [model_name] [learning_rate] [lbd] [seed]
#
# Defaults: LLaMA-3.2-1B-Instruct, lr=2e-5, lbd=0.05, seed=42

set -euo pipefail

base_model_name="${1:-meta-llama/Llama-3.2-1B-Instruct}"
learning_rate="${2:-2e-5}"
lbd="${3:-0.05}"
seed="${4:-42}"

# Determine last_token offset (model-specific chat template quirk)
if [[ "$base_model_name" == google/gemma* ]]; then
  last_token=-2
elif [[ "$base_model_name" == apple/OpenELM* ]]; then
  last_token=-4
elif [[ "$base_model_name" == allenai/OLMo-2* ]]; then
  last_token=-1
elif [[ "$base_model_name" == Qwen/Qwen* ]]; then
  last_token=-3
elif [[ "$base_model_name" == deepseek-ai/DeepSeek* ]]; then
  last_token=-1
else
  last_token=-2   # LLaMA default
fi

predictors=128
num_epochs=3
max_length=2048    # KG examples are longer than NL→regex tasks

model_folder="trained_models/ft-hetionet-jepa-${learning_rate}-${lbd}-${seed}"

echo "=== Hetionet JEPA fine-tuning ==="
echo "  model       : ${base_model_name}"
echo "  lr          : ${learning_rate}"
echo "  lbd         : ${lbd}"
echo "  seed        : ${seed}"
echo "  last_token  : ${last_token}"
echo "  predictors  : ${predictors}"
echo "  max_length  : ${max_length}"
echo "  output_dir  : ${model_folder}"
echo ""

# ---- JEPA run ----
torchrun --nproc_per_node=4 stp.py \
  --train_file datasets/hetionet_train.jsonl \
  --eval_file  datasets/hetionet_test.jsonl \
  --output_dir="${model_folder}" \
  --model_name="${base_model_name}" \
  --learning_rate="${learning_rate}" \
  --num_epochs="${num_epochs}" \
  --finetune_seed="${seed}" \
  --last_token="${last_token}" \
  --lbd="${lbd}" \
  --predictors="${predictors}" \
  --max_length="${max_length}" \
  --linear=random_span

echo ""
echo "=== Training complete. Model saved to ${model_folder} ==="
