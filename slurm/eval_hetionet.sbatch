#!/bin/bash -x
#SBATCH --account=xxxxx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --partition=dc-hwai
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/eval_hetionet_%j.out
#SBATCH --error=slurm/logs/eval_hetionet_%j.err

set -euo pipefail

export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}

module load Stages
module load Python/3.13.5
source ".venv/bin/activate"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/p/project1/xxxxx/.cache/huggingface/datasets
export HF_HUB_CACHE=/p/project1/xxxxxx/.cache/huggingface/hub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
FINETUNED_MODEL="trained_models/ft-hetionet-jepa-2e-5-0.05-42"
TEST_FILE="datasets/hetionet_test.jsonl"
MAX_NEW_TOKENS=512

# ---- Evaluate fine-tuned JEPA model ----
echo ""
echo "============================================================"
echo " Evaluating JEPA fine-tuned model: ${FINETUNED_MODEL}"
echo "============================================================"
python evaluate_hetionet.py \
    --model_name "${FINETUNED_MODEL}" \
    --original_model_name "${BASE_MODEL}" \
    --input_file "${TEST_FILE}" \
    --output_file "eval_hetionet_jepa.jsonl" \
    --max_new_tokens ${MAX_NEW_TOKENS}

# ---- Evaluate base model ----
echo ""
echo "============================================================"
echo " Evaluating base model: ${BASE_MODEL}"
echo "============================================================"
python evaluate_hetionet.py \
    --model_name "${BASE_MODEL}" \
    --original_model_name "${BASE_MODEL}" \
    --input_file "${TEST_FILE}" \
    --output_file "eval_hetionet_base.jsonl" \
    --max_new_tokens ${MAX_NEW_TOKENS}

# ---- Print side-by-side comparison ----
echo ""
echo "============================================================"
echo " SCORE COMPARISON"
echo "============================================================"
python3 - <<'EOF'
import json

def load_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

jepa = load_summary("eval_hetionet_jepa_summary.json")
base = load_summary("eval_hetionet_base_summary.json")

def fmt(val):
    return f"{val:.4f}" if val is not None else "N/A"

print(f"\n{'Metric':<50} {'Base':>8}  {'JEPA':>8}  {'Delta':>8}")
print("-" * 72)

for label, key in [("Overall Precision", "precision"),
                    ("Overall Recall",    "recall"),
                    ("Overall F1",        "f1")]:
    b = base["overall"][key]  if base else None
    j = jepa["overall"][key]  if jepa else None
    delta = f"+{j-b:.4f}" if (j is not None and b is not None) else "N/A"
    print(f"  {label:<48} {fmt(b):>8}  {fmt(j):>8}  {delta:>8}")

print()

# Per-priority comparison
all_labels = set()
if jepa: all_labels.update(jepa.get("per_priority", {}).keys())
if base:  all_labels.update(base.get("per_priority", {}).keys())

if all_labels:
    print(f"  {'Priority Section (Recall / F1)':<48} {'Base R':>8}  {'JEPA R':>8}  {'Δ R':>8}")
    print("  " + "-" * 70)
    for lbl in sorted(all_labels):
        b_r = base["per_priority"].get(lbl, {}).get("recall") if base else None
        j_r = jepa["per_priority"].get(lbl, {}).get("recall") if jepa else None
        delta = f"+{j_r-b_r:.3f}" if (j_r is not None and b_r is not None) else "N/A"
        print(f"  {lbl:<48} {fmt(b_r):>8}  {fmt(j_r):>8}  {delta:>8}")

print()
if jepa: print(f"  JEPA model  : {jepa['model']}")
if base:  print(f"  Base model  : {base['model']}")
if jepa: print(f"  Test set    : {jepa['input_file']}  ({jepa['n_examples']} examples)")
EOF
