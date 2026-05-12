#!/bin/bash -x
#SBATCH --account=xxxxxx
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=10:00:00
#SBATCH --partition=dc-hwai
#SBATCH --gres=gpu:4
#SBATCH --output=slurm/logs/run_hetionet_%j.out
#SBATCH --error=slurm/logs/run_hetionet_%j.err

set -euo pipefail

export PYTHONPATH="$SLURM_SUBMIT_DIR":${PYTHONPATH:-}

module load Stages
module load Python/3.13.5
source ".venv/bin/activate"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/p/project1/xxxxxxx/.cache/huggingface/datasets
export HF_HUB_CACHE=/p/project1/xxxxxxx/.cache/huggingface/hub
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Generate dataset if not already present
if [[ ! -f datasets/hetionet_train.jsonl ]]; then
  echo "=== Generating Hetionet JSONL dataset ==="
  python datasets/convert_hetionet.py \
    --sample_compounds 900 \
    --seed 42 \
    --out_dir datasets/
fi

# Run JEPA fine-tuning
bash run_hetionet_stp.sh
