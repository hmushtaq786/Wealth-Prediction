#!/bin/bash
#SBATCH --job-name=create_single_study
#SBATCH -t 02:00:00
#SBATCH -p grete:interactive
#SBATCH --gres=gpu:1g.10gb
#SBATCH --output=../../slurm_logs/raw_images/create_study/create_study_%A.out
#SBATCH --error=../../slurm_logs/raw_images/create_study/create_study_%A.err

# === Environment setup ===
module load python
module load uv
# uv venv
source ../.venv/bin/activate

echo "Creating study for model: $MODEL"
python -u create_images_optuna_study.py --model "$MODEL"