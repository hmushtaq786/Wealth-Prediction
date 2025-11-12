#!/bin/bash
#SBATCH --job-name=create_study
#SBATCH -t 02:00:00
#SBATCH -p grete:interactive
#SBATCH --gres=gpu:1g.10gb
#SBATCH --output=../../slurm_logs/create_study/create_study_%A.out
#SBATCH --error=../../slurm_logs/create_study/create_study_%A.err

# === Environment setup ===
module load python
module load uv
# uv venv
source ../.venv/bin/activate

echo "Creating study for index: $INDEX and model: $MODEL"
python -u create_optuna_study.py --index "$INDEX" --model "$MODEL"