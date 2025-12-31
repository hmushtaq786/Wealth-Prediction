#!/bin/bash
#SBATCH --job-name=correlation
#SBATCH -t 02:00:00
#SBATCH -p grete:interactive
#SBATCH --gres=gpu:1g.10gb
#SBATCH --output=../../slurm_logs/correlation/correlation_%A.out
#SBATCH --error=../../slurm_logs/correlation/correlation_%A.err

WS_DIR=$(ws_allocate -F ceph-ssd hmushtaq 30)
export WS_DIR
echo "Workspace allocated at $WS_DIR"

# === Environment setup ===
module load python
module load uv
# uv venv
source ../.venv/bin/activate

echo "Calculating correlation for the dataset"
python -u correlation.py