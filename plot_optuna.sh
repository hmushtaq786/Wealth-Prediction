#!/bin/bash
#SBATCH --job-name=plot-optuna-results
#SBATCH -t 1:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --output=../slurm_files/resnet/plot_results.out
#SBATCH --error=../slurm_files/resnet/plot_results.err

# === Environment setup ===
module load python
module load uv
# uv venv
source .venv/bin/activate

echo "Starting plotting at $(date)"
python -u plot_optuna.py
echo "Finished plotting at $(date)"