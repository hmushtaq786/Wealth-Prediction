#!/bin/bash
#SBATCH --job-name=plot-optuna-best-results
#SBATCH -t 1:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --output=../slurm_logs/single/plot_results/best_trials/plot_results_%A_%a.out
#SBATCH --error=../slurm_logs/single/plot_results/best_trials/plot_results_%A_%a.err

# === Environment setup ===
module load python
module load uv
# uv venv
source .venv/bin/activate

echo "Starting plotting best trials"
python -u plot_optuna_best_trial.py --index $INDEX --model $MODEL
echo "Finished plotting best trials"