#!/bin/bash
#SBATCH --job-name=create_studies
#SBATCH -t 1:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --output=../slurm_files/resnet/create_studies.out
#SBATCH --error=../slurm_files/resnet/create_studies.err

# === Environment setup ===
module load python
module load uv
# uv venv
source .venv/bin/activate

echo "Starting plotting at $(date)"
python -u create_studies.py
echo "Finished plotting at $(date)"