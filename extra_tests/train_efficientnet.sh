#!/bin/bash
#SBATCH --job-name=train-efficientnet-gpu
#SBATCH -t 05:00:00                  # estimated time # TODO: adapt to your needs/ the full training run will take approx. 5 h on an A100
#SBATCH -p grete:shared              # the partition you are training on (i.e., which nodes), for nodes see sinfo -o "%25N  %5c  %10m  %32f  %10G %18P " | grep gpu
#SBATCH -G A100:1                   # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --output=../slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=../slurm_files/slurm-%x-%j.err      # where to write slurm error

# === Workspace allocation ===
WS_DIR=$(ws_allocate -F ceph-ssd hmushtaq 30)
export WS_DIR
echo "Workspace allocated at $WS_DIR"

# === Environment setup ===
module load python
module load uv
# uv venv
source .venv/bin/activate

# === Debugging ===
python --version

# === Move training data to workspace if needed ===
# cp -r ~/thesis/data "$WS_DIR/data"

# === Training ===
echo "Starting training at $(date)"
python -u 9_cnn_efficientnet.py
echo "Training finished at $(date)"
