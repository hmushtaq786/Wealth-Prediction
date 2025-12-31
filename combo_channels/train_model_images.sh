#!/bin/bash
#SBATCH --job-name=train-raw-model
#SBATCH -t 20:00:00                                          # estimated time # TODO: adapt to your needs/ the full training run will take approx. 5 h on an A100
#SBATCH -p grete:shared                                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -o "%25N  %5c  %10m  %32f  %10G %18P " | grep gpu
#SBATCH --gres=gpu:A100:1                                   # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH -a 0-4                                            # 5 trials = 5 jobs, Max 5 trials at once
#SBATCH -n 1
#SBATCH -c 8    # Each trial gets 8 CPU cores
#SBATCH --output=../../slurm_logs/raw_images/model_training/training-score_trial_%A_%a.out      # where to write output, %x give job name, %j names job id
#SBATCH --error=../../slurm_logs/raw_images/model_training/training-score_trial_%A_%a.err       # where to write slurm error

# === Workspace allocation ===
WS_DIR=$(ws_allocate -F ceph-ssd hmushtaq 30)
export WS_DIR
echo "Workspace allocated at $WS_DIR"

# === Environment setup ===
module load python
module load uv
# uv venv
source ../.venv/bin/activate

# === Debugging ===
python --version

# === Training ===
echo "Starting training at $(date) for Model: $MODEL"
echo "Starting trial $SLURM_ARRAY_TASK_ID on GPU $CUDA_VISIBLE_DEVICES"
srun python -u train_model_images.py --trial-id $SLURM_ARRAY_TASK_ID --model $MODEL
echo "Training finished at $(date) for Model: $MODEL"
