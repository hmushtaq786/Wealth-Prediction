#!/bin/bash
#SBATCH --job-name=train-efficientnet-msavi
#SBATCH -t 20:00:00                                          # estimated time # TODO: adapt to your needs/ the full training run will take approx. 5 h on an A100
#SBATCH -p grete:shared                                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -o "%25N  %5c  %10m  %32f  %10G %18P " | grep gpu
#SBATCH --gres=gpu:A100:1                                   # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --exclusive
#SBATCH -a 0-9%5                                            # 10 trials = 10 jobs, Max 3 trials at once
#SBATCH -n 1
#SBATCH -c 8    # Each trial gets 16 CPU cores
#SBATCH --output=../../slurm_logs/model_training/efficientnet/msavi/trial_%A_%a.out      # where to write output, %x give job name, %j names job id
#SBATCH --error=../../slurm_logs/model_training/efficientnet/msavi/trial_%A_%a.err       # where to write slurm error

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
echo "Starting training at $(date) for Model: $MODEL and Index: $INDEX"
echo "Starting trial $SLURM_ARRAY_TASK_ID on GPU $CUDA_VISIBLE_DEVICES"
srun python -u train_model.py --trial-id $SLURM_ARRAY_TASK_ID --index $INDEX --model $MODEL
echo "Training finished at $(date) for Model: $MODEL and Index: $INDEX"
