#!/bin/bash
#SBATCH --job-name=train-resnet-gpu-optuna
#SBATCH -t 20:00:00                                          # estimated time # TODO: adapt to your needs/ the full training run will take approx. 5 h on an A100
#SBATCH -p grete:shared                                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -o "%25N  %5c  %10m  %32f  %10G %18P " | grep gpu
#SBATCH --gres=gpu:A100:1                                   # requesting GPU slices, see https://docs.hpc.gwdg.de/usage_guide/slurm/gpu_usage/index.html for more options
#SBATCH --exclude=ggpu142,ggpu201,ggpu212,ggpu121,ggpu122,ggpu130,ggpu140
#SBATCH -a 0-9%5                                            # 10 trials = 10 jobs, Max 3 trials at once
#SBATCH -n 1
#SBATCH -c 8    # Each trial gets 16 CPU cores
#SBATCH --output=../../slurm_files/resnet/ndvi/optuna_trials/trial_%A_%a.out      # where to write output, %x give job name, %j names job id
#SBATCH --error=../../slurm_files/resnet/ndvi/optuna_trials/trial_%A_%a.err       # where to write slurm error

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

# === Move training data to workspace if needed ===
# cp -r ~/thesis/data "$WS_DIR/data"

# === Creating Optuna Database ===
# python -c "import optuna; optuna.create_study(study_name='resnet34_parallel', storage='sqlite:///optuna/storage/resnet34.db', direction='maximize')"

# === Training ===
echo "Starting training at $(date)"
echo "Starting trial $SLURM_ARRAY_TASK_ID on GPU $CUDA_VISIBLE_DEVICES"
srun python -u resnet_ndvi.py --trial-id $SLURM_ARRAY_TASK_ID
echo "Training finished at $(date)"
