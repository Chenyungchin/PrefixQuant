#!/bin/bash

# Start an interactive session with GPU
srun --gpus-per-node=1 --mem=80G --time=1:00:00 --cpus-per-task=3 --pty /bin/bash -c '
# enable detailed CUDA error tracing
export CUDA_LAUNCH_BLOCKING=1

# path to datasets
export HF_HOME=/scratch/gpfs/NVERMA/jim/datasets
export HF_DATASETS_CACHE=/scratch/gpfs/NVERMA/jim/datasets
export HF_DATASETS_OFFLINE=1

# show GPU info
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load cudatoolkit/12.6
module load anaconda3/2025.6
conda activate prefixquant

# THE SHIELD: Block the ~/.local folder
export PYTHONNOUSERSITE=1

# Start zsh shell
exec /bin/zsh
'
