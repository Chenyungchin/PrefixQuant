#!/bin/bash
#SBATCH --job-name=prefixquant   # create a short name for your job=
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # 32 GB total memory on the node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=gpu80       # use bigger GPU memory
#SBATCH --output=slurm_log/out/out_%j.txt
#SBATCH --error=slurm_log/err/err_%j.txt
# #SBATCH --mail-type=begin        # send mail when job begins
# #SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=yc9182@princeton.edu

# enable detailed CUDA error tracing
export CUDA_LAUNCH_BLOCKING=1

# path to datasets
export WIKITEXT_LOCAL_DIR=/scratch/gpfs/NVERMA/jim/datasets/wikitext-2-raw-v1
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

python main.py \
  --model_path ../models/llama-2-7b-hf \
  --model_name llama-2-7b \
  --output_dir ./log/llama-2-7b-static \
  --wbits 16 \
  --input_bits 8 --input_mode static --input_group_size -1 \
  --k_bits 16 --v_bits 16 --kv_mode dynamic --kv_group_size -1 \
  --set_prefixed_tokens \
  --eval_ppl \
  --save_quant_dir ./pre_quantized_models/llama-2-7b-static
