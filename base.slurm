#!/bin/bash
#SBATCH --job-name=base
#SBATCH --output=./logs/test.out
#SBATCH --error=./logs/test.err
#SBATCH --partition=gpuA40x4
#SBATCH --account=beis-delta-gpu
#SBATCH --constraint="projects"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00

module load cuda
module load anaconda3_gpu
module load gcc

# deactivate any base conda, then activate your 'llm' env
source deactivate
export PATH="/u/akanodia/.conda/envs/llm/bin:$PATH"
source ~/.bashrc
conda activate llm

# Setup distributed env
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345

cd /projects/beis/akanodia/Megatron-LM

mkdir -p ./logs
mkdir -p ./tb/base
mkdir -p /projects/beis/akanodia/checkpoints/meg_pretrained

srun python pretrain_gpt.py \
  --num-layers 2 \
  --hidden-size 32 \
  --num-attention-heads 2 \
  --seq-length 32 \
  --max-position-embeddings 32 \
  --micro-batch-size 4 \
  --global-batch-size 16 \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file /projects/beis/akanodia/oscar/gpt2-vocab.json \
  --merge-file /projects/beis/akanodia/oscar/gpt2-merges.txt \
  --data-path /projects/beis/akanodia/meg-gpt2_text_document \
  --train-iters 2000 \
  --log-interval 200 \
  --transformer-impl local \
  --tensorboard-dir ./tb/base \
  --log-timers-to-tensorboard \
  --lr 1e-3 \
  --min-lr 1e-4 \
  --lr-decay-style cosine \
  --lr-warmup-fraction 0.2 \
  --lr-decay-iters 100 \
  --save /projects/beis/akanodia/checkpoints/meg_pretrained \
  --save-interval 500