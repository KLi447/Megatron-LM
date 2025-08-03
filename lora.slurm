#!/bin/bash
#SBATCH --job-name=finetune-lora
#SBATCH --output=./logs/finetune_lora.out
#SBATCH --error=./logs/finetune_lora.err
#SBATCH --partition=gpuA40x4
#SBATCH --account=beis-delta-gpu
#SBATCH --constraint="projects"
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Load environment
module load cuda
module load anaconda3_gpu
module load gcc

# Activate Conda env
source deactivate
export PATH="/u/akanodia/.conda/envs/llm/bin:$PATH"
source ~/.bashrc
conda activate llm

# Distributed env
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345

# Go to Megatron repo
cd /projects/beis/akanodia/Megatron-LM

# Run finetuning
srun python pretrain_gpt.py \
  --num-layers 2 \
  --hidden-size 32 \
  --num-attention-heads 2 \
  --seq-length 32 \
  --max-position-embeddings 32 \
  --micro-batch-size 4 \
  --global-batch-size 16 \
  --train-iters 2000 \
  --lr 1e-4 \
  --min-lr 1e-4 \
  --lr-decay-style constant \
  --lr-warmup-fraction 0.2 \
  --lr-decay-iters 2000 \
  --log-interval 1 \
  --eval-interval 200 \
  --eval-iters 10 \
  --save-interval 1000 \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file /projects/beis/akanodia/oscar/gpt2-vocab.json \
  --merge-file /projects/beis/akanodia/oscar/gpt2-merges.txt \
  --data-path /projects/beis/akanodia/meg-gpt2_text_document \
  --load /projects/beis/akanodia/checkpoints/meg_pretrained \
  --tensorboard-dir ./tb/lora \
  --log-timers-to-tensorboard \
   --no-load-optim \
  --no-load-rng \
  --finetune \
  --transformer-impl local \
  --enable-lora \
  --lora-rank 8 \
  --lora-alpha 8 \
  --lora-dropout 0.1 \
  --dist-ckpt-strictness ignore_all \
