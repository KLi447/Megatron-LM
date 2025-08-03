#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

# These are set by SLURM and exported by the wrapper
: "${GPUS_PER_NODE:?GPUS_PER_NODE not set}"
: "${NUM_NODES:?NUM_NODES not set}"
: "${NODE_RANK:?NODE_RANK not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
MASTER_PORT=12345

CHECKPOINT_PATH=/projects/beis/akanodia/ckpt
TENSORBOARD_LOGS_PATH=/projects/beis/akanodia/tensorboard
VOCAB_FILE=/projects/beis/akanodia/oscar/gpt2-vocab.json
MERGE_FILE=/projects/beis/akanodia/oscar/gpt2-merges.txt
DATA_PATH=/projects/beis/akanodia/meg-gpt2_text_document

GPT_MODEL_ARGS=(
    --num-layers 4 
    --hidden-size 256 
    --num-attention-heads 4 
    --seq-length 128
    --max-position-embeddings 512 
    --attention-backend auto
)

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 16 
    --train-iters 50 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .01 
    --lr-decay-iters 8 
    --transformer-impl local
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 98,2,0
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 100 
    --eval-interval 100
    --eval-iters 5
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

rm -rf $CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH

# Call torchrun with proper SLURM environment variables
PYTHONUNBUFFERED=1 \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
python -m torch.distributed.run \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    pretrain_gpt.py \
    "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}"

rm -rf $CHECKPOINT_PATH
