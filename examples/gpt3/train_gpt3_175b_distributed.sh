#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

# GPUS_PER_NODE=2
# Change for multinode config
# MASTER_ADDR=172.28.23.119
MASTER_PORT=12345
# NUM_NODES=2
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/projects/beis/kli44/ckpt
TENSORBOARD_LOGS_PATH=/projects/beis/kli44/tensorboard
VOCAB_FILE=/projects/beis/kli44/oscar/gpt2-vocab.json
MERGE_FILE=/projects/beis/kli44/oscar/gpt2-merges.txt
DATA_PATH=/projects/beis/kli44/meg-gpt2_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 8 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 256
    --max-position-embeddings 1024 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 4 
    --global-batch-size 64 
    --train-iters 3 
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
	--tensor-model-parallel-size 2 
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
    --save-interval 1000 
    --eval-interval 1000 
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

rm -rf $CHECKPOINT_PATH
mkdir $CHECKPOINT_PATH

echo 'Starting training...'

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}

echo 'done training...'

rm -rf $CHECKPOINT_PATH