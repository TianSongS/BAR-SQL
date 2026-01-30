#!/bin/bash

# 环境依赖安装
pip install "deepspeed>=0.17.6"
pip install "trl>=0.23.1"
pip install transformers==4.57.1
pip install ms-swift==3.9.3

# 环境变量配置
export MODEL_TYPE="qwen3"
export SFT_TYPE="full"
export BATCH_SIZE=2
export EPOCHS=5
export LEARNING_RATE=5e-5
export EVAL_STEPS=50
export SAVE_STEPS=100
export LOGGING_STEPS=5
export MAX_LENGTH=4096
export DDP_ZERO_TYPE=2

# 多机分布式训练配置
export NNODES=<num_nodes>              # 节点数量
export NODE_RANK=<node_rank>           # 当前节点rank (0, 1, 2, ...)
export MASTER_ADDR=<master_ip>         # 主节点IP地址
export MASTER_PORT=<master_port>       # 主节点端口
export NPROC_PER_NODE=<gpus_per_node>  # 每个节点的GPU数量

# 核心训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model <your_model_path> \
    --model_type $MODEL_TYPE \
    --train_type $SFT_TYPE \
    --num_train_epochs $EPOCHS \
    --dataset <your_dataset_path> \
    --torch_dtype bfloat16 \
    --target_modules all-linear \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --learning_rate $LEARNING_RATE \
    --gradient_accumulation_steps 16 \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --max_length $MAX_LENGTH \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir <your_output_path> \
    --deepspeed zero$DDP_ZERO_TYPE \
    --attn_impl flash_attn