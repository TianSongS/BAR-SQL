#!/bin/bash

# 环境依赖安装
pip install --upgrade pip
pip install "deepspeed>=0.17.6"
pip install "trl>=0.23.1"
pip install transformers==4.56.2
pip install "accelerate>=1.10.1"
pip install ms-swift==3.9.3
pip install sqlglot
pip install scipy
pip install scikit-learn

# DeepSpeed兼容性修复（针对torch.load的weights_only警告）
sudo chmod +w /usr/local/lib/python3.11/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py
cp /usr/local/lib/python3.11/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py{,.bak}
sed -i 's/partition = torch.load(path, map_location=map_location)/partition = torch.load(path, map_location=map_location, weights_only=False)/g' /usr/local/lib/python3.11/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py
sudo chmod -w /usr/local/lib/python3.11/site-packages/deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py

# 环境变量配置
export RLHF_TYPE="grpo"
export MODEL_TYPE="qwen3"
export TRAIN_TYPE="full"
export TORCH_DTYPE="bfloat16"
export LORA_RANK=8
export MAX_COMPLETION_LENGTH=2048
export EPOCH=5
export TRAIN_BATCH_SIZE=8
export LEARN_RATE=1e-6
export SAVE_STEP=50
export NUM_GENERATION=2
export TEMPERATURE=1

# 多机分布式训练配置
export NNODES=<num_nodes>              # 节点数量
export NODE_RANK=<node_rank>           # 当前节点rank (0, 1, 2, ...)
export MASTER_ADDR=<master_ip>         # 主节点IP地址
export MASTER_PORT=<master_port>       # 主节点端口
export NPROC_PER_NODE=8                # 每个节点的GPU数量

# NCCL配置
export NCCL_DEBUG=INFO
export TORCH_CPP_SHOW_STACKTRACES=1
export NCCL_ASYNC_ERROR_HANDLING=1

# VLLM推理服务器配置
export ROLLOUT_IP=<vllm_server_ip>     # VLLM推理服务器IP
export ROLLOUT_PORT=8000               # VLLM推理服务器端口

# 核心RLHF训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
    --rlhf_type $RLHF_TYPE \
    --beta 0 \
    --model <your_model_path> \
    --model_type $MODEL_TYPE \
    --external_plugins <your_plugin_path>/plugin.py \
    --reward_funcs sql_acc still_format sql_grammar_cheak soft_overlong \
    --reward_weights 1.5 1.0 0.9 0.8 \
    --max_completion_length $MAX_COMPLETION_LENGTH \
    --soft_cache_length 409 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dynamic_sample true \
    --overlong_filter true \
    --max_resample_times 3 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host $ROLLOUT_IP \
    --vllm_server_port $ROLLOUT_PORT \
    --vllm_server_timeout 240 \
    --async_generate true \
    --num_iterations 2 \
    --train_type $TRAIN_TYPE \
    --torch_dtype $TORCH_DTYPE \
    --lora_rank $LORA_RANK \
    --lora_alpha 32 \
    --target_modules all-linear \
    --dataset <your_dataset_path> \
    --split_dataset_ratio 0 \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 8 \
    --learning_rate $LEARN_RATE \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps $SAVE_STEP \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 4096 \
    --output_dir <your_output_path> \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations $NUM_GENERATION \
    --temperature $TEMPERATURE \
    --system <your_prompt_file_path>/prompt.txt \
    --deepspeed zero2 \
    --log_completions true \
    --dataset_shuffle false \
    --train_dataloader_shuffle true \
    --attn_impl flash_attn