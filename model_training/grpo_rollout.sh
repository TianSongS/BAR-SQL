CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rollout \
    --model <your_model_path> \
    --tensor_parallel_size 4 \
    --data_parallel_size 2 \
    --model_type qwen3