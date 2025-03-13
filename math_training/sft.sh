#!/bin/bash
batch_size=2 # per_device
gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
accumula=$((64 / batch_size / gpu_count))
datasets=("gsm8k") 
learning_rate=2e-5
num_train_epochs=3
save_strategySet=("steps" "epoch")
save_strategy=${save_strategySet[0]}
cd /home/nfs02/laizj/experiment/uncertainty_analysis/math_training

# BASE_MODEL_PATH=/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
BASE_MODEL_PATH=/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e/

MASTER_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

for dataset in "${datasets[@]}"; do
    output_dir=/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/model/${dataset}_${num_train_epochs}_${learning_rate}
    deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port=$MASTER_PORT /home/nfs03/laizj/code/LLaMA-Factory/src/train.py \
        --stage sft \
        --do_train \
        --model_name_or_path $BASE_MODEL_PATH \
        --template empty \
        --dataset ${dataset} \
        --dataset_dir /home/nfs02/laizj/experiment/uncertainty_analysis/math_training/data \
        --deepspeed /home/nfs02/laizj/experiment/uncertainty_analysis/math_training/ds_z2_offload_config.json \
        --finetuning_type full \
        --output_dir $output_dir \
        --overwrite_cache \
        --overwrite_output_dir \
        --max_grad_norm 1 \
        --packing "False" \
        --preprocessing_num_workers 32 \
        --per_device_train_batch_size ${batch_size} \
        --gradient_accumulation_steps ${accumula} \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --warmup_ratio 0.03 \
        --evaluation_strategy "no" \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${num_train_epochs} \
        --max_samples 1000000 \
        --val_size 0 \
        --do_eval False \
        --plot_loss \
        --bf16 \
        --ddp_timeout 180000000 \
        --save_steps 1000000 \
        --save_strategy $save_strategy \
        --optim adamw_torch

    echo "Deleting global_step folders in each checkpoint folder in $output_dir"
    if [ -d "$output_dir" ]; then
        find "$output_dir" -type d -path '*/checkpoint-*' -exec echo "Deleting: {}" \; -exec rm -rf {} +
    else
        echo "Error: Directory $output_dir does not exist!"
    fi

done


