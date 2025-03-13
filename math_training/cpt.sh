#!/bin/bash
REQUIRED_GPUS=2  # 你希望使用的GPU数量
batch_size=8 # per_device
accumula=$((64/$REQUIRED_GPUS/$batch_size))
datasets=("squad_texts") 
learning_rate=5e-6
num_train_epochs=12
save_strategySet=("steps" "epoch")
save_strategy=${save_strategySet[1]}
cd /home/nfs05/hup/enhanceKnowledge

BASE_MODEL_PATH=/home/nfs04/hup/Qwen/Qwen2.5-1.5B

MASTER_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
# 自动找到空闲的GPU
FREE_GPUS=$(python -c "
import subprocess
import re
import sys

# Function to find free GPUs
def find_free_gpus(required_gpus):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free', '--format=csv,nounits,noheader'], universal_newlines=True)
    lines = output.strip().split('\\n')
    free_gpus = []
    for i, line in enumerate(lines):
        util, mem = map(int, re.split(',\\s*', line))
        if mem > 42000:  # Customize GPU memory and utilization thresholds if needed
            free_gpus.append(str(i))
    return free_gpus

required_gpus = $REQUIRED_GPUS
free_gpus = find_free_gpus(required_gpus)
if len(free_gpus) == 0:
    sys.stderr.write('Error: No GPUs available.\\n')
    sys.exit(1)
final_gpus = free_gpus[:min(len(free_gpus), required_gpus)]
if len(final_gpus) < required_gpus:
    print(f'Warning: Only {len(final_gpus)} GPUs available, but {required_gpus} were requested.')
print(','.join(final_gpus))

")

for dataset in "${datasets[@]}"; do
    output_dir=/home/nfs05/hup/enhanceKnowledge/cpt/model/${dataset}_${num_train_epochs}_${learning_rate}
    echo "dataset: $dataset"
    echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
    echo "GPU:$FREE_GPUS"
    echo "lr: $learning_rate epoch:$num_train_epochs save_strategy:$save_strategy"
    deepspeed --include localhost:$FREE_GPUS --master_port=$MASTER_PORT /home/nfs05/hup/enhanceKnowledge/LLaMA-Factory/src/train.py \
        --stage pt \
        --do_train \
        --model_name_or_path $BASE_MODEL_PATH \
        --dataset ${dataset} \
        --dataset_dir /home/nfs05/hup/enhanceKnowledge/cpt/data \
        --deepspeed /home/nfs05/hup/enhanceKnowledge/cpt/ds_z0_config.json \
        --finetuning_type full \
        --output_dir $output_dir \
        --overwrite_cache \
        --overwrite_output_dir \
        --max_grad_norm 1 \
        --packing "False" \
        --preprocessing_num_workers 32 \
        --per_device_train_batch_size $batch_size \
        --gradient_accumulation_steps $accumula \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --warmup_steps 0 \
        --evaluation_strategy "no" \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --max_samples 1000000 \
        --val_size 0 \
        --do_eval False \
        --plot_loss \
        --bf16 \
        --ddp_timeout 180000000 \
        --save_steps 20000 \
        --save_strategy $save_strategy 

    echo "Deleting global_step folders in each checkpoint folder in $output_dir"
    if [ -d "$output_dir" ]; then
        find "$output_dir" -type d -path '*/checkpoint-*/global_step*' -exec echo "Deleting: {}" \; -exec rm -rf {} +
    else
        echo "Error: Directory $output_dir does not exist!"
    fi

done


