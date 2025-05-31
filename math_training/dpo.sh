#!/bin/bash
batch_size=2 # per_device
gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
accumula=$((16 / batch_size / gpu_count))
datasets=("qwen7b_instruct-generate_dpo_pairs-qwen7b_instruct-generate_cot_with_code-qwen72b_instruct-generate_template_and_code-orca_10k_train-disturbed_1.0-code" "qwen7b_instruct-generate_dpo_pairs-qwen7b_instruct-generate_cot_with_code-qwen72b_instruct-generate_template_and_code-orca_10k_train-disturbed_1.0") 
learning_rate=1e-6
num_train_epochs=1
save_strategySet=("steps" "epoch")
save_strategy=${save_strategySet[0]}
cd /home/nfs02/laizj/experiment/uncertainty_analysis/math_training

# BASE_MODEL_PATH=/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2
BASE_MODEL_PATH=/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e/
# BASE_MODEL_PATH=/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d

model_name=qwen7b_instruct
declare -A BASE_MODEL_DICT=(
    ["qwen7b"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e/"
    ["qwen7b_instruct"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
)


MASTER_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

output_dir=/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/model/dpo-$model_name-mixed-${num_train_epochs}-${learning_rate}
# deepspeed --include localhost:$CUDA_VISIBLE_DEVICES --master_port=$MASTER_PORT /home/nfs03/laizj/code/LLaMA-Factory/src/train.py \
#     --stage dpo \
#     --do_train \
#     --model_name_or_path ${BASE_MODEL_DICT[$model_name]} \
#     --template qwen \
#     --dataset "$(IFS=,; echo "${datasets[*]}")" \
#     --dataset_dir /home/nfs02/laizj/experiment/uncertainty_analysis/math_training/data \
#     --deepspeed /home/nfs02/laizj/experiment/uncertainty_analysis/math_training/ds_z2_config.json \
#     --finetuning_type lora \
#     --lora_target "q_proj, v_proj, o_proj" \
#     --lora_rank 64 \
#     --lora_dropout 0.05 \
#     --output_dir $output_dir \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --max_grad_norm 1 \
#     --preprocessing_num_workers 32 \
#     --per_device_train_batch_size ${batch_size} \
#     --gradient_accumulation_steps ${accumula} \
#     --cutoff_len 2048 \
#     --pref_beta 0.1 \
#     --pref_loss sigmoid \
#     --lr_scheduler_type cosine_with_min_lr \
#     --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
#     --optim adamw_torch \
#     --weight_decay 0.01 \
#     --adam_beta1 0.9 \
#     --adam_beta2 0.999 \
#     --warmup_ratio 0.03 \
#     --logging_steps 50 \
#     --evaluation_strategy "no" \
#     --learning_rate ${learning_rate} \
#     --num_train_epochs ${num_train_epochs} \
#     --flash_attn fa2 \
#     --max_samples 1000000 \
#     --val_size 0 \
#     --do_eval False \
#     --plot_loss \
#     --bf16 \
#     --ddp_timeout 180000000 \
#     --save_steps 500 \
#     --save_strategy $save_strategy

# echo "Deleting global_step folders in each checkpoint folder in $output_dir"
# if [ -d "$output_dir" ]; then
#     find "$output_dir" -type d -path '*/checkpoint-*' -exec echo "Deleting: {}" \; -exec rm -rf {} +
# else
#     echo "Error: Directory $output_dir does not exist!"
# fi
bash test.sh $model_name $output_dir

