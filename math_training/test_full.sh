#!/bin/bash

cd /home/nfs02/laizj/experiment/uncertainty_analysis/math_training || exit

test_datasets=("gsm8k" "orca_disturbed" "gsm_sym_main" "gsm_sym_p1" "gsm_sym_p2" "aime2024" "aime2025")
# test_datasets=("gsm_sym_main" "gsm_sym_p1" "gsm_sym_p2")
# load_dir=model/qwen7b_instruct-generate_cot_with_code-generate_cot_with_code-qwen7b_instruct-orca_10k_train-disturbed_1.0-one_3_1e-4
load_dir=$2

model_name=$1
declare -A BASE_MODEL_DICT=(
    ["qwen7b"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e/"
    ["qwen7b_instruct"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
    ["qwen1.5b"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    ["qwen1.5b_instruct"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
)

lora_dirs=($(find ${load_dir} -type d -name "*checkpoint*" | sort -V))
lora_dirs=("$load_dir")
result_path=results/$(basename ${load_dir}).txt


echo -e "Testing with model:\t$(basename $load_dir)"
echo -e "Testing with dataset:\t${test_datasets[*]}"
echo -e "Result path: ${result_path}\n\n"


touch ${result_path}

for dataset in "${test_datasets[@]}"; do
    for lora_path in "${lora_dirs[@]}"; do
        if [ "$(basename ${load_dir})" = "$(basename ${lora_path})" ]; then
            output_path="results/$(basename ${load_dir})-${dataset}.json"
        else
            output_path="results/$(basename ${load_dir})-$(basename ${lora_path})-${dataset}.json"
        fi
        

        echo -e "\n\nTesting with model: $(basename $lora_path) in testset $dataset\n"

        python check_ans.py \
            --model_path "$lora_path" \
            --test_set "$dataset" \
            --output_path "$output_path" \
            --result_path "$result_path"
        
    done
done