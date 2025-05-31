#!/bin/bash

cd /home/nfs02/laizj/experiment/uncertainty_analysis/math_training || exit

test_datasets=("gsm8k" "orca_disturbed" "orca_disturbed_code" "gsm_sym_main" "gsm_sym_p1" "gsm_sym_p2" "aime2024" "aime2025")
# load_dir=model/qwen7b_instruct-generate_cot_with_code-generate_cot_with_code-qwen7b_instruct-orca_10k_train-disturbed_1.0-one_3_1e-4


declare -A BASE_MODEL_DICT=(
    ["qwen7b_sft"]="/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/model/qwen7b-orca_10k_train-repeat-1-1e-6-32-no_shuffle"
    ["qwen7b"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e/"
    ["qwen7b_instruct"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d"
    ["qwen1.5b"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
    ["qwen1.5b_instruct"]="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
)

# for model_name in ${!BASE_MODEL_DICT[@]}; do
# result_path="results/${model_name}.txt"


# echo -e "Testing with model:\t${model_name}"
# echo -e "Testing with dataset:\t${test_datasets[*]}"
# echo -e "Output path: ${result_path}\n\n"


# touch $result_path

# for dataset in "${test_datasets[@]}"; do
#     output_path="results/${model_name}-${dataset}.json"

#     echo -e "\n\nTesting with model: ${model_name} in testset $dataset\n"

#     python check_ans.py \
#         --model_path "${BASE_MODEL_DICT[$model_name]}" \
#         --test_set "$dataset" \
#         --output_path "$output_path" \
#         --result_path "$result_path"
# done

# done

model_name=qwen7b_sft
result_path="results/${model_name}.txt"


echo -e "Testing with model:\t${model_name}"
echo -e "Testing with dataset:\t${test_datasets[*]}"
echo -e "Output path: ${result_path}\n\n"


touch $result_path

for dataset in "${test_datasets[@]}"; do
    output_path="results/${model_name}-${dataset}.json"

    echo -e "\n\nTesting with model: ${model_name} in testset $dataset\n"

    python check_ans.py \
        --model_path "${BASE_MODEL_DICT[$model_name]}" \
        --test_set "$dataset" \
        --output_path "$output_path" \
        --result_path "$result_path"
done

# declare -A TEST_MODEL_DICT=(
#     ["duorc"]="/home/nfs03/hp/enhanceKnowledge/data/hug/duorc/SelfRC/duorc_qa_fewshot.json"
#     ["newsqa"]="/home/nfs03/hp/enhanceKnowledge/data/hug/newsqa/newsqa_qa_fewshot.json"
#     ["search"]="/home/nfs03/hp/enhanceKnowledge/data/hug/searchqa/data/qa_fewshot.json"
#     ["coqa"]="/home/nfs03/hp/enhanceKnowledge/data/hug/coqa/data/coqa_qa_fewshot.json"
#     ["squad"]="/home/nfs03/hp/enhanceKnowledge/data/squad_fewshot.json"
# )