import re
import json
import argparse
import yaml
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import torch
import os
from collections import defaultdict, Counter


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"\\boxed\{(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) == 0:
        res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

# system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. Output each step in a separate line, and explicitly state the final answer after the final step within \\boxed{}."
# system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

test_set = {
    'gsm8k': {
        "path": "/home/nfs02/laizj/experiment/uncertainty_analysis/analysis_unknown/data/raw_data/gsm8k-test.json",
        "query_key": "query",
        "answer_key": "answer",
    },
    'gsm_sym_main': {
        "path": "/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/data/gsm_symbolic_main.json",
        "query_key": "question",
        "answer_key": "answer",
    },
    'gsm_sym_p1': {
        "path": "/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/data/gsm_symbolic_p1.json",
        "query_key": "question",
        "answer_key": "answer",
    },
    'gsm_sym_p2': {
        "path": "/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/data/gsm_symbolic_p2.json",
        "query_key": "question",
        "answer_key": "answer",
    },
    "orca_disturbed": {
        "path": "data/qwen7b_instruct-generate_cot_with_code-qwen72b_instruct-generate_template_and_code-orca_10k_train-disturbed_1.0_16-test.json",
        "query_key": "instruction",
        "answer_key": "answer",
    },
    "orca_disturbed_code": {
        "path": "data/qwen7b_instruct-generate_cot_with_code-qwen72b_instruct-generate_template_and_code-orca_10k_train-disturbed_1.0_16-test-python.json",
        "query_key": "instruction",
        "answer_key": "answer",
    },
    "aime2024": {
        "path": "/home/nfs02/laizj/experiment/uncertainty_analysis/analysis_unknown/data/raw_data/aime2024.json",
        "query_key": "query",
        "answer_key": "answer",
    },
    "aime2025": {
        "path": "/home/nfs02/laizj/experiment/uncertainty_analysis/analysis_unknown/data/raw_data/aime2025.json",
        "query_key": "query",
        "answer_key": "answer",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--lora_path", type=str, default="", help="Path to the LoRA model")
    parser.add_argument("--test_set", type=str, required=True, choices=test_set.keys(), help="Test set name")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output IDs file")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result file")
    args = parser.parse_args()

    test_set_attr = test_set[args.test_set]
    output_path = args.output_path
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if not os.path.exists(output_path):

        # Load prompts from the JSON test set file
        with open(test_set_attr['path'], "r", encoding="utf-8") as file:
            prompts = []
            
            # jsonl
            # for line in file:
            #     input_data = json.loads(line)
            #     prompts.append(input_data['prompt'])
            
            
            # json
            data = json.load(file)
            # prompts = [entry[test_set_attr['query_key']] for entry in data]

            for entry in data:
                messages = [
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": entry[test_set_attr['query_key']]}
                ]
                prompts.append(tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True))

        
        if not isinstance(prompts, list):
            raise ValueError("The test set JSON file must contain a list of prompts.")

        # Define sampling parameters
        # sampling_params = SamplingParams(
        #     temperature=0.7,
        #     top_p=0.9,
        #     max_tokens=1536,
        #     n=5,
        #     stop=[token for token in tokenizer.all_special_tokens if "end" in token]
        # )
        sampling_params = SamplingParams(
            temperature=0,
            top_p=0.9,
            max_tokens=1536,
            n=1,
            stop=[token for token in tokenizer.all_special_tokens if "end" in token]
        )
            

        llm = LLM(model=args.model_path, 
                dtype=torch.bfloat16, 
                tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')),
                gpu_memory_utilization=0.92,
                enable_lora=bool(args.lora_path),
                max_lora_rank=64)

        if args.lora_path:
            outputs = llm.generate(prompts, sampling_params, lora_request=LoRARequest("math_adapter", 1, lora_path=args.lora_path))
        else:
            outputs = llm.generate(prompts, sampling_params)

        results = []
        for i, output in enumerate(outputs):
            data[i]['generated_texts'] = [gen.text for gen in output.outputs]
            results.append(data[i])

        # Save results to a JSON file
        with open(output_path, "w", encoding="utf-8") as out_file:
            json.dump(results, out_file, indent=4, ensure_ascii=False)

    else:
        with open(output_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        

    correct_ids = []
    half_ids = []
    wrong_ids = []
    major = []
    result = defaultdict(list)
    
    for idx, line in enumerate(data):
        # if line["id"] not in ids: continue
        line["id"] = idx
        answer = extract_last_num(str(line[test_set_attr['answer_key']]))
        generated_nums = [extract_last_num(gen) for gen in line["generated_texts"][:5]]
        if any(abs(num - answer) < 1e-2 for num in generated_nums):
            result["pass@5"].append(idx)
        else:
            result["fail_pass@5"].append(idx)
        counted = Counter([round(num, 2) for num in generated_nums])
        most_common_value, _ = counted.most_common(1)[0]
        if abs(most_common_value - answer) < 1e-2:
            result["major@5"].append(idx)
        else:
            result["fail_major@5"].append(idx)
    
    with open(args.result_path, "a", encoding="utf-8") as f:
        f.write("-" * 10 + "  " * 5 + args.test_set + " | " + args.lora_path + "  " * 5 + "-" * 10 + "\n")
        # f.write(f"Correct: {len(correct_ids)}\n")
        # f.write(f"Half: {len(half_ids)}\n")
        # f.write(f"Wrong: {len(wrong_ids)}\n")
        f.write(f'major@5 正确率: {len(result["major@5"])} / {len(data)} = {len(result["major@5"]) / len(data) * 100:.2f}%\n')
        f.write(f'错误ID: {result["fail_major@5"][:20]}\n\n')
        f.write(f'pass@5 正确率: {len(result["pass@5"])} / {len(data)} = {len(result["pass@5"]) / len(data) * 100:.2f}%\n')
        f.write(f"错误ID: {result['fail_pass@5'][:20]}\n")
        f.write("\n\n")
    # with open(args.output_path, "w") as f:
    #     f.write(str(correct_ids))