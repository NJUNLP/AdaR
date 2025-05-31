import argparse
import json
import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
import torch
import os
import sys

# CoT
# messages = [
#     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
#     {"role": "user", "content": {}}
# ]

# TIR
# messages = [
#     {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
#     {"role": "user", "content": prompt}
# ]

MODEL_PATH = {
    "qwen8b": "/home/nfs05/model/Qwen3-8B",
    "qwen7b": "/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e/",
    "qwen7b_instruct": "/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d",
    "qwen1.5b": "/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2",
    "qwen1.5b_instruct": "/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35",
    "qwen7b_sft": "/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/model/qwen7b-orca_10k_train-repeat-1-1e-6-32-no_shuffle"
}


def main():
    if len(sys.argv) < 2:
    # Parse command-line arguments
        with open('config.yaml', 'r') as file:
            args = yaml.safe_load(file)
    else:
        with open(sys.argv[1], 'r') as file:
            args = yaml.safe_load(file)

    output_path = args['data']['output-path']
    if not output_path:
        output_path = os.path.join("results", f"{args['model']['model_name']}-{args['data']['task']}-{args['data']['source_dataset']}.json")
        args['data']['test-set-path'] = os.path.join(args['data']['test-set-path'], f"{args['data']['task']}-{args['data']['source_dataset']}.json")


    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[args['model']['model_name']])

    # Load prompts from the JSON test set file
    with open(args['data']['test-set-path'], "r", encoding="utf-8") as file:
        prompts = []
        
        # jsonl
        # for line in file:
        #     input_data = json.loads(line)
        #     prompts.append(input_data['prompt'])
        
        
        # json
        data = json.load(file)
        prompts = [entry['prompt'] for entry in data]

        # for entry in data:
        #     prompts.append(tokenizer.apply_chat_template(
        #                     messages,
        #                     tokenize=False,
        #                     add_generation_prompt=True))

    
    if not isinstance(prompts, list):
        raise ValueError("The test set JSON file must contain a list of prompts.")

    # Define sampling parameters
    if 'beam_search' in args['inference']:
        sampling_params = BeamSearchParams(
            **args['inference']['beam_search']
        )
    else:
        sampling_params = SamplingParams(
            **args['inference']['sampling-params'],
            stop=[token for token in tokenizer.all_special_tokens if "end" in token],
        )

    llm = LLM(model=MODEL_PATH[args['model']['model_name']], 
              dtype=torch.bfloat16, 
              tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')),
              gpu_memory_utilization=0.95)

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        data[i]['generated_texts'] = [gen.text for gen in output.outputs]
        results.append(data[i])

    # Save results to a JSON file
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)

    print(f"Results saved to {args['data']['output-path']}")

if __name__ == "__main__":
    main()
