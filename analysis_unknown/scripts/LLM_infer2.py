import argparse
import json
import yaml
from vllm import LLM, SamplingParams
import torch
import os

def main():
    # Parse command-line arguments
    with open('config.yaml', 'r') as file:
        args = yaml.safe_load(file)

    # Load prompts from the JSON test set file
    with open("/home/nfs02/laizj/experiment/uncertainty_analysis/analysis_unknown/data/gsm8k-train-disturbed.json", "r", encoding="utf-8") as file:
        prompts = []
        # for line in file:
        #     input_data = json.loads(line)
        #     prompts.append(input_data['prompt'])
        #     chosens.append(input_data['chosen'])
        data = json.load(file)
        prompts = [entry['prompt'] for entry in data]

    
    if not isinstance(prompts, list):
        raise ValueError("The test set JSON file must contain a list of prompts.")

    # Define sampling parameters
    sampling_params = SamplingParams(
        **args['inference']['sampling-params']
    )

    # Initialize the LLM
    llm = LLM(model=args['model']['base-model-path'], 
              dtype=torch.bfloat16, 
              tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')),
              gpu_memory_utilization=0.95)

    # Run inference
    outputs = llm.generate(prompts, sampling_params)

    # Prepare results in JSON format
    results = []
    for i, output in enumerate(outputs):
        result_entry = {
            "prompt": prompts[i],
            "answer": data[i]['answer'],
            "generated_texts": [gen.text for gen in output.outputs]
        }
        results.append(result_entry)

    # Save results to a JSON file
    with open("/home/nfs02/laizj/experiment/uncertainty_analysis/analysis_unknown/results/gsm8k-train-disturbed_inference_0.7_0.75_5.json", "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)

    print(f"Results saved to /home/nfs02/laizj/experiment/uncertainty_analysis/analysis_unknown/results/gsm8k-train-disturbed_inference_0.7_0.75_5.json")

if __name__ == "__main__":
    main()
