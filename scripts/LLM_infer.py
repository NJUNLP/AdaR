import json
import yaml
from transformers import AutoTokenizer
import torch
import os
import sys
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import openai

cfg = None
client = None

async def process_prompt_async(args):
    data_item, sampling_params = args
    
    while True:
        try:
            completion = await client.completions.create(
                model="default",
                prompt=data_item["prompt"],
                **sampling_params,
            )
            data_item['generated_texts'] = [gen.text for gen in completion.choices]
            return data_item
        except Exception as e:
            # pass
            print(f"Retry: {e}", file=sys.stderr)
            # await asyncio.sleep(1)


async def process_sync_in_async(data, sampling_params, output_path):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import BeamSearchParams
    
    if cfg["process"][sys.argv[1]]["generate_model_path"] == -1:
        model_path = cfg["process"][sys.argv[1]]["generate_tokenizer_path"]
    else:
        model_path = cfg["process"][sys.argv[1]]["generate_model_path"]
    
    # 配置tensor并行
    tensor_parallel_size = cfg["process"][sys.argv[1]]["tensor_parallel_size"]
    if tensor_parallel_size == -1:
        tensor_parallel_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(','))
    
    # 初始化LLM
    llm = LLM(
        model=model_path,
        dtype=torch.bfloat16,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=cfg["process"][sys.argv[1]]["gpu_memory_utilization"]
    )
    
    prompts = [item['prompt'] for item in data]
    outputs = llm.generate(prompts, SamplingParams(**sampling_params))
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            data[i]['generated_texts'] = [gen.text for gen in output.outputs]
            f.write(json.dumps(data[i], ensure_ascii=False) + "\n")
    
    print()
    print("="* 40)
    print("Results for template and code generation have been generated!")
    print(f"Total: {len(data)} instances")
    print(f"Output path: {output_path}")
    print(f'Deployed tokenizer: {os.path.basename(cfg["process"][sys.argv[1]]["generate_tokenizer_path"])}')
    print(f'Deployed model: {os.path.basename(model_path)}')
    print("="* 40)
    print()


async def main():
    global cfg
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
        
    from utils import set_seed
    set_seed(cfg["process"]["seed"])

    tokenizer_path = cfg["process"][sys.argv[1]]["generate_tokenizer_path"]
    if sys.argv[1] == "template_and_code_generation":
        input_path = os.path.join(cfg["process"]["tmp_folder"], sys.argv[1], f'{cfg["data"]["dataset_name"]}_prompt.jsonl')
        output_path = os.path.join(cfg["process"]["tmp_folder"], sys.argv[1], f'{cfg["data"]["dataset_name"]}_generated.jsonl')
    else:
        input_path = os.path.join(cfg["process"]["tmp_folder"], sys.argv[1], f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_prompt.jsonl')
        output_path = os.path.join(cfg["process"]["tmp_folder"], sys.argv[1], f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_generated.jsonl')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    is_parallel = cfg["process"][sys.argv[1]]["is_parallel"]
    
    
    data  = []

    with open(input_path, "r", encoding="utf-8") as f:   
        for line in f:
            data_item = json.loads(line)
            data.append(data_item)
    
    sampling_params = cfg["process"][sys.argv[1]]['sampling-params']
    sampling_params["stop"] =  [token for token in tokenizer.all_special_tokens if "end" in token]

    if is_parallel:
        tasks = []
        line_count = len(data)
        batch_size = cfg["process"][sys.argv[1]]["bsz"]
        
        global client
        client = openai.AsyncClient(base_url=cfg["process"][sys.argv[1]]["url"], api_key=cfg["process"][sys.argv[1]]["api_key"])
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(data, 1):
                tasks.append(process_prompt_async((item, sampling_params)))
                
                if i % batch_size == 0 or i == line_count:
                    results = await async_tqdm.gather(
                        *tasks, 
                        desc=f"[{i} / {line_count}]", 
                        total=len(tasks)
                    )
                    f.write('\n'.join([json.dumps(it, ensure_ascii=False) for it in results]) + '\n')
                    tasks = []
        
        print()
        print("="* 40)
        print(f"Results for {sys.argv[1]} have been generated!\n")
        print(f"Total: {len(data)} instances")
        print(f"Output path: {output_path}")
        print(f'Deployed tokenizer: {os.path.basename(cfg["process"][sys.argv[1]]["generate_tokenizer_path"])}')
        print(f'Deployed url: {cfg["process"][sys.argv[1]]["url"]}')
        print(f'Sampling Params: {json.dumps(sampling_params, ensure_ascii=False)}')
        print(f'Inference batch size: {cfg["process"][sys.argv[1]]["bsz"]}')
        print("="* 40)
        print()
                    
    else:
        await process_sync_in_async(data, sampling_params, output_path)

if __name__ == "__main__":
    asyncio.run(main())
