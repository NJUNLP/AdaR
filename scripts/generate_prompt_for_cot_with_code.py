import json
from tqdm import tqdm
import os
import yaml

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
    
from utils import set_seed
set_seed(cfg["process"]["seed"])

input_path = os.path.join(cfg["process"]["tmp_folder"], "controllable_generation", f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}.jsonl')
output_dir = os.path.join(cfg["process"]["tmp_folder"], "sanity_check")
output_path =  os.path.join(output_dir, f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_prompt.jsonl')

os.makedirs(output_dir, exist_ok=True)


prompt = r"""
Your task is to provide a clear chain-of-thought (COT) explanation that answers the user's question. A Python script may be provided as part of the input, but it is not mandatory to follow it closely. If the provided code doesn't align with the real-world scenario or if the values and logic in the code are incorrect or irrelevant to the problem, feel free to disregard the script. Instead, focus on reasoning through the problem using your own judgment and logic.
Interpret the question clearly and begin by understanding the problem. If the Python script can offer guidance, you may refer to it, but it's not a requirement. If the provided code does not match the context or contains errors, you are free to work through the solution from scratch without referring to it.
Explicitly state the final answer after completing your reasoning, enclosed in \boxed{}.
"""

instruction = """
### Query:
{}

### Python Code:
{}

### Resonse:
"""


from utils import get_line_count
line_count = get_line_count(input_path)
with open(input_path, "r") as in_f, open(output_path, "w") as out_f:
    for line in tqdm(in_f, total=line_count):

        item = json.loads(line)
        for inner_idx, inner_item in enumerate(item["perturbed"]):
            messages = [
                {"role": "user", "content": prompt + instruction.format(inner_item["new_query"], inner_item["new_code"])}
            ]
            if cfg["process"]["template_and_code_generation"]["system"]:
                messages.insert(0, {"role": "system", "content": cfg["process"]["template_and_code_generation"]["system"]})
            
            out_f.write(json.dumps({
                "id": item["id"],
                "inner_id": inner_idx,
                # "prompt": tokenizer.apply_chat_template(
                #             messages,
                #             tokenize=False,
                #             add_generation_prompt=True,
                #             enable_thinking=False),
                "messages": messages,
                "max_fluct": inner_item["max_fluct"],
                "instruction": inner_item["new_query"],
                "code": inner_item["new_code"],
                "system": "Please reason step by step, and put your final answer within \\boxed{}.",
                "input": "",
                "history": [],
                "answer": inner_item["new_ans"],
                }, ensure_ascii=False) + "\n")

print()
print("="* 40)
print("Prompts for EVS have been created!\n")
print(f"Total: {line_count} instances")
print(f"Output path: {output_path}")
print("="* 40)
print()