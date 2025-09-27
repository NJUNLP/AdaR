import json
import re
from tqdm import tqdm
from collections import defaultdict
import yaml
import os

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
    
from utils import set_seed
set_seed(cfg["process"]["seed"])

input_path = os.path.join(cfg["process"]["tmp_folder"], "sanity_check", f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_generated.jsonl')
output_path = os.path.join(cfg["process"]["tmp_folder"], "sanity_check", f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_filtered.jsonl')

record = defaultdict(list)
id_set = set()

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"\\boxed\{(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) == 0:
        res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 114.514

from utils import get_line_count
line_count = get_line_count(input_path)
with open(input_path, 'r') as in_f, open(output_path, 'w') as out_f:
    for line in tqdm(in_f, total=line_count):
        item = json.loads(line)
        id = item["id"]
        max_fluct = item["max_fluct"]
        pass_EVS = False
        for generated_text in item["generated_texts"]:
            if abs(extract_last_num(generated_text) - item['answer']) < 1e-3:
                temp_item = {
                    "id": id,
                    "max_fluct": max_fluct,
                    "instruction": item["instruction"],
                    "code": item["code"],
                    "output": generated_text,
                    "system": cfg["data"]["system"],
                    "input": "",
                    "history": [],
                    "answer": item['answer'],
                }
                out_f.write(json.dumps(temp_item, ensure_ascii=False) + "\n")
                # any result match is OK
                pass_EVS = True
                break
        id_set.add(id)
        if pass_EVS:
            record[id].append(item["inner_id"])

successful_perturbation_count = sum(len(item) for item in record.values())

print()
print("="* 40)
print(f"EVS have been executed!\n")
print(f"The success rate of perturbation：{len(record)} / {len(id_set)} = {len(record) / len(id_set) * 100:.2f}%")
print(f"The success rate of each perturbation：{successful_perturbation_count} / {line_count} = {successful_perturbation_count / line_count * 100:.2f}%")
print(f"The average number of successful perturbations for each instance: {successful_perturbation_count / len(id_set):.2f}")
print("="* 40)
print()