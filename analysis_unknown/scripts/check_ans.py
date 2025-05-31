
import re
import json


result_path = "/home/nfs02/laizj/experiment/uncertainty_analysis/analysis_unknown/results/qwen7b_instruct-inference-greedy-qwen72b_instruct-generate-greedy-template_and_code-gsm8k_train_disturbed_2.0.json"
output_path = "/home/nfs02/laizj/experiment/uncertainty_analysis/analysis_unknown/ids.txt"


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

if __name__ == "__main__":
    with open(result_path, "r") as f:
        data = json.load(f)
    with open(output_path, "r") as f:
        ids = eval(f.read())
        
    correct_ids = []
    half_ids = []
    wrong_ids = []
    
    for idx, line in enumerate(data):
        if line["id"] not in ids: continue
        answer = extract_last_num(str(line["new_ans"]))
        correct_num = 0
        for gen in line["generated_texts"]:
            generated_num = extract_last_num(gen)
            if abs(generated_num - answer) < 1e-2:
                correct_num += 1
        if correct_num == 0:
            wrong_ids.append(line["id"])
        elif correct_num == len(line["generated_texts"]):
            correct_ids.append(line["id"])
        else:
            half_ids.append(line["id"])
    
    print("Correct:", len(correct_ids))
    print("Half:", len(half_ids))
    print("Wrong:", len(wrong_ids))
    print(f"成功率: {len(correct_ids) / len(ids)}")
    print(wrong_ids[:20])        
    # with open(output_path, "w") as f:
    #     f.write(str(correct_ids))