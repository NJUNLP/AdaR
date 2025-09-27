
import contextlib
from io import StringIO
import re
import subprocess
import json
from tqdm import tqdm
import os
from contextlib import contextmanager
import signal
import builtins
import threading
import yaml

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
    
from utils import set_seed
set_seed(cfg["process"]["seed"])

input_path = os.path.join(cfg["process"]["tmp_folder"], "template_and_code_generation", f"{cfg['data']['dataset_name']}_generated.jsonl")
output_path = os.path.join(cfg["process"]["tmp_folder"], "template_and_code_generation", f"{cfg['data']['dataset_name']}_parsed.jsonl")

success = 0
template_generation_fault_count = 0 # 无法生成模板
template_generation_mistake_count = 0 # 生成的模板与原始问题不匹配
python_generation_fault_count = 0 # 无法生成 Python 代码
python_run_fault_count = 0 # 无法运行的 Python 代码
python_run_mistake_count = 0 # Python 代码运行结果错误
template_python_not_algined_count = 0 # 模板中的变量与 Python 代码中的变量不对齐

# avoid input() interrupt the program
def mock_input(prompt=""):
    return "" 
builtins.input = mock_input

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise Exception("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def runcode(code):
    """执行 Python 代码并返回输出"""
    output = ""
    try:
        with time_limit(10):
            # 捕获标准输出
            with StringIO() as buf, contextlib.redirect_stdout(buf):
                exec(code, {}, {})
                output = buf.getvalue().strip()
    except Exception as e:
        return None
    
    try:
        return float(output)
    except ValueError:
        if re.search(r'\d', output):
             output = None
        else:
            return output

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"\\boxed\{(\d+(\.\d+)?)", text)  # 匹配 \\boxed
    if len(res) == 0:
        res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 114.514

c = 0

def extract_content(item):
    
    global success, python_run_fault_count, python_run_mistake_count, python_generation_fault_count, template_generation_fault_count, template_generation_mistake_count, template_python_not_algined_count
    
    generation = item["generated_texts"][0]
    # print(generation)
    # 提取第一次出现的### template后的内容
    template_match = re.search(r'### (?:Query|Query Template|Template):(.*?)(?=###|$)', generation, re.DOTALL | re.IGNORECASE)
    template_content = template_match.group(1).strip() if template_match else None
    
    # 提取### python代码块中的代码
    python_code_match = re.search(r'### Python Code:\s*```(?:python)?\s*(.*?)\s*```', generation, re.DOTALL | re.IGNORECASE)
    python_code = python_code_match.group(1).strip() if python_code_match else None
    
    if python_code is None:
        python_generation_fault_count += 1
    elif template_content is None:
        template_generation_fault_count += 1 
    elif abs(template_content.count(' ') - item["query"].count(' ')) / item["query"].count(' ') > 1:
        template_generation_mistake_count += 1
    else:
        python_result = runcode(python_code)
        if python_result is None:
            global c
            python_run_fault_count += 1
        elif isinstance(python_result, str):
            item["answer"] = python_result
        elif abs(python_result - item["answer"]) > 1e-2:
            python_run_mistake_count += 1
        else:
            variables = re.findall(r'<([^>]+?)>', template_content)
            for var in variables:
                pattern = r'\b' + re.escape(var) + r'\s*?='
                if re.search(pattern, python_code) is None:
                    template_python_not_algined_count += 1
                    return False
            else:
                success += 1
                item["template"] = template_content
                item["python"] = python_code
            return True
        
    return False

from utils import get_line_count
line_count = get_line_count(input_path)
with open(input_path, "r") as in_f, open(output_path, "w") as out_f:
    for line in tqdm(in_f, total=line_count):
        item = json.loads(line)
        if extract_content(item):
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

print()
print("="* 40)
print("Results for template and code generation have been parsed!\n")
print(f"Success rate: {success} / {line_count} = {success / line_count}")
print(f"The error from template generation: {template_generation_fault_count}")
print(f"The error from code generation: {python_generation_fault_count}")
print(f"The error from a mismatch between the template and the query: {template_generation_mistake_count}")
print(f"The error from the runtime error: {python_run_fault_count}")
print(f"The error from code running with incorrect results: {python_run_mistake_count}")
print(f"The error from a mismatch between the query and the code: {template_python_not_algined_count}")
print("="* 40)