

import json
import random
import yaml
import os
import sys


with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
    
from utils import set_seed
set_seed(cfg["process"]["seed"])

dataset_name=cfg["data"]["dataset_name"]
input_path = cfg["data"]["source_path"]
output_dir = os.path.join(cfg["process"]["tmp_folder"], "template_and_code_generation")
output_path =  os.path.join(output_dir, f'{cfg["data"]["dataset_name"]}_prompt.jsonl')

os.makedirs(output_dir, exist_ok=True)


instruction = r"""Task Description:
You are given a natural language query and its chain-of-thought response. Your task is to:
Generate a Query Template by abstracting specific values into variables.
Generate Python Code that executes the logic described in the COT response using the abstracted variables.

Input Format:
Query: Original query with specific values
Response: Chain-of-thought reasoning that leads to the answer

Output Requirements:
Query Template:
Replace only concrete values in the query with angle-bracketed placeholders like <variable_name>.
Do not replace names or general nouns (e.g., do not change “Jungkook” to <person_name>).
Preserve the original wording and structure of the query as much as possible.
Python Code:
Begin by defining variables that correspond to the placeholders in the template.
Translate the logic in the response into executable Python code.
The code should end with a print() statement that prints only the final result.
Do not include comments with explanations or reasoning.
Use the same variable names as in the template for consistency.

=== START EXAMPLE ===
### Query:
Find A that satisfies 32×A×A×A=42592

### Response:
To find the value of A that satisfies the equation 32×A×A×A=42592, we can rewrite the equation as:
\(32A^3 = 42592\)
Now, we need to isolate A by dividing both sides of the equation by 32:
\(A^3 = \frac{42592}{32}\)
\(A^3 = 1331\)
Now, we take the cube root of both sides to solve for A:
\(A = \sqrt[3]{1331}\)
\(A = 11\)

### Template:
Find A that satisfies <coefficient>×A×A×A=<result>

### Python Code:
```python
# Variable definitions
coefficient = 32
result = 42592

# Calculation
A_cubed = result / coefficient
A = A_cubed ** (1/3)

# Output
print(A)
```
=== END EXAMPLE ===
"""

prompt = """
Instruction:
### Query:
{}

### Response:
{}

"""


with open(input_path, "r") as f:
    data = json.load(f)

random.shuffle(data)

with open(output_path, "w") as f:
    for idx, item in enumerate(data):
        
        messages = [
            {"role": "user", "content": instruction + prompt.format(item["query"], item["chosen"])},
        ]
        if cfg["process"]["template_and_code_generation"]["system"]:
            messages.insert(0, {"role": "system", "content": cfg["process"]["template_and_code_generation"]["system"]})
        
        result = {
            "id": idx,
            "query": item["query"],
            # "prompt": tokenizer.apply_chat_template(
            #             messages,
            #             tokenize=False,
            #             add_generation_prompt=True),
            "messages": messages,
            "chosen": item["chosen"],
            "answer": item["answer"]
        }

        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
print()
print("="* 40)
print("Prompts for template and code generation have been created!\n")
print(f"Total: {len(data)} instances")
print(f"Output path: {output_path}")
print("="* 40)
print()