#!/bin/bash

cd scripts || exit

# 1 generate the prompt for template and code generation

python generate_prompt_for_template_code.py

# Generation
python LLM_infer.py template_and_code_generation

# 2 parse the result
python parse_generation_result.py

# 3 perturb the values and execute VA & EC
python perturb_variables.py

# 4 generate the prompt for EVS
python generate_prompt_for_cot_with_code.py

# # generation
python LLM_infer.py sanity_check

# # 5 execute EVS
python execute_EVS.py

# # 6 generate prompt for paraphrase
python generate_prompt_for_paraphrase.py

# # generation
python LLM_infer.py query_paraphrase

# # 7 merge paraphrase result
python merge_result.py