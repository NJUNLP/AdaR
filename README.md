# MAKE MATHEMATICAL REASONING ADAPTIVE

Mathematical reasoning is a primary indicator of large language models (LLMs) intelligence. However, existing LLMs exhibit failures of robustness and generalization. This paper attributes these deficiencies to spurious reasoning—i.e., producing answers from superficial features. To address this challenge, we propose the AdaR framework to enable adaptive reasoning, wherein models adapt to varying variable values when the corresponding problem-solving logic is unchanged. AdaR synthesizes data logically equivalent problems by varying variable values and trains models with RLVR on these data to penalize spurious logic while encouraging adaptive logic. To improve data quality, we obtain the corresponding answer by code execution and then apply sanity check. Experimental results demonstrate that AdaR  improves robustness and generalization, achieving substantial improvement in mathematical reasoning while maintaining high data efficiency. Analysis indicates that data synthesis and RLVR function in a coordinated manner to enable adaptive reasoning in LLMs. Subsequent analyses derive key design insights into the effect of critical factors and the applicability to instruct LLMs.

📄 Paper: [link to paper if available]  
💻 Code: [https://github.com/LaiZhejian/AdaR](https://github.com/LaiZhejian/AdaR)

---

## Features

- **Adaptive reasoning training** with synthesized logically equivalent problems.  
- **RLVR objective** to penalize spurious reasoning and encourage adaptive reasoning.  
- **Automatic answer generation and sanity checking** for high-quality synthetic datasets.  

---

## Installation

Clone this repository and set up the environment:

```bash
git clone https://github.com/LaiZhejian/AdaR.git
cd AdaR
uv sync
```

---

## Data Preparation

AdaR expects data in **JSONL format**, where each line is a dictionary with the following keys:

```json
{
  "query": "The math problem statement",
  "chosen": "The chain-of-thought reasoning",
  "answer": "The gold standard answer"
}
```

- `query`: Input problem  
- `chosen`: Chain-of-thought (CoT) reasoning  
- `answer`: Ground truth answer

---

## Configuration

Modify **`scripts/config.yaml`** to set parameters.  

Key options include:

- `is_parallel`:  
  - `true` → Enable parallel querying (API-based).  
  - `false` → Sequential querying (local).  

Other hyperparameters can also be adjusted depending on the dataset and model.

---

## Running Data Synthesis

Once your dataset and configuration are ready, run:

```bash
bash synthesis.sh
```

This script will launch the AdaR data synthesis and training pipeline.
