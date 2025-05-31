import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import torch

def main(base_model_path, lora_model_path, save_model_path):
    # 创建保存模型的目录
    os.makedirs(save_model_path, exist_ok=True)

    # 加载原始模型和Tokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 加载LoRA模型
    model = PeftModel.from_pretrained(model, lora_model_path)

    # 合并LoRA权重到原始模型中
    model = model.merge_and_unload()

    # 保存合并后的模型和Tokenizer
    model.save_pretrained(save_model_path, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(save_model_path)

    print(f"模型已成功合并并保存到 {save_model_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="合并LoRA模型到原始模型")
    # parser.add_argument("--base_model_path", type=str, required=True, help="原始模型的路径")
    # parser.add_argument("--lora_model_path", type=str, required=True, help="LoRA模型的路径")
    # parser.add_argument("--save_model_path", type=str, required=True, help="保存合并后模型的路径")

    # args = parser.parse_args()

    # main(args.base_model_path, args.lora_model_path, args.save_model_path)
    main(
        base_model_path="/home/nfs05/laizj/model/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e/",
        lora_model_path="/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/model/qwen7b-orca_10k_train-repeat-1-1e-6-32-no_shuffle",
        save_model_path="/home/nfs02/laizj/experiment/uncertainty_analysis/math_training/model/qwen7b-orca_10k_train-repeat-1-1e-6-32-no_shuffle"
    )