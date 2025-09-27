import random
import numpy as np
import torch
import os
import subprocess

def set_seed(seed):
    # Python随机库
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)
    # PyTorch GPU（单卡）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensure the determinism of CUDA convolution operations (which may affect performance, optional)
    # torch.backends.cudnn.deterministic = True
    # Disable automatic optimization (to ensure determinism)
    # torch.backends.cudnn.benchmark = False  
    # Environment variable (Some dependencies may read this variable)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
def get_line_count(path):
    result = subprocess.run(['wc', '-l', path], stdout=subprocess.PIPE, text=True)
    return int(result.stdout.split()[0])