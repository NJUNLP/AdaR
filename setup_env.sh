#!/bin/bash

ENV_NAME="YOUR_ENV_NAME"
ENV_PATH="YOUR_ENV_PATH/$ENV_NAME"

echo "Creating Conda environment..."
conda create --prefix $ENV_PATH python=3.10 -y

echo "Activating Conda environment..."
conda activate $ENV_PATH

echo "Installing PyTorch and related libraries..."
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing pip dependencies..."
pip install scipy==1.15.2 \
            pillow==10.4.0 \
            accelerate==1.0.1 \
            datasets==3.1.0 \
            peft==0.12.0 \
            transformers==4.45.2 \
            trl==0.9.6 \
            tensorboardX==2.6.2.2 \
            deepspeed==0.15.4 \
            vllm==0.6.4.post1 \
            matplotlib \
            scikit-learn

echo "Installing additional Conda dependencies..."
conda install -c conda-forge nccl -y

echo "Environment setup complete!"
echo "To activate the environment, run:"
echo "conda activate $ENV_PATH"