#!/bin/bash
set -e

echo "Creating virtual environment..."
python -m venv venv

# Activation path depends on OS, assuming Git Bash on Windows for .sh scripts
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "Installing required packages..."
pip install --upgrade pip setuptools wheel
# Python 3.13 대응을 위해 최신 PyTorch 설치 (CU124는 안정적인 최신 버전)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 핵심 라이브러리 설치 (윈도우/3.13 호환성 확인된 최신 버전들)
pip install transformers datasets peft accelerate bitsandbytes sentencepiece protobuf huggingface_hub pandas tqdm pyyaml

echo "Registering HF_TOKEN..."
read -sp "Enter your Hugging Face Token (HF_TOKEN): " HF_TOKEN
echo "" # For new line
export HF_TOKEN=$HF_TOKEN
echo "HF_TOKEN=$HF_TOKEN" > .env

# Log in to Hugging Face
huggingface-cli login --token $HF_TOKEN

echo "Setup completed successfully."
