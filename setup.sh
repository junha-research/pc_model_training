#!/bin/bash
set -e

echo "=== [1/3] 가상 환경 생성 중 ==="
if command -v python3 &>/dev/null; then
    PYTHON_EXE=python3
else
    PYTHON_EXE=python
fi

$PYTHON_EXE -m venv venv

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "=== [2/3] 필수 패키지 설치 중 (Python 3.13 호환) ==="
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft accelerate bitsandbytes sentencepiece protobuf huggingface_hub pandas tqdm pyyaml

echo "=== [3/3] Hugging Face 인증 (선택 사항) ==="
echo "Llama 3.1, Gemma 2 등 승인제 모델을 사용하려면 토큰이 필요합니다."
echo "Qwen, Solar 모델만 사용할 경우 입력하지 않고 엔터(Enter)를 누르세요."
read -sp "HF_TOKEN (Optional): " HF_TOKEN
echo ""

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN=$HF_TOKEN
    echo "HF_TOKEN=$HF_TOKEN" > .env
    huggingface-cli login --token $HF_TOKEN
    echo "✅ 토큰 인증이 완료되었습니다."
else
    echo "ℹ️ 토큰 입력 없이 진행합니다. (공개 모델만 사용 가능)"
fi

echo "✅ 모든 설정이 완료되었습니다. 이제 'bash run.sh'를 실행하세요."
