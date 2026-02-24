#!/bin/bash
set -e

# 1. 가상 환경 활성화
echo "가상 환경 활성화 중..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# 2. 파일럿 테스트 (환경 검증)
echo "--- 시스템 호환성 파일럿 테스트를 수행하시겠습니까? (추천) [y/n]"
read RUN_PILOT
if [ "$RUN_PILOT" == "y" ]; then
    python pilot_test.py
fi

# 3. 학습 모드 선택
echo ""
echo "--- 학습 모드를 선택하세요 ---"
echo "1. 특정 모델 1개만 학습 (qwen-7b, qwen-14b, llama-3.1, solar, gemma-2 중 선택)"
echo "2. 모든 모델 순차 자동 학습 (순서: Qwen -> Llama -> Solar -> Gemma)"
read MODE

if [ "$MODE" == "2" ]; then
    echo "모든 모델 학습 공정을 시작합니다..."
    python train_all.py
else
    echo "학습할 모델의 이름을 입력하세요 [qwen-7b, qwen-14b, llama-3.1, solar, gemma-2]:"
    read MODEL_NAME
    echo "[$MODEL_NAME] 학습을 시작합니다..."
    python train.py --model $MODEL_NAME
fi

echo "전체 과정이 종료되었습니다."
