#!/bin/bash
set -e

# 1. 가상 환경 활성화 (Linux 우선 처리)
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv/Scripts/activate ]; then
    source venv/Scripts/activate
else
    echo "Error: 가상 환경을 찾을 수 없습니다. 'bash setup.sh'를 먼저 실행해 주세요."
    exit 1
fi

# 2. 환경 변수 로드 (HF_TOKEN 등)
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# 3. 파일럿 테스트 (환경 검증)
echo "--- 시스템 호환성 파일럿 테스트를 수행하시겠습니까? (추천) [y/n]"
read -p "입력: " RUN_PILOT
if [ "$RUN_PILOT" == "y" ]; then
    # python 명령어는 venv 내부의 python을 따름
    python pilot_test.py
fi

# 4. 학습 모드 선택
echo ""
echo "--- 학습 모드를 선택하세요 ---"
echo "1. [Single] config.yaml에 설정된 기본 모델 학습"
echo "2. [All] 모든 지원 모델을 순차적으로 자동 학습 (Qwen, Llama, Solar, Gemma)"
read -p "모드 선택 (1 또는 2): " MODE

if [ "$MODE" == "2" ]; then
    echo "🚀 모든 모델 학습 공정을 순차적으로 시작합니다..."
    python train_all.py
else
    # YAML 설정 혹은 CLI 오버라이드 (여기서는 기본값 사용)
    echo "🚀 기본 모델 학습을 시작합니다..."
    python train.py
fi

echo "✅ 모든 학습 과정이 정상 종료되었습니다."
