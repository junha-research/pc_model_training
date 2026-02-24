# 서술형 채점 모델 학습 파이프라인 (Automated Scoring & Feedback Training)

이 프로젝트는 다양한 LLM(Qwen 2.5, Llama 3.1, Solar, Gemma 2)을 활용하여 서술형 에세이를 채점하고 정성적인 피드백을 생성하는 모델을 학습시키는 자동화된 파이프라인을 제공합니다.

## 🚀 주요 특징
- **다양한 모델 지원**: Qwen 2.5 (7B/14B), Llama 3.1 (8B), Solar 10.7B, Gemma 2 (9B) 지원
- **자동화된 파이프라인**: 환경 설정부터 다수 모델 순차 학습까지 한 번의 명령어로 수행
- **효율적인 학습**: QLoRA(4-bit 양자화)를 적용하여 VRAM 사용량 최소화
- **정교한 데이터 처리**: 점수뿐만 아니라 채점 근거(Reasoning)와 피드백(Feedback)을 함께 학습
- **결과 분석**: 학습 완료 후 테스트 데이터에 대한 추론 결과를 CSV 파일로 자동 저장

## 🛠️ 설치 방법 (Windows/Linux 공통)

Python 3.13 환경에서 최적화되어 있습니다. 다음 명령어를 실행하여 가상 환경 구축 및 필수 라이브러리를 설치하세요.

```bash
bash setup.sh
```

## 📂 설정 (config.yaml)
학습 환경을 수정하려면 `config.yaml` 파일을 편집하세요. 코드를 수정할 필요 없이 다음 항목들을 제어할 수 있습니다:
- `model.selected_model`: 기본으로 학습할 모델 선택
- `training`: 학습률, 에포크, 배치 사이즈 등 하이퍼파라미터
- `lora`: LoRA Rank(r), Alpha 값 조정

## 🏃 실행 방법

### 1. 전체 자동화 실행
가장 권장되는 방식입니다. 실행 후 메뉴에서 학습 모드를 선택할 수 있습니다.

```bash
bash run.sh
```
- **Pilot Test**: 실제 학습 전 시스템 호환성을 1분 내에 검증합니다.
- **Single Model**: 특정 모델 하나만 선택하여 학습합니다.
- **All Models**: `config.yaml`에 정의된 모든 모델을 순차적으로 자동 학습합니다.

### 2. 개별 스크립트 실행
- **환경 검증 전용**: `python pilot_test.py`
- **특정 모델 학습**: `python train.py --model qwen-7b`
- **모든 모델 일괄 학습**: `python train_all.py`

## 📊 결과 확인
학습이 완료되면 다음과 같은 결과물이 생성됩니다.
1. **모델 어댑터**: `scoring_model_{model_name}/final_adapter` 폴더에 저장됩니다.
2. **평가 결과**: `test_results_{model_name}.csv` 파일로 저장되며, [질문, 실제 점수, 모델 예측 점수/피드백]을 포함합니다.

## ⚠️ 주의사항
- **GPU 메모리**: Qwen-14B 등 큰 모델을 학습할 때는 VRAM이 최소 16GB~24GB 이상 필요할 수 있습니다. 메모리가 부족할 경우 `config.yaml`에서 `batch_size`를 1로 유지하고 `gradient_accumulation_steps`를 조절하세요.
- **데이터셋**: 기본적으로 `SJunha/paper-clinic` (Hugging Face Public) 데이터셋을 사용합니다.

## 📄 라이선스
본 프로젝트의 코드는 자유롭게 수정 및 배포가 가능합니다. 단, 사용된 각 베이스 모델(Llama, Qwen 등)의 라이선스 규정을 준수해야 합니다.
