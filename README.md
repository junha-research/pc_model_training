# 서술형 채점 모델 자동화 학습 파이프라인 (LLM Automated Scoring)

이 프로젝트는 Qwen 2.5, Llama 3.1, Solar, Gemma 2 등 최신 LLM을 활용하여 서술형 에세이 채점 및 피드백 생성 모델을 구축하는 자동화 파이프라인입니다.

## 📋 사전 준비 사항 (Prerequisites)

### 1. 모델 접근 권한 승인 (Gated Models)
본 프로젝트에서 사용하는 **Llama 3.1** 및 **Gemma 2** 모델은 Hugging Face에서 사용 승인이 필요합니다.
- [Meta Llama 3.1 페이지](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) 및 [Google Gemma 2 페이지](https://huggingface.co/google/gemma-2-9b-it)에 접속하여 사용 요청(Request Access)을 완료해 주세요.
- 승인된 계정의 **Hugging Face Write Token**이 필요합니다.

### 2. 하드웨어 요구 사항
- **GPU**: NVIDIA GPU (최소 VRAM 16GB 권장, 14B 모델의 경우 24GB 이상 권장)
- **Driver**: CUDA 12.1 ~ 12.4 지원 드라이버

---

## 🚀 실행 순서 (Step-by-Step)

서버에 접속한 후 반드시 다음 순서대로 명령어를 실행해 주세요.

### **Step 1: 실행 권한 부여**
가장 먼저 스크립트 파일들이 실행될 수 있도록 권한을 설정합니다.
```bash
chmod +x setup.sh run.sh
```

### **Step 2: 필수 환경 구축**
가상 환경을 생성하고 필수 라이브러리를 설치합니다.
- **Qwen, Solar 모델만 사용할 경우**: 토큰 입력 없이 엔터를 눌러 넘어가도 됩니다.
- **Llama 3.1, Gemma 2 모델을 사용할 경우**: 해당 모델 페이지에서 승인받은 계정의 토큰을 입력해야 합니다.
```bash
bash setup.sh
```

### **Step 3: 학습 파이프라인 가동**
환경 구축이 끝났다면 통합 스크립트를 통해 검증 및 학습을 시작합니다.
```bash
bash run.sh
```
- **Pilot Test**: 시스템 및 라이브러리 호환성을 1분 내에 체크합니다. (추천)
- **Mode 1**: `config.yaml`에 설정된 단일 모델 학습
- **Mode 2**: 전체 모델(Qwen, Llama, Solar, Gemma) 순차 자동 학습

---

## ⚙️ 설정 관리 (config.yaml)
학습 파라미터나 모델 구성을 바꾸고 싶을 때 `config.yaml` 파일을 편집하세요.
- `model.selected_model`: 단일 학습 시 사용할 기본 모델
- `training.num_epochs`: 학습 횟수 (기본값: 3)
- `training.batch_size`: 메모리 부족 시 1로 유지 권장

---

## 📂 결과물 확인 (Output)

### 1. 채점 및 피드백 결과 (CSV)
학습 종료 후 테스트 데이터에 대한 추론 결과가 `test_results_{model_name}.csv` 파일로 저장됩니다.
- **포함 내용**: 질문, 학생 에세이, 실제 정답(JSON), 모델 예측값(JSON: 점수+피드백)

### 2. 학습된 모델 가중치 (LoRA Adapter)
최종 모델 어댑터는 `scoring_model_{model_name}/final_adapter` 폴더에 로컬로 저장됩니다. 향후 다른 추론 서버에서 이 어댑터만 로드하여 사용할 수 있습니다.

---

## 🛠️ 문제 해결 (Troubleshooting)
- **OOM (Out of Memory)**: 학습 중 메모리 부족 에러가 발생하면 `config.yaml`에서 `batch_size`를 줄이거나 `gradient_accumulation_steps`를 늘리세요.
- **토큰 에러**: 모델 다운로드 시 401/403 에러가 발생하면 `setup.sh`에서 입력한 토큰이 승인된 계정의 것인지 확인하세요.

---

## 📄 라이선스
본 프로젝트의 소스코드는 MIT License를 따르며, 각 모델(Llama, Qwen 등)의 가중치는 해당 제조사의 커뮤니티 라이선스를 따릅니다.
