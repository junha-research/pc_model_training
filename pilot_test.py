import os
import torch
import json
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

def run_pilot():
    print("=== [PILOT TEST] 실제 데이터 기반 호환성 검증 시작 ===")
    
    # 1. 실제 데이터 로드 (Public 저장소)
    dataset_repo_id = "SJunha/paper-clinic"
    print(f"[1/5] 데이터셋 로드 중: {dataset_repo_id}")
    try:
        raw_dataset = load_dataset(dataset_repo_id)
        # 테스트를 위해 4개씩만 추출
        train_small = raw_dataset['train'].select(range(min(4, len(raw_dataset['train']))))
        test_small = raw_dataset['test'].select(range(min(4, len(raw_dataset['test']))))
        print(f"✓ 실제 데이터 {len(train_small)}개 로드 성공")
    except Exception as e:
        print(f"Error: 데이터셋 로드 실패: {e}")
        return

    # 2. 초경량 모델 로드
    model_id = "sshleifer/tiny-gpt2"
    print(f"[2/5] 초경량 모델 로드 중: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    print("✓ 모델 로드 완료")

    # 3. LoRA(PEFT) 적용
    print("[3/5] LoRA 적용 중...")
    config = LoraConfig(
        r=8, lora_alpha=16, 
        target_modules=["c_attn"], 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    print("✓ LoRA 적용 완료")

    # 4. 전처리 및 1단계 학습
    print("[4/5] 학습 루프 테스트 중...")
    def preprocess_function(examples):
        inputs = []
        for i in range(len(examples['instruction'])):
            output_data = {
                "scores": examples['output'][i],
                "feedback": examples.get('feedback', [""]*len(examples['instruction']))[i],
                "reasoning": examples.get('organization_reasoning', [""]*len(examples['instruction']))[i]
            }
            output_str = json.dumps(output_data, ensure_ascii=False)
            prompt = f"### 질문: {examples['question'][i]}\n\n### 지시사항: {examples['instruction'][i]}\n\n### 에세이:\n{examples['input'][i]}\n\n### 채점 결과:\n{output_str}"
            inputs.append(prompt)
        return tokenizer(inputs, truncation=True, padding="max_length", max_length=128)

    tokenized_train = train_small.map(preprocess_function, batched=True)

    args = TrainingArguments(
        output_dir="./pilot_out",
        max_steps=1,
        per_device_train_batch_size=1,
        logging_steps=1,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    print("✓ 학습 파이프라인 무결성 확인")

    # 5. 추론 및 CSV 저장
    print("[5/5] 결과 저장 테스트 중...")
    results = [{"ground_truth": "test", "predicted": "test_pred"}]
    pd.DataFrame(results).to_csv("pilot_test_results.csv", index=False, encoding='utf-8-sig')
    print("✓ CSV 저장 완료")

    print("\n=== [SUCCESS] 모든 시스템 및 라이브러리 호환성이 확인되었습니다! ===")

if __name__ == "__main__":
    run_pilot()
