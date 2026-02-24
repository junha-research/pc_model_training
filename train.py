import os
import torch
import json
import yaml
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def train(config_path, model_override=None):
    # 1. 설정 로드
    config = load_config(config_path)
    
    # 모델 선택 (CLI 인자가 있으면 우선순위, 없으면 YAML 설정 사용)
    model_key = model_override if model_override else config['model']['selected_model']
    model_id = config['model']['model_map'].get(model_key)
    
    if not model_id:
        print(f"Error: 지원하지 않는 모델입니다: {model_key}")
        return

    print(f"--- 학습 시작: {model_key} ({model_id}) ---")

    # 2. 양자화 및 모델 로드
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # 3. LoRA 설정
    lora_cfg = config['lora']
    peft_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        target_modules=lora_cfg['target_modules'],
        lora_dropout=lora_cfg['dropout'],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # 4. 데이터셋 로드
    print(f"Loading dataset: {config['dataset']['repo_id']}")
    raw_dataset = load_dataset(config['dataset']['repo_id'])

    def preprocess_function(examples):
        inputs = []
        for i in range(len(examples['instruction'])):
            output_data = {
                "scores": examples['output'][i],
                "feedback": examples.get('feedback', [None]*len(examples['instruction']))[i],
                "reasoning": examples.get('organization_reasoning', [None]*len(examples['instruction']))[i]
            }
            output_str = json.dumps(output_data, ensure_ascii=False)
            prompt = f"### 질문: {examples['question'][i]}\n\n### 지시사항: {examples['instruction'][i]}\n\n### 에세이:\n{examples['input'][i]}\n\n### 채점 결과:\n{output_str}"
            inputs.append(prompt)
        return tokenizer(inputs, truncation=True, padding="max_length", max_length=config['dataset']['max_length'])

    print("Tokenizing data...")
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, remove_columns=raw_dataset['train'].column_names)

    # 5. 학습 설정
    t_cfg = config['training']
    output_dir = f"{config['output']['dir_prefix']}{model_key}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=t_cfg['batch_size'],
        gradient_accumulation_steps=t_cfg['gradient_accumulation_steps'],
        learning_rate=float(t_cfg['learning_rate']),
        num_train_epochs=t_cfg['num_epochs'],
        logging_steps=t_cfg['logging_steps'],
        evaluation_strategy="steps",
        eval_steps=t_cfg['eval_steps'],
        save_strategy="steps",
        save_steps=t_cfg['save_steps'],
        fp16=t_cfg['fp16'],
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final_adapter")

    # 6. 테스트 결과 CSV 저장
    if config['output']['save_csv']:
        print("Generating test results CSV...")
        model.eval()
        test_data = raw_dataset['test']
        results = []

        for i in tqdm(range(len(test_data))):
            prompt = f"### 질문: {test_data[i]['question']}\n\n### 지시사항: {test_data[i]['instruction']}\n\n### 에세이:\n{test_data[i]['input']}\n\n### 채점 결과:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=config['output']['max_new_tokens'],
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            decoded = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results.append({
                "question": test_data[i]['question'],
                "ground_truth": json.dumps({"scores": test_data[i]['output'], "feedback": test_data[i].get('feedback', "")}, ensure_ascii=False),
                "predicted": decoded.strip()
            })

        df = pd.DataFrame(results)
        df.to_csv(f"test_results_{model_key}.csv", index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--model", type=str, help="Override selected model in config")
    args = parser.parse_args()
    train(args.config, args.model)
