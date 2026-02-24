import json
import os
import argparse
from datasets import Dataset, DatasetDict
from huggingface_hub import login

def upload_to_hf():
    # 1. 파일 경로 설정
    input_path = "data/paperclinic_generated_dataset_gemini.json"
    
    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' 파일을 찾을 수 없습니다. 'data' 폴더에 해당 파일이 있는지 확인하세요.")
        return

    # 2. Hugging Face 로그인 확인
    token = os.environ.get("HF_TOKEN")
    if token:
        print("환경 변수에서 HF_TOKEN을 감지했습니다. 로그인을 시도합니다...")
        login(token=token)
    else:
        print("HF_TOKEN 환경 변수가 설정되지 않았습니다.")
        token = input("Hugging Face Token (HF_TOKEN)을 직접 입력하세요: ").strip()
        if token:
            login(token=token)
        else:
            print("Error: 토큰이 입력되지 않았습니다. 업로드를 중단합니다.")
            return

    # 3. 데이터 로드
    print(f"데이터 로딩 중: {input_path}...")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            full_json = json.load(f)
    except Exception as e:
        print(f"Error: JSON 파일을 읽는 도중 오류가 발생했습니다: {e}")
        return
    
    data_list = full_json.get('data', [])
    if not data_list:
        print("Error: JSON 파일 내에 'data' 키가 없거나 비어 있습니다.")
        return

    # 4. 데이터셋 변환 및 8:2 분할
    print(f"총 {len(data_list)}개의 샘플을 변환 중입니다...")
    full_dataset = Dataset.from_list(data_list)
    
    # 무작위 분할 (seed 고정으로 재현성 확보)
    ds_split = full_dataset.train_test_split(test_size=0.2, seed=42)
    
    ds_dict = DatasetDict({
        'train': ds_split['train'],
        'test': ds_split['test']
    })

    # 5. 저장소 정보 입력 및 업로드
    print("---Hugging Face 업로드 설정 ---")
    repo_id = input("업로드할 Dataset Repository ID를 입력하세요 (예: username/dataset-name): ").strip()
    
    if not repo_id:
        print("Error: Repository ID가 입력되지 않았습니다.")
        return

    print(f"업로드 시작: https://huggingface.co/datasets/{repo_id} (Private 설정 적용)")
    try:
        ds_dict.push_to_hub(repo_id, private=True)
        
        # 6. 향후 학습 스크립트(train.py)에서 참조할 수 있도록 repo_id 저장
        with open(".repo_id", "w") as f:
            f.write(repo_id)
            
        print("[성공] 업로드가 완료되었습니다!")
        print(f"데이터셋 페이지: https://huggingface.co/datasets/{repo_id}")
        print("이제 'bash run.sh'를 실행하여 모델 학습을 시작할 수 있습니다.")
        
    except Exception as e:
        print(f"Error: 업로드 도중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    upload_to_hf()
