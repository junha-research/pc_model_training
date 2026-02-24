import yaml
import subprocess
import sys

def train_all():
    # 1. YAML에서 모델 리스트 로드
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        models = list(config['model']['model_map'].keys())
    except Exception as e:
        print(f"Error: 설정을 읽는 중 오류가 발생했습니다: {e}")
        return

    print(f"총 {len(models)}개의 모델을 순차적으로 학습합니다: {models}")
    print("-" * 50)

    for model_key in models:
        print(f">>> [시작] 모델 학습: {model_key}")
        
        # subprocess를 사용하여 별도 프로세스로 실행 (메모리 초기화 효과)
        # 윈도우와 리눅스 모두 호환되도록 python 실행 방식 설정
        python_exe = sys.executable
        try:
            subprocess.run([python_exe, "train.py", "--model", model_key], check=True)
            print(f">>> [완료] {model_key} 학습 및 결과 저장 성공")
        except subprocess.CalledProcessError as e:
            print(f"!!! [에러] {model_key} 학습 도중 문제가 발생했습니다: {e}")
            print("다음 모델로 넘어갑니다.")
            continue

    print("-" * 50)
    print("모든 모델에 대한 학습 공정이 끝났습니다.")

if __name__ == "__main__":
    train_all()
