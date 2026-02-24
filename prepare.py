import json
import random
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import login

def prepare_data(input_path, repo_id):
    # Load the JSON data
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        full_json = json.load(f)
    
    data_list = full_json.get('data', [])
    if not data_list:
        print("Warning: No data found in the 'data' key.")
        return

    # Shuffle and split (80% train, 20% test)
    random.seed(42)
    random.shuffle(data_list)
    split_idx = int(len(data_list) * 0.8)
    
    train_data = data_list[:split_idx]
    eval_data = data_list[split_idx:]
    
    print(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")

    # Convert to DatasetDict
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    ds_dict = DatasetDict({
        'train': train_dataset,
        'test': eval_dataset
    })

    # Upload to Hugging Face Hub
    print(f"Uploading dataset to {repo_id}...")
    ds_dict.push_to_hub(repo_id, private=True)
    print("Dataset upload complete.")

if __name__ == "__main__":
    input_file = "data/paperclinic_generated_dataset_gemini.json"
    repo_id = input("Enter the Hugging Face repository ID to upload dataset (e.g., username/repo-name): ")
    
    # Check for HF_TOKEN in environment
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    
    prepare_data(input_file, repo_id)
    
    # Save repo_id for train.py (optional but good for automation)
    with open(".repo_id", "w") as f:
        f.write(repo_id)
