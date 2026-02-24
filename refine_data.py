import json
import os

def refine_dataset():
    input_path = "data/paperclinic_generated_dataset_gemini.json"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} 파일을 찾을 수 없습니다.")
        return

    print(f"데이터 정제 시작: {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        full_json = json.load(f)
    
    data_list = full_json.get('data', [])
    
    # 1. 질문별로 데이터 그룹화 (filename + question 조합)
    groups = {}
    for item in data_list:
        # filename과 question이 같으면 같은 질문에 대한 응답들임
        group_key = (item.get('filename', ''), item.get('question', ''))
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(item)

    print(f"총 {len(groups)}개의 고유 질문 그룹을 찾았습니다.")

    # 2. 그룹별로 순회하며 데이터 보정
    refined_count = 0
    for key, items in groups.items():
        # 해당 그룹에서 모범답안(is_original=True) 찾기
        original_item = next((i for i in items if i.get('is_original') is True), None)
        
        if original_item:
            # 모범답안의 evidence_list 추출
            evidence = original_item.get('evidence_list', [])
            # 모범답안의 점수를 100으로 설정
            original_item['score'] = 100
            
            # 해당 그룹의 다른 모든 답안에 evidence_list 채우기
            for item in items:
                if item.get('is_original') is not True:
                    item['evidence_list'] = evidence
                    refined_count += 1

    print(f"보정 완료: {refined_count}개의 변형 답안에 evidence_list를 추가하고, 모범답안 점수를 100점으로 설정했습니다.")

    # 3. 결과 저장
    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(full_json, f, ensure_ascii=False, indent=2)
    
    print("정제된 데이터가 파일에 저장되었습니다.")

if __name__ == "__main__":
    refine_dataset()
