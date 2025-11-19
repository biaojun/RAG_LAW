import json
with open('formal_law_case.json', 'r', encoding='utf-8') as f:
    cases = json.load(f)
count = len(cases)
case_count = int(count / 9)
# 将每个案例写入单独的JSON文件
tmp_cases = []
for i, case in enumerate(cases, 0):
    tmp_cases.append(case)
    if i % case_count == 0:
        with open(f'formal_law_case_data/formal_law_case_{i//case_count}.json', 'w', encoding='utf-8') as f:
            json.dump(tmp_cases, f, ensure_ascii=False, indent=4)
        tmp_cases= []

print(f"已成功将案例分成9个JSON文件")