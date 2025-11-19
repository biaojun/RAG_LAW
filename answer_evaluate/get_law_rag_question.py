import json

def extract_user_inputs(input_file, output_file):
    # 读取输入的JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有user_input字段
    user_inputs = [item['user_input'] for item in data if 'user_input' in item]
    
    # 将结果写入输出的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(user_inputs, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 输入文件路径（替换为你的JSON文件路径）
    input_json = "testset.json"
    # 输出文件路径（结果将保存到这里）
    output_json = "user_inputs.json"
    
    extract_user_inputs(input_json, output_json)
    print(f"已成功提取所有user_input，结果保存至 {output_json}")