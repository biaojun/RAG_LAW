import json
import sys

def remove_response_newlines(json_file_path, output_file_path=None):
    """
    读取JSON文件，去除所有元素中response字段的换行符，并保存结果
    
    参数:
        json_file_path: 输入JSON文件路径
        output_file_path: 输出JSON文件路径，默认为原文件路径加"_processed"后缀
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据是列表类型
        if not isinstance(data, list):
            raise ValueError("JSON文件内容必须是数组形式")
        
        # 处理每个元素的response字段
        for item in data:
            if "response" in item and isinstance(item["response"], str):
                # 去除所有换行符(\n)
                item["response"] = item["response"].replace('\n', '')
        
        # 确定输出文件路径
        if not output_file_path:
            # 生成带后缀的输出文件名（如 input.json -> input_processed.json）
            if '.' in json_file_path:
                parts = json_file_path.rsplit('.', 1)
                output_file_path = f"{parts[0]}_processed.{parts[1]}"
            else:
                output_file_path = f"{json_file_path}_processed"
        
        # 保存处理后的JSON
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成，结果已保存至: {output_file_path}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{json_file_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 文件 '{json_file_path}' 不是有效的JSON格式")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python remove_newlines.py <输入JSON文件路径>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    remove_response_newlines(input_file)