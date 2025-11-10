import pandas as pd
import json
import os
import sys
# import re # 已移除，不再需要

# 移除了 parse_related_laws 函数
# 移除了 parse_keywords 函数

if __name__ == "__main__":
    input_file = 'test_law_case.xlsx'
    if len(sys.argv) < 2:
        print(f"Test Started, only test data in {input_file} will be processed\n")
    elif len(sys.argv) > 2:
        raise ValueError(f"Invalid param: {sys.argv[1:]}")
    else:
        input_file = sys.argv[1]
        print(f"Formal data preprocess started, test data in {input_file} will be processed\n")
    
    # Read file
    workdir = os.getcwd()
    excel_file = pd.ExcelFile(os.path.join(workdir, input_file))

    # Get data from specified worksheet
    df = excel_file.parse('Sheet1')

    # View basic data information
    print('Data basic information:')
    df.info()

    # View number of rows and columns in the dataset
    rows, columns = df.shape
    print(f"\nDataset has {rows} rows and {columns} columns.")

    # 移除了旧的 df.drop() 和列重命名逻辑
    # 我们将保留所有需要的列，并直接使用原始名称

    # 用于存储最终结果的列表
    json_result = []

    # 遍历 DataFrame 的每一行来构建新的 JSON 结构
    for index, row in df.iterrows():
        
        # 1. 处理 'categories' 字段
        categories = ""
        keywords_val = row.get('关键词') # 使用 .get() 避免列不存在的错误
        
        # 检查关键词是否为 NaN 或空字符串
        if pd.isna(keywords_val) or str(keywords_val).strip() == "":
            # 如果关键词为空，则合并一级分类和二级分类
            cats_list = []
            cat1 = str(row['一级分类']) if pd.notna(row['一级分类']) and str(row['一级分类']).strip() else None
            cat2 = str(row['二级分类']) if pd.notna(row['二级分类']) and str(row['二级分类']).strip() else None
            
            if cat1:
                cats_list.append(cat1)
            if cat2:
                cats_list.append(cat2)
            categories = ",".join(cats_list)
        else:
            # 如果关键词不为空，则直接使用
            categories = str(keywords_val)
            
        # 2. 处理 'text' 字段 (拼接)
        # 确保字段存在且在拼接时处理 NaNs
        case_name = str(row.get('案名', '')) if pd.notna(row.get('案名')) else "无"
        case_facts = str(row.get('基本案情', '')) if pd.notna(row.get('基本案情')) else "无"
        judgment_reason = str(row.get('裁判理由', '')) if pd.notna(row.get('裁判理由')) else "无"
        judgement_holding = str(row.get('裁判要旨', '')) if pd.notna(row.get('裁判要旨')) else "无"
        
        text = (
            f"[案名]{case_name}\n"
            f"[基本案情]{case_facts}\n"
            f"[裁判理由]{judgment_reason}\n"
            f"[裁判要旨]{judgement_holding}"
        )
        
        # 3. 处理 'metadata' 字段
        
        # 处理裁判日期 (转换为 YYYY-MM-DD 字符串，处理 NaT/NaN)
        judgment_date = None
        if pd.notna(row.get('裁判日期')):
            try:
                # 尝试将其转换为 datetime 并格式化
                judgment_date = pd.to_datetime(row['裁判日期']).strftime('%Y-%m-%d')
            except Exception:
                # 如果失败，直接转为字符串
                judgment_date = str(row['裁判日期'])
        else:
            judgment_date = "无"

        metadata = {
            "categories": categories,
            "法院": str(row.get('法院名称')) if pd.notna(row.get('法院名称')) else "无",
            "省份": str(row.get('省份')) if pd.notna(row.get('省份')) else "无",
            "裁判日期": judgment_date,
            "案号": str(row.get('案件证号')) if pd.notna(row.get('案件证号')) else "无"
        }
        
        # 4. 组装最终的单条记录
        record = {
            "doc_id": str(row.get('id')) if pd.notna(row.get('id')) else None,
            "text": text,
            "metadata": metadata
        }
        
        json_result.append(record)

  
    
    # Save to file
    output_filename = f"{input_file.strip('.xlsx')}.json"
    json_path = os.path.join(workdir, output_filename)
    
    print(f"\nProcessing complete. Saving to {json_path}")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=4)

    print("Done.")