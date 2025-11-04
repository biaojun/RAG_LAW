import pandas as pd
import json
import os
import sys
import re

def parse_related_laws(text):
    """
    Parse the related_laws field to extract legal names and corresponding article numbers
    Return format: [{
        "law_name": "Full legal name (including document number)",
        "article_numbers": ["Article number 1", "Article number 2"]  # Array of pure numeric article numbers
    }]
    """
    # Handle null values or non-string types
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    # 1. Separate legal provisions part (using "######" as separator, keep only left-side legal content)
    law_text = text.split('######')[0].strip()
    if not law_text:  # Return empty list if no legal provisions exist
        return []
    
    # 2. Regular expression matching: Extract legal names from 《》 and subsequent associated articles
    # Matching rule explanation:
    # 《([^》]+?)》: Match legal names within 《》 (supports parentheses and document numbers, e.g., "（法释〔2000〕33号）")
    # (第[\d]+条(?:第[\d]+款)?(?:、第[\d]+条(?:第[\d]+款)?)?)*: Match subsequent articles (supports single/multiple articles, articles with paragraphs)
    pattern = r'《([^》]+?)》(第[\d]+条(?:第[\d]+款)?(?:、第[\d]+条(?:第[\d]+款)?)?)*'
    matches = re.findall(pattern, law_text)
    
    parsed_result = []
    for law_name, articles_str in matches:
        # Skip matches with no article information (only legal name without articles)
        if not articles_str or articles_str.strip() == "":
            continue
        
        # 3. Split multiple articles (e.g., "第1条、第3条第2款" → split into ["第1条", "第3条第2款"])
        single_articles = [art.strip() for art in articles_str.split('、') if art.strip()]
        
        # 4. Extract pure numeric article numbers (e.g., "第133条"→"133", "第54条第2款"→"54")
        article_nums = []
        for art in single_articles:
            # Regular expression to extract number X from "第X条" (compatible with articles with paragraphs)
            art_match = re.search(r'第(\d+)条', art)
            if art_match:
                article_nums.append(art_match.group(1))  # Keep only pure numeric article numbers
        
        # 5. Assemble result (ensure article number array is not empty)
        if article_nums:
            parsed_result.append({
                "law_name": law_name.strip(),  # Legal name (e.g., "最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释（法释〔2000〕33号）")
                "article_numbers": article_nums  # Array of article numbers (e.g., ["1", "3"])
            })
    return parsed_result

def parse_keywords(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    keywords = text.split(',')
    if not keywords:
        return []
    return keywords

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

    df = df.drop(columns=['入库编号', 'source_id', '入库编号', '法院名称', "案件证号", '庭审', '入库日期', '省份'])
    # Define column name mapping relationship
    column_mapping = {
        'id': 'id',
        '一级分类': 'category',
        '二级分类': 'criminal_charge',
        '案名': 'case_name',
        '裁判日期': 'judgment_date',
        '裁决要旨': 'judgment_summary',
        '关键词': 'keywords',
        '基本案情': 'case_facts',
        '裁判理由': 'judgment_reason',
        '裁判要旨': 'judgement_holding',
        '关联索引': 'related_laws'
    }
    # Rename columns
    df = df.rename(columns=column_mapping)
    df['related_laws'] = df['related_laws'].apply(parse_related_laws)
    
    df['keywords'] = df['keywords'].apply(parse_keywords)

    # Convert DataFrame to JSON format
    result = df.to_json(orient='records', force_ascii=False, indent=4)

    # Parse JSON string into Python object
    json_result = json.loads(result)

    # Save to file
    output_filename = f"{input_file.strip('.xlsx')}.json"
    json_path = os.path.join(workdir, output_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, ensure_ascii=False, indent=4)