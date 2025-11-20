import json
import requests
import os

def update_reply_json():
    # 检查文件是否存在
    if not os.path.exists('user_inputs.json'):
        print("错误：未找到user_inputs.json文件")
        return
    
    if not os.path.exists('reply_by_gemini.json'):
        print("错误：未找到reply_by_gemini.json文件")
        return
    
    # 读取用户问题文件
    try:
        with open('user_inputs.json', 'r', encoding='utf-8') as f:
            user_inputs = json.load(f)
    except json.JSONDecodeError:
        print("错误：user_inputs.json格式不正确")
        return
    
    # 读取回复文件
    try:
        with open('reply_by_gemini.json', 'r', encoding='utf-8') as f:
            gemini_replies = json.load(f)
    except json.JSONDecodeError:
        print("错误：reply_by_gemini.json格式不正确")
        return
    
    # API请求地址
    api_url = 'http://127.0.0.1:8000/api/ask'
    
    # 遍历所有问题
    for idx, question in enumerate(user_inputs, start=1):
        print("dealing with question {}:{}...".format(idx, question[50:]))
        
        # 发送请求
        try:
            response = requests.post(
                api_url,
                headers={'Content-Type': 'application/json'},
                json={'question': question},
                timeout=120
            )
            response.raise_for_status()
            api_result = response.json()
        except Exception as e:
            print(f"请求失败：{str(e)}")
            continue
        
        # 解析API返回结果
        answer = api_result.get('answer', '')
        context = api_result.get('context', {})
        
        # 转换context为retrieved_contexts格式
        retrieved_contexts = []
        
        # 处理法律条文
        for law in context.get('law', []):
            law_entry = {
                "id": law.get('id'),
                "law_name": law.get('law_name'),
                "law_article_num": law.get('law_article_num'),
                "snippet": law.get('snippet'),
                "similarity": law.get('similarity'),
                "rerank_score": law.get('rerank_score'),
            }
            # 过滤空值
            law_entry = {k: v for k, v in law_entry.items() if v is not None}
            retrieved_contexts.append(json.dumps(law_entry, ensure_ascii=False))
        
        # 处理案例
        for case in context.get('case', []):
            case_entry = {
                "doc_id": case.get('id') or case.get('case_id'),
                "text": case.get('snippet')
            }
            # 过滤空值
            case_entry = {k: v for k, v in case_entry.items() if v is not None}
            retrieved_contexts.append(json.dumps(case_entry, ensure_ascii=False))
        
        # 更新reply_by_gemini.json中对应的条目
        for reply_item in gemini_replies:
            if reply_item.get('user_input') == question:
                reply_item['response'] = answer
                reply_item['retrieved_contexts'] = retrieved_contexts
                # 更新reference字段（这里简化处理，使用answer的前半部分）
                break
        print("Finish dealing with question {}/{}...".format(idx, len(user_inputs)))
    
    # 保存更新后的文件
    try:
        with open('reply_by_law_rag.json', 'w', encoding='utf-8') as f:
            json.dump(gemini_replies, f, ensure_ascii=False, indent=2)
        print("处理完成，结果已保存至reply_by_law_rag.json")
    except Exception as e:
        print(f"保存文件失败：{str(e)}")

if __name__ == "__main__":
    update_reply_json()