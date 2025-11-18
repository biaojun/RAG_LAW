import os
import re
import chromadb
import requests
from zhipu_config import read_config
import sys
from typing import List, Dict
from zai import ZhipuAiClient  # 确保 zai.py 在PYTHONPATH中
from cn2an import cn2an
import json
import pandas as pd

# 智谱Embedding3配置


class ZhipuEmbeddingFunction:
    """智谱Embedding3嵌入函数"""
    """（添加name属性）"""
    def name(self):
        """返回嵌入模型名称（Chroma要求的方法）"""
        return "zhipu-embedding-3"  # 关键修复：将name改为方法
    def __call__(self, input: List[str]) -> List[List[float]]:
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": f"Bearer {ZHIPU_API_KEY}"
        # }
        # data = {
        #     "model": "embedding-3",
        #     "input": input
        # }
        # response = requests.post(ZHIPU_EMBEDDING_URL, headers=headers, json=data)
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        if not self.zhipu_api_key:
             raise ValueError("ZHIPU_API_KEY environment variable not set.")
        client = ZhipuAiClient(api_key= self.zhipu_api_key)
        response = client.embeddings.create(
                        model="embedding-3", #填写需要调用的模型编码
                        input=input,
                    )
        # response.raise_for_status()
        # return [item["embedding"] for item in response.json()["data"]]
        return [data_point.embedding for data_point in response.data]

class LegalChromaStore:
    def __init__(self,  workdir = os.getcwd(), law_db_dir: str = "law_chroma_db", law_case_db_dir: str = "law_case_chroma_db", config_file: str = "config/basic_config.json"):
        self.workdir = workdir
        config_path = os.path.join(workdir, config_file)
        law_db_path = os.path.join(workdir, law_db_dir)
        law_case_db_path = os.path.join(workdir, law_case_db_dir)
        
        config = read_config(config_path)
        self.max_batch_size = config["ZHIPU_MAX_BATCH_SIZE"]
        self.law_client = chromadb.PersistentClient(path=law_db_path)
        self.law_case_client = chromadb.PersistentClient(path=law_case_db_path)
        
        
        self.embedding_function = ZhipuEmbeddingFunction()
        self.law_collection = self.law_client.get_or_create_collection(
            name="law_articles",
            embedding_function=self.embedding_function,
            # --- 修改：更新元数据描述 ---
            metadata={"description": "元数据格式严格为{law_name, law_article_num, keywords}"}
        )
        self.law_case_collection = self.law_case_client.get_or_create_collection(
            name="law_case_articles",
            embedding_function=self.embedding_function,
            metadata={"description": "A data base that store law case"}
        )

    def _parse_cleaned_line(self, line: str) -> Dict:
        """解析文本并严格提取法律名称和条款号"""
        line = line.strip().strip("{}")
        if not line:
            return None
        # 严格匹配格式：法律名称第X条：内容
        pattern = r"(.*?)第(.*)条：(.*)"
        match = re.match(pattern, line)
        if not match:
            return None
        law_name = match.group(1).strip()
        try :
            temp_num = match.group(2).strip()
            law_article_num = cn2an(temp_num)
        except ValueError as e:
            
            print(law_name)
            print (temp_num)
            sys.exit(0)
        law_content = match.group(3).strip()
        return {
            # 提取字段严格对应元数据要求
            "law_name": law_name,         # 法律名称（如"中华人民共和国残疾人保障法"）
            "law_article_num": law_article_num,   # 条款号（如"50"）
            "text": f"{law_name}第{law_article_num}条：{law_content}",
            "id": f"{law_name}_{law_article_num}"
        }
    def _parse_case_json(self, json_path: str) -> List[Dict]:
        """新增：解析test_law_case.json格式数据"""
        parsed_cases = []
        with open(json_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        for case in cases:
            # 提取案件核心信息
            doc_id = case.get("doc_id", "")
            txt = case.get("text", "")
            metadata = case.get("metadata", "")

            parsed_cases.append({
                "text": txt,
                "metadata": metadata,
                "id": doc_id
            })

        return parsed_cases

    # --- 新增：生成关键词的辅助方法 ---
    def _generate_keywords(self, client: ZhipuAiClient, prompt,law_text: str) -> List[str]:
        """使用Zhipu GLM-4为法条文本生成关键词"""
        try:
            prompt = prompt
            response = client.chat.completions.create(
                model="glm-glm-4-flashx",  # 使用GLM-4 (GLM-4.6在API中通常用glm-4标识)
                messages=[
                    {"role": "user", "content": f"法条内容：'{law_text}'\n\n{prompt}"}
                ],
                temperature=0.1, # 低温获取更稳定、相关的词
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析关键词，模型可能返回 "词1、词2、词3" 或 "词1, 词2, 词3" 或 "1. 词1 2. 词2"
            # 使用正则表达式提取所有中英文词汇
            keywords = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', content)
            
            # 返回前4个，符合prompt要求
            return keywords[:4] if keywords else []
        except Exception as e:
            print(f"Warning: Keyword generation failed for doc '{law_text[:50]}...': {e}")
            return [] # 发生错误时返回空列表
    # --- 新增结束 ---

    
    def store_law(self, cleaned_data_dir_name):
        """分批入库，解决智谱API单次输入限制（已更新为包含关键词生成）"""
        cleaned_data_dir = os.path.join(self.workdir, cleaned_data_dir_name)
        if self.law_collection.count() > 0:
            print(f"已存在{self.law_collection.count()}条数据，跳过入库")
            return
        
        # 1. 收集所有待入库数据
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for filename in os.listdir(cleaned_data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(cleaned_data_dir, filename), "r", encoding="utf-8") as f:
                    for line in f:
                        parsed = self._parse_cleaned_line(line)
                        if parsed:
                            all_documents.append(parsed["text"])
                            all_metadatas.append({
                                "law_name": parsed["law_name"],
                                "law_article_num": parsed["law_article_num"]
                                # keywords 将在下一步添加
                            })
                            all_ids.append(parsed["id"])
        
        if not all_documents:
            print("未找到有效数据")
            return
        
        total = len(all_documents)
        
        # --- 2. 新增：逐条生成关键词 ---
        print(f"共找到{total}条数据，开始逐条生成关键词（这可能需要一些时间）...")
        try:
            # 初始化ZhipuAI客户端 (用于GLM-4)
            self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
            if not self.zhipu_api_key:
                raise ValueError("ZHIPU_API_KEY environment variable not set.")
            client = ZhipuAiClient(api_key=self.zhipu_api_key)
            # 读取prompt
            with open('prompts_keywords.txt', 'r', encoding='utf-8') as file:
                prompt = file.read()
            # 逐条为all_metadatas添加keywords
            for i in range(total):
                doc_text = all_documents[i]
                # 调用新方法生成关键词
                keywords_list = self._generate_keywords(client,prompt, doc_text)
                
                # 将关键词列表合并为逗号分隔的字符串，存入元数据
                all_metadatas[i]["keywords"] = ", ".join(keywords_list)
                
                if (i + 1) % 10 == 0 or i == total - 1:
                    print(f"已处理 {i + 1}/{total} 条法条的关键词...")

        except Exception as e:
            print(f"Error during keyword generation: {e}")
            print("Aborting storage process.")
            return
        # --- 关键词生成结束 ---
        
        # 3. 分批入库（每批不超过self.max_batch_size）
        print(f"关键词生成完毕。开始分批入库，共{total}条数据，每批最多{self.max_batch_size}条...")
        
        for i in range(0, total, self.max_batch_size):
            end = min(i + self.max_batch_size, total)
            batch_docs = all_documents[i:end]
            batch_metas = all_metadatas[i:end] # 这里的metas已经包含了keywords
            batch_ids = all_ids[i:end]
            
            self.law_collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"已入库第{i+1}-{end}条（共{total}条）")
        
        print(f"全部数据入库完成，共{self.law_collection.count()}条")

    def store_case(self, case_json_path: str):
        """将解析后的案件数据分批入库"""
        
        # 解析案件JSON数据
        parsed_cases = self._parse_case_json(case_json_path)
        if not parsed_cases:
            print("未解析到有效案件数据")
            return
        
        # 分离文档、元数据和ID
        all_documents = [case["text"] for case in parsed_cases]
        all_metadatas = [case["metadata"] for case in parsed_cases]
        all_ids = [case["id"] for case in parsed_cases]
        
        total = len(all_documents)
        print(f"开始入库案件数据，共{total}条，每批最多{self.max_batch_size}条...")

        # 一条一条的入库，因为case的单条长度过长了
        # 记录一下意外没有入库的案例 case_except
        case_except =pd.DataFrame(columns=['e', 'id', 'document', 'metadata'])
        for i in range(0, total):
            batch_docs = all_documents[i]
            batch_metas = all_metadatas[i]
            batch_ids = all_ids[i]
            try:
                self.law_case_collection.add(
                    documents=[batch_docs],
                    metadatas=[batch_metas],
                    ids=[batch_ids]
                )
            except Exception as e:
                new_row = pd.DataFrame([
                                        [e,batch_ids ,batch_docs , batch_metas]
                                        ],
                                        columns=['e', 'id', 'document', 'metadata'])
                # 使用concat方法添加行
                case_except = pd.concat([case_except, new_row], ignore_index=True)
            print(f"已入库案件数据第{i+1}条（共{total}条）")
        
        print(f"案件数据全部入库完成，共{self.law_case_collection.count()}条")
        return case_except
    def get_all_cases(self, limit: int = 100) -> List[Dict]:
        """批量查询案件数据（默认最多100条，可调整）"""
        # 先获取所有案件的ID
        result = self.law_case_collection.peek(limit)
        if not result:
            return []
        
        # 整理结果为列表
        cases = []
        for i in range(len(result["ids"])):
            cases.append({
                "id": result["ids"][i],
                "document": result["documents"][i],
                "metadata": result["metadatas"][i]
            })
        return cases


# Usage:
#   To store test law data:
#       python chroma_store.py law
#   To store formal law data:
#       python chroma_store.py law formal
#   To store test law case data:
#       python chroma_store.py law_case
#   To store formal law case data:
#       python chroma_store.py law_case formal
if __name__ == "__main__":
    data_storage = "test_law_data"
    target = None
    try:
        # Check argument count
        if len(sys.argv) < 2 or len(sys.argv) > 3:
            raise ValueError("Invalid number of arguments. Usage:\n"
                             "  python chroma_store.py law [formal]\n"
                             "  python chroma_store.py law_case [formal]")
        
        target = sys.argv[1]
        if target not in ["law", "law_case"]:
            raise ValueError(f"Invalid target: {target}. Must be 'law' or 'law_case'")
        
        # Check for formal mode
        formal_mode = len(sys.argv) == 3 and sys.argv[2] == "formal"
        if len(sys.argv) == 3 and not formal_mode:
            raise ValueError(f"Invalid argument: {sys.argv[2]}. Only 'formal' is allowed as second argument")
        
        # Set data folder based on mode
        if target == "law":
            data_storage = "formal_law_data" if formal_mode else "test_law_data"
        else:  # law_case
            data_storage = "formal_law_case.json" if formal_mode else "test_law_case.json"
            
        mode = "formal" if formal_mode else "test"
        print(f"{mode.capitalize()} mode activated - storing {mode} {target} data\n")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    workdir = os.getcwd()
    store = LegalChromaStore(workdir=workdir)
    
    # Store appropriate data based on target
    if target == "law":
        store.store_law(data_storage)
    else:
        case_except = store.store_case(os.path.join(workdir, data_storage))
        case_except.to_excel('case_excep.xlsx')  
    # For test, testing the law case retrieval from db 
    if target == "law_case":
        res = store.get_all_cases()
        print("Sample cases:")
        print(res)