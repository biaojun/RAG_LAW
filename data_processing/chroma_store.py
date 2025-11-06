import os
import re
import chromadb
import requests
from zhipu_config import read_config
import sys
from typing import List, Dict
from zai import ZhipuAI
from cn2an import cn2an
import json

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
        client = ZhipuAI(api_key= self.zhipu_api_key)
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
            metadata={"description": "元数据格式严格为{law_name, law_article_num}"}
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
        law_article_num = cn2an(match.group(2).strip())
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
            case_id = case.get("id", "")
            case_name = case.get("case_name", "")
            criminal_charge = case.get("criminal_charge", "")
            case_facts = case.get("case_facts", "").strip()
            judgment_holding = case.get("judgement_holding", "").strip()
            related_laws = case.get("related_laws", [])
            judgement_date = case.get("judgment_date",[])

            # 构建法律关联信息文本
            law_info = {}
            for law in related_laws:
                law_name = law.get("law_name", "")
                article_nums_arr = law.get("article_numbers", [])
                if not law_name or not article_nums_arr:
                    continue
                law_info[law_name] = article_nums_arr
            law_info_str = json.dumps(law_info, ensure_ascii=False)

            # 生成案件文档文本（整合关键信息，便于后续检索）
            case_text = f"""案件ID：{case_id}, 案件名称：{case_name}, 罪名：{criminal_charge}, 关联法律：{law_info}, 裁判要旨：{judgment_holding}, 基本案情：{case_facts}"""
            parsed_cases.append({
                "text": case_text,
                "metadata": {
                    "case_id": str(case_id),
                    "case_name": case_name,
                    "criminal_charge": criminal_charge,
                    "related_laws": law_info_str,
                    "judgement_date": judgement_date,
                },
                "id": f"case_{case_id}"  # 唯一ID：case_+案件ID
            })
        return parsed_cases

    
    def store_law(self, cleaned_data_dir_name):
        """分批入库，解决智谱API单次输入限制"""
        cleaned_data_dir = os.path.join(self.workdir, cleaned_data_dir_name)
        if self.law_collection.count() > 0:
            print(f"已存在{self.law_collection.count()}条数据，跳过入库")
            return
        
        # 收集所有待入库数据
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
                            })
                            all_ids.append(parsed["id"])
        
        if not all_documents:
            print("未找到有效数据")
            return
        
        # 分批入库（每批不超过self.max_batch_size）
        total = len(all_documents)
        print(f"开始入库，共{total}条数据，每批最多{self.max_batch_size}条...")
        
        for i in range(0, total, self.max_batch_size):
            end = min(i + self.max_batch_size, total)
            batch_docs = all_documents[i:end]
            batch_metas = all_metadatas[i:end]
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
        
        # 分批入库
        for i in range(0, total, self.max_batch_size):
            end = min(i + self.max_batch_size, total)
            batch_docs = all_documents[i:end]
            batch_metas = all_metadatas[i:end]
            batch_ids = all_ids[i:end]
            
            self.law_case_collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"已入库案件数据第{i+1}-{end}条（共{total}条）")
        
        print(f"案件数据全部入库完成，共{self.law_case_collection.count()}条")
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
        store.store_case(os.path.join(workdir, data_storage))
    
    # For test, testing the law case retrieval from db 
    if target == "law_case":
        res = store.get_all_cases()
        print("Sample cases:")
        print(res)