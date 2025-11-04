import os
import re
import chromadb
import requests
from zhipu_config import read_config
import sys
from typing import List, Dict
from zai import ZhipuAiClient

# 智谱Embedding3配置


class ZhipuEmbeddingFunction:
    """智谱Embedding3嵌入函数"""
    """（添加name属性）"""
    def name(self,):
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        """返回嵌入模型名称（Chroma要求的方法）"""
        return "zhipu-embedding-3"  # 关键修复：将name改为方法
    def __call__(self, input: List[str]) -> List[List[float]]:
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": f"Bearer {ZHIPU_API_KEY}"
        # }
        data = {
            "model": "embedding-3",
            "input": input
        }
        # response = requests.post(ZHIPU_EMBEDDING_URL, headers=headers, json=data)
        client = ZhipuAiClient(api_key= self.zhipu_api_key)
        response = client.embeddings.create(
        model="embedding-3", #填写需要调用的模型编码
        input=input,
        )
        # response.raise_for_status()
        # return [item["embedding"] for item in response.json()["data"]]
        return [data_point.embedding for data_point in response.data]

class LegalChromaStore:
    def __init__(self,  workdir = os.getcwd(), chroma_persist_dir: str = "chroma_legal_db", config_file: str = "config/basic_config.json"):
        self.workdir = workdir
        config_path = os.path.join(workdir, config_file)
        chroma_persist_dir = os.path.join(workdir, chroma_persist_dir)
        
        config = read_config(config_path)
        self.max_batch_size = config["ZHIPU_MAX_BATCH_SIZE"]
        self.client = chromadb.PersistentClient(path=chroma_persist_dir)
        self.embedding_function = ZhipuEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="legal_articles",
            embedding_function=self.embedding_function,
            metadata={"description": "元数据格式严格为{law_name, law_article_num}"}
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
        return {
            # 提取字段严格对应元数据要求
            "law_name": match.group(1).strip(),          # 法律名称（如"中华人民共和国残疾人保障法"）
            "law_article_num": match.group(2).strip(),   # 条款号（如"50"）
            "text": f"{match.group(1)}第{match.group(2)}条：{match.group(3)}",
            "id": f"{match.group(1)}_{match.group(2)}"
        }

    def store_data(self, cleaned_data_dir_name: str = "cleaned_legal_data"):
        """分批入库，解决智谱API单次输入限制"""
        cleaned_data_dir = os.path.join(self.workdir, cleaned_data_dir_name)
        if self.collection.count() > 0:
            print(f"已存在{self.collection.count()}条数据，跳过入库")
            return
        
        # 收集所有待入库数据
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for filename in os.listdir(cleaned_data_dir):
            if filename.startswith("cleaned_") and filename.endswith(".txt"):
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
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"已入库第{i+1}-{end}条（共{total}条）")
        
        print(f"全部数据入库完成，共{self.collection.count()}条")

# Usage:
# For test:
#     python chroma_store.py
# For data store:
#     python chroma_store.py formal
if __name__ == "__main__":
    data_folder = "test_law_data"
    try:
        # Check for excessive arguments
        if len(sys.argv) > 2:
            raise ValueError(f"Too many arguments: {sys.argv[1:]}. Maximum 1 argument allowed ('test').")
        
        # Handle single argument case
        if len(sys.argv) == 2:
            arg = sys.argv[1]
            if arg != "formal":
                raise ValueError(f"Invalid argument: {arg}. Only 'test' is supported.")
            data_folder = "cleaned_legal_data"
            print("Production mode activated - all data will be stored\n")
        else:
            # No arguments: Production mode
            print("Test mode activated - only test data will be stored\n")

    except ValueError as e:
        # Catch and display argument-related errors, then exit
        print(f"Error: {e}")
        sys.exit(1)
    workdir = os.getcwd()
    store = LegalChromaStore(workdir=workdir)
    
    store.store_data(data_folder)