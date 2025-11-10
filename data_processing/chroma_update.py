import os
import numpy as np
import jieba
import chromadb
import requests
from rank_bm25 import BM25Okapi
from typing import List, Dict
from zai import ZhipuAiClient
import re

# 智谱Embedding3配置
class ZhipuEmbeddingFunction:
    """智谱Embedding3嵌入函数"""
    """（添加name属性）"""
    def name(self):
        """返回嵌入模型名称（Chroma要求的方法）"""
        return "zhipu-embedding-3"  # 关键修复：将name改为方法
    def __call__(self, input: List[str]) -> List[List[float]]:
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        client = ZhipuAiClient(api_key= self.zhipu_api_key)
        response = client.embeddings.create(
        model="embedding-3", #填写需要调用的模型编码
        input=input,
        )
        return [data_point.embedding for data_point in response.data]
    def embed_query(self, input: str) -> List[float]:
        """为查询生成嵌入向量"""
        return self.__call__(input)


class updater:
    def __init__(self, chroma_persist_dir: str = "law_chroma_db"):
        self.workdir = os.getcwd()
        chroma_persist_path = os.path.join(self.workdir, chroma_persist_dir)
        self.client = chromadb.PersistentClient(path=chroma_persist_path)
        self.collection = self.client.get_collection(
            name="law_articles",
            embedding_function=ZhipuEmbeddingFunction()
        )
    def get_all(self):
        alldocs = self.collection.get()

        return alldocs['ids'],alldocs["documents"],alldocs["metadatas"]
    def keywords_update(self):
        ids,docs,metadatas = self.get_all()
        non_keywords = []
        zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        client = ZhipuAiClient(api_key=zhipu_api_key)
        with open('prompts_keywords.txt', 'r', encoding='utf-8') as file:
            prompt = file.read()
            count_f = 0
        for i in range(len(ids)):
            id = ids[i]
            doc = docs[i]
            metadata=metadatas[i]
            metadata['keywords'] = str(self._generate_keywords(client, prompt, doc))
            if  metadata['keywords'] == '':
                non_keywords.append({
                    'id':id,
                    'doc':doc
                })
                count_f+=1
            
            self.collection.update(
            ids = [id],
            metadatas=[metadata]
            )
            print(f"已更新{i+1}条数据，其中{count_f}条不成功")
        return non_keywords
        
    def _generate_keywords( self,client: ZhipuAiClient, prompt,law_text: str) -> List[str]:
        """使用Zhipu GLM-4为法条文本生成关键词"""
        try:
            prompt = prompt
            response = client.chat.completions.create(
                model="glm-4-flashx",  # 使用GLM-4 (GLM-4.6在API中通常用glm-4标识)
                messages=[
                    {"role": "user", "content": f"法条内容：'{law_text}'\n\n{prompt}"}
                ],
                #max_tokens=60,  # 限制输出长度，关键词一般不长
                temperature=0.1, # 低温获取更稳定、相关的词
                # thinking={
                #          "type": "enabled",    # 启用深度思考模式
                # }
            )
            
            content = response.choices[0].message.content.strip()
            print(f"content:{content}")
            # 解析关键词，模型可能返回 "词1、词2、词3" 或 "词1, 词2, 词3" 或 "1. 词1 2. 词2"
            # 使用正则表达式提取所有中英文词汇
            keywords = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', content)
            print(f"keywords:{keywords}")
            # 返回前4个，符合prompt要求
            return keywords[:4] if keywords else []
        except Exception as e:
            print(f"Warning: Keyword generation failed for doc '{law_text[:50]}...': {e}")
            return [] # 发生错误时返回空列
        

if __name__ == "__main__":
    updater_instance = updater()
    non_keywords = updater_instance.keywords_update()
    print(f"以下法条未成功生成关键词，共{len(non_keywords)}条：")
    for item in non_keywords:
        print(f"ID: {item['id']}, 文本片段: {item['doc'][:50]}...")