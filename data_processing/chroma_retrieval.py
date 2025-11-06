import os
import numpy as np
import jieba
import chromadb
import requests
from rank_bm25 import BM25Okapi
from typing import List, Dict
from zai import ZhipuAI
# 智谱Embedding3配置
class ZhipuEmbeddingFunction:
    """智谱Embedding3嵌入函数"""
    """（添加name属性）"""
    def name(self):
        """返回嵌入模型名称（Chroma要求的方法）"""
        return "zhipu-embedding-3"  # 关键修复：将name改为方法
    def __call__(self, input: List[str]) -> List[List[float]]:
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        client = ZhipuAI(api_key= self.zhipu_api_key)
        response = client.embeddings.create(
        model="embedding-3", #填写需要调用的模型编码
        input=input,
        )
        return [data_point.embedding for data_point in response.data]
    def embed_query(self, input: str) -> List[float]:
        """为查询生成嵌入向量"""
        return self.__call__(input)


class LegalChromaRetriever:
    def __init__(self, chroma_persist_dir: str = "law_chroma_db"):
        self.workdir = os.getcwd()
        chroma_persist_path = os.path.join(self.workdir, chroma_persist_dir)
        self.client = chromadb.PersistentClient(path=chroma_persist_path)
        self.collection = self.client.get_collection(
            name="law_articles",
            embedding_function=ZhipuEmbeddingFunction()
        )
        self.metadata_list = self._load_and_validate_metadata()
        if not self.metadata_list:
            raise ValueError("请先运行zhipu_store.py入库数据")
        self._init_bm25()

    def _load_and_validate_metadata(self) -> List[Dict]:
        all_docs = self.collection.get()
        metadata_list = []
        for i in range(len(all_docs["ids"])):
            meta = all_docs["metadatas"][i]
            required_fields = ["law_name", "law_article_num"]
            if not all(field in meta for field in required_fields):
                raise ValueError(f"元数据缺少必要字段，应为{required_fields}，实际为{meta.keys()}")
            metadata_list.append({
                "id": all_docs["ids"][i],
                "text": all_docs["documents"][i],
                "law_name": meta["law_name"],
                "law_article_num": meta["law_article_num"]
            })
        return metadata_list

    def _init_bm25(self):
        self.corpus_texts = [item["text"] for item in self.metadata_list]
        self.tokenized_corpus = [list(jieba.cut(text)) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "law_name": results["metadatas"][0][i]["law_name"],
                "law_article_num": results["metadatas"][0][i]["law_article_num"],
                "similarity": round(1 / (1 + results["distances"][0][i]), 4),
                "type": "vector"
            }
            for i in range(len(results["ids"][0]))
        ]

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id": self.metadata_list[idx]["id"],
                "text": self.metadata_list[idx]["text"],
                "law_name": self.metadata_list[idx]["law_name"],
                "law_article_num": self.metadata_list[idx]["law_article_num"],
                "similarity": round(scores[idx], 4),
                "type": "keyword"
            }
            for idx in top_indices
        ]

    def _hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
        vector_res = self._vector_search(query, top_k=top_k*2)
        keyword_res = self._keyword_search(query, top_k=top_k*2)
        
        score_map = {}
        if vector_res:
            vec_scores = [r["similarity"] for r in vector_res]
            min_v, max_v = min(vec_scores), max(vec_scores)
            for r in vector_res:
                norm_score = (r["similarity"] - min_v) / (max_v - min_v) if (max_v - min_v) else 0
                score_map[r["id"]] = alpha * norm_score
        
        if keyword_res:
            key_scores = [r["similarity"] for r in keyword_res]
            min_k, max_k = min(key_scores), max(key_scores)
            for r in keyword_res:
                norm_score = (r["similarity"] - min_k) / (max_k - min_k) if (max_k - min_k) else 0
                score_map[r["id"]] = score_map.get(r["id"], 0) + (1 - alpha) * norm_score
        
        sorted_ids = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "id": doc_id,
                "text": next(m["text"] for m in self.metadata_list if m["id"] == doc_id),
                "law_name": next(m["law_name"] for m in self.metadata_list if m["id"] == doc_id),
                "law_article_num": next(m["law_article_num"] for m in self.metadata_list if m["id"] == doc_id),
                "similarity": round(score, 4),
                "type": "hybrid"
            }
            for doc_id, score in sorted_ids
        ]

    def retrieve(self, query: str, top_k: int = 5, search_type: str = "hybrid") -> List[Dict]:
        if search_type == "vector":
            return self._vector_search(query, top_k)
        elif search_type == "keyword":
            return self._keyword_search(query, top_k)
        elif search_type == "hybrid":
            return self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"不支持的检索类型：{search_type}")


if __name__ == "__main__":
    retriever = LegalChromaRetriever()
    
    test_queries = [
        "离婚冷静期多久？",
        "公司破产清算的偿付顺序？"
    ]
    
    for query in test_queries:
        print(f"\n===== 查询：{query} =====")
        results = retriever.retrieve(query, top_k=3,search_type= "vector")
        for i, res in enumerate(results, 1):
            print(f"\n第{i}条结果：")
            print(f"  法条：{res['text']}")
            print(f"  来源：{res['law_name']}第{res['law_article_num']}条")
            print(f"  相似度：{res['similarity']}")
            print(f"  类型：{res['type']}")