import os
import jieba
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from .ret import QueryProcessor
import chromadb
from .ret import ZhipuEmbeddingFunction

load_dotenv()
ZHIPU_API_KEY = os.environ.get('Api_Key')


class BM25Retriever:
    def __init__(self, chroma_persist_dir: str = "./law_chroma_db",
                 case_chroma_persist_dir: str = "./law_case_chroma_db"):

        # 初始化Chroma客户端（仅用于获取文档数据）
        self.law_client = chromadb.PersistentClient(path=chroma_persist_dir)
        self.law_collection = self.law_client.get_collection(
            name="law_articles",
            embedding_function=ZhipuEmbeddingFunction()
        )

        self.case_client = chromadb.PersistentClient(path=case_chroma_persist_dir)
        self.case_collection = self.case_client.get_collection(
            name="law_case_articles",
            embedding_function=ZhipuEmbeddingFunction()
        )

        self.query_processor = QueryProcessor(ZHIPU_API_KEY)

        # 加载文档数据并初始化BM25
        self.law_metadata_list = self._load_law_metadata()
        self.case_metadata_list = self._load_case_metadata()

        self._init_bm25()

    def _load_law_metadata(self) -> list:
        """加载法律条文元数据"""
        try:
            all_docs = self.law_collection.get()
            metadata_list = []
            for i in range(len(all_docs["ids"])):
                meta = all_docs["metadatas"][i]
                required_fields = ["law_name", "law_article_num", "keywords"]
                if not all(field in meta for field in required_fields):
                    continue

                metadata_list.append({
                    "id": all_docs["ids"][i],
                    "text": all_docs["documents"][i],
                    "law_name": meta["law_name"],
                    "law_article_num": meta["law_article_num"],
                    "category": meta.get("category", meta["keywords"]),
                    "data_type": "law"
                })
            print(f"BM25 - 成功加载 {len(metadata_list)} 条法律条文数据")
            return metadata_list
        except Exception as e:
            print(f"BM25 - 加载法律条文元数据失败: {e}")
            return []

    def _load_case_metadata(self) -> list:
        """加载案例元数据"""
        try:
            all_docs = self.case_collection.get()
            metadata_list = []
            for i in range(len(all_docs["ids"])):
                meta = all_docs["metadatas"][i]
                metadata_list.append({
                    "id": all_docs["ids"][i],
                    "text": all_docs["documents"][i],
                    "case_id": meta.get("doc_id", all_docs["ids"][i]),
                    "data_type": "case"
                })
            print(f"BM25 - 成功加载 {len(metadata_list)} 条案例数据")
            return metadata_list
        except Exception as e:
            print(f"BM25 - 加载案例元数据失败: {e}")
            return []

    def _init_bm25(self):
        """初始化BM25模型"""
        # 法律条文BM25
        self.law_corpus_texts = [item["text"] for item in self.law_metadata_list]
        self.law_tokenized_corpus = [list(jieba.cut(text)) for text in self.law_corpus_texts]
        self.law_bm25 = BM25Okapi(self.law_tokenized_corpus) if self.law_tokenized_corpus else None

        # 案例BM25
        self.case_corpus_texts = [item["text"] for item in self.case_metadata_list]
        self.case_tokenized_corpus = [list(jieba.cut(text)) for text in self.case_corpus_texts]
        self.case_bm25 = BM25Okapi(self.case_tokenized_corpus) if self.case_tokenized_corpus else None

    def _law_bm25_search(self, query: str, top_k: int) -> list:
        """纯BM25检索法律条文"""
        if not self.law_bm25 or not self.law_metadata_list:
            return []

        query_tokens = list(jieba.cut(query))
        scores = self.law_bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                **self.law_metadata_list[idx],
                "bm25_score": round(scores[idx], 4),
                "type": "bm25"
            }
            for idx in top_indices
        ]

    def _case_bm25_search(self, query: str, top_k: int) -> list:
        """纯BM25检索案例"""
        if not self.case_bm25 or not self.case_metadata_list:
            return []

        query_tokens = list(jieba.cut(query))
        scores = self.case_bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                **self.case_metadata_list[idx],
                "bm25_score": round(scores[idx], 4),
                "type": "bm25"
            }
            for idx in top_indices
        ]

    def retrieve(self, query: str, law_top_k: int = 3, case_top_k: int = 2, search_type: str = "bm25") -> dict:
        """
        纯BM25检索
        返回: {"law": 法律条文结果, "case": 案例结果}
        """
        print(f"BM25检索 - 原始查询: {query}")

        # 查询改写
        rewritten_query = self.query_processor.rewrite_query(query)
        print(f"BM25检索 - 改写后查询: {rewritten_query}")

        # 类别提取（可选，用于分析）
        categories = self.query_processor.extract_category(rewritten_query)
        print(f"BM25检索 - 提取类别: {categories}")

        # 分别进行BM25检索
        law_results = self._law_bm25_search(rewritten_query, law_top_k)
        case_results = self._case_bm25_search(rewritten_query, case_top_k)

        # 添加额外信息
        for result in law_results:
            result["original_query"] = query
            result["rewritten_query"] = rewritten_query
            result["matched_categories"] = categories

        for result in case_results:
            result["original_query"] = query
            result["rewritten_query"] = rewritten_query
            result["matched_categories"] = categories

        return {
            "law": law_results,
            "case": case_results
        }


if __name__ == "__main__":
    # 测试BM25检索
    bm25_retriever = BM25Retriever(
        chroma_persist_dir="./law_chroma_db",
        case_chroma_persist_dir="./law_case_chroma_db"
    )

    test_queries = [
        "离婚冷静期多久？",
        "公司破产清算的偿付顺序？",
        "商标侵权怎么处罚？"
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"BM25检索测试 - 查询：{query}")
        print(f"{'=' * 60}")

        results = bm25_retriever.retrieve(query, law_top_k=3, case_top_k=2)

        # 显示法律条文结果
        print("\n法律条文结果：")
        for i, res in enumerate(results["law"], 1):
            print(f"\n第{i}条法律条文：")
            print(f"  内容：{res['text'][:100]}...")
            print(f"  来源：{res['law_name']}第{res['law_article_num']}条")
            print(f"  BM25得分：{res['bm25_score']}")

        # 显示案例结果
        print("\n案例结果：")
        for i, res in enumerate(results["case"], 1):
            print(f"\n第{i}条案例：")
            print(f"  内容：{res['text'][:100]}...")
            print(f"  案例ID：{res.get('case_id', '未知')}")
            print(f"  BM25得分：{res['bm25_score']}")