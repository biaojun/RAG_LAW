import os
from dotenv import load_dotenv
import chromadb
from zai import ZhipuAiClient
from ret import ZhipuEmbeddingFunction, QueryProcessor

load_dotenv()
ZHIPU_API_KEY = os.environ.get('Api_Key')


class VectorRetriever:
    def __init__(self, chroma_persist_dir: str = "./law_chroma_db",
                 case_chroma_persist_dir: str = "./law_case_chroma_db"):

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

    def _law_vector_search(self, query: str, top_k: int) -> list:
        """纯向量检索法律条文"""
        try:
            law_results = self.law_collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            results = []
            for i in range(len(law_results["ids"][0])):
                results.append({
                    "id": law_results["ids"][0][i],
                    "text": law_results["documents"][0][i],
                    "law_name": law_results["metadatas"][0][i]["law_name"],
                    "law_article_num": law_results["metadatas"][0][i]["law_article_num"],
                    "category": law_results["metadatas"][0][i].get("category", "通用"),
                    "similarity": round(1 / (1 + law_results["distances"][0][i]), 4),
                    "type": "vector",
                    "data_type": "law"
                })
            return results
        except Exception as e:
            print(f"法律条文向量检索失败: {e}")
            return []

    def _case_vector_search(self, query: str, top_k: int) -> list:
        """纯向量检索案例"""
        try:
            case_results = self.case_collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            results = []
            for i in range(len(case_results["ids"][0])):
                results.append({
                    "id": case_results["ids"][0][i],
                    "text": case_results["documents"][0][i],
                    "case_name": case_results["metadatas"][0][i].get("case_name", "未知案例"),
                    "case_id": case_results["metadatas"][0][i].get("doc_id", case_results["ids"][0][i]),
                    "category": case_results["metadatas"][0][i].get("category", "案例"),
                    "similarity": round(1 / (1 + case_results["distances"][0][i]), 4),
                    "type": "vector",
                    "data_type": "case"
                })
            return results
        except Exception as e:
            print(f"案例向量检索失败: {e}")
            return []

    def retrieve(self, query: str, law_top_k: int = 3, case_top_k: int = 2) -> dict:
        """
        纯向量检索
        返回: {"law": 法律条文结果, "case": 案例结果}
        """
        print(f"向量检索 - 原始查询: {query}")

        # 查询改写
        rewritten_query = self.query_processor.rewrite_query(query)
        print(f"向量检索 - 改写后查询: {rewritten_query}")

        # 类别提取（可选，用于分析）
        categories = self.query_processor.extract_category(rewritten_query)
        print(f"向量检索 - 提取类别: {categories}")

        # 分别进行向量检索
        law_results = self._law_vector_search(rewritten_query, law_top_k)
        case_results = self._case_vector_search(rewritten_query, case_top_k)

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
    # 测试向量检索
    vector_retriever = VectorRetriever(
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
        print(f"向量检索测试 - 查询：{query}")
        print(f"{'=' * 60}")

        results = vector_retriever.retrieve(query, law_top_k=3, case_top_k=2)

        # 显示法律条文结果
        print("\n法律条文结果：")
        for i, res in enumerate(results["law"], 1):
            print(f"\n第{i}条法律条文：")
            print(f"  内容：{res['text'][:100]}...")
            print(f"  来源：{res['law_name']}第{res['law_article_num']}条")
            print(f"  相似度：{res['similarity']}")

        # 显示案例结果
        print("\n案例结果：")
        for i, res in enumerate(results["case"], 1):
            print(f"\n第{i}条案例：")
            print(f"  内容：{res['text'][:100]}...")
            print(f"  案例ID：{res.get('case_id', '未知')}")
            print(f"  相似度：{res['similarity']}")