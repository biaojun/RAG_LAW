import os
from difflib import SequenceMatcher

from dotenv import load_dotenv
import numpy as np
import jieba
import chromadb
import requests
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from zai import ZhipuAiClient
import json

load_dotenv()
ZHIPU_API_KEY = os.environ.get('Api_Key')


class ZhipuEmbeddingFunction:
    def name(self):
        return "zhipu-embedding-3"

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            print("警告: 输入为空，返回默认向量")
            return [[0.0] * 1024]

        client = ZhipuAiClient(api_key=ZHIPU_API_KEY)
        response = client.embeddings.create(
            model="embedding-3",
            input=input,
        )
        return [data_point.embedding for data_point in response.data]

    def embed_query(self, input: str) -> List[float]:
        return self.__call__(input)


class QueryProcessor:
    def __init__(self, api_key: str):
        self.client = ZhipuAiClient(api_key=api_key)
        self.all_categories = [
            "宪法",

            # 基本部门法
            "民法",  # 包含合同、物权、婚姻家庭、继承、侵权责任等
            "商法",  # 包含公司、证券、保险、票据等
            "刑法",
            "行政法",
            "经济法",  # 包含反垄断、消费者权益保护、税法等
            "社会法",  # 包含劳动法、社会保障法等
            "诉讼法",  # 包含民事、刑事、行政诉讼法
            "国际法",

            # 重要领域法
            "环境与资源法",
            "知识产权法",
            "军事与国防法",
            "教育科技法",
            "医疗卫生法"
        ]

    def rewrite_query(self, original_query: str) -> str:
        prompt = f"""
        你是一名专业的法律语言优化助手，擅长将用户输入的查询改写为更加正式、准确、易于理解的法律问题表达形式。

        请对以下法律查询进行改写，使其：
        含义保持一致；
        语言更清晰、逻辑更连贯；
        可以适当添加或替换为常见的法律术语、同义表达或更规范的问法；
        输出中不要出现任何解释说明，只返回改写后的句子。

        原查询：{original_query}
        """

        try:
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"查询改写失败，使用原查询: {e}")
            return original_query

    def extract_category(self, query: str) -> List[str]:
        prompt = f"""
        请分析以下法律查询，判断它可能涉及哪些法律类别。请从以下类别列表中选择最相关的1-3个类别：
        {', '.join(self.all_categories)}

        查询内容："{query}"

        请严格按照以下格式返回结果：
        ["类别1", "类别2", "类别3"]

        要求：
        1. 只返回JSON格式的列表，不要有其他任何文字
        2. 选择最相关的1-3个类别，按相关性从高到低排列
        3. 如果确实无法确定类别，可以返回空列表[]
        """

        try:
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()

            try:
                categories = json.loads(result)
                if isinstance(categories, list):
                    valid_categories = [cat for cat in categories if cat in self.all_categories]
                    return valid_categories
            except json.JSONDecodeError:
                print(f"JSON解析失败，返回内容: {result}")

            return []

        except Exception as e:
            print(f"LLM类别提取失败: {e}")
            return []

    def calculate_similarity(self,text1: str, text2: str) -> float:
        # 字符级别相似度
        char_similarity = SequenceMatcher(None, text1, text2).ratio()

        # 词语级别相似度
        words1 = set(jieba.cut(text1))
        words2 = set(jieba.cut(text2))

        if not words1 or not words2:
            return char_similarity

        intersection = words1 & words2
        union = words1 | words2
        word_similarity = len(intersection) / len(union) if union else 0.0

        return max(char_similarity, word_similarity)


class RerankModel:
    def __init__(self, api_key: str):
        self.client = ZhipuAiClient(api_key=api_key)
        self.api_key = api_key
        self.url = "https://open.bigmodel.cn/api/paas/v4/rerank"

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        if len(documents) <= top_k:
            return documents

        document_texts = [doc["text"] for doc in documents]

        payload = {
            "model": "rerank",
            "query": query,
            "top_n": top_k,
            "documents": document_texts
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()

            result_data = response.json()

            if 'results' in result_data:
                reranked_results = []
                for item in result_data["results"]:
                    original_index = item["index"]
                    original_doc = documents[original_index]

                    reranked_doc = original_doc.copy()
                    reranked_doc["rerank_score"] = item.get("relevance_score", 0)
                    reranked_doc["rerank_index"] = item["index"]
                    reranked_results.append(reranked_doc)

                print(f"重排序完成，返回前{len(reranked_results)}个结果")
                return reranked_results
            else:
                print("重排序API返回格式异常，使用原始排序")
                return documents[:top_k]

        except Exception as e:
            print(f"重排序失败: {e}，使用原始排序")
            return documents[:top_k]


class EnhancedLegalRetriever:
    def __init__(self, chroma_persist_dir: str = "./chroma_legal_db"):
        self.client = chromadb.PersistentClient(path=chroma_persist_dir)
        self.collection = self.client.get_collection(
            name="legal_articles",
            embedding_function=ZhipuEmbeddingFunction()
        )

        self.query_processor = QueryProcessor(ZHIPU_API_KEY)
        self.reranker = RerankModel(ZHIPU_API_KEY)
        self.metadata_list = self._load_and_validate_metadata()

        if not self.metadata_list:
            raise ValueError("请先运行zhipu_store.py入库数据")

        self._init_bm25()
        self._init_category_mapping()

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
                "law_article_num": meta["law_article_num"],
                "category": meta.get("category", meta["law_name"])
            })
        return metadata_list

    def _init_bm25(self):
        self.corpus_texts = [item["text"] for item in self.metadata_list]
        self.tokenized_corpus = [list(jieba.cut(text)) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _init_category_mapping(self):
        self.law_name_to_category = {}
        for item in self.metadata_list:
            law_name = item["law_name"]
            self.law_name_to_category[law_name] = item["category"]

    def _should_include_document(self, law_name: str, target_categories: List[str]) -> bool:
        if not target_categories:
            return True

        doc_category = self.law_name_to_category.get(law_name, "")
        if not doc_category:
            return True

        for target_category in target_categories:
            similarity = self.query_processor.calculate_similarity(doc_category, target_category)
            if similarity > 0.3:
                return True

        return False

    def _filter_by_category(self, categories: List[str], documents: List[Dict]) -> List[Dict]:
        if not categories:
            return documents

        filtered_docs = []
        for doc in documents:
            law_name = doc.get("law_name", "")
            if self._should_include_document(law_name, categories):
                filtered_docs.append(doc)

        print(f"类别过滤: 从{len(documents)}个文档过滤到{len(filtered_docs)}个文档")
        return filtered_docs

    def _bm25_search(self, query: str, top_k: int = 50) -> List[Dict]:
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.metadata_list[idx]["id"],
                "text": self.metadata_list[idx]["text"],
                "law_name": self.metadata_list[idx]["law_name"],
                "law_article_num": self.metadata_list[idx]["law_article_num"],
                "category": self.metadata_list[idx]["category"],
                "bm25_score": round(scores[idx], 4),
                "similarity": round(scores[idx], 4),
                "type": "keyword"
            }
            for idx in top_indices
        ]

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
                "category": results["metadatas"][0][i].get("category", "通用"),
                "similarity": round(1 / (1 + results["distances"][0][i]), 4),
                "type": "vector"
            }
            for i in range(len(results["ids"][0]))
        ]

    def _get_metadata_by_categories(self, categories: List[str]) -> List[Dict]:
        """根据类别过滤元数据"""
        if not categories:
            return self.metadata_list

        filtered_metadata = []
        for item in self.metadata_list:
            law_name = item["law_name"]
            if self._should_include_document(law_name, categories):
                filtered_metadata.append(item)

        print(f"类别过滤: 从{len(self.metadata_list)}个文档过滤到{len(filtered_metadata)}个文档")
        return filtered_metadata

    def _bm25_search_on_filtered(self, query: str, filtered_metadata: List[Dict], top_k: int) -> List[Dict]:
        """在过滤后的文档集上进行BM25检索"""
        if not filtered_metadata:
            return []

        # 为过滤后的文档集构建临时BM25
        corpus_texts = [item["text"] for item in filtered_metadata]
        tokenized_corpus = [list(jieba.cut(text)) for text in corpus_texts]
        temp_bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = list(jieba.cut(query))
        scores = temp_bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": filtered_metadata[idx]["id"],
                "text": filtered_metadata[idx]["text"],
                "law_name": filtered_metadata[idx]["law_name"],
                "law_article_num": filtered_metadata[idx]["law_article_num"],
                "category": filtered_metadata[idx]["category"],
                "bm25_score": round(scores[idx], 4),
                "similarity": round(scores[idx], 4),
                "type": "keyword"
            }
            for idx in top_indices
        ]

    def _vector_search_on_filtered(self, query: str, filtered_metadata: List[Dict], top_k: int) -> List[Dict]:
        """在过滤后的文档集上进行向量检索"""
        if not filtered_metadata:
            return []

        # 构建过滤后的ID列表用于chroma查询
        filtered_ids = [item["id"] for item in filtered_metadata]

        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, len(filtered_ids)),
            where={"id": {"$in": filtered_ids}},  # 关键：在查询时直接过滤
            include=["documents", "metadatas", "distances"]
        )

        return [
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "law_name": results["metadatas"][0][i]["law_name"],
                "law_article_num": results["metadatas"][0][i]["law_article_num"],
                "category": results["metadatas"][0][i].get("category", "通用"),
                "similarity": round(1 / (1 + results["distances"][0][i]), 4),
                "type": "vector"
            }
            for i in range(len(results["ids"][0]))
        ]

    def _hybrid_search(self, query: str, categories: List[str], top_k: int = 50) -> List[Dict]:
        filtered_metadata = self._get_metadata_by_categories(categories)

        bm25_results = self._bm25_search_on_filtered(query, filtered_metadata, top_k=top_k * 2)
        vector_results = self._vector_search_on_filtered(query, filtered_metadata, top_k=top_k * 2)

        score_map = {}

        if bm25_results:
            bm25_scores = [r["bm25_score"] for r in bm25_results]
            min_b, max_b = min(bm25_scores), max(bm25_scores)
            for r in bm25_results:
                norm_score = (r["bm25_score"] - min_b) / (max_b - min_b) if (max_b - min_b) else 0
                score_map[r["id"]] = 0.3 * norm_score

        if vector_results:
            vec_scores = [r["similarity"] for r in vector_results]
            min_v, max_v = min(vec_scores), max(vec_scores)
            for r in vector_results:
                norm_score = (r["similarity"] - min_v) / (max_v - min_v) if (max_v - min_v) else 0
                score_map[r["id"]] = score_map.get(r["id"], 0) + 0.7 * norm_score

        sorted_ids = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

        final_results = []
        for doc_id, score in sorted_ids:
            doc_info = next((m for m in self.metadata_list if m["id"] == doc_id), None)
            if doc_info:
                final_results.append({
                    "id": doc_id,
                    "text": doc_info["text"],
                    "law_name": doc_info["law_name"],
                    "law_article_num": doc_info["law_article_num"],
                    "category": doc_info["category"],
                    "similarity": round(score, 4),
                    "bm25_score": next((r["bm25_score"] for r in bm25_results if r["id"] == doc_id), 0),
                    "type": "hybrid"
                })

        return final_results

    def retrieve(self, query: str, top_k: int = 3, search_type: str = "hybrid") -> List[Dict]:
        print(f"原始查询: {query}")

        # 1. 查询改写
        rewritten_query = self.query_processor.rewrite_query(query)
        print(f"改写后查询: {rewritten_query}")

        # 2. 类别提取
        categories = self.query_processor.extract_category(rewritten_query)
        print(f"提取类别: {categories}")

        # 3. 混合检索（包含类别过滤）
        if search_type == "vector":
            initial_results = self._vector_search(rewritten_query, top_k=50)
            initial_results = self._filter_by_category(categories, initial_results)
        elif search_type == "keyword":
            initial_results = self._bm25_search(rewritten_query, top_k=50)
            initial_results = self._filter_by_category(categories, initial_results)
        else:  # hybrid
            initial_results = self._hybrid_search(rewritten_query, categories, top_k=50)

        print(f"类别过滤后结果数: {len(initial_results)}")

        # 4. 重排序
        final_results = self.reranker.rerank(rewritten_query, initial_results, top_k=top_k)

        for result in final_results:
            result["original_query"] = query
            result["rewritten_query"] = rewritten_query
            result["matched_categories"] = categories

        return final_results

if __name__ == "__main__":
    retriever = EnhancedLegalRetriever()

    test_queries = [
        "离婚冷静期多久？",
        "公司破产清算的偿付顺序？",
        "商标侵权怎么处罚？"
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"查询：{query}")
        print(f"{'=' * 60}")

        results = retriever.retrieve(query, top_k=3, search_type="hybrid")

        for i, res in enumerate(results, 1):
            print(f"\n第{i}条结果：")
            print(f"  法条：{res['text']}")
            print(f"  来源：{res['law_name']}第{res['law_article_num']}条")
            print(f"  类别：{res.get('category', '未知')}")
            print(f"  相似度：{res['similarity']}")
            if 'bm25_score' in res:
                print(f"  BM25得分：{res['bm25_score']}")