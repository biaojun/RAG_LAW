import os
from difflib import SequenceMatcher
from dotenv import load_dotenv
import numpy as np
import jieba
import chromadb
import requests
from rank_bm25 import BM25Okapi
from typing import List, Dict
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
        self.reranker = RerankModel(ZHIPU_API_KEY)

        self.law_metadata_list = self._load_and_validate_law_metadata()
        self.case_metadata_list = self._load_and_validate_case_metadata()

        if not self.law_metadata_list and not self.case_metadata_list:
            raise ValueError("请先运行chroma_store.py入库数据")

        self._init_bm25()
        self._init_category_mapping()

    def _load_and_validate_law_metadata(self) -> List[Dict]:
        """加载法律条文元数据"""
        try:
            all_docs = self.law_collection.get()
            metadata_list = []
            for i in range(len(all_docs["ids"])):
                meta = all_docs["metadatas"][i]
                required_fields = ["law_name", "law_article_num","keywords"]
                if not all(field in meta for field in required_fields):
                    print(f"警告: 法律条文元数据缺少必要字段，跳过该条: {meta}")
                    continue

                metadata_list.append({
                    "id": all_docs["ids"][i],
                    "text": all_docs["documents"][i],
                    "law_name": meta["law_name"],
                    "law_article_num": meta["law_article_num"],
                    "category": meta.get("category", meta["keywords"]),
                    "data_type": "law"  # 标记数据类型
                })
            print(f"成功加载 {len(metadata_list)} 条法律条文数据")
            return metadata_list
        except Exception as e:
            print(f"加载法律条文元数据失败: {e}")
            return []

    def _load_and_validate_case_metadata(self) -> List[Dict]:
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
                    "data_type": "case"  # 标记数据类型
                })
            print(f"成功加载 {len(metadata_list)} 条案例数据")
            return metadata_list
        except Exception as e:
            print(f"加载案例元数据失败: {e}")
            return []

    def _init_bm25(self):
        self.law_corpus_texts = [item["text"] for item in self.law_metadata_list]
        self.law_tokenized_corpus = [list(jieba.cut(text)) for text in self.law_corpus_texts]
        self.law_bm25 = BM25Okapi(self.law_tokenized_corpus) if self.law_tokenized_corpus else None

        # 案例BM25
        self.case_corpus_texts = [item["text"] for item in self.case_metadata_list]
        self.case_tokenized_corpus = [list(jieba.cut(text)) for text in self.case_corpus_texts]
        self.case_bm25 = BM25Okapi(self.case_tokenized_corpus) if self.case_tokenized_corpus else None

    def _init_category_mapping(self):
        self.law_name_to_category = {}
        for item in self.law_metadata_list:
            law_name = item["law_name"]
            self.law_name_to_category[law_name] = item["law_name"]

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

    def _get_law_metadata_by_categories(self, categories: List[str]) -> List[Dict]:
        """根据类别过滤法律条文元数据"""
        if not categories:
            return self.law_metadata_list

        filtered_metadata = []
        for item in self.law_metadata_list:
            law_name = item["law_name"]
            if self._should_include_document(law_name, categories):
                filtered_metadata.append(item)

        print(f"法律条文类别过滤: 从{len(self.law_metadata_list)}个文档过滤到{len(filtered_metadata)}个文档")
        return filtered_metadata

    def _law_bm25_search_on_filtered(self, query: str, filtered_metadata: List[Dict], top_k: int) -> List[Dict]:
        """在过滤后的法律条文文档集上进行BM25检索"""
        if not filtered_metadata:
            return []

        corpus_texts = [item["text"] for item in filtered_metadata]
        tokenized_corpus = [list(jieba.cut(text)) for text in corpus_texts]
        temp_bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = list(jieba.cut(query))
        scores = temp_bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                **filtered_metadata[idx],
                "bm25_score": round(scores[idx], 4),
                "type": "keyword"
            }
            for idx in top_indices
        ]

    def _law_vector_search_on_filtered(self, query: str, filtered_metadata: List[Dict], top_k: int) -> List[Dict]:
        """在过滤后的法律条文文档集上进行向量检索"""
        if not filtered_metadata:
            return []

        law_ids = [item["id"] for item in filtered_metadata]

        try:
            law_results = self.law_collection.query(
                query_texts=[query],
                n_results=min(top_k, len(law_ids)),
                where={"id": {"$in": law_ids}},
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
            print(f"过滤后法律条文向量检索失败: {e}")
            return []

    def _case_bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """案例BM25检索（不需要过滤）"""
        if not self.case_bm25 or not self.case_metadata_list:
            return []

        query_tokens = list(jieba.cut(query))
        scores = self.case_bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                **self.case_metadata_list[idx],
                "bm25_score": round(scores[idx], 4),
                "type": "keyword"
            }
            for idx in top_indices
        ]

    def _case_vector_search(self, query: str, top_k: int) -> List[Dict]:
        """案例向量检索（不需要过滤）"""
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

    def _hybrid_search(self, query: str, categories: List[str], law_top_k: int = 50, case_top_k: int = 50) -> Dict[str, List[Dict]]:
        """
        混合检索，分别处理法律条文和案例
        返回: {"law": 法律条文结果, "case": 案例结果}
        """
        # 1. 法律条文检索（带类别过滤）
        filtered_law_metadata = self._get_law_metadata_by_categories(categories)

        law_bm25_results = self._law_bm25_search_on_filtered(query, filtered_law_metadata, top_k=law_top_k)
        law_vector_results = self._law_vector_search_on_filtered(query, filtered_law_metadata, top_k=law_top_k)

        # 计算法律条文混合分数
        law_score_map = {}

        if law_bm25_results:
            law_bm25_scores = [r["bm25_score"] for r in law_bm25_results]
            min_b, max_b = min(law_bm25_scores), max(law_bm25_scores)
            for r in law_bm25_results:
                norm_score = (r["bm25_score"] - min_b) / (max_b - min_b) if (max_b - min_b) else 0
                law_score_map[r["id"]] = 0.3 * norm_score

        if law_vector_results:
            law_vec_scores = [r["similarity"] for r in law_vector_results]
            min_v, max_v = min(law_vec_scores), max(law_vec_scores)
            for r in law_vector_results:
                norm_score = (r["similarity"] - min_v) / (max_v - min_v) if (max_v - min_v) else 0
                law_score_map[r["id"]] = law_score_map.get(r["id"], 0) + 0.7 * norm_score

        # 构建法律条文最终结果
        law_sorted_ids = sorted(law_score_map.items(), key=lambda x: x[1], reverse=True)[:law_top_k]
        law_final_results = []
        for doc_id, score in law_sorted_ids:
            doc_info = next((m for m in filtered_law_metadata if m["id"] == doc_id), None)
            if doc_info:
                law_final_results.append({
                    **doc_info,
                    "bm25_score": next((r["bm25_score"] for r in law_bm25_results if r["id"] == doc_id), 0),
                    "similarity": round(score, 4),
                    "type": "hybrid"
                })

        # 2. 案例检索（不带过滤）
        case_bm25_results = self._case_bm25_search(query, top_k=case_top_k)
        case_vector_results = self._case_vector_search(query, top_k=case_top_k)

        # 计算案例混合分数
        case_score_map = {}

        if case_bm25_results:
            case_bm25_scores = [r["bm25_score"] for r in case_bm25_results]
            min_b, max_b = min(case_bm25_scores), max(case_bm25_scores)
            for r in case_bm25_results:
                norm_score = (r["bm25_score"] - min_b) / (max_b - min_b) if (max_b - min_b) else 0
                case_score_map[r["id"]] = 0.3 * norm_score

        if case_vector_results:
            case_vec_scores = [r["similarity"] for r in case_vector_results]
            min_v, max_v = min(case_vec_scores), max(case_vec_scores)
            for r in case_vector_results:
                norm_score = (r["similarity"] - min_v) / (max_v - min_v) if (max_v - min_v) else 0
                case_score_map[r["id"]] = case_score_map.get(r["id"], 0) + 0.7 * norm_score

        # 构建案例最终结果
        case_sorted_ids = sorted(case_score_map.items(), key=lambda x: x[1], reverse=True)[:case_top_k]
        case_final_results = []
        for doc_id, score in case_sorted_ids:
            doc_info = next((m for m in self.case_metadata_list if m["id"] == doc_id), None)
            if doc_info:
                case_final_results.append({
                    **doc_info,
                    "similarity": round(score, 4),
                    "bm25_score": next((r["bm25_score"] for r in law_bm25_results if r["id"] == doc_id), 0),
                    "type": "hybrid"
                })

        return {
            "law": law_final_results,
            "case": case_final_results
        }

    def retrieve(self, query: str, law_top_k: int = 3, case_top_k: int = 2, search_type: str = "hybrid") -> Dict[str, List[Dict]]:
        """
        检索法律条文和案例
        返回: {"law": 法律条文结果, "case": 案例结果}
        """
        print(f"原始查询: {query}")

        # 1. 查询改写
        rewritten_query = self.query_processor.rewrite_query(query)
        print(f"改写后查询: {rewritten_query}")

        # 2. 类别提取
        categories = self.query_processor.extract_category(rewritten_query)
        print(f"提取类别: {categories}")

        # 3. 混合检索
        if search_type == "hybrid":
            initial_results = self._hybrid_search(rewritten_query, categories, law_top_k=50, case_top_k=50)
        else:
            # 其他检索类型可以在这里扩展
            raise ValueError(f"不支持的检索类型: {search_type}")

        print(f"法律条文初始结果数: {len(initial_results['law'])}")
        print(f"案例初始结果数: {len(initial_results['case'])}")

        # 4. 分别重排序
        law_final_results = self.reranker.rerank(rewritten_query, initial_results["law"], top_k=law_top_k)
        case_final_results = self.reranker.rerank(rewritten_query, initial_results["case"], top_k=case_top_k)

        # 5. 添加额外信息
        for result in law_final_results:
            result["original_query"] = query
            result["rewritten_query"] = rewritten_query
            result["matched_categories"] = categories

        for result in case_final_results:
            result["original_query"] = query
            result["rewritten_query"] = rewritten_query
            result["matched_categories"] = categories

        return {
            "law": law_final_results,
            "case": case_final_results
        }

if __name__ == "__main__":
    # 修改初始化，传入两个数据库路径
    retriever = EnhancedLegalRetriever(
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
        print(f"查询：{query}")
        print(f"{'=' * 60}")

        results = retriever.retrieve(query, law_top_k=3, case_top_k=2, search_type="hybrid")

        # 显示法律条文结果
        print("\n法律条文结果：")
        for i, res in enumerate(results["law"], 1):
            print(f"\n第{i}条法律条文：")
            print(f"  内容：{res['text'][:100]}...")
            print(f"  来源：{res['law_name']}第{res['law_article_num']}条")
            print(f"  类别：{res.get('category', '未知')}")
            print(f"  相似度：{res['similarity']}")
            if 'bm25_score' in res:
                print(f"  BM25得分：{res['bm25_score']}")

        # 显示案例结果
        print("\n案例结果：")
        for i, res in enumerate(results["case"], 1):
            print(f"\n第{i}条案例：")
            print(f"  内容：{res['text'][:100]}...")
            print(f"  案例ID：{res.get('case_id', '未知')}")
            print(f"  相似度：{res['similarity']}")
            if 'bm25_score' in res:
                print(f"  BM25得分：{res['bm25_score']}")