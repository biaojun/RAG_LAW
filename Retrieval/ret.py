import os
import numpy as np
import jieba
import chromadb
import requests
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from zhipuai import ZhipuAI
import json
import time

# 智谱API配置
ZHIPU_API_KEY = "d03dac744afc4393b521ae5ebefbc7ac.yBJNhreyUJcVEfiK"


class ZhipuEmbeddingFunction:
    """智谱Embedding3嵌入函数"""

    def name(self):
        return "zhipu-embedding-3"

    def __call__(self, input: List[str]) -> List[List[float]]:

        if not input:
            print("警告: 输入为空，返回默认向量")
            # 返回一个默认的向量（根据你的向量维度调整）
            return [[0.0] * 1024]  # embedding-3通常是1024维

        print(f"处理后的输入: {input}")
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
        response = client.embeddings.create(
            model="embedding-3",
            input=input,
        )
        return [data_point.embedding for data_point in response.data]

    def embed_query(self, input: str) -> List[float]:
        return self.__call__(input)


class QueryProcessor:
    """查询处理器：负责查询改写和类别提取"""

    def __init__(self, api_key: str):
        self.client = ZhipuAI(api_key=api_key)
        self.law_categories = {
            "民法": ["婚姻", "继承", "合同", "物权", "侵权", "离婚", "财产", "债务"],
            "刑法": ["犯罪", "刑罚", "盗窃", "伤害", "抢劫", "诈骗", "毒品", "杀人"],
            "商法": ["公司", "破产", "证券", "保险", "票据", "企业", "股东", "清算"],
            "行政法": ["行政许可", "行政处罚", "行政复议", "政府", "行政机关"],
            "劳动法": ["劳动合同", "工资", "工伤", "劳动争议", "社保", "加班"],
            "知识产权": ["专利", "商标", "版权", "著作权", "发明", "侵权"]
        }

    def rewrite_query(self, original_query: str) -> str:
        """LLM查询改写"""
        prompt = f"""
        你是一名专业的法律语言优化助手，擅长将用户输入的查询改写为更加正式、准确、易于理解的法律问题表达形式。
        
        请对以下法律查询进行改写，使其：
        含义保持一致；
        语言更清晰、逻辑更连贯；
        可以适当添加或替换为常见的法律术语、同义表达或更规范的问法；
        输出中不要出现任何解释说明，只返回改写后的句子。
        
        原生成：{original_query}
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
        """LLM提取查询类别"""
        # 基于关键词的简单分类
        categories = []
        query_lower = query.lower()

        for category, keywords in self.law_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                categories.append(category)

        # 如果没匹配到，使用LLM进行智能分类
        if not categories:
            try:
                prompt = f"""
                请判断以下法律查询属于哪些法律类别（例如民法、刑法、商法、行政法、劳动法、知识产权等等）：
                查询：{query}
                只返回类别名称，用逗号分隔：
                """

                response = self.client.chat.completions.create(
                    model="glm-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50
                )
                llm_categories = response.choices[0].message.content.strip().split(',')
                categories = [cat.strip() for cat in llm_categories if cat.strip()]
            except Exception as e:
                print(f"LLM分类失败: {e}")
                categories = ["通用"]  # 默认类别

        return categories if categories else ["通用"]


class RerankModel:
    def __init__(self, api_key: str):
        self.client = ZhipuAI(api_key=api_key)
        self.api_key = api_key
        self.url = "https://open.bigmodel.cn/api/paas/v4/rerank"

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """对检索结果进行重排序"""
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
            response.raise_for_status()  # 如果请求失败会抛出异常

            result_data = response.json()

            # 解析重排序结果
            if 'results' in result_data:
                reranked_results = []
                for item in result_data["results"]:
                    # 根据返回的索引获取对应的原始文档
                    original_index = item["index"]
                    original_doc = documents[original_index]

                    # 添加重排序得分
                    reranked_doc = original_doc.copy()
                    reranked_doc["rerank_score"] = item.get("relevance_score", 0)
                    reranked_doc["rerank_index"] = item["index"]
                    reranked_results.append(reranked_doc)

                print(f"重排序完成，返回前{len(reranked_results)}个结果")
                return reranked_results
            else:
                print("重排序API返回格式异常，使用原始排序")
                return documents[:top_k]

        except requests.exceptions.RequestException as e:
            print(f"重排序请求失败: {e}，使用原始排序")
            return documents[:top_k]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"重排序结果解析失败: {e}，使用原始排序")
            return documents[:top_k]
        except Exception as e:
            print(f"重排序过程中发生未知错误: {e}，使用原始排序")
            return documents[:top_k]


class EnhancedLegalRetriever:
    """增强版法律检索器（按照流程图实现）"""

    def __init__(self, chroma_persist_dir: str = "./chroma_legal_db"):
        self.client = chromadb.PersistentClient(path=chroma_persist_dir)
        self.collection = self.client.get_collection(
            name="legal_articles",
            embedding_function=ZhipuEmbeddingFunction()
        )

        # 初始化各个组件
        self.query_processor = QueryProcessor(ZHIPU_API_KEY)
        self.reranker = RerankModel(ZHIPU_API_KEY)
        self.metadata_list = self._load_and_validate_metadata()

        if not self.metadata_list:
            raise ValueError("请先运行zhipu_store.py入库数据")

        self._init_bm25()
        self._init_category_index()

    def _load_and_validate_metadata(self) -> List[Dict]:
        """加载并验证元数据"""
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
                "category": meta.get("category", "通用")  # 添加类别字段
            })
        return metadata_list

    def _init_bm25(self):
        """初始化BM25"""
        self.corpus_texts = [item["text"] for item in self.metadata_list]
        self.tokenized_corpus = [list(jieba.cut(text)) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _init_category_index(self):
        """初始化类别索引"""
        self.category_index = {}
        for item in self.metadata_list:
            category = item["category"]
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(item)

    def _filter_by_category(self, categories: List[str], documents: List[Dict]) -> List[Dict]:
        """按类别过滤文档"""
        if "通用" in categories or not categories:
            return documents  # 不进行过滤

        filtered_docs = []
        for doc in documents:
            if doc.get("category") in categories:
                filtered_docs.append(doc)

        return filtered_docs if filtered_docs else documents  # 如果没有匹配，返回所有

    def _bm25_search(self, query: str, top_k: int = 50) -> List[Dict]:
        """BM25检索"""
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
                "similarity": round(scores[idx], 4),  # 为了兼容性
                "type": "keyword"
            }
            for idx in top_indices
        ]

    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量检索"""
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

    def _hybrid_search(self, query: str, top_k: int = 50) -> List[Dict]:
        """混合检索（BM25权重0.3 + 向量检索权重0.7）"""
        bm25_results = self._bm25_search(query, top_k=top_k * 2)
        vector_results = self._vector_search(query, top_k=top_k * 2)

        # 创建评分映射
        score_map = {}

        # BM25结果（权重0.3）
        if bm25_results:
            bm25_scores = [r["bm25_score"] for r in bm25_results]
            min_b, max_b = min(bm25_scores), max(bm25_scores)
            for r in bm25_results:
                norm_score = (r["bm25_score"] - min_b) / (max_b - min_b) if (max_b - min_b) else 0
                score_map[r["id"]] = 0.3 * norm_score

        # 向量检索结果（权重0.7）
        if vector_results:
            vec_scores = [r["similarity"] for r in vector_results]
            min_v, max_v = min(vec_scores), max(vec_scores)
            for r in vector_results:
                norm_score = (r["similarity"] - min_v) / (max_v - min_v) if (max_v - min_v) else 0
                score_map[r["id"]] = score_map.get(r["id"], 0) + 0.7 * norm_score

        # 按综合分数排序
        sorted_ids = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 构建最终结果
        final_results = []
        for doc_id, score in sorted_ids:
            # 查找文档信息
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
        """
        完整的检索流程（按照流程图）
        1. 查询改写 → 2. 类别提取 → 3. 类别过滤 → 4. 混合检索 → 5. 重排序
        """
        print(f"原始查询: {query}")

        # 1. 查询改写
        #rewritten_query = query
        rewritten_query = self.query_processor.rewrite_query(query)
        print(f"改写后查询: {rewritten_query}")

        # 2. 类别提取
        categories = self.query_processor.extract_category(rewritten_query)
        print(f"提取类别: {categories}")

        # 3. 混合检索（Top50）
        if search_type == "vector":
            initial_results = self._vector_search(rewritten_query, top_k=50)
        elif search_type == "keyword":
            initial_results = self._bm25_search(rewritten_query, top_k=50)
        else:  # hybrid
            initial_results = self._hybrid_search(rewritten_query, top_k=50)

        print(f"初步检索结果数: {len(initial_results)}")

        # 4. 按类别过滤
        # filtered_results = self._filter_by_category(categories, initial_results)
        # print(f"类别过滤后结果数: {len(filtered_results)}")

        # 5. 重排序（Top50 → Top3）
        final_results = self.reranker.rerank(rewritten_query, initial_results, top_k=top_k)

        # 添加检索过程信息
        for result in final_results:
            result["original_query"] = query
            result["rewritten_query"] = rewritten_query
            result["matched_categories"] = categories

        return final_results, rewritten_query


# 测试代码
if __name__ == "__main__":
    # 初始化增强版检索器
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

        start_time = time.time()
        results = retriever.retrieve(query, top_k=3, search_type="hybrid")
        end_time = time.time()

        print(f"检索耗时: {end_time - start_time:.2f}秒")

        for i, res in enumerate(results, 1):
            print(f"\n第{i}条结果：")
            print(f"  法条：{res['text']}")
            print(f"  来源：{res['law_name']}第{res['law_article_num']}条")
            print(f"  类别：{res.get('category', '未知')}")
            print(f"  相似度：{res['similarity']}")
            print(f"  检索类型：{res['type']}")
            if 'bm25_score' in res:
                print(f"  BM25得分：{res['bm25_score']}")
            print(f"  匹配类别：{res.get('matched_categories', [])}")