# RAG_LAW

一个基于 RAG（检索增强生成）的中文法律问答服务：
- 检索：本地 Chroma 向量库 + BM25 混合检索，支持重排序
- 生成：调用智谱 GLM 模型生成结构化中文回答
- 服务：FastAPI 对外提供 HTTP 接口

## 快速开始

### 环境准备

建议使用虚拟环境（可选）：

```zsh
python3 -m venv .venv
source .venv/bin/activate
```

安装依赖（按代码用到的库）：

```zsh
pip install fastapi uvicorn pydantic chromadb zhipuai requests numpy jieba rank_bm25 cn2an
```

设置智谱 API Key（务必换成你自己的密钥）：

```zsh
export ZHIPU_API_KEY="<你的智谱APIKey>"
```
(可选)设置向量数据库retrieval 模式
```zsh
# LAW_RAG_OPERATION_VECTOR 只用向量进行retrieval
# LAW_RAG_OPERATION_BM25 只用BM25进行retrieval
# 其他值或不设 混合模式
export LAW_RAG_OPERATION_MODE = LAW_RAG_OPERATION_VECTOR
```

### 启动服务

#### 方式一：启动 API 服务（推荐用于生产环境）

```zsh
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

启动成功后访问：`http://127.0.0.1:8000`

#### 方式二：启动可视化后端（带 Web 界面）

```zsh
python visualization/backend.py
```

或者：

```zsh
cd visualization
python backend.py
```

启动成功后访问：
- Web 界面：`http://localhost:8000` 或 `http://localhost:8000/static/rag.html`
- API 接口：`http://localhost:8000/api/ask`

**注意事项**：
- 后端会在启动时自动预初始化 RAG 系统（包括 BM25 索引和向量数据库），首次启动可能需要 10-30 秒
- 初始化完成后会显示 "RAG 系统预热完成，可以接受请求！"
- 预初始化后，所有后续请求都会直接使用已加载的模型，响应速度更快

### 调用示例

接口：`POST /api/ask`

请求体：`{ "question": "离婚冷静期多久？" }`

示例：

```zsh
curl -X POST "http://127.0.0.1:8000/api/ask" \
	-H "Content-Type: application/json" \
	-d '{"question":"离婚冷静期多久？"}'
```

返回字段说明：
- `answer`：模型生成的中文回答（包含结论与条文依据）
- `context`：用于溯源的检索结果摘要（法典名、条号、相似度等）

## 数据与向量库

- 项目根目录包含示例向量库目录 `chroma_legal_db/`，可直接使用。
- 如需重建或替换数据，可使用 `data_processing/chroma_store.py` 入库：

```zsh
# 法条入库（测试数据）
python data_processing/chroma_store.py law

# 案例入库（测试数据）
python data_processing/chroma_store.py law_case
```

注意：检索端（`Retrieval/ret.py`）默认读取 `./chroma_legal_db` 下集合 `legal_articles`；
而入库脚本默认写入 `law_chroma_db` 下集合 `law_articles`。如果你要用入库脚本重建数据，请确保二者目录与集合名对齐（修改检索端或入库脚本其一即可）。

## 主要文件

- `api_server.py`：FastAPI 服务入口，提供 `/api/ask` 接口
- `rag_pipeline.py`：RAG 主流程（检索→提示构建→生成）
- `Retrieval/ret.py`：增强检索器（BM25、向量检索、混合检索与重排序）
- `data_processing/chroma_store.py`：数据入库脚本（法条/案例）

## RAG Evaluation Metrics
### Context Recall
#### description
In legal RAG systems, context recall ensures retrieved legal provisions, cases, and details fully cover key grounds needed for answers, preventing distortions from missing information—critical in legal scenarios reliant on specific statutes. For example, when a user asks about legal consequences of having someone take the blame after a traffic accident, context recall checks if relevant provisions on traffic offenses, judicial interpretations defining such acts as fleeing, and harboring crimes are all retrieved. Omissions risk flawed responses; full retrieval guarantees legitimacy and comprehensiveness.
- Range of values: 0-1 (0 means no relevant key information is retrieved, and 1 means all necessary key information is completely retrieved.)
- Whether higher values are better: Yes. In legal scenarios, a higher context recall value indicates more comprehensive retrieved information, which minimizes answer errors caused by missing key provisions, cases, or interpretations, thereby enhancing the reliability and accuracy of legal opinions.
### Context Precision
#### Description
In legal RAG systems, context precision measures the proportion of retrieved context (legal provisions, cases, etc.) that is truly relevant to the user,s question, preventing irrelevant information from distorting responses—a key priority in legal scenarios, where unrelated legal details can cause misunderstandings or flawed reasoning. For example, when a user asks about adjusting excessive liquidated damages in contract disputes, the system should retrieve only relevant Civil Code provisions (e.g., Article 585) and interpretations. Context precision checks for irrelevant content (like tort liability clauses) in results: low precision means too much noise, harming response relevance; high precision ensures focus on core legal grounds.
- Range of values: 0-1 (0 indicates all retrieved information is irrelevant; 1 means all is fully relevant to the question.)
- Whether higher values are better: Yes. In legal contexts, higher context precision reduces redundant or unrelated legal information, minimizing interference and enhancing response accuracy, thus avoiding biased legal opinions from irrelevant provisions or cases.
### Context Entities Recall
#### Description
In legal RAG systems, context entities recall measures how many critical legal entities (such as specific statutes, case names, parties, legal terms, or key factual details) from the reference answer are successfully covered in the retrieved context. It ensures that essential entities—vital for accurate legal reasoning and response generation—are not missing from the retrieved information. For example, if a reference answer about a "breach of employment contract" mentions entities like "Labor Contract Law Article 39" and "employee misconduct"  context entities recall checks whether all these entities are present in the retrieved context. Low recall indicates missing key entities, which can weaken the legal basis of the response; high recall ensures all critical entities are included, strengthening the validity of the answer.
- Range of values: 0-1 (0 means no critical entities from the reference are retrieved; 1 indicates all critical entities are fully covered in the retrieved context.)
- Whether higher values are better: Yes. In legal scenarios, higher context entities recall ensures that all essential legal entities (statutes, cases, terms, etc.) needed for robust reasoning are included, reducing the risk of incomplete or unsupported legal conclusions and enhancing the reliability of the response.

### Faithfulness
#### Description
In legal RAG systems, faithfulness measures how factually consistent the generated response is with the retrieved legal contexts (such as statutes, case precedents, or judicial interpretations). It ensures all claims in the response are directly supported by the retrieved information, eliminating hallucinations or unsupported legal assertions. For example, if retrieved contexts cite "Criminal Law Article 291-2 (High-altitude Littering Crime)" and a case confirming "fines for serious circumstances," faithfulness checks whether the response,s claims (e.g., "high-altitude littering may result in imprisonment or fines") are fully inferred from these contexts. Low faithfulness means the response contains unsubstantiated legal statements, while high faithfulness confirms alignment with retrieved legal grounds.
- Range of values: 0-1 (0 indicates no claims in the response are supported by retrieved contexts; 1 means all claims are fully backed by the retrieved information.)
- Whether higher values are better: Yes. In legal scenarios, higher faithfulness is critical—it ensures the response avoids fabricated legal details or misinterpretations, upholding the credibility and validity of legal advice and preventing misleading outcomes for users.
### Response Relevancy
#### Description
In legal RAG systems, response relevancy measures how closely the generated response addresses the core of the user,s legal query, focusing on the alignment between the response content and the user,s specific legal concerns (e.g., liability, penalties, remedies). It ensures the response directly targets the user,s question without irrelevant digressions, such as unrelated legal provisions or non-pertinent case details. For example, if a user asks "Whether shoplifting goods worth 1,000 yuan constitutes a crime," response relevancy checks if the response focuses on "theft conviction standards," "criminal vs. administrative liability," and "mitigating measures"—rather than diverging into topics like shop operation regulations. Low relevancy means the response fails to address the user,s key concern, while high relevancy ensures targeted, purposeful answers.
- Range of values: 0-1 (0 indicates the response is completely unrelated to the user,s query; 1 means the response fully addresses the core legal issue raised by the user.)
- Whether higher values are better: Yes. In legal scenarios, higher response relevancy is essential—it ensures users receive direct guidance for their specific legal predicament, avoiding confusion from irrelevant information and enabling informed decision-making.
