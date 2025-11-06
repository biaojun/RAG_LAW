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

### 启动服务

```zsh
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

启动成功后访问：`http://127.0.0.1:8000`

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

