import os
from typing import List, Dict, Tuple, Optional

# 复用你现有的检索与智谱客户端实现
try:
    # 优先通过包路径导入（如果从仓库根目录运行，RAG_LAW 会作为命名空间包生效）
    from RAG_LAW.Retrieval.ret import EnhancedLegalRetriever, ZHIPU_API_KEY
except Exception:
    # 回退到相对路径导入（如果当前工作目录就在 RAG_LAW）
    from Retrieval.ret import EnhancedLegalRetriever, ZHIPU_API_KEY  # type: ignore

# 为避免直接依赖第三方包导入错误，这里复用 ret.py 内已导入并暴露在模块命名空间的 ZhipuAI
try:
    # 与上方的导入路径保持一致
    from RAG_LAW.Retrieval.ret import ZhipuAI  # type: ignore
except Exception:
    try:
        from Retrieval.ret import ZhipuAI  # type: ignore
    except Exception as e:
        raise ImportError(
            "未能通过 ret.py 获取 ZhipuAI，请确保 ret.py 可被导入（包含对 zai 的可用依赖）。"
        ) from e


_retriever: Optional[EnhancedLegalRetriever] = None
_client: Optional[ZhipuAI] = None


def _get_retriever() -> EnhancedLegalRetriever:
    global _retriever
    if _retriever is None:
        # 若本地向量库未初始化，ret.py 内部会抛出清晰的报错提示
        _retriever = EnhancedLegalRetriever()
    return _retriever


def _get_client() -> ZhipuAI:
    global _client
    if _client is None:
        api_key = os.environ.get("ZHIPU_API_KEY", ZHIPU_API_KEY)
        _client = ZhipuAI(api_key=api_key)
    return _client


def _format_context(passages: List[Dict]) -> str:
    """将检索到的法条整理为可读上下文，供大模型参考。"""
    lines: List[str] = []
    for i, p in enumerate(passages, 1):
        law_name = p.get("law_name", "未知法典")
        art = p.get("law_article_num", "?")
        text = (p.get("text") or "").strip()
        # 控制单条上下文长度，避免提示过长（可按需调整）
        if len(text) > 800:
            text = text[:800] + "..."
        lines.append(f"[法条{i}] 《{law_name}》第{art}条：\n{text}")
    return "\n\n".join(lines)


def _build_prompt(question: str, passages: List[Dict]) -> Tuple[str, str]:
    """
    返回 (system_prompt, user_prompt)。
    将检索到的法条拼接到 user 部分，约束回答格式、引用规则与语言要求。
    """
    context_block = _format_context(passages) if passages else "(未检索到相关法条)"

    system_prompt = (
        "你是一名专业的中文法律助手，擅长依据中国法律条文给出准确、清晰、可执行的解答。\n"
        "- 你的语言必须为简体中文。\n"
        "- 优先依据提供的法条作答，逐条对齐问题进行推理与说明。\n"
        "- 明确引用出处：用《法典名》第X条标注。\n"
        "- 若法条不足以直接回答，请指出不确定性并给出合规的下一步建议（如咨询律师/补充材料）。\n"
        "- 禁止编造不存在的条文或案例。\n"
    )

    user_prompt = (
        f"用户问题：\n{question}\n\n"
        f"检索到的相关法条与内容（优先作为依据）：\n{context_block}\n\n"
        "请完成以下任务：\n"
        "1) 先给出凝练直接的结论性回答（1-3句）。\n"
        "2) 列出支撑依据：逐条引用对应《法典名》第X条并用简短语句解释其与问题的关联。\n"
        "3) 如仍有不确定性或需要补充信息，给出合规的建议与下一步。\n"
        "4) 全文保持中文，避免空洞套话，禁止生成与法律无关的内容。\n"
    )

    return system_prompt, user_prompt


def retrieve_and_generate(
    question: str,
    *,
    top_k: int = 3,
    search_type: str = "hybrid",
    model: str = "glm-4",
    temperature: float = 1.0,
    max_tokens: int = 1024,
) -> Tuple[str, List[Dict]]:
    """
    完整 RAG：检索 -> 提示构建 -> 大模型生成。

    返回：
    - answer: 最终回答（中文）
    - context: 引用的检索结果（便于前端展示溯源）
    """
    retriever = _get_retriever()

    # 召回
    results,rewrite_query = retriever.retrieve(
        query=question, top_k=top_k, search_type=search_type
    )

    # 构建提示
    # sys_prompt, usr_prompt = _build_prompt(question, results)
    sys_prompt, usr_prompt = _build_prompt(rewrite_query, results)
    print(usr_prompt)  # 调试时可查看完整提示内容
    # 生成
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        # 降级策略：不阻断整体流程，返回简短错误提示与上下文
        answer = f"抱歉，生成过程中出现异常：{e}。以下为检索到的相关法条供参考。"

    # 精简 context，避免过多字段
    context = [
        {
            "id": r.get("id"),
            "law_name": r.get("law_name"),
            "law_article_num": r.get("law_article_num"),
            "similarity": r.get("similarity"),
            "rerank_score": r.get("rerank_score"),
            "snippet": (r.get("text") or "").strip(),
        }
        for r in results
    ]

    return answer, context


__all__ = ["retrieve_and_generate"]
