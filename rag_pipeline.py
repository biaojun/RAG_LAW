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
    from RAG_LAW.Retrieval.ret import ZhipuAiClient  # type: ignore
except Exception:
    try:
        from Retrieval.ret import ZhipuAiClient  # type: ignore
    except Exception as e:
        raise ImportError(
            "未能通过 ret.py 获取 ZhipuAI，请确保 ret.py 可被导入（包含对 zai 的可用依赖）。"
        ) from e


_retriever: Optional[EnhancedLegalRetriever] = None
_client: Optional[ZhipuAiClient] = None


def _get_retriever() -> EnhancedLegalRetriever:
    global _retriever
    if _retriever is None:
        # 若本地向量库未初始化，ret.py 内部会抛出清晰的报错提示
        _retriever = EnhancedLegalRetriever()
    return _retriever


def _get_client() -> ZhipuAiClient:
    global _client
    if _client is None:
        api_key = os.environ.get("ZHIPU_API_KEY", ZHIPU_API_KEY)
        _client = ZhipuAiClient(api_key=api_key)
    return _client


def initialize_rag_system():
    """
    预初始化 RAG 系统（包括 retriever 和 client）。
    建议在应用启动时调用，避免首次请求时的初始化延迟。
    """
    print("正在初始化 RAG 系统...")
    try:
        retriever = _get_retriever()
        print("✓ Retriever 初始化完成（包括 BM25 索引）")
    except Exception as e:
        print(f"✗ Retriever 初始化失败: {e}")
        raise
    
    try:
        client = _get_client()
        print("✓ Zhipu Client 初始化完成")
    except Exception as e:
        print(f"✗ Zhipu Client 初始化失败: {e}")
        raise
    
    print("RAG 系统初始化完成！")
    return retriever, client


def _format_context(passages: List[Dict]) -> str:
    """将检索到的法条和案例整理为可读上下文，供大模型参考。"""
    lines: List[str] = []
    law_counter = 1
    case_counter = 1
    
    for p in passages:
        text = (p.get("text") or "").strip()
        # 控制单条上下文长度，避免提示过长（可按需调整）
        if len(text) > 800:
            text = text[:800] + "..."
        
        # 判断数据类型
        data_type = p.get("data_type", "law")
        if data_type == "case":
            case_id = p.get("case_id", "未知案例")
            lines.append(f"[案例{case_counter}] 案例ID: {case_id}\n{text}")
            case_counter += 1
        else:
            law_name = p.get("law_name", "未知法典")
            art = p.get("law_article_num", "?")
            lines.append(f"[法条{law_counter}] 《{law_name}》第{art}条：\n{text}")
            law_counter += 1
    
    return "\n\n".join(lines)


def _build_prompt(question: str, passages: List[Dict]) -> Tuple[str, str]:
    """
    返回 (system_prompt, user_prompt)。
    将检索到的法条和案例拼接到 user 部分，约束回答格式、引用规则与语言要求。
    """
    context_block = _format_context(passages) if passages else "(未检索到相关法条和案例)"

    system_prompt = (
        "你是一名专业的中文法律助手，擅长依据中国法律条文和相关案例给出准确、清晰、可执行的解答。\n"
        "- 你的语言必须为简体中文。\n"
        "- 优先依据提供的法条作答，逐条对齐问题进行推理与说明。\n"
        "- 明确引用出处：用《法典名》第X条标注法律条文，用案例ID标注参考案例。\n"
        "- 如果提供了相关案例，可以参考案例来辅助说明法律的实际应用。\n"
        "- 若法条和案例不足以直接回答，请指出不确定性并给出合规的下一步建议（如咨询律师/补充材料）。\n"
        "- 禁止编造不存在的条文或案例。\n"
    )

    user_prompt = (
f"用户问题：\n{question}\n\n"
f"检索到的法条与案例片段（可能包含无关内容，仅供参考）：\n{context_block}\n\n"
"请根据以下要求直接生成最终回答：\n"
"1) 先回答用户的问题。\n"
"2) 仅引用真正与问题具有直接法律关联的法条：以“《法典名》第X条：…… → 关联理由”格式呈现。\n"
"   - 若检索内容中存在不相关或冲突的条文，请完全忽略。\n"
"   - 若检索内容均与问题无关，则请基于你自身掌握的一般法律知识回答，但不得编造不存在的条号，必要时可仅描述原则而不写具体条号。\n"
"3) 若有相关案例，仅说明其如何支持条文含义；无关案例请忽略。\n"
"4) 禁止输出与法律无关的内容，禁止补充检索中不存在的法规内容。\n"


    )

    return system_prompt, user_prompt


def retrieve_and_generate(
    question: str,
    *,
    top_k: int = 5,
    top_p: float = 1.0,
    search_type: str = "hybrid",
    model: str = "glm-4",
    temperature: float = 1.0,
    max_tokens: int = 65536,
) -> Tuple[str, List[Dict]]:
    """
    完整 RAG：检索 -> 提示构建 -> 大模型生成。

    返回：
    - answer: 最终回答（中文）
    - context: 引用的检索结果（便于前端展示溯源）
    """
    retriever = _get_retriever()

    # 召回
    results = retriever.retrieve(
        query=question, law_top_k=3, case_top_k=2, search_type=search_type
    )
    
    # 合并法律条文和案例
    all_results = results.get("law", []) + results.get("case", [])
    rewritten_query = all_results[0]["rewritten_query"] if all_results else question

    # 构建提示
    # sys_prompt, usr_prompt = _build_prompt(question, all_results)
    sys_prompt, usr_prompt = _build_prompt(rewritten_query, all_results)
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
            temperature=temperature
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        # 降级策略：不阻断整体流程，返回简短错误提示与上下文
        answer = f"抱歉，生成过程中出现异常：{e}。以下为检索到的相关法条供参考。"



# {'law': [{...}, {...}, {...}], 'case': [{...}, {...}]}
# # special variables:
# # function variables:
# # 'law': [{'id': '中华人民共和国民法典_1082', 'text': '中华人民共和国民法典第1082条：女方在怀孕期间、分娩后一年内或者终止妊娠后六个月内，男方不得提出离婚；但是，女方提出离婚或者人民法院认为确有必要受理男方离婚请求的除外。', 'law_name': '中华人民共和国民法典', 'law_article_num': 1082, 'category': "['怀孕离婚', '男方限制', '终止妊娠', '离婚请求']", 'data_type': 'law', 'bm25_score': 11.816, 'similarity': 0.098, 'type': 'hybrid', 'rerank_score': 1.0, 'rerank_index': 12, 'original_query': '离婚冷静期多久？', 'rewritten_query': '离婚诉讼中的冷静期限是多久？', 'matched_categories': [...]}, {'id': '中华人民共和国民法典_1077', 'text': '中华人民共和国民法典第1077条：自婚姻登记机关收到离婚登记申请之日起三十日内，任何一方不愿意离婚的，可以向婚姻登记机关撤回离婚登记申请。前款规定期限届满后三十日内，双方应当亲自到婚姻登记机关申请发给离婚证；未申请的，视为撤回离婚登记申请。', 'law_name': '中华人民共和国民法典', 'law_article_num': 1077, 'category': "['离婚登记', '撤回申请', '期限届满', '申请离婚证']", 'data_type': 'law', 'bm25_score': 15.9366, 'similarity': 0.267, 'type': 'hybrid', 'rerank_score': 1.0, 'rerank_index': 1, 'original_query': '离婚冷静期多久？', 'rewritten_query': '离婚诉讼中的冷静期限是多久？', 'matched_categories': [...]}, {'id': '中华人民共和国民事诉讼法_151', 'text': '中华人民共和国民事诉讼法第151条：人民法院对公开审理或者不公开审理的案件，一律公开宣告判决。当庭宣判的，应当在十日内发送判决书；定期宣判的，宣判后立即发给判决书。宣告判...上诉期限和上诉的法院。宣告离婚判决，必须告知当事人在判决发生法律效力前不得另行结婚。', 'law_name': '中华人民共和国民事诉讼法', 'law_article_num': 151, 'category': "['公开宣判', '判决书', '上诉权利', '离婚判决']", 'data_type': 'law', 'bm25_score': 11.1579, 'similarity': 0.071, 'type': 'hybrid', 'rerank_score': 0.9999999, 'rerank_index': 14, 'original_query': '离婚冷静期多久？', 'rewritten_query': '离婚诉讼中的冷静期限是多久？', 'matched_categories': [...]}]
# # 'case': [{'id': '3438', 'text': '[案名]杨某乙诉杨某甲离婚后财产纠纷案\n[基本案情] 杨某乙与杨某甲于2001年9月相识，2003年10月15日登记结婚，双方均系再婚。婚后未生育子女。在婚后的共同生活中...系存续期间，夫妻一方以个人财产购置的房屋等不动产仍应归个人所有，不属于夫妻共同财产。', 'case_id': '3438', 'data_type': 'case', 'similarity': 0.1387, 'bm25_score': 0, 'type': 'hybrid', 'rerank_score': 0.9999993, 'rerank_index': 33, 'original_query': '离婚冷静期多久？', 'rewritten_query': '离婚诉讼中的冷静期限是多久？', 'matched_categories': [...]}, {'id': '2147', 'text': '[案名]徐某贵诉张某琴离婚纠纷案\n[基本案情] 徐某贵与张某琴于1988年相识，2014年4月经他人介绍确定男女朋友关系，2015年5月开始同居生活，2016年2月自愿登...的，人民法院应当在判决准许双方离婚的同时依法判令过错方对无过错方承担损害赔偿责任。 ', 'case_id': '2147', 'data_type': 'case', 'similarity': 0.4003, 'bm25_score': 0, 'type': 'hybrid', 'rerank_score': 0.99999905, 'rerank_index': 8, 'original_query': '离婚冷静期多久？', 'rewritten_query': '离婚诉讼中的冷静期限是多久？', 'matched_categories': [...]}]
# # len(): 2
    # 精简 context，避免过多字段
    # 精简 context,避免过多字段
    context_law = [
        {
            "id": r.get("id"),
            "law_name": r.get("law_name"),
            "law_article_num": r.get("law_article_num"),
            "similarity": r.get("similarity"),
            "rerank_score": r.get("rerank_score"),
            "snippet": (r.get("text") or "").strip(),
            "data_type": "law"
        }
        for r in results.get("law", [])
    ]

    context_case = [
        {
            "id": r.get("id"),
            "case_id": r.get("case_id"),
            "similarity": r.get("similarity"),
            "rerank_score": r.get("rerank_score"),
            "snippet": (r.get("text") or "").strip(),
            "data_type": "case"
        }
        for r in results.get("case", [])
    ]

    # 合并法律条文和案例作为完整上下文
    context = {
        "law": context_law,
        "case": context_case
    }

    return answer, context


__all__ = ["retrieve_and_generate", "initialize_rag_system"]
