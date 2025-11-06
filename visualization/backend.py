from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import json
import os
import sys
from datetime import datetime
import uuid
import glob
import re

app = FastAPI(title="本地问答系统", version="1.0.0")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 基准目录：与本文件同级，避免受当前工作目录影响
BASE_DIR = os.path.dirname(__file__)

# 数据与静态目录（均相对于本文件目录）
DATA_DIR = os.path.join(BASE_DIR, "chat_data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 优先尝试导入 RAG 流程；失败则在运行时回退到本地演示回答
RAG_AVAILABLE = False
try:
    from rag_pipeline import retrieve_and_generate as rag_retrieve_and_generate  # 从仓库根运行
    RAG_AVAILABLE = True
except Exception:
    try:
        # 兼容：从本文件的上级目录作为仓库根加入路径后再导入
        REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
        if REPO_ROOT not in sys.path:
            sys.path.append(REPO_ROOT)
        from rag_pipeline import retrieve_and_generate as rag_retrieve_and_generate
        RAG_AVAILABLE = True
    except Exception:
        RAG_AVAILABLE = False


class QuestionRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class QuestionResponse(BaseModel):
    answer: str
    context: list
    message_id: str
    timestamp: str
    conversation_id: str


def parse_message_from_file(filepath):
    """从txt文件中解析消息内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析文件内容
        message_id = re.search(r'消息ID: (.+)', content)
        message_type = re.search(r'类型: (.+)', content)
        timestamp = re.search(r'时间: (.+)', content)
        question = re.search(r'问题: (.+)', content)
        answer = re.search(r'回答: (.+)', content)

        return {
            'id': message_id.group(1) if message_id else os.path.basename(filepath),
            'type': message_type.group(1) if message_type else 'unknown',
            'timestamp': timestamp.group(1) if timestamp else '',
            'question': question.group(1) if question else '',
            'answer': answer.group(1) if answer else '',
            'filepath': filepath
        }
    except Exception as e:
        print(f"解析文件失败 {filepath}: {e}")
        return None


def save_to_txt(message_data: dict, message_type: str, conversation_id: str = None):
    """将消息保存为txt文件"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        message_id = message_data.get('message_id', str(uuid.uuid4())[:8])

        # 使用会话ID作为目录
        if conversation_id:
            conv_dir = os.path.join(DATA_DIR, conversation_id)
        else:
            conv_dir = os.path.join(DATA_DIR, "default")

        os.makedirs(conv_dir, exist_ok=True)

        filename = f"{timestamp}_{message_type}_{message_id}.txt"
        filepath = os.path.join(conv_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"消息ID: {message_id}\n")
            f.write(f"类型: {message_type}\n")
            f.write(f"会话ID: {conversation_id or 'default'}\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")

            if message_type == "question":
                f.write(f"问题: {message_data['question']}\n")
            elif message_type == "answer":
                f.write(f"问题: {message_data['question']}\n")
                f.write(f"回答: {message_data['answer']}\n")
                f.write(f"上下文: {', '.join(message_data.get('context', []))}\n")

        print(f"消息已保存: {filepath}")
        return message_id
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None


def generate_answer(question: str) -> tuple:
    """生成回答的逻辑"""
    answers = {
        "你好": "你好！我是本地问答助手，很高兴为你服务。",
        "你是谁": "我是一个部署在本地的问答系统，可以回答你的各种问题。",
        "今天天气怎么样": "我无法获取实时天气信息，建议你查看天气预报应用。",
        "谢谢": "不客气！如果有其他问题，随时问我。",
    }

    for key, value in answers.items():
        if key in question:
            return value, [f"匹配关键词: {key}"]

    default_answer = f"我已经收到你的问题：'{question}'。这是一个本地部署的问答系统，你可以根据需要自定义回答逻辑。"
    return default_answer, ["系统默认回答"]


def rag_answer(question: str) -> tuple:
    """调用真实的 RAG 流程，并将上下文格式化为可展示的字符串列表。"""
    try:
        answer, ctx = rag_retrieve_and_generate(question)
        display_ctx = []
        for c in (ctx or []):
            try:
                law = c.get("law_name") or "未知法典"
                art = c.get("law_article_num") or "?"
                display_ctx.append(f"《{law}》第{art}条")
            except Exception:
                continue
        return answer, display_ctx
    except Exception as e:
        # 回退：不影响前端使用
        fallback_ans, fallback_ctx = generate_answer(question)
        safe_err = str(e)
        return f"{fallback_ans}\n\n[提示] RAG 调用异常，已回退本地回答：{safe_err}", fallback_ctx


@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """处理问答请求"""
    try:
        print(f"收到问题: {request.question}, 会话ID: {request.conversation_id}")

        message_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # 如果没有提供会话ID，创建一个新的
        conversation_id = request.conversation_id or str(uuid.uuid4())[:8]
        print(f"使用的会话ID: {conversation_id}")

        # 保存问题到txt文件
        question_data = {
            "question": request.question,
            "message_id": message_id,
            "timestamp": timestamp
        }
        save_to_txt(question_data, "question", conversation_id)

        # 生成回答：优先调用 RAG，失败或不可用时回退到本地演示逻辑
        if RAG_AVAILABLE:
            answer, context = rag_answer(request.question)
        else:
            answer, context = generate_answer(request.question)

        # 准备响应数据
        response_data = {
            "answer": answer,
            "context": context,
            "message_id": message_id,
            "timestamp": timestamp,
            "conversation_id": conversation_id
        }

        # 保存回答到txt文件
        answer_data = {
            "question": request.question,
            "answer": answer,
            "context": context,
            "message_id": message_id,
            "timestamp": timestamp
        }
        save_to_txt(answer_data, "answer", conversation_id)

        return QuestionResponse(**response_data)

    except Exception as e:
        print(f"处理问题出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理问题时出错: {str(e)}")


@app.get("/api/history")
async def get_chat_history(limit: int = 50, offset: int = 0):
    """获取对话历史（从保存的文件中读取）"""
    try:
        # 获取所有对话文件
        pattern = os.path.join(DATA_DIR, "**", "*_answer_*.txt")
        answer_files = glob.glob(pattern, recursive=True)

        # 解析所有对话
        all_chats = []
        for filepath in answer_files:
            chat_data = parse_message_from_file(filepath)
            if chat_data and chat_data.get('question') and chat_data.get('answer'):
                all_chats.append(chat_data)

        # 按时间排序（最新的在前面）
        all_chats.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # 分页处理
        total = len(all_chats)
        start_idx = offset
        end_idx = offset + limit
        paginated_chats = all_chats[start_idx:end_idx]

        # 格式化返回数据
        history = []
        for chat in paginated_chats:
            history.append({
                "id": chat['id'],
                "question": chat['question'],
                "answer": chat['answer'],
                "timestamp": chat['timestamp'],
                "type": "qa"
            })

        return {
            "history": history,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录时出错: {str(e)}")


@app.get("/api/conversations")
async def get_conversations():
    """获取所有会话列表"""
    try:
        conversations = []

        # 获取所有会话目录
        if os.path.exists(DATA_DIR):
            for item in os.listdir(DATA_DIR):
                item_path = os.path.join(DATA_DIR, item)
                if os.path.isdir(item_path):
                    # 查找该会话中的第一个问题文件
                    first_question = None
                    question_files = glob.glob(os.path.join(item_path, "*_question_*.txt"))
                    if question_files:
                        first_question_file = min(question_files)
                        with open(first_question_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            question_match = re.search(r'问题: (.+)', content)
                            if question_match:
                                first_question = question_match.group(1)

                    # 获取创建时间
                    created_time = ""
                    if question_files:
                        with open(min(question_files), 'r', encoding='utf-8') as f:
                            content = f.read()
                            time_match = re.search(r'时间: (.+)', content)
                            if time_match:
                                created_time = time_match.group(1)

                    conversations.append({
                        "id": item,
                        "title": first_question or "新对话",
                        "created_at": created_time
                    })

        # 按创建时间排序
        conversations.sort(key=lambda x: x["created_at"], reverse=True)

        return {"conversations": conversations}

    except Exception as e:
        print(f"获取会话列表出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话列表时出错: {str(e)}")


@app.get("/api/conversation/{conversation_id}")
async def get_conversation_messages(conversation_id: str):
    """获取特定会话的所有消息"""
    try:
        conversation_dir = os.path.join(DATA_DIR, conversation_id)
        if not os.path.exists(conversation_dir):
            return {"messages": []}

        # 获取会话中的所有文件
        all_files = glob.glob(os.path.join(conversation_dir, "*.txt"))
        messages = []

        for filepath in sorted(all_files):
            chat_data = parse_message_from_file(filepath)
            if chat_data:
                messages.append({
                    "id": chat_data['id'],
                    "type": chat_data['type'],
                    "content": chat_data['question'] if chat_data['type'] == 'question' else chat_data['answer'],
                    "timestamp": chat_data['timestamp']
                })

        return {"messages": messages}

    except Exception as e:
        print(f"获取会话消息出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话消息时出错: {str(e)}")


@app.delete("/api/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除整个会话"""
    try:
        conversation_dir = os.path.join(DATA_DIR, conversation_id)
        if not os.path.exists(conversation_dir):
            raise HTTPException(status_code=404, detail=f"未找到会话: {conversation_id}")

        # 删除整个目录
        import shutil
        shutil.rmtree(conversation_dir)

        return {
            "status": "success",
            "message": f"已删除会话: {conversation_id}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话时出错: {str(e)}")


# 提供静态文件服务（使用绝对路径，挂载到 /static 以避免与根路由冲突）
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    print("启动本地问答系统...")
    print("访问地址: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)