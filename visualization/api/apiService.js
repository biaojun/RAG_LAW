// API 基础配置
const API_CONFIG = {
  baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  timeout: 30000, // 30秒超时
  headers: {
    'Content-Type': 'application/json',
  },
};

// API 错误处理类
class APIError extends Error {
  constructor(message, status, code) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.code = code;
  }
}

// 核心 API 服务类
class QAService {
  constructor(config = {}) {
    this.config = { ...API_CONFIG, ...config };
  }

  // 统一的请求方法
  async _request(endpoint, options = {}) {
    const url = `${this.config.baseURL}${endpoint}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          ...this.config.headers,
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new APIError(
          errorData.detail || `HTTP error! status: ${response.status}`,
          response.status,
          errorData.code
        );
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        throw new APIError('请求超时，请稍后重试', 408, 'TIMEOUT');
      }
      throw error;
    }
  }

  // 发送问题
  async askQuestion(question, options = {}) {
    return this._request('/api/ask', {
      method: 'POST',
      body: JSON.stringify({
        question: question.trim(),
        ...options,
      }),
    });
  }

  // 流式问答（适合长回答）
  async askQuestionStream(question, onChunk, options = {}) {
    const url = `${this.config.baseURL}/api/ask/stream`;
    const controller = new AbortController();
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...this.config.headers,
        },
        body: JSON.stringify({
          question: question.trim(),
          stream: true,
          ...options,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new APIError(`HTTP error! status: ${response.status}`, response.status);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              onChunk(data);
            } catch (e) {
              // 忽略解析错误，继续处理其他数据
            }
          }
        }
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        throw error;
      }
    }

    return () => controller.abort();
  }

  // 获取对话历史
  async getChatHistory(limit = 50, offset = 0) {
    return this._request(`/api/chat/history?limit=${limit}&offset=${offset}`);
  }

  // 删除对话记录
  async deleteChat(chatId) {
    return this._request(`/api/chat/${chatId}`, {
      method: 'DELETE',
    });
  }

  // 反馈接口（点赞/点踩）
  async sendFeedback(chatId, feedback, comment = '') {
    return this._request('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({
        chat_id: chatId,
        feedback, // 'like' 或 'dislike'
        comment,
      }),
    });
  }
}

// 创建单例实例
const qaService = new QAService();

export { qaService, QAService, APIError };