import React, { useState } from 'react';
import { useQAChat } from './useQAChat';

const ChatComponent = () => {
  const [input, setInput] = useState('');
  const {
    loading,
    error,
    chatHistory,
    askQuestion,
    askQuestionStream,
    stopStreaming,
    sendFeedback,
    clearHistory,
  } = useQAChat();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    try {
      // 普通请求
      // await askQuestion(input);
      
      // 或者使用流式请求
      await askQuestionStream(input);
      setInput('');
    } catch (err) {
      // 错误已经在 hook 中处理
      console.error('提问失败:', err);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-container">
      {/* 错误显示 */}
      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* 对话历史 */}
      <div className="chat-history">
        {chatHistory.map((item) => (
          <div key={item.id} className="message-pair">
            <div className="question">
              <strong>用户:</strong> {item.question}
            </div>
            <div className="answer">
              <strong>助手:</strong> 
              {item.answer}
              {item.isStreaming && <span className="typing-indicator">...</span>}
              
              {/* 反馈按钮 */}
              {!item.isStreaming && !item.error && (
                <div className="feedback-buttons">
                  <button 
                    onClick={() => sendFeedback(item.id, 'like')}
                    className={item.feedback === 'like' ? 'active' : ''}
                  >
                    👍
                  </button>
                  <button 
                    onClick={() => sendFeedback(item.id, 'dislike')}
                    className={item.feedback === 'dislike' ? 'active' : ''}
                  >
                    👎
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* 输入区域 */}
      <form onSubmit={handleSubmit} className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="请输入您的问题..."
          disabled={loading}
          rows={3}
        />
        <div className="button-group">
          <button 
            type="submit" 
            disabled={!input.trim() || loading}
          >
            {loading ? '发送中...' : '发送'}
          </button>
          {loading && (
            <button type="button" onClick={stopStreaming}>
              停止
            </button>
          )}
          {chatHistory.length > 0 && (
            <button type="button" onClick={clearHistory}>
              清空对话
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default ChatComponent;