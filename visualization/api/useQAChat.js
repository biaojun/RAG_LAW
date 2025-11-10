import { useState, useCallback, useRef } from 'react';
import { qaService, APIError } from './apiService';

// 自定义问答 Hook
export const useQAChat = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const abortControllerRef = useRef(null);

  // 发送问题
  const askQuestion = useCallback(async (question, options = {}) => {
    setLoading(true);
    setError(null);

    try {
      const response = await qaService.askQuestion(question, options);
      
      const newQAPair = {
        id: Date.now().toString(),
        question,
        answer: response.answer,
        context: response.context || [],
        timestamp: new Date().toISOString(),
      };

      setChatHistory(prev => [...prev, newQAPair]);
      return newQAPair;
    } catch (err) {
      const errorMessage = err instanceof APIError 
        ? err.message 
        : '网络错误，请稍后重试';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // 流式问答
  const askQuestionStream = useCallback(async (question, options = {}) => {
    setLoading(true);
    setError(null);

    const messageId = Date.now().toString();
    const newQAPair = {
      id: messageId,
      question,
      answer: '',
      context: [],
      timestamp: new Date().toISOString(),
      isStreaming: true,
    };

    setChatHistory(prev => [...prev, newQAPair]);

    const onChunk = (chunk) => {
      setChatHistory(prev => prev.map(item => {
        if (item.id === messageId) {
          return {
            ...item,
            answer: item.answer + (chunk.content || ''),
            context: chunk.context || item.context,
            isStreaming: !chunk.is_finished,
          };
        }
        return item;
      }));
    };

    try {
      abortControllerRef.current = await qaService.askQuestionStream(
        question, 
        onChunk, 
        options
      );
    } catch (err) {
      const errorMessage = err instanceof APIError 
        ? err.message 
        : '网络错误，请稍后重试';
      setError(errorMessage);
      
      // 标记消息为错误状态
      setChatHistory(prev => prev.map(item => 
        item.id === messageId 
          ? { ...item, error: true, isStreaming: false }
          : item
      ));
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // 停止流式响应
  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current();
      abortControllerRef.current = null;
      setLoading(false);
      
      // 更新最后一条消息状态
      setChatHistory(prev => prev.map((item, index) => 
        index === prev.length - 1 && item.isStreaming
          ? { ...item, isStreaming: false, stopped: true }
          : item
      ));
    }
  }, []);

  // 发送反馈
  const sendFeedback = useCallback(async (chatId, feedback, comment = '') => {
    try {
      await qaService.sendFeedback(chatId, feedback, comment);
      
      // 更新本地状态
      setChatHistory(prev => prev.map(item =>
        item.id === chatId
          ? { ...item, feedback, feedbackComment: comment }
          : item
      ));
    } catch (err) {
      console.error('发送反馈失败:', err);
      throw err;
    }
  }, []);

  // 清空对话历史
  const clearHistory = useCallback(() => {
    setChatHistory([]);
    setError(null);
  }, []);

  // 删除单条对话
  const deleteMessage = useCallback(async (chatId) => {
    try {
      await qaService.deleteChat(chatId);
      setChatHistory(prev => prev.filter(item => item.id !== chatId));
    } catch (err) {
      console.error('删除消息失败:', err);
      throw err;
    }
  }, []);

  return {
    // 状态
    loading,
    error,
    chatHistory,
    
    // 方法
    askQuestion,
    askQuestionStream,
    stopStreaming,
    sendFeedback,
    clearHistory,
    deleteMessage,
    
    // 工具方法
    hasHistory: chatHistory.length > 0,
  };
};