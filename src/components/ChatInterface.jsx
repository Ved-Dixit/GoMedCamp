import React, { useState, useEffect, useRef, useCallback } from 'react';
import api from '../services/api'; // Your API service
import '../App.css'; // Ensure styles are here

const ChatInterface = ({ connectionId, currentUser, chatWithName, chatWithUserType, campName, onClose }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [loadingMessages, setLoadingMessages] = useState(true);
  const [sendingMessage, setSendingMessage] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null); // To scroll to bottom

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const fetchMessages = useCallback(async () => {
    if (!connectionId) return;
    setLoadingMessages(true);
    try {
      const response = await api.get(`/chat/conversation/${connectionId}/messages`);
      setMessages(response.data || []);
      setError('');
    } catch (err) {
      setError('Failed to load messages. ' + (err.response?.data?.error || err.message));
      console.error("Fetch messages error:", err);
    } finally {
      setLoadingMessages(false);
    }
  }, [connectionId]);

  useEffect(() => {
    fetchMessages();
    // Optional: Implement polling for new messages
    const intervalId = setInterval(fetchMessages, 15000); // Poll every 15 seconds
    return () => clearInterval(intervalId);
  }, [fetchMessages]);

  useEffect(scrollToBottom, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!newMessage.trim() || !connectionId || sendingMessage) return;

    setSendingMessage(true);
    setError('');
    try {
      const response = await api.post(`/chat/conversation/${connectionId}/message`, {
        text: newMessage.trim(),
      });
      // Add new message to state immediately for better UX, or rely on next poll
      // setMessages(prevMessages => [...prevMessages, response.data.chatMessage]);
      setNewMessage('');
      fetchMessages(); // Or rely on polling, but this gives immediate feedback
    } catch (err) {
      setError('Failed to send message. ' + (err.response?.data?.error || err.message));
      console.error("Send message error:", err);
    } finally {
      setSendingMessage(false);
    }
  };

  return (
    <div className="chat-interface-container">
      <div className="chat-header">
        {onClose && (
          <button onClick={onClose} className="back-button" title="Close Chat">
            &larr;
          </button>
        )}
        <div className="chat-title-info">
          <h2>Chat with {chatWithName} ({chatWithUserType})</h2>
          {campName && <p className="camp-context">Regarding Camp: {campName}</p>}
        </div>
      </div>

      {error && <div className="error-message" style={{ margin: '10px' }}>{error}</div>}

      <div className="messages-area">
        {loadingMessages && messages.length === 0 && <p>Loading messages...</p>}
        {!loadingMessages && messages.length === 0 && <p>No messages yet. Start the conversation!</p>}
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`message-bubble ${msg.sender_id === currentUser.id ? 'sent' : 'received'}`}
          >
            <span className="message-sender">
              {msg.sender_id === currentUser.id ? 'You' : msg.sender_name || chatWithName}
            </span>
            <p className="message-text">{msg.message_text}</p>
            <span className="message-time">{new Date(msg.sent_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="message-input-form">
        <input
          type="text"
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Type your message..."
          disabled={sendingMessage || loadingMessages}
        />
        <button type="submit" disabled={sendingMessage || !newMessage.trim()}>
          {sendingMessage ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;
