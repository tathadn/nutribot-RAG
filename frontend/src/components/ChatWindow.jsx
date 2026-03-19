import React, { useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";

const EXAMPLE_QUERIES = [
  "What are the health benefits of omega-3 fatty acids?",
  "How does fiber affect gut health?",
  "What vitamins are important for bone density?",
  "How does intermittent fasting affect metabolism?",
];

export default function ChatWindow({ messages, isLoading, onSend }) {
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const val = inputRef.current?.value?.trim();
    if (!val) return;
    onSend(val);
    inputRef.current.value = "";
  };

  const handleExample = (query) => {
    onSend(query);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="chat-area">
      <div className="chat-header">
        Ask questions grounded in your uploaded research articles and books
      </div>

      {isEmpty ? (
        <div className="welcome">
          <h2>What would you like to know?</h2>
          <p>
            Ask any health or nutrition question. NutriBot will search your
            documents and answer with cited sources.
          </p>
          <div className="example-queries">
            {EXAMPLE_QUERIES.map((q, i) => (
              <button
                key={i}
                className="example-query"
                onClick={() => handleExample(q)}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="messages">
          {messages.map((msg, i) => (
            <MessageBubble key={i} message={msg} />
          ))}

          {isLoading && (
            <div className="message bot">
              <div className="message-avatar">N</div>
              <div className="message-content">
                <div className="loading-dots">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      )}

      <div className="input-area">
        <form className="input-wrapper" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            type="text"
            placeholder="Ask a health or nutrition question..."
            disabled={isLoading}
            autoFocus
          />
          <button className="send-btn" type="submit" disabled={isLoading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
