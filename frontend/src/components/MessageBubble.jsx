import React from "react";
import SourceCard from "./SourceCard";

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={`message ${message.role}`}>
      <div className="message-avatar">
        {isUser ? "U" : "N"}
      </div>
      <div className="message-content">
        {message.content.split("\n").map((paragraph, i) =>
          paragraph.trim() ? <p key={i}>{paragraph}</p> : null
        )}

        {!isUser && message.sources?.length > 0 && (
          <div className="sources-section">
            <div className="sources-label">Sources</div>
            {message.sources.map((src, i) => (
              <SourceCard key={i} source={src} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
