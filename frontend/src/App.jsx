import React, { useState, useEffect } from "react";
import ChatWindow from "./components/ChatWindow";
import UploadPanel from "./components/UploadPanel";
import { useChat, fetchStatus } from "./hooks/useChat";

export default function App() {
  const { messages, isLoading, sendMessage } = useChat();
  const [status, setStatus] = useState({ ready: false, num_chunks: 0 });

  const loadStatus = async () => {
    try {
      const data = await fetchStatus();
      setStatus(data);
    } catch {
      setStatus({ ready: false, num_chunks: 0 });
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a10 10 0 110 20 10 10 0 010-20z" />
              <path d="M12 6v6l4 2" />
            </svg>
            NutriBot
          </h1>
          <p>Health & Nutrition Q&A</p>
        </div>

        <div className="sidebar-content">
          {/* Status */}
          <div className="sidebar-section" style={{ marginBottom: 20 }}>
            <h3>Index status</h3>
            <div className={`status-badge ${status.ready ? "" : "offline"}`}>
              <span className="status-dot" />
              {status.ready ? "Ready" : "No index"}
            </div>
            <div style={{ marginTop: 10 }}>
              <div className="stat-row">
                <span>Chunks indexed</span>
                <span>{status.num_chunks}</span>
              </div>
            </div>
          </div>

          {/* Upload */}
          <div className="sidebar-section">
            <h3>Documents</h3>
            <UploadPanel onReindexComplete={loadStatus} />
          </div>
        </div>
      </aside>

      {/* Chat */}
      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        onSend={sendMessage}
      />
    </div>
  );
}
