import { useState, useCallback } from "react";

const API_BASE = "/api";

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = useCallback(async (question) => {
    if (!question.trim() || isLoading) return;

    // Add user message
    const userMsg = { role: "user", content: question };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      const botMsg = {
        role: "bot",
        content: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const errorMsg = {
        role: "bot",
        content: `Sorry, something went wrong: ${err.message}`,
        sources: [],
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const clearMessages = useCallback(() => setMessages([]), []);

  return { messages, isLoading, sendMessage, clearMessages };
}

export async function fetchStatus() {
  const res = await fetch(`${API_BASE}/status`);
  return res.json();
}

export async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function reindex() {
  const res = await fetch(`${API_BASE}/reindex`, { method: "POST" });
  return res.json();
}
