import React, { useRef, useState } from "react";
import { uploadDocument, reindex } from "../hooks/useChat";

export default function UploadPanel({ onReindexComplete }) {
  const fileRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const handleDrop = async (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file) await doUpload(file);
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (file) await doUpload(file);
  };

  const doUpload = async (file) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setFeedback({ type: "error", text: "Only PDF files are supported." });
      return;
    }
    setUploading(true);
    setFeedback(null);
    try {
      const res = await uploadDocument(file);
      setFeedback({ type: "success", text: `Uploaded ${file.name}` });
    } catch (err) {
      setFeedback({ type: "error", text: err.message });
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const handleReindex = async () => {
    setReindexing(true);
    setFeedback(null);
    try {
      const res = await reindex();
      setFeedback({
        type: "success",
        text: `Indexed ${res.num_chunks} chunks`,
      });
      onReindexComplete?.();
    } catch (err) {
      setFeedback({ type: "error", text: "Reindexing failed" });
    } finally {
      setReindexing(false);
    }
  };

  return (
    <div>
      <div
        className="upload-zone"
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <div className="upload-icon">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <p>{uploading ? "Uploading..." : "Drop a PDF or click to upload"}</p>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf"
          hidden
          onChange={handleFileSelect}
        />
      </div>

      <button
        className="reindex-btn"
        onClick={handleReindex}
        disabled={reindexing}
      >
        {reindexing ? "Reindexing..." : "Reindex documents"}
      </button>

      {feedback && (
        <p
          style={{
            marginTop: 10,
            fontSize: 13,
            color: feedback.type === "error" ? "#b35900" : "#2d6a4f",
          }}
        >
          {feedback.text}
        </p>
      )}
    </div>
  );
}
