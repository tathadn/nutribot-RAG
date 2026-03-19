import React from "react";

export default function SourceCard({ source }) {
  return (
    <div className="source-card">
      <div className="source-name">
        {source.source} — p.{source.page}
      </div>
      {source.text_preview && (
        <div className="source-preview">{source.text_preview}</div>
      )}
      <div className="source-score">
        relevance: {(source.score * 100).toFixed(1)}%
      </div>
    </div>
  );
}
