"""
Claude Sonnet integration for RAG-based question answering.
"""

import anthropic
from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS


SYSTEM_PROMPT = """You are NutriBot, a knowledgeable health and nutrition assistant. \
You answer questions ONLY based on the provided context from research articles and books.

Rules:
1. Answer the user's question using ONLY the information in the provided context chunks.
2. Cite your sources by referencing the document name and page number, e.g. [Source: filename.pdf, p.12].
3. If the context does not contain enough information to answer, say so honestly. Do NOT make up information.
4. Keep answers clear, concise, and well-structured.
5. When multiple sources provide information, synthesize them and cite each.
6. If the question is outside the health/nutrition domain, politely redirect the user.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block for the prompt."""
    if not chunks:
        return "No relevant context found in the knowledge base."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"--- Chunk {i} ---\n"
            f"Source: {chunk['source']}, Page: {chunk['page']}\n"
            f"Relevance Score: {chunk['score']:.3f}\n\n"
            f"{chunk['text']}\n"
        )

    return "\n".join(context_parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Send the query + retrieved context to Claude Sonnet and return the answer.

    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "page": int, "score": float}, ...]
        }
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    context_block = build_context_block(chunks)

    user_message = (
        f"Context from knowledge base:\n\n{context_block}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Please answer based on the context above, citing your sources."
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text

    # Deduplicate sources for the response
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "score": chunk["score"],
                "text_preview": chunk["text"][:150] + "...",
            })

    return {"answer": answer_text, "sources": sources}
