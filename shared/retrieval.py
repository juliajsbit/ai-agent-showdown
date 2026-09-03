"""The document search tool. Framework-agnostic on purpose.

Pipeline: embed query -> pgvector cosine search (recall) -> torch cross-encoder
rerank (precision) -> top-k. Every framework's agent calls search_documents()
with the same signature, so only the agent wiring differs between frameworks.
"""

from dataclasses import dataclass

import httpx
import psycopg
from pgvector.psycopg import register_vector

from .config import DB_DSN, RERANKER_URL
from .embeddings import embed_one


@dataclass
class Hit:
    id: str
    title: str
    text: str
    score: float  # reranker relevance, 0..1


def _vector_search(query: str, limit: int) -> list[Hit]:
    """First-stage recall: nearest neighbours by cosine distance in pgvector."""
    qvec = embed_one(query)
    with psycopg.connect(DB_DSN) as conn:
        register_vector(conn)
        rows = conn.execute(
            # <=> is cosine distance; smaller is closer. Score here is a rough
            # 1 - distance, later overwritten by the reranker.
            """
            SELECT id, title, text, 1 - (embedding <=> %s::vector) AS sim
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, limit),
        ).fetchall()
    return [Hit(id=r[0], title=r[1], text=r[2], score=float(r[3])) for r in rows]


def _rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Second stage: torch cross-encoder rescoring. Falls back to vector order
    if the reranker service is down, so the agent still works without it."""
    try:
        resp = httpx.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "documents": [h.text for h in hits], "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPError:
        return hits[:top_k]

    reranked = []
    for r in resp.json()["results"]:
        h = hits[r["index"]]
        reranked.append(Hit(id=h.id, title=h.title, text=h.text, score=r["score"]))
    return reranked


def search_documents(query: str, k: int = 3, recall: int = 8) -> list[Hit]:
    """Search the Meridian Robotics knowledge base.

    Args:
        query: natural-language question.
        k: number of documents to return after reranking.
        recall: how many candidates to pull from pgvector before reranking.
    """
    candidates = _vector_search(query, limit=recall)
    if not candidates:
        return []
    return _rerank(query, candidates, top_k=k)


if __name__ == "__main__":
    for h in search_documents("how many vacation days do I get?"):
        print(f"{h.score:.3f}  {h.id:20s} {h.title}")
