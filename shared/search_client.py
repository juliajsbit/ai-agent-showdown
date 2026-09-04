"""Thin retrieval client the frameworks use.

Talks to the retrieval service over HTTP and returns the same Hit shape the old
in-process search returned, so framework code doesn't change. Depends only on
httpx - no torch, no pgvector - so it installs cleanly in every framework's venv.
"""

from dataclasses import dataclass

import httpx

from .config import RETRIEVAL_URL


@dataclass
class Hit:
    id: str
    title: str
    text: str
    score: float


def search_documents(query: str, k: int = 3) -> list[Hit]:
    """Search the Meridian Robotics knowledge base via the retrieval service."""
    resp = httpx.post(f"{RETRIEVAL_URL}/search", json={"query": query, "k": k}, timeout=30)
    resp.raise_for_status()
    return [Hit(**doc) for doc in resp.json()["results"]]
