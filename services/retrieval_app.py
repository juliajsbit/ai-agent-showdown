"""Retrieval as an HTTP service.

Owns the heavy dependencies - pgvector, sentence-transformers embeddings, and the
call to the torch reranker. Every framework calls /search over HTTP instead of
importing any of that, so a framework's venv only needs the framework itself plus
a thin HTTP client. This is what lets each framework live in its own isolated
venv without dependency conflicts.

Run (from the main venv):
    ./venv/bin/python services/retrieval_app.py
"""

from fastapi import FastAPI
from pydantic import BaseModel

from shared.retrieval import search_documents

app = FastAPI(title="retrieval service")


class SearchRequest(BaseModel):
    query: str
    k: int = 3


class Doc(BaseModel):
    id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    results: list[Doc]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    hits = search_documents(req.query, k=req.k)
    return SearchResponse(results=[Doc(id=h.id, title=h.title, text=h.text, score=h.score) for h in hits])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8200)
