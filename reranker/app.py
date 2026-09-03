"""Cross-encoder reranker served over FastAPI.

This is the one PyTorch piece in the whole project. Every framework's agent
calls the same /rerank endpoint as a tool, so the reranker never changes when
we swap frameworks.

We load a pretrained cross-encoder (ms-marco-MiniLM) and run the forward pass
in plain torch on purpose - tokenize, model(**inputs).logits, sigmoid - so the
torch part is visible, not hidden behind a black-box .predict().
"""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Filled in on startup. Kept module-level so the model loads once, not per request.
_state: dict = {}


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # no dropout, no grad tracking during inference
    _state["tokenizer"] = tokenizer
    _state["model"] = model
    _state["device"] = device
    print(f"reranker ready on {device}")
    yield
    _state.clear()


app = FastAPI(title="cross-encoder reranker", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int | None = None  # if set, return only the top_k results


class Scored(BaseModel):
    index: int  # position in the original documents list
    text: str
    score: float


class RerankResponse(BaseModel):
    results: list[Scored]  # sorted best-first


@torch.no_grad()
def _score(query: str, documents: list[str]) -> list[float]:
    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]

    # A cross-encoder reads (query, document) as one joined sequence and outputs
    # a single relevance logit. That is why it beats bi-encoder cosine - it sees
    # both texts together instead of comparing two separate embeddings.
    pairs = [[query, doc] for doc in documents]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    logits = model(**inputs).logits.squeeze(-1)  # shape: (num_documents,)
    scores = torch.sigmoid(logits)  # map to 0..1 so scores are readable
    return scores.tolist()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": _state.get("device")}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    if not req.documents:
        return RerankResponse(results=[])

    scores = _score(req.query, req.documents)
    ranked = sorted(
        (Scored(index=i, text=doc, score=s) for i, (doc, s) in enumerate(zip(req.documents, scores))),
        key=lambda r: r.score,
        reverse=True,
    )
    if req.top_k is not None:
        ranked = ranked[: req.top_k]
    return RerankResponse(results=ranked)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8100)
