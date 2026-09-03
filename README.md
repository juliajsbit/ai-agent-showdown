# Agent Framework Showdown

One agent task, built in 6 frameworks, run through one eval harness.

The task: a support agent that answers questions about a company knowledge base
(the fictional "Meridian Robotics"). Same task, same tools, same model, same
metrics for every framework - so the comparison measures the framework, not the
model or the prompt.

Frameworks compared: LangGraph, CrewAI, AutoGen, OpenAI Agents SDK, Pydantic AI,
LlamaIndex.

Metrics: task success, cost, latency, reliability, developer experience.

## What each tool does in this project

| Tool / library | Role here |
|---|---|
| **Postgres + pgvector** | Vector database. Stores the knowledge base as embeddings and does the first-stage similarity search. |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Turns text into embeddings. Runs locally, so retrieval is free and offline. |
| **PyTorch + Transformers** (`ms-marco-MiniLM` cross-encoder) | The reranker. Rereads the top candidates from pgvector and reorders them by real relevance. Runs on Apple MPS. |
| **FastAPI + Uvicorn** | Serves the reranker as an HTTP tool. Every framework calls the same endpoint, so the reranker never changes when we swap frameworks. |
| **Claude Haiku 4.5** | The single LLM behind every agent. One model across all six so the comparison is fair. |
| **httpx / psycopg** | HTTP client for the reranker, Postgres driver for pgvector. |
| **Docker Compose** | Runs the pgvector database locally. |

## How retrieval works

Two stages:

1. **pgvector** pulls ~8 candidate documents by embedding similarity. Fast, but
   rough - it matches on word overlap, so plausible-but-wrong docs can rank high.
2. **The PyTorch reranker** rereads each candidate against the question and
   rescores it. It pushes generic matches down and pulls truly relevant docs up.

The agent then answers using only the reranked top-k documents, with citations.

## Layout

```
shared/       task, corpus, tools (retrieval + reranker client), config, contract
reranker/     PyTorch cross-encoder served over FastAPI
frameworks/   one agent implementation per framework
data/         corpus.json (knowledge base) + eval_set.json (questions + gold answers)
notes/        field notes per framework (become the write-ups)
```

## Running it

```bash
# 1. start pgvector
docker compose up -d

# 2. load the knowledge base into pgvector (run once)
./venv/bin/python -m shared.ingest

# 3. start the reranker service
./venv/bin/python reranker/app.py

# 4. try retrieval end to end
./venv/bin/python -m shared.retrieval
```

## Status

- [x] Shared corpus + eval set
- [x] pgvector retrieval
- [x] PyTorch reranker as a tool
- [x] Shared search tool
- [x] Eval harness + comparison table
- [x] LangGraph agent (baseline)
- [x] CrewAI agent
- [x] AutoGen agent
- [ ] Remaining 3 frameworks (OpenAI Agents SDK, Pydantic AI, LlamaIndex)

## Results

<!-- RESULTS:START -->
| Framework | Task success | Citations | Reliability | Avg latency | Avg tool calls | Cost / 10 Q |
|---|---|---|---|---|---|---|
| langgraph | 100% | 100% | 100% | 4.47s | 1.1 | $0.0263 |
| autogen | 100% | 100% | 100% | 2.84s | 1.2 | $0.0274 |
| crewai | 100% | 100% | 100% | 3.40s | 1.1 | $0.0338 |
<!-- RESULTS:END -->
