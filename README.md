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

## Architecture: isolated venvs + services

The frameworks have conflicting dependencies - OpenAI Agents SDK wants `openai`
3.x, CrewAI wants 2.x, and they pull incompatible protobuf/tiktoken versions.
They cannot share one venv. So:

- Each framework lives in its own venv (`frameworks/<name>/venv`) with its own
  `requirements.txt`.
- The heavy shared work (pgvector, embeddings, torch reranker) runs as HTTP
  services, so a framework's venv only needs the framework plus a thin HTTP
  client. No torch, no pgvector in the framework venvs.
- The eval orchestrator runs each framework as a subprocess in its own venv,
  reads back scored rows, and aggregates. One process never imports two
  frameworks.

```
shared/       contract, config, thin search client, eval core + orchestrator
services/     retrieval service (pgvector + embeddings + reranker call)
reranker/     PyTorch cross-encoder served over FastAPI
frameworks/   one isolated venv + agent per framework
data/         corpus.json (knowledge base) + eval_set.json (questions + gold answers)
results/      one scorecard JSON per framework
notes/        field notes per framework (become the write-ups)
```

## Running it

```bash
# 1. start pgvector and load the knowledge base (once)
docker compose up -d
./venv/bin/python -m shared.ingest

# 2. start the two services (each in its own terminal)
./venv/bin/python reranker/app.py            # torch reranker on :8100
./venv/bin/python services/retrieval_app.py  # retrieval on :8200

# 3. create a framework's venv (once)
python3.12 -m venv frameworks/langgraph/venv
frameworks/langgraph/venv/bin/pip install -r frameworks/langgraph/requirements.txt

# 4. run the eval for one framework (subprocesses into its venv)
./venv/bin/python -m shared.evaluate langgraph

# 5. rebuild the comparison table in this README
./venv/bin/python -m shared.compare --readme
```

## Status

- [x] Shared corpus + eval set
- [x] pgvector retrieval + PyTorch reranker (as services)
- [x] Isolated venv per framework + subprocess orchestrator
- [x] Eval harness + comparison table
- [x] LangGraph, CrewAI, AutoGen, OpenAI Agents SDK
- [ ] Remaining 2 frameworks (Pydantic AI, LlamaIndex)

## Results

<!-- RESULTS:START -->
| Framework | Task success | Citations | Reliability | Avg latency | Avg tool calls | Cost / 10 Q |
|---|---|---|---|---|---|---|
| langgraph | 100% | 100% | 100% | 5.50s | 1.1 | $0.0263 |
| openai_sdk | 100% | 100% | 100% | 3.94s | 1.1 | $0.0265 |
| autogen | 100% | 100% | 100% | 3.52s | 1.1 | $0.0269 |
| crewai | 100% | 100% | 100% | 4.00s | 1.1 | $0.0338 |
<!-- RESULTS:END -->
