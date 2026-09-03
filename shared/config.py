"""Shared config for every framework's agent. Nothing framework-specific here."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.json"
EVAL_PATH = DATA_DIR / "eval_set.json"

# Local Postgres from docker-compose. 5433 on host to avoid clashing with a
# system postgres on 5432.
DB_DSN = os.getenv(
    "SHOWDOWN_DB_DSN",
    "postgresql://showdown:showdown@127.0.0.1:5433/showdown",
)

# Local sentence-transformers model - retrieval stays free and offline. Only the
# agent's LLM calls cost money.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# torch reranker service (reranker/app.py)
RERANKER_URL = os.getenv("SHOWDOWN_RERANKER_URL", "http://127.0.0.1:8100")

# One model for all six frameworks - otherwise we'd be measuring the model, not
# the framework.
MODEL = os.getenv("SHOWDOWN_MODEL", "claude-haiku-4-5")

# Haiku 4.5 list price, USD per million tokens. Used to turn token counts into a
# dollar cost column in the results table.
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * PRICE_IN_PER_MTOK + (output_tokens / 1_000_000) * PRICE_OUT_PER_MTOK
