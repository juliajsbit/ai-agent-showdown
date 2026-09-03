"""Load the corpus, embed it, and store it in pgvector.

Run once (or after changing the corpus):
    ./venv/bin/python -m shared.ingest
"""

import json

import psycopg
from pgvector.psycopg import register_vector

from .config import CORPUS_PATH, DB_DSN, EMBED_DIM
from .embeddings import embed


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(DB_DSN)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def _create_table(conn: psycopg.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute(
        f"""
        CREATE TABLE documents (
            id        TEXT PRIMARY KEY,
            title     TEXT NOT NULL,
            text      TEXT NOT NULL,
            embedding vector({EMBED_DIM}) NOT NULL
        )
        """
    )


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text())
    # Embed title + text together so a query can match on either.
    vectors = embed([f"{d['title']}. {d['text']}" for d in corpus])

    with _connect() as conn:
        _create_table(conn)
        with conn.cursor() as cur:
            for doc, vec in zip(corpus, vectors):
                cur.execute(
                    "INSERT INTO documents (id, title, text, embedding) VALUES (%s, %s, %s, %s)",
                    (doc["id"], doc["title"], doc["text"], vec),
                )
        conn.commit()
        count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    print(f"ingested {count} documents into pgvector")


if __name__ == "__main__":
    main()
