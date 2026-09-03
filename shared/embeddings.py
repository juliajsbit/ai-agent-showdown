"""One shared embedding model, loaded lazily and once."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. normalize_embeddings=True so cosine == dot product."""
    vecs = _model().encode(texts, normalize_embeddings=True)
    return vecs.tolist()


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
