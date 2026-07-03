"""Retrieval: embed the query, fetch top-k, optionally rerank."""
from __future__ import annotations

from typing import List

from .config import Settings
from .embeddings import Embedder
from .vectorstore import Hit, get_store


class Retriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings)
        self.store = get_store(settings)
        self.store.load()
        self._reranker = None
        if settings.use_reranker:
            self._reranker = _load_reranker()

    def retrieve(self, query: str, k: int | None = None) -> List[Hit]:
        k = k or self.settings.top_k
        # Over-fetch when reranking so the reranker has candidates to reorder.
        fetch_k = k * 4 if self._reranker else k
        qvec = self.embedder.embed_one(query)
        hits = self.store.search(qvec, fetch_k)
        if self._reranker is not None:
            hits = self._rerank(query, hits)
        return hits[:k]

    def _rerank(self, query: str, hits: List[Hit]) -> List[Hit]:
        pairs = [(query, h.chunk.text) for h in hits]
        scores = self._reranker.predict(pairs)
        for h, s in zip(hits, scores):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)


def _load_reranker():  # pragma: no cover - optional heavy dep
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        # Fall back to a lexical-overlap reranker so USE_RERANKER never crashes.
        return _LexicalReranker()


class _LexicalReranker:
    def predict(self, pairs):
        out = []
        for query, text in pairs:
            q = set(query.lower().split())
            t = set(text.lower().split())
            out.append(len(q & t) / (len(q) + 1e-9))
        return out
