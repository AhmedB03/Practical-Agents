"""Embedding backends: Azure OpenAI / OpenAI / sentence-transformers / hashing.

The hashing backend is a deterministic bag-of-words projection with no
dependencies. It is intentionally weak, but it guarantees the whole pipeline —
and therefore the evaluation — runs offline with reproducible numbers.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import List

import numpy as np

from .config import Settings

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


class Embedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.embed_provider
        self.dim = settings.embed_dim
        self._client = None
        self._st = None
        if self.provider == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(settings.embed_model)
            self.dim = self._st.get_sentence_embedding_dimension()
        elif self.provider in ("azure", "openai"):
            self._client = _openai_client(settings)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.provider == "hashing":
            vecs = np.stack([self._hash_embed(t) for t in texts])
        elif self.provider == "sentence-transformers":
            vecs = np.asarray(
                self._st.encode(texts, normalize_embeddings=False), dtype=np.float32
            )
        else:
            vecs = self._openai_embed(texts)
        return _l2_normalize(vecs.astype(np.float32))

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    # -- backends --------------------------------------------------------
    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hashed TF vector with sublinear term weighting."""
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: dict[int, int] = {}
        for tok in _tokenize(text):
            h = int.from_bytes(
                hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little"
            )
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            counts[idx] = counts.get(idx, 0)
            vec[idx] += sign
            counts[idx] += 1
        # sublinear damping so long chunks don't dominate
        for idx, c in counts.items():
            if c > 1:
                vec[idx] = (vec[idx] / c) * (1.0 + math.log(c))
        return vec

    def _openai_embed(self, texts: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(
            model=self.settings.embed_model, input=texts
        )
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


@lru_cache(maxsize=1)
def _openai_client(settings: Settings):  # pragma: no cover - needs network
    if settings.embed_provider == "azure" or settings.llm_provider == "azure":
        from openai import AzureOpenAI
        import os

        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
        )
    from openai import OpenAI

    return OpenAI()
