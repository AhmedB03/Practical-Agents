"""Vector store backends.

`FaissStore` is the default local backend. It uses `faiss` if installed, else a
pure-NumPy exact search so the project never hard-fails on a missing wheel.
`AzureSearchStore` is the production backend (Azure AI Search vector index).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np

from .config import Settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int


@dataclass
class Hit:
    chunk: Chunk
    score: float


class FaissStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dir = settings.index_dir
        self._matrix: np.ndarray | None = None
        self._chunks: List[Chunk] = []
        self._faiss_index = None

    # -- build/persist ---------------------------------------------------
    def build(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        self._chunks = chunks
        self._matrix = vectors.astype(np.float32)
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "vectors.npy", self._matrix)
        with open(self.dir / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c)) + "\n")

    def load(self) -> None:
        self._matrix = np.load(self.dir / "vectors.npy")
        self._chunks = []
        with open(self.dir / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                self._chunks.append(Chunk(**json.loads(line)))
        try:
            import faiss

            index = faiss.IndexFlatIP(self._matrix.shape[1])
            index.add(self._matrix)
            self._faiss_index = index
        except Exception:
            self._faiss_index = None

    # -- query -----------------------------------------------------------
    def search(self, query_vec: np.ndarray, k: int) -> List[Hit]:
        if self._matrix is None:
            self.load()
        q = query_vec.astype(np.float32).reshape(1, -1)
        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, min(k, len(self._chunks)))
            pairs = zip(idxs[0].tolist(), scores[0].tolist())
        else:
            sims = (self._matrix @ q[0])
            top = np.argsort(-sims)[:k]
            pairs = ((int(i), float(sims[i])) for i in top)
        return [Hit(chunk=self._chunks[i], score=s) for i, s in pairs if i >= 0]


class AzureSearchStore:  # pragma: no cover - requires Azure
    def __init__(self, settings: Settings):
        import os
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient

        self.settings = settings
        self.client = SearchClient(
            endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
            index_name=os.environ["AZURE_SEARCH_INDEX"],
            credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
        )

    def build(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        docs = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "ordinal": c.ordinal,
                "text": c.text,
                "vector": vectors[i].tolist(),
            }
            for i, c in enumerate(chunks)
        ]
        for start in range(0, len(docs), 1000):
            self.client.upload_documents(docs[start : start + 1000])

    def load(self) -> None:
        return None

    def search(self, query_vec: np.ndarray, k: int) -> List[Hit]:
        from azure.search.documents.models import VectorizedQuery

        vq = VectorizedQuery(
            vector=query_vec.tolist(), k_nearest_neighbors=k, fields="vector"
        )
        results = self.client.search(search_text=None, vector_queries=[vq], top=k)
        hits: List[Hit] = []
        for r in results:
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=r["chunk_id"],
                        doc_id=r["doc_id"],
                        text=r["text"],
                        ordinal=r["ordinal"],
                    ),
                    score=float(r.get("@search.score", 0.0)),
                )
            )
        return hits


def get_store(settings: Settings):
    if settings.vectorstore == "azure_search":
        return AzureSearchStore(settings)
    return FaissStore(settings)
