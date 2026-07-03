"""Chunk the corpus, embed the chunks, and build the vector index."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from .config import Settings, describe, load_settings
from .embeddings import Embedder
from .vectorstore import Chunk, get_store

# Approximate tokens as words * 1.3; good enough for chunk sizing and keeps the
# ingest path dependency-free (no tokenizer download).
_TOKENS_PER_WORD = 1.3


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, chunk_tokens: int, overlap: int) -> List[str]:
    """Greedy paragraph-packing with word-level overlap between chunks."""
    max_words = int(chunk_tokens / _TOKENS_PER_WORD)
    overlap_words = int(overlap / _TOKENS_PER_WORD)
    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0
    for para in _split_paragraphs(text):
        pw = len(para.split())
        if cur_words + pw > max_words and cur:
            chunks.append("\n\n".join(cur))
            # carry the tail of the previous chunk forward as overlap
            tail = "\n\n".join(cur).split()[-overlap_words:] if overlap_words else []
            cur = ([" ".join(tail)] if tail else [])
            cur_words = len(tail)
        cur.append(para)
        cur_words += pw
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def build_chunks(settings: Settings) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(Path(settings.corpus_dir).glob("*.md")):
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        for ordinal, piece in enumerate(
            chunk_text(text, settings.chunk_tokens, settings.chunk_overlap)
        ):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{ordinal}",
                    doc_id=doc_id,
                    text=piece,
                    ordinal=ordinal,
                )
            )
    return chunks


def ingest(settings: Settings | None = None) -> int:
    settings = settings or load_settings()
    print(f"[ingest] {describe(settings)}")
    chunks = build_chunks(settings)
    embedder = Embedder(settings)
    vectors = embedder.embed([c.text for c in chunks])
    store = get_store(settings)
    store.build(chunks, vectors)
    print(f"[ingest] indexed {len(chunks)} chunks from {settings.corpus_dir}")
    return len(chunks)


if __name__ == "__main__":
    ingest()
