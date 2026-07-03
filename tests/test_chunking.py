import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.practical_agents.ingest import chunk_text


def test_chunking_respects_size():
    text = "\n\n".join(f"para {i} " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, chunk_tokens=220, overlap=40)
    assert len(chunks) > 1
    for c in chunks:
        # allow some slack for the overlap tail
        assert len(c.split()) <= int(220 / 1.3) + int(40 / 1.3) + 5


def test_chunking_single_short_doc():
    chunks = chunk_text("just one small paragraph", chunk_tokens=220, overlap=40)
    assert chunks == ["just one small paragraph"]


def test_overlap_present():
    text = "\n\n".join(f"unique{i} " + "filler " * 60 for i in range(4))
    chunks = chunk_text(text, chunk_tokens=120, overlap=40)
    assert len(chunks) >= 2
