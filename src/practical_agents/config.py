"""Provider and runtime configuration.

Resolution order is Azure-first, then OpenAI, then a fully local fallback so the
project runs and its evaluation reproduces with no API keys at all. The rest of
the codebase asks this module which backend is active rather than reading env
vars directly, so the same code path is exercised locally and on Azure.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load a .env file if python-dotenv is installed (optional dependency).
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]


def _env(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


@dataclass(frozen=True)
class Settings:
    # --- generation / judge backend ---
    llm_provider: str          # "azure" | "openai" | "local"
    chat_model: str
    judge_model: str
    # --- embedding backend ---
    embed_provider: str        # "azure" | "openai" | "sentence-transformers" | "hashing"
    embed_model: str
    embed_dim: int
    # --- vector store backend ---
    vectorstore: str           # "azure_search" | "faiss"
    # --- retrieval knobs (overridable for ablations) ---
    top_k: int
    use_reranker: bool
    chunk_tokens: int
    chunk_overlap: int
    # --- paths ---
    corpus_dir: Path
    index_dir: Path

    @property
    def is_local_llm(self) -> bool:
        return self.llm_provider == "local"


def load_settings(**overrides) -> Settings:
    """Build Settings from the environment, with optional keyword overrides.

    Overrides let the ablation harness sweep knobs (top_k, use_reranker,
    chunk_tokens, ...) without mutating the environment.
    """
    # ---- LLM provider ----
    # Groq is checked first: it's an explicit opt-in (GROQ_API_KEY) and should win
    # over a stale OPENAI_API_KEY lingering in the shell environment. Groq serves
    # chat via an OpenAI-compatible endpoint but has no embeddings API, so the
    # embedding block below falls through to the local embedder.
    if _env("GROQ_API_KEY"):
        llm_provider = "groq"
        chat_model = _env("GROQ_CHAT_MODEL") or "llama-3.3-70b-versatile"
        judge_model = _env("GROQ_JUDGE_MODEL") or chat_model
    elif _env("AZURE_OPENAI_API_KEY") and _env("AZURE_OPENAI_ENDPOINT"):
        llm_provider = "azure"
        chat_model = _env("AZURE_OPENAI_CHAT_DEPLOYMENT") or "gpt-4o-mini"
        judge_model = _env("AZURE_OPENAI_JUDGE_DEPLOYMENT") or chat_model
    elif _env("OPENAI_API_KEY"):
        llm_provider = "openai"
        chat_model = _env("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
        judge_model = _env("OPENAI_JUDGE_MODEL") or chat_model
    else:
        llm_provider = "local"
        chat_model = "local-extractive"
        judge_model = "local-heuristic"

    # ---- Embedding provider ----
    if llm_provider == "azure" and _env("AZURE_OPENAI_EMBED_DEPLOYMENT"):
        embed_provider = "azure"
        embed_model = _env("AZURE_OPENAI_EMBED_DEPLOYMENT")
        embed_dim = int(_env("EMBED_DIM") or 1536)
    elif llm_provider == "openai":
        embed_provider = "openai"
        embed_model = _env("OPENAI_EMBED_MODEL") or "text-embedding-3-small"
        embed_dim = int(_env("EMBED_DIM") or 1536)
    else:
        # Prefer real local embeddings; fall back to deterministic hashing if
        # sentence-transformers isn't installed (keeps CI light).
        try:
            import sentence_transformers  # noqa: F401

            embed_provider = "sentence-transformers"
            embed_model = _env("ST_EMBED_MODEL") or "all-MiniLM-L6-v2"
            embed_dim = 384
        except Exception:
            embed_provider = "hashing"
            embed_model = "hashing-2048"
            embed_dim = 2048

    # ---- Vector store ----
    if _env("AZURE_SEARCH_ENDPOINT") and _env("AZURE_SEARCH_API_KEY"):
        vectorstore = "azure_search"
    else:
        vectorstore = "faiss"

    settings = Settings(
        llm_provider=llm_provider,
        chat_model=chat_model,
        judge_model=judge_model,
        embed_provider=embed_provider,
        embed_model=embed_model,
        embed_dim=embed_dim,
        vectorstore=vectorstore,
        top_k=int(_env("TOP_K") or 5),
        use_reranker=(_env("USE_RERANKER") or "false").lower() == "true",
        chunk_tokens=int(_env("CHUNK_TOKENS") or 220),
        chunk_overlap=int(_env("CHUNK_OVERLAP") or 40),
        corpus_dir=Path(_env("CORPUS_DIR") or (ROOT / "corpus")),
        index_dir=Path(_env("INDEX_DIR") or (ROOT / "eval" / "index")),
    )
    if overrides:
        from dataclasses import replace

        settings = replace(settings, **overrides)
    return settings


def describe(settings: Settings) -> str:
    return (
        f"llm={settings.llm_provider}:{settings.chat_model} | "
        f"embed={settings.embed_provider}:{settings.embed_model}({settings.embed_dim}d) | "
        f"store={settings.vectorstore} | top_k={settings.top_k} "
        f"rerank={settings.use_reranker} chunk={settings.chunk_tokens}/{settings.chunk_overlap}"
    )
