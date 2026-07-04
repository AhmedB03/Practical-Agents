"""Chat/completion client with a local fallback.

`chat()` returns a normalized dict: {text, prompt_tokens, completion_tokens}.
The local provider is deterministic and network-free: it powers reproducible
offline eval, doing extractive answering over whatever context it is given and
abstaining when the context does not contain the answer.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import List

# rough per-1k-token USD prices for cost estimation (gpt-4o-mini defaults)
PRICE_PER_1K = {
    "prompt": float(os.getenv("PRICE_PROMPT_PER_1K", "0.00015")),
    "completion": float(os.getenv("PRICE_COMPLETION_PER_1K", "0.00060")),
}

ABSTAIN = "I don't have enough information in the documentation to answer that."


class LLM:
    def __init__(self, settings):
        self.settings = settings
        self.provider = settings.llm_provider
        self._client = None
        if self.provider in ("azure", "openai", "groq"):
            self._client = _client(settings)

    def chat(self, system: str, user: str, model: str | None = None) -> dict:
        model = model or self.settings.chat_model
        if self.provider == "local":
            return self._local_chat(system, user)
        return self._api_chat(system, user, model)

    # -- API path --------------------------------------------------------
    def _api_chat(self, system: str, user: str, model: str) -> dict:  # pragma: no cover
        resp = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        usage = resp.usage
        return {
            "text": resp.choices[0].message.content or "",
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
        }

    # -- Local path ------------------------------------------------------
    def _local_chat(self, system: str, user: str) -> dict:
        """Extractive answerer over the CONTEXT block in `user`.

        Picks the context sentences with the highest lexical overlap with the
        question. Abstains when overlap is negligible, which is exactly the
        behavior the eval measures for out-of-scope questions.
        """
        question, context = _split_context(user)
        sentences = _sentences(context)
        q_terms = _content_terms(question)
        scored = []
        for s in sentences:
            s_terms = _content_terms(s)
            overlap = len(q_terms & s_terms)
            if overlap:
                scored.append((overlap, s))
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        if not scored or scored[0][0] < 1:
            text = ABSTAIN
        else:
            text = " ".join(s for _, s in scored[:2]).strip()
        return {
            "text": text,
            "prompt_tokens": _approx_tokens(system) + _approx_tokens(user),
            "completion_tokens": _approx_tokens(text),
        }


# -- cost -----------------------------------------------------------------
def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1000 * PRICE_PER_1K["prompt"]
        + completion_tokens / 1000 * PRICE_PER_1K["completion"]
    )


# -- helpers --------------------------------------------------------------
_STOP = set(
    "the a an of to in and or is are for with on by as at be it this that from "
    "you your can do does what how when where which who why into per not no".split()
)


def _content_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOP}


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*`") for p in parts if p.strip(" -*`")]


def _split_context(user: str) -> tuple[str, str]:
    if "CONTEXT:" in user:
        q, ctx = user.split("CONTEXT:", 1)
        q = q.replace("QUESTION:", "").strip()
        return q, ctx
    return user, user


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# Free tiers (esp. Groq) enforce tight tokens-per-minute limits. The OpenAI SDK
# honors 429 Retry-After headers and backs off, so a generous retry budget lets
# a long eval pace itself under the limit instead of crashing.
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "12"))


@lru_cache(maxsize=1)
def _client(settings):  # pragma: no cover - needs network
    if settings.llm_provider == "azure":
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            max_retries=_MAX_RETRIES,
        )
    from openai import OpenAI

    if settings.llm_provider == "groq":
        # Groq exposes an OpenAI-compatible API; only base_url + key differ.
        return OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            max_retries=_MAX_RETRIES,
        )
    return OpenAI(max_retries=_MAX_RETRIES)
