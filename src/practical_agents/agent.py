"""The tool-using RAG agent.

For API providers the agent uses real tool-calling: the model decides whether to
call `search_docs`, may call it multiple times to refine the query, and then
answers. For the local provider it runs a single deterministic retrieve→answer
step so the offline eval is stable. Both paths return the same `AgentResult`, so
the evaluation harness treats them identically.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List

from .config import Settings, load_settings
from .llm import LLM, ABSTAIN, estimate_cost
from .retriever import Retriever
from .vectorstore import Hit

SYSTEM = (
    "You are a documentation assistant. Answer strictly from the provided "
    "context. If the context does not contain the answer, reply exactly: "
    f'"{ABSTAIN}" Cite the doc ids you used in brackets, e.g. [configuration].'
)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the product documentation for relevant passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"}
            },
            "required": ["query"],
        },
    },
}


@dataclass
class AgentResult:
    question: str
    answer: str
    retrieved: List[Hit] = field(default_factory=list)
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    abstained: bool = False

    @property
    def retrieved_doc_ids(self) -> List[str]:
        seen, out = set(), []
        for h in self.retrieved:
            if h.chunk.doc_id not in seen:
                seen.add(h.chunk.doc_id)
                out.append(h.chunk.doc_id)
        return out

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.prompt_tokens, self.completion_tokens)


class Agent:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.retriever = Retriever(self.settings)
        self.llm = LLM(self.settings)

    def answer(self, question: str) -> AgentResult:
        t0 = time.perf_counter()
        if self.settings.is_local_llm:
            result = self._answer_local(question)
        else:  # pragma: no cover - needs network
            result = self._answer_agentic(question)
        result.latency_s = time.perf_counter() - t0
        result.abstained = ABSTAIN[:24].lower() in result.answer.lower()
        return result

    # -- deterministic local path ---------------------------------------
    def _answer_local(self, question: str) -> AgentResult:
        hits = self.retriever.retrieve(question)
        context = _format_context(hits)
        user = f"QUESTION: {question}\nCONTEXT:\n{context}"
        resp = self.llm.chat(SYSTEM, user)
        return AgentResult(
            question=question,
            answer=resp["text"],
            retrieved=hits,
            tool_calls=1,
            prompt_tokens=resp["prompt_tokens"],
            completion_tokens=resp["completion_tokens"],
        )

    # -- real tool-calling path -----------------------------------------
    def _answer_agentic(self, question: str, max_steps: int = 3) -> AgentResult:  # pragma: no cover
        client = self.llm._client
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        all_hits: List[Hit] = []
        pt = ct = tool_calls = 0
        for _ in range(max_steps):
            resp = client.chat.completions.create(
                model=self.settings.chat_model,
                messages=messages,
                tools=[SEARCH_TOOL],
                temperature=0,
            )
            pt += resp.usage.prompt_tokens
            ct += resp.usage.completion_tokens
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return AgentResult(
                    question=question,
                    answer=msg.content or "",
                    retrieved=all_hits,
                    tool_calls=tool_calls,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                )
            messages.append(msg)
            for call in msg.tool_calls:
                tool_calls += 1
                args = json.loads(call.function.arguments or "{}")
                hits = self.retriever.retrieve(args.get("query", question))
                all_hits.extend(hits)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _format_context(hits),
                    }
                )
        # ran out of steps; force a final answer from what we have
        resp = client.chat.completions.create(
            model=self.settings.chat_model, messages=messages, temperature=0
        )
        pt += resp.usage.prompt_tokens
        ct += resp.usage.completion_tokens
        return AgentResult(
            question=question,
            answer=resp.choices[0].message.content or "",
            retrieved=all_hits,
            tool_calls=tool_calls,
            prompt_tokens=pt,
            completion_tokens=ct,
        )


def _format_context(hits: List[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk.doc_id}] {h.chunk.text}" for h in hits)
