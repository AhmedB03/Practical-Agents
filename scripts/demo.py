"""Interactive CLI: ask the agent questions over the ingested corpus.

    python scripts/demo.py
    python scripts/demo.py "How many retries by default?"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.practical_agents.agent import Agent
from src.practical_agents.config import describe, load_settings


def ask(agent, q):
    res = agent.answer(q)
    print(f"\nQ: {q}")
    print(f"A: {res.answer}")
    print(
        f"   [docs: {', '.join(res.retrieved_doc_ids) or '—'} | "
        f"tool_calls={res.tool_calls} | {res.latency_s*1000:.0f}ms | "
        f"${res.cost_usd:.5f}]"
    )


def main():
    settings = load_settings()
    print(f"config: {describe(settings)}")
    agent = Agent(settings)
    if len(sys.argv) > 1:
        ask(agent, " ".join(sys.argv[1:]))
        return
    print("Ask a question (blank line to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        ask(agent, q)


if __name__ == "__main__":
    main()
