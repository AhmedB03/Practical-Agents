"""Run the evaluation harness over the golden set and write a markdown report.

Usage:
    python -m eval.run_eval                # single run with current settings
    python -m eval.run_eval --ablate       # sweep configs and compare
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.practical_agents.agent import Agent
from src.practical_agents.config import describe, load_settings
from eval import metrics
from eval.judge import Judge

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden.jsonl"
RESULTS = ROOT / "eval" / "results"


def load_golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(settings) -> dict:
    from src.practical_agents.agent import _format_context

    agent = Agent(settings)
    judge = Judge(settings)
    golden = load_golden()
    rows = []
    for item in golden:
        res = agent.answer(item["question"])
        context = _format_context(res.retrieved)
        graded = judge.score(
            item["question"], item["reference"], res.answer, context, item["answerable"]
        )
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": res.answer,
                "answerable": item["answerable"],
                "abstained": res.abstained,
                "retrieved_docs": res.retrieved_doc_ids,
                "recall_at_k": metrics.recall_at_k(
                    res.retrieved_doc_ids, item["relevant_docs"], settings.top_k
                ),
                "reciprocal_rank": metrics.reciprocal_rank(
                    res.retrieved_doc_ids, item["relevant_docs"]
                ),
                "context_precision": metrics.context_precision(
                    res.retrieved_doc_ids, item["relevant_docs"], settings.top_k
                ),
                "correctness": graded["correctness"],
                "faithfulness": graded["faithfulness"],
                "tool_calls": res.tool_calls,
                "latency_s": res.latency_s,
                "cost_usd": res.cost_usd,
            }
        )
    summary = metrics.aggregate(rows, settings.top_k)
    return {"config": describe(settings), "summary": summary, "rows": rows}


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.3f}"
    return str(v)


def write_report(result: dict, settings) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "raw.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    s = result["summary"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Evaluation Report",
        "",
        f"_Generated {ts}_",
        "",
        f"**Config:** `{result['config']}`",
        "",
        f"Graded {len(result['rows'])} questions from `eval/golden.jsonl`.",
    ]
    if settings.is_local_llm:
        lines += [
            "",
            "> Running the **local** backend: answers are extractive and grading "
            "uses a transparent lexical heuristic. Configure Azure OpenAI in "
            "`.env` for model-generated answers and LLM-judged grading.",
        ]
    lines += ["", "## Summary", "", "| Metric | Value |", "| --- | --- |"]
    for key in [
        f"recall@{settings.top_k}", "mrr", "context_precision",
        "correctness", "faithfulness",
        "abstention_precision", "abstention_recall", "abstention_accuracy",
        "avg_tool_calls", "latency_p50_s", "latency_p95_s", "cost_per_1k_usd",
    ]:
        lines.append(f"| {key} | {_fmt(s[key])} |")

    lines += ["", "## Per-question", "",
              "| id | ok? | abstain | correct | faithful | recall@k | docs |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["rows"]:
        lines.append(
            f"| {r['id']} | {'—' if r['answerable'] else 'N/A'} | "
            f"{'yes' if r['abstained'] else 'no'} | {_fmt(r['correctness'])} | "
            f"{_fmt(r['faithfulness'])} | {_fmt(r['recall_at_k'])} | "
            f"{','.join(r['retrieved_docs']) or '—'} |"
        )
    report = RESULTS / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def ablate() -> Path:
    configs = [
        {"top_k": 3, "use_reranker": False},
        {"top_k": 5, "use_reranker": False},
        {"top_k": 5, "use_reranker": True},
        {"top_k": 5, "use_reranker": False, "chunk_tokens": 120},
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for overrides in configs:
        settings = load_settings(**overrides)
        # re-ingest for chunking changes so the index matches the config
        from src.practical_agents.ingest import ingest

        ingest(settings)
        result = evaluate(settings)
        s = result["summary"]
        label = f"k={settings.top_k}, rerank={settings.use_reranker}, chunk={settings.chunk_tokens}"
        rows.append((label, s))
        print(f"[ablate] {label}: "
              f"recall@{settings.top_k}={_fmt(s[f'recall@{settings.top_k}'])} "
              f"faith={_fmt(s['faithfulness'])} correct={_fmt(s['correctness'])}")

    lines = ["# Ablation Report", "",
             "| Config | recall@k | faithfulness | correctness | abstain_acc | p95_latency_s | $/1k |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, s in rows:
        rk = next(v for k, v in s.items() if k.startswith("recall@"))
        lines.append(
            f"| {label} | {_fmt(rk)} | {_fmt(s['faithfulness'])} | "
            f"{_fmt(s['correctness'])} | {_fmt(s['abstention_accuracy'])} | "
            f"{_fmt(s['latency_p95_s'])} | {_fmt(s['cost_per_1k_usd'])} |"
        )
    out = RESULTS / "ablation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true", help="sweep configs")
    args = ap.parse_args()
    if args.ablate:
        out = ablate()
        print(f"[eval] wrote {out}")
        return
    settings = load_settings()
    result = evaluate(settings)
    report = write_report(result, settings)
    s = result["summary"]
    print(f"[eval] {result['config']}")
    print(f"[eval] correctness={_fmt(s['correctness'])} "
          f"faithfulness={_fmt(s['faithfulness'])} "
          f"recall@{settings.top_k}={_fmt(s[f'recall@{settings.top_k}'])} "
          f"abstain_acc={_fmt(s['abstention_accuracy'])}")
    print(f"[eval] wrote {report}")


if __name__ == "__main__":
    main()
