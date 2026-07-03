"""Pure-function metrics for retrieval and agent behavior.

Kept dependency-free and side-effect-free so they are unit-tested directly
(see tests/test_metrics.py). The LLM-graded metrics live in judge.py.
"""
from __future__ import annotations

from statistics import median
from typing import Dict, List, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k retrieved doc ids."""
    if not relevant:
        return float("nan")  # undefined; excluded from the mean
    top = list(dict.fromkeys(retrieved))[:k]
    hit = sum(1 for r in relevant if r in top)
    return hit / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1/rank of the first relevant doc; 0 if none retrieved."""
    if not relevant:
        return float("nan")
    for i, doc in enumerate(dict.fromkeys(retrieved), start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def context_precision(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top-k retrieved docs that are relevant (noise measure)."""
    if not relevant:
        return float("nan")
    top = list(dict.fromkeys(retrieved))[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def abstention_scores(rows: List[dict]) -> Dict[str, float]:
    """Precision/recall/accuracy of the agent's decision to abstain.

    Positive class = 'should abstain' (question is not answerable).
    """
    tp = fp = tn = fn = 0
    for r in rows:
        should = not r["answerable"]
        did = r["abstained"]
        if should and did:
            tp += 1
        elif should and not did:
            fn += 1
        elif not should and did:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    accuracy = (tp + tn) / len(rows) if rows else float("nan")
    return {
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_accuracy": accuracy,
    }


def _mean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x == x]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(rows: List[dict], k: int) -> Dict[str, float]:
    """Roll per-question rows up into the summary metric table."""
    latencies = [r["latency_s"] for r in rows]
    latencies_sorted = sorted(latencies)
    p95_idx = max(0, int(round(0.95 * (len(latencies_sorted) - 1))))
    summary = {
        f"recall@{k}": _mean([r["recall_at_k"] for r in rows]),
        "mrr": _mean([r["reciprocal_rank"] for r in rows]),
        "context_precision": _mean([r["context_precision"] for r in rows]),
        "correctness": _mean([r.get("correctness", float("nan")) for r in rows]),
        "faithfulness": _mean([r.get("faithfulness", float("nan")) for r in rows]),
        "avg_tool_calls": _mean([r["tool_calls"] for r in rows]),
        "latency_p50_s": median(latencies) if latencies else float("nan"),
        "latency_p95_s": latencies_sorted[p95_idx] if latencies_sorted else float("nan"),
        "cost_per_1k_usd": _mean([r["cost_usd"] for r in rows]) * 1000,
    }
    summary.update(abstention_scores(rows))
    return summary
