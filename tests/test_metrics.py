import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics


def test_recall_at_k_full_and_partial():
    assert metrics.recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0
    assert metrics.recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5
    assert metrics.recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0  # a is outside top-2


def test_recall_undefined_when_no_relevant():
    assert math.isnan(metrics.recall_at_k(["a"], [], 3))


def test_reciprocal_rank():
    assert metrics.reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5
    assert metrics.reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert metrics.reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_context_precision():
    assert metrics.context_precision(["a", "b", "x", "y"], ["a", "b"], 4) == 0.5


def test_abstention_scores_perfect():
    rows = [
        {"answerable": False, "abstained": True},
        {"answerable": True, "abstained": False},
    ]
    s = metrics.abstention_scores(rows)
    assert s["abstention_precision"] == 1.0
    assert s["abstention_recall"] == 1.0
    assert s["abstention_accuracy"] == 1.0


def test_abstention_penalizes_hallucination():
    # should abstain but didn't -> recall drops
    rows = [
        {"answerable": False, "abstained": False},
        {"answerable": True, "abstained": False},
    ]
    s = metrics.abstention_scores(rows)
    assert s["abstention_recall"] == 0.0


def test_aggregate_drops_nan():
    rows = [
        {"recall_at_k": 1.0, "reciprocal_rank": 1.0, "context_precision": 1.0,
         "correctness": 1.0, "faithfulness": 1.0, "tool_calls": 1,
         "latency_s": 0.01, "cost_usd": 0.0, "answerable": True, "abstained": False},
        {"recall_at_k": float("nan"), "reciprocal_rank": float("nan"),
         "context_precision": float("nan"), "correctness": 1.0, "faithfulness": 1.0,
         "tool_calls": 1, "latency_s": 0.02, "cost_usd": 0.0,
         "answerable": False, "abstained": True},
    ]
    s = metrics.aggregate(rows, 5)
    assert s["recall@5"] == 1.0  # NaN row excluded
    assert s["correctness"] == 1.0
