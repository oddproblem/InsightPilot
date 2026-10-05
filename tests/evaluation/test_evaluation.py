"""Tests for evaluation module metrics."""

from app.evaluation.metrics import compute_eval_scorecard
from app.evaluation.retrieval import calculate_retrieval_metrics


def test_calculate_retrieval_metrics() -> None:
    retrieved = ["chunk_1", "chunk_2", "chunk_3", "chunk_4", "chunk_5"]
    gold = ["chunk_2", "chunk_5"]

    metrics = calculate_retrieval_metrics(retrieved, gold, k=5)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["precision_at_k"] == 0.4
    assert metrics["mrr"] == 0.5  # First hit is chunk_2 at rank 2


def test_compute_eval_scorecard() -> None:
    results = [
        {"recall_at_k": 1.0, "precision_at_k": 0.5, "faithfulness": 0.9},
        {"recall_at_k": 0.8, "precision_at_k": 0.4, "faithfulness": 0.8},
    ]
    card = compute_eval_scorecard(results)
    assert card["avg_recall"] == 0.9
    assert card["avg_precision"] == 0.45
    assert card["avg_faithfulness"] == 0.85
