"""Aggregate evaluation scorecard computation."""


def compute_eval_scorecard(results: list[dict[str, float]]) -> dict[str, float]:
    """Compute aggregate averages across an evaluation run."""
    if not results:
        return {"avg_recall": 0.0, "avg_precision": 0.0, "avg_faithfulness": 0.0}

    total_recall = sum(r.get("recall_at_k", 0.0) for r in results)
    total_precision = sum(r.get("precision_at_k", 0.0) for r in results)
    total_faithfulness = sum(r.get("faithfulness", 0.0) for r in results)

    n = len(results)
    return {
        "avg_recall": round(total_recall / n, 4),
        "avg_precision": round(total_precision / n, 4),
        "avg_faithfulness": round(total_faithfulness / n, 4),
    }
