"""Retrieval evaluation metrics: Recall@K, Precision@K, and MRR."""


def calculate_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    gold_chunk_ids: list[str],
    k: int = 5,
) -> dict[str, float]:
    """Compute standard IR metrics: Recall@K, Precision@K, and Reciprocal Rank."""
    if not gold_chunk_ids:
        return {"recall_at_k": 1.0, "precision_at_k": 1.0, "mrr": 1.0}

    top_k = retrieved_chunk_ids[:k]
    hits = [cid for cid in top_k if cid in gold_chunk_ids]

    recall = len(hits) / len(gold_chunk_ids)
    precision = len(hits) / len(top_k) if top_k else 0.0

    # Mean Reciprocal Rank (MRR)
    mrr = 0.0
    for idx, cid in enumerate(top_k, 1):
        if cid in gold_chunk_ids:
            mrr = 1.0 / idx
            break

    return {
        "recall_at_k": round(recall, 4),
        "precision_at_k": round(precision, 4),
        "mrr": round(mrr, 4),
    }
