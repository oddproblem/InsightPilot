# InsightPilot Evaluation Harness

To ensure reliability for financial reasoning and document synthesis, InsightPilot includes an offline evaluation framework to benchmark routing, retrieval, and generation quality.

## Evaluation Pipeline

```
Benchmark Dataset (`evals/dataset.jsonl`)
         │
         ▼
  `evals/run_evaluation.py`
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
Routing Accuracy          Retrieval & Groundedness
(Document vs. Web         (Recall@K, Precision@K,
 vs. Calculator)           Faithfulness Score)
    │                               │
    └────┬──────────────────────────┘
         ▼
Scorecard Output (`evals/results/latest_run.json`)
```

## Key Metrics

1. **Routing Accuracy**:
   - Measures whether the Query Analyzer correctly picks `document_search`, `web_search`, `calculator`, or `direct`.
   - Target: >95%.

2. **Recall@K & Precision@K**:
   - Evaluates whether the gold relevant chunks appear in the top-K retrieved vector passages from pgvector.

3. **Faithfulness Score**:
   - Measures the lexical and factual alignment between the generated response and the retrieved source text.

4. **Refusal on Insufficient Evidence**:
   - Tests whether the model reliably outputs `"The uploaded documents do not contain sufficient information..."` when data is absent, preventing hallucinations.

## Running the Evaluation

```bash
python evals/run_evaluation.py
```
Results are saved to `evals/results/latest_run.json`.
