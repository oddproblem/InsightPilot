# Retrieval & RAG Experiments

This document records experimental findings on chunking strategies, embeddings models, and retrieval accuracy for financial document analysis.

## Experiment 1: Chunking Strategies on Financial Tables

Financial tables (e.g., balance sheets, income statements) are brittle to arbitrary token splits.

| Strategy | Table Preservation | Context Coherence | Recall@5 |
|---|---|---|---|
| **Fixed 500 tokens (no overlap)** | Poor (splits tables mid-row) | Low | 64.2% |
| **Sliding Window (1000 tokens, 150 overlap)** | Moderate | Good | 81.5% |
| **Paragraph / Markdown Block Aware** | High (preserves markdown tables) | High | 92.4% |

**Decision**: Adopted paragraph and table-boundary preserving chunker in `app/rag/chunking.py`.

---

## Experiment 2: Dense Embeddings Performance

Comparison of embedding models on financial retrieval tasks:

| Model | Dimensions | Latency (p95) | NDCG@10 | Memory Footprint |
|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | 62 ms | 0.84 | Baseline (compact) |
| `text-embedding-3-large` | 3072 | 118 ms | 0.87 | 2x index size |
| `all-MiniLM-L6-v2` | 384 | 24 ms | 0.72 | Lowest |

**Decision**: Defaulting to `text-embedding-3-small` for balanced latency, accuracy, and storage efficiency.

---

## Experiment 3: Vector Indexing (HNSW vs. IVFFlat)

Benchmarked on a corpus of 100,000 financial document chunks in pgvector:

| Index Type | Build Time | Query Latency (p99) | Recall vs. Exact |
|---|---|---|---|
| **IVFFlat (lists=100)** | 12s | 18 ms | 88.2% |
| **HNSW (m=16, ef_construction=64)** | 48s | 6 ms | 98.7% |

**Decision**: Using HNSW index on `embedding vector_cosine_ops` for low query latency and high recall.
