# InsightPilot System Architecture

InsightPilot is an agentic document intelligence system built for analyzing complex financial and business documents (e.g., 10-K, 10-Q, quarterly earnings, annual reports).

Unlike conventional RAG chatbots that mechanically retrieve documents for every question, InsightPilot implements an intelligent, multi-stage reasoning pipeline where the model analyzes intent and selectively activates domain tools.

## High-Level Pipeline

```
                 USER QUERY
                     │
                     ▼
          ┌───────────────────────┐
          │     Input Guard       │  (Prompt Injection & Adversarial Check)
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │    Query Analyzer     │  (Classifies Intent & Strategy)
          └──────────┬────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────────┐┌──────────┐┌───────────┐
│Document Search││Web Search││Calculator │
└───────┬──────┘└────┬─────┘└─────┬─────┘
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │    Evidence Layer     │  (Synthesizes & grounds atomic facts)
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │   Answer Generator    │  (Structured financial response synthesis)
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │  Citation Validator   │  (Verifies source grounding & extracts citations)
          └──────────┬────────────┘
                     │
                     ▼
          ┌───────────────────────┐
          │     Output Guard      │  (PII, credential, and leak scrubbing)
          └──────────┬────────────┘
                     │
                     ▼
                 RESPONSE
```

## Architectural Components

### 1. Security Guardrails (`app/security/`)
- **Input Guard (`input_guard.py`)**: Sanitizes incoming user queries, blocking prompt injection and system jailbreak attempts.
- **Document Guard (`document_guard.py`)**: Checks uploaded files for indirect prompt injection, embedded instructions, and accidental secret exposures.
- **Tool Guard (`tool_guard.py`)**: Validates parameters and sanitizes math expressions before tool execution.
- **Output Guard (`output_guard.py`)**: Redacts PII (SSNs, credit cards) and sensitive API keys.

### 2. Query Analyzer & Routing (`app/graph/routing.py`, `app/graph/prompts.py`)
- Distinguishes between document-grounded financial queries, calculations (CAGR, margin variance), live web searches, and direct responses.

### 3. Business-Document RAG Engine (`app/rag/`)
- **Chunking (`chunking.py`)**: Layout-aware chunking preserving tables, financial notes, and section headings.
- **Embeddings (`embeddings.py`)**: Dense vector representation using `text-embedding-3-small`.
- **Retriever (`retriever.py`)**: HNSW-indexed cosine distance search in PostgreSQL via `pgvector` with strict tenant isolation.
- **Citation Validator (`citations.py`)**: Line-by-line verification ensuring answers reference verified source snippets.

### 4. Persistence & API Layer (`app/db/`, `app/routers/`, `app/services/`)
- Threaded connection pooling in PostgreSQL.
- Parameterized SQL for multi-turn sessions and message histories.
- Hashed API key authentication (SHA-256 for fast lookup, Bcrypt for storage).
