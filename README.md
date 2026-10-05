<div align="center">

<img src="frontend/assets/logo.jpg" alt="InsightPilot - Autonomous Financial Intelligence Agent" width="220" style="border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 0 25px rgba(6, 182, 212, 0.4);">

# InsightPilot

### Autonomous Business & Financial Document Intelligence System

A production-grade, multi-tenant agentic intelligence platform engineered for rigorous analysis of corporate disclosures (SEC Forms 10-K, 10-Q, 8-K, quarterly earnings calls, and financial statements). Combines multi-stage query routing, layout-aware RAG with PostgreSQL pgvector, an AST-sandboxed mathematical calculation engine, line-by-line citation verification, and defense-in-depth prompt injection guardrails.

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-FF6F00?logo=chainlink&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector_0.8.2-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

[Architecture](./docs/architecture.md) | [Evaluation Harness](./docs/evaluation.md) | [Threat Model](./docs/threat-model.md) | [Experiments](./docs/experiments.md) | [Attributions](./ATTRIBUTIONS.md)

</div>

---

## Executive Overview

Conventional retrieval-augmented generation (RAG) applications often fail in enterprise financial contexts. Naive vector retrieval splits structured tables mid-row, arbitrary token windows tear numeric footnotes from their header context, and language models hallucinate compound growth metrics when computing percentage changes across reporting cycles.

InsightPilot addresses these systemic limitations with a structured agentic architecture:
- **Financial Intent Analysis**: Before any vector search is executed, an analyzer evaluates query intent and dispatches execution to dedicated domain nodes: document vector retrieval, an abstract syntax tree (AST) sandboxed calculator, live web search, or direct conversational synthesis.
- **Layout-Aware Ingestion**: Document processing preserves Markdown table structures, financial section hierarchies, and multi-column balances.
- **Strict Multi-Tenant Isolation**: Every vector chunk, conversational session, and message is explicitly bound to a tenant identifier at the database layer with composite indexing on `(tenant_id, session_id)`.
- **Verifiable Citation Grounding**: Generated answers are systematically scanned to ensure factual assertions are linked to specific source passages and page numbers.
- **Defense-in-Depth Guardrails**: Inbound user queries, uploaded document payloads, and outbound model generations are filtered through dedicated security guards to neutralize direct and indirect prompt injections.

---

## User Interface

InsightPilot includes a responsive web interface designed for document exploration, real-time agent execution, and citation inspection.

### Financial Reasoning Dashboard

![InsightPilot Dashboard](docs/assets/dashboard_ui.png)

The dashboard provides three unified workspaces:
1. **Document Corpus Explorer (Left Panel)**: Supports multipart file uploads (PDF, TXT, DOCX, Markdown) with real-time indexing status, token accounting, and per-tenant corpus segregation.
2. **Conversational Agent Canvas (Center Panel)**: Displays multi-turn session dialogs, real-time server-sent event (SSE) streams, query routing badges, and active guardrail status indicators.
3. **Citation Inspector (Right Panel)**: Clicking any numerical assertion or footnote badge in the agent response reveals the exact source passage, similarity score, and document metadata extracted from PostgreSQL.

### Interactive OpenAPI Documentation

![InsightPilot API Documentation](docs/assets/api_docs.png)

Complete OpenAPI 3.1 specifications and Swagger UI are accessible at `/docs` during development, providing interactive execution for document ingestion, agent runs, session management, and API key administration.

---

## System Architecture

### 1. End-to-End Request Pipeline

```mermaid
flowchart TD
    Client["Client / Web UI"] -->|HTTP / SSE| ReverseProxy["Reverse Proxy / TLS Termination"]
    ReverseProxy -->|Port 8000| FastAPI["FastAPI Application"]
    
    subgraph MiddlewareLayer["Middleware Pipeline"]
        FastAPI --> LogMid["LoggingMiddleware (Correlation ID, JSON Logs)"]
        LogMid --> AuthMid["AuthMiddleware (Bearer API Key, Tenant Extraction)"]
    end
    
    subgraph RoutingLayer["API Routers"]
        AuthMid --> AgentRouter["/v1/agents (Chat, Streaming, Sessions)"]
        AuthMid --> DocRouter["/v1/documents (Upload, Ingest, List, Delete)"]
        AuthMid --> KeyRouter["/v1/keys (Create, Revoke, List)"]
        AuthMid --> HealthRouter["/health, /health/detailed"]
    end
    
    subgraph ServiceLayer["Application Services"]
        AgentRouter --> AgentSvc["AgentService"]
        DocRouter --> DocSvc["DocumentService"]
        KeyRouter --> KeySvc["ApiKeyService"]
    end
    
    subgraph StorageLayer["PostgreSQL + pgvector"]
        AgentSvc --> DB_Msg[("agent_messages")]
        AgentSvc --> DB_Sess[("agent_sessions")]
        DocSvc --> DB_Docs[("documents (1536-dim vectors)")]
        KeySvc --> DB_Keys[("api_keys")]
    end
```

---

### 2. LangGraph Agent Workflow State Machine

The core intelligence layer operates as a compiled LangGraph state graph. Unlike cyclical black-box agents that loop unpredictably, InsightPilot follows a deterministic directed acyclic graph (DAG) with guarded tool activation:

```mermaid
flowchart TD
    Start([User Query Received]) --> InputGuard["Input Security Guard<br/>(Prompt Injection & Jailbreak Defense)"]
    InputGuard --> QueryAnalyzer["Query Analyzer Node<br/>(Intent Classification)"]
    
    QueryAnalyzer --> BranchDecision{"Routing Decision"}
    
    BranchDecision -->|document_search| RetrieveNode["Retrieve Documents Node<br/>(pgvector Cosine Similarity)"]
    BranchDecision -->|calculator| CalcNode["Run Calculator Node<br/>(AST Sandboxed Math Engine)"]
    BranchDecision -->|web_search| WebNode["Tavily Web Search Node<br/>(Live Market Data)"]
    BranchDecision -->|direct| AnswerNode["Generate Answer Node<br/>(Direct LLM Synthesis)"]
    
    RetrieveNode --> AnswerNode
    CalcNode --> AnswerNode
    WebNode --> AnswerNode
    
    AnswerNode --> CiteNode["Cite Sources Node<br/>(Footnote Attribution)"]
    CiteNode --> OutputGuard["Output Security Guard<br/>(PII & Secret Sanitization)"]
    OutputGuard --> Complete([Streamed Response to Client])
```

---

### 3. Multi-Tenant RAG & Vector Pipeline

```mermaid
flowchart LR
    Upload["Financial Document<br/>(PDF, DOCX, TXT, MD)"] --> SecurityScan["Document Guard<br/>(Scan Hidden Instructions)"]
    SecurityScan --> Chunker["Boundary-Aware Chunker<br/>(Preserve Markdown Tables)"]
    Chunker --> Embedder["Embedding Generator<br/>(text-embedding-3-small)"]
    Embedder --> VectorDB[("PostgreSQL pgvector<br/>Tenant-Partitioned Chunks")]
    
    UserQuery["User Analytical Query"] --> QueryEmbed["Query Vectorization"]
    QueryEmbed --> CosineSearch["Cosine Similarity Search<br/>WHERE tenant_id = :tenant_id"]
    VectorDB --> CosineSearch
    CosineSearch --> TopK["Top-K Relevant Chunks"]
    TopK --> LLMContext["LLM Context Window"]
```

---

### 4. Database Schema & Multi-Tenant Isolation

```mermaid
erDiagram
    api_keys {
        uuid id PK
        varchar key_hash
        varchar lookup_hash UK
        varchar name
        varchar tenant_id
        varchar role
        timestamptz created_at
        timestamptz last_used_at
        timestamptz revoked_at
    }

    agent_sessions {
        uuid id PK
        varchar session_id UK
        varchar tenant_id
        timestamptz created_at
        timestamptz last_active_at
        int message_count
    }

    agent_messages {
        uuid id PK
        varchar session_id FK
        varchar tenant_id
        varchar role
        text content
        jsonb metadata
        timestamptz created_at
    }

    documents {
        uuid id PK
        varchar document_id
        varchar filename
        text content
        vector embedding
        jsonb metadata
        varchar tenant_id
        timestamptz created_at
    }

    agent_sessions ||--o{ agent_messages : "contains"
    api_keys ||--o{ agent_sessions : "authorizes"
```

---

## Core Engineering Features

### 1. Layout-Aware Financial Document Ingestion
Financial disclosures contain dense numeric tables, balance sheet footnotes, and segmented disclosures. Standard token splitters frequently sever tabular rows, destroying column associations. 

InsightPilot implements a specialized chunking engine (`app/rag/chunking.py`):
- Detects Markdown table boundaries (`|---|---|`) and prevents splitting across table rows.
- Maintains a default target of 500 tokens with 50-token contextual overlaps.
- Extracts document metadata including page numbers, header structures, and source filing names.

### 2. Strict Tenant-Isolated Vector Retrieval
All data operations are strictly scoped to the authenticated caller's `tenant_id` (`app/rag/retriever.py`):
- Embeddings are indexed using 1536-dimensional vectors via PostgreSQL `pgvector`.
- The database schema enforces compound indexing on `(tenant_id, session_id)` and `(tenant_id, document_id)`.
- Cosine similarity matching explicitly injects `WHERE tenant_id = :tenant_id` at the SQL query level, preventing cross-tenant information disclosure.

### 3. AST-Sandboxed Financial Calculator
Financial calculations (such as Compound Annual Growth Rate [CAGR], operating margin variance, and percentage changes) require deterministic arithmetic rather than language model approximations.

InsightPilot implements an abstract syntax tree evaluator (`app/graph/tools.py`):
- Parses formulas into a Python AST and only allows safe mathematical operators (`Add`, `Sub`, `Mult`, `Div`, `Pow`).
- Disallows all standard built-ins (`eval`, `exec`, `open`, `__import__`) to prevent arbitrary code execution vulnerabilities.
- Handles standard financial shorthand (such as `B` for billions and `M` for millions).

### 4. Verification & Citation Footnotes
Every factual claim generated from retrieved context includes structured citation anchors (`app/rag/citations.py`):
- References are formatted as clickable numeric footnotes (e.g., `[1]`, `[2]`).
- Each footnote links to the corresponding chunk UUID, source document name, and similarity score.
- The UI Citation Inspector displays the raw, unedited context chunk alongside the agent's synthesis.

### 5. Multi-Layer Guardrails
Security defenses are implemented as modular filters (`app/security/`):
- **Input Guard**: Scans incoming messages for prompt injection patterns, system prompt override attempts, and delimiters.
- **Document Guard**: Scans uploaded files for hidden instructions, zero-width characters, and indirect jailbreak attempts before vectorization.
- **Output Guard**: Verifies outgoing generation text, scrubbing potential credential leaks or internal prompt exposures.

---

## API Reference

All requests must include standard Bearer authentication headers:
```http
Authorization: Bearer <api-key>
```

### Endpoints Overview

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/health` | Service liveness probe | No |
| `GET` | `/health/detailed` | Database connection & graph readiness probe | Yes (Admin) |
| `POST` | `/v1/agent/run` | Synchronous agent execution | Yes |
| `POST` | `/v1/agents/chat/stream` | Server-Sent Events (SSE) streaming chat | Yes |
| `GET` | `/v1/agent/sessions/{session_id}` | Retrieve complete conversation history | Yes |
| `POST` | `/v1/documents/ingest` | Upload and vectorize document (PDF, TXT, DOCX, MD) | Yes |
| `GET` | `/v1/documents` | List all indexed documents for caller tenant | Yes |
| `GET` | `/v1/documents/{document_id}` | Retrieve metadata and chunks for specific document | Yes |
| `DELETE` | `/v1/documents/{document_id}` | Delete document and remove all vector embeddings | Yes |
| `POST` | `/v1/keys` | Generate new API key with specific role | Yes (Admin) |
| `DELETE` | `/v1/keys/{key_id}` | Revoke an existing API key | Yes (Admin) |
| `GET` | `/v1/auth/dev-key` | Retrieve local development credentials | No (Dev only) |

---

### Request & Response Examples

#### 1. Document Ingestion (`POST /v1/documents/ingest`)

Upload a file using `multipart/form-data`:

```bash
curl -X POST http://localhost:8000/v1/documents/ingest \
  -H "Authorization: Bearer <api-key>" \
  -F "file=@annual_report_2025.pdf" \
  -F "company_name=MedTech Global" \
  -F "period=FY2025"
```

Response (`201 Created`):
```json
{
  "document_id": "doc_9c82b17a",
  "filename": "annual_report_2025.pdf",
  "chunks_created": 24,
  "status": "indexed",
  "tenant_id": "default",
  "created_at": "2026-10-05T10:30:00Z"
}
```

---

#### 2. Agent Execution (`POST /v1/agent/run`)

Execute a multi-stage financial query:

```bash
curl -X POST http://localhost:8000/v1/agent/run \
  -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-financial-analysis",
    "message": "What was the operating margin decline in FY2025 and why did it occur?"
  }'
```

Response (`200 OK`):
```json
{
  "session_id": "session-financial-analysis",
  "response": "Operating margin declined by 180 basis points from 24.2% to 22.4% in FY2025 [1]. This contraction was primarily driven by higher research and development investments in APAC expansion and transitory supply chain friction [2].",
  "run_id": "run_20261005T103522",
  "usage": {
    "input_tokens": 642,
    "output_tokens": 128
  },
  "citations": [
    {
      "citation_id": 1,
      "source_document": "annual_report_2025.pdf",
      "chunk_id": "chk_3f92",
      "snippet": "Operating margin was 22.4% for fiscal 2025, compared to 24.2% in fiscal 2024..."
    }
  ]
}
```

---

## Local Development & Installation

### Prerequisites
- Python 3.11 or higher
- PostgreSQL with `pgvector` extension enabled (or an active Supabase PostgreSQL instance)
- OpenRouter or OpenAI API key

---

### Step 1: Clone Repository
```bash
git clone https://github.com/oddproblem/InsightPilot.git
cd InsightPilot
```

---

### Step 2: Environment Configuration
Create a `.env` configuration file from the template:
```bash
cp .env.example .env
```

Configure your credentials in `.env`:
```ini
# Database (PostgreSQL with pgvector)
DATABASE_URL=postgresql://postgres.yourproject:yourpassword@aws-0-region.pooler.supabase.com:5432/postgres?sslmode=require
POSTGRES_PASSWORD=yourpassword
DATABASE_POOL_SIZE=10

# Language Model Configuration
LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=sk-or-v1-...

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
LOG_FORMAT=text
```

*Security Note: The `.env` file is excluded in `.gitignore` and must never be committed.*

---

### Step 3: Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

---

### Step 4: Run Database Migrations
Execute Alembic migrations to apply schemas up to head:
```bash
alembic upgrade head
```

---

### Step 5: Generate Initial Admin Credentials
```bash
python scripts/create_api_key.py --name "local-admin" --role "admin"
```

Save the generated `ip_...` token for API authentication.

---

### Step 6: Start Application Server
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Access the dashboard at `http://localhost:8000` or the interactive API docs at `http://localhost:8000/docs`.

---

## Testing & Quality Assurance

InsightPilot maintains an automated test suite verifying routing logic, vector retrieval, security boundaries, and tenant isolation:

```bash
# Run complete test suite (124 tests)
pytest -q

# Run with verbose diagnostic output
pytest -v

# Run linting and code quality analysis
ruff check app/ tests/ scripts/

# Run static type verification
mypy app/ --ignore-missing-imports
```

---

## Production Deployment

InsightPilot is containerized as a standalone multi-stage image.

### Production Dockerfile
The production container runs Gunicorn managing Uvicorn worker processes:
```bash
docker build -t insightpilot:latest .
docker run -d \
  --name insightpilot \
  -p 8000:8000 \
  --env-file .env \
  insightpilot:latest
```

### 1-Click Render Deployment
The repository includes a ready-to-use [`render.yaml`](./render.yaml) blueprint specification:
1. Connect your repository in the [Render Dashboard](https://dashboard.render.com/).
2. Select **New Web Service** and choose **Docker Runtime**.
3. Provide your environment variables (`DATABASE_URL`, `POSTGRES_PASSWORD`, `OPENROUTER_API_KEY`).
4. Set Health Check path to `/health`.
5. Deploy.

*Note for URL passwords: If your database password contains special characters such as `@`, ensure it is properly URL-encoded (e.g., `%40`) in the `DATABASE_URL` string.*

---

## Repository Structure

```text
InsightPilot/
├── app/
│   ├── config.py                 # Pydantic v2 application settings
│   ├── main.py                   # FastAPI factory, lifespan, and static mounting
│   ├── db/
│   │   ├── connection.py         # Threaded connection pool management
│   │   └── queries.py            # Parameterized multi-tenant SQL queries
│   ├── graph/
│   │   ├── graph.py              # Compiled LangGraph workflow definition
│   │   ├── nodes.py              # Execution nodes (Analyze, Retrieve, Math, Generate)
│   │   ├── routing.py            # Financial intent regex & semantic routing
│   │   ├── state.py              # AgentState typed data structures
│   │   └── tools.py              # AST-sandboxed calculator and vector tools
│   ├── middleware/
│   │   ├── auth.py               # Bearer token validation and tenant propagation
│   │   └── logging.py            # Structured JSON request logging
│   ├── models/
│   │   ├── requests.py           # Pydantic request payloads
│   │   └── responses.py          # Unified response schemas and error envelopes
│   ├── rag/
│   │   ├── chunking.py           # Layout-aware markdown table chunker
│   │   ├── citations.py          # Footnote attribution engine
│   │   ├── embeddings.py         # Dense vector embedding generator
│   │   ├── ingestion.py          # PDF/DOCX/TXT file parser
│   │   └── retriever.py          # Isolated pgvector cosine similarity search
│   ├── routers/
│   │   ├── agents.py             # Chat, streaming, and session endpoints
│   │   ├── api_keys.py           # API credential administration
│   │   ├── documents.py          # Corpus management and ingestion endpoints
│   │   └── health.py             # Liveness and readiness endpoints
│   ├── security/
│   │   ├── document_guard.py     # Document payload injection scanner
│   │   ├── input_guard.py        # Query prompt injection defense
│   │   ├── output_guard.py       # PII and sensitive data scrubbing
│   │   └── tool_guard.py         # Tool parameter authorization
│   └── services/
│       ├── agent_service.py      # LangGraph invocation and message persistence
│       ├── api_key_service.py    # Key hashing and lookup validation
│       └── document_service.py   # Document lifecycle management
├── docs/
│   ├── architecture.md           # Deep-dive architectural specification
│   ├── evaluation.md             # Benchmark evaluation methodology
│   ├── experiments.md            # RAG experimentation logs
│   ├── threat-model.md           # Security analysis and threat mitigation
│   └── assets/                   # Architecture diagrams and UI screenshots
├── frontend/
│   ├── index.html                # Responsive dashboard structure
│   ├── style.css                 # Dark-mode styling tokens
│   ├── app.js                    # Client state management and SSE streaming
│   └── assets/                   # UI branding and visual assets
├── migrations/
│   ├── versions/                 # Alembic versioned schema migrations
│   └── env.py                    # Migration database configuration
├── scripts/
│   ├── create_api_key.py         # Administrative key generation utility
│   ├── revoke_api_key.py         # Key revocation utility
│   ├── health_check.py           # Container health probe
│   └── test_live_pgvector_roundtrip.py # End-to-end vector pipeline verification
├── tests/
│   ├── integration/              # Live database and RAG integration tests
│   ├── security/                 # Tenant isolation and guardrail tests
│   ├── test_graph/               # LangGraph node and execution tests
│   └── unit/                     # Chunker, router, and tool unit tests
├── Dockerfile                    # Multi-stage production container definition
├── Makefile                      # Developer workflow orchestration
├── pyproject.toml                # Project metadata and dependencies
└── render.yaml                   # Cloud PaaS deployment blueprint
```

---

## License

InsightPilot is released under the [MIT License](./LICENSE).
