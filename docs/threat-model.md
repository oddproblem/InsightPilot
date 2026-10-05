# InsightPilot Threat Model & Security Posture

This document outlines the threat landscape, attack vectors, and implemented defensive controls for the InsightPilot platform.

## 1. Threat Vectors

| Threat Vector | Description | Severity | Defensive Control |
|---|---|---|---|
| **Direct Prompt Injection** | Adversary submits adversarial instructions to bypass safety guidelines or leak system prompts. | High | `app/security/input_guard.py` pattern filters & system prompt boundary encapsulation. |
| **Indirect Prompt Injection** | Malicious instructions hidden within uploaded business documents (e.g., hidden white text, HTML comments). | Critical | `app/security/document_guard.py` scans during ingestion and sanitizes untrusted file content. |
| **Cross-Tenant Data Leakage** | User from Tenant A queries or retrieves documents belonging to Tenant B. | Critical | Mandatory `tenant_id` filtering in all database queries and pgvector similarity lookups. |
| **Arbitrary Code Execution in Tools** | Malicious mathematical expressions evaluated in calculator tool (e.g., `__import__`, `eval`). | Critical | `app/security/tool_guard.py` strictly restricts calculator characters to safe math operators. |
| **Sensitive Data & PII Exposure** | Financial documents containing SSNs, credit card numbers, or internal keys reflected in responses. | High | `app/security/output_guard.py` applies regex redaction prior to delivering API responses. |
| **Hallucinated Financial Advice** | Model outputs ungrounded numbers or misattributed citations. | High | `app/rag/citations.py` checks that facts match retrieved evidence; sets `insufficient_evidence` flag if absent. |

## 2. Security Boundaries

1. **Network & Auth Boundary**:
   - Every request is validated by `AuthMiddleware` using Bearer API keys hashed with SHA-256 for fast lookup.
   - Admin routes (`POST /v1/keys`, `DELETE /v1/keys`) strictly require the `admin` role.

2. **Data Boundary**:
   - Documents and sessions are partitioned by `tenant_id`.
   - Vector queries enforce `WHERE tenant_id = %(tenant_id)s`.

3. **Tool Boundary**:
   - Calculators use isolated math engines with input character whitelisting.
   - External web searches via Tavily enforce query length limits.
