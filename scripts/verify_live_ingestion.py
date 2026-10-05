"""Live verification of document ingestion, pgvector storage, and agent retrieval.

Tests the full end-to-end chain against live Supabase PostgreSQL and OpenRouter:
  real PDF
  -> POST /v1/documents/ingest
  -> pypdf extraction
  -> chunking
  -> real embedding API call (OpenRouter openai/text-embedding-3-small)
  -> document/chunk insertion into live Supabase PostgreSQL
  -> POST /v1/agent/run
  -> query embedding
  -> pgvector similarity retrieval (<=>)
  -> LangGraph answer generation (OpenRouter openai/gpt-4o-mini)
  -> citation validation
  -> final API response
"""

import io
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.connection import db_conn
from app.main import app
from app.services.api_key_service import ApiKeyService
from app.services.document_service import DocumentService


def generate_sample_pdf() -> bytes:
    """Generate a clean 2-page financial report PDF with reportlab."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    # Page 1: Financial Performance Highlights
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "Acme Healthtech Q3 2025 Financial Results")
    c.setFont("Helvetica", 11)
    c.drawString(
        72,
        690,
        "Acme Healthtech reported consolidated revenue of USD 48.2 million "
        "for the third quarter of 2025, up 17 percent year-over-year.",
    )
    c.drawString(
        72,
        670,
        "Operating profit expanded to USD 9.6 million, delivering an operating margin "
        "of 19.9 percent compared to 16.2 percent in Q3 2024.",
    )
    c.drawString(
        72,
        650,
        "Net income reached USD 7.8 million, with diluted earnings per share of USD 0.42.",
    )
    c.showPage()

    # Page 2: Segment Breakdown and Cash Position
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, "Segment Performance and Balance Sheet")
    c.setFont("Helvetica", 11)
    c.drawString(
        72,
        690,
        "Enterprise AI Platform subscription revenue contributed USD 31.4 million, "
        "representing 65 percent of total company revenues.",
    )
    c.drawString(
        72,
        670,
        "Professional services and clinical integration revenue stood at USD 16.8 million.",
    )
    c.drawString(
        72,
        650,
        "Cash, cash equivalents, and short-term marketable securities totaled USD 112.5 million "
        "with zero long-term debt on the balance sheet.",
    )
    c.save()
    return buf.getvalue()


def run_live_verification() -> dict[str, Any]:
    tenant_id = f"verify-{uuid.uuid4().hex[:8]}"
    session_id = f"sess-{uuid.uuid4().hex[:8]}"

    evidence: dict[str, Any] = {}

    with TestClient(app) as client:
        # Step 0: Create dedicated verification API key in live PostgreSQL
        key_svc = ApiKeyService()
        raw_key, key_info = key_svc.create_key(
            name="Live Ingestion Verifier",
            role="user",
            tenant_id=tenant_id,
        )
        evidence["tenant_id"] = tenant_id
        evidence["api_key_created"] = True
        auth_headers = {"Authorization": f"Bearer {raw_key}"}

        # Step 1: Generate real PDF with financial metrics
        pdf_bytes = generate_sample_pdf()
        evidence["pdf_bytes_len"] = len(pdf_bytes)

        # Step 2: Upload PDF via POST /v1/documents/ingest
        upload_resp = client.post(
            "/v1/documents/ingest",
            headers=auth_headers,
            files={"file": ("acme_healthtech_q3_2025.pdf", pdf_bytes, "application/pdf")},
            data={
                "document_name": "Acme Healthtech Q3 2025 Financial Results",
                "doc_type": "earnings",
                "company": "Acme Healthtech",
                "financial_year": "Q3 2025",
                "chunk_size": 1000,
                "chunk_overlap": 150,
            },
        )
        evidence["upload_status_code"] = upload_resp.status_code
        evidence["upload_response"] = upload_resp.json()

        if upload_resp.status_code != 201:
            raise RuntimeError(f"Ingest failed: {upload_resp.text}")

        ingest_data = upload_resp.json()
        doc_id = ingest_data["document_id"]
        evidence["document_id"] = doc_id
        evidence["chunks_indexed"] = ingest_data.get("chunks_indexed", 0)

        # Step 3: Verify live PostgreSQL database storage
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, document_name, source, page_number, chunk_index,
                           content, length(content), (embedding IS NOT NULL) AS has_embedding
                    FROM documents
                    WHERE tenant_id = %(tenant_id)s AND document_id = %(doc_id)s
                    ORDER BY chunk_index ASC
                    """,
                    {"tenant_id": tenant_id, "doc_id": doc_id},
                )
                db_chunks = cur.fetchall()
                evidence["db_chunk_count"] = len(db_chunks)
                evidence["db_chunks_sample"] = [
                    {
                        "page_number": row[3],
                        "chunk_index": row[4],
                        "content_length": row[6],
                        "has_embedding": row[7],
                        "snippet": row[5][:100] + "...",
                    }
                    for row in db_chunks
                ]

        # Step 4: Run Agent query via POST /v1/agent/run
        agent_query = (
            "Based on the uploaded financial report, what was Acme Healthtech's total revenue, "
            "operating margin, and cash position in Q3 2025?"
        )
        agent_resp = client.post(
            "/v1/agent/run",
            headers=auth_headers,
            json={
                "message": agent_query,
                "session_id": session_id,
            },
        )
        evidence["agent_status_code"] = agent_resp.status_code
        evidence["agent_response"] = agent_resp.json()

        if agent_resp.status_code != 200:
            raise RuntimeError(f"Agent run failed: {agent_resp.text}")

        agent_data = agent_resp.json()
        final_text = agent_data.get("response", "")
        evidence["agent_response_text"] = final_text
        evidence["agent_usage"] = agent_data.get("usage", {})

        # Step 5: Verify session and messages persisted in live DB
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM agent_messages
                    WHERE tenant_id = %(tenant_id)s AND session_id = %(session_id)s
                    ORDER BY created_at ASC
                    """,
                    {"tenant_id": tenant_id, "session_id": session_id},
                )
                stored_msgs = cur.fetchall()
                evidence["persisted_messages_count"] = len(stored_msgs)
                evidence["persisted_messages"] = [
                    {"role": r[0], "snippet": r[1][:120]} for r in stored_msgs
                ]

        # Step 6: Clean up test artifacts from live database
        doc_svc = DocumentService()
        deleted = doc_svc.delete_document(document_id=doc_id, tenant_id=tenant_id)
        evidence["document_cleanup"] = deleted

        key_svc.revoke_key(key_id=key_info.id, tenant_id=tenant_id)
        evidence["key_revoked"] = True

    return evidence


if __name__ == "__main__":
    print("=" * 80)
    print("INSIGHTPILOT LIVE INGESTION & AGENT PIPELINE VERIFICATION")
    print("=" * 80)
    results = run_live_verification()
    print("\nEXECUTION SUMMARY:")
    print(json.dumps(results, indent=2))
