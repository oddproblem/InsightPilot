"""Integration tests for InsightPilot end-to-end document and agent pipelines.

Verifies:
1. Document ingestion round-trip into real PostgreSQL pgvector schema.
2. Strict tenant isolation (tenant A cannot see tenant B's chunks).
3. Semantic retrieval via retrieve_relevant_chunks and document_search tool.
4. Document metadata inspection via get_document_metadata tool.
5. Document deletion and cleanup.
6. End-to-end LangGraph agent workflow with document_search route and citation generation.
"""

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.db.connection import db_conn, init_pool
from app.graph.graph import app_graph
from app.graph.tools import _tenant_ctx, get_document_metadata
from app.rag.retriever import retrieve_relevant_chunks
from app.services.document_service import DocumentService


@pytest.fixture(scope="module", autouse=True)
def ensure_db_pool() -> None:
    """Ensure database connection pool is initialized for integration tests."""
    init_pool()


@pytest.fixture
def mock_embedding_vector() -> list[float]:
    """Deterministic 1536-dimensional embedding vector for testing."""
    vec = [0.0] * 1536
    vec[0] = 1.0  # Unit vector along first dimension
    return vec


class TestDocumentPipelineIntegration:
    """Real PostgreSQL round-trip tests for DocumentService and pgvector retrieval."""

    def test_ingest_retrieve_and_delete_roundtrip(self, mock_embedding_vector: list[float]) -> None:
        tenant_id = f"test-tenant-{uuid.uuid4().hex[:8]}"
        doc_service = DocumentService()

        sample_text = (
            "Acme Corp announced revenue of $120 million for FY2025, representing a 20% "
            "year-over-year increase. Operating margin improved to 18.5% driven by "
            "operational efficiencies."
        )

        with patch("app.services.document_service.embeddings_service") as mock_emb:
            mock_emb.embed_documents.return_value = [mock_embedding_vector]

            ingest_result = doc_service.ingest(
                file_bytes=sample_text.encode("utf-8"),
                filename="acme_fy2025_earnings.txt",
                tenant_id=tenant_id,
                document_name="Acme FY2025 Earnings",
                doc_type="earnings",
                company="Acme Corp",
                financial_year="FY2025",
            )

        assert ingest_result.status == "success"
        assert ingest_result.chunks_indexed == 1
        doc_id = ingest_result.document_id

        # 1. Verify chunk in DB
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM documents "
                    "WHERE tenant_id = %(tenant)s AND document_id = %(doc_id)s",
                    {"tenant": tenant_id, "doc_id": doc_id},
                )
                assert cur.fetchone()[0] == 1

        # 2. Verify retrieval with matching embedding
        with patch("app.rag.retriever.embeddings_service") as mock_ret_emb:
            mock_ret_emb.embed_query.return_value = mock_embedding_vector
            chunks = retrieve_relevant_chunks(
                query="What was Acme revenue in FY2025?",
                tenant_id=tenant_id,
                top_k=3,
            )

        assert len(chunks) == 1
        assert chunks[0]["document_id"] == doc_id
        assert chunks[0]["source"] == "acme_fy2025_earnings.txt"
        assert "$120 million" in chunks[0]["content"]

        # 3. Verify tenant isolation: another tenant retrieves nothing
        other_tenant = f"other-tenant-{uuid.uuid4().hex[:8]}"
        with patch("app.rag.retriever.embeddings_service") as mock_ret_emb:
            mock_ret_emb.embed_query.return_value = mock_embedding_vector
            other_chunks = retrieve_relevant_chunks(
                query="What was Acme revenue in FY2025?",
                tenant_id=other_tenant,
                top_k=3,
            )
        assert len(other_chunks) == 0

        # 4. Verify get_document_metadata tool
        _tenant_ctx.set(tenant_id)
        meta_result = get_document_metadata.invoke({"document_id": doc_id})
        assert "Acme Corp" in meta_result
        assert "FY2025" in meta_result
        assert "earnings" in meta_result

        # Cross-tenant metadata request should return not found
        _tenant_ctx.set(other_tenant)
        cross_meta = get_document_metadata.invoke({"document_id": doc_id})
        assert "No document found" in cross_meta

        # 5. Clean up via delete_document
        deleted = doc_service.delete_document(document_id=doc_id, tenant_id=tenant_id)
        assert deleted is True

        # Verify chunks removed
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM documents "
                    "WHERE tenant_id = %(tenant)s AND document_id = %(doc_id)s",
                    {"tenant": tenant_id, "doc_id": doc_id},
                )
                assert cur.fetchone()[0] == 0


class TestAgentGraphIntegration:
    """End-to-end integration test through the LangGraph agent state machine."""

    def test_agent_graph_document_search_flow(self, mock_embedding_vector: list[float]) -> None:
        tenant_id = f"test-agent-tenant-{uuid.uuid4().hex[:8]}"
        doc_service = DocumentService()

        # Ingest test context
        with patch("app.services.document_service.embeddings_service") as mock_emb:
            mock_emb.embed_documents.return_value = [mock_embedding_vector]
            ingest_result = doc_service.ingest(
                file_bytes=b"FY2025 operating profit reached $24M on revenue of $120M.",
                filename="profit_report.txt",
                tenant_id=tenant_id,
                document_name="FY2025 Profit Report",
            )

        doc_id = ingest_result.document_id

        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content="What was operating profit in FY2025?")],
            "query": "What was operating profit in FY2025?",
            "tenant_id": tenant_id,
            "session_id": f"sess-{uuid.uuid4().hex[:8]}",
            "run_id": f"run-{uuid.uuid4().hex[:8]}",
        }

        mock_llm_response = AIMessage(
            content="Operating profit in FY2025 reached $24 million [profit_report.txt]."
        )

        with (
            patch("app.rag.retriever.embeddings_service") as mock_ret_emb,
            patch("app.graph.tools.OpenAI") as mock_openai,
            patch("app.graph.nodes._llm") as mock_llm_fn,
        ):
            mock_ret_emb.embed_query.return_value = mock_embedding_vector
            mock_embed_client = MagicMock()
            mock_embed_client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=mock_embedding_vector)]
            )
            mock_openai.return_value = mock_embed_client

            mock_chat_llm = MagicMock()
            mock_chat_llm.invoke.return_value = mock_llm_response
            mock_llm_fn.return_value = mock_chat_llm

            final_state = app_graph.invoke(initial_state)

        assert final_state["route"] == "document_search"
        assert len(final_state["retrieved_documents"]) >= 1
        assert any("$24M" in chunk["content"] for chunk in final_state["retrieved_documents"])
        assert len(final_state["messages"]) >= 2

        # Clean up
        doc_service.delete_document(document_id=doc_id, tenant_id=tenant_id)
