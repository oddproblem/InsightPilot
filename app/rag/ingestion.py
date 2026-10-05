"""Document ingestion workflow: parsing, chunking, embedding, and storage."""

import logging
from typing import Any

from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


def ingest_document_text(
    content: str,
    source: str,
    tenant_id: str = "default",
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict[str, Any]:
    """Ingest, chunk, embed, and store document in pgvector with tenant isolation."""
    meta = metadata or {}
    svc = DocumentService()
    res = svc.ingest(
        file_bytes=content.encode("utf-8"),
        filename=source,
        tenant_id=tenant_id,
        document_name=meta.get("document_name", source),
        doc_type=meta.get("doc_type", ""),
        company=meta.get("company", ""),
        financial_year=meta.get("financial_year", ""),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return {
        "chunks_indexed": res.chunks_indexed,
        "source": source,
        "status": res.status,
        "document_id": res.document_id,
    }
