"""pgvector document retrieval interface with tenant isolation."""

import logging
from typing import Any

from app.db.connection import db_conn
from app.rag.embeddings import embeddings_service

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    query: str,
    tenant_id: str = "default",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve top-k document chunks matching the query using cosine distance in pgvector."""
    try:
        embedding = embeddings_service.embed_query(query)

        with db_conn() as conn:
            with conn.cursor() as cur:
                # Ensure documents table exists
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = 'documents'
                    )
                    """
                )
                row = cur.fetchone()
                if row is None or not row[0]:
                    logger.info("documents_table_missing")
                    return []

                cur.execute(
                    """
                    SELECT id, content, source, metadata, page_number, document_id, document_name,
                           (embedding <=> %(embedding)s::vector) AS distance
                    FROM documents
                    WHERE tenant_id = %(tenant_id)s
                    ORDER BY distance ASC
                    LIMIT %(top_k)s
                    """,
                    {
                        "embedding": str(embedding),
                        "tenant_id": tenant_id,
                        "top_k": top_k,
                    },
                )
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for r_id, content, source, meta, page_num, doc_id, doc_name, distance in rows:
            merged_meta = dict(meta) if isinstance(meta, dict) else {}
            if page_num is not None:
                merged_meta["page_number"] = page_num
            if doc_id:
                merged_meta["document_id"] = str(doc_id)
            if doc_name:
                merged_meta["document_name"] = doc_name
            results.append(
                {
                    "id": str(r_id),
                    "content": content,
                    "source": source,
                    "metadata": merged_meta,
                    "score": 1.0 - float(distance),
                    "page_number": page_num,
                    "document_id": str(doc_id) if doc_id else None,
                    "document_name": doc_name,
                }
            )

        return results

    except Exception as e:
        logger.warning("retrieval_failed", extra={"query": query, "error": str(e)})
        return []
