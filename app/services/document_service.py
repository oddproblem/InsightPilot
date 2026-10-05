"""Document service: ingestion, listing, metadata retrieval, and deletion.

This is the only layer that touches the documents table directly.
All SQL is imported from app.db.queries (parameterized, no string interpolation).
"""

import io
import json
import logging
import uuid
from typing import Any

from app.db.connection import db_conn
from app.db.queries import (
    DELETE_DOCUMENT,
    GET_DOCUMENT_BY_ID,
    INSERT_DOCUMENT_CHUNK,
    LIST_DOCUMENTS,
)
from app.models.responses import DocumentMetadataResponse, IngestDocumentResponse
from app.rag.chunking import chunk_text
from app.rag.embeddings import embeddings_service
from app.security.document_guard import check_document_safety

logger = logging.getLogger(__name__)

# Maximum raw file size accepted (20 MB)
MAX_FILE_BYTES = 20 * 1024 * 1024

# Accepted MIME / extension types
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


class DocumentService:
    """Service layer for document ingestion and retrieval."""

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        tenant_id: str,
        document_name: str = "",
        doc_type: str = "",
        company: str = "",
        financial_year: str = "",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> IngestDocumentResponse:
        """Extract text from a file, chunk, embed, and store in pgvector.

        Supports: .pdf, .txt, .docx, .md

        Args:
            file_bytes:     Raw bytes of the uploaded file.
            filename:       Original filename (used to detect format & as source label).
            tenant_id:      Tenant scope for the document.
            document_name:  Human-readable name override; defaults to filename.
            doc_type:       Optional document type hint (e.g. '10-K').
            company:        Optional company name.
            financial_year: Optional fiscal year string.
            chunk_size:     Target characters per chunk.
            chunk_overlap:  Characters of overlap between consecutive chunks.

        Returns:
            IngestDocumentResponse with document_id, chunk count, and status.
        """
        if len(file_bytes) > MAX_FILE_BYTES:
            raise ValueError(f"File too large: {len(file_bytes)} bytes (max {MAX_FILE_BYTES})")

        ext = _file_extension(filename)
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Extract text + page-number mapping
        pages: list[tuple[int, str]] = _extract_text_by_page(file_bytes, ext)
        if not pages:
            return IngestDocumentResponse(
                document_id=str(uuid.uuid4()),
                document_name=document_name.strip() or filename,
                chunks_indexed=0,
                source=filename,
                status="empty",
            )

        # Security scan on raw text
        full_text = "\n".join(text for _, text in pages)
        safety_flags = check_document_safety(full_text)
        if safety_flags:
            logger.warning(
                "document_safety_flags",
                extra={"doc_filename": filename, "tenant_id": tenant_id, "flags": safety_flags},
            )
            raise ValueError(f"Document rejected by security scan: {'; '.join(safety_flags)}")

        doc_name = document_name.strip() or filename
        document_id = str(uuid.uuid4())
        source = filename

        # Build metadata template
        base_metadata: dict[str, Any] = {
            "doc_type": doc_type or None,
            "company": company or None,
            "financial_year": financial_year or None,
            "filename": filename,
        }

        # Chunk each page independently to preserve page numbers
        all_chunks = []
        for page_num, page_text in pages:
            page_meta = {**base_metadata, "page_number": page_num}
            page_chunks = chunk_text(
                text=page_text,
                source=source,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=page_meta,
            )
            for chunk in page_chunks:
                all_chunks.append((page_num, chunk))

        if not all_chunks:
            return IngestDocumentResponse(
                document_id=document_id,
                document_name=doc_name,
                chunks_indexed=0,
                source=source,
                status="empty",
            )

        # Embed all chunks in a single API call
        chunk_texts = [chunk.content for _, chunk in all_chunks]
        embeddings = embeddings_service.embed_documents(chunk_texts)

        # Persist to DB
        with db_conn() as conn:
            with conn.cursor() as cur:
                for idx, ((page_num, chunk), emb) in enumerate(
                    zip(all_chunks, embeddings, strict=True)
                ):
                    cur.execute(
                        INSERT_DOCUMENT_CHUNK,
                        {
                            "id": str(uuid.uuid4()),
                            "tenant_id": tenant_id,
                            "document_id": document_id,
                            "document_name": doc_name,
                            "source": source,
                            "content": chunk.content,
                            "page_number": page_num,
                            "chunk_index": idx,
                            "doc_type": doc_type or None,
                            "company": company or None,
                            "financial_year": financial_year or None,
                            "metadata": json.dumps(chunk.metadata),
                            "embedding": str(emb),
                        },
                    )

        logger.info(
            "document_ingested",
            extra={
                "document_id": document_id,
                "document_name": doc_name,
                "chunks": len(all_chunks),
                "tenant_id": tenant_id,
            },
        )

        return IngestDocumentResponse(
            document_id=document_id,
            document_name=doc_name,
            chunks_indexed=len(all_chunks),
            source=source,
            status="success",
        )

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_documents(self, tenant_id: str) -> list[DocumentMetadataResponse]:
        """Return file-level metadata for every document in the tenant."""
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(LIST_DOCUMENTS, {"tenant_id": tenant_id})
                rows = cur.fetchall()

        results = []
        for row in rows:
            doc_id, doc_name, source, dtype, company, fy, upload_date, chunks, pages = row
            results.append(
                DocumentMetadataResponse(
                    document_id=str(doc_id),
                    document_name=doc_name,
                    source=source,
                    doc_type=dtype,
                    company=company,
                    financial_year=fy,
                    upload_date=upload_date,
                    chunk_count=int(chunks),
                    page_count=int(pages) if pages else None,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_document(self, document_id: str, tenant_id: str) -> DocumentMetadataResponse | None:
        """Return metadata for a single document; None if not found."""
        try:
            parsed_id = str(uuid.UUID(document_id))
        except ValueError:
            return None

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    GET_DOCUMENT_BY_ID,
                    {"document_id": parsed_id, "tenant_id": tenant_id},
                )
                row = cur.fetchone()

        if row is None:
            return None

        doc_id, doc_name, source, dtype, company, fy, upload_date, chunks, pages = row
        return DocumentMetadataResponse(
            document_id=str(doc_id),
            document_name=doc_name,
            source=source,
            doc_type=dtype,
            company=company,
            financial_year=fy,
            upload_date=upload_date,
            chunk_count=int(chunks),
            page_count=int(pages) if pages else None,
        )

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str, tenant_id: str) -> bool:
        """Delete all chunks belonging to a document. Returns True if anything was deleted."""
        try:
            parsed_id = str(uuid.UUID(document_id))
        except ValueError:
            return False

        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    DELETE_DOCUMENT,
                    {"document_id": parsed_id, "tenant_id": tenant_id},
                )
                deleted = bool(cur.rowcount and cur.rowcount > 0)

        if deleted:
            logger.info(
                "document_deleted",
                extra={"document_id": document_id, "tenant_id": tenant_id},
            )
        return deleted


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _file_extension(filename: str) -> str:
    """Return lowercased file extension including the dot, e.g. '.pdf'."""
    dot = filename.rfind(".")
    if dot == -1:
        return ""
    return filename[dot:].lower()


def _extract_text_by_page(file_bytes: bytes, ext: str) -> list[tuple[int, str]]:
    """Extract text keyed by 1-indexed page number.

    Returns a list of (page_number, page_text) tuples.
    For formats without page concepts (.txt, .md), returns a single entry with page=1.
    """
    if ext in (".txt", ".md"):
        text = file_bytes.decode("utf-8", errors="replace").strip()
        return [(1, text)] if text else []

    if ext == ".pdf":
        return _extract_pdf_pages(file_bytes)

    if ext == ".docx":
        return _extract_docx_pages(file_bytes)

    return []


def _extract_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text per page from a PDF using pypdf."""
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((i, text))
        return pages
    except ImportError:
        logger.warning("pypdf_not_installed — falling back to raw text decode")
        # Graceful degradation: treat as plain bytes
        text = file_bytes.decode("utf-8", errors="replace").strip()
        return [(1, text)] if text else []
    except Exception as exc:
        logger.error("pdf_extraction_failed", extra={"error": str(exc)})
        raise ValueError(f"PDF parsing failed: {exc}") from exc


def _extract_docx_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text from a DOCX file using python-docx.

    DOCX has no hard page boundaries; we group every 30 paragraphs as a
    logical 'page' to preserve some structural metadata.
    """
    try:
        import docx

        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        if not paragraphs:
            return []

        # Group into logical pages of ~30 paragraphs
        page_size = 30
        pages = []
        for page_idx, start in enumerate(range(0, len(paragraphs), page_size), start=1):
            page_text = "\n\n".join(paragraphs[start : start + page_size])
            if page_text.strip():
                pages.append((page_idx, page_text))
        return pages
    except ImportError:
        logger.warning("python-docx_not_installed — falling back to raw text decode")
        text = file_bytes.decode("utf-8", errors="replace").strip()
        return [(1, text)] if text else []
    except Exception as exc:
        logger.error("docx_extraction_failed", extra={"error": str(exc)})
        raise ValueError(f"DOCX parsing failed: {exc}") from exc
