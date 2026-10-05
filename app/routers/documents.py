"""Documents router: upload, list, inspect, and delete documents.

Endpoints:
  POST   /v1/documents/ingest          — upload a file + optional metadata
  GET    /v1/documents                 — list all documents for the tenant
  GET    /v1/documents/{document_id}   — metadata for a single document
  DELETE /v1/documents/{document_id}   — delete all chunks for a document
"""

import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.middleware.auth import require_role
from app.models.responses import DocumentMetadataResponse, IngestDocumentResponse
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/documents", tags=["documents"])

_service = DocumentService()


def _tenant_id(request: Request) -> str:
    """Extract tenant_id from the request state (set by AuthMiddleware)."""
    return str(getattr(request.state, "tenant_id", "default"))


# ---------------------------------------------------------------------------
# POST /v1/documents/ingest
# ---------------------------------------------------------------------------


@router.post(
    "/ingest",
    response_model=IngestDocumentResponse,
    status_code=201,
    summary="Upload and ingest a document",
    description=(
        "Upload a PDF, TXT, DOCX, or Markdown file. "
        "The system extracts text, chunks it, embeds each chunk, "
        "and stores it in the pgvector knowledge base for the tenant."
    ),
    dependencies=[require_role("user")],
)
async def ingest_document(
    request: Request,
    file: UploadFile,
    document_name: str = "",
    doc_type: str = "",
    company: str = "",
    financial_year: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> IngestDocumentResponse:
    """Ingest a document into the tenant's knowledge base."""
    tenant_id = _tenant_id(request)

    if file.filename is None or file.filename.strip() == "":
        raise HTTPException(
            status_code=400,
            detail={"code": "MISSING_FILENAME", "message": "No filename provided."},
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail={"code": "EMPTY_FILE", "message": "Uploaded file is empty."},
        )

    try:
        result = _service.ingest(
            file_bytes=file_bytes,
            filename=file.filename,
            tenant_id=tenant_id,
            document_name=document_name,
            doc_type=doc_type,
            company=company,
            financial_year=financial_year,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "INGESTION_ERROR", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.error(
            "document_ingest_failed",
            extra={"error": str(exc), "doc_filename": file.filename},
        )
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "message": "Document ingestion failed."},
        ) from exc

    return result


# ---------------------------------------------------------------------------
# GET /v1/documents
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[DocumentMetadataResponse],
    summary="List all documents for the tenant",
    dependencies=[require_role("user")],
)
async def list_documents(request: Request) -> list[DocumentMetadataResponse]:
    """Return file-level metadata for every document in the caller's tenant."""
    tenant_id = _tenant_id(request)
    return _service.list_documents(tenant_id=tenant_id)


# ---------------------------------------------------------------------------
# GET /v1/documents/{document_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{document_id}",
    response_model=DocumentMetadataResponse,
    summary="Get metadata for a single document",
    dependencies=[require_role("user")],
)
async def get_document(document_id: str, request: Request) -> DocumentMetadataResponse:
    """Return metadata (type, company, FY, chunk count, etc.) for one document."""
    tenant_id = _tenant_id(request)
    doc = _service.get_document(document_id=document_id, tenant_id=tenant_id)
    if doc is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"Document {document_id} not found."},
        )
    return doc


# ---------------------------------------------------------------------------
# DELETE /v1/documents/{document_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/{document_id}",
    status_code=204,
    summary="Delete a document and all its chunks",
    dependencies=[require_role("user")],
)
async def delete_document(document_id: str, request: Request) -> None:
    """Remove all vector chunks for the specified document from the tenant's knowledge base."""
    tenant_id = _tenant_id(request)
    deleted = _service.delete_document(document_id=document_id, tenant_id=tenant_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"Document {document_id} not found."},
        )
