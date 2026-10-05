"""Integration tests for the documents router.

Tests the full HTTP layer: auth enforcement, successful ingestion,
listing, retrieval, and deletion — all with mocked service calls.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from httpx import AsyncClient

from app.models.responses import DocumentMetadataResponse, IngestDocumentResponse

# Shared fake document metadata
FAKE_DOC_ID = str(uuid.uuid4())
FAKE_META = DocumentMetadataResponse(
    document_id=FAKE_DOC_ID,
    document_name="Annual Report FY2025",
    source="annual_report.pdf",
    doc_type="10-K",
    company="Acme Corp",
    financial_year="FY2025",
    upload_date=datetime(2025, 1, 15, tzinfo=UTC),
    chunk_count=42,
    page_count=10,
)
FAKE_INGEST = IngestDocumentResponse(
    document_id=FAKE_DOC_ID,
    document_name="Annual Report FY2025",
    chunks_indexed=42,
    source="annual_report.pdf",
    status="success",
)


# ---------------------------------------------------------------------------
# POST /v1/documents/ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_requires_auth(client: AsyncClient) -> None:
    """Upload without Authorization header must return 401."""
    response = await client.post(
        "/v1/documents/ingest",
        files={"file": ("report.txt", b"Revenue: $10M", "text/plain")},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_ingest_success(client: AsyncClient, user_key_header: dict[str, str]) -> None:
    """Successful upload returns 201 with document_id and chunk count."""
    with patch("app.routers.documents._service.ingest", return_value=FAKE_INGEST):
        response = await client.post(
            "/v1/documents/ingest",
            headers=user_key_header,
            files={"file": ("annual_report.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            params={"document_name": "Annual Report FY2025", "doc_type": "10-K"},
        )

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == FAKE_DOC_ID
    assert body["chunks_indexed"] == 42
    assert body["status"] == "success"


@pytest.mark.asyncio
async def test_ingest_empty_file_returns_400(
    client: AsyncClient, user_key_header: dict[str, str]
) -> None:
    """Empty file body must return 400."""
    response = await client.post(
        "/v1/documents/ingest",
        headers=user_key_header,
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body.get("code") == "EMPTY_FILE"


@pytest.mark.asyncio
async def test_ingest_unsupported_format_returns_422(
    client: AsyncClient, user_key_header: dict[str, str]
) -> None:
    """Unsupported file type propagates as 422 INGESTION_ERROR."""
    with patch(
        "app.routers.documents._service.ingest",
        side_effect=ValueError("Unsupported file type '.xlsx'"),
    ):
        response = await client.post(
            "/v1/documents/ingest",
            headers=user_key_header,
            files={"file": ("data.xlsx", b"PK...", "application/octet-stream")},
        )

    assert response.status_code == 422
    body = response.json()
    assert body.get("code") == "INGESTION_ERROR"


@pytest.mark.asyncio
async def test_ingest_tenant_isolation(
    client: AsyncClient, user_key_header: dict[str, str]
) -> None:
    """The tenant_id from the auth token is passed through to the service."""
    captured: list[dict] = []

    def capture_ingest(**kwargs):  # type: ignore[no-untyped-def]
        captured.append(kwargs)
        return FAKE_INGEST

    with patch("app.routers.documents._service.ingest", side_effect=capture_ingest):
        await client.post(
            "/v1/documents/ingest",
            headers=user_key_header,
            files={"file": ("report.txt", b"Some content", "text/plain")},
        )

    assert len(captured) == 1
    # tenant_id must be set (not None or empty)
    assert captured[0]["tenant_id"] not in (None, "")


# ---------------------------------------------------------------------------
# GET /v1/documents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_documents_requires_auth(client: AsyncClient) -> None:
    response = await client.get("/v1/documents")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_list_documents_empty(client: AsyncClient, user_key_header: dict[str, str]) -> None:
    with patch("app.routers.documents._service.list_documents", return_value=[]):
        response = await client.get("/v1/documents", headers=user_key_header)
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_documents_returns_items(
    client: AsyncClient, user_key_header: dict[str, str]
) -> None:
    with patch("app.routers.documents._service.list_documents", return_value=[FAKE_META]):
        response = await client.get("/v1/documents", headers=user_key_header)
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 1
    assert items[0]["document_name"] == "Annual Report FY2025"
    assert items[0]["chunk_count"] == 42


# ---------------------------------------------------------------------------
# GET /v1/documents/{document_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_document_not_found(client: AsyncClient, user_key_header: dict[str, str]) -> None:
    with patch("app.routers.documents._service.get_document", return_value=None):
        response = await client.get(f"/v1/documents/{FAKE_DOC_ID}", headers=user_key_header)
    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


@pytest.mark.asyncio
async def test_get_document_found(client: AsyncClient, user_key_header: dict[str, str]) -> None:
    with patch("app.routers.documents._service.get_document", return_value=FAKE_META):
        response = await client.get(f"/v1/documents/{FAKE_DOC_ID}", headers=user_key_header)
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == FAKE_DOC_ID
    assert body["doc_type"] == "10-K"
    assert body["financial_year"] == "FY2025"


# ---------------------------------------------------------------------------
# DELETE /v1/documents/{document_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_document_not_found(
    client: AsyncClient, user_key_header: dict[str, str]
) -> None:
    with patch("app.routers.documents._service.delete_document", return_value=False):
        response = await client.delete(f"/v1/documents/{FAKE_DOC_ID}", headers=user_key_header)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_success(
    client: AsyncClient, user_key_header: dict[str, str]
) -> None:
    with patch("app.routers.documents._service.delete_document", return_value=True):
        response = await client.delete(f"/v1/documents/{FAKE_DOC_ID}", headers=user_key_header)
    assert response.status_code == 204
