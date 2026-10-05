from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class CitationItem(BaseModel):
    source: str
    snippet: str
    verified: bool = False
    page: int | None = None


class ToolTraceItem(BaseModel):
    tool_name: str
    tool_input: dict[str, Any] | str = Field(default_factory=dict)
    output: str = ""
    status: str = "ok"


class RunAgentResponse(BaseModel):
    session_id: str
    response: str
    run_id: str
    usage: TokenUsage
    route: str | None = None
    confidence: str | None = None
    citations: list[CitationItem] = Field(default_factory=list)
    tools: list[ToolTraceItem] = Field(default_factory=list)


class ApiKeyCreatedResponse(BaseModel):
    id: str
    key: str  # Plaintext — returned once only
    name: str
    role: str
    tenant_id: str
    created_at: datetime


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    role: str
    tenant_id: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime


class SessionResponse(BaseModel):
    session_id: str
    message_count: int
    created_at: datetime
    last_active_at: datetime
    messages: list[MessageResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str


class DetailedHealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    database: Literal["ok", "error"]
    graph: Literal["ok", "error"]
    checks: dict[str, str]


class ErrorResponse(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class IngestDocumentResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_indexed: int
    source: str
    status: str


class DocumentMetadataResponse(BaseModel):
    document_id: str
    document_name: str
    source: str
    doc_type: str | None
    company: str | None
    financial_year: str | None
    upload_date: datetime
    chunk_count: int
    page_count: int | None
