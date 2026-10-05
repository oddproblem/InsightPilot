from typing import Literal

from pydantic import BaseModel, Field


class RunAgentRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=255)
    message: str = Field(..., min_length=1, max_length=10000)
    stream: bool = False


class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    role: Literal["user", "admin"] = "user"
    tenant_id: str = Field(default="default", min_length=1, max_length=100)


class IngestDocumentRequest(BaseModel):
    """Optional metadata provided alongside an uploaded file."""

    document_name: str = Field(default="", max_length=500)
    doc_type: str = Field(default="", max_length=100)  # e.g. '10-K', 'earnings'
    company: str = Field(default="", max_length=255)
    financial_year: str = Field(default="", max_length=20)  # e.g. 'FY2025'
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=500)
