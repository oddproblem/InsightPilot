"""Add documents table for pgvector RAG storage

Revision ID: 002
Revises: 001
Create Date: 2026-10-05

Stores document chunks with 1536-dim embeddings (text-embedding-3-small).
Tenant-isolated via tenant_id column.  Cosine-distance index via ivfflat.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # documents: one row per text chunk; embedding is a 1536-dim vector.
    # page_number and document_id allow grouping chunks back to their source file.
    # financial_year, doc_type, company are pre-parsed metadata for tool filtering.
    op.execute("""
        CREATE TABLE documents (
            id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            tenant_id    TEXT NOT NULL DEFAULT 'default',
            document_id  UUID NOT NULL,
            document_name TEXT NOT NULL,
            source       TEXT NOT NULL,
            content      TEXT NOT NULL,
            page_number  INTEGER,
            chunk_index  INTEGER NOT NULL DEFAULT 0,
            doc_type     TEXT,
            company      TEXT,
            financial_year TEXT,
            upload_date  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            metadata     JSONB NOT NULL DEFAULT '{}',
            embedding    vector(1536) NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # Primary retrieval path: cosine similarity within tenant
    op.execute(
        "CREATE INDEX documents_embedding_idx "
        "ON documents USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 100)"
    )
    op.execute("CREATE INDEX documents_tenant_idx ON documents (tenant_id)")
    op.execute("CREATE INDEX documents_document_id_idx ON documents (document_id)")
    op.execute(
        "CREATE INDEX documents_tenant_docid_idx ON documents (tenant_id, document_id)"
    )
    op.execute(
        "CREATE INDEX documents_tenant_doctype_idx ON documents (tenant_id, doc_type) "
        "WHERE doc_type IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS documents")
