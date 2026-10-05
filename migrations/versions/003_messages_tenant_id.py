"""Add tenant_id to agent_messages for strict tenant isolation

Revision ID: 003
Revises: 002
Create Date: 2026-10-05

Adds tenant_id to agent_messages to ensure conversation history cannot leak
across tenants sharing identical or colliding session IDs.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add tenant_id column with default 'default' to backfill existing messages safely
    op.execute(
        "ALTER TABLE agent_messages ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'"
    )

    # 2. Add tenant-scoped indexes for session message retrieval and ordering
    op.execute(
        "CREATE INDEX messages_tenant_session_idx ON agent_messages (tenant_id, session_id)"
    )
    op.execute(
        "CREATE INDEX messages_tenant_created_idx ON agent_messages (tenant_id, session_id, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS messages_tenant_created_idx")
    op.execute("DROP INDEX IF EXISTS messages_tenant_session_idx")
    op.execute("ALTER TABLE agent_messages DROP COLUMN IF EXISTS tenant_id")
