"""Add tenant_id to agent_messages for strict tenant isolation

Revision ID: 003
Revises: 002
Create Date: 2026-10-05

Adds tenant_id to agent_messages to ensure conversation history cannot leak
across tenants sharing identical or colliding session IDs.

Backfill Strategy:
1. Add tenant_id as nullable TEXT without an arbitrary default.
2. Validate that no session_id in agent_messages maps to multiple distinct tenants.
3. Backfill tenant_id from agent_sessions where session_id matches.
4. Verify that 0 NULL rows remain. Fail migration if unmapped messages exist.
5. Alter column to NOT NULL.
6. Add composite indexes on (tenant_id, session_id) and (tenant_id, session_id, created_at).
"""

from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add tenant_id column as nullable initially (no arbitrary default)
    op.execute("ALTER TABLE agent_messages ADD COLUMN tenant_id TEXT")

    # 2. Check for ambiguity and backfill from agent_sessions using PL/pgSQL block
    op.execute("""
        DO $$
        DECLARE
            ambiguous_count INTEGER;
            orphan_count INTEGER;
        BEGIN
            -- Check for ambiguous session_ids mapping to multiple tenants
            SELECT COUNT(*) INTO ambiguous_count
            FROM (
                SELECT session_id
                FROM agent_sessions
                WHERE session_id IN (
                    SELECT DISTINCT session_id FROM agent_messages WHERE tenant_id IS NULL
                )
                GROUP BY session_id
                HAVING COUNT(DISTINCT tenant_id) > 1
            ) ambiguous;

            IF ambiguous_count > 0 THEN
                RAISE EXCEPTION
                    'Migration 003 failed: % session_id(s) map to multiple tenants.',
                    ambiguous_count;
            END IF;

            -- Backfill tenant_id from the owning agent_sessions record
            UPDATE agent_messages m
            SET tenant_id = s.tenant_id
            FROM agent_sessions s
            WHERE m.session_id = s.session_id
              AND m.tenant_id IS NULL;

            -- Fail if any rows remain unmapped (orphaned messages)
            SELECT COUNT(*) INTO orphan_count
            FROM agent_messages
            WHERE tenant_id IS NULL;

            IF orphan_count > 0 THEN
                RAISE EXCEPTION
                    'Migration 003 failed: % message(s) have no corresponding session.',
                    orphan_count;
            END IF;
        END $$;
    """)

    # 3. Enforce NOT NULL constraint now that all existing rows are safely verified
    op.execute("ALTER TABLE agent_messages ALTER COLUMN tenant_id SET NOT NULL")

    # 4. Add tenant-scoped indexes for session message retrieval and ordering
    op.execute(
        "CREATE INDEX messages_tenant_session_idx ON agent_messages (tenant_id, session_id)"
    )
    op.execute(
        "CREATE INDEX messages_tenant_created_idx ON agent_messages "
        "(tenant_id, session_id, created_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS messages_tenant_created_idx")
    op.execute("DROP INDEX IF EXISTS messages_tenant_session_idx")
    op.execute("ALTER TABLE agent_messages DROP COLUMN IF EXISTS tenant_id")
