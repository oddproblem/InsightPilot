"""Tests for strict multi-tenant isolation across sessions, messages, and API keys.

Verifies:
1. Same tenant + same session_id -> messages visible.
2. Different tenant + same session_id -> messages NOT visible (cross-tenant leakage prevented).
3. Tenant A cannot create, overwrite, or access Tenant B's conversation.
4. Missing or empty tenant_id fails safely.
5. All queries enforce composite keys (session_id, tenant_id).
6. API key revocation enforces tenant scoping (IDOR prevention).
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.db.queries import (
    CREATE_SESSION,
    GET_SESSION,
    GET_SESSION_MESSAGES,
    INCREMENT_SESSION_MESSAGE_COUNT,
    INSERT_MESSAGE,
    REVOKE_API_KEY,
)
from app.services.agent_service import AgentService
from app.services.api_key_service import ApiKeyService


class MockTenantDatabase:
    """In-memory mock representing PostgreSQL tables with tenant isolation."""

    def __init__(self) -> None:
        # (session_id, tenant_id) -> session dict
        self.sessions: dict[tuple[str, str], dict[str, Any]] = {}
        # list of message dicts with tenant_id, session_id, role, content, etc.
        self.messages: list[dict[str, Any]] = []
        # (key_id, tenant_id) -> key dict
        self.keys: dict[tuple[str, str], dict[str, Any]] = {}
        self.executed_queries: list[tuple[str, dict[str, Any]]] = []

    def cursor(self) -> "MockTenantCursor":
        return MockTenantCursor(self)


class MockTenantCursor:
    def __init__(self, db: MockTenantDatabase) -> None:
        self.db = db
        self.last_result: list[Any] = []
        self.fetchone_result: Any = None

    def __enter__(self) -> "MockTenantCursor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        params = params or {}
        self.db.executed_queries.append((query, params))

        if query == CREATE_SESSION:
            key = (params["session_id"], params["tenant_id"])
            if key not in self.db.sessions:
                self.db.sessions[key] = {
                    "id": params["id"],
                    "session_id": params["session_id"],
                    "tenant_id": params["tenant_id"],
                    "created_at": datetime.now(tz=UTC),
                    "last_active_at": datetime.now(tz=UTC),
                    "message_count": 0,
                }
            else:
                self.db.sessions[key]["last_active_at"] = datetime.now(tz=UTC)

        elif query == GET_SESSION:
            key = (params["session_id"], params["tenant_id"])
            sess = self.db.sessions.get(key)
            if sess:
                self.fetchone_result = (
                    sess["id"],
                    sess["session_id"],
                    sess["tenant_id"],
                    sess["created_at"],
                    sess["last_active_at"],
                    sess["message_count"],
                )
            else:
                self.fetchone_result = None

        elif query == GET_SESSION_MESSAGES:
            # Strictly filter by BOTH session_id AND tenant_id
            target_sid = params["session_id"]
            target_tid = params["tenant_id"]
            matched = [
                (
                    m["id"],
                    m["session_id"],
                    m["tenant_id"],
                    m["role"],
                    m["content"],
                    m["metadata"],
                    m["created_at"],
                )
                for m in self.db.messages
                if m["session_id"] == target_sid and m["tenant_id"] == target_tid
            ]
            self.last_result = matched

        elif query == INSERT_MESSAGE:
            self.db.messages.append(
                {
                    "id": params["id"],
                    "session_id": params["session_id"],
                    "tenant_id": params["tenant_id"],
                    "role": params["role"],
                    "content": params["content"],
                    "metadata": params.get("metadata", "{}"),
                    "created_at": datetime.now(tz=UTC),
                }
            )

        elif query == INCREMENT_SESSION_MESSAGE_COUNT:
            key = (params["session_id"], params["tenant_id"])
            if key in self.db.sessions:
                self.db.sessions[key]["message_count"] += 1

        elif query == REVOKE_API_KEY:
            key = (params["id"], params["tenant_id"])
            if key in self.db.keys and self.db.keys[key]["revoked_at"] is None:
                self.db.keys[key]["revoked_at"] = datetime.now(tz=UTC)
                self.fetchone_result = (params["id"],)
            else:
                self.fetchone_result = None

    def fetchall(self) -> list[Any]:
        return self.last_result

    def fetchone(self) -> Any:
        return self.fetchone_result


@pytest.mark.asyncio
async def test_same_tenant_same_session_messages_visible(_patch_config: None) -> None:
    """A tenant querying their own session retrieves their conversation messages."""
    db = MockTenantDatabase()
    sid = "financial-review-2026"
    tid = "acme-corp"

    # Pre-populate session and messages for acme-corp
    db.sessions[(sid, tid)] = {
        "id": "sess-1",
        "session_id": sid,
        "tenant_id": tid,
        "created_at": datetime.now(tz=UTC),
        "last_active_at": datetime.now(tz=UTC),
        "message_count": 2,
    }
    db.messages.append(
        {
            "id": "m1",
            "session_id": sid,
            "tenant_id": tid,
            "role": "user",
            "content": "What was our Q3 EBITDA?",
            "metadata": "{}",
            "created_at": datetime.now(tz=UTC),
        }
    )
    db.messages.append(
        {
            "id": "m2",
            "session_id": sid,
            "tenant_id": tid,
            "role": "assistant",
            "content": "Acme Q3 EBITDA was $42M.",
            "metadata": "{}",
            "created_at": datetime.now(tz=UTC),
        }
    )

    mock_graph = MagicMock()
    service = AgentService(mock_graph)

    with patch("app.services.agent_service.db_conn") as mock_conn:
        mock_conn.return_value.__enter__.return_value = db
        session = await service.get_session(session_id=sid, tenant_id=tid)

    assert session is not None
    assert session.session_id == sid
    assert session.message_count == 2
    assert len(session.messages) == 2
    assert session.messages[0].content == "What was our Q3 EBITDA?"
    assert session.messages[1].content == "Acme Q3 EBITDA was $42M."


@pytest.mark.asyncio
async def test_different_tenant_same_session_messages_not_visible(_patch_config: None) -> None:
    """A tenant requesting an identical session_id belonging to another tenant
    CANNOT view messages.
    """
    db = MockTenantDatabase()
    sid = "shared-session-name"
    tenant_a = "tenant-a-finance"
    tenant_b = "tenant-b-competitor"

    # Populate session and confidential messages for Tenant A
    db.sessions[(sid, tenant_a)] = {
        "id": "sess-a",
        "session_id": sid,
        "tenant_id": tenant_a,
        "created_at": datetime.now(tz=UTC),
        "last_active_at": datetime.now(tz=UTC),
        "message_count": 2,
    }
    db.messages.append(
        {
            "id": "m-secret-1",
            "session_id": sid,
            "tenant_id": tenant_a,
            "role": "user",
            "content": "Confidential merger details between A and X",
            "metadata": "{}",
            "created_at": datetime.now(tz=UTC),
        }
    )
    db.messages.append(
        {
            "id": "m-secret-2",
            "session_id": sid,
            "tenant_id": tenant_a,
            "role": "assistant",
            "content": "The acquisition offer is priced at $55 per share.",
            "metadata": "{}",
            "created_at": datetime.now(tz=UTC),
        }
    )

    mock_graph = MagicMock()
    service = AgentService(mock_graph)

    with patch("app.services.agent_service.db_conn") as mock_conn:
        mock_conn.return_value.__enter__.return_value = db
        # Tenant B queries the exact same session ID
        session_b = await service.get_session(session_id=sid, tenant_id=tenant_b)

    # Tenant B has no session record with that (session_id, tenant_id) pair -> returns None
    assert session_b is None

    # Even if Tenant B has a session created with the same session_id,
    # message retrieval filters by tenant_id and returns 0 messages from Tenant A
    db.sessions[(sid, tenant_b)] = {
        "id": "sess-b",
        "session_id": sid,
        "tenant_id": tenant_b,
        "created_at": datetime.now(tz=UTC),
        "last_active_at": datetime.now(tz=UTC),
        "message_count": 0,
    }

    with patch("app.services.agent_service.db_conn") as mock_conn:
        mock_conn.return_value.__enter__.return_value = db
        session_b = await service.get_session(session_id=sid, tenant_id=tenant_b)

    assert session_b is not None
    assert session_b.session_id == sid
    assert len(session_b.messages) == 0  # Tenant A's messages are strictly invisible!


@pytest.mark.asyncio
async def test_tenant_a_cannot_access_or_mutate_tenant_b_conversation(_patch_config: None) -> None:
    """Two tenants using the same session_id execute turns independently without context bleed."""
    db = MockTenantDatabase()
    sid = "quarterly-review"
    tenant_a = "pharma-client"
    tenant_b = "retail-client"

    mock_graph = MagicMock()

    def mock_invoke(state: dict[str, Any]) -> dict[str, Any]:
        caller_tenant = state["tenant_id"]
        # Verify that history contains only messages from the caller tenant
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                content_str = str(msg.content).lower()
                if caller_tenant == tenant_a:
                    assert "retail" not in content_str
                elif caller_tenant == tenant_b:
                    assert "pharma" not in content_str

        res_msg = AIMessage(content=f"Response for {caller_tenant}")
        res_msg.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 10,
            "total_tokens": 20,
        }
        return {
            **state,
            "messages": [*state["messages"], res_msg],
            "input_tokens": 10,
            "output_tokens": 10,
        }

    mock_graph.invoke.side_effect = mock_invoke
    service = AgentService(mock_graph)

    with patch("app.services.agent_service.db_conn") as mock_conn:
        mock_conn.return_value.__enter__.return_value = db

        # 1. Tenant A runs a turn
        res_a = await service.run(
            session_id=sid,
            tenant_id=tenant_a,
            message="Pharma clinical trial phase 3 results",
        )
        assert res_a.response == f"Response for {tenant_a}"

        # 2. Tenant B runs a turn on the SAME session_id
        res_b = await service.run(
            session_id=sid,
            tenant_id=tenant_b,
            message="Retail same-store sales growth",
        )
        assert res_b.response == f"Response for {tenant_b}"

        # 3. Tenant A inspects session
        sess_a = await service.get_session(session_id=sid, tenant_id=tenant_a)
        assert sess_a is not None
        assert sess_a.message_count == 2
        assert len(sess_a.messages) == 2
        assert "Pharma" in sess_a.messages[0].content
        assert "pharma-client" in sess_a.messages[1].content

        # 4. Tenant B inspects session
        sess_b = await service.get_session(session_id=sid, tenant_id=tenant_b)
        assert sess_b is not None
        assert sess_b.message_count == 2
        assert len(sess_b.messages) == 2
        assert "Retail" in sess_b.messages[0].content
        assert "retail-client" in sess_b.messages[1].content

    # Total messages in DB should be 4 (2 for tenant A, 2 for tenant B)
    assert len(db.messages) == 4
    for m in db.messages:
        assert m["tenant_id"] in (tenant_a, tenant_b)


@pytest.mark.asyncio
async def test_missing_or_empty_tenant_id_fails_safely(_patch_config: None) -> None:
    """AgentService rejects missing or whitespace tenant_id safely."""
    mock_graph = MagicMock()
    service = AgentService(mock_graph)

    # Empty tenant_id on run
    with pytest.raises(ValueError, match="tenant_id is required"):
        await service.run(session_id="s1", tenant_id="", message="hello")

    with pytest.raises(ValueError, match="tenant_id is required"):
        await service.run(session_id="s1", tenant_id="   ", message="hello")

    # Empty session_id on run
    with pytest.raises(ValueError, match="session_id is required"):
        await service.run(session_id="", tenant_id="t1", message="hello")

    # Empty tenant_id on get_session
    with pytest.raises(ValueError, match="tenant_id is required"):
        await service.get_session(session_id="s1", tenant_id="")

    with pytest.raises(ValueError, match="tenant_id is required"):
        await service.get_session(session_id="s1", tenant_id="   ")

    # Empty session_id on get_session
    with pytest.raises(ValueError, match="session_id is required"):
        await service.get_session(session_id="", tenant_id="t1")


def test_sql_queries_strictly_enforce_tenant_scoping() -> None:
    """Verify that all conversation and session SQL queries contain tenant_id parameters."""
    # 1. GET_SESSION_MESSAGES must filter by both session_id and tenant_id
    assert "session_id = %(session_id)s" in GET_SESSION_MESSAGES
    assert "tenant_id = %(tenant_id)s" in GET_SESSION_MESSAGES
    assert "tenant_id" in GET_SESSION_MESSAGES

    # 2. INSERT_MESSAGE must store tenant_id
    assert "tenant_id" in INSERT_MESSAGE
    assert "%(tenant_id)s" in INSERT_MESSAGE

    # 3. INCREMENT_SESSION_MESSAGE_COUNT must be scoped by tenant_id
    assert "session_id = %(session_id)s" in INCREMENT_SESSION_MESSAGE_COUNT
    assert "tenant_id = %(tenant_id)s" in INCREMENT_SESSION_MESSAGE_COUNT

    # 4. CREATE_SESSION must include tenant_id
    assert "tenant_id" in CREATE_SESSION
    assert "%(tenant_id)s" in CREATE_SESSION

    # 5. GET_SESSION must include tenant_id
    assert "session_id = %(session_id)s" in GET_SESSION
    assert "tenant_id = %(tenant_id)s" in GET_SESSION

    # 6. REVOKE_API_KEY must filter by tenant_id (prevent cross-tenant key revocation)
    assert "id = %(id)s" in REVOKE_API_KEY
    assert "tenant_id = %(tenant_id)s" in REVOKE_API_KEY


def test_api_key_revocation_enforces_tenant_isolation(_patch_config: None) -> None:
    """Tenant A cannot revoke Tenant B's API key."""
    db = MockTenantDatabase()
    key_a = "key-uuid-111"
    key_b = "key-uuid-222"

    db.keys[(key_a, "tenant-a")] = {
        "id": key_a,
        "tenant_id": "tenant-a",
        "revoked_at": None,
    }
    db.keys[(key_b, "tenant-b")] = {
        "id": key_b,
        "tenant_id": "tenant-b",
        "revoked_at": None,
    }

    service = ApiKeyService()

    with patch("app.services.api_key_service.db_conn") as mock_conn:
        mock_conn.return_value.__enter__.return_value = db

        # Tenant A attempts to revoke Tenant B's key -> must fail (return False)
        revoked_by_attacker = service.revoke_key(key_id=key_b, tenant_id="tenant-a")
        assert revoked_by_attacker is False
        assert db.keys[(key_b, "tenant-b")]["revoked_at"] is None

        # Tenant B revokes their own key -> succeeds
        revoked_by_owner = service.revoke_key(key_id=key_b, tenant_id="tenant-b")
        assert revoked_by_owner is True
        assert db.keys[(key_b, "tenant-b")]["revoked_at"] is not None
