import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from app.services.api_key_service import ApiKeyService


@pytest.mark.asyncio
async def test_create_key_returns_valid_format(_patch_config: None) -> None:
    service = ApiKeyService()
    key_id = uuid.uuid4()
    now = datetime.now(UTC)
    fake_row = (key_id, "test", "default", "user", now)

    with patch("app.services.api_key_service.db_conn") as mock_db:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = fake_row
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        key, response = service.create_key("test", "user", "default")

    assert key.startswith("sk-")
    assert len(key) > 10
    assert response.name == "test"
    assert response.role == "user"
    assert response.tenant_id == "default"
    assert response.id == str(key_id)


@pytest.mark.asyncio
async def test_validate_key_succeeds_with_correct_key(_patch_config: None) -> None:
    service = ApiKeyService()
    key_id = uuid.uuid4()
    fake_row = (key_id, "b_hash", "l_hash", "test-validate", "default", "user", None, None)

    with patch("app.services.api_key_service.db_conn") as mock_db:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = fake_row
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        validated = service.validate_key("sk-correctkey123")

    assert validated is not None
    assert validated.id == str(key_id)
    assert validated.role == "user"


@pytest.mark.asyncio
async def test_validate_key_fails_with_wrong_key(_patch_config: None) -> None:
    service = ApiKeyService()
    with patch("app.services.api_key_service.db_conn") as mock_db:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        validated = service.validate_key("sk-wrongkey12345wrongwrongwrong")

    assert validated is None


@pytest.mark.asyncio
async def test_revoke_key_invalidates_it(_patch_config: None) -> None:
    service = ApiKeyService()
    key_id = str(uuid.uuid4())

    with patch("app.services.api_key_service.db_conn") as mock_db:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (uuid.UUID(key_id),)
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        revoked = service.revoke_key(key_id, "default")

    assert revoked is True


@pytest.mark.asyncio
async def test_revoke_nonexistent_key_returns_false(_patch_config: None) -> None:
    service = ApiKeyService()
    with patch("app.services.api_key_service.db_conn") as mock_db:
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_db.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = (
            mock_cursor
        )

        result = service.revoke_key("00000000-0000-0000-0000-000000000000", "default")

    assert result is False
