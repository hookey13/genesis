"""Unit tests for Credential Manager."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from genesis.security.credential_manager import CredentialManager
from genesis.security.vault_manager import VaultManager
from genesis.core.exceptions import SecurityException


@pytest.fixture
def mock_vault_manager():
    """Create mock Vault manager."""
    vault = MagicMock(spec=VaultManager)
    vault._get_with_retry = AsyncMock()
    vault.get_secret = AsyncMock()
    vault.write_secret = AsyncMock()
    return vault


@pytest.fixture
def credential_manager(mock_vault_manager):
    """Create CredentialManager instance."""
    return CredentialManager(mock_vault_manager)


@pytest.mark.asyncio
async def test_get_database_credentials_new(credential_manager, mock_vault_manager):
    """Test getting new database credentials."""
    mock_vault_manager._get_with_retry.return_value = {
        "username": "genesis_user_abc123",
        "password": "temp_pass_xyz789",
        "lease_duration": 3600
    }
    
    username, password = await credential_manager.get_database_credentials("test-role")
    
    assert username == "genesis_user_abc123"
    assert password == "temp_pass_xyz789"
    assert "test-role" in credential_manager._active_credentials
    
    mock_vault_manager._get_with_retry.assert_called_once_with("database/creds/test-role")


@pytest.mark.asyncio
async def test_get_database_credentials_cached(credential_manager, mock_vault_manager):
    """Test getting cached database credentials."""
    # Pre-populate cache
    credential_manager._active_credentials["test-role"] = {
        "username": "cached_user",
        "password": "cached_pass",
        "expires_at": datetime.utcnow() + timedelta(hours=1),
        "lease_duration": 3600
    }
    
    username, password = await credential_manager.get_database_credentials("test-role")
    
    assert username == "cached_user"
    assert password == "cached_pass"
    # Should not call Vault
    mock_vault_manager._get_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_get_database_credentials_expired_cache(credential_manager, mock_vault_manager):
    """Test expired cached credentials trigger renewal."""
    # Pre-populate with expired cache
    credential_manager._active_credentials["test-role"] = {
        "username": "old_user",
        "password": "old_pass",
        "expires_at": datetime.utcnow() - timedelta(minutes=1),
        "lease_duration": 3600
    }
    
    mock_vault_manager._get_with_retry.return_value = {
        "username": "new_user",
        "password": "new_pass",
        "lease_duration": 3600
    }
    
    username, password = await credential_manager.get_database_credentials("test-role")
    
    assert username == "new_user"
    assert password == "new_pass"
    mock_vault_manager._get_with_retry.assert_called_once()


@pytest.mark.asyncio
async def test_database_credentials_fallback(credential_manager, mock_vault_manager):
    """Test fallback to static credentials on failure."""
    mock_vault_manager._get_with_retry.side_effect = Exception("Vault error")
    mock_vault_manager.get_secret.return_value = {
        "username": "static_user",
        "password": "static_pass"
    }
    
    username, password = await credential_manager.get_database_credentials("test-role")
    
    assert username == "static_user"
    assert password == "static_pass"
    mock_vault_manager.get_secret.assert_called_once_with("database/static-creds")


@pytest.mark.asyncio
async def test_rotate_database_credentials(credential_manager, mock_vault_manager):
    """Test database credential rotation."""
    mock_vault_manager._get_with_retry.return_value = {
        "username": "rotated_user",
        "password": "rotated_pass",
        "lease_duration": 3600
    }
    
    await credential_manager.rotate_database_credentials("test-role")
    
    assert "test-role" in credential_manager._active_credentials
    creds = credential_manager._active_credentials["test-role"]
    assert creds["username"] == "rotated_user"
    assert creds["password"] == "rotated_pass"


@pytest.mark.asyncio
async def test_get_api_key(credential_manager, mock_vault_manager):
    """Test retrieving API key."""
    mock_vault_manager.get_secret.return_value = {"api_key": "test_api_key_123"}
    
    api_key = await credential_manager.get_api_key("binance")
    
    assert api_key == "test_api_key_123"
    mock_vault_manager.get_secret.assert_called_once_with("api-keys/binance")


@pytest.mark.asyncio
async def test_get_api_credentials(credential_manager, mock_vault_manager):
    """Test retrieving API key and secret."""
    mock_vault_manager.get_secret.return_value = {
        "api_key": "test_key_123",
        "api_secret": "test_secret_456"
    }
    
    api_key, api_secret = await credential_manager.get_api_credentials("binance")
    
    assert api_key == "test_key_123"
    assert api_secret == "test_secret_456"
    mock_vault_manager.get_secret.assert_called_once_with("api-keys/binance")


@pytest.mark.asyncio
async def test_get_jwt_signing_key(credential_manager, mock_vault_manager):
    """Test retrieving JWT signing key."""
    mock_vault_manager.get_secret.return_value = {"key": "jwt_secret_key_789"}
    
    jwt_key = await credential_manager.get_jwt_signing_key()
    
    assert jwt_key == "jwt_secret_key_789"
    mock_vault_manager.get_secret.assert_called_once_with("jwt/signing-key")


@pytest.mark.asyncio
async def test_rotate_jwt_signing_key(credential_manager, mock_vault_manager):
    """Test JWT signing key rotation."""
    mock_vault_manager.get_secret.return_value = {"key": "old_jwt_key"}
    
    with patch('secrets.token_urlsafe', return_value="new_jwt_key_abc"):
        new_key = await credential_manager.rotate_jwt_signing_key()
    
    assert new_key == "new_jwt_key_abc"
    
    # Should save old key
    mock_vault_manager.write_secret.assert_any_call(
        "jwt/signing-key-old",
        {"key": "old_jwt_key", "rotated_at": unittest.mock.ANY}
    )
    
    # Should save new key
    mock_vault_manager.write_secret.assert_any_call(
        "jwt/signing-key",
        {"key": "new_jwt_key_abc", "created_at": unittest.mock.ANY}
    )


@pytest.mark.asyncio
async def test_database_connection_context_manager(credential_manager, mock_vault_manager):
    """Test database connection context manager."""
    mock_vault_manager._get_with_retry.return_value = {
        "username": "db_user",
        "password": "db_pass",
        "lease_duration": 3600
    }
    
    with patch('genesis.security.credential_manager.create_async_engine') as mock_engine:
        mock_engine_instance = AsyncMock()
        mock_engine.return_value = mock_engine_instance
        
        async with credential_manager.get_database_connection("test-role") as engine:
            assert engine == mock_engine_instance
        
        # Engine should be disposed
        mock_engine_instance.dispose.assert_called_once()


@pytest.mark.asyncio
async def test_credential_rotation_scheduling(credential_manager, mock_vault_manager):
    """Test credential rotation scheduling."""
    mock_vault_manager._get_with_retry.return_value = {
        "username": "user",
        "password": "pass",
        "lease_duration": 100  # Short duration for testing
    }
    
    # Get credentials (should schedule rotation)
    await credential_manager.get_database_credentials("test-role")
    
    # Verify rotation task was created
    assert "test-role" in credential_manager._rotation_tasks
    task = credential_manager._rotation_tasks["test-role"]
    assert not task.done()
    
    # Cancel task for cleanup
    task.cancel()


@pytest.mark.asyncio
async def test_cleanup(credential_manager):
    """Test cleanup of credential manager."""
    # Add some test data
    credential_manager._active_credentials["test"] = {"data": "test"}
    
    # Create a dummy task
    async def dummy():
        await asyncio.sleep(10)
    
    task = asyncio.create_task(dummy())
    credential_manager._rotation_tasks["test"] = task
    
    # Cleanup
    await credential_manager.cleanup()
    
    # Verify cleanup
    assert len(credential_manager._active_credentials) == 0
    assert task.cancelled()


# Import unittest.mock for ANY matcher
import unittest.mock