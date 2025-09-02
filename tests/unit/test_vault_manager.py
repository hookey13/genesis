"""Unit tests for Vault Manager."""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest

from genesis.security.vault_manager import (
    VaultManager, CircuitBreakerState
)
from genesis.config.vault_config import VaultConfig
from genesis.core.exceptions import SecurityException, ValidationException


@pytest.fixture
def vault_config():
    """Create test Vault configuration."""
    return VaultConfig(
        vault_url="http://localhost:8200",
        vault_token="test-token",
        cache_ttl_seconds=300,
        max_retries=3,
        retry_delay_seconds=0.1,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=5
    )


@pytest.fixture
def mock_hvac_client():
    """Create mock hvac client."""
    client = MagicMock()
    client.is_authenticated.return_value = True
    client.token = "test-token"
    return client


@pytest.fixture
async def vault_manager(vault_config, mock_hvac_client):
    """Create VaultManager instance with mocked client."""
    manager = VaultManager(vault_config)
    
    with patch('genesis.security.vault_manager.hvac.Client', return_value=mock_hvac_client):
        await manager.initialize()
        manager._client = mock_hvac_client
    
    return manager


@pytest.mark.asyncio
async def test_vault_initialization(vault_config, mock_hvac_client):
    """Test Vault manager initialization."""
    manager = VaultManager(vault_config)
    
    with patch('genesis.security.vault_manager.hvac.Client', return_value=mock_hvac_client):
        await manager.initialize()
        
        assert manager._client is not None
        assert manager._circuit_breaker_state == CircuitBreakerState.CLOSED
        mock_hvac_client.is_authenticated.assert_called_once()


@pytest.mark.asyncio
async def test_get_secret_success(vault_manager):
    """Test successful secret retrieval."""
    test_secret = {"username": "testuser", "password": "testpass"}
    
    # Mock Vault response
    vault_manager._client.secrets.kv.v2.read_secret_version = AsyncMock(
        return_value={"data": {"data": test_secret}}
    )
    
    # Get secret
    result = await vault_manager.get_secret("test/path")
    
    assert result == test_secret
    assert "test/path" in vault_manager._cache


@pytest.mark.asyncio
async def test_get_secret_with_key(vault_manager):
    """Test retrieving specific key from secret."""
    test_secret = {"username": "testuser", "password": "testpass"}
    
    vault_manager._client.secrets.kv.v2.read_secret_version = AsyncMock(
        return_value={"data": {"data": test_secret}}
    )
    
    result = await vault_manager.get_secret("test/path", "username")
    
    assert result == "testuser"


@pytest.mark.asyncio
async def test_cache_functionality(vault_manager):
    """Test secret caching."""
    test_secret = {"key": "value"}
    
    vault_manager._client.secrets.kv.v2.read_secret_version = AsyncMock(
        return_value={"data": {"data": test_secret}}
    )
    
    # First call - should hit Vault
    result1 = await vault_manager.get_secret("cached/path")
    assert vault_manager._client.secrets.kv.v2.read_secret_version.call_count == 1
    
    # Second call - should use cache
    result2 = await vault_manager.get_secret("cached/path")
    assert vault_manager._client.secrets.kv.v2.read_secret_version.call_count == 1
    
    assert result1 == result2 == test_secret


@pytest.mark.asyncio
async def test_cache_expiration(vault_manager):
    """Test cache TTL expiration."""
    test_secret = {"key": "value"}
    
    vault_manager._client.secrets.kv.v2.read_secret_version = AsyncMock(
        return_value={"data": {"data": test_secret}}
    )
    
    # Get secret and cache it
    await vault_manager.get_secret("expiring/path")
    
    # Manually expire cache
    vault_manager._cache["expiring/path"]["expires_at"] = datetime.utcnow() - timedelta(seconds=1)
    
    # Should hit Vault again
    await vault_manager.get_secret("expiring/path")
    assert vault_manager._client.secrets.kv.v2.read_secret_version.call_count == 2


@pytest.mark.asyncio
async def test_circuit_breaker_opens(vault_manager):
    """Test circuit breaker opens after threshold failures."""
    vault_manager._client.secrets.kv.v2.read_secret_version = AsyncMock(
        side_effect=Exception("Vault error")
    )
    
    # Fail multiple times to open circuit
    for _ in range(vault_manager.config.circuit_breaker_threshold):
        with pytest.raises(SecurityException):
            await vault_manager.get_secret("failing/path")
    
    assert vault_manager._circuit_breaker_state == CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_recovery(vault_manager):
    """Test circuit breaker recovery after timeout."""
    # Open circuit breaker
    vault_manager._circuit_breaker_state = CircuitBreakerState.OPEN
    vault_manager._circuit_breaker_last_failure = datetime.utcnow() - timedelta(
        seconds=vault_manager.config.circuit_breaker_timeout + 1
    )
    
    # Should transition to HALF_OPEN and allow request
    assert vault_manager._is_circuit_closed() == True
    assert vault_manager._circuit_breaker_state == CircuitBreakerState.HALF_OPEN


@pytest.mark.asyncio
async def test_retry_logic(vault_manager):
    """Test exponential backoff retry."""
    call_count = 0
    
    async def mock_read(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary error")
        return {"data": {"data": {"key": "value"}}}
    
    vault_manager._client.secrets.kv.v2.read_secret_version = mock_read
    
    result = await vault_manager.get_secret("retry/path")
    
    assert result == {"key": "value"}
    assert call_count == 3


@pytest.mark.asyncio
async def test_write_secret(vault_manager):
    """Test writing secret to Vault."""
    test_data = {"api_key": "secret123"}
    
    vault_manager._client.secrets.kv.v2.create_or_update_secret = AsyncMock()
    
    await vault_manager.write_secret("new/secret", test_data)
    
    vault_manager._client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
        path="new/secret",
        secret=test_data,
        mount_point=vault_manager.config.kv_mount_point
    )


@pytest.mark.asyncio
async def test_delete_secret(vault_manager):
    """Test deleting secret from Vault."""
    vault_manager._client.secrets.kv.v2.delete_metadata_and_all_versions = AsyncMock()
    
    # Add to cache first
    vault_manager._cache["delete/me"] = {
        "data": {"key": "value"},
        "expires_at": datetime.utcnow() + timedelta(minutes=5)
    }
    
    await vault_manager.delete_secret("delete/me")
    
    # Should be removed from cache
    assert "delete/me" not in vault_manager._cache
    vault_manager._client.secrets.kv.v2.delete_metadata_and_all_versions.assert_called_once()


@pytest.mark.asyncio
async def test_list_secrets(vault_manager):
    """Test listing secrets."""
    mock_response = {"data": {"keys": ["secret1", "secret2", "secret3"]}}
    
    vault_manager._client.secrets.kv.v2.list_secrets = AsyncMock(
        return_value=mock_response
    )
    
    result = await vault_manager.list_secrets("test/path")
    
    assert result == ["secret1", "secret2", "secret3"]


@pytest.mark.asyncio
async def test_break_glass_cache_save(vault_manager, tmp_path):
    """Test saving break-glass cache."""
    vault_manager.config.break_glass_cache_file = str(tmp_path / "cache.json")
    
    # Add secrets to cache
    vault_manager._cache["secret1"] = {
        "data": {"key1": "value1"},
        "expires_at": datetime.utcnow() + timedelta(minutes=5)
    }
    vault_manager._cache["secret2"] = {
        "data": {"key2": "value2"},
        "expires_at": datetime.utcnow() + timedelta(minutes=5)
    }
    
    await vault_manager.save_break_glass_cache()
    
    # Verify cache was saved
    with open(vault_manager.config.break_glass_cache_file, "r") as f:
        saved_cache = json.load(f)
    
    assert "secret1" in saved_cache
    assert "secret2" in saved_cache
    assert saved_cache["secret1"]["key1"] == "value1"


@pytest.mark.asyncio
async def test_break_glass_cache_load(vault_manager, tmp_path):
    """Test loading break-glass cache."""
    cache_file = tmp_path / "cache.json"
    vault_manager.config.break_glass_cache_file = str(cache_file)
    
    # Create cache file
    cache_data = {
        "emergency/secret": {"username": "admin", "password": "emergency123"}
    }
    with open(cache_file, "w") as f:
        json.dump(cache_data, f)
    
    # Load from break-glass cache
    result = await vault_manager._load_break_glass_cache("emergency/secret")
    
    assert result == {"username": "admin", "password": "emergency123"}


@pytest.mark.asyncio
async def test_token_renewal(vault_manager):
    """Test token renewal logic."""
    vault_manager._token_lease_duration = 3600  # 1 hour
    vault_manager._token_renewed_at = datetime.utcnow() - timedelta(minutes=55)
    vault_manager.config.token_renewal_threshold = 600  # 10 minutes
    
    # Should need renewal
    assert vault_manager._should_renew_token() == True
    
    # Mock renewal
    vault_manager._client.auth.token.renew_self = AsyncMock(
        return_value={"auth": {"lease_duration": 3600}}
    )
    
    await vault_manager._renew_token()
    
    vault_manager._client.auth.token.renew_self.assert_called_once()
    assert vault_manager._token_lease_duration == 3600


@pytest.mark.asyncio
async def test_appRole_authentication(vault_config):
    """Test AppRole authentication."""
    vault_config.vault_auth_method = "approle"
    vault_config.vault_role_id = "test-role-id"
    vault_config.vault_secret_id = "test-secret-id"
    
    mock_client = MagicMock()
    mock_client.is_authenticated.return_value = True
    mock_client.auth.approle.login = AsyncMock(
        return_value={
            "auth": {
                "client_token": "new-token",
                "lease_duration": 3600
            }
        }
    )
    
    manager = VaultManager(vault_config)
    
    with patch('genesis.security.vault_manager.hvac.Client', return_value=mock_client):
        await manager.initialize()
    
    mock_client.auth.approle.login.assert_called_once_with(
        role_id="test-role-id",
        secret_id="test-secret-id"
    )
    assert mock_client.token == "new-token"