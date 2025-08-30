"""Unit tests for Vault client module."""

import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pytest

from genesis.security.vault_client import VaultClient, SecretCache


class TestSecretCache:
    """Test SecretCache class."""
    
    def test_cache_not_expired(self):
        """Test that cache is not expired within TTL."""
        cache = SecretCache(
            value="test_secret",
            fetched_at=datetime.now(),
            ttl_seconds=3600
        )
        assert not cache.is_expired()
    
    def test_cache_expired(self):
        """Test that cache is expired after TTL."""
        cache = SecretCache(
            value="test_secret",
            fetched_at=datetime.now() - timedelta(hours=2),
            ttl_seconds=3600
        )
        assert cache.is_expired()


class TestVaultClient:
    """Test VaultClient class."""
    
    @pytest.fixture
    def mock_hvac_client(self):
        """Create a mock hvac client."""
        with patch("genesis.security.vault_client.hvac") as mock_hvac:
            mock_client = Mock()
            mock_client.is_authenticated.return_value = True
            mock_hvac.Client.return_value = mock_client
            yield mock_client
    
    @pytest.fixture
    def vault_client_with_vault(self, mock_hvac_client):
        """Create VaultClient with Vault enabled."""
        client = VaultClient(
            vault_url="https://vault.example.com:8200",
            vault_token="test_token",
            use_vault=True,
            cache_ttl=60
        )
        client.client = mock_hvac_client
        return client
    
    @pytest.fixture
    def vault_client_without_vault(self):
        """Create VaultClient with Vault disabled (env fallback)."""
        return VaultClient(use_vault=False)
    
    def test_init_with_vault_success(self, mock_hvac_client):
        """Test successful Vault initialization."""
        client = VaultClient(
            vault_url="https://vault.example.com:8200",
            vault_token="test_token",
            use_vault=True
        )
        assert client.use_vault is True
        assert client.vault_url == "https://vault.example.com:8200"
        assert client.vault_token == "test_token"
    
    def test_init_without_vault(self):
        """Test initialization without Vault (development mode)."""
        client = VaultClient(use_vault=False)
        assert client.use_vault is False
        assert client.client is None
    
    def test_init_vault_auth_failure(self):
        """Test Vault initialization with authentication failure."""
        with patch("genesis.security.vault_client.hvac") as mock_hvac:
            mock_client = Mock()
            mock_client.is_authenticated.return_value = False
            mock_hvac.Client.return_value = mock_client
            
            client = VaultClient(
                vault_url="https://vault.example.com:8200",
                vault_token="invalid_token",
                use_vault=True
            )
            assert client.use_vault is False
            assert client.client is None
    
    def test_get_secret_from_cache(self, vault_client_with_vault):
        """Test retrieving secret from cache."""
        # Populate cache
        vault_client_with_vault._cache["test_path"] = SecretCache(
            value="cached_secret",
            fetched_at=datetime.now(),
            ttl_seconds=3600
        )
        
        # Get from cache
        secret = vault_client_with_vault.get_secret("test_path")
        assert secret == "cached_secret"
        
        # Vault should not be called
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.assert_not_called()
    
    def test_get_secret_cache_expired(self, vault_client_with_vault):
        """Test retrieving secret when cache is expired."""
        # Populate expired cache
        vault_client_with_vault._cache["test_path"] = SecretCache(
            value="old_secret",
            fetched_at=datetime.now() - timedelta(hours=2),
            ttl_seconds=3600
        )
        
        # Mock Vault response
        mock_response = {
            "data": {
                "data": {
                    "secret_key": "new_secret"
                }
            }
        }
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # Get with expired cache
        secret = vault_client_with_vault.get_secret("test_path")
        assert secret == {"secret_key": "new_secret"}
        
        # Verify Vault was called
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.assert_called_once()
    
    def test_get_secret_force_refresh(self, vault_client_with_vault):
        """Test force refresh bypasses cache."""
        # Populate cache
        vault_client_with_vault._cache["test_path"] = SecretCache(
            value="cached_secret",
            fetched_at=datetime.now(),
            ttl_seconds=3600
        )
        
        # Mock Vault response
        mock_response = {
            "data": {
                "data": {
                    "secret_key": "fresh_secret"
                }
            }
        }
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # Force refresh
        secret = vault_client_with_vault.get_secret("test_path", force_refresh=True)
        assert secret == {"secret_key": "fresh_secret"}
        
        # Verify Vault was called despite cache
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.assert_called_once()
    
    def test_get_secret_from_environment(self, vault_client_without_vault, monkeypatch):
        """Test retrieving secret from environment variables."""
        # Set environment variables
        monkeypatch.setenv("BINANCE_API_KEY", "test_api_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_api_secret")
        
        # Get exchange keys
        secret = vault_client_without_vault.get_secret(
            VaultClient.EXCHANGE_API_KEYS_PATH
        )
        assert secret == {
            "api_key": "test_api_key",
            "api_secret": "test_api_secret"
        }
    
    def test_get_exchange_api_keys(self, vault_client_with_vault):
        """Test getting exchange API keys."""
        # Mock Vault response
        mock_response = {
            "data": {
                "data": {
                    "api_key": "main_key",
                    "api_secret": "main_secret",
                    "api_key_read": "read_key",
                    "api_secret_read": "read_secret"
                }
            }
        }
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # Get trading keys
        keys = vault_client_with_vault.get_exchange_api_keys(read_only=False)
        assert keys == {
            "api_key": "main_key",
            "api_secret": "main_secret"
        }
        
        # Get read-only keys
        keys = vault_client_with_vault.get_exchange_api_keys(read_only=True)
        assert keys == {
            "api_key": "read_key",
            "api_secret": "read_secret"
        }
    
    def test_store_secret(self, vault_client_with_vault):
        """Test storing a secret in Vault."""
        # Store secret
        result = vault_client_with_vault.store_secret(
            "/test/path",
            {"key": "value"}
        )
        assert result is True
        
        # Verify Vault was called
        vault_client_with_vault.client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            path="test/path",
            secret={"key": "value"},
            cas=None,
            mount_point="secret"
        )
    
    def test_rotate_secret(self, vault_client_with_vault):
        """Test rotating a secret."""
        # Mock current secret
        mock_response = {
            "data": {
                "data": {
                    "api_key": "old_key",
                    "api_secret": "old_secret"
                }
            }
        }
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # Rotate api_key
        result = vault_client_with_vault.rotate_secret(
            "/genesis/exchange/api-keys",
            "api_key",
            "new_key"
        )
        assert result is True
        
        # Verify new secret was stored
        expected_data = {
            "api_key": "new_key",
            "api_secret": "old_secret"
        }
        vault_client_with_vault.client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            path="genesis/exchange/api-keys",
            secret=expected_data,
            cas=None,
            mount_point="secret"
        )
    
    def test_clear_cache_specific_path(self, vault_client_with_vault):
        """Test clearing cache for specific path."""
        # Populate cache
        vault_client_with_vault._cache["/path1:key1"] = SecretCache("value1", datetime.now(), 60)
        vault_client_with_vault._cache["/path1:key2"] = SecretCache("value2", datetime.now(), 60)
        vault_client_with_vault._cache["/path2:key1"] = SecretCache("value3", datetime.now(), 60)
        
        # Clear specific path
        vault_client_with_vault.clear_cache("/path1")
        
        # Check cache
        assert "/path1:key1" not in vault_client_with_vault._cache
        assert "/path1:key2" not in vault_client_with_vault._cache
        assert "/path2:key1" in vault_client_with_vault._cache
    
    def test_clear_cache_all(self, vault_client_with_vault):
        """Test clearing entire cache."""
        # Populate cache
        vault_client_with_vault._cache["/path1"] = SecretCache("value1", datetime.now(), 60)
        vault_client_with_vault._cache["/path2"] = SecretCache("value2", datetime.now(), 60)
        
        # Clear all
        vault_client_with_vault.clear_cache()
        
        # Check cache is empty
        assert len(vault_client_with_vault._cache) == 0
    
    def test_is_connected(self, vault_client_with_vault, vault_client_without_vault):
        """Test connection status check."""
        assert vault_client_with_vault.is_connected() is True
        assert vault_client_without_vault.is_connected() is False
    
    def test_health_check_with_vault(self, vault_client_with_vault):
        """Test health check with Vault connection."""
        vault_client_with_vault.client.sys.read_health_status.return_value = {"status": "active"}
        
        health = vault_client_with_vault.health_check()
        assert health["status"] == "healthy"
        assert health["mode"] == "vault"
        assert health["authenticated"] is True
        assert health["vault_status"] == {"status": "active"}
    
    def test_health_check_without_vault(self, vault_client_without_vault):
        """Test health check without Vault (fallback mode)."""
        health = vault_client_without_vault.health_check()
        assert health["status"] == "fallback"
        assert health["mode"] == "environment_variables"
    
    def test_get_database_encryption_key(self, vault_client_with_vault):
        """Test getting database encryption key."""
        # Mock Vault response
        mock_response = {
            "data": {
                "data": {
                    "key": "db_encryption_key_123"
                }
            }
        }
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        key = vault_client_with_vault.get_database_encryption_key()
        assert key == "db_encryption_key_123"
    
    def test_get_tls_certificates(self, vault_client_with_vault):
        """Test getting TLS certificates."""
        # Mock Vault response
        mock_response = {
            "data": {
                "data": {
                    "cert": "/path/to/cert.pem",
                    "key": "/path/to/key.pem",
                    "ca": "/path/to/ca.pem"
                }
            }
        }
        vault_client_with_vault.client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        certs = vault_client_with_vault.get_tls_certificates()
        assert certs == {
            "cert": "/path/to/cert.pem",
            "key": "/path/to/key.pem",
            "ca": "/path/to/ca.pem"
        }