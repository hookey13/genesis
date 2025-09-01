"""
Unit tests for SecretsManager with multi-backend support.
Tests Vault, AWS, and local encrypted storage backends.
"""

import os
import json
import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import hvac
from cryptography.fernet import Fernet
from botocore.exceptions import ClientError

from genesis.security.secrets_manager import (
    SecretsManager,
    SecretBackend,
    VaultBackend,
    AWSSecretsManagerBackend,
    LocalEncryptedBackend,
    SecretAccess
)
from genesis.core.exceptions import (
    SecurityError,
    VaultConnectionError,
    EncryptionError
)


class TestLocalEncryptedBackend:
    """Test local encrypted storage backend."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    async def local_backend(self, temp_storage):
        """Create local backend instance."""
        backend = LocalEncryptedBackend(temp_storage)
        return backend
    
    @pytest.mark.asyncio
    async def test_put_and_get_secret(self, local_backend):
        """Test storing and retrieving encrypted secrets."""
        secret_data = {
            "api_key": "test_key_123",
            "api_secret": "test_secret_456"
        }
        
        # Store secret
        success = await local_backend.put_secret("test/path", secret_data)
        assert success is True
        
        # Retrieve secret
        retrieved = await local_backend.get_secret("test/path")
        assert retrieved == secret_data
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_secret(self, local_backend):
        """Test retrieving non-existent secret returns None."""
        result = await local_backend.get_secret("nonexistent/path")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete_secret(self, local_backend):
        """Test deleting encrypted secrets."""
        secret_data = {"key": "value"}
        
        # Store and verify
        await local_backend.put_secret("test/delete", secret_data)
        assert await local_backend.get_secret("test/delete") == secret_data
        
        # Delete
        success = await local_backend.delete_secret("test/delete")
        assert success is True
        
        # Verify deleted
        assert await local_backend.get_secret("test/delete") is None
    
    @pytest.mark.asyncio
    async def test_list_secrets(self, local_backend):
        """Test listing secrets with prefix."""
        # Store multiple secrets
        await local_backend.put_secret("app/db/password", {"value": "pass1"})
        await local_backend.put_secret("app/db/username", {"value": "user1"})
        await local_backend.put_secret("app/api/key", {"value": "key1"})
        
        # List with prefix
        db_secrets = await local_backend.list_secrets("app/db")
        assert len(db_secrets) == 2
        assert any("password" in s for s in db_secrets)
        assert any("username" in s for s in db_secrets)
    
    @pytest.mark.asyncio
    async def test_rotate_secret(self, local_backend):
        """Test secret rotation with backup."""
        original = {"api_key": "old_key"}
        new_secret = {"api_key": "new_key"}
        
        # Store original
        await local_backend.put_secret("api/key", original)
        
        # Rotate
        success = await local_backend.rotate_secret("api/key", new_secret)
        assert success is True
        
        # Verify new secret
        current = await local_backend.get_secret("api/key")
        assert current == new_secret
        
        # Check backup exists
        backups = await local_backend.list_secrets("api/key_backup")
        assert len(backups) > 0
    
    @pytest.mark.asyncio
    async def test_encryption_integrity(self, temp_storage):
        """Test that secrets are actually encrypted on disk."""
        backend = LocalEncryptedBackend(temp_storage)
        secret_data = {"sensitive": "data123"}
        
        await backend.put_secret("test/encrypted", secret_data)
        
        # Read encrypted file directly
        secret_file = temp_storage / "test_encrypted.encrypted"
        assert secret_file.exists()
        
        with open(secret_file, "rb") as f:
            raw_data = f.read()
        
        # Verify it's encrypted (not plaintext JSON)
        assert b"sensitive" not in raw_data
        assert b"data123" not in raw_data
        
        # Verify decryption works
        retrieved = await backend.get_secret("test/encrypted")
        assert retrieved == secret_data
    
    @pytest.mark.asyncio
    async def test_master_key_generation(self, temp_storage):
        """Test master key generation and persistence."""
        backend1 = LocalEncryptedBackend(temp_storage)
        secret_data = {"test": "data"}
        
        # Store with first backend
        await backend1.put_secret("test", secret_data)
        
        # Create new backend with same storage (should reuse key)
        backend2 = LocalEncryptedBackend(temp_storage)
        retrieved = await backend2.get_secret("test")
        
        assert retrieved == secret_data
    
    @pytest.mark.asyncio
    async def test_health_check(self, local_backend, temp_storage):
        """Test health check for local backend."""
        assert await local_backend.health_check() is True
        
        # Remove master key
        master_key_path = temp_storage / "master.key"
        master_key_path.unlink()
        
        assert await local_backend.health_check() is False


class TestVaultBackend:
    """Test HashiCorp Vault backend."""
    
    @pytest.fixture
    def mock_vault_client(self):
        """Create mock Vault client."""
        client = Mock(spec=hvac.Client)
        client.secrets = Mock()
        client.secrets.kv = Mock()
        client.secrets.kv.v2 = Mock()
        client.sys = Mock()
        return client
    
    @pytest.fixture
    def vault_backend(self, mock_vault_client):
        """Create Vault backend with mocked client."""
        backend = VaultBackend("http://localhost:8200", "test_token")
        backend.client = mock_vault_client
        return backend
    
    @pytest.mark.asyncio
    async def test_get_secret(self, vault_backend, mock_vault_client):
        """Test retrieving secret from Vault."""
        mock_response = {
            "data": {
                "data": {
                    "api_key": "vault_key",
                    "api_secret": "vault_secret"
                }
            }
        }
        
        mock_vault_client.secrets.kv.v2.read_secret_version = Mock(
            return_value=mock_response
        )
        
        result = await vault_backend.get_secret("test/path")
        assert result == {"api_key": "vault_key", "api_secret": "vault_secret"}
    
    @pytest.mark.asyncio
    async def test_get_secret_not_found(self, vault_backend, mock_vault_client):
        """Test retrieving non-existent secret from Vault."""
        mock_vault_client.secrets.kv.v2.read_secret_version = Mock(
            side_effect=hvac.exceptions.InvalidPath()
        )
        
        result = await vault_backend.get_secret("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_put_secret(self, vault_backend, mock_vault_client):
        """Test storing secret in Vault."""
        secret_data = {"key": "value"}
        
        mock_vault_client.secrets.kv.v2.create_or_update_secret = Mock()
        
        success = await vault_backend.put_secret("test/path", secret_data)
        assert success is True
        
        mock_vault_client.secrets.kv.v2.create_or_update_secret.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rotate_secret(self, vault_backend, mock_vault_client):
        """Test secret rotation in Vault."""
        old_secret = {"api_key": "old"}
        new_secret = {"api_key": "new"}
        
        # Mock getting current secret
        mock_vault_client.secrets.kv.v2.read_secret_version = Mock(
            return_value={"data": {"data": old_secret}}
        )
        mock_vault_client.secrets.kv.v2.create_or_update_secret = Mock()
        
        success = await vault_backend.rotate_secret("api/key", new_secret)
        assert success is True
        
        # Verify backup and update were called
        assert mock_vault_client.secrets.kv.v2.create_or_update_secret.call_count == 2
    
    @pytest.mark.asyncio
    async def test_health_check(self, vault_backend, mock_vault_client):
        """Test Vault health check."""
        mock_vault_client.sys.is_sealed = Mock(return_value=False)
        
        is_healthy = await vault_backend.health_check()
        assert is_healthy is True
        
        # Test sealed vault
        mock_vault_client.sys.is_sealed = Mock(return_value=True)
        is_healthy = await vault_backend.health_check()
        assert is_healthy is False


class TestAWSSecretsManagerBackend:
    """Test AWS Secrets Manager backend."""
    
    @pytest.fixture
    def mock_aws_client(self):
        """Create mock AWS Secrets Manager client."""
        return Mock()
    
    @pytest.fixture
    def aws_backend(self, mock_aws_client):
        """Create AWS backend with mocked client."""
        backend = AWSSecretsManagerBackend()
        backend.client = mock_aws_client
        return backend
    
    @pytest.mark.asyncio
    async def test_get_secret(self, aws_backend, mock_aws_client):
        """Test retrieving secret from AWS."""
        secret_data = {"api_key": "aws_key"}
        mock_aws_client.get_secret_value = Mock(
            return_value={"SecretString": json.dumps(secret_data)}
        )
        
        result = await aws_backend.get_secret("test-secret")
        assert result == secret_data
    
    @pytest.mark.asyncio
    async def test_get_secret_not_found(self, aws_backend, mock_aws_client):
        """Test retrieving non-existent secret from AWS."""
        error_response = {"Error": {"Code": "ResourceNotFoundException"}}
        mock_aws_client.get_secret_value = Mock(
            side_effect=ClientError(error_response, "GetSecretValue")
        )
        
        result = await aws_backend.get_secret("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_put_secret_update(self, aws_backend, mock_aws_client):
        """Test updating existing secret in AWS."""
        secret_data = {"key": "value"}
        mock_aws_client.update_secret = Mock()
        
        success = await aws_backend.put_secret("test-secret", secret_data)
        assert success is True
        
        mock_aws_client.update_secret.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_put_secret_create(self, aws_backend, mock_aws_client):
        """Test creating new secret in AWS."""
        secret_data = {"key": "value"}
        
        # Update fails with not found
        error_response = {"Error": {"Code": "ResourceNotFoundException"}}
        mock_aws_client.update_secret = Mock(
            side_effect=ClientError(error_response, "UpdateSecret")
        )
        mock_aws_client.create_secret = Mock()
        
        success = await aws_backend.put_secret("new-secret", secret_data)
        assert success is True
        
        mock_aws_client.create_secret.assert_called_once()


class TestSecretsManager:
    """Test main SecretsManager class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    async def secrets_manager(self, temp_storage):
        """Create SecretsManager with local backend."""
        config = {"storage_path": temp_storage}
        manager = SecretsManager(SecretBackend.LOCAL, config)
        return manager
    
    @pytest.mark.asyncio
    async def test_get_secret_with_cache(self, secrets_manager):
        """Test secret retrieval with caching."""
        secret_data = {"api_key": "test123"}
        
        # Store secret
        await secrets_manager.put_secret("test/cache", secret_data)
        
        # First retrieval (from backend)
        result1 = await secrets_manager.get_secret("test/cache")
        assert result1 == secret_data
        
        # Second retrieval (from cache)
        result2 = await secrets_manager.get_secret("test/cache")
        assert result2 == secret_data
        
        # Verify cache hit
        assert "test/cache" in secrets_manager.cache
    
    @pytest.mark.asyncio
    async def test_get_secret_env_fallback(self, secrets_manager):
        """Test fallback to environment variables."""
        os.environ["TEST_ENV_SECRET"] = json.dumps({"env_key": "env_value"})
        
        result = await secrets_manager.get_secret(
            "test/env/secret",
            fallback_to_env=True
        )
        
        assert result == {"env_key": "env_value"}
        
        # Cleanup
        del os.environ["TEST_ENV_SECRET"]
    
    @pytest.mark.asyncio
    async def test_put_secret_invalidates_cache(self, secrets_manager):
        """Test that putting a secret invalidates cache."""
        original = {"version": "1"}
        updated = {"version": "2"}
        
        # Store and cache
        await secrets_manager.put_secret("test/invalidate", original)
        await secrets_manager.get_secret("test/invalidate")
        assert "test/invalidate" in secrets_manager.cache
        
        # Update secret
        await secrets_manager.put_secret("test/invalidate", updated)
        assert "test/invalidate" not in secrets_manager.cache
        
        # Verify updated value
        result = await secrets_manager.get_secret("test/invalidate")
        assert result == updated
    
    @pytest.mark.asyncio
    async def test_rotate_api_keys(self, secrets_manager):
        """Test API key rotation with dual-key strategy."""
        # Store initial keys
        initial_keys = {
            "api_key": "original_key",
            "api_secret": "original_secret"
        }
        await secrets_manager.put_secret("/genesis/exchange/api-keys", initial_keys)
        
        # Rotate keys
        result = await secrets_manager.rotate_api_keys()
        
        assert result["status"] == "success"
        assert "rotation_id" in result
        assert "grace_period_end" in result
        
        # Verify dual keys stored
        current = await secrets_manager.get_secret(
            "/genesis/exchange/api-keys",
            use_cache=False
        )
        assert "primary" in current
        assert "secondary" in current
        assert current["secondary"] == initial_keys
    
    @pytest.mark.asyncio
    async def test_generate_temporary_credential(self, secrets_manager):
        """Test temporary credential generation."""
        temp_cred = await secrets_manager.generate_temporary_credential(
            purpose="test_operation",
            ttl=timedelta(hours=2),
            permissions=["read", "write"]
        )
        
        assert "credential_id" in temp_cred
        assert temp_cred["purpose"] == "test_operation"
        assert temp_cred["permissions"] == ["read", "write"]
        assert "token" in temp_cred
        assert "expires_at" in temp_cred
        
        # Verify stored
        path = f"/genesis/temp/{temp_cred['credential_id']}"
        stored = await secrets_manager.get_secret(path, use_cache=False)
        assert stored == temp_cred
    
    @pytest.mark.asyncio
    async def test_audit_logging(self, secrets_manager):
        """Test audit trail for secret operations."""
        # Perform operations
        await secrets_manager.put_secret("audit/test", {"data": "value"})
        await secrets_manager.get_secret("audit/test")
        await secrets_manager.delete_secret("audit/test")
        
        # Check audit log
        audit_log = secrets_manager.get_audit_log()
        assert len(audit_log) >= 3
        
        # Verify log entries
        actions = [entry["action"] for entry in audit_log]
        assert "put" in actions
        assert "get" in actions
        assert "delete" in actions
    
    @pytest.mark.asyncio
    async def test_audit_log_filtering(self, secrets_manager):
        """Test audit log filtering capabilities."""
        # Create some audit entries
        await secrets_manager.put_secret("app/db/password", {"value": "pass"})
        await secrets_manager.put_secret("app/api/key", {"value": "key"})
        await secrets_manager.get_secret("app/db/password")
        
        # Filter by path
        db_logs = secrets_manager.get_audit_log(path_filter="app/db")
        assert all("app/db" in log["path"] for log in db_logs)
        
        # Filter by time
        start_time = datetime.utcnow() - timedelta(minutes=1)
        recent_logs = secrets_manager.get_audit_log(start_time=start_time)
        assert len(recent_logs) > 0
    
    @pytest.mark.asyncio
    async def test_health_check(self, secrets_manager):
        """Test health check functionality."""
        health = await secrets_manager.health_check()
        
        assert health["backend"] == "local"
        assert "backend_healthy" in health
        assert "cache_size" in health
        assert "audit_log_size" in health
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_backend_error_handling(self):
        """Test error handling for backend failures."""
        # Create manager with invalid Vault config
        manager = SecretsManager(
            SecretBackend.VAULT,
            {"vault_url": "http://invalid:8200", "vault_token": "invalid"}
        )
        
        with pytest.raises(VaultConnectionError):
            await manager.get_secret("test/path", fallback_to_env=False)
    
    @pytest.mark.asyncio
    async def test_encryption_error_handling(self, temp_storage):
        """Test handling of encryption errors."""
        backend = LocalEncryptedBackend(temp_storage)
        
        # Corrupt the cipher
        backend.cipher = Mock(spec=Fernet)
        backend.cipher.decrypt = Mock(side_effect=Exception("Decryption failed"))
        
        # Store a secret normally
        good_backend = LocalEncryptedBackend(temp_storage)
        await good_backend.put_secret("test", {"data": "value"})
        
        # Try to read with corrupted cipher
        with pytest.raises(EncryptionError):
            await backend.get_secret("test")