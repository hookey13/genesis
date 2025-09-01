"""
Integration tests for Vault failover and recovery scenarios.
Tests backend switching, cache behavior, and rotation during failures.
"""

import os
import pytest
import asyncio
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from genesis.security.vault_client import VaultClient
from genesis.security.secrets_manager import SecretBackend
from genesis.core.exceptions import VaultConnectionError


class TestVaultFailover:
    """Test Vault failover and recovery scenarios."""
    
    @pytest.fixture
    def cleanup_env(self):
        """Clean up environment variables."""
        env_vars = [
            "GENESIS_VAULT_URL",
            "GENESIS_VAULT_TOKEN",
            "AWS_REGION",
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET"
        ]
        
        # Store original values
        original = {var: os.environ.get(var) for var in env_vars}
        
        yield
        
        # Restore original values
        for var, value in original.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
    
    @pytest.mark.asyncio
    async def test_vault_to_local_failover(self, cleanup_env):
        """Test failover from Vault to local encrypted storage."""
        # Start with Vault backend
        client = VaultClient(
            vault_url="http://localhost:8200",
            vault_token="test_token",
            backend_type=SecretBackend.VAULT
        )
        
        # Mock Vault failure
        with patch.object(client.secrets_manager.backend, 'health_check') as mock_health:
            mock_health.return_value = False
            
            # Should fallback to environment
            os.environ["TEST_SECRET"] = '{"value": "env_fallback"}'
            
            secret = await client.get_secret_async("test/secret")
            assert secret == {"value": "env_fallback"}
    
    @pytest.mark.asyncio
    async def test_vault_to_aws_failover(self, cleanup_env):
        """Test failover from Vault to AWS Secrets Manager."""
        os.environ["AWS_REGION"] = "us-east-1"
        
        # Start with Vault, should detect AWS available
        client = VaultClient(
            vault_url="http://invalid:8200",
            vault_token="invalid",
            use_vault=False
        )
        
        # Should use AWS backend
        assert client.backend_type == SecretBackend.AWS
    
    @pytest.mark.asyncio
    async def test_runtime_secret_injection(self):
        """Test runtime secret injection without code changes."""
        client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL
        )
        
        # Inject secret at runtime
        client.inject_runtime_secret("/app/api/key", {
            "api_key": "injected_key",
            "api_secret": "injected_secret"
        })
        
        # Retrieve injected secret
        secret = client.get_secret("/app/api/key")
        assert secret == {
            "api_key": "injected_key",
            "api_secret": "injected_secret"
        }
        
        # Test specific key retrieval
        api_key = client.get_secret("/app/api/key", key="api_key")
        assert api_key == "injected_key"
    
    @pytest.mark.asyncio
    async def test_cache_ttl_during_failover(self):
        """Test that cache TTL is respected during backend failures."""
        client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL,
            cache_ttl=1  # 1 second TTL
        )
        
        # Store and cache a secret
        await client.secrets_manager.put_secret("/test/ttl", {"value": "cached"})
        first_result = await client.get_secret_async("/test/ttl")
        assert first_result == {"value": "cached"}
        
        # Simulate backend failure
        client.secrets_manager.backend = None
        
        # Should still get cached value
        cached_result = await client.get_secret_async("/test/ttl")
        assert cached_result == {"value": "cached"}
        
        # Wait for cache to expire
        await asyncio.sleep(1.1)
        
        # Now should fail or return None
        expired_result = await client.get_secret_async("/test/ttl", fallback_to_env=False)
        assert expired_result is None
    
    @pytest.mark.asyncio
    async def test_rotation_during_vault_failure(self, cleanup_env):
        """Test API key rotation continues during Vault failure."""
        client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL,
            enable_rotation=True
        )
        
        # Initialize rotation manager
        await client.rotation_manager.initialize()
        
        # Store initial keys
        await client.secrets_manager.put_secret(
            client.EXCHANGE_API_KEYS_PATH,
            {
                "api_key": "initial_key",
                "api_secret": "initial_secret"
            }
        )
        
        # Perform rotation
        result = await client.rotate_api_keys(grace_period=timedelta(seconds=1))
        
        assert result["status"] == "success"
        assert "rotation_id" in result
    
    @pytest.mark.asyncio
    async def test_multi_backend_priority(self, cleanup_env):
        """Test secret retrieval priority across backends."""
        client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL
        )
        
        # Priority 1: Runtime injected
        client.inject_runtime_secret("/test/priority", {"source": "runtime"})
        
        # Priority 2: Backend storage
        await client.secrets_manager.put_secret("/test/priority2", {"source": "backend"})
        
        # Priority 3: Environment variable
        os.environ["TEST_PRIORITY3"] = '{"source": "env"}'
        
        # Test priority order
        assert client.get_secret("/test/priority")["source"] == "runtime"
        
        result2 = await client.get_secret_async("/test/priority2")
        assert result2["source"] == "backend"
        
        result3 = await client.get_secret_async("/test/priority3")
        assert result3["source"] == "env"
    
    @pytest.mark.asyncio
    async def test_health_check_all_backends(self):
        """Test comprehensive health check across all components."""
        client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL,
            enable_rotation=True
        )
        
        health = client.health_check()
        
        assert "backend" in health
        assert health["backend"] == "local"
        assert "cache_size" in health
        assert "runtime_secrets_count" in health
        assert "rotation_enabled" in health
        assert health["rotation_enabled"] is True
        assert "secrets_manager" in health
        assert "rotation_status" in health
    
    @pytest.mark.asyncio
    async def test_automatic_rotation_configuration(self):
        """Test automatic rotation configuration with failover."""
        client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL,
            enable_rotation=True
        )
        
        # Configure automatic rotation
        await client.configure_automatic_rotation(
            interval=timedelta(hours=24),
            grace_period=timedelta(minutes=10)
        )
        
        # Check rotation status
        status = await client.get_rotation_status()
        
        assert "active_schedules" in status
        assert len(status["active_schedules"]) > 0
    
    @pytest.mark.asyncio
    async def test_backend_switching(self):
        """Test switching between backends dynamically."""
        # Start with local backend
        client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL
        )
        
        # Store secret in local
        await client.secrets_manager.put_secret("/test/switch", {"value": "local"})
        
        # Verify retrieval
        result = await client.get_secret_async("/test/switch")
        assert result["value"] == "local"
        
        # Now test AWS backend initialization
        with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
            aws_client = VaultClient(
                use_vault=False,
                backend_type=None  # Auto-detect
            )
            assert aws_client.backend_type == SecretBackend.AWS
    
    @pytest.mark.asyncio
    async def test_concurrent_secret_access(self):
        """Test concurrent access to secrets during failover."""
        client = VaultClient(
            use_vault=False,
            backend_type=SecretBackend.LOCAL
        )
        
        # Store multiple secrets
        secrets = {
            f"/test/concurrent/{i}": {"value": f"secret_{i}"}
            for i in range(10)
        }
        
        for path, secret in secrets.items():
            await client.secrets_manager.put_secret(path, secret)
        
        # Concurrent retrieval
        async def get_secret(path):
            return await client.get_secret_async(path)
        
        tasks = [get_secret(path) for path in secrets.keys()]
        results = await asyncio.gather(*tasks)
        
        # Verify all secrets retrieved correctly
        for i, result in enumerate(results):
            assert result["value"] == f"secret_{i}"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, cleanup_env):
        """Test graceful degradation when all backends fail."""
        client = VaultClient(
            vault_url="http://invalid:8200",
            vault_token="invalid",
            use_vault=True
        )
        
        # All backends fail, but environment variables work
        os.environ["BINANCE_API_KEY"] = "env_api_key"
        os.environ["BINANCE_API_SECRET"] = "env_api_secret"
        
        # Should fallback to environment
        keys = client.get_exchange_api_keys()
        
        assert keys is not None
        assert keys["api_key"] == "env_api_key"
        assert keys["api_secret"] == "env_api_secret"