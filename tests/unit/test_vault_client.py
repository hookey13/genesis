"""
Unit tests for HashiCorp Vault client
Tests the new hvac-based implementation with proper secret engines
"""

import os
import json
import time
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
import pytest
import hvac
from hvac.exceptions import VaultError, InvalidPath, Forbidden

from genesis.security.vault_client import (
    VaultConfig,
    VaultClient,
    AsyncVaultClient,
    DatabaseCredentials,
    EncryptedData,
    get_vault_client,
    get_secret,
    encrypt,
    decrypt
)


class TestVaultConfig:
    """Test VaultConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = VaultConfig()
        assert config.url == 'http://localhost:8200'
        assert config.namespace == 'genesis'
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.cache_ttl == 300
        assert config.verify_ssl == True
    
    def test_config_from_environment(self):
        """Test configuration from environment variables"""
        with patch.dict(os.environ, {
            'VAULT_ADDR': 'https://vault.example.com:8200',
            'VAULT_TOKEN': 'test-token',
            'VAULT_NAMESPACE': 'custom',
            'VAULT_VERIFY_SSL': 'false'
        }):
            config = VaultConfig()
            assert config.url == 'https://vault.example.com:8200'
            assert config.token == 'test-token'
            assert config.namespace == 'custom'
            assert config.verify_ssl == False


class TestVaultClient:
    """Test VaultClient class"""
    
    @pytest.fixture
    def mock_hvac_client(self):
        """Create mock hvac client"""
        mock_client = MagicMock(spec=hvac.Client)
        mock_client.is_authenticated.return_value = True
        # Create nested mock structure for hvac client
        mock_client.secrets.kv.v2.read_secret_version = MagicMock()
        mock_client.secrets.kv.v2.create_or_update_secret = MagicMock()
        mock_client.secrets.kv.v2.delete_metadata_and_all_versions = MagicMock()
        mock_client.secrets.kv.v2.list_secrets = MagicMock()
        mock_client.secrets.database.generate_credentials = MagicMock()
        mock_client.secrets.transit.encrypt_data = MagicMock()
        mock_client.secrets.transit.decrypt_data = MagicMock()
        mock_client.secrets.transit.rewrap_data = MagicMock()
        mock_client.sys.revoke_lease = MagicMock()
        mock_client.sys.read_health_status = MagicMock()
        mock_client.auth.token.lookup_self = MagicMock()
        mock_client.auth.token.renew_self = MagicMock()
        mock_client.adapter.close = MagicMock()
        return mock_client
    
    @pytest.fixture
    def vault_client(self, mock_hvac_client):
        """Create VaultClient with mocked hvac"""
        with patch('genesis.security.vault_client.hvac.Client', return_value=mock_hvac_client):
            config = VaultConfig(token='test-token')
            client = VaultClient(config)
            client._client = mock_hvac_client
            return client
    
    def test_initialization(self, mock_hvac_client):
        """Test client initialization"""
        with patch('genesis.security.vault_client.hvac.Client', return_value=mock_hvac_client):
            config = VaultConfig(token='test-token')
            client = VaultClient(config)
            
            assert client.config == config
            assert len(client._cache) == 0
            assert len(client._credentials_cache) == 0
            mock_hvac_client.is_authenticated.assert_called_once()
    
    def test_initialization_fails_without_auth(self):
        """Test initialization fails when not authenticated"""
        mock_client = MagicMock(spec=hvac.Client)
        mock_client.is_authenticated.return_value = False
        
        with patch('genesis.security.vault_client.hvac.Client', return_value=mock_client):
            config = VaultConfig(token='invalid-token')
            with pytest.raises(VaultError, match="Failed to authenticate"):
                VaultClient(config)
    
    def test_get_secret_success(self, vault_client, mock_hvac_client):
        """Test successful secret retrieval"""
        mock_response = {
            'data': {
                'data': {
                    'api_key': 'test-key',
                    'api_secret': 'test-secret'
                }
            }
        }
        mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # Get entire secret
        result = vault_client.get_secret('app/config')
        assert result == {'api_key': 'test-key', 'api_secret': 'test-secret'}
        
        # Get specific key
        result = vault_client.get_secret('app/config', 'api_key')
        assert result == 'test-key'
    
    def test_get_secret_caching(self, vault_client, mock_hvac_client):
        """Test secret caching mechanism"""
        mock_response = {
            'data': {
                'data': {
                    'value': 'cached-value'
                }
            }
        }
        mock_hvac_client.secrets.kv.v2.read_secret_version.return_value = mock_response
        
        # First call - should hit Vault
        result1 = vault_client.get_secret('test/path')
        assert mock_hvac_client.secrets.kv.v2.read_secret_version.call_count == 1
        
        # Second call - should use cache
        result2 = vault_client.get_secret('test/path')
        assert mock_hvac_client.secrets.kv.v2.read_secret_version.call_count == 1
        assert result1 == result2
        
        # Invalidate cache
        vault_client._cache.clear()
        
        # Third call - should hit Vault again
        result3 = vault_client.get_secret('test/path')
        assert mock_hvac_client.secrets.kv.v2.read_secret_version.call_count == 2
    
    def test_get_secret_invalid_path(self, vault_client, mock_hvac_client):
        """Test get_secret with invalid path"""
        mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = InvalidPath()
        
        with pytest.raises(InvalidPath):
            vault_client.get_secret('invalid/path')
    
    def test_get_secret_forbidden(self, vault_client, mock_hvac_client):
        """Test get_secret with forbidden access"""
        mock_hvac_client.secrets.kv.v2.read_secret_version.side_effect = Forbidden()
        
        with pytest.raises(Forbidden):
            vault_client.get_secret('forbidden/path')
    
    def test_put_secret(self, vault_client, mock_hvac_client):
        """Test storing a secret"""
        data = {'key': 'value', 'another': 'secret'}
        
        vault_client.put_secret('app/test', data)
        
        mock_hvac_client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            mount_point='genesis-secrets',
            path='app/test',
            secret=data
        )
    
    def test_delete_secret(self, vault_client, mock_hvac_client):
        """Test deleting a secret"""
        vault_client.delete_secret('app/test')
        
        mock_hvac_client.secrets.kv.v2.delete_metadata_and_all_versions.assert_called_once_with(
            mount_point='genesis-secrets',
            path='app/test'
        )
    
    def test_list_secrets(self, vault_client, mock_hvac_client):
        """Test listing secrets"""
        mock_response = {
            'data': {
                'keys': ['secret1/', 'secret2/', 'secret3']
            }
        }
        mock_hvac_client.secrets.kv.v2.list_secrets.return_value = mock_response
        
        result = vault_client.list_secrets('app')
        assert result == ['secret1/', 'secret2/', 'secret3']
    
    def test_get_database_credentials(self, vault_client, mock_hvac_client):
        """Test getting dynamic database credentials"""
        mock_response = {
            'data': {
                'username': 'v-token-genesis-app-1234',
                'password': 'generated-password'
            },
            'lease_duration': 3600,
            'lease_id': 'database/creds/genesis-app/abcd1234'
        }
        mock_hvac_client.secrets.database.generate_credentials.return_value = mock_response
        
        creds = vault_client.get_database_credentials('genesis-app')
        
        assert isinstance(creds, DatabaseCredentials)
        assert creds.username == 'v-token-genesis-app-1234'
        assert creds.password == 'generated-password'
        assert creds.ttl == 3600
        assert creds.lease_id == 'database/creds/genesis-app/abcd1234'
        assert isinstance(creds.expires_at, datetime)
    
    def test_database_credentials_caching(self, vault_client, mock_hvac_client):
        """Test database credentials caching and renewal"""
        mock_response = {
            'data': {
                'username': 'cached-user',
                'password': 'cached-pass'
            },
            'lease_duration': 3600,
            'lease_id': 'lease-123'
        }
        mock_hvac_client.secrets.database.generate_credentials.return_value = mock_response
        
        # First call - generate new credentials
        creds1 = vault_client.get_database_credentials('genesis-app')
        assert mock_hvac_client.secrets.database.generate_credentials.call_count == 1
        
        # Second call - use cached credentials
        creds2 = vault_client.get_database_credentials('genesis-app')
        assert mock_hvac_client.secrets.database.generate_credentials.call_count == 1
        assert creds1.username == creds2.username
    
    def test_revoke_database_credentials(self, vault_client, mock_hvac_client):
        """Test revoking database credentials"""
        lease_id = 'database/creds/genesis-app/test123'
        
        vault_client.revoke_database_credentials(lease_id)
        
        mock_hvac_client.sys.revoke_lease.assert_called_once_with(lease_id=lease_id)
    
    def test_encrypt_data(self, vault_client, mock_hvac_client):
        """Test data encryption"""
        mock_response = {
            'data': {
                'ciphertext': 'vault:v1:encrypted-data',
                'key_version': 1
            }
        }
        mock_hvac_client.secrets.transit.encrypt_data.return_value = mock_response
        
        result = vault_client.encrypt_data('sensitive-data')
        
        assert isinstance(result, EncryptedData)
        assert result.ciphertext == 'vault:v1:encrypted-data'
        assert result.key_version == 1
    
    def test_encrypt_data_with_context(self, vault_client, mock_hvac_client):
        """Test data encryption with context"""
        mock_response = {
            'data': {
                'ciphertext': 'vault:v1:encrypted-with-context',
                'key_version': 2
            }
        }
        mock_hvac_client.secrets.transit.encrypt_data.return_value = mock_response
        
        context = {'user_id': '123', 'purpose': 'api_key'}
        result = vault_client.encrypt_data('sensitive-data', context)
        
        assert result.ciphertext == 'vault:v1:encrypted-with-context'
        assert result.encryption_context == context
    
    def test_decrypt_data(self, vault_client, mock_hvac_client):
        """Test data decryption"""
        import base64
        plaintext = 'decrypted-data'
        mock_response = {
            'data': {
                'plaintext': base64.b64encode(plaintext.encode()).decode()
            }
        }
        mock_hvac_client.secrets.transit.decrypt_data.return_value = mock_response
        
        result = vault_client.decrypt_data('vault:v1:encrypted-data')
        
        assert result == plaintext
    
    def test_rewrap_data(self, vault_client, mock_hvac_client):
        """Test data rewrapping"""
        mock_response = {
            'data': {
                'ciphertext': 'vault:v2:rewrapped-data'
            }
        }
        mock_hvac_client.secrets.transit.rewrap_data.return_value = mock_response
        
        result = vault_client.rewrap_data('vault:v1:old-encrypted-data')
        
        assert result == 'vault:v2:rewrapped-data'
    
    def test_health_check(self, vault_client, mock_hvac_client):
        """Test health check"""
        mock_response = {
            'initialized': True,
            'sealed': False,
            'standby': False,
            'performance_standby': False,
            'version': '1.15.0',
            'cluster_name': 'genesis-vault',
            'cluster_id': 'cluster-123'
        }
        mock_hvac_client.sys.read_health_status.return_value = mock_response
        
        result = vault_client.health_check()
        
        assert result['initialized'] == True
        assert result['sealed'] == False
        assert result['version'] == '1.15.0'
        assert result['cluster_name'] == 'genesis-vault'
    
    def test_health_check_error(self, vault_client, mock_hvac_client):
        """Test health check with error"""
        mock_hvac_client.sys.read_health_status.side_effect = Exception("Connection failed")
        
        result = vault_client.health_check()
        
        assert result['initialized'] == False
        assert result['sealed'] == True
        assert 'error' in result
    
    def test_get_token_info(self, vault_client, mock_hvac_client):
        """Test getting token information"""
        mock_response = {
            'data': {
                'accessor': 'accessor-123',
                'creation_time': 1234567890,
                'creation_ttl': 2764800,
                'display_name': 'token-genesis-app',
                'expire_time': '2024-01-01T00:00:00Z',
                'policies': ['genesis-app', 'default']
            }
        }
        mock_hvac_client.auth.token.lookup_self.return_value = mock_response
        
        result = vault_client.get_token_info()
        
        assert result['accessor'] == 'accessor-123'
        assert 'genesis-app' in result['policies']
    
    def test_renew_token(self, vault_client, mock_hvac_client):
        """Test token renewal"""
        vault_client.renew_token()
        
        mock_hvac_client.auth.token.renew_self.assert_called_once()
    
    def test_clear_cache(self, vault_client):
        """Test cache clearing"""
        # Add some data to caches
        vault_client._cache['test'] = ('value', time.time())
        vault_client._credentials_cache['role'] = Mock()
        
        vault_client.clear_cache()
        
        assert len(vault_client._cache) == 0
        assert len(vault_client._credentials_cache) == 0
    
    def test_retry_operation(self, vault_client):
        """Test retry logic"""
        mock_operation = Mock()
        mock_operation.side_effect = [
            VaultError("First failure"),
            VaultError("Second failure"),
            "Success"
        ]
        
        with patch('time.sleep'):  # Speed up test
            result = vault_client._retry_operation(mock_operation)
        
        assert result == "Success"
        assert mock_operation.call_count == 3
    
    def test_retry_operation_max_retries(self, vault_client):
        """Test retry logic reaches max retries"""
        mock_operation = Mock()
        mock_operation.side_effect = VaultError("Always fails")
        
        with patch('time.sleep'):  # Speed up test
            with pytest.raises(VaultError, match="Always fails"):
                vault_client._retry_operation(mock_operation)
        
        assert mock_operation.call_count == vault_client.config.max_retries
    
    def test_context_manager(self, mock_hvac_client):
        """Test using VaultClient as context manager"""
        with patch('genesis.security.vault_client.hvac.Client', return_value=mock_hvac_client):
            config = VaultConfig(token='test-token')
            
            with VaultClient(config) as client:
                assert client._client is not None
            
            # After exiting context, client should be closed
            mock_hvac_client.adapter.close.assert_called_once()


class TestAsyncVaultClient:
    """Test AsyncVaultClient class"""
    
    @pytest.mark.asyncio
    async def test_async_get_secret(self):
        """Test async secret retrieval"""
        mock_sync_client = Mock()
        mock_sync_client.get_secret.return_value = {'key': 'value'}
        
        async_client = AsyncVaultClient()
        async_client._sync_client = mock_sync_client
        
        result = await async_client.get_secret('test/path')
        assert result == {'key': 'value'}
    
    @pytest.mark.asyncio
    async def test_async_encrypt_data(self):
        """Test async data encryption"""
        mock_encrypted = EncryptedData(
            ciphertext='vault:v1:encrypted',
            key_version=1
        )
        mock_sync_client = Mock()
        mock_sync_client.encrypt_data.return_value = mock_encrypted
        
        async_client = AsyncVaultClient()
        async_client._sync_client = mock_sync_client
        
        result = await async_client.encrypt_data('plaintext')
        assert result.ciphertext == 'vault:v1:encrypted'
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager"""
        async with AsyncVaultClient() as client:
            assert client._sync_client is not None


class TestSingletonFunctions:
    """Test singleton and convenience functions"""
    
    def test_get_vault_client_singleton(self):
        """Test singleton pattern for get_vault_client"""
        with patch('genesis.security.vault_client.VaultClient') as MockVaultClient:
            mock_instance = Mock()
            MockVaultClient.return_value = mock_instance
            
            # Reset singleton
            import genesis.security.vault_client as vault_module
            vault_module._vault_client = None
            
            # First call creates instance
            client1 = get_vault_client()
            assert MockVaultClient.call_count == 1
            
            # Second call returns same instance
            client2 = get_vault_client()
            assert MockVaultClient.call_count == 1
            assert client1 is client2
    
    def test_get_secret_convenience(self):
        """Test get_secret convenience function"""
        with patch('genesis.security.vault_client.get_vault_client') as mock_get_client:
            mock_client = Mock()
            mock_client.get_secret.return_value = 'secret-value'
            mock_get_client.return_value = mock_client
            
            result = get_secret('path/to/secret')
            
            assert result == 'secret-value'
            mock_client.get_secret.assert_called_once_with('path/to/secret', None)
    
    def test_encrypt_convenience(self):
        """Test encrypt convenience function"""
        with patch('genesis.security.vault_client.get_vault_client') as mock_get_client:
            mock_client = Mock()
            mock_encrypted = EncryptedData(ciphertext='encrypted', key_version=1)
            mock_client.encrypt_data.return_value = mock_encrypted
            mock_get_client.return_value = mock_client
            
            result = encrypt('plaintext')
            
            assert result == 'encrypted'
            mock_client.encrypt_data.assert_called_once_with('plaintext')
    
    def test_decrypt_convenience(self):
        """Test decrypt convenience function"""
        with patch('genesis.security.vault_client.get_vault_client') as mock_get_client:
            mock_client = Mock()
            mock_client.decrypt_data.return_value = 'decrypted'
            mock_get_client.return_value = mock_client
            
            result = decrypt('ciphertext')
            
            assert result == 'decrypted'
            mock_client.decrypt_data.assert_called_once_with('ciphertext')


class TestCacheValidation:
    """Test cache validation logic"""
    
    def test_cache_expiry(self):
        """Test cache expiry detection"""
        config = VaultConfig(cache_ttl=1)  # 1 second TTL
        
        with patch('genesis.security.vault_client.hvac.Client') as MockClient:
            mock_client = MockClient.return_value
            mock_client.is_authenticated.return_value = True
            
            vault_client = VaultClient(config)
            
            # Fresh timestamp
            fresh_time = time.time()
            assert vault_client._is_cache_valid(fresh_time) == True
            
            # Expired timestamp
            old_time = time.time() - 2  # 2 seconds ago
            assert vault_client._is_cache_valid(old_time) == False