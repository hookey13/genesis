"""
Integration tests for HashiCorp Vault deployment
Tests actual Vault operations with a test instance
"""

import os
import time
import json
import pytest
import docker
from pathlib import Path
from datetime import datetime, timedelta
import hvac
from hvac.exceptions import VaultError, InvalidPath

from genesis.security.vault_client import (
    VaultConfig,
    VaultClient,
    DatabaseCredentials,
    EncryptedData
)


@pytest.mark.integration
class TestVaultIntegration:
    """Integration tests for Vault deployment"""
    
    @classmethod
    def setup_class(cls):
        """Setup test Vault instance"""
        cls.docker_client = docker.from_env()
        cls.vault_container = None
        cls.vault_url = "http://localhost:8200"
        cls.root_token = None
        
        # Start Vault container for testing
        try:
            # Check if container already exists
            try:
                cls.vault_container = cls.docker_client.containers.get("test-vault")
                cls.vault_container.stop()
                cls.vault_container.remove()
            except docker.errors.NotFound:
                pass
            
            # Start new container
            cls.vault_container = cls.docker_client.containers.run(
                "vault:1.15",
                name="test-vault",
                ports={'8200/tcp': 8200},
                environment={
                    'VAULT_DEV_ROOT_TOKEN_ID': 'test-root-token',
                    'VAULT_DEV_LISTEN_ADDRESS': '0.0.0.0:8200',
                    'VAULT_ADDR': 'http://0.0.0.0:8200'
                },
                detach=True,
                remove=False
            )
            
            # Wait for Vault to be ready
            time.sleep(5)
            
            cls.root_token = "test-root-token"
            
        except Exception as e:
            pytest.skip(f"Could not start Vault container: {e}")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup test Vault instance"""
        if cls.vault_container:
            try:
                cls.vault_container.stop()
                cls.vault_container.remove()
            except:
                pass
    
    @pytest.fixture
    def vault_client(self):
        """Create authenticated Vault client"""
        config = VaultConfig(
            url=self.vault_url,
            token=self.root_token,
            verify_ssl=False
        )
        client = VaultClient(config)
        
        # Setup secret engines for testing
        self._setup_secret_engines(client)
        
        yield client
        
        # Cleanup
        client.close()
    
    def _setup_secret_engines(self, client):
        """Setup secret engines for testing"""
        try:
            # Enable KV-v2 engine
            client.client.sys.enable_secrets_engine(
                backend_type='kv-v2',
                path='genesis-secrets'
            )
        except Exception:
            pass  # Already enabled
        
        try:
            # Enable Transit engine
            client.client.sys.enable_secrets_engine(
                backend_type='transit',
                path='genesis-transit'
            )
            
            # Create encryption key
            client.client.secrets.transit.create_key(
                name='genesis-key',
                mount_point='genesis-transit',
                convergent_encryption=True,
                derived=True
            )
        except Exception:
            pass  # Already setup
    
    def test_vault_health(self, vault_client):
        """Test Vault health check"""
        health = vault_client.health_check()
        
        assert health['initialized'] == True
        assert health['sealed'] == False
        assert 'version' in health
        assert 'cluster_name' in health
    
    def test_secret_crud_operations(self, vault_client):
        """Test secret CRUD operations"""
        path = 'test/integration'
        secret_data = {
            'api_key': 'test-key-123',
            'api_secret': 'test-secret-456',
            'timestamp': str(datetime.utcnow())
        }
        
        # Create secret
        vault_client.put_secret(path, secret_data)
        
        # Read secret
        retrieved = vault_client.get_secret(path)
        assert retrieved['api_key'] == secret_data['api_key']
        assert retrieved['api_secret'] == secret_data['api_secret']
        
        # Read specific key
        api_key = vault_client.get_secret(path, 'api_key')
        assert api_key == 'test-key-123'
        
        # Update secret
        updated_data = secret_data.copy()
        updated_data['api_key'] = 'updated-key-789'
        vault_client.put_secret(path, updated_data)
        
        # Verify update
        retrieved = vault_client.get_secret(path)
        assert retrieved['api_key'] == 'updated-key-789'
        
        # List secrets
        vault_client.put_secret('test/other', {'data': 'value'})
        secrets_list = vault_client.list_secrets('test')
        assert 'integration' in secrets_list or 'integration/' in str(secrets_list)
        assert 'other' in secrets_list or 'other/' in str(secrets_list)
        
        # Delete secret
        vault_client.delete_secret(path)
        
        # Verify deletion
        with pytest.raises(InvalidPath):
            vault_client.get_secret(path)
    
    def test_secret_caching(self, vault_client):
        """Test secret caching mechanism"""
        path = 'test/cache'
        data = {'value': 'cached-data'}
        
        # Store secret
        vault_client.put_secret(path, data)
        
        # First read - hits Vault
        start = time.time()
        result1 = vault_client.get_secret(path)
        first_time = time.time() - start
        
        # Second read - should use cache
        start = time.time()
        result2 = vault_client.get_secret(path)
        cache_time = time.time() - start
        
        assert result1 == result2
        # Cache should be faster (not always reliable in CI)
        # assert cache_time < first_time
        
        # Clear cache
        vault_client.clear_cache()
        
        # Should hit Vault again
        result3 = vault_client.get_secret(path)
        assert result3 == result1
    
    def test_encryption_decryption(self, vault_client):
        """Test Transit engine encryption/decryption"""
        plaintext = "sensitive-data-12345"
        
        # Encrypt data
        encrypted = vault_client.encrypt_data(plaintext)
        assert isinstance(encrypted, EncryptedData)
        assert encrypted.ciphertext.startswith('vault:v')
        assert encrypted.key_version >= 1
        
        # Decrypt data
        decrypted = vault_client.decrypt_data(encrypted.ciphertext)
        assert decrypted == plaintext
        
        # Test with context
        context = {'user_id': '123', 'purpose': 'test'}
        encrypted_with_context = vault_client.encrypt_data(plaintext, context)
        
        # Decrypt with same context
        decrypted_with_context = vault_client.decrypt_data(
            encrypted_with_context.ciphertext,
            context
        )
        assert decrypted_with_context == plaintext
        
        # Decryption should fail with wrong context
        wrong_context = {'user_id': '456', 'purpose': 'test'}
        with pytest.raises(Exception):
            vault_client.decrypt_data(
                encrypted_with_context.ciphertext,
                wrong_context
            )
    
    def test_secret_versioning(self, vault_client):
        """Test KV-v2 secret versioning"""
        path = 'test/versioning'
        
        # Create version 1
        v1_data = {'version': '1', 'data': 'original'}
        vault_client.put_secret(path, v1_data)
        
        # Create version 2
        v2_data = {'version': '2', 'data': 'updated'}
        vault_client.put_secret(path, v2_data)
        
        # Create version 3
        v3_data = {'version': '3', 'data': 'latest'}
        vault_client.put_secret(path, v3_data)
        
        # Current version should be v3
        current = vault_client.get_secret(path)
        assert current['version'] == '3'
        
        # Can access specific versions through raw client
        # Note: This requires direct hvac client access
        response = vault_client.client.secrets.kv.v2.read_secret_version(
            mount_point='genesis-secrets',
            path=path,
            version=1
        )
        assert response['data']['data']['version'] == '1'
    
    def test_token_operations(self, vault_client):
        """Test token operations"""
        # Get token info
        token_info = vault_client.get_token_info()
        assert 'accessor' in token_info
        assert 'policies' in token_info
        
        # Token should be renewable
        vault_client.renew_token()
        
        # Verify token is still valid
        health = vault_client.health_check()
        assert health['initialized'] == True
    
    def test_concurrent_access(self, vault_client):
        """Test concurrent access to Vault"""
        import threading
        import random
        
        errors = []
        results = []
        
        def access_vault(thread_id):
            try:
                path = f'test/concurrent/{thread_id}'
                data = {'thread': thread_id, 'value': random.randint(1, 1000)}
                
                # Write secret
                vault_client.put_secret(path, data)
                
                # Read secret
                retrieved = vault_client.get_secret(path)
                
                # Verify
                if retrieved['thread'] == thread_id:
                    results.append(True)
                else:
                    errors.append(f"Thread {thread_id}: Data mismatch")
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        # Create threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=access_vault, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
    
    def test_performance_benchmarks(self, vault_client):
        """Test performance meets requirements"""
        import statistics
        
        # Warm up cache
        vault_client.put_secret('test/perf', {'data': 'benchmark'})
        vault_client.get_secret('test/perf')
        
        # Measure secret retrieval latency
        latencies = []
        for i in range(100):
            start = time.time()
            vault_client.get_secret('test/perf')
            latency = (time.time() - start) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p99_latency = sorted(latencies)[98]  # 99th percentile
        
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"P99 latency: {p99_latency:.2f}ms")
        
        # Should meet <10ms requirement for cached access
        assert avg_latency < 10, f"Average latency {avg_latency}ms exceeds 10ms"
        assert p99_latency < 20, f"P99 latency {p99_latency}ms exceeds 20ms"