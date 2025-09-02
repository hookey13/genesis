"""
HashiCorp Vault Client for Genesis Trading Platform
Provides secure secret management and dynamic credential generation
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from functools import lru_cache
import hvac
from hvac.exceptions import VaultError, InvalidPath, Forbidden

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class VaultConfig:
    """Vault configuration settings"""
    url: str = field(default_factory=lambda: os.getenv('VAULT_ADDR', 'http://localhost:8200'))
    token: Optional[str] = field(default_factory=lambda: os.getenv('VAULT_TOKEN'))
    namespace: Optional[str] = field(default_factory=lambda: os.getenv('VAULT_NAMESPACE', 'genesis'))
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_ttl: int = 300  # 5 minutes
    verify_ssl: bool = field(default_factory=lambda: os.getenv('VAULT_VERIFY_SSL', 'true').lower() == 'true')


@dataclass
class DatabaseCredentials:
    """Dynamic database credentials from Vault"""
    username: str
    password: str
    ttl: int
    lease_id: str
    expires_at: datetime


@dataclass
class EncryptedData:
    """Encrypted data response from Transit engine"""
    ciphertext: str
    key_version: int
    encryption_context: Optional[Dict[str, str]] = None


class VaultClient:
    """
    Thread-safe Vault client with caching and retry logic
    """
    
    def __init__(self, config: Optional[VaultConfig] = None):
        """Initialize Vault client with configuration"""
        self.config = config or VaultConfig()
        self._client: Optional[hvac.Client] = None
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._credentials_cache: Dict[str, DatabaseCredentials] = {}
        self._lock = asyncio.Lock()
        self._initialize_client()
        
    def _initialize_client(self) -> None:
        """Initialize the HVAC client"""
        try:
            self._client = hvac.Client(
                url=self.config.url,
                token=self.config.token,
                namespace=self.config.namespace,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl
            )
            
            # Verify authentication
            if not self._client.is_authenticated():
                raise VaultError("Failed to authenticate with Vault")
                
            logger.info("Vault client initialized", 
                       vault_url=self.config.url,
                       namespace=self.config.namespace)
                       
        except Exception as e:
            logger.error("Failed to initialize Vault client", error=str(e))
            raise
    
    @property
    def client(self) -> hvac.Client:
        """Get the HVAC client, reinitializing if necessary"""
        if not self._client or not self._client.is_authenticated():
            self._initialize_client()
        return self._client
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached data is still valid"""
        return (time.time() - timestamp) < self.config.cache_ttl
    
    def _retry_operation(self, operation, *args, **kwargs) -> Any:
        """Execute operation with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                return operation(*args, **kwargs)
            except (VaultError, ConnectionError) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Vault operation failed, retrying...", 
                                 attempt=attempt + 1,
                                 error=str(e))
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                    
                    # Reinitialize client on connection errors
                    if isinstance(e, ConnectionError):
                        self._initialize_client()
        
        logger.error("Vault operation failed after retries", 
                    error=str(last_error))
        raise last_error
    
    # Secret Management Methods
    
    def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """
        Get secret from KV-v2 engine
        
        Args:
            path: Secret path (e.g., 'app/config')
            key: Optional specific key from secret
            
        Returns:
            Secret value or entire secret dict
        """
        cache_key = f"secret:{path}:{key}"
        
        # Check cache
        if cache_key in self._cache:
            value, timestamp = self._cache[cache_key]
            if self._is_cache_valid(timestamp):
                logger.debug("Returning cached secret", path=path, key=key)
                return value
        
        try:
            # Read from Vault
            response = self._retry_operation(
                self.client.secrets.kv.v2.read_secret_version,
                mount_point='genesis-secrets',
                path=path
            )
            
            data = response['data']['data']
            
            # Extract specific key if requested
            if key:
                value = data.get(key)
                if value is None:
                    raise KeyError(f"Key '{key}' not found in secret '{path}'")
            else:
                value = data
            
            # Update cache
            self._cache[cache_key] = (value, time.time())
            
            logger.info("Secret retrieved", path=path, key=key)
            return value
            
        except InvalidPath:
            logger.error("Secret not found", path=path)
            raise
        except Forbidden:
            logger.error("Access denied to secret", path=path)
            raise
    
    def put_secret(self, path: str, data: Dict[str, Any]) -> None:
        """
        Store secret in KV-v2 engine
        
        Args:
            path: Secret path
            data: Secret data dictionary
        """
        try:
            self._retry_operation(
                self.client.secrets.kv.v2.create_or_update_secret,
                mount_point='genesis-secrets',
                path=path,
                secret=data
            )
            
            # Invalidate cache for this path
            cache_keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"secret:{path}")]
            for key in cache_keys_to_remove:
                del self._cache[key]
            
            logger.info("Secret stored", path=path)
            
        except Exception as e:
            logger.error("Failed to store secret", path=path, error=str(e))
            raise
    
    def delete_secret(self, path: str) -> None:
        """Delete secret from KV-v2 engine"""
        try:
            self._retry_operation(
                self.client.secrets.kv.v2.delete_metadata_and_all_versions,
                mount_point='genesis-secrets',
                path=path
            )
            
            # Invalidate cache
            cache_keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"secret:{path}")]
            for key in cache_keys_to_remove:
                del self._cache[key]
            
            logger.info("Secret deleted", path=path)
            
        except Exception as e:
            logger.error("Failed to delete secret", path=path, error=str(e))
            raise
    
    def list_secrets(self, path: str = "") -> List[str]:
        """List secrets at given path"""
        try:
            response = self._retry_operation(
                self.client.secrets.kv.v2.list_secrets,
                mount_point='genesis-secrets',
                path=path
            )
            return response.get('data', {}).get('keys', [])
            
        except InvalidPath:
            return []
    
    # Database Credential Management
    
    def get_database_credentials(self, role: str = 'genesis-app') -> DatabaseCredentials:
        """
        Get dynamic database credentials
        
        Args:
            role: Database role name
            
        Returns:
            DatabaseCredentials object
        """
        # Check if we have valid cached credentials
        if role in self._credentials_cache:
            creds = self._credentials_cache[role]
            # Renew if less than 20% of TTL remaining
            remaining_ttl = (creds.expires_at - datetime.utcnow()).total_seconds()
            if remaining_ttl > (creds.ttl * 0.2):
                logger.debug("Using cached database credentials", 
                           role=role, 
                           remaining_ttl=remaining_ttl)
                return creds
        
        try:
            response = self._retry_operation(
                self.client.secrets.database.generate_credentials,
                name=role,
                mount_point='genesis-database'
            )
            
            creds = DatabaseCredentials(
                username=response['data']['username'],
                password=response['data']['password'],
                ttl=response['lease_duration'],
                lease_id=response['lease_id'],
                expires_at=datetime.utcnow() + timedelta(seconds=response['lease_duration'])
            )
            
            # Cache the credentials
            self._credentials_cache[role] = creds
            
            logger.info("Database credentials generated", 
                       role=role,
                       username=creds.username,
                       ttl=creds.ttl)
            
            return creds
            
        except Exception as e:
            logger.error("Failed to get database credentials", 
                        role=role,
                        error=str(e))
            raise
    
    def revoke_database_credentials(self, lease_id: str) -> None:
        """Revoke database credentials by lease ID"""
        try:
            self._retry_operation(
                self.client.sys.revoke_lease,
                lease_id=lease_id
            )
            
            # Remove from cache
            for role, creds in list(self._credentials_cache.items()):
                if creds.lease_id == lease_id:
                    del self._credentials_cache[role]
            
            logger.info("Database credentials revoked", lease_id=lease_id)
            
        except Exception as e:
            logger.error("Failed to revoke credentials", 
                        lease_id=lease_id,
                        error=str(e))
    
    # Encryption Services
    
    def encrypt_data(self, 
                    plaintext: str,
                    context: Optional[Dict[str, str]] = None) -> EncryptedData:
        """
        Encrypt data using Transit engine
        
        Args:
            plaintext: Data to encrypt
            context: Optional encryption context for convergent encryption
            
        Returns:
            EncryptedData object
        """
        try:
            import base64
            
            # Encode plaintext to base64
            encoded = base64.b64encode(plaintext.encode()).decode()
            
            # Prepare request
            request_data = {'plaintext': encoded}
            if context:
                request_data['context'] = base64.b64encode(
                    json.dumps(context).encode()
                ).decode()
            
            response = self._retry_operation(
                self.client.secrets.transit.encrypt_data,
                mount_point='genesis-transit',
                name='genesis-key',
                **request_data
            )
            
            return EncryptedData(
                ciphertext=response['data']['ciphertext'],
                key_version=response['data'].get('key_version', 1),
                encryption_context=context
            )
            
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            raise
    
    def decrypt_data(self,
                    ciphertext: str,
                    context: Optional[Dict[str, str]] = None) -> str:
        """
        Decrypt data using Transit engine
        
        Args:
            ciphertext: Encrypted data
            context: Optional decryption context
            
        Returns:
            Decrypted plaintext
        """
        try:
            import base64
            
            # Prepare request
            request_data = {'ciphertext': ciphertext}
            if context:
                request_data['context'] = base64.b64encode(
                    json.dumps(context).encode()
                ).decode()
            
            response = self._retry_operation(
                self.client.secrets.transit.decrypt_data,
                mount_point='genesis-transit',
                name='genesis-key',
                **request_data
            )
            
            # Decode from base64
            plaintext = base64.b64decode(response['data']['plaintext']).decode()
            
            return plaintext
            
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise
    
    def rewrap_data(self, ciphertext: str) -> str:
        """
        Rewrap encrypted data with latest key version
        
        Args:
            ciphertext: Encrypted data to rewrap
            
        Returns:
            Rewrapped ciphertext
        """
        try:
            response = self._retry_operation(
                self.client.secrets.transit.rewrap_data,
                mount_point='genesis-transit',
                name='genesis-key',
                ciphertext=ciphertext
            )
            
            return response['data']['ciphertext']
            
        except Exception as e:
            logger.error("Rewrap failed", error=str(e))
            raise
    
    # Health and Status Methods
    
    def health_check(self) -> Dict[str, Any]:
        """Check Vault health status"""
        try:
            response = self.client.sys.read_health_status()
            return {
                'initialized': response.get('initialized', False),
                'sealed': response.get('sealed', True),
                'standby': response.get('standby', False),
                'performance_standby': response.get('performance_standby', False),
                'version': response.get('version', 'unknown'),
                'cluster_name': response.get('cluster_name', 'unknown'),
                'cluster_id': response.get('cluster_id', 'unknown')
            }
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                'initialized': False,
                'sealed': True,
                'error': str(e)
            }
    
    def get_token_info(self) -> Dict[str, Any]:
        """Get current token information"""
        try:
            response = self.client.auth.token.lookup_self()
            return response['data']
        except Exception as e:
            logger.error("Failed to get token info", error=str(e))
            raise
    
    def renew_token(self) -> None:
        """Renew current token"""
        try:
            self.client.auth.token.renew_self()
            logger.info("Token renewed successfully")
        except Exception as e:
            logger.error("Failed to renew token", error=str(e))
            raise
    
    # Cleanup Methods
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._cache.clear()
        self._credentials_cache.clear()
        logger.info("Cache cleared")
    
    def close(self) -> None:
        """Close client and cleanup resources"""
        self.clear_cache()
        if self._client:
            self._client.adapter.close()
        logger.info("Vault client closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Async wrapper for async applications
class AsyncVaultClient:
    """Async wrapper for VaultClient"""
    
    def __init__(self, config: Optional[VaultConfig] = None):
        self._sync_client = VaultClient(config)
        self._executor = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous function in executor"""
        loop = asyncio.get_event_loop()
        if self._executor is None:
            import concurrent.futures
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        return await loop.run_in_executor(self._executor, func, *args, **kwargs)
    
    async def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """Async get secret"""
        return await self._run_sync(self._sync_client.get_secret, path, key)
    
    async def put_secret(self, path: str, data: Dict[str, Any]) -> None:
        """Async put secret"""
        await self._run_sync(self._sync_client.put_secret, path, data)
    
    async def get_database_credentials(self, role: str = 'genesis-app') -> DatabaseCredentials:
        """Async get database credentials"""
        return await self._run_sync(self._sync_client.get_database_credentials, role)
    
    async def encrypt_data(self, plaintext: str, context: Optional[Dict[str, str]] = None) -> EncryptedData:
        """Async encrypt data"""
        return await self._run_sync(self._sync_client.encrypt_data, plaintext, context)
    
    async def decrypt_data(self, ciphertext: str, context: Optional[Dict[str, str]] = None) -> str:
        """Async decrypt data"""
        return await self._run_sync(self._sync_client.decrypt_data, ciphertext, context)
    
    async def health_check(self) -> Dict[str, Any]:
        """Async health check"""
        return await self._run_sync(self._sync_client.health_check)
    
    async def close(self) -> None:
        """Close async client"""
        if self._executor:
            self._executor.shutdown(wait=True)
        self._sync_client.close()


# Singleton instance for application use
_vault_client: Optional[VaultClient] = None


def get_vault_client(config: Optional[VaultConfig] = None) -> VaultClient:
    """Get or create singleton Vault client"""
    global _vault_client
    if _vault_client is None:
        _vault_client = VaultClient(config)
    return _vault_client


# Convenience functions
def get_secret(path: str, key: Optional[str] = None) -> Any:
    """Get secret using singleton client"""
    return get_vault_client().get_secret(path, key)


def encrypt(plaintext: str) -> str:
    """Encrypt data using singleton client"""
    return get_vault_client().encrypt_data(plaintext).ciphertext


def decrypt(ciphertext: str) -> str:
    """Decrypt data using singleton client"""
    return get_vault_client().decrypt_data(ciphertext)