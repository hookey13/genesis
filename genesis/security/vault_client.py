"""HashiCorp Vault client for secure secrets management.

This module provides integration with HashiCorp Vault for production secrets
management with caching, TTL management, and fallback to environment variables
for local development.
"""

import os
import json
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import structlog
from decimal import Decimal

logger = structlog.get_logger(__name__)


@dataclass
class SecretCache:
    """Cache entry for a secret with TTL management."""
    
    value: Any
    fetched_at: datetime
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if the cached secret has expired."""
        elapsed = datetime.now() - self.fetched_at
        return elapsed.total_seconds() > self.ttl_seconds


class VaultClient:
    """Client for interacting with HashiCorp Vault or AWS Secrets Manager.
    
    Provides secure secret retrieval with caching, TTL management, and
    fallback to environment variables for local development.
    """
    
    DEFAULT_TTL = 3600  # 1 hour default TTL for cached secrets
    
    # Secret paths in Vault
    EXCHANGE_API_KEYS_PATH = "/genesis/exchange/api-keys"
    DATABASE_ENCRYPTION_KEY_PATH = "/genesis/database/encryption-key"
    TLS_CERTIFICATES_PATH = "/genesis/tls/certificates"
    BACKUP_ENCRYPTION_KEY_PATH = "/genesis/backup/encryption-key"
    
    def __init__(
        self,
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
        use_vault: bool = True,
        cache_ttl: int = DEFAULT_TTL
    ):
        """Initialize the Vault client.
        
        Args:
            vault_url: URL of the Vault server (e.g., https://vault.example.com:8200)
            vault_token: Authentication token for Vault
            use_vault: Whether to use Vault (False for local development)
            cache_ttl: Default TTL for cached secrets in seconds
        """
        self.vault_url = vault_url or os.environ.get("VAULT_URL")
        self.vault_token = vault_token or os.environ.get("VAULT_TOKEN")
        self.use_vault = use_vault and bool(self.vault_url and self.vault_token)
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, SecretCache] = {}
        
        if self.use_vault:
            try:
                import hvac
                self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
                if not self.client.is_authenticated():
                    logger.error("Failed to authenticate with Vault")
                    self.use_vault = False
                    self.client = None
                else:
                    logger.info("Successfully connected to Vault", vault_url=self.vault_url)
            except ImportError:
                logger.warning("hvac library not installed, falling back to environment variables")
                self.use_vault = False
                self.client = None
            except Exception as e:
                logger.error("Failed to connect to Vault", error=str(e))
                self.use_vault = False
                self.client = None
        else:
            logger.info("Using environment variables for secrets (development mode)")
            self.client = None
    
    def get_secret(
        self,
        path: str,
        key: Optional[str] = None,
        ttl: Optional[int] = None,
        force_refresh: bool = False
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve a secret from Vault or environment.
        
        Args:
            path: Path to the secret in Vault
            key: Specific key within the secret to retrieve
            ttl: Custom TTL for this secret (seconds)
            force_refresh: Force refresh from Vault, bypassing cache
            
        Returns:
            The secret value or None if not found
        """
        cache_key = f"{path}:{key}" if key else path
        
        # Check cache unless force refresh
        if not force_refresh and cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if not cache_entry.is_expired():
                logger.debug("Returning cached secret", path=path, key=key)
                return cache_entry.value
        
        # Fetch from Vault or environment
        if self.use_vault:
            secret_value = self._fetch_from_vault(path, key)
        else:
            secret_value = self._fetch_from_environment(path, key)
        
        # Cache the secret if found
        if secret_value is not None:
            ttl_seconds = ttl or self.cache_ttl
            self._cache[cache_key] = SecretCache(
                value=secret_value,
                fetched_at=datetime.now(),
                ttl_seconds=ttl_seconds
            )
            logger.debug("Cached secret", path=path, key=key, ttl=ttl_seconds)
        
        return secret_value
    
    def _fetch_from_vault(self, path: str, key: Optional[str] = None) -> Optional[Any]:
        """Fetch a secret from HashiCorp Vault.
        
        Args:
            path: Path to the secret in Vault
            key: Specific key within the secret
            
        Returns:
            The secret value or None if not found
        """
        if not self.client:
            return None
        
        try:
            # Remove leading slash if present for hvac compatibility
            vault_path = path.lstrip('/')
            
            # Read secret from Vault (KV v2 secrets engine)
            response = self.client.secrets.kv.v2.read_secret_version(
                path=vault_path,
                mount_point='secret'
            )
            
            if response and 'data' in response and 'data' in response['data']:
                secret_data = response['data']['data']
                
                if key:
                    return secret_data.get(key)
                return secret_data
            
            logger.warning("Secret not found in Vault", path=path, key=key)
            return None
            
        except Exception as e:
            logger.error("Failed to fetch secret from Vault", path=path, key=key, error=str(e))
            return None
    
    def _fetch_from_environment(self, path: str, key: Optional[str] = None) -> Optional[str]:
        """Fetch a secret from environment variables (fallback for development).
        
        Args:
            path: Path to map to environment variable
            key: Specific key within the secret
            
        Returns:
            The secret value or None if not found
        """
        # Map Vault paths to environment variables
        env_mapping = {
            self.EXCHANGE_API_KEYS_PATH: {
                "api_key": "BINANCE_API_KEY",
                "api_secret": "BINANCE_API_SECRET",
                "api_key_read": "BINANCE_API_KEY_READ",
                "api_secret_read": "BINANCE_API_SECRET_READ"
            },
            self.DATABASE_ENCRYPTION_KEY_PATH: {
                "key": "DATABASE_ENCRYPTION_KEY"
            },
            self.TLS_CERTIFICATES_PATH: {
                "cert": "TLS_CERT_PATH",
                "key": "TLS_KEY_PATH",
                "ca": "TLS_CA_PATH"
            },
            self.BACKUP_ENCRYPTION_KEY_PATH: {
                "key": "BACKUP_ENCRYPTION_KEY"
            }
        }
        
        if path in env_mapping:
            if key and key in env_mapping[path]:
                env_var = env_mapping[path][key]
                value = os.environ.get(env_var)
                if value:
                    logger.debug("Retrieved secret from environment", env_var=env_var)
                return value
            elif not key:
                # Return all keys for this path
                result = {}
                for k, env_var in env_mapping[path].items():
                    value = os.environ.get(env_var)
                    if value:
                        result[k] = value
                return result if result else None
        
        logger.warning("Secret not found in environment", path=path, key=key)
        return None
    
    def get_exchange_api_keys(self, read_only: bool = False) -> Optional[Dict[str, str]]:
        """Get exchange API keys from Vault.
        
        Args:
            read_only: Whether to get read-only keys
            
        Returns:
            Dictionary with 'api_key' and 'api_secret' or None
        """
        keys = self.get_secret(self.EXCHANGE_API_KEYS_PATH)
        if not keys:
            return None
        
        if read_only:
            return {
                "api_key": keys.get("api_key_read"),
                "api_secret": keys.get("api_secret_read")
            }
        else:
            return {
                "api_key": keys.get("api_key"),
                "api_secret": keys.get("api_secret")
            }
    
    def get_database_encryption_key(self) -> Optional[str]:
        """Get database encryption key from Vault.
        
        Returns:
            The encryption key or None
        """
        return self.get_secret(self.DATABASE_ENCRYPTION_KEY_PATH, key="key")
    
    def get_backup_encryption_key(self) -> Optional[str]:
        """Get backup encryption key from Vault.
        
        Returns:
            The encryption key or None
        """
        return self.get_secret(self.BACKUP_ENCRYPTION_KEY_PATH, key="key")
    
    def get_tls_certificates(self) -> Optional[Dict[str, str]]:
        """Get TLS certificates from Vault.
        
        Returns:
            Dictionary with 'cert', 'key', and 'ca' paths or None
        """
        return self.get_secret(self.TLS_CERTIFICATES_PATH)
    
    def store_secret(
        self,
        path: str,
        data: Dict[str, Any],
        cas: Optional[int] = None
    ) -> bool:
        """Store a secret in Vault (admin operation).
        
        Args:
            path: Path to store the secret
            data: Secret data to store
            cas: Check-and-set version for concurrent updates
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_vault or not self.client:
            logger.error("Cannot store secrets without Vault connection")
            return False
        
        try:
            vault_path = path.lstrip('/')
            
            # Write secret to Vault
            self.client.secrets.kv.v2.create_or_update_secret(
                path=vault_path,
                secret=data,
                cas=cas,
                mount_point='secret'
            )
            
            # Invalidate cache for this path
            cache_keys_to_remove = [k for k in self._cache if k.startswith(path)]
            for k in cache_keys_to_remove:
                del self._cache[k]
            
            logger.info("Successfully stored secret", path=path)
            return True
            
        except Exception as e:
            logger.error("Failed to store secret", path=path, error=str(e))
            return False
    
    def rotate_secret(
        self,
        path: str,
        key: str,
        new_value: str
    ) -> bool:
        """Rotate a specific secret value.
        
        Args:
            path: Path to the secret
            key: Key within the secret to rotate
            new_value: New value for the key
            
        Returns:
            True if successful, False otherwise
        """
        # Get current secret data
        current_data = self.get_secret(path, force_refresh=True)
        if not current_data:
            current_data = {}
        elif not isinstance(current_data, dict):
            logger.error("Cannot rotate non-dict secret", path=path)
            return False
        
        # Update the specific key
        current_data[key] = new_value
        
        # Store the updated secret
        return self.store_secret(path, current_data)
    
    def clear_cache(self, path: Optional[str] = None):
        """Clear cached secrets.
        
        Args:
            path: Specific path to clear, or None to clear all
        """
        if path:
            cache_keys_to_remove = [k for k in self._cache if k.startswith(path)]
            for k in cache_keys_to_remove:
                del self._cache[k]
            logger.debug("Cleared cache for path", path=path)
        else:
            self._cache.clear()
            logger.debug("Cleared entire secret cache")
    
    def is_connected(self) -> bool:
        """Check if connected to Vault.
        
        Returns:
            True if connected to Vault, False if using environment variables
        """
        return self.use_vault and self.client is not None
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the Vault connection.
        
        Returns:
            Health check results
        """
        if not self.use_vault:
            return {
                "status": "fallback",
                "mode": "environment_variables",
                "cache_size": len(self._cache)
            }
        
        if not self.client:
            return {
                "status": "error",
                "mode": "vault",
                "error": "No client connection"
            }
        
        try:
            is_authenticated = self.client.is_authenticated()
            sys_health = self.client.sys.read_health_status()
            
            return {
                "status": "healthy" if is_authenticated else "unhealthy",
                "mode": "vault",
                "authenticated": is_authenticated,
                "vault_status": sys_health,
                "cache_size": len(self._cache),
                "vault_url": self.vault_url
            }
        except Exception as e:
            return {
                "status": "error",
                "mode": "vault",
                "error": str(e),
                "cache_size": len(self._cache)
            }