"""HashiCorp Vault client for secure secrets management.

This module provides integration with HashiCorp Vault for production secrets
management with caching, TTL management, and fallback to environment variables
for local development. Enhanced to use the new SecretsManager infrastructure.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import asyncio

import structlog

from genesis.security.secrets_manager import SecretsManager, SecretBackend
from genesis.security.api_key_rotation import APIKeyRotationManager, APIKeySet

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
    """Enhanced Vault client using the new SecretsManager infrastructure.
    
    Provides secure secret retrieval with multi-backend support, runtime injection,
    and seamless integration with API key rotation system.
    """

    DEFAULT_TTL = 3600  # 1 hour default TTL for cached secrets

    # Secret paths in Vault
    EXCHANGE_API_KEYS_PATH = "/genesis/exchange/api-keys"
    DATABASE_ENCRYPTION_KEY_PATH = "/genesis/database/encryption-key"
    TLS_CERTIFICATES_PATH = "/genesis/tls/certificates"
    BACKUP_ENCRYPTION_KEY_PATH = "/genesis/backup/encryption-key"

    def __init__(
        self,
        vault_url: str | None = None,
        vault_token: str | None = None,
        use_vault: bool = True,
        cache_ttl: int = DEFAULT_TTL,
        backend_type: Optional[SecretBackend] = None,
        enable_rotation: bool = True
    ):
        """Initialize the enhanced Vault client with SecretsManager.
        
        Args:
            vault_url: URL of the Vault server (e.g., https://vault.example.com:8200)
            vault_token: Authentication token for Vault
            use_vault: Whether to use Vault (False for local development)
            cache_ttl: Default TTL for cached secrets in seconds
            backend_type: Override backend type (Vault, AWS, Local)
            enable_rotation: Enable API key rotation manager
        """
        self.vault_url = vault_url or os.environ.get("GENESIS_VAULT_URL")
        self.vault_token = vault_token or os.environ.get("GENESIS_VAULT_TOKEN")
        self.use_vault = use_vault and bool(self.vault_url and self.vault_token)
        self.cache_ttl = cache_ttl
        self._cache: dict[str, SecretCache] = {}
        
        # Determine backend type
        if backend_type:
            self.backend_type = backend_type
        elif self.use_vault:
            self.backend_type = SecretBackend.VAULT
        elif os.environ.get("AWS_REGION"):
            self.backend_type = SecretBackend.AWS
        else:
            self.backend_type = SecretBackend.LOCAL
        
        # Initialize SecretsManager with appropriate backend
        self._init_secrets_manager()
        
        # Initialize API key rotation manager if enabled
        self.rotation_manager = None
        if enable_rotation:
            self._init_rotation_manager()
        
        # Runtime secret injection storage
        self._runtime_secrets: Dict[str, Any] = {}
        
        logger.info(
            "Enhanced Vault client initialized",
            backend=self.backend_type.value,
            rotation_enabled=enable_rotation
        )
    
    def _init_secrets_manager(self):
        """Initialize the SecretsManager with appropriate backend."""
        config = {}
        
        if self.backend_type == SecretBackend.VAULT:
            config = {
                "vault_url": self.vault_url,
                "vault_token": self.vault_token,
                "mount_point": "secret"
            }
        elif self.backend_type == SecretBackend.AWS:
            config = {
                "aws_region": os.environ.get("AWS_REGION", "us-east-1")
            }
        
        self.secrets_manager = SecretsManager(
            backend=self.backend_type,
            config=config
        )
    
    def _init_rotation_manager(self):
        """Initialize the API key rotation manager."""
        try:
            self.rotation_manager = APIKeyRotationManager(
                secrets_manager=self.secrets_manager,
                exchange_api=None,  # Will be set by application
                verification_callback=None  # Will be set by application
            )
            
            # Initialize with existing keys
            asyncio.create_task(self.rotation_manager.initialize())
            
        except Exception as e:
            logger.error("Failed to initialize rotation manager", error=str(e))
            self.rotation_manager = None
    
    def inject_runtime_secret(self, path: str, value: Any):
        """
        Inject a secret at runtime without code changes.
        
        Args:
            path: Secret path
            value: Secret value to inject
        """
        self._runtime_secrets[path] = value
        logger.info("Runtime secret injected", path=path)

    async def get_secret_async(
        self,
        path: str,
        key: str | None = None,
        ttl: int | None = None,
        force_refresh: bool = False
    ) -> str | dict[str, Any] | None:
        """
        Async version using SecretsManager for enhanced functionality.
        
        Args:
            path: Path to the secret
            key: Specific key within the secret
            ttl: Custom TTL for caching
            force_refresh: Force refresh from backend
        
        Returns:
            Secret value or None
        """
        # Check runtime injected secrets first
        if path in self._runtime_secrets:
            runtime_value = self._runtime_secrets[path]
            if key and isinstance(runtime_value, dict):
                return runtime_value.get(key)
            return runtime_value
        
        # Use SecretsManager for retrieval
        secret = await self.secrets_manager.get_secret(
            path,
            use_cache=not force_refresh,
            fallback_to_env=True
        )
        
        if secret and key:
            return secret.get(key)
        
        return secret

    def get_secret(
        self,
        path: str,
        key: str | None = None,
        ttl: int | None = None,
        force_refresh: bool = False
    ) -> str | dict[str, Any] | None:
        """Enhanced secret retrieval with multi-backend support.
        
        Synchronous wrapper around async functionality for backward compatibility.
        
        Args:
            path: Path to the secret in Vault
            key: Specific key within the secret to retrieve
            ttl: Custom TTL for this secret (seconds)
            force_refresh: Force refresh from Vault, bypassing cache
            
        Returns:
            The secret value or None if not found
        """
        # Check runtime injected secrets first
        if path in self._runtime_secrets:
            runtime_value = self._runtime_secrets[path]
            if key and isinstance(runtime_value, dict):
                return runtime_value.get(key)
            return runtime_value
        
        # Try to get existing event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, create task
            future = asyncio.create_task(
                self.get_secret_async(path, key, ttl, force_refresh)
            )
            # Can't await in sync function, fallback to old method
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
            
        except RuntimeError:
            # No event loop, use sync version
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

    def _fetch_from_vault(self, path: str, key: str | None = None) -> Any | None:
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

    def _fetch_from_environment(self, path: str, key: str | None = None) -> str | None:
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

    def get_exchange_api_keys(self, read_only: bool = False) -> dict[str, str] | None:
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

    def get_database_encryption_key(self) -> str | None:
        """Get database encryption key from Vault.
        
        Returns:
            The encryption key or None
        """
        return self.get_secret(self.DATABASE_ENCRYPTION_KEY_PATH, key="key")

    def get_backup_encryption_key(self) -> str | None:
        """Get backup encryption key from Vault.
        
        Returns:
            The encryption key or None
        """
        return self.get_secret(self.BACKUP_ENCRYPTION_KEY_PATH, key="key")

    def get_tls_certificates(self) -> dict[str, str] | None:
        """Get TLS certificates from Vault.
        
        Returns:
            Dictionary with 'cert', 'key', and 'ca' paths or None
        """
        return self.get_secret(self.TLS_CERTIFICATES_PATH)

    def store_secret(
        self,
        path: str,
        data: dict[str, Any],
        cas: int | None = None
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

    def clear_cache(self, path: str | None = None):
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

    async def rotate_api_keys(
        self,
        grace_period: timedelta = timedelta(minutes=5)
    ) -> Dict[str, Any]:
        """
        Rotate API keys using the rotation manager.
        
        Args:
            grace_period: Time to maintain both old and new keys
        
        Returns:
            Rotation result
        """
        if not self.rotation_manager:
            return {
                "status": "error",
                "error": "Rotation manager not initialized"
            }
        
        return await self.rotation_manager.rotate_keys(
            key_path=self.EXCHANGE_API_KEYS_PATH,
            grace_period=grace_period
        )
    
    async def configure_automatic_rotation(
        self,
        interval: timedelta = timedelta(days=30),
        grace_period: timedelta = timedelta(minutes=5)
    ):
        """
        Configure automatic API key rotation.
        
        Args:
            interval: Rotation interval
            grace_period: Grace period for each rotation
        """
        if not self.rotation_manager:
            raise ValueError("Rotation manager not initialized")
        
        await self.rotation_manager.configure_automatic_rotation(
            key_path=self.EXCHANGE_API_KEYS_PATH,
            interval=interval,
            grace_period=grace_period
        )
        
        logger.info(
            "Configured automatic key rotation",
            interval=str(interval),
            grace_period=str(grace_period)
        )
    
    async def get_rotation_status(self) -> Dict[str, Any]:
        """Get current rotation status."""
        if not self.rotation_manager:
            return {"status": "disabled"}
        
        return self.rotation_manager.get_status()
    
    def health_check(self) -> dict[str, Any]:
        """Enhanced health check including SecretsManager status.
        
        Returns:
            Health check results
        """
        health = {
            "backend": self.backend_type.value,
            "cache_size": len(self._cache),
            "runtime_secrets_count": len(self._runtime_secrets),
            "rotation_enabled": bool(self.rotation_manager)
        }
        
        # Add SecretsManager health
        if hasattr(self, 'secrets_manager'):
            try:
                loop = asyncio.new_event_loop()
                sm_health = loop.run_until_complete(
                    self.secrets_manager.health_check()
                )
                health["secrets_manager"] = sm_health
            except Exception as e:
                health["secrets_manager_error"] = str(e)
        
        # Add rotation manager status
        if self.rotation_manager:
            health["rotation_status"] = self.rotation_manager.get_status()
        
        # Legacy Vault health check
        if self.use_vault and hasattr(self, 'client') and self.client:
            try:
                import hvac
                if isinstance(self.client, hvac.Client):
                    is_authenticated = self.client.is_authenticated()
                    health["vault_authenticated"] = is_authenticated
                    health["vault_url"] = self.vault_url
            except Exception as e:
                health["vault_error"] = str(e)
        
        return health
