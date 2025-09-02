"""HashiCorp Vault integration for secret management."""

import asyncio
import json
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import hvac
import structlog
from hvac.exceptions import Forbidden, InvalidPath

from genesis.config.vault_config import VaultConfig
from genesis.core.exceptions import (
    ConfigurationException,
    SecurityException,
    ValidationException,
)

logger = structlog.get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class VaultManager:
    """Manages HashiCorp Vault operations with caching and retry logic."""

    def __init__(self, config: VaultConfig | None = None):
        """Initialize Vault manager.
        
        Args:
            config: Vault configuration, defaults to loading from environment
        """
        self.config = config or VaultConfig()
        self._client: hvac.Client | None = None
        self._cache: dict[str, dict[str, Any]] = {}
        self._circuit_breaker_state = CircuitBreakerState.CLOSED
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: datetime | None = None
        self._token_lease_duration: int | None = None
        self._token_renewed_at: datetime | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize Vault client and authenticate."""
        async with self._lock:
            try:
                # Create Vault client
                self._client = hvac.Client(
                    url=self.config.vault_url,
                    namespace=self.config.vault_namespace,
                    verify=self._get_verify_option()
                )

                # Authenticate based on method
                await self._authenticate()

                # Verify authentication
                if not self._client.is_authenticated():
                    raise SecurityException("Failed to authenticate with Vault")

                # Start token renewal task if using renewable token
                if self.config.vault_auth_method in ["token", "approle"]:
                    asyncio.create_task(self._token_renewal_loop())

                logger.info("Vault manager initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Vault manager: {e}")
                raise SecurityException(f"Vault initialization failed: {e}")

    def _get_verify_option(self) -> bool | str:
        """Get SSL verification option."""
        if not self.config.verify_ssl:
            return False
        if self.config.ca_cert_path:
            return self.config.ca_cert_path
        return True

    async def _authenticate(self) -> None:
        """Authenticate with Vault based on configured method."""
        if self.config.vault_auth_method == "token":
            if not self.config.vault_token:
                raise ConfigurationException("Vault token not configured")
            self._client.token = self.config.vault_token

        elif self.config.vault_auth_method == "approle":
            if not self.config.vault_role_id or not self.config.vault_secret_id:
                raise ConfigurationException("AppRole credentials not configured")

            response = await asyncio.to_thread(
                self._client.auth.approle.login,
                role_id=self.config.vault_role_id,
                secret_id=self.config.vault_secret_id
            )
            self._client.token = response["auth"]["client_token"]
            self._token_lease_duration = response["auth"]["lease_duration"]
            self._token_renewed_at = datetime.utcnow()

        else:
            raise ConfigurationException(
                f"Unsupported auth method: {self.config.vault_auth_method}"
            )

    async def _token_renewal_loop(self) -> None:
        """Background task to renew token before expiry."""
        while True:
            try:
                if self._should_renew_token():
                    await self._renew_token()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Token renewal error: {e}")
                await asyncio.sleep(10)  # Retry sooner on error

    def _should_renew_token(self) -> bool:
        """Check if token should be renewed."""
        if not self._token_lease_duration or not self._token_renewed_at:
            return False

        time_since_renewal = (datetime.utcnow() - self._token_renewed_at).total_seconds()
        time_until_expiry = self._token_lease_duration - time_since_renewal

        return time_until_expiry <= self.config.token_renewal_threshold

    async def _renew_token(self) -> None:
        """Renew the current token."""
        try:
            response = await asyncio.to_thread(self._client.auth.token.renew_self)
            self._token_lease_duration = response["auth"]["lease_duration"]
            self._token_renewed_at = datetime.utcnow()
            logger.info("Token renewed successfully")
        except Exception as e:
            logger.error(f"Failed to renew token: {e}")
            # Re-authenticate if renewal fails
            await self._authenticate()

    async def get_secret(self, path: str, key: str | None = None) -> Any:
        """Retrieve a secret from Vault with caching.
        
        Args:
            path: Secret path in Vault
            key: Specific key within the secret (optional)
            
        Returns:
            Secret value or entire secret dict
        """
        # Check circuit breaker
        if not self._is_circuit_closed():
            return await self._get_from_cache_or_fail(path, key)

        # Check cache first
        cached = self._get_cached_secret(path, key)
        if cached is not None:
            return cached

        # Retrieve from Vault with retry
        try:
            secret_data = await self._get_with_retry(path)

            # Cache the result
            self._cache_secret(path, secret_data)

            # Reset circuit breaker on success
            self._reset_circuit_breaker()

            # Return specific key or entire secret
            if key:
                return secret_data.get(key)
            return secret_data

        except Exception as e:
            self._record_circuit_failure()
            logger.error(f"Failed to retrieve secret {path}: {e}")

            # Try cache as fallback
            cached = self._get_cached_secret(path, key, ignore_ttl=True)
            if cached is not None:
                logger.warning(f"Using stale cached secret for {path}")
                return cached

            raise SecurityException(f"Failed to retrieve secret: {e}")

    def _is_circuit_closed(self) -> bool:
        """Check if circuit breaker allows requests."""
        if self._circuit_breaker_state == CircuitBreakerState.CLOSED:
            return True

        if self._circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if self._circuit_breaker_last_failure:
                time_since_failure = (
                    datetime.utcnow() - self._circuit_breaker_last_failure
                ).total_seconds()

                if time_since_failure >= self.config.circuit_breaker_timeout:
                    self._circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                    return True

            return False

        # HALF_OPEN state - allow one request
        return True

    def _record_circuit_failure(self) -> None:
        """Record a failure for circuit breaker."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.utcnow()

        if self._circuit_breaker_failures >= self.config.circuit_breaker_threshold:
            self._circuit_breaker_state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker opened due to repeated failures")

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker after successful request."""
        self._circuit_breaker_state = CircuitBreakerState.CLOSED
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None

    def _get_cached_secret(
        self,
        path: str,
        key: str | None = None,
        ignore_ttl: bool = False
    ) -> Any | None:
        """Get secret from cache if available and not expired."""
        if path not in self._cache:
            return None

        cached = self._cache[path]

        # Check TTL unless ignored
        if not ignore_ttl:
            if datetime.utcnow() > cached["expires_at"]:
                del self._cache[path]
                return None

        data = cached["data"]
        if key:
            return data.get(key)
        return data

    def _cache_secret(self, path: str, data: dict[str, Any]) -> None:
        """Cache a secret with TTL."""
        expires_at = datetime.utcnow() + timedelta(seconds=self.config.cache_ttl_seconds)
        self._cache[path] = {
            "data": data,
            "expires_at": expires_at
        }

    async def _get_with_retry(self, path: str) -> dict[str, Any]:
        """Get secret from Vault with exponential backoff retry."""
        last_exception = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries):
            try:
                # Read from KV v2 secrets engine
                response = await asyncio.to_thread(
                    self._client.secrets.kv.v2.read_secret_version,
                    path=path,
                    mount_point=self.config.kv_mount_point
                )

                return response["data"]["data"]

            except InvalidPath:
                # Secret doesn't exist - don't retry
                raise ValidationException(f"Secret not found: {path}")

            except Forbidden:
                # Permission denied - don't retry
                raise SecurityException(f"Access denied to secret: {path}")

            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff

        raise last_exception or SecurityException("Failed to retrieve secret")

    async def _get_from_cache_or_fail(
        self,
        path: str,
        key: str | None = None
    ) -> Any:
        """Get from cache when circuit breaker is open."""
        # Try break-glass cache if enabled
        if self.config.break_glass_enabled:
            cached = await self._load_break_glass_cache(path, key)
            if cached is not None:
                logger.warning(f"Using break-glass cache for {path}")
                return cached

        # Try memory cache (even if expired)
        cached = self._get_cached_secret(path, key, ignore_ttl=True)
        if cached is not None:
            logger.warning(f"Using expired cache for {path} (circuit open)")
            return cached

        raise SecurityException(
            "Vault unavailable and no cached secret available. "
            "Check break-glass procedures."
        )

    async def _load_break_glass_cache(
        self,
        path: str,
        key: str | None = None
    ) -> Any | None:
        """Load encrypted break-glass cache."""
        cache_file = self.config.break_glass_cache_file
        if not os.path.exists(cache_file):
            return None

        try:
            # In production, this would decrypt the cache file
            # For now, we'll use simple JSON
            with open(cache_file) as f:
                cache_data = json.load(f)

            if path in cache_data:
                data = cache_data[path]
                if key:
                    return data.get(key)
                return data

        except Exception as e:
            logger.error(f"Failed to load break-glass cache: {e}")

        return None

    async def save_break_glass_cache(self) -> None:
        """Save current cache to break-glass file."""
        if not self.config.break_glass_enabled:
            return

        try:
            # Extract non-expired secrets
            cache_data = {}
            for path, cached in self._cache.items():
                if datetime.utcnow() <= cached["expires_at"]:
                    cache_data[path] = cached["data"]

            # In production, this would encrypt the cache
            # For now, we'll use simple JSON
            with open(self.config.break_glass_cache_file, "w") as f:
                json.dump(cache_data, f, indent=2, default=str)

            logger.info("Break-glass cache updated")

        except Exception as e:
            logger.error(f"Failed to save break-glass cache: {e}")

    async def write_secret(
        self,
        path: str,
        data: dict[str, Any]
    ) -> None:
        """Write a secret to Vault.
        
        Args:
            path: Secret path in Vault
            data: Secret data to write
        """
        try:
            await asyncio.to_thread(
                self._client.secrets.kv.v2.create_or_update_secret,
                path=path,
                secret=data,
                mount_point=self.config.kv_mount_point
            )

            # Invalidate cache for this path
            if path in self._cache:
                del self._cache[path]

            logger.info(f"Secret written to {path}")

        except Exception as e:
            logger.error(f"Failed to write secret {path}: {e}")
            raise SecurityException(f"Failed to write secret: {e}")

    async def delete_secret(self, path: str) -> None:
        """Delete a secret from Vault.
        
        Args:
            path: Secret path to delete
        """
        try:
            await asyncio.to_thread(
                self._client.secrets.kv.v2.delete_metadata_and_all_versions,
                path=path,
                mount_point=self.config.kv_mount_point
            )

            # Remove from cache
            if path in self._cache:
                del self._cache[path]

            logger.info(f"Secret deleted: {path}")

        except Exception as e:
            logger.error(f"Failed to delete secret {path}: {e}")
            raise SecurityException(f"Failed to delete secret: {e}")

    async def list_secrets(self, path: str = "") -> list[str]:
        """List secrets at a given path.
        
        Args:
            path: Path to list secrets from
            
        Returns:
            List of secret paths
        """
        try:
            response = await asyncio.to_thread(
                self._client.secrets.kv.v2.list_secrets,
                path=path,
                mount_point=self.config.kv_mount_point
            )

            return response.get("data", {}).get("keys", [])

        except InvalidPath:
            return []
        except Exception as e:
            logger.error(f"Failed to list secrets at {path}: {e}")
            raise SecurityException(f"Failed to list secrets: {e}")

    async def close(self) -> None:
        """Close Vault connection and save break-glass cache."""
        await self.save_break_glass_cache()
        self._client = None
        logger.info("Vault manager closed")
