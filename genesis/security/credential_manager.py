"""Dynamic credential generation and management using Vault."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from contextlib import asynccontextmanager

import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from genesis.security.vault_manager import VaultManager
from genesis.core.exceptions import SecurityException, DatabaseException


logger = structlog.get_logger(__name__)


class CredentialManager:
    """Manages dynamic database credentials and API keys."""
    
    def __init__(self, vault_manager: VaultManager):
        """Initialize credential manager.
        
        Args:
            vault_manager: Initialized Vault manager instance
        """
        self.vault = vault_manager
        self._active_credentials: Dict[str, Dict] = {}
        self._credential_lock = asyncio.Lock()
        self._rotation_tasks: Dict[str, asyncio.Task] = {}
        
    async def get_database_credentials(
        self,
        role_name: str = "genesis-app"
    ) -> Tuple[str, str]:
        """Get dynamic database credentials from Vault.
        
        Args:
            role_name: Vault database role name
            
        Returns:
            Tuple of (username, password)
        """
        try:
            # Check for active credentials
            async with self._credential_lock:
                if role_name in self._active_credentials:
                    creds = self._active_credentials[role_name]
                    # Check if credentials are still valid
                    if datetime.utcnow() < creds["expires_at"]:
                        return creds["username"], creds["password"]
            
            # Request new credentials from Vault
            path = f"database/creds/{role_name}"
            response = await self.vault._get_with_retry(path)
            
            username = response["username"]
            password = response["password"]
            lease_duration = response.get("lease_duration", 3600)
            
            # Store active credentials
            async with self._credential_lock:
                self._active_credentials[role_name] = {
                    "username": username,
                    "password": password,
                    "expires_at": datetime.utcnow() + timedelta(seconds=lease_duration),
                    "lease_duration": lease_duration
                }
            
            # Schedule rotation before expiry
            await self._schedule_rotation(role_name, lease_duration)
            
            logger.info(
                f"Generated dynamic database credentials for role {role_name}",
                username=username,
                expires_in=lease_duration
            )
            
            return username, password
            
        except Exception as e:
            logger.error(f"Failed to get database credentials: {e}")
            # Fall back to static credentials if available
            return await self._get_static_db_credentials()
    
    async def _get_static_db_credentials(self) -> Tuple[str, str]:
        """Get static database credentials as fallback."""
        try:
            secret = await self.vault.get_secret("database/static-creds")
            return secret["username"], secret["password"]
        except Exception as e:
            raise SecurityException(
                f"Failed to get database credentials: {e}"
            )
    
    async def _schedule_rotation(
        self,
        role_name: str,
        lease_duration: int
    ) -> None:
        """Schedule credential rotation before expiry.
        
        Args:
            role_name: Role name for credentials
            lease_duration: Credential lease duration in seconds
        """
        # Cancel existing rotation task if any
        if role_name in self._rotation_tasks:
            self._rotation_tasks[role_name].cancel()
        
        # Rotate at 75% of lease duration
        rotation_delay = int(lease_duration * 0.75)
        
        async def rotate():
            await asyncio.sleep(rotation_delay)
            try:
                await self.rotate_database_credentials(role_name)
            except Exception as e:
                logger.error(f"Credential rotation failed: {e}")
        
        self._rotation_tasks[role_name] = asyncio.create_task(rotate())
    
    async def rotate_database_credentials(
        self,
        role_name: str = "genesis-app"
    ) -> None:
        """Rotate database credentials proactively.
        
        Args:
            role_name: Role name to rotate credentials for
        """
        logger.info(f"Starting credential rotation for {role_name}")
        
        try:
            # Get new credentials
            new_username, new_password = await self.get_database_credentials(role_name)
            
            # Update all active connections (handled by connection pool)
            # In production, this would trigger connection pool refresh
            
            logger.info(f"Successfully rotated credentials for {role_name}")
            
        except Exception as e:
            logger.error(f"Failed to rotate credentials: {e}")
            raise SecurityException(f"Credential rotation failed: {e}")
    
    async def get_api_key(self, service: str) -> str:
        """Get API key for external service.
        
        Args:
            service: Service name (e.g., "binance", "coingecko")
            
        Returns:
            API key string
        """
        try:
            path = f"api-keys/{service}"
            secret = await self.vault.get_secret(path)
            return secret.get("api_key") or secret.get("key")
        except Exception as e:
            logger.error(f"Failed to get API key for {service}: {e}")
            raise SecurityException(f"Failed to retrieve API key: {e}")
    
    async def get_api_credentials(
        self,
        service: str
    ) -> Tuple[str, str]:
        """Get API key and secret for external service.
        
        Args:
            service: Service name
            
        Returns:
            Tuple of (api_key, api_secret)
        """
        try:
            path = f"api-keys/{service}"
            secret = await self.vault.get_secret(path)
            
            api_key = secret.get("api_key") or secret.get("key")
            api_secret = secret.get("api_secret") or secret.get("secret")
            
            if not api_key or not api_secret:
                raise ValidationException(
                    f"Incomplete API credentials for {service}"
                )
            
            return api_key, api_secret
            
        except Exception as e:
            logger.error(f"Failed to get API credentials for {service}: {e}")
            raise SecurityException(f"Failed to retrieve API credentials: {e}")
    
    async def get_jwt_signing_key(self) -> str:
        """Get JWT signing key from Vault.
        
        Returns:
            JWT signing key
        """
        try:
            secret = await self.vault.get_secret("jwt/signing-key")
            return secret["key"]
        except Exception as e:
            logger.error(f"Failed to get JWT signing key: {e}")
            raise SecurityException("Failed to retrieve JWT signing key")
    
    async def rotate_jwt_signing_key(self) -> str:
        """Rotate JWT signing key.
        
        Returns:
            New JWT signing key
        """
        try:
            # Generate new key (in production, use cryptographically secure method)
            import secrets
            new_key = secrets.token_urlsafe(64)
            
            # Store old key for grace period
            current_key = await self.get_jwt_signing_key()
            await self.vault.write_secret(
                "jwt/signing-key-old",
                {"key": current_key, "rotated_at": datetime.utcnow().isoformat()}
            )
            
            # Store new key
            await self.vault.write_secret(
                "jwt/signing-key",
                {"key": new_key, "created_at": datetime.utcnow().isoformat()}
            )
            
            logger.info("JWT signing key rotated successfully")
            return new_key
            
        except Exception as e:
            logger.error(f"Failed to rotate JWT signing key: {e}")
            raise SecurityException("Failed to rotate JWT signing key")
    
    @asynccontextmanager
    async def get_database_connection(self, role_name: str = "genesis-app"):
        """Context manager for database connection with dynamic credentials.
        
        Args:
            role_name: Database role name
            
        Yields:
            AsyncEngine instance
        """
        username, password = await self.get_database_credentials(role_name)
        
        # Create connection string
        connection_string = (
            f"postgresql+asyncpg://{username}:{password}@"
            f"localhost:5432/genesis"
        )
        
        # Create engine
        engine = create_async_engine(
            connection_string,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
        
        try:
            yield engine
        finally:
            await engine.dispose()
    
    async def cleanup(self) -> None:
        """Clean up rotation tasks and active credentials."""
        # Cancel all rotation tasks
        for task in self._rotation_tasks.values():
            task.cancel()
        
        # Clear credentials
        self._active_credentials.clear()
        
        logger.info("Credential manager cleaned up")