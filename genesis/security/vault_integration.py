"""Vault integration for seamless secret management across the application."""

import os
from typing import Optional, Tuple
from contextlib import asynccontextmanager

import structlog
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from genesis.security.vault_manager import VaultManager
from genesis.security.credential_manager import CredentialManager
from genesis.security.envelope_encryption import EnvelopeEncryption
from genesis.security.secret_rotation import SecretRotation
from genesis.config.vault_config import VaultConfig
from genesis.core.exceptions import SecurityException


logger = structlog.get_logger(__name__)


class VaultIntegration:
    """Central integration point for Vault services."""
    
    _instance: Optional['VaultIntegration'] = None
    
    def __init__(self):
        """Initialize Vault integration components."""
        self.config = VaultConfig()
        self.vault_manager: Optional[VaultManager] = None
        self.credential_manager: Optional[CredentialManager] = None
        self.envelope_encryption: Optional[EnvelopeEncryption] = None
        self.secret_rotation: Optional[SecretRotation] = None
        self._initialized = False
        
    @classmethod
    async def get_instance(cls) -> 'VaultIntegration':
        """Get or create singleton instance.
        
        Returns:
            VaultIntegration singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
            await cls._instance.initialize()
        return cls._instance
    
    async def initialize(self) -> None:
        """Initialize all Vault components."""
        if self._initialized:
            return
            
        try:
            # Initialize Vault manager
            self.vault_manager = VaultManager(self.config)
            await self.vault_manager.initialize()
            
            # Initialize credential manager
            self.credential_manager = CredentialManager(self.vault_manager)
            
            # Initialize envelope encryption
            self.envelope_encryption = EnvelopeEncryption(self.vault_manager)
            
            # Initialize secret rotation
            self.secret_rotation = SecretRotation(
                self.vault_manager,
                self.credential_manager
            )
            await self.secret_rotation.initialize()
            
            self._initialized = True
            logger.info("Vault integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vault integration: {e}")
            raise SecurityException(f"Vault integration failed: {e}")
    
    async def get_database_url(self, role: str = "genesis-app") -> str:
        """Get database connection URL with dynamic credentials.
        
        Args:
            role: Database role name
            
        Returns:
            PostgreSQL connection URL
        """
        if not self.credential_manager:
            raise SecurityException("Vault not initialized")
            
        username, password = await self.credential_manager.get_database_credentials(role)
        
        # Get database connection details from Vault
        db_config = await self.vault_manager.get_secret("database/config")
        
        host = db_config.get("host", "localhost")
        port = db_config.get("port", 5432)
        database = db_config.get("database", "genesis_trading")
        
        return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    
    async def get_api_credentials(self, service: str) -> Tuple[str, str]:
        """Get API credentials for external service.
        
        Args:
            service: Service name (e.g., "binance")
            
        Returns:
            Tuple of (api_key, api_secret)
        """
        if not self.credential_manager:
            raise SecurityException("Vault not initialized")
            
        return await self.credential_manager.get_api_credentials(service)
    
    async def get_jwt_signing_key(self) -> str:
        """Get JWT signing key.
        
        Returns:
            JWT signing key
        """
        if not self.credential_manager:
            raise SecurityException("Vault not initialized")
            
        return await self.credential_manager.get_jwt_signing_key()
    
    async def encrypt_sensitive_data(self, data: any) -> dict:
        """Encrypt sensitive data using envelope encryption.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted envelope
        """
        if not self.envelope_encryption:
            raise SecurityException("Vault not initialized")
            
        return await self.envelope_encryption.encrypt_data(data)
    
    async def decrypt_sensitive_data(self, encrypted_envelope: dict) -> any:
        """Decrypt sensitive data.
        
        Args:
            encrypted_envelope: Encrypted data envelope
            
        Returns:
            Decrypted data
        """
        if not self.envelope_encryption:
            raise SecurityException("Vault not initialized")
            
        return await self.envelope_encryption.decrypt_data(encrypted_envelope)
    
    @asynccontextmanager
    async def get_database_engine(self, role: str = "genesis-app"):
        """Context manager for database engine with dynamic credentials.
        
        Args:
            role: Database role name
            
        Yields:
            AsyncEngine instance
        """
        connection_url = await self.get_database_url(role)
        
        engine = create_async_engine(
            connection_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
        
        try:
            yield engine
        finally:
            await engine.dispose()
    
    async def rotate_secrets(self, secret_type: str = None) -> bool:
        """Trigger secret rotation.
        
        Args:
            secret_type: Specific secret type to rotate (optional)
            
        Returns:
            True if rotation successful
        """
        if not self.secret_rotation:
            raise SecurityException("Vault not initialized")
            
        if secret_type:
            return await self.secret_rotation.rotate_secret(secret_type)
        else:
            # Rotate all secrets
            results = {}
            for secret_type in ["jwt-signing-key", "database-credentials", "api-keys"]:
                results[secret_type] = await self.secret_rotation.rotate_secret(secret_type)
            return all(results.values())
    
    async def emergency_rotation(self) -> dict:
        """Perform emergency rotation of all secrets.
        
        Returns:
            Rotation results
        """
        if not self.secret_rotation:
            raise SecurityException("Vault not initialized")
            
        return await self.secret_rotation.emergency_rotation()
    
    async def shutdown(self) -> None:
        """Shutdown Vault integration."""
        if self.secret_rotation:
            await self.secret_rotation.shutdown()
        
        if self.credential_manager:
            await self.credential_manager.cleanup()
        
        if self.vault_manager:
            await self.vault_manager.close()
        
        self._initialized = False
        logger.info("Vault integration shutdown complete")


# Convenience functions for easy access
async def get_vault() -> VaultIntegration:
    """Get Vault integration instance.
    
    Returns:
        VaultIntegration instance
    """
    return await VaultIntegration.get_instance()


async def get_database_url(role: str = "genesis-app") -> str:
    """Get database URL with dynamic credentials.
    
    Args:
        role: Database role
        
    Returns:
        Database connection URL
    """
    vault = await get_vault()
    return await vault.get_database_url(role)


async def get_api_credentials(service: str) -> Tuple[str, str]:
    """Get API credentials.
    
    Args:
        service: Service name
        
    Returns:
        Tuple of (api_key, api_secret)
    """
    vault = await get_vault()
    return await vault.get_api_credentials(service)


async def get_jwt_key() -> str:
    """Get JWT signing key.
    
    Returns:
        JWT signing key
    """
    vault = await get_vault()
    return await vault.get_jwt_signing_key()


async def encrypt_data(data: any) -> dict:
    """Encrypt sensitive data.
    
    Args:
        data: Data to encrypt
        
    Returns:
        Encrypted envelope
    """
    vault = await get_vault()
    return await vault.encrypt_sensitive_data(data)


async def decrypt_data(encrypted: dict) -> any:
    """Decrypt sensitive data.
    
    Args:
        encrypted: Encrypted envelope
        
    Returns:
        Decrypted data
    """
    vault = await get_vault()
    return await vault.decrypt_sensitive_data(encrypted)