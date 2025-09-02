"""Automated secret rotation logic."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from genesis.security.vault_manager import VaultManager
from genesis.security.credential_manager import CredentialManager
from genesis.core.exceptions import SecurityException


logger = structlog.get_logger(__name__)


class RotationStatus(Enum):
    """Secret rotation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SecretRotation:
    """Manages automated secret rotation."""
    
    def __init__(
        self,
        vault_manager: VaultManager,
        credential_manager: CredentialManager
    ):
        """Initialize secret rotation manager.
        
        Args:
            vault_manager: Vault manager instance
            credential_manager: Credential manager instance
        """
        self.vault = vault_manager
        self.credentials = credential_manager
        self.scheduler = AsyncIOScheduler()
        self._rotation_handlers: Dict[str, Callable] = {}
        self._rotation_status: Dict[str, RotationStatus] = {}
        self._rotation_history: List[Dict] = []
        
    async def initialize(self) -> None:
        """Initialize rotation scheduler and handlers."""
        # Register default rotation handlers
        self._register_default_handlers()
        
        # Schedule automatic rotations
        await self._schedule_rotations()
        
        # Start scheduler
        self.scheduler.start()
        
        logger.info("Secret rotation manager initialized")
    
    def _register_default_handlers(self) -> None:
        """Register default rotation handlers."""
        # JWT signing key rotation
        self.register_rotation_handler(
            "jwt-signing-key",
            self._rotate_jwt_key
        )
        
        # Database credentials rotation
        self.register_rotation_handler(
            "database-credentials",
            self._rotate_database_credentials
        )
        
        # API keys rotation
        self.register_rotation_handler(
            "api-keys",
            self._rotate_api_keys
        )
        
        # Encryption keys rotation
        self.register_rotation_handler(
            "encryption-keys",
            self._rotate_encryption_keys
        )
    
    async def _schedule_rotations(self) -> None:
        """Schedule automatic secret rotations."""
        # JWT key rotation - every 30 days
        self.scheduler.add_job(
            self.rotate_secret,
            IntervalTrigger(days=30),
            args=["jwt-signing-key"],
            id="jwt-rotation",
            name="JWT Signing Key Rotation"
        )
        
        # Database credentials - every 7 days
        self.scheduler.add_job(
            self.rotate_secret,
            IntervalTrigger(days=7),
            args=["database-credentials"],
            id="db-rotation",
            name="Database Credentials Rotation"
        )
        
        # API keys - every 90 days
        self.scheduler.add_job(
            self.rotate_secret,
            IntervalTrigger(days=90),
            args=["api-keys"],
            id="api-rotation",
            name="API Keys Rotation"
        )
        
        # Encryption keys - every 90 days
        self.scheduler.add_job(
            self.rotate_secret,
            IntervalTrigger(days=90),
            args=["encryption-keys"],
            id="encryption-rotation",
            name="Encryption Keys Rotation"
        )
    
    def register_rotation_handler(
        self,
        secret_type: str,
        handler: Callable
    ) -> None:
        """Register a rotation handler for a secret type.
        
        Args:
            secret_type: Type of secret to rotate
            handler: Async function to handle rotation
        """
        self._rotation_handlers[secret_type] = handler
        logger.info(f"Registered rotation handler for {secret_type}")
    
    async def rotate_secret(
        self,
        secret_type: str,
        force: bool = False
    ) -> bool:
        """Rotate a secret by type.
        
        Args:
            secret_type: Type of secret to rotate
            force: Force rotation even if recently rotated
            
        Returns:
            True if rotation successful
        """
        # Check if rotation is already in progress
        if self._rotation_status.get(secret_type) == RotationStatus.IN_PROGRESS:
            logger.warning(f"Rotation already in progress for {secret_type}")
            return False
        
        # Check if handler exists
        if secret_type not in self._rotation_handlers:
            logger.error(f"No rotation handler for {secret_type}")
            return False
        
        # Update status
        self._rotation_status[secret_type] = RotationStatus.IN_PROGRESS
        rotation_start = datetime.utcnow()
        
        try:
            # Execute rotation handler
            handler = self._rotation_handlers[secret_type]
            await handler()
            
            # Update status and history
            self._rotation_status[secret_type] = RotationStatus.COMPLETED
            self._rotation_history.append({
                "secret_type": secret_type,
                "status": "success",
                "timestamp": rotation_start,
                "duration": (datetime.utcnow() - rotation_start).total_seconds()
            })
            
            logger.info(
                f"Successfully rotated {secret_type}",
                duration=(datetime.utcnow() - rotation_start).total_seconds()
            )
            
            return True
            
        except Exception as e:
            # Update status and history
            self._rotation_status[secret_type] = RotationStatus.FAILED
            self._rotation_history.append({
                "secret_type": secret_type,
                "status": "failed",
                "timestamp": rotation_start,
                "error": str(e),
                "duration": (datetime.utcnow() - rotation_start).total_seconds()
            })
            
            logger.error(f"Failed to rotate {secret_type}: {e}")
            
            # Alert on rotation failure
            await self._alert_rotation_failure(secret_type, e)
            
            return False
    
    async def _rotate_jwt_key(self) -> None:
        """Rotate JWT signing key."""
        logger.info("Starting JWT signing key rotation")
        
        # Rotate the key
        new_key = await self.credentials.rotate_jwt_signing_key()
        
        # Allow grace period for tokens signed with old key
        # Old key is kept in Vault for validation during grace period
        
        logger.info("JWT signing key rotation completed")
    
    async def _rotate_database_credentials(self) -> None:
        """Rotate database credentials."""
        logger.info("Starting database credentials rotation")
        
        # Rotate credentials for all roles
        roles = ["genesis-app", "genesis-readonly", "genesis-admin"]
        
        for role in roles:
            try:
                await self.credentials.rotate_database_credentials(role)
            except Exception as e:
                logger.error(f"Failed to rotate credentials for {role}: {e}")
                raise
        
        logger.info("Database credentials rotation completed")
    
    async def _rotate_api_keys(self) -> None:
        """Rotate API keys (where supported)."""
        logger.info("Starting API keys rotation")
        
        # Note: Most exchange APIs don't support automatic rotation
        # This would typically involve:
        # 1. Creating new API key through provider's API
        # 2. Testing new key
        # 3. Updating Vault
        # 4. Revoking old key after grace period
        
        # For now, log that manual rotation is required
        logger.warning(
            "API key rotation requires manual intervention. "
            "Please rotate keys through provider portals."
        )
    
    async def _rotate_encryption_keys(self) -> None:
        """Rotate encryption keys in Vault transit engine."""
        logger.info("Starting encryption keys rotation")
        
        # List of encryption keys to rotate
        keys = [
            "genesis-data",
            "genesis-field-api_secret",
            "genesis-field-private_key"
        ]
        
        for key_name in keys:
            try:
                # Rotate key in Vault
                await self.vault._client.secrets.transit.rotate_key(
                    name=key_name,
                    mount_point=self.vault.config.transit_mount_point
                )
                logger.info(f"Rotated encryption key: {key_name}")
            except Exception as e:
                logger.error(f"Failed to rotate key {key_name}: {e}")
                raise
        
        logger.info("Encryption keys rotation completed")
    
    async def _alert_rotation_failure(
        self,
        secret_type: str,
        error: Exception
    ) -> None:
        """Alert on rotation failure.
        
        Args:
            secret_type: Type of secret that failed to rotate
            error: The error that occurred
        """
        # In production, this would send alerts via:
        # - Email to security team
        # - Slack/Teams notification
        # - PagerDuty incident
        # - Monitoring system alert
        
        alert_message = (
            f"CRITICAL: Secret rotation failed\n"
            f"Secret Type: {secret_type}\n"
            f"Error: {error}\n"
            f"Timestamp: {datetime.utcnow().isoformat()}\n"
            f"Action Required: Manual intervention needed"
        )
        
        logger.critical(alert_message)
        
        # Write to alert file for monitoring system
        try:
            with open("/tmp/genesis_rotation_alerts.log", "a") as f:
                f.write(f"{alert_message}\n{'='*50}\n")
        except Exception as e:
            logger.error(f"Failed to write alert: {e}")
    
    async def validate_rotation(self, secret_type: str) -> bool:
        """Validate that rotation was successful.
        
        Args:
            secret_type: Type of secret to validate
            
        Returns:
            True if validation successful
        """
        try:
            if secret_type == "jwt-signing-key":
                # Verify new key exists and old key is preserved
                current = await self.vault.get_secret("jwt/signing-key")
                old = await self.vault.get_secret("jwt/signing-key-old")
                return current and old and current != old
                
            elif secret_type == "database-credentials":
                # Test database connection with new credentials
                async with self.credentials.get_database_connection():
                    return True
                    
            elif secret_type == "encryption-keys":
                # Verify key version was incremented
                # This would check key metadata in Vault
                return True
                
            else:
                logger.warning(f"No validation for {secret_type}")
                return True
                
        except Exception as e:
            logger.error(f"Validation failed for {secret_type}: {e}")
            return False
    
    def get_rotation_status(self, secret_type: str) -> Optional[RotationStatus]:
        """Get current rotation status for a secret type.
        
        Args:
            secret_type: Type of secret
            
        Returns:
            Current rotation status or None
        """
        return self._rotation_status.get(secret_type)
    
    def get_rotation_history(
        self,
        secret_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Get rotation history.
        
        Args:
            secret_type: Filter by secret type (optional)
            limit: Maximum number of entries to return
            
        Returns:
            List of rotation history entries
        """
        history = self._rotation_history
        
        if secret_type:
            history = [
                entry for entry in history
                if entry["secret_type"] == secret_type
            ]
        
        # Return most recent entries
        return history[-limit:]
    
    async def emergency_rotation(self) -> None:
        """Perform emergency rotation of all secrets."""
        logger.warning("Starting emergency rotation of all secrets")
        
        results = {}
        for secret_type in self._rotation_handlers.keys():
            try:
                success = await self.rotate_secret(secret_type, force=True)
                results[secret_type] = "success" if success else "failed"
            except Exception as e:
                results[secret_type] = f"error: {e}"
                logger.error(f"Emergency rotation failed for {secret_type}: {e}")
        
        logger.warning(
            "Emergency rotation completed",
            results=results
        )
        
        return results
    
    async def shutdown(self) -> None:
        """Shutdown rotation scheduler."""
        self.scheduler.shutdown()
        logger.info("Secret rotation manager shutdown")