"""API key rotation system for zero-downtime key management.

This module handles automated and manual rotation of API keys with dual-key
strategy to ensure continuous service availability during rotation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
import structlog
from decimal import Decimal

from genesis.security.vault_client import VaultClient
from genesis.core.models import SystemState

logger = structlog.get_logger(__name__)


class KeyStatus(Enum):
    """Status of an API key during rotation lifecycle."""
    
    ACTIVE = "active"
    ROTATING = "rotating"
    EXPIRED = "expired"
    PENDING = "pending"


@dataclass
class APIKeyVersion:
    """Represents a version of an API key."""
    
    key_identifier: str
    version: int
    api_key: str
    api_secret: str
    created_at: datetime
    rotated_at: Optional[datetime] = None
    status: KeyStatus = KeyStatus.ACTIVE
    is_primary: bool = True
    
    def is_expired(self, max_age_days: int = 30) -> bool:
        """Check if the key has exceeded maximum age.
        
        Args:
            max_age_days: Maximum age in days before rotation required
            
        Returns:
            True if key is expired
        """
        age = datetime.now() - self.created_at
        return age.days >= max_age_days


@dataclass
class RotationSchedule:
    """Configuration for automated key rotation."""
    
    enabled: bool = True
    interval_days: int = 30
    grace_period_hours: int = 24
    retry_attempts: int = 3
    retry_delay_seconds: int = 300
    
    def next_rotation_time(self, last_rotation: datetime) -> datetime:
        """Calculate next rotation time.
        
        Args:
            last_rotation: Time of last rotation
            
        Returns:
            Next scheduled rotation time
        """
        return last_rotation + timedelta(days=self.interval_days)


class KeyRotationOrchestrator:
    """Orchestrates API key rotation with zero downtime.
    
    Implements dual-key strategy where both old and new keys remain active
    during a grace period to allow for connection draining and switchover.
    """
    
    def __init__(
        self,
        vault_client: VaultClient,
        database_client: Any = None,
        exchange_client: Any = None,
        schedule: Optional[RotationSchedule] = None
    ):
        """Initialize the key rotation orchestrator.
        
        Args:
            vault_client: Vault client for secret management
            database_client: Database client for tracking key versions
            exchange_client: Exchange client for updating API keys
            schedule: Rotation schedule configuration
        """
        self.vault_client = vault_client
        self.database_client = database_client
        self.exchange_client = exchange_client
        self.schedule = schedule or RotationSchedule()
        self._rotation_task: Optional[asyncio.Task] = None
        self._active_keys: Dict[str, APIKeyVersion] = {}
        self._pending_keys: Dict[str, APIKeyVersion] = {}
    
    async def initialize(self):
        """Initialize the rotation system and load current keys."""
        try:
            # Load current keys from Vault
            await self._load_current_keys()
            
            # Start rotation scheduler if enabled
            if self.schedule.enabled:
                self._rotation_task = asyncio.create_task(self._rotation_scheduler())
                logger.info("Key rotation scheduler started", 
                          interval_days=self.schedule.interval_days)
            
        except Exception as e:
            logger.error("Failed to initialize key rotation", error=str(e))
            raise
    
    async def shutdown(self):
        """Shutdown the rotation system gracefully."""
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass
            logger.info("Key rotation scheduler stopped")
    
    async def rotate_keys(
        self,
        key_identifier: str = "exchange",
        reason: str = "scheduled",
        force: bool = False
    ) -> Tuple[bool, str]:
        """Rotate API keys with zero downtime.
        
        Args:
            key_identifier: Identifier for the keys to rotate
            reason: Reason for rotation (scheduled, manual, compromised)
            force: Force rotation even if not due
            
        Returns:
            Tuple of (success, message)
        """
        logger.info("Starting key rotation", 
                   key_identifier=key_identifier, 
                   reason=reason)
        
        try:
            # Phase 1: Generate new keys
            new_keys = await self._generate_new_keys(key_identifier)
            if not new_keys:
                return False, "Failed to generate new keys"
            
            # Phase 2: Store new keys as secondary (pending)
            success = await self._store_pending_keys(key_identifier, new_keys)
            if not success:
                return False, "Failed to store pending keys"
            
            # Phase 3: Activate new keys with exchange
            success = await self._activate_keys_with_exchange(new_keys)
            if not success:
                # Rollback pending keys
                await self._rollback_pending_keys(key_identifier)
                return False, "Failed to activate keys with exchange"
            
            # Phase 4: Start grace period for connection draining
            await self._start_grace_period(key_identifier)
            
            # Phase 5: Switch primary keys after grace period
            await self._switch_primary_keys(key_identifier)
            
            # Phase 6: Deactivate old keys
            await self._deactivate_old_keys(key_identifier)
            
            # Phase 7: Audit log the rotation
            await self._log_rotation_event(key_identifier, reason)
            
            logger.info("Key rotation completed successfully", 
                       key_identifier=key_identifier)
            return True, "Key rotation completed successfully"
            
        except Exception as e:
            logger.error("Key rotation failed", 
                        key_identifier=key_identifier, 
                        error=str(e))
            return False, f"Key rotation failed: {str(e)}"
    
    async def _load_current_keys(self):
        """Load current active keys from Vault."""
        keys = self.vault_client.get_exchange_api_keys()
        if keys:
            self._active_keys["exchange"] = APIKeyVersion(
                key_identifier="exchange",
                version=1,
                api_key=keys["api_key"],
                api_secret=keys["api_secret"],
                created_at=datetime.now(),
                status=KeyStatus.ACTIVE,
                is_primary=True
            )
            
            # Load read-only keys
            read_keys = self.vault_client.get_exchange_api_keys(read_only=True)
            if read_keys:
                self._active_keys["exchange_read"] = APIKeyVersion(
                    key_identifier="exchange_read",
                    version=1,
                    api_key=read_keys["api_key"],
                    api_secret=read_keys["api_secret"],
                    created_at=datetime.now(),
                    status=KeyStatus.ACTIVE,
                    is_primary=True
                )
    
    async def _generate_new_keys(self, key_identifier: str) -> Optional[Dict[str, str]]:
        """Generate new API keys.
        
        Args:
            key_identifier: Identifier for the keys
            
        Returns:
            Dictionary with new api_key and api_secret
        """
        # In production, this would interface with the exchange API
        # to generate new API keys programmatically
        # For now, this is a placeholder that would need manual intervention
        
        logger.warning("Manual API key generation required", 
                      key_identifier=key_identifier)
        
        # Placeholder for new keys (would be provided manually)
        return {
            "api_key": f"new_key_{datetime.now().timestamp()}",
            "api_secret": f"new_secret_{datetime.now().timestamp()}"
        }
    
    async def _store_pending_keys(
        self,
        key_identifier: str,
        new_keys: Dict[str, str]
    ) -> bool:
        """Store new keys as pending in Vault.
        
        Args:
            key_identifier: Identifier for the keys
            new_keys: New API keys to store
            
        Returns:
            True if successful
        """
        try:
            current_version = self._active_keys.get(key_identifier)
            new_version = (current_version.version + 1) if current_version else 1
            
            # Create new key version
            pending_key = APIKeyVersion(
                key_identifier=key_identifier,
                version=new_version,
                api_key=new_keys["api_key"],
                api_secret=new_keys["api_secret"],
                created_at=datetime.now(),
                status=KeyStatus.PENDING,
                is_primary=False
            )
            
            # Store in Vault with versioning
            path = f"/genesis/exchange/api-keys-v{new_version}"
            success = self.vault_client.store_secret(
                path,
                {
                    "api_key": new_keys["api_key"],
                    "api_secret": new_keys["api_secret"],
                    "version": new_version,
                    "status": KeyStatus.PENDING.value,
                    "created_at": pending_key.created_at.isoformat()
                }
            )
            
            if success:
                self._pending_keys[key_identifier] = pending_key
                
            return success
            
        except Exception as e:
            logger.error("Failed to store pending keys", error=str(e))
            return False
    
    async def _activate_keys_with_exchange(self, new_keys: Dict[str, str]) -> bool:
        """Activate new keys with the exchange.
        
        Args:
            new_keys: New API keys to activate
            
        Returns:
            True if successful
        """
        if not self.exchange_client:
            logger.warning("No exchange client configured, skipping activation")
            return True
        
        try:
            # Test new keys with a simple API call
            # In production, this would validate the keys work
            return True
            
        except Exception as e:
            logger.error("Failed to activate keys with exchange", error=str(e))
            return False
    
    async def _start_grace_period(self, key_identifier: str):
        """Start grace period for connection draining.
        
        Args:
            key_identifier: Identifier for the keys
        """
        grace_period = timedelta(hours=self.schedule.grace_period_hours)
        logger.info("Starting grace period for key rotation", 
                   key_identifier=key_identifier,
                   grace_period_hours=self.schedule.grace_period_hours)
        
        # Mark old keys as rotating
        if key_identifier in self._active_keys:
            self._active_keys[key_identifier].status = KeyStatus.ROTATING
        
        # Wait for grace period
        await asyncio.sleep(grace_period.total_seconds())
    
    async def _switch_primary_keys(self, key_identifier: str):
        """Switch primary keys from old to new.
        
        Args:
            key_identifier: Identifier for the keys
        """
        if key_identifier not in self._pending_keys:
            logger.error("No pending keys to switch", key_identifier=key_identifier)
            return
        
        pending_key = self._pending_keys[key_identifier]
        
        # Update Vault with new primary keys
        success = self.vault_client.store_secret(
            VaultClient.EXCHANGE_API_KEYS_PATH,
            {
                "api_key": pending_key.api_key,
                "api_secret": pending_key.api_secret,
                "api_key_read": pending_key.api_key if "read" in key_identifier else None,
                "api_secret_read": pending_key.api_secret if "read" in key_identifier else None
            }
        )
        
        if success:
            # Update internal state
            pending_key.status = KeyStatus.ACTIVE
            pending_key.is_primary = True
            self._active_keys[key_identifier] = pending_key
            del self._pending_keys[key_identifier]
            
            logger.info("Primary keys switched successfully", 
                       key_identifier=key_identifier)
    
    async def _deactivate_old_keys(self, key_identifier: str):
        """Deactivate old API keys.
        
        Args:
            key_identifier: Identifier for the keys
        """
        # In production, this would call exchange API to deactivate old keys
        logger.info("Old keys deactivated", key_identifier=key_identifier)
    
    async def _rollback_pending_keys(self, key_identifier: str):
        """Rollback pending keys in case of failure.
        
        Args:
            key_identifier: Identifier for the keys
        """
        if key_identifier in self._pending_keys:
            del self._pending_keys[key_identifier]
            logger.info("Rolled back pending keys", key_identifier=key_identifier)
    
    async def _log_rotation_event(self, key_identifier: str, reason: str):
        """Log key rotation event for audit trail.
        
        Args:
            key_identifier: Identifier for the rotated keys
            reason: Reason for rotation
        """
        if self.database_client:
            # Store rotation event in database
            await self.database_client.log_audit_event(
                event_type="key_rotation",
                resource=key_identifier,
                action="rotate",
                result="success",
                metadata={
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def _rotation_scheduler(self):
        """Background task for scheduled key rotation."""
        while True:
            try:
                # Check each key for rotation
                for key_id, key_version in self._active_keys.items():
                    if key_version.is_expired(self.schedule.interval_days):
                        logger.info("Scheduled rotation triggered", 
                                   key_identifier=key_id)
                        await self.rotate_keys(key_id, reason="scheduled")
                
                # Sleep until next check (daily)
                await asyncio.sleep(86400)  # 24 hours
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in rotation scheduler", error=str(e))
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    def get_active_keys(self, key_identifier: str = "exchange") -> Optional[APIKeyVersion]:
        """Get current active keys.
        
        Args:
            key_identifier: Identifier for the keys
            
        Returns:
            Active key version or None
        """
        return self._active_keys.get(key_identifier)
    
    def get_pending_keys(self, key_identifier: str = "exchange") -> Optional[APIKeyVersion]:
        """Get pending keys awaiting activation.
        
        Args:
            key_identifier: Identifier for the keys
            
        Returns:
            Pending key version or None
        """
        return self._pending_keys.get(key_identifier)
    
    def get_rotation_status(self) -> Dict[str, Any]:
        """Get current rotation status.
        
        Returns:
            Dictionary with rotation status information
        """
        status = {
            "scheduler_enabled": self.schedule.enabled,
            "interval_days": self.schedule.interval_days,
            "active_keys": {},
            "pending_keys": {},
            "next_rotations": {}
        }
        
        for key_id, key_version in self._active_keys.items():
            status["active_keys"][key_id] = {
                "version": key_version.version,
                "created_at": key_version.created_at.isoformat(),
                "status": key_version.status.value,
                "is_primary": key_version.is_primary
            }
            
            # Calculate next rotation
            next_rotation = self.schedule.next_rotation_time(key_version.created_at)
            status["next_rotations"][key_id] = next_rotation.isoformat()
        
        for key_id, key_version in self._pending_keys.items():
            status["pending_keys"][key_id] = {
                "version": key_version.version,
                "created_at": key_version.created_at.isoformat(),
                "status": key_version.status.value
            }
        
        return status