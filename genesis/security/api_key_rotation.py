"""
API Key Rotation System with zero-downtime dual-key strategy.
Implements gradual transition, verification, and automatic revocation.
"""

import asyncio
import os
import hashlib
import hmac
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum

from genesis.security.secrets_manager import SecretsManager, SecretBackend
from genesis.core.exceptions import SecurityError, GenesisException

logger = structlog.get_logger(__name__)


class RotationState(Enum):
    """API key rotation states."""
    IDLE = "idle"
    PREPARING = "preparing"
    TRANSITIONING = "transitioning"
    VERIFYING = "verifying"
    COMPLETING = "completing"
    FAILED = "failed"


@dataclass
class APIKeySet:
    """Container for API key credentials."""
    api_key: str
    api_secret: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if the key set has expired."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "permissions": self.permissions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIKeySet":
        """Create from dictionary."""
        return cls(
            api_key=data["api_key"],
            api_secret=data["api_secret"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            permissions=data.get("permissions", []),
            metadata=data.get("metadata", {})
        )


@dataclass
class RotationSchedule:
    """Configuration for automatic key rotation."""
    interval: timedelta
    grace_period: timedelta = timedelta(minutes=5)
    max_retries: int = 3
    retry_delay: timedelta = timedelta(minutes=1)
    enabled: bool = True
    last_rotation: Optional[datetime] = None
    next_rotation: Optional[datetime] = None
    
    def is_due(self) -> bool:
        """Check if rotation is due."""
        if not self.enabled or not self.next_rotation:
            return False
        return datetime.utcnow() >= self.next_rotation
    
    def update_schedule(self):
        """Update rotation schedule after successful rotation."""
        self.last_rotation = datetime.utcnow()
        self.next_rotation = self.last_rotation + self.interval


class APIKeyRotationManager:
    """
    Manages API key rotation with zero-downtime strategy.
    Implements dual-key approach for seamless transitions.
    """
    
    def __init__(
        self,
        secrets_manager: SecretsManager,
        exchange_api: Optional[Any] = None,
        verification_callback: Optional[Callable] = None
    ):
        """
        Initialize the rotation manager.
        
        Args:
            secrets_manager: SecretsManager instance for key storage
            exchange_api: Exchange API client for key generation
            verification_callback: Callback to verify new keys work
        """
        self.secrets_manager = secrets_manager
        self.exchange_api = exchange_api
        self.verification_callback = verification_callback
        
        self.current_state = RotationState.IDLE
        self.primary_keys: Optional[APIKeySet] = None
        self.secondary_keys: Optional[APIKeySet] = None
        self.rotation_id: Optional[str] = None
        self.rotation_lock = asyncio.Lock()
        
        self.schedules: Dict[str, RotationSchedule] = {}
        self.rotation_history: List[Dict[str, Any]] = []
        
        self.logger = structlog.get_logger(__name__)
    
    async def initialize(self, key_path: str = "/genesis/exchange/api-keys"):
        """
        Initialize the manager with current keys.
        
        Args:
            key_path: Path to API keys in secrets manager
        """
        try:
            # Load current keys
            current_secret = await self.secrets_manager.get_secret(key_path)
            
            if current_secret:
                # Check if we have dual keys from incomplete rotation
                if "primary" in current_secret and "secondary" in current_secret:
                    self.primary_keys = APIKeySet.from_dict(current_secret["primary"])
                    self.secondary_keys = APIKeySet.from_dict(current_secret["secondary"])
                    
                    # Check if grace period has expired
                    grace_end = current_secret.get("grace_period_end")
                    if grace_end and datetime.fromisoformat(grace_end) < datetime.utcnow():
                        # Grace period expired, complete rotation
                        await self._complete_rotation(key_path)
                else:
                    # Single key set, use as primary
                    self.primary_keys = APIKeySet(
                        api_key=current_secret.get("api_key", ""),
                        api_secret=current_secret.get("api_secret", ""),
                        created_at=datetime.utcnow(),
                        permissions=current_secret.get("permissions", ["trade", "read"])
                    )
            
            self.logger.info("API key rotation manager initialized", has_keys=bool(self.primary_keys))
            
        except Exception as e:
            self.logger.error("Failed to initialize rotation manager", error=str(e))
            raise SecurityError(f"Rotation manager initialization failed: {str(e)}")
    
    async def rotate_keys(
        self,
        key_path: str = "/genesis/exchange/api-keys",
        grace_period: timedelta = timedelta(minutes=5),
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Perform API key rotation with dual-key strategy.
        
        Args:
            key_path: Path to store keys in secrets manager
            grace_period: Time to maintain both old and new keys
            force: Force rotation even if not scheduled
        
        Returns:
            Rotation result with status and new keys
        """
        async with self.rotation_lock:
            if self.current_state != RotationState.IDLE and not force:
                return {
                    "status": "in_progress",
                    "state": self.current_state.value,
                    "rotation_id": self.rotation_id
                }
            
            try:
                self.current_state = RotationState.PREPARING
                self.rotation_id = os.urandom(16).hex()
                
                self.logger.info(
                    "Starting API key rotation",
                    rotation_id=self.rotation_id,
                    grace_period=str(grace_period)
                )
                
                # Step 1: Generate new keys
                new_keys = await self._generate_new_keys()
                
                # Step 2: Transition to dual-key mode
                self.current_state = RotationState.TRANSITIONING
                await self._transition_to_dual_keys(key_path, new_keys, grace_period)
                
                # Step 3: Verify new keys work
                self.current_state = RotationState.VERIFYING
                verification_success = await self._verify_new_keys(new_keys)
                
                if not verification_success:
                    # Rollback if verification fails
                    await self._rollback_rotation(key_path)
                    raise SecurityError("New API keys verification failed")
                
                # Step 4: Schedule old key revocation
                asyncio.create_task(
                    self._schedule_revocation(key_path, grace_period)
                )
                
                # Update rotation history
                self.rotation_history.append({
                    "rotation_id": self.rotation_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "old_key_id": self._hash_key(self.primary_keys.api_key) if self.primary_keys else None,
                    "new_key_id": self._hash_key(new_keys.api_key),
                    "grace_period": str(grace_period),
                    "status": "success"
                })
                
                # Update schedule if configured
                schedule_key = f"{key_path}_schedule"
                if schedule_key in self.schedules:
                    self.schedules[schedule_key].update_schedule()
                
                self.current_state = RotationState.IDLE
                
                return {
                    "status": "success",
                    "rotation_id": self.rotation_id,
                    "new_key_id": self._hash_key(new_keys.api_key),
                    "grace_period_end": (datetime.utcnow() + grace_period).isoformat(),
                    "verification": "passed"
                }
                
            except Exception as e:
                self.current_state = RotationState.FAILED
                self.logger.error(
                    "API key rotation failed",
                    rotation_id=self.rotation_id,
                    error=str(e)
                )
                
                self.rotation_history.append({
                    "rotation_id": self.rotation_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": "failed",
                    "error": str(e)
                })
                
                raise
            finally:
                if self.current_state == RotationState.FAILED:
                    self.current_state = RotationState.IDLE
    
    async def _generate_new_keys(self) -> APIKeySet:
        """
        Generate new API keys.
        
        Returns:
            New APIKeySet
        """
        if self.exchange_api:
            # Use exchange API to generate real keys
            try:
                response = await self.exchange_api.create_api_key(
                    permissions=["trade", "read"],
                    label=f"genesis_rotation_{self.rotation_id}"
                )
                
                return APIKeySet(
                    api_key=response["api_key"],
                    api_secret=response["api_secret"],
                    created_at=datetime.utcnow(),
                    permissions=response.get("permissions", ["trade", "read"]),
                    metadata={
                        "rotation_id": self.rotation_id,
                        "label": response.get("label")
                    }
                )
            except Exception as e:
                self.logger.error("Failed to generate keys via API", error=str(e))
                raise SecurityError(f"Key generation failed: {str(e)}")
        else:
            # Generate mock keys for testing
            return APIKeySet(
                api_key=f"new_key_{os.urandom(16).hex()}",
                api_secret=f"new_secret_{os.urandom(32).hex()}",
                created_at=datetime.utcnow(),
                permissions=["trade", "read"],
                metadata={"rotation_id": self.rotation_id}
            )
    
    async def _transition_to_dual_keys(
        self,
        key_path: str,
        new_keys: APIKeySet,
        grace_period: timedelta
    ):
        """
        Transition to dual-key mode during rotation.
        
        Args:
            key_path: Storage path for keys
            new_keys: New APIKeySet to activate
            grace_period: Duration to maintain both keys
        """
        # Move current primary to secondary
        if self.primary_keys:
            self.secondary_keys = self.primary_keys
            self.secondary_keys.expires_at = datetime.utcnow() + grace_period
        
        # Set new keys as primary
        self.primary_keys = new_keys
        
        # Store dual keys in secrets manager
        dual_keys_secret = {
            "primary": self.primary_keys.to_dict(),
            "secondary": self.secondary_keys.to_dict() if self.secondary_keys else None,
            "grace_period_end": (datetime.utcnow() + grace_period).isoformat(),
            "rotation_id": self.rotation_id,
            "rotation_state": "dual_keys"
        }
        
        await self.secrets_manager.put_secret(
            key_path,
            dual_keys_secret,
            metadata={"action": "rotation_transition"}
        )
        
        self.logger.info(
            "Transitioned to dual-key mode",
            rotation_id=self.rotation_id,
            grace_period=str(grace_period)
        )
    
    async def _verify_new_keys(self, new_keys: APIKeySet) -> bool:
        """
        Verify that new API keys work correctly.
        
        Args:
            new_keys: Keys to verify
        
        Returns:
            True if verification successful
        """
        if self.verification_callback:
            try:
                result = await self.verification_callback(new_keys)
                self.logger.info(
                    "Key verification completed",
                    rotation_id=self.rotation_id,
                    success=result
                )
                return result
            except Exception as e:
                self.logger.error(
                    "Key verification failed",
                    rotation_id=self.rotation_id,
                    error=str(e)
                )
                return False
        else:
            # Default verification - check if we can authenticate
            if self.exchange_api:
                try:
                    # Try a simple authenticated request
                    await self.exchange_api.get_account_info(
                        api_key=new_keys.api_key,
                        api_secret=new_keys.api_secret
                    )
                    return True
                except Exception:
                    return False
            
            # No verification available, assume success
            self.logger.warning("No verification method available, assuming success")
            return True
    
    async def _schedule_revocation(self, key_path: str, grace_period: timedelta):
        """
        Schedule revocation of old keys after grace period.
        
        Args:
            key_path: Storage path for keys
            grace_period: Time to wait before revocation
        """
        await asyncio.sleep(grace_period.total_seconds())
        
        try:
            await self._complete_rotation(key_path)
        except Exception as e:
            self.logger.error(
                "Failed to complete rotation after grace period",
                rotation_id=self.rotation_id,
                error=str(e)
            )
    
    async def _complete_rotation(self, key_path: str):
        """
        Complete rotation by revoking old keys.
        
        Args:
            key_path: Storage path for keys
        """
        self.current_state = RotationState.COMPLETING
        
        try:
            # Revoke old keys if we have exchange API
            if self.secondary_keys and self.exchange_api:
                try:
                    await self.exchange_api.revoke_api_key(
                        api_key=self.secondary_keys.api_key
                    )
                    self.logger.info(
                        "Revoked old API key",
                        rotation_id=self.rotation_id,
                        key_id=self._hash_key(self.secondary_keys.api_key)
                    )
                except Exception as e:
                    self.logger.warning(
                        "Failed to revoke old key via API",
                        error=str(e)
                    )
            
            # Clear secondary keys
            self.secondary_keys = None
            
            # Update storage to single key set
            single_key_secret = self.primary_keys.to_dict()
            single_key_secret["rotation_completed"] = datetime.utcnow().isoformat()
            
            await self.secrets_manager.put_secret(
                key_path,
                single_key_secret,
                metadata={"action": "rotation_complete"}
            )
            
            self.logger.info(
                "API key rotation completed",
                rotation_id=self.rotation_id
            )
            
        finally:
            self.current_state = RotationState.IDLE
    
    async def _rollback_rotation(self, key_path: str):
        """
        Rollback failed rotation.
        
        Args:
            key_path: Storage path for keys
        """
        self.logger.warning(
            "Rolling back rotation",
            rotation_id=self.rotation_id
        )
        
        # Restore original keys
        if self.secondary_keys:
            self.primary_keys = self.secondary_keys
            self.secondary_keys = None
            
            await self.secrets_manager.put_secret(
                key_path,
                self.primary_keys.to_dict(),
                metadata={"action": "rotation_rollback"}
            )
    
    def _hash_key(self, api_key: str) -> str:
        """
        Generate a hash of API key for logging.
        
        Args:
            api_key: Key to hash
        
        Returns:
            Hashed key identifier
        """
        return hashlib.sha256(api_key.encode()).hexdigest()[:8]
    
    async def configure_automatic_rotation(
        self,
        key_path: str,
        interval: timedelta,
        grace_period: timedelta = timedelta(minutes=5)
    ):
        """
        Configure automatic key rotation schedule.
        
        Args:
            key_path: Path to keys to rotate
            interval: Rotation interval
            grace_period: Grace period for each rotation
        """
        schedule = RotationSchedule(
            interval=interval,
            grace_period=grace_period,
            enabled=True,
            next_rotation=datetime.utcnow() + interval
        )
        
        schedule_key = f"{key_path}_schedule"
        self.schedules[schedule_key] = schedule
        
        self.logger.info(
            "Configured automatic rotation",
            key_path=key_path,
            interval=str(interval),
            next_rotation=schedule.next_rotation.isoformat()
        )
        
        # Start rotation scheduler
        asyncio.create_task(self._rotation_scheduler(key_path, schedule_key))
    
    async def _rotation_scheduler(self, key_path: str, schedule_key: str):
        """
        Background task for automatic rotation.
        
        Args:
            key_path: Path to keys to rotate
            schedule_key: Key for schedule configuration
        """
        while schedule_key in self.schedules:
            schedule = self.schedules[schedule_key]
            
            if not schedule.enabled:
                await asyncio.sleep(60)  # Check every minute
                continue
            
            if schedule.is_due():
                try:
                    self.logger.info(
                        "Executing scheduled rotation",
                        key_path=key_path
                    )
                    
                    await self.rotate_keys(
                        key_path=key_path,
                        grace_period=schedule.grace_period
                    )
                    
                except Exception as e:
                    self.logger.error(
                        "Scheduled rotation failed",
                        key_path=key_path,
                        error=str(e)
                    )
            
            # Sleep until next check (1 minute)
            await asyncio.sleep(60)
    
    def get_current_keys(self) -> Optional[APIKeySet]:
        """
        Get current active API keys.
        
        Returns:
            Primary APIKeySet or None
        """
        return self.primary_keys
    
    def get_all_active_keys(self) -> List[APIKeySet]:
        """
        Get all currently active keys (including grace period).
        
        Returns:
            List of active APIKeySets
        """
        active = []
        
        if self.primary_keys:
            active.append(self.primary_keys)
        
        if self.secondary_keys and not self.secondary_keys.is_expired():
            active.append(self.secondary_keys)
        
        return active
    
    async def generate_temporary_key(
        self,
        purpose: str,
        ttl: timedelta = timedelta(hours=1),
        permissions: Optional[List[str]] = None
    ) -> APIKeySet:
        """
        Generate temporary API key for specific operations.
        
        Args:
            purpose: Purpose of temporary key
            ttl: Time to live
            permissions: Specific permissions for key
        
        Returns:
            Temporary APIKeySet
        """
        temp_key = APIKeySet(
            api_key=f"temp_{purpose}_{os.urandom(16).hex()}",
            api_secret=os.urandom(32).hex(),
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + ttl,
            permissions=permissions or ["read"],
            metadata={
                "purpose": purpose,
                "temporary": True
            }
        )
        
        # Store in secrets manager
        path = f"/genesis/temp_keys/{temp_key.api_key}"
        await self.secrets_manager.put_secret(
            path,
            temp_key.to_dict(),
            metadata={"purpose": purpose, "ttl": str(ttl)}
        )
        
        self.logger.info(
            "Generated temporary key",
            purpose=purpose,
            ttl=str(ttl),
            key_id=self._hash_key(temp_key.api_key)
        )
        
        # Schedule automatic cleanup
        asyncio.create_task(self._cleanup_temp_key(path, ttl))
        
        return temp_key
    
    async def _cleanup_temp_key(self, path: str, ttl: timedelta):
        """
        Clean up temporary key after TTL expires.
        
        Args:
            path: Storage path of temporary key
            ttl: Time to wait before cleanup
        """
        await asyncio.sleep(ttl.total_seconds())
        
        try:
            await self.secrets_manager.delete_secret(path)
            self.logger.info("Cleaned up expired temporary key", path=path)
        except Exception as e:
            self.logger.error("Failed to cleanup temporary key", path=path, error=str(e))
    
    def get_rotation_history(
        self,
        limit: int = 10,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get rotation history.
        
        Args:
            limit: Maximum number of entries
            since: Filter entries after this time
        
        Returns:
            List of rotation history entries
        """
        history = self.rotation_history
        
        if since:
            history = [
                h for h in history
                if datetime.fromisoformat(h["timestamp"]) > since
            ]
        
        return history[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current rotation manager status.
        
        Returns:
            Status dictionary
        """
        return {
            "state": self.current_state.value,
            "rotation_id": self.rotation_id,
            "has_primary_keys": bool(self.primary_keys),
            "has_secondary_keys": bool(self.secondary_keys),
            "active_schedules": list(self.schedules.keys()),
            "rotation_history_count": len(self.rotation_history),
            "last_rotation": self.rotation_history[-1] if self.rotation_history else None
        }