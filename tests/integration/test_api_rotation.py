"""
Integration tests for API key rotation system.
Tests dual-key strategy, verification, and automatic rotation.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

from genesis.security.secrets_manager import SecretsManager, SecretBackend
from genesis.security.api_key_rotation import (
    APIKeyRotationManager,
    APIKeySet,
    RotationState,
    RotationSchedule
)
from genesis.core.exceptions import SecurityError


class TestAPIKeyRotation:
    """Test API key rotation with zero-downtime strategy."""
    
    @pytest.fixture
    async def secrets_manager(self):
        """Create SecretsManager with temporary storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecretsManager(
                backend=SecretBackend.LOCAL,
                config={"storage_path": Path(tmpdir)}
            )
            yield manager
    
    @pytest.fixture
    async def rotation_manager(self, secrets_manager):
        """Create APIKeyRotationManager instance."""
        manager = APIKeyRotationManager(
            secrets_manager=secrets_manager,
            exchange_api=None,
            verification_callback=None
        )
        yield manager
    
    @pytest.fixture
    async def mock_exchange_api(self):
        """Create mock exchange API."""
        api = AsyncMock()
        api.create_api_key = AsyncMock(return_value={
            "api_key": "new_api_key_123",
            "api_secret": "new_api_secret_456",
            "permissions": ["trade", "read"],
            "label": "genesis_test"
        })
        api.revoke_api_key = AsyncMock(return_value=True)
        api.get_account_info = AsyncMock(return_value={"account": "test"})
        return api
    
    @pytest.mark.asyncio
    async def test_initialize_with_no_keys(self, rotation_manager, secrets_manager):
        """Test initialization with no existing keys."""
        await rotation_manager.initialize()
        
        assert rotation_manager.primary_keys is None
        assert rotation_manager.secondary_keys is None
        assert rotation_manager.current_state == RotationState.IDLE
    
    @pytest.mark.asyncio
    async def test_initialize_with_existing_keys(self, rotation_manager, secrets_manager):
        """Test initialization with existing single key set."""
        # Store existing keys
        existing_keys = {
            "api_key": "existing_key",
            "api_secret": "existing_secret",
            "permissions": ["trade", "read"]
        }
        await secrets_manager.put_secret("/genesis/exchange/api-keys", existing_keys)
        
        # Initialize
        await rotation_manager.initialize()
        
        assert rotation_manager.primary_keys is not None
        assert rotation_manager.primary_keys.api_key == "existing_key"
        assert rotation_manager.secondary_keys is None
    
    @pytest.mark.asyncio
    async def test_initialize_with_dual_keys(self, rotation_manager, secrets_manager):
        """Test initialization with dual keys from incomplete rotation."""
        # Store dual keys
        dual_keys = {
            "primary": {
                "api_key": "primary_key",
                "api_secret": "primary_secret",
                "created_at": datetime.utcnow().isoformat(),
                "permissions": ["trade"]
            },
            "secondary": {
                "api_key": "secondary_key",
                "api_secret": "secondary_secret",
                "created_at": datetime.utcnow().isoformat(),
                "permissions": ["trade"]
            },
            "grace_period_end": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        await secrets_manager.put_secret("/genesis/exchange/api-keys", dual_keys)
        
        # Initialize
        await rotation_manager.initialize()
        
        assert rotation_manager.primary_keys.api_key == "primary_key"
        assert rotation_manager.secondary_keys.api_key == "secondary_key"
    
    @pytest.mark.asyncio
    async def test_rotate_keys_success(self, rotation_manager, secrets_manager):
        """Test successful key rotation."""
        # Setup initial keys
        initial_keys = APIKeySet(
            api_key="old_key",
            api_secret="old_secret",
            created_at=datetime.utcnow(),
            permissions=["trade"]
        )
        rotation_manager.primary_keys = initial_keys
        
        # Perform rotation
        result = await rotation_manager.rotate_keys(
            grace_period=timedelta(seconds=1)
        )
        
        assert result["status"] == "success"
        assert "rotation_id" in result
        assert "new_key_id" in result
        
        # Verify dual keys stored
        stored = await secrets_manager.get_secret("/genesis/exchange/api-keys")
        assert "primary" in stored
        assert "secondary" in stored
        assert stored["secondary"]["api_key"] == "old_key"
    
    @pytest.mark.asyncio
    async def test_rotate_keys_with_verification(self, rotation_manager, secrets_manager):
        """Test key rotation with verification callback."""
        # Setup verification callback
        verification_called = False
        
        async def verify_keys(keys: APIKeySet) -> bool:
            nonlocal verification_called
            verification_called = True
            return keys.api_key.startswith("new_key_")
        
        rotation_manager.verification_callback = verify_keys
        
        # Perform rotation
        result = await rotation_manager.rotate_keys()
        
        assert result["status"] == "success"
        assert verification_called
        assert result["verification"] == "passed"
    
    @pytest.mark.asyncio
    async def test_rotate_keys_verification_failure(self, rotation_manager, secrets_manager):
        """Test rotation rollback on verification failure."""
        # Setup initial keys
        initial_keys = APIKeySet(
            api_key="old_key",
            api_secret="old_secret",
            created_at=datetime.utcnow(),
            permissions=["trade"]
        )
        rotation_manager.primary_keys = initial_keys
        
        # Setup failing verification
        async def verify_keys(keys: APIKeySet) -> bool:
            return False
        
        rotation_manager.verification_callback = verify_keys
        
        # Attempt rotation
        with pytest.raises(SecurityError, match="verification failed"):
            await rotation_manager.rotate_keys()
        
        # Verify rollback
        assert rotation_manager.primary_keys.api_key == "old_key"
        assert rotation_manager.secondary_keys is None
    
    @pytest.mark.asyncio
    async def test_rotate_keys_with_exchange_api(
        self,
        rotation_manager,
        secrets_manager,
        mock_exchange_api
    ):
        """Test rotation using real exchange API."""
        rotation_manager.exchange_api = mock_exchange_api
        
        # Perform rotation
        result = await rotation_manager.rotate_keys()
        
        assert result["status"] == "success"
        
        # Verify API was called
        mock_exchange_api.create_api_key.assert_called_once()
        
        # Verify new keys from API
        assert rotation_manager.primary_keys.api_key == "new_api_key_123"
    
    @pytest.mark.asyncio
    async def test_grace_period_expiration(self, rotation_manager, secrets_manager):
        """Test automatic revocation after grace period."""
        # Setup with exchange API
        mock_api = AsyncMock()
        mock_api.revoke_api_key = AsyncMock(return_value=True)
        rotation_manager.exchange_api = mock_api
        
        # Setup initial keys
        initial_keys = APIKeySet(
            api_key="old_key",
            api_secret="old_secret",
            created_at=datetime.utcnow(),
            permissions=["trade"]
        )
        rotation_manager.primary_keys = initial_keys
        
        # Rotate with short grace period
        await rotation_manager.rotate_keys(grace_period=timedelta(seconds=0.1))
        
        # Wait for grace period to expire
        await asyncio.sleep(0.2)
        
        # Verify old key was revoked
        mock_api.revoke_api_key.assert_called_with(api_key="old_key")
        
        # Verify secondary keys cleared
        assert rotation_manager.secondary_keys is None
    
    @pytest.mark.asyncio
    async def test_generate_temporary_key(self, rotation_manager, secrets_manager):
        """Test temporary key generation."""
        temp_key = await rotation_manager.generate_temporary_key(
            purpose="testing",
            ttl=timedelta(hours=1),
            permissions=["read"]
        )
        
        assert temp_key.api_key.startswith("temp_testing_")
        assert temp_key.expires_at is not None
        assert temp_key.permissions == ["read"]
        assert temp_key.metadata["temporary"] is True
        
        # Verify stored in secrets manager
        path = f"/genesis/temp_keys/{temp_key.api_key}"
        stored = await secrets_manager.get_secret(path)
        assert stored is not None
    
    @pytest.mark.asyncio
    async def test_temporary_key_cleanup(self, rotation_manager, secrets_manager):
        """Test automatic cleanup of expired temporary keys."""
        # Generate temp key with very short TTL
        temp_key = await rotation_manager.generate_temporary_key(
            purpose="cleanup_test",
            ttl=timedelta(seconds=0.1)
        )
        
        path = f"/genesis/temp_keys/{temp_key.api_key}"
        
        # Verify key exists
        assert await secrets_manager.get_secret(path) is not None
        
        # Wait for cleanup
        await asyncio.sleep(0.2)
        
        # Verify key was cleaned up
        assert await secrets_manager.get_secret(path) is None
    
    @pytest.mark.asyncio
    async def test_automatic_rotation_schedule(self, rotation_manager, secrets_manager):
        """Test automatic rotation scheduling."""
        # Configure automatic rotation
        await rotation_manager.configure_automatic_rotation(
            key_path="/genesis/exchange/api-keys",
            interval=timedelta(seconds=0.2),
            grace_period=timedelta(seconds=0.1)
        )
        
        # Verify schedule created
        schedule_key = "/genesis/exchange/api-keys_schedule"
        assert schedule_key in rotation_manager.schedules
        
        schedule = rotation_manager.schedules[schedule_key]
        assert schedule.enabled is True
        assert schedule.interval == timedelta(seconds=0.2)
    
    @pytest.mark.asyncio
    async def test_get_all_active_keys(self, rotation_manager):
        """Test getting all active keys including grace period."""
        # Setup dual keys
        primary = APIKeySet(
            api_key="primary",
            api_secret="primary_secret",
            created_at=datetime.utcnow()
        )
        secondary = APIKeySet(
            api_key="secondary",
            api_secret="secondary_secret",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=5)
        )
        
        rotation_manager.primary_keys = primary
        rotation_manager.secondary_keys = secondary
        
        active_keys = rotation_manager.get_all_active_keys()
        
        assert len(active_keys) == 2
        assert active_keys[0].api_key == "primary"
        assert active_keys[1].api_key == "secondary"
    
    @pytest.mark.asyncio
    async def test_get_all_active_keys_expired_secondary(self, rotation_manager):
        """Test that expired secondary keys are not returned."""
        # Setup dual keys with expired secondary
        primary = APIKeySet(
            api_key="primary",
            api_secret="primary_secret",
            created_at=datetime.utcnow()
        )
        secondary = APIKeySet(
            api_key="secondary",
            api_secret="secondary_secret",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() - timedelta(minutes=5)  # Expired
        )
        
        rotation_manager.primary_keys = primary
        rotation_manager.secondary_keys = secondary
        
        active_keys = rotation_manager.get_all_active_keys()
        
        assert len(active_keys) == 1
        assert active_keys[0].api_key == "primary"
    
    @pytest.mark.asyncio
    async def test_rotation_history(self, rotation_manager, secrets_manager):
        """Test rotation history tracking."""
        # Perform multiple rotations
        for i in range(3):
            rotation_manager.primary_keys = APIKeySet(
                api_key=f"key_{i}",
                api_secret=f"secret_{i}",
                created_at=datetime.utcnow()
            )
            await rotation_manager.rotate_keys()
        
        # Check history
        history = rotation_manager.get_rotation_history(limit=5)
        
        assert len(history) == 3
        assert all(h["status"] == "success" for h in history)
        assert all("rotation_id" in h for h in history)
    
    @pytest.mark.asyncio
    async def test_rotation_history_with_failures(self, rotation_manager, secrets_manager):
        """Test that failed rotations are tracked in history."""
        # Setup failing verification
        async def fail_verify(keys):
            return False
        
        rotation_manager.verification_callback = fail_verify
        rotation_manager.primary_keys = APIKeySet(
            api_key="test",
            api_secret="test",
            created_at=datetime.utcnow()
        )
        
        # Attempt rotation (will fail)
        try:
            await rotation_manager.rotate_keys()
        except SecurityError:
            pass
        
        # Check history
        history = rotation_manager.get_rotation_history()
        
        assert len(history) == 1
        assert history[0]["status"] == "failed"
        assert "error" in history[0]
    
    @pytest.mark.asyncio
    async def test_concurrent_rotation_prevention(self, rotation_manager, secrets_manager):
        """Test that concurrent rotations are prevented."""
        rotation_manager.primary_keys = APIKeySet(
            api_key="test",
            api_secret="test",
            created_at=datetime.utcnow()
        )
        
        # Start first rotation
        rotation1 = asyncio.create_task(
            rotation_manager.rotate_keys(grace_period=timedelta(seconds=1))
        )
        
        # Small delay to ensure first rotation starts
        await asyncio.sleep(0.01)
        
        # Attempt second rotation
        result2 = await rotation_manager.rotate_keys()
        
        # Second rotation should indicate in progress
        assert result2["status"] == "in_progress"
        assert result2["state"] != RotationState.IDLE.value
        
        # Wait for first rotation to complete
        result1 = await rotation1
        assert result1["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_get_status(self, rotation_manager):
        """Test status reporting."""
        rotation_manager.primary_keys = APIKeySet(
            api_key="test",
            api_secret="test",
            created_at=datetime.utcnow()
        )
        
        # Get initial status
        status = rotation_manager.get_status()
        
        assert status["state"] == RotationState.IDLE.value
        assert status["has_primary_keys"] is True
        assert status["has_secondary_keys"] is False
        assert status["rotation_history_count"] == 0
        
        # Perform rotation
        await rotation_manager.rotate_keys()
        
        # Get updated status
        status = rotation_manager.get_status()
        
        assert status["has_secondary_keys"] is True
        assert status["rotation_history_count"] == 1
        assert status["last_rotation"] is not None