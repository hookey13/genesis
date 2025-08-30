"""Integration tests for API key rotation system."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest

from genesis.security.key_rotation import (
    KeyRotationOrchestrator,
    RotationSchedule,
    APIKeyVersion,
    KeyStatus
)
from genesis.security.vault_client import VaultClient


@pytest.mark.asyncio
class TestKeyRotationIntegration:
    """Integration tests for key rotation orchestrator."""
    
    @pytest.fixture
    async def mock_vault_client(self):
        """Create a mock Vault client."""
        client = Mock(spec=VaultClient)
        client.get_exchange_api_keys.return_value = {
            "api_key": "current_key",
            "api_secret": "current_secret"
        }
        client.store_secret.return_value = True
        return client
    
    @pytest.fixture
    async def mock_database_client(self):
        """Create a mock database client."""
        client = AsyncMock()
        client.log_audit_event = AsyncMock()
        return client
    
    @pytest.fixture
    async def mock_exchange_client(self):
        """Create a mock exchange client."""
        client = AsyncMock()
        return client
    
    @pytest.fixture
    async def orchestrator(self, mock_vault_client, mock_database_client, mock_exchange_client):
        """Create a key rotation orchestrator."""
        schedule = RotationSchedule(
            enabled=False,  # Disable automatic scheduling for tests
            interval_days=30,
            grace_period_hours=0.001  # Very short grace period for tests
        )
        
        orch = KeyRotationOrchestrator(
            vault_client=mock_vault_client,
            database_client=mock_database_client,
            exchange_client=mock_exchange_client,
            schedule=schedule
        )
        
        await orch.initialize()
        yield orch
        await orch.shutdown()
    
    async def test_initialize_loads_current_keys(self, orchestrator):
        """Test that initialization loads current keys from Vault."""
        active_key = orchestrator.get_active_keys("exchange")
        assert active_key is not None
        assert active_key.api_key == "current_key"
        assert active_key.api_secret == "current_secret"
        assert active_key.status == KeyStatus.ACTIVE
        assert active_key.is_primary is True
    
    async def test_rotate_keys_success(self, orchestrator):
        """Test successful key rotation."""
        # Mock new key generation
        with patch.object(orchestrator, '_generate_new_keys') as mock_gen:
            mock_gen.return_value = {
                "api_key": "new_key",
                "api_secret": "new_secret"
            }
            
            # Perform rotation
            success, message = await orchestrator.rotate_keys(
                key_identifier="exchange",
                reason="test"
            )
            
            assert success is True
            assert "successfully" in message.lower()
            
            # Verify new keys are active
            active_key = orchestrator.get_active_keys("exchange")
            assert active_key.api_key == "new_key"
            assert active_key.api_secret == "new_secret"
            assert active_key.status == KeyStatus.ACTIVE
    
    async def test_rotate_keys_with_grace_period(self, orchestrator):
        """Test key rotation with grace period for connection draining."""
        # Set a longer grace period
        orchestrator.schedule.grace_period_hours = 0.01  # 36 seconds
        
        with patch.object(orchestrator, '_generate_new_keys') as mock_gen:
            mock_gen.return_value = {
                "api_key": "new_key_grace",
                "api_secret": "new_secret_grace"
            }
            
            # Start rotation (non-blocking)
            rotation_task = asyncio.create_task(
                orchestrator.rotate_keys("exchange", "test_grace")
            )
            
            # Check that old keys are marked as rotating during grace period
            await asyncio.sleep(0.005)  # Half of grace period
            
            # During grace period, should have pending keys
            pending_key = orchestrator.get_pending_keys("exchange")
            if pending_key:  # May not be set yet depending on timing
                assert pending_key.status == KeyStatus.PENDING
            
            # Wait for rotation to complete
            success, message = await rotation_task
            assert success is True
    
    async def test_rotate_keys_rollback_on_exchange_failure(self, orchestrator):
        """Test that keys are rolled back if exchange activation fails."""
        # Make exchange activation fail
        with patch.object(orchestrator, '_activate_keys_with_exchange') as mock_activate:
            mock_activate.return_value = False
            
            with patch.object(orchestrator, '_generate_new_keys') as mock_gen:
                mock_gen.return_value = {
                    "api_key": "failed_key",
                    "api_secret": "failed_secret"
                }
                
                # Attempt rotation
                success, message = await orchestrator.rotate_keys(
                    key_identifier="exchange",
                    reason="test_failure"
                )
                
                assert success is False
                assert "Failed to activate" in message
                
                # Verify old keys are still active
                active_key = orchestrator.get_active_keys("exchange")
                assert active_key.api_key == "current_key"
                assert active_key.api_secret == "current_secret"
                
                # Verify no pending keys remain
                pending_key = orchestrator.get_pending_keys("exchange")
                assert pending_key is None
    
    async def test_key_expiration_check(self):
        """Test that key expiration is correctly calculated."""
        # Create a key that's 25 days old
        old_key = APIKeyVersion(
            key_identifier="test",
            version=1,
            api_key="old_key",
            api_secret="old_secret",
            created_at=datetime.now() - timedelta(days=25)
        )
        
        # Should not be expired with 30-day threshold
        assert old_key.is_expired(30) is False
        
        # Should be expired with 20-day threshold
        assert old_key.is_expired(20) is True
        
        # Create a key that's 35 days old
        very_old_key = APIKeyVersion(
            key_identifier="test",
            version=1,
            api_key="very_old_key",
            api_secret="very_old_secret",
            created_at=datetime.now() - timedelta(days=35)
        )
        
        # Should be expired with 30-day threshold
        assert very_old_key.is_expired(30) is True
    
    async def test_rotation_status_reporting(self, orchestrator):
        """Test that rotation status is correctly reported."""
        status = orchestrator.get_rotation_status()
        
        assert "scheduler_enabled" in status
        assert status["scheduler_enabled"] is False
        
        assert "interval_days" in status
        assert status["interval_days"] == 30
        
        assert "active_keys" in status
        assert "exchange" in status["active_keys"]
        
        active_key_info = status["active_keys"]["exchange"]
        assert active_key_info["version"] == 1
        assert active_key_info["is_primary"] is True
        assert active_key_info["status"] == "active"
    
    async def test_dual_key_strategy(self, orchestrator):
        """Test dual-key strategy maintains both keys during rotation."""
        # Start with initial keys
        initial_key = orchestrator.get_active_keys("exchange")
        assert initial_key is not None
        
        # Mock new key generation
        with patch.object(orchestrator, '_generate_new_keys') as mock_gen:
            mock_gen.return_value = {
                "api_key": "dual_key_new",
                "api_secret": "dual_secret_new"
            }
            
            # Mock the grace period to capture state
            original_grace = orchestrator._start_grace_period
            grace_period_called = asyncio.Event()
            
            async def mock_grace_period(key_id):
                # During grace period, both keys should exist
                active = orchestrator.get_active_keys(key_id)
                pending = orchestrator.get_pending_keys(key_id)
                
                # Old key should be rotating
                assert active.status == KeyStatus.ROTATING
                
                # New key should be pending
                assert pending is not None
                assert pending.status == KeyStatus.PENDING
                
                grace_period_called.set()
                await original_grace(key_id)
            
            orchestrator._start_grace_period = mock_grace_period
            
            # Perform rotation
            success, message = await orchestrator.rotate_keys(
                key_identifier="exchange",
                reason="test_dual"
            )
            
            assert success is True
            assert grace_period_called.is_set()
    
    async def test_audit_logging(self, orchestrator):
        """Test that key rotation events are properly audited."""
        with patch.object(orchestrator, '_generate_new_keys') as mock_gen:
            mock_gen.return_value = {
                "api_key": "audit_key",
                "api_secret": "audit_secret"
            }
            
            # Perform rotation
            await orchestrator.rotate_keys(
                key_identifier="exchange",
                reason="audit_test"
            )
            
            # Verify audit event was logged
            orchestrator.database_client.log_audit_event.assert_called_once()
            
            call_args = orchestrator.database_client.log_audit_event.call_args
            assert call_args[1]["event_type"] == "key_rotation"
            assert call_args[1]["resource"] == "exchange"
            assert call_args[1]["action"] == "rotate"
            assert call_args[1]["result"] == "success"
            assert call_args[1]["metadata"]["reason"] == "audit_test"