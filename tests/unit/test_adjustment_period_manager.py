"""Unit tests for adjustment period manager."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from genesis.core.exceptions import StateError, ValidationError
from genesis.tilt.adjustment_period_manager import (
    AdjustmentPeriodManager,
    AdjustmentPhase,
    PeriodMetrics,
)


class TestAdjustmentPeriodManager:
    """Test suite for AdjustmentPeriodManager."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.query.return_value.filter_by.return_value.first.return_value = None
        session.commit = MagicMock()
        session.rollback = MagicMock()
        session.add = MagicMock()
        return session

    @pytest.fixture
    def manager(self, mock_session):
        """Create AdjustmentPeriodManager instance with mocked dependencies."""
        with patch(
            "genesis.tilt.adjustment_period_manager.get_session",
            return_value=mock_session,
        ):
            return AdjustmentPeriodManager(account_id="test-account-123")

    @pytest.mark.asyncio
    async def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.account_id == "test-account-123"
        assert manager.session is not None
        assert manager.monitoring_task is None

    @pytest.mark.asyncio
    async def test_start_adjustment_period(self, manager):
        """Test starting adjustment period."""
        period_id = await manager.start_adjustment_period(
            tier="HUNTER", duration_hours=48
        )

        assert period_id is not None
        assert isinstance(period_id, str)

        # Verify period was created
        status = await manager.get_adjustment_status()
        assert status is not None
        assert status.is_active
        assert status.current_phase == AdjustmentPhase.INITIAL
        assert status.position_limit_multiplier == Decimal("0.25")

    @pytest.mark.asyncio
    async def test_start_adjustment_period_already_active(self, manager):
        """Test starting period when one is already active."""
        # Start first period
        await manager.start_adjustment_period("HUNTER", 48)

        # Try to start second period
        with pytest.raises(StateError) as exc_info:
            await manager.start_adjustment_period("STRATEGIST", 48)

        assert "already active" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_phase_transitions(self, manager):
        """Test automatic phase transitions."""
        await manager.start_adjustment_period("HUNTER", 48)

        # Initial phase
        status = await manager.get_adjustment_status()
        assert status.current_phase == AdjustmentPhase.INITIAL
        assert status.position_limit_multiplier == Decimal("0.25")

        # Simulate time passing to early phase (after 12 hours)
        with patch("genesis.tilt.adjustment_period_manager.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(hours=13)
            await manager._check_phase_transition()

            status = await manager.get_adjustment_status()
            assert status.current_phase == AdjustmentPhase.EARLY
            assert status.position_limit_multiplier == Decimal("0.50")

        # Simulate time passing to mid phase (after 24 hours)
        with patch("genesis.tilt.adjustment_period_manager.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(hours=25)
            await manager._check_phase_transition()

            status = await manager.get_adjustment_status()
            assert status.current_phase == AdjustmentPhase.MID
            assert status.position_limit_multiplier == Decimal("0.75")

        # Simulate time passing to final phase (after 36 hours)
        with patch("genesis.tilt.adjustment_period_manager.datetime") as mock_dt:
            mock_dt.utcnow.return_value = datetime.utcnow() + timedelta(hours=37)
            await manager._check_phase_transition()

            status = await manager.get_adjustment_status()
            assert status.current_phase == AdjustmentPhase.FINAL
            assert status.position_limit_multiplier == Decimal("1.00")

    @pytest.mark.asyncio
    async def test_get_current_limits(self, manager):
        """Test retrieving current position limits."""
        initial_limit = Decimal("1000")

        await manager.start_adjustment_period("HUNTER", 48)

        # Initial phase should have 25% of limit
        current_limit = await manager.get_current_limits(initial_limit)
        assert current_limit == Decimal("250")

        # Test with different phases
        manager.adjustment_status.current_phase = AdjustmentPhase.MID
        current_limit = await manager.get_current_limits(initial_limit)
        assert current_limit == Decimal("750")

    @pytest.mark.asyncio
    async def test_record_intervention(self, manager):
        """Test recording interventions during adjustment."""
        await manager.start_adjustment_period("HUNTER", 48)

        await manager.record_intervention(
            "Position size violation", "Reduced position to 25% limit"
        )

        status = await manager.get_adjustment_status()
        assert status.interventions_triggered == 1

    @pytest.mark.asyncio
    async def test_complete_adjustment_period(self, manager):
        """Test completing adjustment period."""
        await manager.start_adjustment_period("HUNTER", 48)

        # Complete the period
        await manager.complete_adjustment_period()

        status = await manager.get_adjustment_status()
        assert not status.is_active

    @pytest.mark.asyncio
    async def test_complete_nonexistent_period(self, manager):
        """Test completing period that doesn't exist."""
        with pytest.raises(ValidationError) as exc_info:
            await manager.complete_adjustment_period()

        assert "No active adjustment period" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extend_adjustment_period(self, manager):
        """Test extending adjustment period."""
        await manager.start_adjustment_period("HUNTER", 48)

        original_status = await manager.get_adjustment_status()
        original_end = original_status.end_time

        # Extend by 24 hours
        await manager.extend_adjustment_period(24)

        extended_status = await manager.get_adjustment_status()
        assert extended_status.end_time > original_end

    @pytest.mark.asyncio
    async def test_monitoring_sensitivity(self, manager):
        """Test monitoring sensitivity multipliers by phase."""
        await manager.start_adjustment_period("HUNTER", 48)

        # Initial phase has 3x sensitivity
        status = await manager.get_adjustment_status()
        assert status.monitoring_sensitivity_multiplier == 3.0

        # Early phase has 2.5x
        manager.adjustment_status.current_phase = AdjustmentPhase.EARLY
        status = await manager.get_adjustment_status()
        assert status.monitoring_sensitivity_multiplier == 2.5

        # Mid phase has 2x
        manager.adjustment_status.current_phase = AdjustmentPhase.MID
        status = await manager.get_adjustment_status()
        assert status.monitoring_sensitivity_multiplier == 2.0

        # Final phase has 1.5x
        manager.adjustment_status.current_phase = AdjustmentPhase.FINAL
        status = await manager.get_adjustment_status()
        assert status.monitoring_sensitivity_multiplier == 1.5

    @pytest.mark.asyncio
    async def test_get_period_metrics(self, manager):
        """Test retrieving period metrics."""
        await manager.start_adjustment_period("HUNTER", 48)

        # Record some interventions
        await manager.record_intervention("Intervention 1", "Action 1")
        await manager.record_intervention("Intervention 2", "Action 2")

        metrics = await manager.get_period_metrics()

        assert isinstance(metrics, PeriodMetrics)
        assert metrics.total_duration_hours == 48
        assert metrics.interventions_count == 2
        assert metrics.current_phase == AdjustmentPhase.INITIAL

    @pytest.mark.asyncio
    async def test_database_persistence(self, manager, mock_session):
        """Test database persistence of adjustment period."""
        await manager.start_adjustment_period("HUNTER", 48)

        # Verify database methods were called
        assert mock_session.add.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_automatic_monitoring_start_stop(self, manager):
        """Test automatic monitoring task management."""
        # Start period should start monitoring
        await manager.start_adjustment_period("HUNTER", 48)
        assert manager.monitoring_task is not None

        # Complete period should stop monitoring
        await manager.complete_adjustment_period()
        # Task should be cancelled
        assert manager.monitoring_task.cancelled() or manager.monitoring_task.done()

    @pytest.mark.asyncio
    async def test_error_handling_in_monitoring(self, manager):
        """Test error handling in monitoring loop."""
        await manager.start_adjustment_period("HUNTER", 48)

        # Inject error into monitoring
        with patch.object(
            manager, "_check_phase_transition", side_effect=Exception("Test error")
        ):
            # Should not crash, just log error
            await asyncio.sleep(0.1)  # Let monitoring run

        # Manager should still be functional
        status = await manager.get_adjustment_status()
        assert status is not None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, manager):
        """Test handling concurrent operations."""
        await manager.start_adjustment_period("HUNTER", 48)

        # Concurrent interventions
        tasks = [
            manager.record_intervention(f"Intervention {i}", f"Action {i}")
            for i in range(5)
        ]

        await asyncio.gather(*tasks)

        status = await manager.get_adjustment_status()
        assert status.interventions_triggered == 5
