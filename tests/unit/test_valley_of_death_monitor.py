"""Unit tests for Valley of Death transition monitoring."""

import asyncio
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from genesis.tilt.valley_of_death_monitor import (
    TransitionMonitor,
    TransitionProximity,
)


class TestTransitionProximity:
    """Test TransitionProximity dataclass."""

    def test_is_critical_true(self):
        """Test critical proximity detection when >95%."""
        proximity = TransitionProximity(
            current_balance=Decimal('1950'),
            current_tier='SNIPER',
            next_tier='HUNTER',
            threshold=Decimal('2000'),
            distance_dollars=Decimal('50'),
            distance_percentage=Decimal('97.5'),
            monitoring_multiplier=3.0,
            is_approaching=True
        )

        assert proximity.is_critical is True

    def test_is_critical_false(self):
        """Test critical proximity detection when <95%."""
        proximity = TransitionProximity(
            current_balance=Decimal('1800'),
            current_tier='SNIPER',
            next_tier='HUNTER',
            threshold=Decimal('2000'),
            distance_dollars=Decimal('200'),
            distance_percentage=Decimal('90'),
            monitoring_multiplier=1.5,
            is_approaching=True
        )

        assert proximity.is_critical is False


class TestTransitionMonitor:
    """Test TransitionMonitor class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        return session

    @pytest.fixture
    def monitor(self, mock_session):
        """Create TransitionMonitor instance."""
        monitor = TransitionMonitor(session=mock_session)
        # Clear any events from previous tests
        monitor._approach_events = []
        return monitor

    def test_check_approaching_transition_not_approaching(self, monitor):
        """Test when balance is not approaching threshold."""
        proximity = monitor.check_approaching_transition(
            balance=Decimal('1000'),
            current_tier='SNIPER'
        )

        assert proximity.current_balance == Decimal('1000')
        assert proximity.current_tier == 'SNIPER'
        assert proximity.next_tier == 'HUNTER'
        assert proximity.threshold == Decimal('2000')
        assert proximity.distance_dollars == Decimal('1000')
        assert proximity.distance_percentage == Decimal('50')
        assert proximity.monitoring_multiplier == 1.0
        assert proximity.is_approaching is False

    def test_check_approaching_transition_90_percent(self, monitor):
        """Test when balance is at 90% of threshold."""
        proximity = monitor.check_approaching_transition(
            balance=Decimal('1800'),
            current_tier='SNIPER'
        )

        assert proximity.current_balance == Decimal('1800')
        assert proximity.distance_percentage == Decimal('90')
        assert proximity.monitoring_multiplier == 1.5
        assert proximity.is_approaching is True

    def test_check_approaching_transition_95_percent(self, monitor):
        """Test when balance is at 95% of threshold."""
        proximity = monitor.check_approaching_transition(
            balance=Decimal('1900'),
            current_tier='SNIPER'
        )

        assert proximity.distance_percentage == Decimal('95')
        assert proximity.monitoring_multiplier == 2.0
        assert proximity.is_approaching is True
        assert proximity.is_critical is True

    def test_check_approaching_transition_98_percent(self, monitor):
        """Test when balance is at 98% of threshold."""
        proximity = monitor.check_approaching_transition(
            balance=Decimal('1960'),
            current_tier='SNIPER'
        )

        assert proximity.distance_percentage == Decimal('98')
        assert proximity.monitoring_multiplier == 3.0
        assert proximity.is_approaching is True
        assert proximity.is_critical is True

    def test_check_approaching_transition_highest_tier(self, monitor):
        """Test when at highest tier (EMPEROR)."""
        proximity = monitor.check_approaching_transition(
            balance=Decimal('300000'),
            current_tier='EMPEROR'
        )

        assert proximity.next_tier == 'NONE'
        assert proximity.threshold == Decimal('0')
        assert proximity.is_approaching is False

    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitor):
        """Test starting account monitoring."""
        account_id = str(uuid.uuid4())

        # Mock the monitoring task
        with patch.object(monitor, '_monitor_account', new_callable=AsyncMock) as mock_monitor:
            await monitor.start_monitoring(account_id, check_interval_seconds=10)

            assert account_id in monitor._monitoring_tasks
            assert len(monitor._monitoring_tasks) == 1

            # Clean up
            await monitor.stop_monitoring(account_id)

    @pytest.mark.asyncio
    async def test_start_monitoring_already_active(self, monitor):
        """Test starting monitoring when already active."""
        account_id = str(uuid.uuid4())

        with patch.object(monitor, '_monitor_account', new_callable=AsyncMock):
            await monitor.start_monitoring(account_id)

            # Try to start again
            await monitor.start_monitoring(account_id)

            # Should still only have one task
            assert len(monitor._monitoring_tasks) == 1

            # Clean up
            await monitor.stop_monitoring(account_id)

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitor):
        """Test stopping account monitoring."""
        account_id = str(uuid.uuid4())

        with patch.object(monitor, '_monitor_account', new_callable=AsyncMock):
            await monitor.start_monitoring(account_id)
            assert account_id in monitor._monitoring_tasks

            await monitor.stop_monitoring(account_id)
            assert account_id not in monitor._monitoring_tasks

    @pytest.mark.asyncio
    async def test_stop_monitoring_not_active(self, monitor):
        """Test stopping monitoring when not active."""
        account_id = str(uuid.uuid4())

        # Should not raise error
        await monitor.stop_monitoring(account_id)
        assert account_id not in monitor._monitoring_tasks

    @pytest.mark.asyncio
    async def test_monitor_account_loop(self, monitor, mock_session):
        """Test the monitoring loop for an account."""
        account_id = str(uuid.uuid4())

        # Mock account
        mock_account = MagicMock()
        mock_account.balance_usdt = Decimal('1900')
        mock_account.current_tier = 'SNIPER'

        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_account

        # Mock handlers
        with patch.object(monitor, '_handle_approach_detected', new_callable=AsyncMock) as mock_handle:
            # Run monitoring for a short time
            task = asyncio.create_task(
                monitor._monitor_account(account_id, check_interval_seconds=0.1)
            )

            await asyncio.sleep(0.2)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Should have detected approach
            assert mock_handle.called

    @pytest.mark.asyncio
    async def test_handle_approach_detected_new(self, monitor, mock_session):
        """Test handling new approach detection."""
        account_id = str(uuid.uuid4())

        proximity = TransitionProximity(
            current_balance=Decimal('1900'),
            current_tier='SNIPER',
            next_tier='HUNTER',
            threshold=Decimal('2000'),
            distance_dollars=Decimal('100'),
            distance_percentage=Decimal('95'),
            monitoring_multiplier=2.0,
            is_approaching=True
        )

        with patch.object(monitor, '_create_transition_record', new_callable=AsyncMock) as mock_create:
            with patch.object(monitor, '_enhance_behavioral_monitoring', new_callable=AsyncMock) as mock_enhance:
                await monitor._handle_approach_detected(
                    account_id=account_id,
                    proximity=proximity,
                    last_proximity=None
                )

                mock_create.assert_called_once_with(account_id, proximity)
                mock_enhance.assert_called_once_with(account_id, proximity)

                # Check events were stored (both APPROACH_DETECTED and CRITICAL_PROXIMITY since 95% is critical)
                assert len(monitor._approach_events) == 2
                assert monitor._approach_events[0]['event_type'] == 'APPROACH_DETECTED'
                assert monitor._approach_events[1]['event_type'] == 'CRITICAL_PROXIMITY'

    @pytest.mark.asyncio
    async def test_handle_critical_proximity(self, monitor):
        """Test handling critical proximity."""
        account_id = str(uuid.uuid4())

        proximity = TransitionProximity(
            current_balance=Decimal('1980'),
            current_tier='SNIPER',
            next_tier='HUNTER',
            threshold=Decimal('2000'),
            distance_dollars=Decimal('20'),
            distance_percentage=Decimal('99'),
            monitoring_multiplier=3.0,
            is_approaching=True
        )

        await monitor._handle_critical_proximity(account_id, proximity)

        # Check critical event was stored
        events = [e for e in monitor._approach_events if e['event_type'] == 'CRITICAL_PROXIMITY']
        assert len(events) == 1
        assert events[0]['account_id'] == account_id

    @pytest.mark.asyncio
    async def test_create_transition_record(self, monitor, mock_session):
        """Test creating transition record in database."""
        account_id = str(uuid.uuid4())

        proximity = TransitionProximity(
            current_balance=Decimal('1900'),
            current_tier='SNIPER',
            next_tier='HUNTER',
            threshold=Decimal('2000'),
            distance_dollars=Decimal('100'),
            distance_percentage=Decimal('95'),
            monitoring_multiplier=2.0,
            is_approaching=True
        )

        # Mock no existing transition
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        await monitor._create_transition_record(account_id, proximity)

        # Should have added new transition
        assert mock_session.add.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_enhance_behavioral_monitoring(self, monitor, mock_session):
        """Test enhancing behavioral monitoring."""
        account_id = str(uuid.uuid4())

        proximity = TransitionProximity(
            current_balance=Decimal('1900'),
            current_tier='SNIPER',
            next_tier='HUNTER',
            threshold=Decimal('2000'),
            distance_dollars=Decimal('100'),
            distance_percentage=Decimal('95'),
            monitoring_multiplier=2.0,
            is_approaching=True
        )

        # Mock tilt profile - use a simple object instead of MagicMock for attribute setting
        class MockProfile:
            def __init__(self):
                self.monitoring_sensitivity = 1.0

        mock_profile = MockProfile()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_profile

        await monitor._enhance_behavioral_monitoring(account_id, proximity)

        # Should have updated sensitivity
        assert mock_profile.monitoring_sensitivity == 2.0
        assert mock_session.commit.called

    def test_get_next_tier_info_valid(self, monitor):
        """Test getting next tier info for valid tier."""
        next_tier, threshold = monitor._get_next_tier_info('SNIPER')
        assert next_tier == 'HUNTER'
        assert threshold == Decimal('2000')

        next_tier, threshold = monitor._get_next_tier_info('HUNTER')
        assert next_tier == 'STRATEGIST'
        assert threshold == Decimal('10000')

    def test_get_next_tier_info_highest(self, monitor):
        """Test getting next tier info for highest tier."""
        next_tier, threshold = monitor._get_next_tier_info('EMPEROR')
        assert next_tier is None
        assert threshold is None

    def test_get_next_tier_info_invalid(self, monitor):
        """Test getting next tier info for invalid tier."""
        next_tier, threshold = monitor._get_next_tier_info('INVALID')
        assert next_tier is None
        assert threshold is None

    def test_get_monitoring_stats(self, monitor):
        """Test getting monitoring statistics."""
        # Add some test data
        monitor._monitoring_tasks = {'acc1': Mock(), 'acc2': Mock()}
        monitor._approach_events = [
            {'account_id': 'acc1', 'event_type': 'APPROACH_DETECTED'},
            {'account_id': 'acc2', 'event_type': 'CRITICAL_PROXIMITY'}
        ]

        stats = monitor.get_monitoring_stats()

        assert stats['active_monitors'] == 2
        assert 'acc1' in stats['monitored_accounts']
        assert 'acc2' in stats['monitored_accounts']
        assert stats['approach_events'] == 2
        assert len(stats['recent_events']) == 2

    @pytest.mark.asyncio
    async def test_cleanup(self, monitor):
        """Test cleanup of monitoring tasks."""
        account_ids = [str(uuid.uuid4()) for _ in range(3)]

        with patch.object(monitor, '_monitor_account', new_callable=AsyncMock):
            # Start monitoring for multiple accounts
            for account_id in account_ids:
                await monitor.start_monitoring(account_id)

            assert len(monitor._monitoring_tasks) == 3

            # Cleanup
            await monitor.cleanup()

            assert len(monitor._monitoring_tasks) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
