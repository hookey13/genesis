"""Unit tests for forced break manager."""

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from genesis.core.events import EventType
from genesis.core.models import TradingTier, Account
from genesis.tilt.forced_break_manager import ForcedBreakManager


class TestForcedBreakManager(unittest.TestCase):
    """Test cases for ForcedBreakManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_repo = MagicMock()
        self.mock_event_bus = MagicMock()
        self.manager = ForcedBreakManager(
            repository=self.mock_repo, event_bus=self.mock_event_bus
        )
        self.account_id = "test_account_123"

    @patch("genesis.utils.decorators.get_current_tier")
    def test_check_consecutive_losses_with_profile(self, mock_get_tier):
        """Test checking consecutive losses when profile exists."""
        # Mock tier check to pass
        mock_get_tier.return_value = TradingTier.HUNTER

        # Setup mock
        self.mock_repo.get_tilt_profile.return_value = {
            "account_id": self.account_id,
            "consecutive_losses": 2,
        }

        # Execute
        result = self.manager.check_consecutive_losses(self.account_id)

        # Verify
        self.assertEqual(result, 2)
        self.mock_repo.get_tilt_profile.assert_called_once_with(self.account_id)
        self.assertEqual(self.manager._loss_counters[self.account_id], 2)

    @patch("genesis.utils.decorators.get_current_tier")
    def test_check_consecutive_losses_no_profile(self, mock_get_tier):
        """Test checking consecutive losses when no profile exists."""
        # Mock tier check to pass
        mock_get_tier.return_value = TradingTier.HUNTER

        # Setup mock
        self.mock_repo.get_tilt_profile.return_value = None

        # Execute
        result = self.manager.check_consecutive_losses(self.account_id)

        # Verify
        self.assertEqual(result, 0)
        self.mock_repo.get_tilt_profile.assert_called_once_with(self.account_id)

    @patch("genesis.utils.decorators.get_current_tier")
    def test_check_consecutive_losses_exception(self, mock_get_tier):
        """Test checking consecutive losses handles exceptions gracefully."""
        # Mock tier check to pass
        mock_get_tier.return_value = TradingTier.HUNTER

        # Setup mock to raise exception
        self.mock_repo.get_tilt_profile.side_effect = Exception("DB error")

        # Execute
        result = self.manager.check_consecutive_losses(self.account_id)

        # Verify
        self.assertEqual(result, 0)

    def test_record_trade_result_profitable(self):
        """Test recording a profitable trade resets loss counter."""
        # Setup initial losses
        self.manager._loss_counters[self.account_id] = 2

        # Execute
        result = self.manager.record_trade_result(self.account_id, is_profitable=True)

        # Verify
        self.assertIsNone(result)
        self.assertEqual(self.manager._loss_counters[self.account_id], 0)
        self.mock_repo.update_tilt_profile.assert_called_once_with(
            self.account_id, {"consecutive_losses": 0}
        )

    def test_record_trade_result_loss_no_break(self):
        """Test recording a loss that doesn't trigger break."""
        # Setup initial losses
        self.manager._loss_counters[self.account_id] = 1

        # Execute
        result = self.manager.record_trade_result(self.account_id, is_profitable=False)

        # Verify
        self.assertIsNone(result)
        self.assertEqual(self.manager._loss_counters[self.account_id], 2)
        self.mock_repo.update_tilt_profile.assert_called_once_with(
            self.account_id, {"consecutive_losses": 2}
        )

    @patch("genesis.tilt.forced_break_manager.datetime")
    def test_record_trade_result_loss_triggers_break(self, mock_datetime):
        """Test recording a loss that triggers forced break."""
        # Setup
        now = datetime(2025, 8, 26, 10, 0, 0, tzinfo=UTC)
        expected_expiration = now + timedelta(minutes=30)
        mock_datetime.now.return_value = now
        self.manager._loss_counters[self.account_id] = 2

        # Mock enforce_trading_break
        with patch.object(
            self.manager, "enforce_trading_break", return_value=expected_expiration
        ) as mock_enforce:
            # Execute
            result = self.manager.record_trade_result(
                self.account_id, is_profitable=False
            )

            # Verify
            self.assertEqual(result, expected_expiration)
            self.assertEqual(self.manager._loss_counters[self.account_id], 3)
            mock_enforce.assert_called_once_with(self.account_id)

    def test_record_trade_result_exception(self):
        """Test recording trade result handles exceptions."""
        # Setup mock to raise exception
        self.mock_repo.update_tilt_profile.side_effect = Exception("DB error")

        # Execute
        result = self.manager.record_trade_result(self.account_id, is_profitable=False)

        # Verify
        self.assertIsNone(result)

    @patch("genesis.tilt.forced_break_manager.datetime")
    def test_enforce_trading_break_default_duration(self, mock_datetime):
        """Test enforcing trading break with default duration."""
        # Setup
        now = datetime(2025, 8, 26, 10, 0, 0, tzinfo=UTC)
        expected_expiration = now + timedelta(minutes=30)
        
        # Create a mock datetime class with proper UTC method
        mock_datetime.now.return_value.replace.return_value = now
        mock_datetime.now.return_value = now
        mock_datetime.fromisoformat = datetime.fromisoformat

        # Execute
        result = self.manager.enforce_trading_break(self.account_id)

        # Verify
        self.assertEqual(result, expected_expiration)
        self.assertEqual(
            self.manager._active_breaks[self.account_id], expected_expiration
        )

        # Check database update
        self.mock_repo.update_tilt_profile.assert_called_once_with(
            self.account_id,
            {
                "lockout_expiration": expected_expiration,
                "journal_entries_required": 1,
                "recovery_required": True,
            },
        )

        # Check event published
        self.mock_event_bus.publish.assert_called_once()
        event = self.mock_event_bus.publish.call_args[0][0]
        self.assertEqual(event["type"], EventType.FORCED_BREAK_INITIATED)
        self.assertEqual(event["account_id"], self.account_id)
        self.assertEqual(event["duration_minutes"], 30)

    @patch("genesis.tilt.forced_break_manager.datetime")
    def test_enforce_trading_break_custom_duration(self, mock_datetime):
        """Test enforcing trading break with custom duration."""
        # Setup
        now = datetime(2025, 8, 26, 10, 0, 0, tzinfo=UTC)
        expected_expiration = now + timedelta(minutes=60)
        mock_datetime.now.return_value = now
        mock_datetime.fromisoformat = datetime.fromisoformat

        # Execute
        result = self.manager.enforce_trading_break(
            self.account_id, duration_minutes=60
        )

        # Verify
        self.assertEqual(result, expected_expiration)
        self.assertEqual(
            self.manager._active_breaks[self.account_id], expected_expiration
        )

    def test_is_on_break_active_in_cache(self):
        """Test checking if account is on break when active in cache."""
        # Setup
        future_time = datetime.now(UTC) + timedelta(minutes=10)
        self.manager._active_breaks[self.account_id] = future_time

        # Execute
        result = self.manager.is_on_break(self.account_id)

        # Verify
        self.assertTrue(result)
        # Should not check database if found in cache
        self.mock_repo.get_tilt_profile.assert_not_called()

    def test_is_on_break_expired_in_cache(self):
        """Test checking if account is on break when expired in cache."""
        # Setup
        past_time = datetime.now(UTC) - timedelta(minutes=10)
        self.manager._active_breaks[self.account_id] = past_time

        # Mock database check
        self.mock_repo.get_tilt_profile.return_value = None

        # Execute
        result = self.manager.is_on_break(self.account_id)

        # Verify
        self.assertFalse(result)
        self.assertNotIn(self.account_id, self.manager._active_breaks)
        self.mock_repo.get_tilt_profile.assert_called_once_with(self.account_id)

    def test_is_on_break_active_in_database(self):
        """Test checking if account is on break when active in database."""
        # Setup
        future_time = datetime.now(UTC) + timedelta(minutes=10)
        self.mock_repo.get_tilt_profile.return_value = {
            "lockout_expiration": future_time.isoformat()
        }

        # Execute
        result = self.manager.is_on_break(self.account_id)

        # Verify
        self.assertTrue(result)
        self.assertEqual(self.manager._active_breaks[self.account_id], future_time)

    def test_is_on_break_not_on_break(self):
        """Test checking if account is not on break."""
        # Setup
        self.mock_repo.get_tilt_profile.return_value = None

        # Execute
        result = self.manager.is_on_break(self.account_id)

        # Verify
        self.assertFalse(result)

    @patch("genesis.tilt.forced_break_manager.datetime")
    def test_get_break_status_on_break(self, mock_datetime):
        """Test getting break status when on break."""
        # Setup
        now = datetime(2025, 8, 26, 10, 0, 0, tzinfo=UTC)
        expiration = now + timedelta(minutes=15)
        mock_datetime.now.return_value = now
        mock_datetime.fromisoformat = datetime.fromisoformat

        self.manager._active_breaks[self.account_id] = expiration
        self.manager._loss_counters[self.account_id] = 3

        self.mock_repo.get_tilt_profile.return_value = {
            "journal_entries_required": 1,
            "recovery_required": True,
        }

        # Execute
        status = self.manager.get_break_status(self.account_id)

        # Verify
        self.assertEqual(status["account_id"], self.account_id)
        self.assertTrue(status["is_on_break"])
        self.assertEqual(status["expiration"], expiration.isoformat())
        self.assertEqual(status["remaining_minutes"], 15)
        self.assertEqual(status["consecutive_losses"], 3)
        self.assertTrue(status["journal_required"])
        self.assertTrue(status["recovery_required"])

    def test_get_break_status_not_on_break(self):
        """Test getting break status when not on break."""
        # Setup
        self.mock_repo.get_tilt_profile.return_value = None

        # Execute
        status = self.manager.get_break_status(self.account_id)

        # Verify
        self.assertEqual(status["account_id"], self.account_id)
        self.assertFalse(status["is_on_break"])
        self.assertIsNone(status["expiration"])
        self.assertEqual(status["remaining_minutes"], 0)
        self.assertEqual(status["consecutive_losses"], 0)
        self.assertFalse(status["journal_required"])
        self.assertFalse(status["recovery_required"])

    @patch("genesis.tilt.forced_break_manager.datetime")
    def test_clear_break_with_journal(self, mock_datetime):
        """Test clearing break with journal completed."""
        # Setup
        now = datetime(2025, 8, 26, 10, 0, 0, tzinfo=UTC)
        mock_datetime.now.return_value = now
        mock_datetime.fromisoformat = datetime.fromisoformat

        self.manager._active_breaks[self.account_id] = now + timedelta(minutes=10)
        self.manager._loss_counters[self.account_id] = 3

        self.mock_repo.get_tilt_profile.return_value = {"journal_entries_required": 1}

        # Execute
        result = self.manager.clear_break(self.account_id, journal_completed=True)

        # Verify
        self.assertTrue(result)
        self.assertNotIn(self.account_id, self.manager._active_breaks)
        self.assertEqual(self.manager._loss_counters[self.account_id], 0)

        # Check database update
        self.mock_repo.update_tilt_profile.assert_called_once_with(
            self.account_id,
            {
                "lockout_expiration": None,
                "consecutive_losses": 0,
                "journal_entries_required": 0,
                "recovery_required": False,
            },
        )

        # Check event published
        self.mock_event_bus.publish.assert_called_once()
        event = self.mock_event_bus.publish.call_args[0][0]
        self.assertEqual(event["type"], EventType.FORCED_BREAK_CLEARED)

    def test_clear_break_without_required_journal(self):
        """Test clearing break without required journal fails."""
        # Setup
        self.mock_repo.get_tilt_profile.return_value = {"journal_entries_required": 1}

        # Execute
        result = self.manager.clear_break(self.account_id, journal_completed=False)

        # Verify
        self.assertFalse(result)
        # Should not update database
        self.mock_repo.update_tilt_profile.assert_not_called()
        self.mock_event_bus.publish.assert_not_called()

    def test_clear_break_no_journal_required(self):
        """Test clearing break when no journal required."""
        # Setup
        self.manager._active_breaks[self.account_id] = datetime.now(UTC) + timedelta(
            minutes=10
        )
        self.mock_repo.get_tilt_profile.return_value = {"journal_entries_required": 0}

        # Execute
        result = self.manager.clear_break(self.account_id, journal_completed=False)

        # Verify
        self.assertTrue(result)
        self.assertNotIn(self.account_id, self.manager._active_breaks)

    def test_load_active_breaks_success(self):
        """Test loading active breaks from database on startup."""
        # Setup
        future_time = datetime.now(UTC) + timedelta(minutes=30)
        past_time = datetime.now(UTC) - timedelta(minutes=30)

        self.mock_repo.get_profiles_with_lockouts.return_value = [
            {
                "account_id": "account1",
                "lockout_expiration": future_time.isoformat(),
                "consecutive_losses": 3,
            },
            {
                "account_id": "account2",
                "lockout_expiration": past_time.isoformat(),  # Expired
                "consecutive_losses": 2,
            },
            {
                "account_id": "account3",
                "lockout_expiration": future_time,  # Already datetime object
                "consecutive_losses": 4,
            },
        ]

        # Execute
        self.manager.load_active_breaks()

        # Verify
        self.assertEqual(len(self.manager._active_breaks), 2)
        self.assertIn("account1", self.manager._active_breaks)
        self.assertIn("account3", self.manager._active_breaks)
        self.assertNotIn("account2", self.manager._active_breaks)  # Expired
        self.assertEqual(self.manager._loss_counters["account1"], 3)
        self.assertEqual(self.manager._loss_counters["account3"], 4)

    def test_load_active_breaks_exception(self):
        """Test loading active breaks handles exceptions gracefully."""
        # Setup
        self.mock_repo.get_profiles_with_lockouts.side_effect = Exception("DB error")

        # Execute (should not raise)
        self.manager.load_active_breaks()

        # Verify
        self.assertEqual(len(self.manager._active_breaks), 0)


if __name__ == "__main__":
    unittest.main()
