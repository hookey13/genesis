"""Unit tests for strategy restriction manager."""

import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from genesis.engine.strategy_restriction import (
    StrategyFilter,
    StrategyRestrictionManager,
    recovery_mode,
)


class TestStrategyRestrictionManager(unittest.TestCase):
    """Test cases for StrategyRestrictionManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_repo = MagicMock()
        self.manager = StrategyRestrictionManager(repository=self.mock_repo)
        self.account_id = "test_account_123"

    @patch("genesis.engine.strategy_restriction.requires_tier")
    def test_restrict_strategies(self, mock_requires_tier):
        """Test restricting strategies for an account."""
        # Mock tier check to pass
        mock_requires_tier.return_value = lambda f: f

        # Setup
        allowed_strategies = ["simple_arb", "spread_capture"]

        # Execute
        self.manager.restrict_strategies(self.account_id, allowed_strategies)

        # Verify
        self.assertIn(self.account_id, self.manager._restricted_accounts)
        self.assertEqual(
            self.manager._restricted_accounts[self.account_id], set(allowed_strategies)
        )
        self.mock_repo.save_strategy_restrictions.assert_called_once()

        # Check arguments
        call_args = self.mock_repo.save_strategy_restrictions.call_args[0]
        self.assertEqual(call_args[0], self.account_id)
        self.assertEqual(call_args[1], allowed_strategies)
        self.assertIsInstance(call_args[2], datetime)

    def test_get_highest_winrate_strategy_with_cache(self):
        """Test getting highest win-rate strategy with cached data."""
        # Setup cache
        self.manager._last_cache_update = datetime.now(UTC)
        self.manager._strategy_performance = {
            "simple_arb": {"win_rate": Decimal("0.65")},
            "spread_capture": {"win_rate": Decimal("0.75")},
            "mean_reversion": {"win_rate": Decimal("0.55")},
        }

        # Execute
        result = self.manager.get_highest_winrate_strategy()

        # Verify
        self.assertEqual(result, "spread_capture")
        # Should not refresh cache
        self.mock_repo.get_trades_since.assert_not_called()

    @patch("genesis.engine.strategy_restriction.datetime")
    def test_get_highest_winrate_strategy_refresh_cache(self, mock_datetime):
        """Test getting highest win-rate strategy refreshes expired cache."""
        # Setup
        now = datetime(2025, 8, 26, 10, 0, 0, tzinfo=UTC)
        mock_datetime.now.return_value = now
        mock_datetime.fromisoformat = datetime.fromisoformat

        # Expired cache
        self.manager._last_cache_update = now - timedelta(hours=2)

        # Mock trades
        self.mock_repo.get_trades_since.return_value = [
            {"strategy_name": "simple_arb", "profit_loss": Decimal("100")},
            {"strategy_name": "simple_arb", "profit_loss": Decimal("-50")},
            {"strategy_name": "simple_arb", "profit_loss": Decimal("75")},
            {"strategy_name": "spread_capture", "profit_loss": Decimal("200")},
            {"strategy_name": "spread_capture", "profit_loss": Decimal("150")},
            {"strategy_name": "mean_reversion", "profit_loss": Decimal("-100")},
        ]

        # Execute
        result = self.manager.get_highest_winrate_strategy()

        # Verify
        self.assertEqual(result, "spread_capture")  # 2/2 = 100% win rate
        self.mock_repo.get_trades_since.assert_called_once()
        self.assertEqual(self.manager._last_cache_update, now)

    def test_get_highest_winrate_strategy_no_data(self):
        """Test getting highest win-rate strategy with no data."""
        # Setup - no cache and no trades
        self.manager._last_cache_update = None
        self.mock_repo.get_trades_since.return_value = []

        # Execute
        result = self.manager.get_highest_winrate_strategy()

        # Verify
        self.assertEqual(result, "simple_arb")  # Default fallback

    def test_get_highest_winrate_strategy_exception(self):
        """Test getting highest win-rate strategy handles exceptions."""
        # Setup
        self.manager._last_cache_update = None
        self.mock_repo.get_trades_since.side_effect = Exception("DB error")

        # Execute
        result = self.manager.get_highest_winrate_strategy()

        # Verify
        self.assertEqual(result, "simple_arb")  # Default fallback

    def test_is_strategy_allowed_no_restrictions(self):
        """Test checking if strategy is allowed when no restrictions."""
        # Execute
        result = self.manager.is_strategy_allowed(self.account_id, "any_strategy")

        # Verify
        self.assertTrue(result)

    def test_is_strategy_allowed_with_restrictions(self):
        """Test checking if strategy is allowed with restrictions."""
        # Setup
        self.manager._restricted_accounts[self.account_id] = {
            "simple_arb",
            "spread_capture",
        }

        # Execute
        allowed_result = self.manager.is_strategy_allowed(self.account_id, "simple_arb")
        not_allowed_result = self.manager.is_strategy_allowed(
            self.account_id, "mean_reversion"
        )

        # Verify
        self.assertTrue(allowed_result)
        self.assertFalse(not_allowed_result)

    def test_remove_restrictions(self):
        """Test removing strategy restrictions."""
        # Setup
        self.manager._restricted_accounts[self.account_id] = {"simple_arb"}

        # Execute
        self.manager.remove_restrictions(self.account_id)

        # Verify
        self.assertNotIn(self.account_id, self.manager._restricted_accounts)
        self.mock_repo.remove_strategy_restrictions.assert_called_once_with(
            self.account_id
        )

    def test_remove_restrictions_not_restricted(self):
        """Test removing restrictions when account not restricted."""
        # Execute
        self.manager.remove_restrictions(self.account_id)

        # Verify - should not error
        self.mock_repo.remove_strategy_restrictions.assert_not_called()

    @patch("genesis.engine.strategy_restriction.requires_tier")
    def test_apply_recovery_restrictions(self, mock_requires_tier):
        """Test applying recovery mode restrictions."""
        # Mock tier check to pass
        mock_requires_tier.return_value = lambda f: f

        # Setup best strategy
        with patch.object(
            self.manager, "get_highest_winrate_strategy", return_value="spread_capture"
        ):
            # Execute
            self.manager.apply_recovery_restrictions(self.account_id)

            # Verify
            self.assertIn(self.account_id, self.manager._restricted_accounts)
            self.assertEqual(
                self.manager._restricted_accounts[self.account_id], {"spread_capture"}
            )

    def test_should_refresh_cache_no_update(self):
        """Test cache refresh check when never updated."""
        # Setup
        self.manager._last_cache_update = None

        # Execute
        result = self.manager._should_refresh_cache()

        # Verify
        self.assertTrue(result)

    def test_should_refresh_cache_fresh(self):
        """Test cache refresh check when cache is fresh."""
        # Setup
        self.manager._last_cache_update = datetime.now(UTC) - timedelta(minutes=30)

        # Execute
        result = self.manager._should_refresh_cache()

        # Verify
        self.assertFalse(result)

    def test_should_refresh_cache_expired(self):
        """Test cache refresh check when cache is expired."""
        # Setup
        self.manager._last_cache_update = datetime.now(UTC) - timedelta(hours=2)

        # Execute
        result = self.manager._should_refresh_cache()

        # Verify
        self.assertTrue(result)

    @patch("genesis.engine.strategy_restriction.datetime")
    def test_refresh_strategy_performance(self, mock_datetime):
        """Test refreshing strategy performance statistics."""
        # Setup
        now = datetime(2025, 8, 26, 10, 0, 0, tzinfo=UTC)
        mock_datetime.now.return_value = now

        # Mock trades with various outcomes
        self.mock_repo.get_trades_since.return_value = [
            {"strategy_name": "simple_arb", "profit_loss": Decimal("100")},
            {"strategy_name": "simple_arb", "profit_loss": Decimal("-50")},
            {"strategy_name": "simple_arb", "profit_loss": Decimal("75")},
            {"strategy_name": "spread_capture", "profit_loss": Decimal("200")},
            {"strategy_name": "spread_capture", "profit_loss": Decimal("150")},
            {"strategy_name": "mean_reversion", "profit_loss": Decimal("-100")},
            {"strategy_name": "mean_reversion", "profit_loss": Decimal("-50")},
            {"strategy_name": "mean_reversion", "profit_loss": Decimal("25")},
        ]

        # Execute
        self.manager._refresh_strategy_performance(30)

        # Verify
        self.assertEqual(len(self.manager._strategy_performance), 3)

        # Check simple_arb: 2 wins out of 3 trades
        simple_arb = self.manager._strategy_performance["simple_arb"]
        self.assertEqual(simple_arb["total_trades"], 3)
        self.assertEqual(simple_arb["winning_trades"], 2)
        self.assertEqual(simple_arb["win_rate"], Decimal("2") / Decimal("3"))
        self.assertEqual(simple_arb["total_profit"], Decimal("175"))
        self.assertEqual(simple_arb["total_loss"], Decimal("50"))

        # Check spread_capture: 2 wins out of 2 trades
        spread_capture = self.manager._strategy_performance["spread_capture"]
        self.assertEqual(spread_capture["total_trades"], 2)
        self.assertEqual(spread_capture["winning_trades"], 2)
        self.assertEqual(spread_capture["win_rate"], Decimal("1"))

        # Check mean_reversion: 1 win out of 3 trades
        mean_reversion = self.manager._strategy_performance["mean_reversion"]
        self.assertEqual(mean_reversion["total_trades"], 3)
        self.assertEqual(mean_reversion["winning_trades"], 1)
        self.assertEqual(mean_reversion["win_rate"], Decimal("1") / Decimal("3"))

        self.assertEqual(self.manager._last_cache_update, now)

    def test_refresh_strategy_performance_no_trades(self):
        """Test refreshing strategy performance with no trades."""
        # Setup
        self.mock_repo.get_trades_since.return_value = []

        # Execute
        self.manager._refresh_strategy_performance(30)

        # Verify
        self.assertEqual(len(self.manager._strategy_performance), 0)


class TestRecoveryModeDecorator(unittest.TestCase):
    """Test cases for recovery_mode decorator."""

    def test_recovery_mode_decorator_with_required_args(self):
        """Test recovery mode decorator with required arguments."""

        # Setup
        @recovery_mode
        def test_function(account_id=None, strategy_name=None):
            return f"Executed: {account_id}, {strategy_name}"

        # Execute
        result = test_function(account_id="test123", strategy_name="simple_arb")

        # Verify
        self.assertEqual(result, "Executed: test123, simple_arb")

    def test_recovery_mode_decorator_missing_args(self):
        """Test recovery mode decorator with missing arguments."""

        # Setup
        @recovery_mode
        def test_function(value=None):
            return f"Executed: {value}"

        # Execute
        result = test_function(value="test")

        # Verify
        self.assertEqual(result, "Executed: test")


class TestStrategyFilter(unittest.TestCase):
    """Test cases for StrategyFilter."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_manager = MagicMock(spec=StrategyRestrictionManager)
        self.filter = StrategyFilter(restriction_manager=self.mock_manager)
        self.account_id = "test_account_123"

    def test_filter_available_strategies_no_restrictions(self):
        """Test filtering strategies with no restrictions."""
        # Setup
        all_strategies = ["simple_arb", "spread_capture", "mean_reversion"]
        self.mock_manager.is_strategy_allowed.return_value = True

        # Execute
        result = self.filter.filter_available_strategies(
            self.account_id, all_strategies
        )

        # Verify
        self.assertEqual(result, all_strategies)
        self.assertEqual(self.mock_manager.is_strategy_allowed.call_count, 3)

    def test_filter_available_strategies_with_restrictions(self):
        """Test filtering strategies with restrictions."""
        # Setup
        all_strategies = ["simple_arb", "spread_capture", "mean_reversion"]

        def is_allowed_side_effect(account_id, strategy):
            return strategy in ["simple_arb", "mean_reversion"]

        self.mock_manager.is_strategy_allowed.side_effect = is_allowed_side_effect

        # Execute
        result = self.filter.filter_available_strategies(
            self.account_id, all_strategies
        )

        # Verify
        self.assertEqual(result, ["simple_arb", "mean_reversion"])

    def test_filter_available_strategies_none_allowed(self):
        """Test filtering strategies when none are allowed."""
        # Setup
        all_strategies = ["simple_arb", "spread_capture", "mean_reversion"]
        self.mock_manager.is_strategy_allowed.return_value = False

        # Execute
        result = self.filter.filter_available_strategies(
            self.account_id, all_strategies
        )

        # Verify - should return first strategy as default
        self.assertEqual(result, ["simple_arb"])

    def test_filter_available_strategies_empty_list(self):
        """Test filtering strategies with empty strategy list."""
        # Setup
        all_strategies = []

        # Execute
        result = self.filter.filter_available_strategies(
            self.account_id, all_strategies
        )

        # Verify
        self.assertEqual(result, [])
        self.mock_manager.is_strategy_allowed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
