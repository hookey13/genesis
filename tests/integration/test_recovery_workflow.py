"""Integration tests for drawdown recovery workflow."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from genesis.analytics.drawdown_detector import DrawdownDetector
from genesis.analytics.recovery_pattern_analyzer import RecoveryPatternAnalyzer
from genesis.core.events import EventType
from genesis.core.models import TradingTier
from genesis.engine.strategy_restriction import StrategyRestrictionManager
from genesis.tilt.forced_break_manager import ForcedBreakManager
from genesis.tilt.recovery_protocols import RecoveryProtocolManager, RecoveryStage


class TestRecoveryWorkflowIntegration:
    """Integration tests for complete recovery workflow."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository with basic functionality."""
        repo = Mock()
        repo.get_account = Mock()
        repo.get_tilt_profile = Mock()
        repo.update_tilt_profile = Mock()
        repo.get_peak_balance = Mock()
        repo.update_peak_balance = Mock()
        repo.save_recovery_protocol = AsyncMock()
        repo.update_recovery_protocol = AsyncMock()
        repo.get_trades_since = Mock(return_value=[])
        repo.save_strategy_restrictions = Mock()
        repo.get_recovery_history = Mock(return_value=[])
        return repo

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        bus = Mock()
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def recovery_system(self, mock_repository, mock_event_bus):
        """Create complete recovery system components."""
        return {
            "detector": DrawdownDetector(mock_repository, mock_event_bus),
            "protocol_manager": RecoveryProtocolManager(
                mock_repository, mock_event_bus
            ),
            "break_manager": ForcedBreakManager(mock_repository, mock_event_bus),
            "restriction_manager": StrategyRestrictionManager(mock_repository),
            "pattern_analyzer": RecoveryPatternAnalyzer(mock_repository),
        }

    @pytest.mark.asyncio
    async def test_full_drawdown_recovery_flow(
        self, recovery_system, mock_repository, mock_event_bus
    ):
        """Test complete drawdown recovery workflow."""
        # Setup account with drawdown
        account = Mock()
        account.balance = Decimal("800")  # 20% drawdown from 1000
        account.tier = TradingTier.HUNTER

        mock_repository.get_account.return_value = account
        mock_repository.get_peak_balance.return_value = Decimal("1000")

        # Step 1: Detect drawdown
        detector = recovery_system["detector"]
        is_breach = detector.detect_drawdown_breach("test_account", Decimal("0.10"))

        assert is_breach is True
        assert mock_event_bus.publish.called

        # Step 2: Initiate recovery protocol
        protocol_manager = recovery_system["protocol_manager"]
        protocol = await protocol_manager.initiate_drawdown_recovery(
            "test_account", Decimal("0.20")
        )

        assert protocol.is_drawdown_recovery is True
        assert protocol.recovery_stage == RecoveryStage.STAGE_1  # 50% position size
        assert protocol.drawdown_percentage == Decimal("0.20")

        # Step 3: Apply strategy restrictions
        restriction_manager = recovery_system["restriction_manager"]
        restriction_manager.apply_recovery_restrictions("test_account")

        assert "test_account" in restriction_manager._restricted_accounts

        # Step 4: Simulate trades during recovery
        # Profitable trade
        await protocol_manager.record_trade_result(
            "test_account", Decimal("50"), is_profitable=True
        )

        # Update recovery progress
        trade_result = {"profit_loss": Decimal("50")}
        protocol_manager.update_recovery_progress(
            protocol.protocol_id, trade_result
        )

        assert protocol.profitable_trades_count == 1
        assert protocol.total_profit == Decimal("50")

        # Step 5: Check if still in recovery
        active_protocol = protocol_manager.get_active_protocol("test_account")
        assert active_protocol is not None
        assert active_protocol.is_active is True

    @pytest.mark.asyncio
    async def test_consecutive_losses_trigger_forced_break(
        self, recovery_system, mock_repository, mock_event_bus
    ):
        """Test that consecutive losses trigger forced break."""
        break_manager = recovery_system["break_manager"]

        # Record first loss
        break_expiration = break_manager.record_trade_result(
            "test_account", is_profitable=False
        )
        assert break_expiration is None  # Not enough losses yet

        # Record second loss
        break_expiration = break_manager.record_trade_result(
            "test_account", is_profitable=False
        )
        assert break_expiration is None  # Still not enough

        # Record third loss - should trigger break
        break_expiration = break_manager.record_trade_result(
            "test_account", is_profitable=False
        )
        assert break_expiration is not None
        assert break_manager.is_on_break("test_account") is True

        # Verify event was published
        mock_event_bus.publish.assert_called()
        last_event = mock_event_bus.publish.call_args[0][0]
        assert last_event["type"] == EventType.FORCED_BREAK_INITIATED
        assert last_event["reason"] == "consecutive_losses"

    @pytest.mark.asyncio
    async def test_recovery_milestone_progression(
        self, recovery_system, mock_repository, mock_event_bus
    ):
        """Test recovery milestone progression through stages."""
        protocol_manager = recovery_system["protocol_manager"]

        # Create recovery protocol
        protocol = await protocol_manager.initiate_drawdown_recovery(
            "test_account", Decimal("0.30")  # 30% drawdown
        )

        assert (
            protocol.recovery_stage == RecoveryStage.STAGE_0
        )  # Start at 25% position size

        # Simulate profitable trades to progress through stages
        for _ in range(5):
            await protocol_manager.record_trade_result(
                "test_account", Decimal("100"), is_profitable=True
            )

            trade_result = {"profit_loss": Decimal("100")}
            protocol_manager.update_recovery_progress(
                protocol.protocol_id, trade_result
            )

        # Should have progressed through stages
        assert protocol.profitable_trades_count > 0
        assert protocol.total_profit > Decimal("0")

    @pytest.mark.asyncio
    async def test_strategy_restriction_during_recovery(
        self, recovery_system, mock_repository
    ):
        """Test strategy restrictions are enforced during recovery."""
        restriction_manager = recovery_system["restriction_manager"]

        # Mock trade history for performance calculation
        mock_repository.get_trades_since.return_value = [
            {"strategy_name": "simple_arb", "profit_loss": Decimal("100")},
            {"strategy_name": "simple_arb", "profit_loss": Decimal("-50")},
            {"strategy_name": "spread_capture", "profit_loss": Decimal("200")},
            {"strategy_name": "spread_capture", "profit_loss": Decimal("150")},
        ]

        # Apply recovery restrictions
        restriction_manager.apply_recovery_restrictions("test_account")

        # Check that only highest win-rate strategy is allowed
        assert restriction_manager.is_strategy_allowed("test_account", "spread_capture")
        assert not restriction_manager.is_strategy_allowed("test_account", "simple_arb")

    @pytest.mark.asyncio
    async def test_recovery_pattern_analysis(self, recovery_system, mock_repository):
        """Test recovery pattern analysis functionality."""
        analyzer = recovery_system["pattern_analyzer"]

        # Mock recovery history
        mock_repository.get_recovery_history.return_value = [
            {
                "protocol_id": "proto1",
                "start_time": datetime.now(UTC) - timedelta(days=10),
                "end_time": datetime.now(UTC) - timedelta(days=8),
                "is_successful": True,
                "initial_drawdown": Decimal("0.15"),
                "recovery_duration_hours": 48,
                "strategies_used": ["spread_capture"],
                "total_trades": 20,
                "profitable_trades": 15,
            },
            {
                "protocol_id": "proto2",
                "start_time": datetime.now(UTC) - timedelta(days=20),
                "end_time": datetime.now(UTC) - timedelta(days=18),
                "is_successful": True,
                "initial_drawdown": Decimal("0.20"),
                "recovery_duration_hours": 48,
                "strategies_used": ["simple_arb"],
                "total_trades": 25,
                "profitable_trades": 18,
            },
        ]

        # Analyze recovery patterns
        analysis = analyzer.analyze_recovery_patterns("test_account")

        assert analysis.total_recoveries == 2
        assert analysis.successful_recoveries == 2
        assert analysis.success_rate == Decimal("1.0")
        assert analysis.avg_recovery_duration_hours > 0
        assert len(analysis.best_strategies) > 0

    @pytest.mark.asyncio
    async def test_recovery_completion_flow(
        self, recovery_system, mock_repository, mock_event_bus
    ):
        """Test successful recovery completion flow."""
        protocol_manager = recovery_system["protocol_manager"]
        restriction_manager = recovery_system["restriction_manager"]

        # Start recovery
        protocol = await protocol_manager.initiate_drawdown_recovery(
            "test_account", Decimal("0.15")
        )

        # Apply restrictions
        restriction_manager.apply_recovery_restrictions("test_account")

        # Simulate successful recovery (balance restored)
        mock_repository.get_account.return_value.balance = Decimal("1000")
        mock_repository.get_peak_balance.return_value = Decimal("1000")

        # Complete recovery
        await protocol_manager.complete_recovery(protocol.protocol_id)

        # Verify restrictions removed
        restriction_manager.remove_restrictions("test_account")
        assert "test_account" not in restriction_manager._restricted_accounts

        # Verify event published
        mock_event_bus.publish.assert_called()
        events = [call[0][0] for call in mock_event_bus.publish.call_args_list]
        recovery_complete_events = [
            e for e in events if e.get("type") == EventType.RECOVERY_COMPLETED
        ]
        assert len(recovery_complete_events) > 0
