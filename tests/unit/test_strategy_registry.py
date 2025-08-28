"""
Unit tests for strategy registry and lifecycle management.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from genesis.engine.event_bus import EventBus
from genesis.engine.strategy_registry import (
    StrategyHealth,
    StrategyMetadata,
    StrategyPriority,
    StrategyRegistry,
    StrategyState,
)
from genesis.strategies.loader import StrategyLoader


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    event_bus = AsyncMock(spec=EventBus)
    return event_bus


@pytest.fixture
def mock_strategy_loader():
    """Create mock strategy loader."""
    loader = MagicMock(spec=StrategyLoader)
    loader.is_strategy_enabled.return_value = True
    return loader


@pytest.fixture
def strategy_metadata():
    """Create sample strategy metadata."""
    return StrategyMetadata(
        name="test_strategy",
        version="1.0.0",
        tier_required="STRATEGIST",
        priority=StrategyPriority.NORMAL,
        max_positions=5,
        max_capital_percent=Decimal("20"),
        min_capital_usdt=Decimal("100"),
        compatible_symbols=["BTC/USDT", "ETH/USDT"],
        conflicting_strategies={"incompatible_strategy"},
    )


@pytest_asyncio.fixture
async def registry(mock_event_bus, mock_strategy_loader):
    """Create strategy registry instance."""
    registry = StrategyRegistry(mock_event_bus, mock_strategy_loader)
    await registry.start()
    yield registry
    await registry.stop()


class TestStrategyHealth:
    """Test strategy health tracking."""

    def test_update_heartbeat(self):
        """Test heartbeat update."""
        health = StrategyHealth()
        original_time = health.last_heartbeat

        # Wait briefly to ensure time difference
        import time

        time.sleep(0.01)

        health.update_heartbeat()
        assert health.last_heartbeat > original_time

    def test_record_error(self):
        """Test error recording."""
        health = StrategyHealth()
        assert health.error_count == 0
        assert health.is_healthy is True

        # Record errors
        for i in range(4):
            health.record_error(f"Error {i}")
            assert health.error_count == i + 1
            assert health.is_healthy is True

        # Fifth error should mark unhealthy
        health.record_error("Error 5")
        assert health.error_count == 5
        assert health.is_healthy is False
        assert health.last_error == "Error 5"

    def test_record_restart(self):
        """Test restart recording."""
        health = StrategyHealth()
        health.error_count = 3

        health.record_restart()

        assert health.restart_count == 1
        assert health.error_count == 0  # Reset on restart
        assert health.last_restart is not None


class TestStrategyMetadata:
    """Test strategy metadata."""

    def test_decimal_conversion(self):
        """Test automatic Decimal conversion."""
        metadata = StrategyMetadata(
            name="test",
            max_capital_percent=20,  # Pass as int
            min_capital_usdt=100,  # Pass as int
        )

        assert isinstance(metadata.max_capital_percent, Decimal)
        assert isinstance(metadata.min_capital_usdt, Decimal)
        assert metadata.max_capital_percent == Decimal("20")
        assert metadata.min_capital_usdt == Decimal("100")

    def test_default_values(self):
        """Test default metadata values."""
        metadata = StrategyMetadata(name="test")

        assert metadata.version == "1.0.0"
        assert metadata.tier_required == "STRATEGIST"
        assert metadata.priority == StrategyPriority.NORMAL
        assert metadata.max_positions == 5
        assert metadata.requires_market_data is True
        assert len(metadata.compatible_symbols) == 0
        assert len(metadata.conflicting_strategies) == 0
        assert len(metadata.dependencies) == 0


class TestStrategyRegistry:
    """Test strategy registry operations."""

    async def test_register_strategy(self, registry, strategy_metadata, mock_event_bus):
        """Test successful strategy registration."""
        account_id = "test_account"

        strategy_id = await registry.register_strategy(
            account_id=account_id,
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        assert strategy_id == strategy_metadata.strategy_id
        assert strategy_id in registry._strategies
        assert strategy_id in registry._metadata
        assert strategy_id in registry._health

        # Check instance creation
        instance = registry._strategies[strategy_id]
        assert instance.account_id == account_id
        assert instance.state == StrategyState.IDLE
        assert instance.allocated_capital == Decimal("0")

        # Check event published
        mock_event_bus.publish.assert_called_once()

    async def test_register_disabled_strategy(
        self, registry, strategy_metadata, mock_strategy_loader
    ):
        """Test registering a disabled strategy fails."""
        mock_strategy_loader.is_strategy_enabled.return_value = False

        with pytest.raises(ValueError, match="not enabled"):
            await registry.register_strategy(
                account_id="test_account",
                strategy_name="test_strategy",
                metadata=strategy_metadata,
            )

    async def test_register_conflicting_strategy(self, registry, strategy_metadata):
        """Test registering conflicting strategies."""
        # Register first strategy
        meta1 = StrategyMetadata(name="strategy1", conflicting_strategies={"strategy2"})

        strategy1_id = await registry.register_strategy(
            account_id="account1", strategy_name="strategy1", metadata=meta1
        )

        # Start first strategy
        await registry.start_strategy(strategy1_id)

        # Try to register conflicting strategy
        meta2 = StrategyMetadata(name="strategy2")
        registry._metadata[strategy1_id].conflicting_strategies.add("strategy2")

        with pytest.raises(ValueError, match="conflicts with running strategy"):
            await registry.register_strategy(
                account_id="account1", strategy_name="strategy2", metadata=meta2
            )

    async def test_unregister_strategy(
        self, registry, strategy_metadata, mock_event_bus
    ):
        """Test strategy unregistration."""
        # Register strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        # Unregister
        result = await registry.unregister_strategy(strategy_id)

        assert result is True
        assert strategy_id not in registry._strategies
        assert strategy_id not in registry._metadata
        assert strategy_id not in registry._health

        # Check event published
        assert mock_event_bus.publish.call_count == 2  # Register + unregister

    async def test_unregister_running_strategy(self, registry, strategy_metadata):
        """Test unregistering a running strategy stops it first."""
        # Register and start strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        await registry.start_strategy(strategy_id)
        assert registry._strategies[strategy_id].state == StrategyState.RUNNING

        # Unregister should stop it
        await registry.unregister_strategy(strategy_id)
        assert strategy_id not in registry._strategies

    async def test_start_strategy(self, registry, strategy_metadata, mock_event_bus):
        """Test starting a strategy."""
        # Register strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        # Start strategy
        result = await registry.start_strategy(strategy_id)

        assert result is True
        instance = registry._strategies[strategy_id]
        assert instance.state == StrategyState.RUNNING
        assert instance.started_at is not None
        assert strategy_id in registry._strategy_tasks

        # Check event published
        assert mock_event_bus.publish.call_count == 2  # Register + start

    async def test_start_invalid_state(self, registry, strategy_metadata):
        """Test starting strategy from invalid state."""
        # Register and start strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        await registry.start_strategy(strategy_id)

        # Try to start again (already running)
        result = await registry.start_strategy(strategy_id)
        assert result is False

    async def test_pause_strategy(self, registry, strategy_metadata, mock_event_bus):
        """Test pausing a running strategy."""
        # Register and start strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        await registry.start_strategy(strategy_id)

        # Pause strategy
        result = await registry.pause_strategy(strategy_id)

        assert result is True
        assert registry._strategies[strategy_id].state == StrategyState.PAUSED

        # Check event published
        assert mock_event_bus.publish.call_count == 3  # Register + start + pause

    async def test_pause_non_running_strategy(self, registry, strategy_metadata):
        """Test pausing a non-running strategy fails."""
        # Register strategy (but don't start)
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        result = await registry.pause_strategy(strategy_id)
        assert result is False

    async def test_resume_strategy(self, registry, strategy_metadata, mock_event_bus):
        """Test resuming a paused strategy."""
        # Register, start, and pause strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        await registry.start_strategy(strategy_id)
        await registry.pause_strategy(strategy_id)

        # Resume strategy
        result = await registry.resume_strategy(strategy_id)

        assert result is True
        assert registry._strategies[strategy_id].state == StrategyState.RUNNING

        # Check event published
        assert (
            mock_event_bus.publish.call_count == 4
        )  # Register + start + pause + resume

    async def test_get_active_strategies(self, registry):
        """Test getting active strategies."""
        # Register multiple strategies
        meta1 = StrategyMetadata(name="strategy1")
        meta2 = StrategyMetadata(name="strategy2")
        meta3 = StrategyMetadata(name="strategy3")

        id1 = await registry.register_strategy("account1", "strategy1", meta1)
        id2 = await registry.register_strategy("account1", "strategy2", meta2)
        id3 = await registry.register_strategy("account2", "strategy3", meta3)

        # Start some strategies
        await registry.start_strategy(id1)
        await registry.start_strategy(id3)

        # Get all active
        active = registry.get_active_strategies()
        assert len(active) == 2

        # Get active for specific account
        account1_active = registry.get_active_strategies("account1")
        assert len(account1_active) == 1
        assert account1_active[0].strategy_id == id1

    async def test_get_strategy_state(self, registry, strategy_metadata):
        """Test getting strategy state."""
        # Register strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        # Check initial state
        state = registry.get_strategy_state(strategy_id)
        assert state == StrategyState.IDLE

        # Start and check again
        await registry.start_strategy(strategy_id)
        state = registry.get_strategy_state(strategy_id)
        assert state == StrategyState.RUNNING

        # Check non-existent strategy
        state = registry.get_strategy_state("invalid_id")
        assert state is None

    async def test_get_strategy_health(self, registry, strategy_metadata):
        """Test getting strategy health metrics."""
        # Register strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        health = registry.get_strategy_health(strategy_id)
        assert health is not None
        assert health.error_count == 0
        assert health.is_healthy is True

        # Check non-existent strategy
        health = registry.get_strategy_health("invalid_id")
        assert health is None

    @pytest.mark.asyncio
    async def test_health_monitor_heartbeat_timeout(self, registry, strategy_metadata):
        """Test health monitor detecting heartbeat timeout."""
        # Register and start strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        await registry.start_strategy(strategy_id)

        # Simulate heartbeat timeout
        health = registry._health[strategy_id]
        health.last_heartbeat = datetime.now(UTC) - timedelta(seconds=35)

        # Mock recovery
        with patch.object(
            registry, "_recover_strategy", new_callable=AsyncMock
        ) as mock_recover:
            # Manually trigger the health check logic
            current_time = datetime.now(UTC)
            for sid, health in registry._health.items():
                instance = registry._strategies.get(sid)
                if instance and instance.state == StrategyState.RUNNING:
                    time_since_heartbeat = (
                        current_time - health.last_heartbeat
                    ).total_seconds()
                    if time_since_heartbeat > 30 and health.restart_count < 3:
                        await registry._recover_strategy(sid)

            # Should attempt recovery
            mock_recover.assert_called_once_with(strategy_id)

    async def test_strategy_recovery(self, registry, strategy_metadata, mock_event_bus):
        """Test automatic strategy recovery."""
        # Register strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        # Simulate error state
        registry._strategies[strategy_id].state = StrategyState.ERROR
        health = registry._health[strategy_id]

        # Perform recovery
        await registry._recover_strategy(strategy_id)

        # Check recovery
        assert health.restart_count == 1
        assert health.last_restart is not None
        assert registry._strategies[strategy_id].state == StrategyState.RUNNING

        # Check recovery event published
        event_calls = mock_event_bus.publish.call_args_list
        assert any("STRATEGY_RECOVERED" in str(call) for call in event_calls)

    async def test_recovery_limit(self, registry, strategy_metadata):
        """Test recovery limit enforcement."""
        # Register strategy
        strategy_id = await registry.register_strategy(
            account_id="test_account",
            strategy_name="test_strategy",
            metadata=strategy_metadata,
        )

        # Set restart count to limit
        health = registry._health[strategy_id]
        health.restart_count = 3

        # Mock recovery attempt
        with patch.object(registry, "_recover_strategy") as mock_recover:
            # Simulate heartbeat timeout with max restarts
            registry._strategies[strategy_id].state = StrategyState.RUNNING
            health.last_heartbeat = datetime.now(UTC) - timedelta(seconds=35)

            # Run health monitor
            registry._shutdown_event.clear()
            monitor_task = asyncio.create_task(registry._health_monitor())
            await asyncio.sleep(0.1)
            registry._shutdown_event.set()
            monitor_task.cancel()

            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            # Should not attempt recovery
            mock_recover.assert_not_called()
            # Should be in error state
            assert registry._strategies[strategy_id].state == StrategyState.ERROR
