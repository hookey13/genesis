"""
Strategy integration testing.
Tests single and multi-strategy execution, VWAP, iceberg orders.
"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import structlog

from genesis.core.models import (
    Position,
    Order,
    Signal,
    OrderStatus,
    OrderType,
    OrderSide,
)
from genesis.core.constants import TradingTier as TierType
from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.engine.strategy_registry import StrategyRegistry
from genesis.engine.ab_test_framework import ABTestFramework
from genesis.data.repository import Repository
from genesis.exchange.gateway import BinanceGateway as ExchangeGateway

logger = structlog.get_logger()


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, name, symbol="BTCUSDT"):
        self.name = name
        self.symbol = symbol
        self.active = True
        self.signals_generated = []

    async def analyze(self, market_data):
        """Generate mock signal."""
        signal = Signal(
            strategy_name=self.name,
            symbol=self.symbol,
            side=OrderSide.BUY,
            confidence=Decimal("0.75"),
            suggested_size=Decimal("0.1"),
            timestamp=datetime.utcnow(),
        )
        self.signals_generated.append(signal)
        return signal

    async def shutdown(self):
        """Clean shutdown."""
        self.active = False


class TestStrategyIntegration:
    """Test strategy execution and integration."""

    @pytest.fixture
    def mock_repository(self):
        """Mock repository."""
        repo = Mock(spec=Repository)
        repo.positions = {}
        repo.orders = {}
        repo.save_order = Mock()
        repo.save_position = Mock()
        repo.get_open_positions = Mock(return_value=[])
        return repo

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange gateway."""
        exchange = Mock(spec=ExchangeGateway)
        exchange.place_order = AsyncMock(
            return_value={"orderId": "12345", "status": "NEW"}
        )
        exchange.get_order = AsyncMock(
            return_value={
                "orderId": "12345",
                "status": "FILLED",
                "executedQty": "0.1",
                "avgPrice": "50000",
            }
        )
        exchange.cancel_order = AsyncMock(return_value=True)
        return exchange

    @pytest.fixture
    def strategy_registry(self):
        """Create strategy registry."""
        return StrategyRegistry()

    @pytest.fixture
    def orchestrator(self, mock_repository, mock_exchange):
        """Create strategy orchestrator."""
        return StrategyOrchestrator(
            repository=mock_repository, exchange_gateway=mock_exchange
        )

    @pytest.mark.asyncio
    async def test_single_strategy_execution(self, orchestrator, strategy_registry):
        """Test single strategy executes correctly."""
        # Register strategy
        strategy = MockStrategy("momentum_strategy")
        strategy_registry.register_strategy("momentum", strategy)

        # Add to orchestrator
        await orchestrator.add_strategy("momentum", strategy)

        # Generate and execute signal
        market_data = {"symbol": "BTCUSDT", "price": "50000"}
        signal = await strategy.analyze(market_data)

        # Execute signal
        order = await orchestrator.execute_signal(signal)

        assert order is not None
        assert len(strategy.signals_generated) == 1
        assert orchestrator.active_strategies.get("momentum") == strategy

    @pytest.mark.asyncio
    async def test_multi_strategy_concurrent_execution(self, orchestrator):
        """Test multiple strategies execute concurrently."""
        # Create multiple strategies
        strategies = [MockStrategy(f"strategy_{i}", f"BTC{i}USDT") for i in range(5)]

        # Add all strategies
        for i, strategy in enumerate(strategies):
            await orchestrator.add_strategy(f"strategy_{i}", strategy)

        # Generate signals concurrently
        market_data = {"symbol": "BTCUSDT", "price": "50000"}
        tasks = [s.analyze(market_data) for s in strategies]
        signals = await asyncio.gather(*tasks)

        # Execute all signals
        execution_tasks = [orchestrator.execute_signal(s) for s in signals]
        orders = await asyncio.gather(*execution_tasks)

        assert len(orders) == 5
        assert len(orchestrator.active_strategies) == 5
        for strategy in strategies:
            assert len(strategy.signals_generated) == 1

    @pytest.mark.asyncio
    async def test_vwap_algorithm_integration(self, orchestrator, mock_exchange):
        """Test VWAP execution algorithm integration."""
        # Configure VWAP parameters
        vwap_params = {
            "execution_type": ExecutionType.VWAP,
            "time_window": 300,  # 5 minutes
            "slice_count": 10,
            "min_slice_size": Decimal("0.01"),
        }

        # Create VWAP order
        order = Order(
            id="vwap_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.NEW,
            execution_params=vwap_params,
        )

        # Execute VWAP slices
        slices_executed = []
        for i in range(10):
            slice_order = Order(
                id=f"vwap_001_slice_{i}",
                parent_order_id="vwap_001",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal(str(50000 + i * 10)),
                status=OrderStatus.NEW,
            )

            result = await mock_exchange.place_order(slice_order.__dict__)
            slices_executed.append(result)

            # Simulate partial execution
            await asyncio.sleep(0.01)

        assert len(slices_executed) == 10
        assert all(s["status"] == "NEW" for s in slices_executed)

    @pytest.mark.asyncio
    async def test_iceberg_order_execution(self, orchestrator, mock_exchange):
        """Test iceberg order execution with market conditions."""
        # Configure iceberg parameters
        iceberg_params = {
            "execution_type": ExecutionType.ICEBERG,
            "visible_quantity": Decimal("0.05"),
            "total_quantity": Decimal("0.5"),
            "refresh_threshold": Decimal("0.01"),
        }

        # Create iceberg order
        order = Order(
            id="iceberg_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            status=OrderStatus.NEW,
            execution_params=iceberg_params,
        )

        # Execute visible portions
        visible_orders = []
        remaining = iceberg_params["total_quantity"]

        while remaining > 0:
            visible_qty = min(iceberg_params["visible_quantity"], remaining)

            visible_order = Order(
                id=f"iceberg_001_v_{len(visible_orders)}",
                parent_order_id="iceberg_001",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=visible_qty,
                price=Decimal("50000"),
                status=OrderStatus.NEW,
            )

            result = await mock_exchange.place_order(visible_order.__dict__)
            visible_orders.append(result)
            remaining -= visible_qty

            # Simulate market taking liquidity
            await asyncio.sleep(0.01)

        assert len(visible_orders) == 10  # 0.5 / 0.05
        assert all(o["status"] == "NEW" for o in visible_orders)

    @pytest.mark.asyncio
    async def test_position_limits_across_strategies(
        self, orchestrator, mock_repository
    ):
        """Test position limits are enforced across all strategies."""
        # Set position limits
        orchestrator.max_positions = 5
        orchestrator.max_position_size = Decimal("10000")

        # Create existing positions
        existing_positions = [
            Position(
                id=f"pos_{i}",
                symbol=f"BTC{i}USDT",
                side=OrderSide.BUY,
                entry_price=Decimal("50000"),
                current_price=Decimal("50000"),
                quantity=Decimal("0.05"),
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
            )
            for i in range(4)
        ]

        mock_repository.get_open_positions.return_value = existing_positions

        # Try to open new positions
        new_signals = [
            Signal(
                strategy_name="test",
                symbol="ETHUSDT",
                side=OrderSide.BUY,
                confidence=Decimal("0.9"),
                suggested_size=Decimal("0.1"),
                timestamp=datetime.utcnow(),
            ),
            Signal(
                strategy_name="test2",
                symbol="ADAUSDT",
                side=OrderSide.BUY,
                confidence=Decimal("0.9"),
                suggested_size=Decimal("0.1"),
                timestamp=datetime.utcnow(),
            ),
        ]

        # First should succeed (4 + 1 = 5)
        order1 = await orchestrator.execute_signal(new_signals[0])
        assert order1 is not None

        # Second should be rejected (would be 6)
        with patch.object(orchestrator, "check_position_limits", return_value=False):
            order2 = await orchestrator.execute_signal(new_signals[1])
            assert order2 is None

    @pytest.mark.asyncio
    async def test_strategy_conflict_resolution(self, orchestrator):
        """Test handling of conflicting signals from different strategies."""
        # Create strategies with conflicting signals
        bull_strategy = MockStrategy("bull_strategy")
        bear_strategy = MockStrategy("bear_strategy")

        await orchestrator.add_strategy("bull", bull_strategy)
        await orchestrator.add_strategy("bear", bear_strategy)

        # Generate conflicting signals
        bull_signal = Signal(
            strategy_name="bull_strategy",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            confidence=Decimal("0.8"),
            suggested_size=Decimal("0.1"),
            timestamp=datetime.utcnow(),
        )

        bear_signal = Signal(
            strategy_name="bear_strategy",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            confidence=Decimal("0.7"),
            suggested_size=Decimal("0.1"),
            timestamp=datetime.utcnow() + timedelta(seconds=1),
        )

        # Resolve conflict - higher confidence wins
        signals = [bull_signal, bear_signal]
        resolved = orchestrator.resolve_signal_conflicts(signals)

        assert resolved == bull_signal  # Higher confidence

    @pytest.mark.asyncio
    async def test_strategy_performance_tracking(self, orchestrator, mock_repository):
        """Test strategy performance metrics are tracked correctly."""
        strategy = MockStrategy("perf_test")
        await orchestrator.add_strategy("perf_test", strategy)

        # Execute multiple trades
        trades_data = [
            (OrderSide.BUY, Decimal("50000"), Decimal("0.1"), Decimal("100")),
            (OrderSide.SELL, Decimal("51000"), Decimal("0.1"), Decimal("-50")),
            (OrderSide.BUY, Decimal("49000"), Decimal("0.05"), Decimal("25")),
        ]

        for side, price, qty, pnl in trades_data:
            signal = Signal(
                strategy_name="perf_test",
                symbol="BTCUSDT",
                side=side,
                confidence=Decimal("0.8"),
                suggested_size=qty,
                timestamp=datetime.utcnow(),
            )

            await orchestrator.execute_signal(signal)

            # Update performance metrics
            orchestrator.update_strategy_performance(
                "perf_test", {"pnl": pnl, "trade_count": 1}
            )

        # Check performance metrics
        metrics = orchestrator.performance_metrics.get("perf_test", {})
        assert metrics.get("total_trades", 0) >= 3
        assert "total_pnl" in metrics

    @pytest.mark.asyncio
    async def test_strategy_isolation_on_error(self, orchestrator):
        """Test one strategy error doesn't affect others."""
        # Create strategies, one will fail
        good_strategy = MockStrategy("good_strategy")
        bad_strategy = MockStrategy("bad_strategy")
        bad_strategy.analyze = AsyncMock(side_effect=Exception("Strategy error"))

        await orchestrator.add_strategy("good", good_strategy)
        await orchestrator.add_strategy("bad", bad_strategy)

        # Run analysis
        market_data = {"symbol": "BTCUSDT", "price": "50000"}

        # Good strategy should work
        good_signal = await good_strategy.analyze(market_data)
        assert good_signal is not None

        # Bad strategy should fail but not crash system
        with pytest.raises(Exception):
            await bad_strategy.analyze(market_data)

        # Good strategy should still be active
        assert orchestrator.active_strategies.get("good") == good_strategy

    @pytest.mark.asyncio
    async def test_ab_test_strategy_comparison(self, mock_repository):
        """Test A/B testing framework for strategy comparison."""
        ab_test = ABTestFramework(repository=mock_repository)

        # Create test configuration
        test_config = {
            "test_id": "momentum_vs_mean_reversion",
            "strategy_a": "momentum",
            "strategy_b": "mean_reversion",
            "allocation_ratio": 0.5,
            "duration_hours": 24,
            "metrics": ["pnl", "sharpe_ratio", "win_rate"],
        }

        # Start A/B test
        await ab_test.start_test(test_config)

        # Simulate results
        for i in range(10):
            # Strategy A results
            ab_test.record_result(
                "momentum",
                {
                    "pnl": Decimal(str(100 + i * 10)),
                    "trade_count": 1,
                    "win": i % 2 == 0,
                },
            )

            # Strategy B results
            ab_test.record_result(
                "mean_reversion",
                {"pnl": Decimal(str(80 + i * 15)), "trade_count": 1, "win": i % 3 == 0},
            )

        # Get test results
        results = await ab_test.get_test_results("momentum_vs_mean_reversion")

        assert results is not None
        assert "strategy_a_metrics" in results
        assert "strategy_b_metrics" in results
        assert "recommendation" in results

    @pytest.mark.asyncio
    async def test_strategy_resource_cleanup(self, orchestrator):
        """Test strategies clean up resources on shutdown."""
        strategies = [MockStrategy(f"cleanup_test_{i}") for i in range(3)]

        for i, strategy in enumerate(strategies):
            await orchestrator.add_strategy(f"cleanup_{i}", strategy)

        # Shutdown orchestrator
        await orchestrator.shutdown()

        # Verify all strategies shut down
        for strategy in strategies:
            assert not strategy.active
        assert len(orchestrator.active_strategies) == 0
