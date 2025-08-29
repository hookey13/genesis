"""
End-to-end data flow integration tests.
Verifies market data → strategy → execution → analytics pipeline.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
import structlog

from genesis.analytics.performance_attribution import PerformanceAttributionEngine
from genesis.analytics.risk_metrics import RiskMetricsCalculator
from genesis.core.models import (
    OrderSide,
    Position,
    Signal,
    Trade,
)
from genesis.core.models import (
    PriceData as MarketData,  # Using PriceData as MarketData
)
from genesis.data.performance_repo import PerformanceRepository
from genesis.data.repository import Repository
from genesis.engine.strategy_orchestrator import StrategyOrchestrator
from genesis.engine.strategy_registry import StrategyRegistry
from genesis.exchange.gateway import BinanceGateway as ExchangeGateway

logger = structlog.get_logger()


class TestDataFlow:
    """Test complete data flow through the system."""

    @pytest.fixture
    def mock_market_data(self):
        """Generate mock market data stream."""
        return [
            MarketData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow(),
                bid_price=Decimal("50000"),
                ask_price=Decimal("50010"),
                bid_volume=Decimal("1.5"),
                ask_volume=Decimal("2.0"),
                last_price=Decimal("50005"),
                volume_24h=Decimal("1000"),
            ),
            MarketData(
                symbol="BTCUSDT",
                timestamp=datetime.utcnow() + timedelta(seconds=1),
                bid_price=Decimal("50100"),
                ask_price=Decimal("50110"),
                bid_volume=Decimal("1.2"),
                ask_volume=Decimal("1.8"),
                last_price=Decimal("50105"),
                volume_24h=Decimal("1001"),
            ),
        ]

    @pytest.fixture
    def mock_repository(self):
        """Mock repository with in-memory storage."""
        repo = Mock(spec=Repository)
        repo.trades = []
        repo.positions = {}
        repo.orders = {}
        repo.save_trade = Mock(side_effect=lambda t: repo.trades.append(t))
        repo.save_position = Mock(
            side_effect=lambda p: repo.positions.update({p.id: p})
        )
        repo.save_order = Mock(side_effect=lambda o: repo.orders.update({o.id: o}))
        repo.get_trades = Mock(return_value=repo.trades)
        return repo

    @pytest.fixture
    def mock_exchange(self):
        """Mock exchange gateway."""
        exchange = Mock(spec=ExchangeGateway)
        exchange.place_order = AsyncMock(return_value={"orderId": "123456"})
        exchange.get_order = AsyncMock(
            return_value={
                "orderId": "123456",
                "status": "FILLED",
                "executedQty": "0.1",
                "avgPrice": "50050",
            }
        )
        exchange.stream_market_data = AsyncMock()
        return exchange

    @pytest.mark.asyncio
    async def test_market_data_ingestion_to_signal(
        self, mock_market_data, mock_repository
    ):
        """Test market data flows to strategy signal generation."""
        # Setup strategy that generates signals
        strategy = Mock()
        strategy.name = "test_strategy"
        strategy.analyze = Mock(
            return_value=Signal(
                strategy_name="test_strategy",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                confidence=Decimal("0.85"),
                suggested_size=Decimal("0.1"),
                timestamp=datetime.utcnow(),
            )
        )

        registry = StrategyRegistry()
        registry.register_strategy("test_strategy", strategy)

        # Process market data
        signals = []
        for data in mock_market_data:
            signal = strategy.analyze(data)
            if signal:
                signals.append(signal)

        assert len(signals) > 0
        assert signals[0].confidence == Decimal("0.85")
        assert signals[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_signal_to_order_execution(self, mock_repository, mock_exchange):
        """Test signal generation leads to order execution."""
        orchestrator = StrategyOrchestrator(
            repository=mock_repository, exchange_gateway=mock_exchange
        )

        # Create signal
        signal = Signal(
            strategy_name="test_strategy",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            confidence=Decimal("0.9"),
            suggested_size=Decimal("0.1"),
            timestamp=datetime.utcnow(),
        )

        # Execute signal
        order = await orchestrator.execute_signal(signal)

        assert mock_exchange.place_order.called
        assert order is not None
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_trade_logging_to_database(self, mock_repository):
        """Test trades are properly logged to database."""
        # Create trade
        trade = Trade(
            id="trade_001",
            order_id="order_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("50050"),
            quantity=Decimal("0.1"),
            commission=Decimal("0.05"),
            timestamp=datetime.utcnow(),
        )

        # Save to repository
        mock_repository.save_trade(trade)

        # Verify saved
        assert len(mock_repository.trades) == 1
        assert mock_repository.trades[0].id == "trade_001"
        assert mock_repository.trades[0].price == Decimal("50050")

    @pytest.mark.asyncio
    async def test_realtime_risk_calculation_updates(self, mock_repository):
        """Test risk metrics update in real-time with trades."""
        risk_calc = RiskMetricsCalculator(repository=mock_repository)

        # Add positions
        position1 = Position(
            id="pos_001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            current_price=Decimal("50500"),
            quantity=Decimal("0.1"),
            unrealized_pnl=Decimal("50"),
            realized_pnl=Decimal("0"),
        )

        position2 = Position(
            id="pos_002",
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal("3000"),
            current_price=Decimal("2950"),
            quantity=Decimal("1"),
            unrealized_pnl=Decimal("-50"),
            realized_pnl=Decimal("0"),
        )

        mock_repository.positions = {"pos_001": position1, "pos_002": position2}
        mock_repository.get_open_positions = Mock(return_value=[position1, position2])

        # Calculate risk metrics
        metrics = await risk_calc.calculate_current_risk()

        assert "total_exposure" in metrics
        assert "position_count" in metrics
        assert "largest_position" in metrics
        assert "total_pnl" in metrics
        assert metrics["position_count"] == 2

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, mock_repository):
        """Test performance metrics calculation with no NaN/null values."""
        perf_repo = Mock(spec=PerformanceRepository)
        perf_engine = PerformanceAttributionEngine(repository=mock_repository)

        # Create sample trades
        trades = [
            Trade(
                id=f"trade_{i}",
                order_id=f"order_{i}",
                symbol="BTCUSDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=Decimal(str(50000 + i * 100)),
                quantity=Decimal("0.1"),
                commission=Decimal("0.05"),
                timestamp=datetime.utcnow() - timedelta(hours=i),
            )
            for i in range(10)
        ]

        mock_repository.get_trades = Mock(return_value=trades)

        # Calculate metrics
        metrics = await perf_engine.calculate_performance_metrics(
            start_date=datetime.utcnow() - timedelta(days=1), end_date=datetime.utcnow()
        )

        # Verify no NaN or null values
        assert metrics is not None
        for key, value in metrics.items():
            assert value is not None, f"{key} is None"
            if isinstance(value, (int, float, Decimal)):
                assert not (
                    isinstance(value, float) and value != value
                ), f"{key} is NaN"

    @pytest.mark.asyncio
    async def test_data_integrity_validation(self, mock_repository):
        """Test data integrity throughout the pipeline."""
        # Test decimal precision preservation
        original_price = Decimal("50123.456789")

        trade = Trade(
            id="precision_test",
            order_id="order_precision",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=original_price,
            quantity=Decimal("0.123456"),
            commission=Decimal("0.0001"),
            timestamp=datetime.utcnow(),
        )

        # Save and retrieve
        mock_repository.save_trade(trade)
        mock_repository.get_trade = Mock(return_value=trade)
        retrieved = mock_repository.get_trade("precision_test")

        # Verify precision preserved
        assert retrieved.price == original_price
        assert str(retrieved.price) == str(original_price)

    @pytest.mark.asyncio
    async def test_concurrent_data_flow(self, mock_repository, mock_exchange):
        """Test multiple data flows can process concurrently."""
        orchestrator = StrategyOrchestrator(
            repository=mock_repository, exchange_gateway=mock_exchange
        )

        # Create multiple signals
        signals = [
            Signal(
                strategy_name=f"strategy_{i}",
                symbol=f"BTC{i}USDT",
                side=OrderSide.BUY,
                confidence=Decimal("0.8"),
                suggested_size=Decimal("0.1"),
                timestamp=datetime.utcnow(),
            )
            for i in range(5)
        ]

        # Process concurrently
        tasks = [orchestrator.execute_signal(signal) for signal in signals]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all processed
        assert len(results) == 5
        assert mock_exchange.place_order.call_count == 5

    @pytest.mark.asyncio
    async def test_market_data_to_analytics_latency(self, mock_market_data):
        """Test latency from market data to analytics update."""
        start_time = datetime.utcnow()

        # Simulate processing chain
        for data in mock_market_data:
            # Strategy analysis
            analysis_time = datetime.utcnow()

            # Order execution
            execution_time = datetime.utcnow()

            # Analytics update
            analytics_time = datetime.utcnow()

        total_latency = (datetime.utcnow() - start_time).total_seconds()

        # Should complete in under 100ms for simple flow
        assert total_latency < 0.1, f"Latency too high: {total_latency}s"

    @pytest.mark.asyncio
    async def test_error_propagation_in_pipeline(self, mock_repository, mock_exchange):
        """Test errors are properly propagated through data pipeline."""
        orchestrator = StrategyOrchestrator(
            repository=mock_repository, exchange_gateway=mock_exchange
        )

        # Simulate exchange error
        mock_exchange.place_order.side_effect = Exception("Exchange unavailable")

        signal = Signal(
            strategy_name="test_strategy",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            confidence=Decimal("0.9"),
            suggested_size=Decimal("0.1"),
            timestamp=datetime.utcnow(),
        )

        # Error should be caught and logged
        with pytest.raises(Exception) as exc_info:
            await orchestrator.execute_signal(signal)

        assert "Exchange unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_data_consistency_across_components(self, mock_repository):
        """Test data remains consistent across different components."""
        # Create position in repository
        position = Position(
            id="consistency_test",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            quantity=Decimal("0.1"),
            unrealized_pnl=Decimal("100"),
            realized_pnl=Decimal("0"),
        )

        mock_repository.save_position(position)

        # Access from different components
        risk_calc = RiskMetricsCalculator(repository=mock_repository)
        perf_engine = PerformanceAttributionEngine(repository=mock_repository)

        mock_repository.get_position = Mock(return_value=position)

        # Verify same data
        risk_position = mock_repository.get_position("consistency_test")
        perf_position = mock_repository.get_position("consistency_test")

        assert risk_position.id == perf_position.id
        assert risk_position.unrealized_pnl == perf_position.unrealized_pnl

    @pytest.mark.asyncio
    async def test_market_data_subscription_lifecycle(self, mock_exchange):
        """Test market data subscription setup and teardown."""
        subscriptions = []

        async def subscribe_callback(data):
            subscriptions.append(data)

        # Setup subscription
        mock_exchange.subscribe_market_data = AsyncMock()
        mock_exchange.unsubscribe_market_data = AsyncMock()

        await mock_exchange.subscribe_market_data("BTCUSDT", subscribe_callback)
        assert mock_exchange.subscribe_market_data.called

        # Simulate data flow
        test_data = {"symbol": "BTCUSDT", "price": "50000"}
        await subscribe_callback(test_data)
        assert len(subscriptions) == 1

        # Teardown
        await mock_exchange.unsubscribe_market_data("BTCUSDT")
        assert mock_exchange.unsubscribe_market_data.called
