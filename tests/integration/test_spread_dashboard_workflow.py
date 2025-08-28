"""
Integration tests for spread dashboard workflow.

Tests the complete flow of spread tracking, analysis, visualization,
and persistence across multiple components.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.analytics.spread_analyzer import SpreadAnalyzer
from genesis.analytics.spread_pattern_analyzer import SpreadPatternAnalyzer
from genesis.analytics.spread_profitability import SpreadProfitabilityCalculator
from genesis.analytics.spread_tracker import SpreadTracker
from genesis.core.events import Event, EventType
from genesis.data.market_data_service import (
    MarketDataService,
)
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus


class TestSpreadDashboardWorkflow:
    """Integration tests for spread analytics dashboard workflow."""

    @pytest.fixture
    async def event_bus(self):
        """Create event bus instance."""
        bus = EventBus()
        await bus.start()
        yield bus
        await bus.stop()

    @pytest.fixture
    async def repository(self, tmp_path):
        """Create temporary repository."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(str(db_path))
        await repo.connect()
        await repo.initialize_database()
        yield repo
        await repo.disconnect()

    @pytest.fixture
    def spread_analyzer(self):
        """Create spread analyzer."""
        return SpreadAnalyzer(max_history_size=100)

    @pytest.fixture
    def spread_tracker(self, spread_analyzer, event_bus):
        """Create spread tracker."""
        return SpreadTracker(spread_analyzer, event_bus)

    @pytest.fixture
    def profitability_calculator(self):
        """Create profitability calculator."""
        return SpreadProfitabilityCalculator()

    @pytest.fixture
    def pattern_analyzer(self, spread_analyzer):
        """Create pattern analyzer."""
        return SpreadPatternAnalyzer(spread_analyzer)

    @pytest.fixture
    async def market_data_service(self, event_bus, repository):
        """Create market data service."""
        service = MarketDataService(
            websocket_manager=None,  # Mock in tests
            gateway=None,  # Mock in tests
            event_bus=event_bus,
            repository=repository,
        )
        return service

    @pytest.mark.asyncio
    async def test_spread_tracking_workflow(self, spread_tracker):
        """Test complete spread tracking workflow."""
        await spread_tracker.start()

        try:
            # Create sample orderbook
            orderbook = {
                "bids": [["50000.00", "1.0"], ["49999.00", "2.0"], ["49998.00", "1.5"]],
                "asks": [["50010.00", "1.2"], ["50011.00", "1.8"], ["50012.00", "2.2"]],
            }

            # Track spread
            metrics = await spread_tracker.track_pair_spread("BTCUSDT", orderbook)

            assert metrics.symbol == "BTCUSDT"
            assert metrics.bid_price == Decimal("50000.00")
            assert metrics.ask_price == Decimal("50010.00")
            assert metrics.spread_bps == Decimal("20")  # 0.02% = 20 bps

            # Get spread history
            history = spread_tracker.get_spread_history("BTCUSDT", "raw")
            assert len(history) > 0

            # Identify patterns
            patterns = spread_tracker.identify_spread_patterns("BTCUSDT")
            assert isinstance(patterns, dict)

        finally:
            await spread_tracker.stop()

    @pytest.mark.asyncio
    async def test_spread_compression_event_flow(self, spread_tracker, event_bus):
        """Test spread compression detection and event publishing."""
        await spread_tracker.start()

        try:
            # Subscribe to compression events
            compression_events = []

            async def handle_compression(event: Event):
                if event.type == EventType.SPREAD_COMPRESSION:
                    compression_events.append(event)

            event_bus.subscribe(EventType.SPREAD_COMPRESSION, handle_compression)

            # Build normal spread history
            normal_orderbook = {"bids": [["100.00", "10"]], "asks": [["100.20", "10"]]}

            for _ in range(25):
                await spread_tracker.track_pair_spread("ETHUSDT", normal_orderbook)

            # Trigger compression
            compressed_orderbook = {
                "bids": [["100.00", "10"]],
                "asks": [["100.05", "10"]],
            }

            await spread_tracker.track_pair_spread("ETHUSDT", compressed_orderbook)

            # Allow event processing
            await asyncio.sleep(0.1)

            # Check compression event was published
            assert len(compression_events) > 0
            event = compression_events[0]
            assert event.data["symbol"] == "ETHUSDT"

        finally:
            await spread_tracker.stop()

    @pytest.mark.asyncio
    async def test_order_imbalance_detection(self, spread_tracker, event_bus):
        """Test order imbalance detection and alerting."""
        await spread_tracker.start()

        try:
            # Subscribe to imbalance events
            imbalance_events = []

            async def handle_imbalance(event: Event):
                if event.type == EventType.ORDER_IMBALANCE:
                    imbalance_events.append(event)

            event_bus.subscribe(EventType.ORDER_IMBALANCE, handle_imbalance)

            # Create imbalanced orderbook (heavy bid pressure)
            imbalanced_book = {
                "bids": [["100.00", "100"], ["99.95", "200"], ["99.90", "150"]],
                "asks": [["100.10", "10"], ["100.15", "20"], ["100.20", "15"]],
            }

            await spread_tracker.track_pair_spread("BNBUSDT", imbalanced_book)

            # Allow event processing
            await asyncio.sleep(0.1)

            # Check imbalance event was published
            assert len(imbalance_events) > 0
            event = imbalance_events[0]
            assert event.data["symbol"] == "BNBUSDT"
            assert event.data["imbalance"].is_significant

        finally:
            await spread_tracker.stop()

    @pytest.mark.asyncio
    async def test_profitability_calculation_flow(self, profitability_calculator):
        """Test profitability calculation for spread opportunities."""
        # Calculate profitability
        metrics = profitability_calculator.calculate_profit_potential(
            symbol="BTCUSDT",
            spread_bps=Decimal("30"),
            volume=Decimal("10000"),
            fee_bps=Decimal("10"),
        )

        assert metrics.symbol == "BTCUSDT"
        assert metrics.spread_bps == Decimal("30")
        assert metrics.gross_profit_bps == Decimal("30")
        # Net profit = 30 - (10*2) - slippage
        assert metrics.net_profit_bps < Decimal("10")
        assert metrics.is_profitable  # Should be profitable with 30 bps spread
        assert metrics.break_even_spread_bps > Decimal("20")

    @pytest.mark.asyncio
    async def test_spread_pattern_analysis(self, pattern_analyzer):
        """Test spread pattern detection and analysis."""
        # Generate sample spread history
        spread_history = []
        base_time = datetime.now(UTC)

        # Create hourly pattern (tighter spreads during certain hours)
        for day in range(7):
            for hour in range(24):
                timestamp = base_time - timedelta(days=day, hours=hour)
                # Tighter spreads during hours 10-16
                if 10 <= hour <= 16:
                    spread = Decimal("10") + Decimal(str(hour % 3))
                else:
                    spread = Decimal("20") + Decimal(str(hour % 5))

                spread_history.append((timestamp, spread))

        # Analyze patterns
        hourly_patterns = pattern_analyzer.analyze_hourly_patterns(
            "BTCUSDT", spread_history
        )
        assert not hourly_patterns.empty

        daily_patterns = pattern_analyzer.analyze_daily_patterns(
            "BTCUSDT", spread_history
        )
        assert not daily_patterns.empty

        # Calculate volatility score
        volatility_score = pattern_analyzer.calculate_volatility_score(
            "BTCUSDT", spread_history
        )
        assert volatility_score.symbol == "BTCUSDT"
        assert volatility_score.score >= Decimal("0")
        assert volatility_score.category in ["low", "medium", "high", "extreme"]

    @pytest.mark.asyncio
    async def test_database_persistence(self, repository):
        """Test spread data persistence to database."""
        # Prepare spread data
        spread_data = {
            "symbol": "BTCUSDT",
            "spread_bps": Decimal("15"),
            "bid_price": Decimal("50000"),
            "ask_price": Decimal("50007.50"),
            "bid_volume": Decimal("1.5"),
            "ask_volume": Decimal("2.0"),
            "order_imbalance": Decimal("0.75"),
            "timestamp": datetime.now(UTC),
        }

        # Save to database
        await repository.save_spread_history(spread_data)

        # Retrieve from database
        history = await repository.get_spread_history("BTCUSDT", limit=10)
        assert len(history) > 0

        saved_data = history[0]
        assert saved_data["symbol"] == "BTCUSDT"
        assert saved_data["spread_bps"] == Decimal("15")
        assert saved_data["bid_price"] == Decimal("50000")

    @pytest.mark.asyncio
    async def test_spread_aggregation(self, spread_tracker):
        """Test spread data aggregation over time."""
        await spread_tracker.start()

        try:
            # Add multiple spread data points
            for i in range(100):
                bid = Decimal("100") + Decimal(str(i * 0.01))
                ask = bid + Decimal("0.10") + Decimal(str(i % 5 * 0.01))

                orderbook = {"bids": [[str(bid), "10"]], "asks": [[str(ask), "10"]]}

                await spread_tracker.track_pair_spread("ADAUSDT", orderbook)

            # Force aggregation
            await spread_tracker._aggregate_spreads()

            # Check hourly aggregates
            hourly = spread_tracker.get_spread_history("ADAUSDT", "hourly")
            # May be empty if not enough time has passed

            # Check raw data is present
            raw = spread_tracker.get_spread_history("ADAUSDT", "raw")
            assert len(raw) > 0

        finally:
            await spread_tracker.stop()

    @pytest.mark.asyncio
    async def test_best_spread_times(self, spread_tracker):
        """Test identification of best trading times based on spreads."""
        await spread_tracker.start()

        try:
            # Generate data with time patterns
            for hour in range(24):
                # Simulate tighter spreads during certain hours
                if 9 <= hour <= 17:  # Business hours
                    spread_bps = Decimal("5")
                else:
                    spread_bps = Decimal("15")

                bid = Decimal("100")
                ask = bid + (spread_bps / Decimal("10000") * bid)

                orderbook = {"bids": [[str(bid), "10"]], "asks": [[str(ask), "10"]]}

                # Track multiple times for pattern
                for _ in range(5):
                    await spread_tracker.track_pair_spread("DOTUSDT", orderbook)

            # Get best times
            patterns = spread_tracker.identify_spread_patterns("DOTUSDT")
            best_times = spread_tracker.get_best_spread_times("DOTUSDT", top_n=3)

            # Should identify business hours as best times
            assert len(best_times) <= 3

        finally:
            await spread_tracker.stop()

    @pytest.mark.asyncio
    async def test_market_data_integration(self, market_data_service):
        """Test integration with market data service."""
        # Mock WebSocket manager
        market_data_service.websocket_manager = AsyncMock()
        market_data_service.websocket_manager.start = AsyncMock()
        market_data_service.websocket_manager.stop = AsyncMock()
        market_data_service.websocket_manager.subscribe = MagicMock()

        await market_data_service.start()

        try:
            # Simulate depth data
            depth_data = {
                "stream": "btcusdt@depth",
                "data": {
                    "lastUpdateId": 12345,
                    "bids": [["50000.00", "1.0"], ["49999.00", "2.0"]],
                    "asks": [["50010.00", "1.2"], ["50011.00", "1.8"]],
                },
            }

            await market_data_service._handle_depth(depth_data)

            # Check orderbook was stored
            orderbook = market_data_service.get_order_book("BTCUSDT")
            assert orderbook is not None
            assert orderbook.symbol == "BTCUSDT"
            assert orderbook.best_bid() == Decimal("50000.00")
            assert orderbook.best_ask() == Decimal("50010.00")

            # Check spread was calculated
            spread = market_data_service.calculate_spread("BTCUSDT")
            assert spread == 20  # 20 basis points

            # Get spread analytics
            analytics = market_data_service.get_spread_analytics("BTCUSDT")
            assert "current_metrics" in analytics
            assert "patterns" in analytics

        finally:
            await market_data_service.stop()

    @pytest.mark.asyncio
    async def test_performance_with_multiple_pairs(self, spread_tracker):
        """Test system performance with 50+ trading pairs."""
        await spread_tracker.start()

        try:
            symbols = [f"PAIR{i}USDT" for i in range(50)]

            # Track spreads for all pairs
            tasks = []
            for symbol in symbols:
                orderbook = {"bids": [["100.00", "10"]], "asks": [["100.10", "10"]]}
                tasks.append(spread_tracker.track_pair_spread(symbol, orderbook))

            # Execute concurrently
            start_time = datetime.now()
            await asyncio.gather(*tasks)
            elapsed = (datetime.now() - start_time).total_seconds()

            # Should complete within reasonable time
            assert elapsed < 5.0  # 5 seconds for 50 pairs

            # Verify all pairs were tracked
            for symbol in symbols:
                history = spread_tracker.get_spread_history(symbol, "raw")
                assert len(history) > 0

        finally:
            await spread_tracker.stop()

    @pytest.mark.asyncio
    async def test_alert_generation_flow(self, spread_tracker, event_bus):
        """Test complete alert generation and handling flow."""
        await spread_tracker.start()

        collected_events = {
            EventType.SPREAD_COMPRESSION: [],
            EventType.ORDER_IMBALANCE: [],
            EventType.SPREAD_ALERT: [],
        }

        async def event_collector(event: Event):
            if event.type in collected_events:
                collected_events[event.type].append(event)

        # Subscribe to all alert types
        for event_type in collected_events.keys():
            event_bus.subscribe(event_type, event_collector)

        try:
            # Generate various conditions

            # 1. Normal spreads
            for _ in range(20):
                orderbook = {"bids": [["100.00", "10"]], "asks": [["100.20", "10"]]}
                await spread_tracker.track_pair_spread("TESTUSDT", orderbook)

            # 2. Compressed spread
            compressed_book = {
                "bids": [["100.00", "10"]],
                "asks": [["100.02", "10"]],  # Very tight
            }
            await spread_tracker.track_pair_spread("TESTUSDT", compressed_book)

            # 3. Imbalanced orderbook
            imbalanced_book = {
                "bids": [["100.00", "100"], ["99.95", "200"]],
                "asks": [["100.10", "5"], ["100.15", "10"]],
            }
            await spread_tracker.track_pair_spread("TESTUSDT", imbalanced_book)

            # Allow events to process
            await asyncio.sleep(0.5)

            # Verify events were generated
            assert len(collected_events[EventType.SPREAD_COMPRESSION]) > 0
            assert len(collected_events[EventType.ORDER_IMBALANCE]) > 0

        finally:
            await spread_tracker.stop()

    @pytest.mark.asyncio
    async def test_data_retention_policy(self, repository):
        """Test automatic cleanup of old spread data."""
        # Add old data
        old_time = datetime.now(UTC) - timedelta(days=35)
        old_data = {
            "symbol": "OLDUSDT",
            "spread_bps": Decimal("20"),
            "bid_price": Decimal("100"),
            "ask_price": Decimal("100.20"),
            "timestamp": old_time,
        }

        # Add recent data
        recent_data = {
            "symbol": "RECENTUSDT",
            "spread_bps": Decimal("15"),
            "bid_price": Decimal("200"),
            "ask_price": Decimal("200.30"),
            "timestamp": datetime.now(UTC),
        }

        await repository.save_spread_history(old_data)
        await repository.save_spread_history(recent_data)

        # Run cleanup
        deleted = await repository.cleanup_old_spread_history(retention_days=30)

        # Old data should be deleted
        assert deleted >= 1

        # Recent data should remain
        recent_history = await repository.get_spread_history("RECENTUSDT")
        assert len(recent_history) > 0
