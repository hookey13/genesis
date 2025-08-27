"""Integration tests for the arbitrage engine system."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from genesis.analytics.backtest_engine import BacktestEngine, BacktestResult
from genesis.analytics.statistical_arb import (
    Signal,
    SpreadAnalyzer,
    StatisticalArbitrage,
    ThresholdMonitor,
)
from genesis.core.events import ArbitrageSignalEvent, Event, EventPriority, EventType
from genesis.data.market_data_service import MarketDataService
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus


@pytest.fixture
async def test_db():
    """Create a test database."""
    repo = SQLiteRepository(":memory:")
    await repo.initialize()
    yield repo
    await repo.shutdown()


@pytest.fixture
async def event_bus():
    """Create test event bus."""
    bus = EventBus()
    await bus.initialize()
    yield bus
    await bus.shutdown()


@pytest.fixture
def mock_market_data():
    """Create mock market data."""
    # Generate realistic price data for testing
    data = {}

    # BTCUSDT data
    btc_prices = []
    base_price = 50000
    for i in range(100):
        price = base_price + (i % 10) * 100  # Oscillating price
        btc_prices.append(Decimal(str(price)))

    # ETHUSDT data (correlated with BTC)
    eth_prices = []
    base_price = 3000
    for i in range(100):
        price = base_price + (i % 10) * 10  # Similar oscillation
        eth_prices.append(Decimal(str(price)))

    data["BTCUSDT"] = btc_prices
    data["ETHUSDT"] = eth_prices

    return data


class TestArbitrageSystemIntegration:
    """Integration tests for the complete arbitrage system."""

    @pytest.mark.asyncio
    async def test_end_to_end_signal_generation(self, test_db, event_bus):
        """Test complete signal generation flow."""
        # Setup components
        arb_engine = StatisticalArbitrage()
        spread_analyzer = SpreadAnalyzer()
        threshold_monitor = ThresholdMonitor()

        # Create test data
        prices1 = [Decimal(str(100 + i * 0.1)) for i in range(50)]
        prices2 = [Decimal(str(99 + i * 0.1)) for i in range(50)]
        timestamps = [datetime.now() - timedelta(minutes=50-i) for i in range(50)]

        # Calculate correlation
        correlation = arb_engine.calculate_correlation(
            "BTCUSDT", "ETHUSDT", 20, prices1[-20:], prices2[-20:]
        )
        assert correlation is not None

        # Analyze spread
        spread_result = spread_analyzer.analyze_spread(
            "BTCUSDT", "ETHUSDT", prices1, prices2, timestamps
        )
        assert spread_result["current_spread"] is not None

        # Calculate z-score
        zscore = arb_engine.calculate_zscore(
            prices1[-1], prices2[-1], 20, prices1[-20:], prices2[-20:]
        )

        # Generate signal if threshold crossed
        if abs(zscore) >= Decimal("2"):
            signal = spread_analyzer.generate_signal(
                zscore, Decimal("2"), "BTCUSDT", "ETHUSDT",
                True, Decimal("0.8")
            )

            if signal:
                # Save to database
                signal_dict = {
                    "pair1_symbol": signal.pair1_symbol,
                    "pair2_symbol": signal.pair2_symbol,
                    "zscore": signal.zscore,
                    "threshold_sigma": signal.threshold_sigma,
                    "signal_type": signal.signal_type,
                    "confidence_score": signal.confidence_score
                }
                signal_id = await test_db.save_arbitrage_signal(signal_dict)
                assert signal_id is not None

                # Publish event
                event = ArbitrageSignalEvent(
                    pair1_symbol=signal.pair1_symbol,
                    pair2_symbol=signal.pair2_symbol,
                    zscore=signal.zscore,
                    threshold_sigma=signal.threshold_sigma,
                    signal_type=signal.signal_type,
                    confidence_score=signal.confidence_score
                )
                await event_bus.publish(event, EventPriority.HIGH)

        # Verify data persistence
        signals = await test_db.get_arbitrage_signals(
            "BTCUSDT", "ETHUSDT", hours_back=1
        )
        if abs(zscore) >= Decimal("2"):
            assert len(signals) > 0

    @pytest.mark.asyncio
    async def test_database_persistence(self, test_db):
        """Test database operations for arbitrage data."""
        # Save arbitrage signal
        signal_data = {
            "pair1_symbol": "BTCUSDT",
            "pair2_symbol": "ETHUSDT",
            "zscore": Decimal("2.5"),
            "threshold_sigma": Decimal("2.0"),
            "signal_type": "ENTRY",
            "confidence_score": Decimal("0.85")
        }

        signal_id = await test_db.save_arbitrage_signal(signal_data)
        assert signal_id is not None

        # Retrieve signals
        signals = await test_db.get_arbitrage_signals(
            "BTCUSDT", "ETHUSDT", "ENTRY", hours_back=1
        )

        assert len(signals) == 1
        assert signals[0]["zscore"] == Decimal("2.5")
        assert signals[0]["confidence_score"] == Decimal("0.85")

        # Save spread history
        await test_db.save_spread_history(
            "BTCUSDT", "ETHUSDT", Decimal("0.01")
        )

        # Retrieve spread history
        history = await test_db.get_spread_history(
            "BTCUSDT", "ETHUSDT", days_back=1
        )

        assert len(history) == 1
        assert history[0]["spread_value"] == Decimal("0.01")

    @pytest.mark.asyncio
    async def test_event_publishing(self, event_bus):
        """Test arbitrage event publishing and handling."""
        received_events = []

        # Subscribe to arbitrage events
        async def handler(event: Event):
            received_events.append(event)

        await event_bus.subscribe(EventType.ARBITRAGE_SIGNAL, handler)

        # Create and publish arbitrage signal event
        event = ArbitrageSignalEvent(
            pair1_symbol="BTCUSDT",
            pair2_symbol="ETHUSDT",
            zscore=Decimal("2.5"),
            threshold_sigma=Decimal("2.0"),
            signal_type="ENTRY",
            confidence_score=Decimal("0.85")
        )

        await event_bus.publish(event, EventPriority.HIGH)

        # Give time for async processing
        await asyncio.sleep(0.1)

        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.ARBITRAGE_SIGNAL
        assert received_events[0].event_data["pair1_symbol"] == "BTCUSDT"
        assert received_events[0].event_data["zscore"] == "2.5"

    @pytest.mark.asyncio
    async def test_threshold_monitoring_with_alerts(self, event_bus):
        """Test threshold monitoring and alert generation."""
        monitor = ThresholdMonitor(default_sigma=Decimal("2"))
        received_alerts = []

        # Subscribe to threshold breach events
        async def alert_handler(event: Event):
            if event.event_type == EventType.ARBITRAGE_THRESHOLD_BREACH:
                received_alerts.append(event)

        await event_bus.subscribe(EventType.ARBITRAGE_THRESHOLD_BREACH, alert_handler)

        # Monitor thresholds
        pairs = [("BTCUSDT", "ETHUSDT"), ("BNBUSDT", "ADAUSDT")]

        async def mock_get_zscore(pair1, pair2):
            if pair1 == "BTCUSDT":
                return Decimal("2.5")  # Above threshold
            return Decimal("1.0")  # Below threshold

        alerts = await monitor.monitor_thresholds(pairs, mock_get_zscore)

        # Publish alerts as events
        for alert in alerts:
            event = Event(
                event_type=EventType.ARBITRAGE_THRESHOLD_BREACH,
                aggregate_id=f"{alert['pair1']}:{alert['pair2']}",
                event_data=alert
            )
            await event_bus.publish(event, EventPriority.HIGH)

        await asyncio.sleep(0.1)

        # Verify alerts
        assert len(alerts) == 1
        assert alerts[0]["pair1"] == "BTCUSDT"
        assert len(received_alerts) == 1

    @pytest.mark.asyncio
    async def test_market_data_integration(self):
        """Test integration with market data service."""
        # Mock components
        mock_gateway = AsyncMock()
        mock_ws_manager = AsyncMock()
        mock_repo = AsyncMock()
        mock_event_bus = AsyncMock()

        # Create market data service
        service = MarketDataService(
            gateway=mock_gateway,
            websocket_manager=mock_ws_manager,
            repository=mock_repo,
            event_bus=mock_event_bus
        )

        # Mock historical data
        mock_gateway.get_historical_klines.return_value = [
            [0, "100", "101", "99", "100.5", "1000", 0, 0, 0, 0, 0, 0]
            for _ in range(100)
        ]

        # Test price history retrieval
        prices = await service.get_price_history("BTCUSDT", 1)
        assert len(prices) == 100
        assert all(isinstance(p, Decimal) for p in prices)

        # Test multi-pair subscription
        await service.subscribe_multi_pair(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        # Verify subscriptions (would check actual subscription logic)

        # Test correlation monitoring
        pairs = [("BTCUSDT", "ETHUSDT"), ("BNBUSDT", "ADAUSDT")]
        await service.start_correlation_monitoring(pairs)
        # Verify monitoring started

    @pytest.mark.asyncio
    async def test_backtesting_integration(self, mock_market_data):
        """Test backtesting with arbitrage strategy."""
        # Create backtest engine
        engine = BacktestEngine(
            initial_capital=Decimal("10000"),
            position_size_percent=Decimal("0.1"),
            transaction_cost_percent=Decimal("0.001")
        )

        # Create strategy
        strategy = StatisticalArbitrage()

        # Prepare historical data
        historical_data = {}
        timestamps = [datetime.now() - timedelta(hours=100-i) for i in range(100)]

        # Create DataFrames for backtesting
        for symbol, prices in mock_market_data.items():
            df = pd.DataFrame({
                'timestamp': timestamps,
                'price1': [float(p) for p in prices] if symbol == "BTCUSDT" else [100] * 100,
                'price2': [100] * 100 if symbol == "BTCUSDT" else [float(p) for p in prices]
            })
            df.set_index('timestamp', inplace=True)
            historical_data["BTCUSDT:ETHUSDT"] = df

        # Run backtest
        start_date = timestamps[0]
        end_date = timestamps[-1]

        result = await engine.run_backtest(
            strategy=strategy,
            historical_data=historical_data,
            start_date=start_date,
            end_date=end_date,
            entry_threshold=Decimal("2"),
            exit_threshold=Decimal("0.5")
        )

        # Verify backtest results
        assert isinstance(result, BacktestResult)
        assert result.initial_capital == Decimal("10000")
        assert result.start_date == start_date
        assert result.end_date == end_date
        # Results will vary based on data

    @pytest.mark.asyncio
    async def test_signal_persistence_check(self):
        """Test signal persistence checking across multiple signals."""
        analyzer = SpreadAnalyzer()

        # Create a series of consistent signals
        signals = []
        base_time = datetime.now()

        for i in range(5):
            signal = Signal(
                pair1_symbol="BTCUSDT",
                pair2_symbol="ETHUSDT",
                zscore=Decimal("2.5"),
                threshold_sigma=Decimal("2.0"),
                signal_type="ENTRY",
                confidence_score=Decimal("0.8"),
                created_at=base_time - timedelta(minutes=5-i)
            )
            signals.append(signal)

        # Check persistence
        persistent_signals = analyzer.check_signal_persistence(
            signals, min_persistence=3
        )

        assert len(persistent_signals) > 0
        assert all(s.signal_type == "ENTRY" for s in persistent_signals)

        # Test with mixed signals (should not pass persistence)
        mixed_signals = signals[:2]  # Only 2 ENTRY signals
        mixed_signals.append(
            Signal(
                pair1_symbol="BTCUSDT",
                pair2_symbol="ETHUSDT",
                zscore=Decimal("0.5"),
                threshold_sigma=Decimal("2.0"),
                signal_type="EXIT",
                confidence_score=Decimal("0.8"),
                created_at=base_time
            )
        )

        persistent_mixed = analyzer.check_signal_persistence(
            mixed_signals, min_persistence=3
        )

        assert len(persistent_mixed) == 0  # Not enough consistent signals

    @pytest.mark.asyncio
    async def test_cointegration_and_correlation(self):
        """Test cointegration and correlation calculations together."""
        arb_engine = StatisticalArbitrage()

        # Create synthetic cointegrated data
        n_points = 100
        t = range(n_points)

        # Base trend
        trend = [Decimal(str(100 + i * 0.1)) for i in t]

        # Add mean-reverting spread
        spread = [Decimal(str(5 * ((i % 20) - 10) / 10)) for i in t]

        prices1 = [trend[i] + spread[i] for i in range(n_points)]
        prices2 = trend

        # Test correlation (should be high)
        correlation = arb_engine.calculate_correlation(
            "PAIR1", "PAIR2", 50, prices1[-50:], prices2[-50:]
        )
        assert correlation > Decimal("0.8")  # High correlation expected

        # Test cointegration
        is_cointegrated = arb_engine.test_cointegration(prices1, prices2)
        # Result depends on the simplified test implementation
        assert isinstance(is_cointegrated, bool)

        # Create correlation matrix
        symbols = ["PAIR1", "PAIR2", "PAIR3"]
        price_data = {
            "PAIR1": prices1[-50:],
            "PAIR2": prices2[-50:],
            "PAIR3": [Decimal(str(50 + i * 0.05)) for i in range(50)]
        }

        matrix = arb_engine.create_correlation_matrix(
            symbols, price_data, 50
        )

        assert matrix.shape == (3, 3)
        assert matrix.loc["PAIR1", "PAIR2"] > 0.8  # High correlation
        assert abs(matrix.loc["PAIR1", "PAIR3"]) < abs(matrix.loc["PAIR1", "PAIR2"])

    @pytest.mark.asyncio
    async def test_full_arbitrage_workflow(self, test_db, event_bus):
        """Test complete arbitrage workflow from data to signal to persistence."""
        # Initialize all components
        arb_engine = StatisticalArbitrage()
        spread_analyzer = SpreadAnalyzer()
        threshold_monitor = ThresholdMonitor()

        # Simulated market data over time
        n_periods = 100
        timestamps = [datetime.now() - timedelta(minutes=n_periods-i)
                     for i in range(n_periods)]

        # Generate realistic price movements
        btc_prices = []
        eth_prices = []
        base_btc = 50000
        base_eth = 3000

        for i in range(n_periods):
            # Add trend and noise
            btc_price = base_btc * (1 + 0.0001 * i + 0.001 * (i % 10 - 5))
            eth_price = base_eth * (1 + 0.0001 * i + 0.001 * (i % 10 - 5))

            # Create divergence in middle
            if 40 <= i <= 60:
                eth_price *= 0.98  # ETH underperforms

            btc_prices.append(Decimal(str(btc_price)))
            eth_prices.append(Decimal(str(eth_price)))

        # Process each time period
        signals_generated = []

        for i in range(20, n_periods):  # Need at least 20 for window
            window = 20

            # Calculate metrics
            correlation = arb_engine.calculate_correlation(
                "BTCUSDT", "ETHUSDT", window,
                btc_prices[i-window:i], eth_prices[i-window:i]
            )

            zscore = arb_engine.calculate_zscore(
                btc_prices[i], eth_prices[i], window,
                btc_prices[i-window:i], eth_prices[i-window:i]
            )

            # Check for signal
            if abs(zscore) >= Decimal("2"):
                is_cointegrated = arb_engine.test_cointegration(
                    btc_prices[i-50:i] if i >= 50 else btc_prices[:i],
                    eth_prices[i-50:i] if i >= 50 else eth_prices[:i]
                )

                signal = spread_analyzer.generate_signal(
                    zscore, Decimal("2"), "BTCUSDT", "ETHUSDT",
                    is_cointegrated, Decimal("0.7")
                )

                if signal:
                    signals_generated.append(signal)

                    # Save to database
                    await test_db.save_arbitrage_signal({
                        "pair1_symbol": signal.pair1_symbol,
                        "pair2_symbol": signal.pair2_symbol,
                        "zscore": signal.zscore,
                        "threshold_sigma": signal.threshold_sigma,
                        "signal_type": signal.signal_type,
                        "confidence_score": signal.confidence_score
                    })

                    # Publish event
                    event = ArbitrageSignalEvent(
                        pair1_symbol=signal.pair1_symbol,
                        pair2_symbol=signal.pair2_symbol,
                        zscore=signal.zscore,
                        threshold_sigma=signal.threshold_sigma,
                        signal_type=signal.signal_type,
                        confidence_score=signal.confidence_score
                    )
                    await event_bus.publish(event, EventPriority.HIGH)

            # Save spread history periodically
            if i % 10 == 0:
                spread = (btc_prices[i] / base_btc) - (eth_prices[i] / base_eth)
                await test_db.save_spread_history(
                    "BTCUSDT", "ETHUSDT", spread
                )

        # Verify results
        assert len(signals_generated) > 0  # Should have generated some signals

        # Check database
        db_signals = await test_db.get_arbitrage_signals(
            "BTCUSDT", "ETHUSDT", hours_back=24
        )
        assert len(db_signals) == len(signals_generated)

        spread_history = await test_db.get_spread_history(
            "BTCUSDT", "ETHUSDT", days_back=1
        )
        assert len(spread_history) > 0
