"""
Integration tests for market data streams.

Tests the full market data pipeline including WebSocket streams,
data aggregation, and event publishing.
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from genesis.core.events import EventType, EventPriority
from genesis.data.market_data_service import MarketDataService, MarketState
from genesis.engine.event_bus import EventBus
from genesis.exchange.websocket_manager import WebSocketManager


@pytest_asyncio.fixture
async def event_bus():
    """Create event bus instance."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest_asyncio.fixture
async def market_data_service(event_bus):
    """Create market data service with real event bus."""
    with patch("genesis.exchange.websocket_manager.get_settings") as mock_settings:
        mock_settings.return_value.trading.trading_pairs = ["BTC/USDT"]
        mock_settings.return_value.exchange.binance_testnet = True
        
        service = MarketDataService(event_bus=event_bus)
        yield service
        await service.stop()


class TestMarketDataIntegration:
    """Test market data integration."""
    
    @pytest.mark.asyncio
    async def test_full_market_data_pipeline(self, market_data_service, event_bus):
        """Test complete market data pipeline."""
        received_events = []
        
        # Subscribe to market data events
        def event_handler(event):
            received_events.append(event)
        
        event_bus.subscribe(
            callback=event_handler,
            event_types={
                EventType.MARKET_DATA_UPDATED,
                EventType.SPREAD_ALERT,
                EventType.VOLUME_ANOMALY
            }
        )
        
        # Start service
        with patch.object(market_data_service.websocket_manager, "start", new_callable=AsyncMock):
            await market_data_service.start()
        
        # Simulate trade data
        trade_data = {
            "stream": "btcusdt@trade",
            "data": {
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "0.1",
                "T": int(datetime.now().timestamp() * 1000),
                "m": False
            }
        }
        
        await market_data_service._handle_trade(trade_data)
        
        # Wait for event processing
        await asyncio.sleep(0.2)
        
        # Check event was received
        assert len(received_events) > 0
        event = received_events[0]
        assert event.event_type == EventType.MARKET_DATA_UPDATED
        assert event.aggregate_id == "BTCUSDT"
        assert event.event_data["price"] == "50000.00"
    
    @pytest.mark.asyncio
    async def test_spread_alert_generation(self, market_data_service, event_bus):
        """Test spread alert generation for tight spreads."""
        received_alerts = []
        
        # Subscribe to spread alerts
        def alert_handler(event):
            if event.event_type == EventType.SPREAD_ALERT:
                received_alerts.append(event)
        
        event_bus.subscribe(
            callback=alert_handler,
            event_types={EventType.SPREAD_ALERT}
        )
        
        # Simulate tight spread
        depth_data = {
            "stream": "btcusdt@depth20",
            "data": {
                "lastUpdateId": 12345,
                "bids": [["49998.00", "10.0"]],  # Very tight spread
                "asks": [["50000.00", "10.0"]]
            }
        }
        
        await market_data_service._handle_depth(depth_data)
        
        # Wait for event processing
        await asyncio.sleep(0.2)
        
        # Check alert was generated
        assert len(received_alerts) > 0
        alert = received_alerts[0]
        assert alert.event_type == EventType.SPREAD_ALERT
        assert alert.aggregate_id == "BTCUSDT"
        assert alert.event_data["spread_bp"] < 10
    
    @pytest.mark.asyncio
    async def test_volume_anomaly_detection(self, market_data_service, event_bus):
        """Test volume anomaly detection."""
        received_anomalies = []
        
        # Subscribe to volume anomalies
        def anomaly_handler(event):
            if event.event_type == EventType.VOLUME_ANOMALY:
                received_anomalies.append(event)
        
        event_bus.subscribe(
            callback=anomaly_handler,
            event_types={EventType.VOLUME_ANOMALY}
        )
        
        # Set up normal volume profile
        symbol = "BTCUSDT"
        profile = market_data_service.volume_profiles.get(symbol)
        if not profile:
            from genesis.data.market_data_service import VolumeProfile
            profile = VolumeProfile(symbol=symbol)
            market_data_service.volume_profiles[symbol] = profile
        
        # Add normal volumes
        for hour in range(10):
            profile.add_volume(hour, Decimal("1000"))
        
        # Trigger anomaly check with high volume
        current_hour = datetime.now().hour
        profile.add_volume(current_hour, Decimal("5000"))  # 5x normal
        
        # Run volume analysis task once
        await market_data_service._volume_analysis_task()
        
        # Wait for event processing
        await asyncio.sleep(0.2)
        
        # Check anomaly was detected
        assert len(received_anomalies) > 0
        anomaly = received_anomalies[0]
        assert anomaly.event_type == EventType.VOLUME_ANOMALY
        assert anomaly.aggregate_id == "BTCUSDT"
    
    @pytest.mark.asyncio
    async def test_candle_aggregation_from_trades(self, market_data_service):
        """Test candle aggregation from trade ticks."""
        # Simulate multiple trades in same minute
        base_time = int(datetime.now().timestamp() * 1000)
        
        trades = [
            {"p": "50000.00", "q": "0.1"},
            {"p": "50010.00", "q": "0.2"},
            {"p": "49990.00", "q": "0.15"},
            {"p": "50005.00", "q": "0.25"},
        ]
        
        for i, trade in enumerate(trades):
            trade_data = {
                "stream": "btcusdt@trade",
                "data": {
                    "s": "BTCUSDT",
                    "p": trade["p"],
                    "q": trade["q"],
                    "T": base_time + i * 100,
                    "m": False
                }
            }
            await market_data_service._handle_trade(trade_data)
        
        # Check aggregator
        agg = market_data_service.candle_aggregators.get("BTCUSDT")
        assert agg is not None
        assert agg["open"] == Decimal("50000.00")
        assert agg["high"] == Decimal("50010.00")
        assert agg["low"] == Decimal("49990.00")
        assert agg["close"] == Decimal("50005.00")
        assert agg["volume"] == Decimal("0.70")
        assert agg["trades"] == 4
    
    @pytest.mark.asyncio
    async def test_market_state_classification(self, market_data_service):
        """Test market state classification based on data."""
        symbol = "BTCUSDT"
        
        # Add candles with normal volatility
        from genesis.data.market_data_service import Candle
        for i in range(25):
            candle = Candle(
                symbol=symbol,
                open=Decimal("50000"),
                high=Decimal("50100"),
                low=Decimal("49900"),
                close=Decimal("50050"),
                volume=Decimal("1000"),
                timestamp=datetime.now()
            )
            market_data_service.candles[symbol].append(candle)
        
        # Add normal volume profile
        from genesis.data.market_data_service import VolumeProfile
        profile = VolumeProfile(symbol=symbol)
        profile.rolling_24h_volume = Decimal("100000")
        market_data_service.volume_profiles[symbol] = profile
        
        # Test normal state
        state = market_data_service.classify_market_state(symbol)
        assert state == MarketState.NORMAL
        
        # Add high volatility candles
        for i in range(5):
            candle = Candle(
                symbol=symbol,
                open=Decimal("50000"),
                high=Decimal("52000"),  # High volatility
                low=Decimal("48000"),
                close=Decimal("51000"),
                volume=Decimal("5000"),
                timestamp=datetime.now()
            )
            market_data_service.candles[symbol].append(candle)
        
        # Test volatile state
        state = market_data_service.classify_market_state(symbol)
        assert state in [MarketState.VOLATILE, MarketState.PANIC]
    
    @pytest.mark.asyncio
    async def test_concurrent_stream_handling(self, market_data_service):
        """Test handling multiple concurrent streams."""
        # Simulate concurrent data from multiple streams
        tasks = []
        
        # Trade stream
        trade_data = {
            "stream": "btcusdt@trade",
            "data": {
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "0.1",
                "T": int(datetime.now().timestamp() * 1000),
                "m": False
            }
        }
        tasks.append(market_data_service._handle_trade(trade_data))
        
        # Depth stream
        depth_data = {
            "stream": "btcusdt@depth20",
            "data": {
                "lastUpdateId": 12345,
                "bids": [["49900.00", "1.0"]],
                "asks": [["50000.00", "0.5"]]
            }
        }
        tasks.append(market_data_service._handle_depth(depth_data))
        
        # Kline stream
        kline_data = {
            "data": {
                "k": {
                    "s": "BTCUSDT",
                    "o": "50000.00",
                    "h": "50100.00",
                    "l": "49900.00",
                    "c": "50050.00",
                    "v": "100.0",
                    "t": int(datetime.now().timestamp() * 1000),
                    "n": 500,
                    "x": True
                }
            }
        }
        tasks.append(market_data_service._handle_kline(kline_data))
        
        # Ticker stream
        ticker_data = {
            "stream": "btcusdt@ticker",
            "data": {
                "s": "BTCUSDT",
                "v": "10000.0"
            }
        }
        tasks.append(market_data_service._handle_ticker(ticker_data))
        
        # Process all concurrently
        await asyncio.gather(*tasks)
        
        # Verify all data was processed
        assert "BTCUSDT" in market_data_service.current_prices
        assert "BTCUSDT" in market_data_service.order_books
        assert len(market_data_service.candles["BTCUSDT"]) > 0
        assert "BTCUSDT" in market_data_service.volume_profiles
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, market_data_service):
        """Test memory-efficient circular buffers."""
        symbol = "BTCUSDT"
        
        # Add more than maxlen ticks
        for i in range(1500):
            from genesis.data.market_data_service import Tick
            tick = Tick(
                symbol=symbol,
                price=Decimal(f"{50000 + i}"),
                quantity=Decimal("0.1"),
                timestamp=float(i),
                is_buyer_maker=True
            )
            market_data_service.ticks[symbol].append(tick)
        
        # Check buffer size is limited
        assert len(market_data_service.ticks[symbol]) <= 1000
        
        # Add more than maxlen candles
        from genesis.data.market_data_service import Candle
        for i in range(1500):
            candle = Candle(
                symbol=symbol,
                open=Decimal("50000"),
                high=Decimal("50100"),
                low=Decimal("49900"),
                close=Decimal("50050"),
                volume=Decimal("100"),
                timestamp=datetime.now()
            )
            market_data_service.candles[symbol].append(candle)
        
        # Check buffer size is limited
        assert len(market_data_service.candles[symbol]) <= 1000
    
    @pytest.mark.asyncio
    async def test_event_priority_handling(self, event_bus):
        """Test event priority processing."""
        received_order = []
        
        def handler(event):
            received_order.append(event.event_data["priority"])
        
        event_bus.subscribe(callback=handler)
        
        # Publish events with different priorities
        from genesis.core.events import Event
        
        # Publish in reverse priority order
        event_low = Event(event_data={"priority": "low"})
        await event_bus.publish(event_low, EventPriority.LOW)
        
        event_normal = Event(event_data={"priority": "normal"})
        await event_bus.publish(event_normal, EventPriority.NORMAL)
        
        event_high = Event(event_data={"priority": "high"})
        await event_bus.publish(event_high, EventPriority.HIGH)
        
        event_critical = Event(event_data={"priority": "critical"})
        await event_bus.publish(event_critical, EventPriority.CRITICAL)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Check critical and high priority were processed first
        assert received_order[0] == "critical"
        assert received_order[1] == "high"
    
    @pytest.mark.asyncio
    async def test_subscription_lifecycle(self, market_data_service):
        """Test market data subscription lifecycle."""
        symbol = "BTCUSDT"
        received_ticks = []
        
        # Add initial tick
        from genesis.data.market_data_service import Tick
        tick1 = Tick(
            symbol=symbol,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            timestamp=1234567890.0,
            is_buyer_maker=True
        )
        market_data_service.ticks[symbol].append(tick1)
        
        # Subscribe
        subscription = market_data_service.subscribe_market_data(symbol)
        
        # Get first tick
        first_tick = await anext(subscription)
        assert first_tick.price == Decimal("50000")
        
        # Add another tick
        tick2 = Tick(
            symbol=symbol,
            price=Decimal("50100"),
            quantity=Decimal("0.2"),
            timestamp=1234567891.0,
            is_buyer_maker=False
        )
        market_data_service.ticks[symbol].append(tick2)
        
        # Unsubscribe
        market_data_service.active_subscriptions.remove(symbol)
        
        # Verify subscription was removed
        assert symbol not in market_data_service.active_subscriptions