"""
Unit tests for Market Data Service.

Tests market data ingestion, order book management, spread calculations,
and volume profile analysis.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.core.events import Event, EventPriority, EventType
from genesis.data.market_data_service import (
    Candle,
    MarketDataService,
    MarketState,
    OrderBook,
    OrderBookLevel,
    Tick,
    VolumeProfile,
)


@pytest.fixture
def mock_websocket_manager():
    """Create mock WebSocket manager."""
    manager = AsyncMock()
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.subscribe = MagicMock()
    manager.get_statistics = MagicMock(return_value={})
    return manager


@pytest.fixture
def mock_gateway():
    """Create mock gateway."""
    gateway = AsyncMock()
    gateway.get_recent_trades = AsyncMock(return_value=[])
    gateway.get_order_book = AsyncMock(return_value={})
    return gateway


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def market_data_service(mock_websocket_manager, mock_gateway, mock_event_bus):
    """Create market data service instance."""
    service = MarketDataService(
        websocket_manager=mock_websocket_manager,
        gateway=mock_gateway,
        event_bus=mock_event_bus
    )
    return service


class TestTick:
    """Test Tick dataclass."""
    
    def test_tick_creation(self):
        """Test creating a tick."""
        tick = Tick(
            symbol="BTCUSDT",
            price=Decimal("50000.00"),
            quantity=Decimal("0.1"),
            timestamp=1234567890.0,
            is_buyer_maker=True
        )
        
        assert tick.symbol == "BTCUSDT"
        assert tick.price == Decimal("50000.00")
        assert tick.quantity == Decimal("0.1")
        assert tick.timestamp == 1234567890.0
        assert tick.is_buyer_maker is True


class TestOrderBook:
    """Test OrderBook functionality."""
    
    def test_order_book_creation(self):
        """Test creating an order book."""
        book = OrderBook(symbol="BTCUSDT")
        
        # Add bids
        book.bids = [
            OrderBookLevel(Decimal("49900"), Decimal("1.0")),
            OrderBookLevel(Decimal("49800"), Decimal("2.0")),
        ]
        
        # Add asks
        book.asks = [
            OrderBookLevel(Decimal("50000"), Decimal("0.5")),
            OrderBookLevel(Decimal("50100"), Decimal("1.5")),
        ]
        
        assert book.symbol == "BTCUSDT"
        assert len(book.bids) == 2
        assert len(book.asks) == 2
    
    def test_best_bid_ask(self):
        """Test getting best bid and ask."""
        book = OrderBook(symbol="BTCUSDT")
        
        book.bids = [
            OrderBookLevel(Decimal("49900"), Decimal("1.0")),
            OrderBookLevel(Decimal("49800"), Decimal("2.0")),
        ]
        book.asks = [
            OrderBookLevel(Decimal("50000"), Decimal("0.5")),
            OrderBookLevel(Decimal("50100"), Decimal("1.5")),
        ]
        
        assert book.best_bid() == Decimal("49900")
        assert book.best_ask() == Decimal("50000")
    
    def test_spread_calculation(self):
        """Test spread calculations."""
        book = OrderBook(symbol="BTCUSDT")
        
        book.bids = [OrderBookLevel(Decimal("49900"), Decimal("1.0"))]
        book.asks = [OrderBookLevel(Decimal("50000"), Decimal("0.5"))]
        
        assert book.spread() == Decimal("100")
        assert book.spread_basis_points() == 20  # (100 / 49950) * 10000
        assert book.mid_price() == Decimal("49950")
    
    def test_empty_order_book(self):
        """Test empty order book handling."""
        book = OrderBook(symbol="BTCUSDT")
        
        assert book.best_bid() is None
        assert book.best_ask() is None
        assert book.spread() is None
        assert book.spread_basis_points() is None
        assert book.mid_price() is None


class TestVolumeProfile:
    """Test VolumeProfile functionality."""
    
    def test_volume_profile_creation(self):
        """Test creating a volume profile."""
        profile = VolumeProfile(symbol="BTCUSDT")
        
        assert profile.symbol == "BTCUSDT"
        assert profile.rolling_24h_volume == Decimal("0")
        assert profile.average_hourly_volume == Decimal("0")
        assert len(profile.hour_volumes) == 0
    
    def test_add_volume(self):
        """Test adding volume data."""
        profile = VolumeProfile(symbol="BTCUSDT")
        
        profile.add_volume(10, Decimal("1000"))
        profile.add_volume(10, Decimal("500"))
        profile.add_volume(11, Decimal("2000"))
        
        assert profile.hour_volumes[10] == Decimal("1500")
        assert profile.hour_volumes[11] == Decimal("2000")
        assert profile.rolling_24h_volume == Decimal("3500")
        assert profile.average_hourly_volume == Decimal("1750")
    
    def test_volume_anomaly_detection(self):
        """Test volume anomaly detection."""
        profile = VolumeProfile(symbol="BTCUSDT")
        
        # Set up normal volumes
        for hour in range(10):
            profile.add_volume(hour, Decimal("1000"))
        
        # Test normal volume
        assert profile.is_volume_anomaly(Decimal("1200")) is False
        
        # Test high volume anomaly
        assert profile.is_volume_anomaly(Decimal("3000")) is True
        
        # Test low volume anomaly
        assert profile.is_volume_anomaly(Decimal("300")) is True


class TestMarketDataService:
    """Test MarketDataService functionality."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, market_data_service):
        """Test service initialization."""
        assert market_data_service is not None
        assert len(market_data_service.current_prices) == 0
        assert len(market_data_service.order_books) == 0
        assert len(market_data_service.active_subscriptions) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop(self, market_data_service):
        """Test starting and stopping the service."""
        await market_data_service.start()
        
        # Verify WebSocket manager was started
        market_data_service.websocket_manager.start.assert_called_once()
        
        # Verify subscriptions were set up
        assert market_data_service.websocket_manager.subscribe.call_count == 4
        
        await market_data_service.stop()
        market_data_service.websocket_manager.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_current_price(self, market_data_service):
        """Test getting current price."""
        # Set a price
        market_data_service.current_prices["BTCUSDT"] = Decimal("50000")
        
        price = market_data_service.get_current_price("BTCUSDT")
        assert price == Decimal("50000")
        
        # Test unknown symbol
        price = market_data_service.get_current_price("UNKNOWN")
        assert price is None
    
    @pytest.mark.asyncio
    async def test_get_order_book(self, market_data_service):
        """Test getting order book."""
        # Create order book
        book = OrderBook(symbol="BTCUSDT")
        book.bids = [
            OrderBookLevel(Decimal("49900"), Decimal("1.0")),
            OrderBookLevel(Decimal("49800"), Decimal("2.0")),
            OrderBookLevel(Decimal("49700"), Decimal("3.0")),
        ]
        book.asks = [
            OrderBookLevel(Decimal("50000"), Decimal("0.5")),
            OrderBookLevel(Decimal("50100"), Decimal("1.5")),
            OrderBookLevel(Decimal("50200"), Decimal("2.5")),
        ]
        
        market_data_service.order_books["BTCUSDT"] = book
        
        # Get full book
        retrieved_book = market_data_service.get_order_book("BTCUSDT")
        assert retrieved_book is not None
        assert len(retrieved_book.bids) == 3
        assert len(retrieved_book.asks) == 3
        
        # Get limited depth
        limited_book = market_data_service.get_order_book("BTCUSDT", depth=2)
        assert limited_book is not None
        assert len(limited_book.bids) == 2
        assert len(limited_book.asks) == 2
    
    @pytest.mark.asyncio
    async def test_calculate_spread(self, market_data_service):
        """Test spread calculation."""
        # Create order book
        book = OrderBook(symbol="BTCUSDT")
        book.bids = [OrderBookLevel(Decimal("49900"), Decimal("1.0"))]
        book.asks = [OrderBookLevel(Decimal("50000"), Decimal("0.5"))]
        
        market_data_service.order_books["BTCUSDT"] = book
        
        spread = market_data_service.calculate_spread("BTCUSDT")
        assert spread == 20  # Basis points
        
        # Test unknown symbol
        spread = market_data_service.calculate_spread("UNKNOWN")
        assert spread is None
    
    @pytest.mark.asyncio
    async def test_classify_market_state(self, market_data_service):
        """Test market state classification."""
        # Test with no data
        state = market_data_service.classify_market_state("BTCUSDT")
        assert state == MarketState.DEAD
        
        # Add some candles
        for i in range(25):
            candle = Candle(
                symbol="BTCUSDT",
                open=Decimal("50000"),
                high=Decimal("50100"),
                low=Decimal("49900"),
                close=Decimal("50050"),
                volume=Decimal("100"),
                timestamp=datetime.now() - timedelta(minutes=i)
            )
            market_data_service.candles["BTCUSDT"].append(candle)
        
        # Add volume profile
        profile = VolumeProfile(symbol="BTCUSDT")
        profile.rolling_24h_volume = Decimal("50000")
        market_data_service.volume_profiles["BTCUSDT"] = profile
        
        state = market_data_service.classify_market_state("BTCUSDT")
        assert state == MarketState.NORMAL
    
    @pytest.mark.asyncio
    async def test_handle_trade(self, market_data_service):
        """Test handling trade data."""
        trade_data = {
            "data": {
                "s": "BTCUSDT",
                "p": "50000.00",
                "q": "0.1",
                "T": 1234567890000,
                "m": False
            }
        }
        
        await market_data_service._handle_trade(trade_data)
        
        # Check price was updated
        assert market_data_service.current_prices["BTCUSDT"] == Decimal("50000.00")
        
        # Check tick was stored
        assert len(market_data_service.ticks["BTCUSDT"]) == 1
        tick = market_data_service.ticks["BTCUSDT"][0]
        assert tick.price == Decimal("50000.00")
        assert tick.quantity == Decimal("0.1")
        
        # Check event was published
        market_data_service.event_bus.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_handle_depth(self, market_data_service):
        """Test handling depth data."""
        depth_data = {
            "stream": "btcusdt@depth20",
            "data": {
                "lastUpdateId": 12345,
                "bids": [
                    ["49900.00", "1.0"],
                    ["49800.00", "2.0"]
                ],
                "asks": [
                    ["50000.00", "0.5"],
                    ["50100.00", "1.5"]
                ]
            }
        }
        
        await market_data_service._handle_depth(depth_data)
        
        # Check order book was updated
        book = market_data_service.order_books.get("BTCUSDT")
        assert book is not None
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.best_bid() == Decimal("49900.00")
        assert book.best_ask() == Decimal("50000.00")
        
        # Check spread history
        assert len(market_data_service.spread_history["BTCUSDT"]) == 1
    
    @pytest.mark.asyncio
    async def test_handle_kline(self, market_data_service):
        """Test handling kline data."""
        kline_data = {
            "data": {
                "k": {
                    "s": "BTCUSDT",
                    "o": "50000.00",
                    "h": "50100.00",
                    "l": "49900.00",
                    "c": "50050.00",
                    "v": "100.0",
                    "t": 1234567890000,
                    "n": 500,
                    "x": True  # Closed candle
                }
            }
        }
        
        await market_data_service._handle_kline(kline_data)
        
        # Check candle was stored
        assert len(market_data_service.candles["BTCUSDT"]) == 1
        candle = market_data_service.candles["BTCUSDT"][0]
        assert candle.open == Decimal("50000.00")
        assert candle.high == Decimal("50100.00")
        assert candle.low == Decimal("49900.00")
        assert candle.close == Decimal("50050.00")
        assert candle.volume == Decimal("100.0")
        
        # Check volume profile was updated
        profile = market_data_service.volume_profiles.get("BTCUSDT")
        assert profile is not None
    
    @pytest.mark.asyncio
    async def test_subscribe_market_data(self, market_data_service):
        """Test subscribing to market data."""
        # Add some ticks
        tick1 = Tick(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            timestamp=1234567890.0,
            is_buyer_maker=True
        )
        market_data_service.ticks["BTCUSDT"].append(tick1)
        
        # Subscribe
        subscription = market_data_service.subscribe_market_data("BTCUSDT")
        
        # Get first tick
        tick = await anext(subscription)
        assert tick.price == Decimal("50000")
        
        # Check subscription was recorded
        assert "BTCUSDT" in market_data_service.active_subscriptions
    
    @pytest.mark.asyncio
    async def test_spread_compression_alert(self, market_data_service):
        """Test spread compression alert."""
        depth_data = {
            "stream": "btcusdt@depth20",
            "data": {
                "lastUpdateId": 12345,
                "bids": [["49995.00", "1.0"]],  # Tight spread
                "asks": [["50000.00", "0.5"]]
            }
        }
        
        await market_data_service._handle_depth(depth_data)
        
        # Check spread alert event was published
        calls = market_data_service.event_bus.publish.call_args_list
        assert len(calls) > 0
        
        # Find spread alert
        for call in calls:
            event = call[0][0]
            if event.event_type == EventType.SPREAD_ALERT:
                assert event.aggregate_id == "BTCUSDT"
                assert event.event_data["spread_bp"] < 10
                break
        else:
            pytest.fail("Spread alert not found")
    
    @pytest.mark.asyncio
    async def test_candle_aggregation(self, market_data_service):
        """Test candle aggregation from ticks."""
        # Simulate multiple ticks in same minute
        base_time = 1234567890.0
        
        for i in range(5):
            tick = Tick(
                symbol="BTCUSDT",
                price=Decimal(f"{50000 + i * 10}"),
                quantity=Decimal("0.1"),
                timestamp=base_time + i,
                is_buyer_maker=True
            )
            await market_data_service._update_candle_aggregator("BTCUSDT", tick)
        
        # Check aggregator was updated
        agg = market_data_service.candle_aggregators.get("BTCUSDT")
        assert agg is not None
        assert agg["open"] == Decimal("50000")
        assert agg["high"] == Decimal("50040")
        assert agg["low"] == Decimal("50000")
        assert agg["close"] == Decimal("50040")
        assert agg["volume"] == Decimal("0.5")
        assert agg["trades"] == 5
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, market_data_service):
        """Test getting service statistics."""
        # Add some data
        market_data_service.current_prices["BTCUSDT"] = Decimal("50000")
        market_data_service.active_subscriptions.add("BTCUSDT")
        
        stats = market_data_service.get_statistics()
        
        assert stats["active_subscriptions"] == 1
        assert stats["tracked_symbols"] == 1
        assert "websocket_stats" in stats