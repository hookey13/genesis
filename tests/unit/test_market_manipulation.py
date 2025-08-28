"""Unit tests for Market Manipulation Detection."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from collections import deque

import pytest

from genesis.analytics.market_manipulation import (
    ManipulationType,
    OrderActivity,
    ManipulationPattern,
    MarketManipulationDetector
)
from genesis.exchange.order_book_manager import OrderBookSnapshot, OrderBookLevel
from genesis.engine.event_bus import EventBus
from genesis.core.events import Event


class TestOrderActivity:
    """Test OrderActivity dataclass."""
    
    def test_order_activity_creation(self):
        """Test creating order activity."""
        activity = OrderActivity(
            order_id="order_123",
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            price=Decimal("50000"),
            quantity=Decimal("1"),
            side="bid",
            action="place",
            levels_from_best=2
        )
        
        assert activity.order_id == "order_123"
        assert activity.notional == Decimal("50000")
        assert activity.action == "place"


class TestManipulationPattern:
    """Test ManipulationPattern dataclass."""
    
    def test_pattern_creation(self):
        """Test creating manipulation pattern."""
        pattern = ManipulationPattern(
            pattern_id="spoof_123",
            symbol="BTCUSDT",
            manipulation_type=ManipulationType.SPOOFING,
            start_time=datetime.now(timezone.utc),
            confidence=Decimal("0.8"),
            severity="high"
        )
        
        assert pattern.pattern_id == "spoof_123"
        assert pattern.manipulation_type == ManipulationType.SPOOFING
        assert pattern.is_significant()
    
    def test_cancellation_rate(self):
        """Test cancellation rate calculation."""
        pattern = ManipulationPattern(
            pattern_id="spoof_123",
            symbol="BTCUSDT",
            manipulation_type=ManipulationType.SPOOFING,
            start_time=datetime.now(timezone.utc)
        )
        
        # Add orders
        for _ in range(10):
            pattern.orders.append(OrderActivity(
                order_id=f"order_{_}",
                symbol="BTCUSDT",
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50000"),
                quantity=Decimal("1"),
                side="bid",
                action="place",
                levels_from_best=2
            ))
        
        # Add cancellations
        for _ in range(8):
            pattern.orders.append(OrderActivity(
                order_id=f"order_{_}",
                symbol="BTCUSDT",
                timestamp=datetime.now(timezone.utc),
                price=Decimal("50000"),
                quantity=Decimal("1"),
                side="bid",
                action="cancel",
                levels_from_best=2
            ))
        
        assert pattern.cancellation_rate == Decimal("0.8")
        assert pattern.total_volume == Decimal("10")


@pytest.mark.asyncio
class TestMarketManipulationDetector:
    """Test MarketManipulationDetector class."""
    
    @pytest.fixture
    def event_bus(self):
        """Create mock event bus."""
        return AsyncMock(spec=EventBus)
    
    @pytest.fixture
    def detector(self, event_bus):
        """Create manipulation detector instance."""
        return MarketManipulationDetector(
            event_bus=event_bus,
            cancellation_threshold=Decimal("0.8"),
            time_window_seconds=30,
            min_orders_for_pattern=5
        )
    
    async def test_track_order_placement(self, detector):
        """Test tracking order placement."""
        await detector.track_order_placement(
            symbol="BTCUSDT",
            order_id="order_1",
            price=Decimal("50000"),
            quantity=Decimal("1"),
            side="bid"
        )
        
        assert "BTCUSDT" in detector.active_orders
        assert "order_1" in detector.active_orders["BTCUSDT"]
        assert len(detector.order_history["BTCUSDT"]) == 1
    
    async def test_track_order_cancellation(self, detector):
        """Test tracking order cancellation."""
        # Place order first
        await detector.track_order_placement(
            symbol="BTCUSDT",
            order_id="order_1",
            price=Decimal("50000"),
            quantity=Decimal("1"),
            side="bid"
        )
        
        # Cancel order
        await detector.track_order_cancellation(
            symbol="BTCUSDT",
            order_id="order_1"
        )
        
        assert "order_1" not in detector.active_orders["BTCUSDT"]
        assert len(detector.order_history["BTCUSDT"]) == 2
        assert detector.order_history["BTCUSDT"][-1].action == "cancel"
    
    async def test_spoofing_detection(self, detector, event_bus):
        """Test detecting spoofing pattern."""
        # Create spoofing pattern: quick placement and cancellation
        timestamp = datetime.now(timezone.utc)
        
        # Place multiple orders
        for i in range(10):
            await detector.track_order_placement(
                symbol="BTCUSDT",
                order_id=f"order_{i}",
                price=Decimal("49990") - Decimal(i),  # Away from best
                quantity=Decimal("10"),
                side="bid",
                timestamp=timestamp
            )
        
        # Quickly cancel most orders
        await asyncio.sleep(0.001)  # Very short delay
        for i in range(9):  # Cancel 9 out of 10
            await detector.track_order_cancellation(
                symbol="BTCUSDT",
                order_id=f"order_{i}",
                timestamp=timestamp + timedelta(seconds=2)
            )
        
        # Check if spoofing was detected
        event_calls = event_bus.publish.call_args_list
        manipulation_detected = False
        
        for call in event_calls:
            if call and len(call[0]) > 0:
                event = call[0][0]
                if event.type == "market_manipulation_detected":
                    if event.data["manipulation_type"] == "spoofing":
                        manipulation_detected = True
                        break
        
        assert manipulation_detected
    
    async def test_layering_detection(self, detector, event_bus):
        """Test detecting layering pattern."""
        # Place multiple orders at different price levels
        timestamp = datetime.now(timezone.utc)
        
        for i in range(6):  # 6 orders at different levels
            await detector.track_order_placement(
                symbol="BTCUSDT",
                order_id=f"layer_{i}",
                price=Decimal("50000") - Decimal(i * 10),
                quantity=Decimal("5"),
                side="bid",
                timestamp=timestamp + timedelta(seconds=i)
            )
        
        # Check if layering was detected
        event_calls = event_bus.publish.call_args_list
        layering_detected = False
        
        for call in event_calls:
            if call and len(call[0]) > 0:
                event = call[0][0]
                if event.type == "market_manipulation_detected":
                    if event.data["manipulation_type"] == "layering":
                        layering_detected = True
                        break
        
        assert layering_detected
    
    def test_calculate_levels_from_best(self, detector):
        """Test calculating distance from best price."""
        # Setup order book
        snapshot = OrderBookSnapshot(
            symbol="BTCUSDT",
            bids=[
                OrderBookLevel(Decimal("50000"), Decimal("1")),
                OrderBookLevel(Decimal("49999"), Decimal("2")),
                OrderBookLevel(Decimal("49998"), Decimal("1.5"))
            ],
            asks=[
                OrderBookLevel(Decimal("50001"), Decimal("1")),
                OrderBookLevel(Decimal("50002"), Decimal("1.5")),
                OrderBookLevel(Decimal("50003"), Decimal("2"))
            ]
        )
        
        detector.update_order_book(snapshot)
        
        # Test bid side
        assert detector._calculate_levels_from_best("BTCUSDT", Decimal("50000"), "bid") == 0
        assert detector._calculate_levels_from_best("BTCUSDT", Decimal("49999"), "bid") == 1
        assert detector._calculate_levels_from_best("BTCUSDT", Decimal("49997"), "bid") == 3
        
        # Test ask side
        assert detector._calculate_levels_from_best("BTCUSDT", Decimal("50001"), "ask") == 0
        assert detector._calculate_levels_from_best("BTCUSDT", Decimal("50002"), "ask") == 1
    
    def test_calculate_cancellation_rate(self, detector):
        """Test cancellation rate calculation."""
        orders = [
            OrderActivity("1", "BTCUSDT", datetime.now(timezone.utc), 
                         Decimal("50000"), Decimal("1"), "bid", "place", 0),
            OrderActivity("2", "BTCUSDT", datetime.now(timezone.utc),
                         Decimal("50000"), Decimal("1"), "bid", "place", 0),
            OrderActivity("1", "BTCUSDT", datetime.now(timezone.utc),
                         Decimal("50000"), Decimal("1"), "bid", "cancel", 0),
        ]
        
        rate = detector._calculate_cancellation_rate(orders)
        assert rate == Decimal("0.5")  # 1 cancel / 2 placements
    
    def test_spoofing_confidence_calculation(self, detector):
        """Test spoofing confidence calculation."""
        # High confidence scenario
        confidence = detector._calculate_spoofing_confidence(
            cancellation_rate=Decimal("0.95"),
            duration=timedelta(seconds=1),
            avg_levels=4
        )
        assert confidence >= Decimal("0.8")
        
        # Low confidence scenario
        confidence = detector._calculate_spoofing_confidence(
            cancellation_rate=Decimal("0.7"),
            duration=timedelta(seconds=10),
            avg_levels=1
        )
        assert confidence < Decimal("0.5")
    
    def test_layering_confidence_calculation(self, detector):
        """Test layering confidence calculation."""
        # Create orders at multiple levels
        orders = []
        for i in range(5):
            orders.append(OrderActivity(
                f"order_{i}",
                "BTCUSDT",
                datetime.now(timezone.utc),
                Decimal("50000") - Decimal(i * 10),  # Different prices
                Decimal("10"),  # Same quantity (coordinated)
                "bid",
                "place",
                i
            ))
        
        confidence = detector._calculate_layering_confidence(orders)
        assert confidence >= Decimal("0.7")
    
    def test_severity_determination(self, detector):
        """Test severity level determination."""
        # High severity
        assert detector._determine_severity(Decimal("0.95"), Decimal("2000000")) == "high"
        
        # Medium severity
        assert detector._determine_severity(Decimal("0.75"), Decimal("200000")) == "medium"
        
        # Low severity
        assert detector._determine_severity(Decimal("0.6"), Decimal("10000")) == "low"
    
    async def test_quote_stuffing_detection(self, detector, event_bus):
        """Test detecting quote stuffing."""
        # Place and cancel many orders rapidly
        timestamp = datetime.now(timezone.utc)
        
        for i in range(60):  # 60 orders in quick succession
            detector.order_history.setdefault("BTCUSDT", deque(maxlen=1000))
            
            # Add directly to history to simulate rapid activity
            detector.order_history["BTCUSDT"].append(OrderActivity(
                f"stuff_{i}",
                "BTCUSDT",
                timestamp + timedelta(milliseconds=i * 10),
                Decimal("50000"),
                Decimal("1"),
                "bid",
                "place",
                1
            ))
            
            # Immediate cancellation
            detector.order_history["BTCUSDT"].append(OrderActivity(
                f"stuff_{i}",
                "BTCUSDT",
                timestamp + timedelta(milliseconds=i * 10 + 5),
                Decimal("50000"),
                Decimal("1"),
                "bid",
                "cancel",
                1
            ))
        
        # Trigger detection
        await detector._detect_quote_stuffing("BTCUSDT")
        
        # Check if detected
        event_bus.publish.assert_called()
    
    def test_get_manipulation_statistics(self, detector):
        """Test getting manipulation statistics."""
        # Add some order history
        detector.order_history["BTCUSDT"] = deque([
            OrderActivity(f"o{i}", "BTCUSDT", datetime.now(timezone.utc),
                         Decimal("50000"), Decimal("1"), "bid", 
                         "place" if i % 2 == 0 else "cancel", 0)
            for i in range(10)
        ])
        
        stats = detector.get_manipulation_statistics("BTCUSDT")
        
        assert "cancellation_rate" in stats
        assert "active_orders" in stats
        assert "recent_order_count" in stats
        assert stats["cancellation_rate"] == 0.5  # 5 cancels / 5 placements