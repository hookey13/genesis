"""Unit tests for Large Trader Detection."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

import pytest

from genesis.analytics.large_trader_detection import (
    WhaleActivity,
    TradeCluster,
    VPINData,
    LargeTraderDetector
)
from genesis.engine.event_bus import EventBus
from genesis.core.events import Event


class TestWhaleActivity:
    """Test WhaleActivity dataclass."""
    
    def test_whale_activity_creation(self):
        """Test creating whale activity record."""
        activity = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            trade_size=Decimal("10"),
            price=Decimal("50000"),
            side="buy",
            percentile=Decimal("98"),
            vpin_score=Decimal("0.4"),
            confidence=Decimal("0.8")
        )
        
        assert activity.symbol == "BTCUSDT"
        assert activity.trade_size == Decimal("10")
        assert activity.notional == Decimal("500000")
        assert activity.percentile == Decimal("98")
    
    def test_significance_check(self):
        """Test significance checking."""
        activity = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            trade_size=Decimal("10"),
            price=Decimal("50000"),
            side="buy",
            percentile=Decimal("98"),
            vpin_score=Decimal("0.4"),
            confidence=Decimal("0.8")
        )
        
        # Significant: high percentile and confidence
        assert activity.is_significant()
        
        # Not significant: low percentile
        activity.percentile = Decimal("90")
        assert not activity.is_significant()
        
        # Not significant: low confidence
        activity.percentile = Decimal("98")
        activity.confidence = Decimal("0.5")
        assert not activity.is_significant()


class TestTradeCluster:
    """Test TradeCluster dataclass."""
    
    def test_cluster_creation(self):
        """Test creating trade cluster."""
        cluster = TradeCluster(
            cluster_id="BTCUSDT_123",
            symbol="BTCUSDT"
        )
        
        assert cluster.cluster_id == "BTCUSDT_123"
        assert cluster.symbol == "BTCUSDT"
        assert cluster.total_volume == Decimal("0")
        assert cluster.avg_trade_size == Decimal("0")
    
    def test_add_trades(self):
        """Test adding trades to cluster."""
        cluster = TradeCluster(
            cluster_id="BTCUSDT_123",
            symbol="BTCUSDT"
        )
        
        # Add buy trades
        for i in range(3):
            activity = WhaleActivity(
                symbol="BTCUSDT",
                timestamp=datetime.now(timezone.utc),
                trade_size=Decimal("10"),
                price=Decimal("50000"),
                side="buy",
                percentile=Decimal("98"),
                vpin_score=Decimal("0.4")
            )
            cluster.add_trade(activity)
        
        assert len(cluster.trades) == 3
        assert cluster.total_volume == Decimal("30")
        assert cluster.avg_trade_size == Decimal("10")
        assert cluster.dominant_side == "buy"
        assert all(t.cluster_id == "BTCUSDT_123" for t in cluster.trades)
    
    def test_mixed_sides(self):
        """Test cluster with mixed buy/sell trades."""
        cluster = TradeCluster(
            cluster_id="BTCUSDT_123",
            symbol="BTCUSDT"
        )
        
        # Add more buys than sells
        for _ in range(3):
            cluster.add_trade(WhaleActivity(
                symbol="BTCUSDT",
                timestamp=datetime.now(timezone.utc),
                trade_size=Decimal("10"),
                price=Decimal("50000"),
                side="buy",
                percentile=Decimal("98"),
                vpin_score=Decimal("0.4")
            ))
        
        for _ in range(2):
            cluster.add_trade(WhaleActivity(
                symbol="BTCUSDT",
                timestamp=datetime.now(timezone.utc),
                trade_size=Decimal("5"),
                price=Decimal("50000"),
                side="sell",
                percentile=Decimal("96"),
                vpin_score=Decimal("0.3")
            ))
        
        assert cluster.total_volume == Decimal("40")  # 30 buy + 10 sell
        assert cluster.dominant_side == "buy"  # More buy volume


class TestVPINData:
    """Test VPINData dataclass."""
    
    def test_vpin_data_creation(self):
        """Test creating VPIN data record."""
        vpin_data = VPINData(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            vpin=Decimal("0.35"),
            buy_volume=Decimal("60"),
            sell_volume=Decimal("40"),
            bucket_size=Decimal("100"),
            confidence=Decimal("0.8")
        )
        
        assert vpin_data.vpin == Decimal("0.35")
        assert vpin_data.indicates_informed_trading()
        
        # Low VPIN - no informed trading
        vpin_data.vpin = Decimal("0.2")
        assert not vpin_data.indicates_informed_trading()


@pytest.mark.asyncio
class TestLargeTraderDetector:
    """Test LargeTraderDetector class."""
    
    @pytest.fixture
    def event_bus(self):
        """Create mock event bus."""
        return AsyncMock(spec=EventBus)
    
    @pytest.fixture
    def detector(self, event_bus):
        """Create large trader detector instance."""
        return LargeTraderDetector(
            event_bus=event_bus,
            percentile_threshold=Decimal("95"),
            cluster_time_window=60,
            vpin_bucket_size=Decimal("100")
        )
    
    async def test_analyze_normal_trade(self, detector):
        """Test analyzing normal-sized trade."""
        # Add historical trades for distribution
        detector.trade_history["BTCUSDT"] = deque([Decimal(i) for i in range(1, 101)])
        
        # Analyze small trade (should not be whale)
        result = await detector.analyze_trade(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            quantity=Decimal("50"),  # 50th percentile
            side="buy"
        )
        
        assert result is None  # Not a whale
    
    async def test_detect_whale_trade(self, detector, event_bus):
        """Test detecting whale trade."""
        # Add historical trades for distribution
        detector.trade_history["BTCUSDT"] = deque([Decimal(i) for i in range(1, 101)])
        
        # Analyze large trade
        result = await detector.analyze_trade(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            quantity=Decimal("100"),  # 99th percentile
            side="buy"
        )
        
        assert result is not None
        assert result.percentile >= Decimal("95")
        assert result.is_significant()
        
        # Check event was published
        event_bus.publish.assert_called()
        call_args = event_bus.publish.call_args[0][0]
        assert call_args.type == "whale_activity_detected"
    
    def test_calculate_size_percentile(self, detector):
        """Test size percentile calculation."""
        # Setup distribution
        detector.trade_history["BTCUSDT"] = deque([Decimal(i) for i in range(1, 101)])
        
        # Test various sizes
        assert detector._calculate_size_percentile("BTCUSDT", Decimal("50")) == Decimal("49")
        assert detector._calculate_size_percentile("BTCUSDT", Decimal("95")) == Decimal("94")
        assert detector._calculate_size_percentile("BTCUSDT", Decimal("150")) == Decimal("100")
        
        # Test with no history
        assert detector._calculate_size_percentile("ETHUSDT", Decimal("100")) == Decimal("50")
    
    async def test_vpin_calculation(self, detector):
        """Test VPIN calculation."""
        # Add trades to build VPIN buckets
        for _ in range(10):
            await detector._update_vpin("BTCUSDT", Decimal("10"), "buy")
        for _ in range(5):
            await detector._update_vpin("BTCUSDT", Decimal("10"), "sell")
        
        # Check VPIN was calculated
        assert "BTCUSDT" in detector.vpin_buckets
        assert len(detector.vpin_history["BTCUSDT"]) > 0
        
        # VPIN should show imbalance
        latest_vpin = detector.vpin_history["BTCUSDT"][-1]
        assert latest_vpin.vpin > Decimal("0")
    
    def test_vpin_bucket_logic(self, detector):
        """Test VPIN bucket creation and management."""
        buckets = []
        
        # Fill one bucket
        buckets.append((Decimal("60"), Decimal("40")))  # Total 100
        vpin = detector._calculate_vpin(buckets)
        
        # VPIN = |60-40|/100 = 0.2
        assert abs(vpin - Decimal("0.2")) < Decimal("0.01")
        
        # Multiple buckets with varying imbalance
        buckets = [
            (Decimal("70"), Decimal("30")),  # 0.4 imbalance
            (Decimal("55"), Decimal("45")),  # 0.1 imbalance
            (Decimal("90"), Decimal("10")),  # 0.8 imbalance
        ]
        vpin = detector._calculate_vpin(buckets)
        
        # Average imbalance = (0.4 + 0.1 + 0.8) / 3 = 0.433
        assert abs(vpin - Decimal("0.433")) < Decimal("0.01")
    
    async def test_clustering(self, detector):
        """Test whale trade clustering."""
        timestamp = datetime.now(timezone.utc)
        
        # Create first whale trade
        whale1 = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=timestamp,
            trade_size=Decimal("100"),
            price=Decimal("50000"),
            side="buy",
            percentile=Decimal("98"),
            vpin_score=Decimal("0.4")
        )
        
        cluster = detector._find_or_create_cluster(whale1)
        assert cluster is not None
        assert cluster.symbol == "BTCUSDT"
        
        # Create similar trade within time window
        whale2 = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=timestamp + timedelta(seconds=30),
            trade_size=Decimal("90"),  # Similar size
            price=Decimal("50100"),
            side="buy",  # Same side
            percentile=Decimal("97"),
            vpin_score=Decimal("0.35")
        )
        
        cluster2 = detector._find_or_create_cluster(whale2)
        assert cluster2 == cluster  # Should be same cluster
    
    def test_similar_pattern_detection(self, detector):
        """Test pattern similarity for clustering."""
        cluster = TradeCluster(
            cluster_id="BTCUSDT_123",
            symbol="BTCUSDT"
        )
        
        # Add initial trades
        for _ in range(3):
            cluster.add_trade(WhaleActivity(
                symbol="BTCUSDT",
                timestamp=datetime.now(timezone.utc),
                trade_size=Decimal("100"),
                price=Decimal("50000"),
                side="buy",
                percentile=Decimal("98"),
                vpin_score=Decimal("0.4")
            ))
        
        # Similar trade
        similar = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            trade_size=Decimal("110"),  # Within 50% of average
            price=Decimal("50100"),
            side="buy",  # Same side
            percentile=Decimal("98"),
            vpin_score=Decimal("0.4")
        )
        assert detector._is_similar_pattern(similar, cluster)
        
        # Different side
        different_side = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            trade_size=Decimal("100"),
            price=Decimal("50000"),
            side="sell",  # Different side
            percentile=Decimal("98"),
            vpin_score=Decimal("0.4")
        )
        assert not detector._is_similar_pattern(different_side, cluster)
        
        # Very different size
        different_size = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            trade_size=Decimal("200"),  # >50% different
            price=Decimal("50000"),
            side="buy",
            percentile=Decimal("99"),
            vpin_score=Decimal("0.5")
        )
        assert not detector._is_similar_pattern(different_size, cluster)
    
    def test_confidence_calculation(self, detector):
        """Test detection confidence calculation."""
        # High confidence: high percentile and VPIN
        confidence = detector._calculate_confidence(
            percentile=Decimal("99"),
            vpin=Decimal("0.5")
        )
        assert confidence > Decimal("0.8")
        
        # Medium confidence
        confidence = detector._calculate_confidence(
            percentile=Decimal("95"),
            vpin=Decimal("0.3")
        )
        assert Decimal("0.5") < confidence < Decimal("0.8")
        
        # Low confidence
        confidence = detector._calculate_confidence(
            percentile=Decimal("91"),
            vpin=Decimal("0.1")
        )
        assert confidence < Decimal("0.5")
    
    def test_entity_tracking(self, detector):
        """Test cumulative entity volume tracking."""
        whale = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            trade_size=Decimal("100"),
            price=Decimal("50000"),
            side="buy",
            percentile=Decimal("98"),
            vpin_score=Decimal("0.4"),
            cluster_id="cluster_1"
        )
        
        detector._update_entity_tracking(whale)
        
        assert "BTCUSDT" in detector.entity_volumes
        assert "cluster_1" in detector.entity_volumes["BTCUSDT"]
        assert detector.entity_volumes["BTCUSDT"]["cluster_1"] == Decimal("100")
        
        # Add more volume to same entity
        whale2 = WhaleActivity(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            trade_size=Decimal("50"),
            price=Decimal("50100"),
            side="buy",
            percentile=Decimal("96"),
            vpin_score=Decimal("0.35"),
            cluster_id="cluster_1"
        )
        
        detector._update_entity_tracking(whale2)
        assert detector.entity_volumes["BTCUSDT"]["cluster_1"] == Decimal("150")
    
    def test_get_whale_statistics(self, detector):
        """Test getting whale statistics."""
        # Add trade history
        detector.trade_history["BTCUSDT"] = deque([
            Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40"),
            Decimal("50"), Decimal("60"), Decimal("70"), Decimal("80"),
            Decimal("90"), Decimal("100")
        ])
        
        stats = detector.get_whale_statistics("BTCUSDT")
        
        assert "mean_size" in stats
        assert "median_size" in stats
        assert "std_dev" in stats
        assert "p95_threshold" in stats
        assert "p99_threshold" in stats
        assert stats["mean_size"] == 55.0
        assert stats["median_size"] == 55.0
    
    def test_vpin_trend(self, detector):
        """Test VPIN trend detection."""
        # Create increasing VPIN trend
        detector.vpin_history["BTCUSDT"] = deque([
            VPINData(
                symbol="BTCUSDT",
                timestamp=datetime.now(timezone.utc),
                vpin=Decimal(str(i/10)),
                buy_volume=Decimal("50"),
                sell_volume=Decimal("50"),
                bucket_size=Decimal("100"),
                confidence=Decimal("0.8")
            )
            for i in range(1, 11)
        ])
        
        trend = detector.get_vpin_trend("BTCUSDT")
        assert trend == "increasing"
        
        # Reverse for decreasing trend
        for vpin_data in detector.vpin_history["BTCUSDT"]:
            vpin_data.vpin = Decimal("1") - vpin_data.vpin
        
        trend = detector.get_vpin_trend("BTCUSDT")
        assert trend == "decreasing"