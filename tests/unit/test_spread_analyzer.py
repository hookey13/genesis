"""
Unit tests for spread analyzer module.

Tests spread calculation, compression detection, order imbalance,
and metrics tracking functionality.
"""

from decimal import Decimal

import pytest

from genesis.analytics.spread_analyzer import (
    OrderImbalance,
    SpreadAnalyzer,
    SpreadCompressionEvent,
    SpreadMetrics,
)
from genesis.core.exceptions import ValidationError


class TestSpreadAnalyzer:
    """Test suite for SpreadAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a SpreadAnalyzer instance."""
        return SpreadAnalyzer(max_history_size=100)

    @pytest.fixture
    def sample_orderbook(self):
        """Create a sample orderbook."""
        return {
            "bids": [
                ["100.00", "10.0"],
                ["99.95", "20.0"],
                ["99.90", "15.0"],
                ["99.85", "25.0"],
                ["99.80", "30.0"]
            ],
            "asks": [
                ["100.10", "12.0"],
                ["100.15", "18.0"],
                ["100.20", "22.0"],
                ["100.25", "16.0"],
                ["100.30", "28.0"]
            ]
        }

    def test_track_spread_valid(self, analyzer):
        """Test tracking spread with valid prices."""
        metrics = analyzer.track_spread("BTCUSDT", Decimal("100.00"), Decimal("100.10"))

        assert metrics.symbol == "BTCUSDT"
        assert metrics.bid_price == Decimal("100.00")
        assert metrics.ask_price == Decimal("100.10")
        # Use approximate comparison for spread calculations
        assert abs(metrics.spread_bps - Decimal("10")) < Decimal("0.01")  # 0.1% = 10 bps
        assert abs(metrics.current_spread - Decimal("10")) < Decimal("0.01")
        assert abs(metrics.avg_spread - Decimal("10")) < Decimal("0.01")  # First entry
        assert metrics.volatility == Decimal("0")  # Single data point

    def test_track_spread_invalid_prices(self, analyzer):
        """Test tracking spread with invalid prices."""
        # Negative bid
        with pytest.raises(ValidationError):
            analyzer.track_spread("BTCUSDT", Decimal("-100"), Decimal("100.10"))

        # Zero ask
        with pytest.raises(ValidationError):
            analyzer.track_spread("BTCUSDT", Decimal("100"), Decimal("0"))

        # Ask less than bid
        with pytest.raises(ValidationError):
            analyzer.track_spread("BTCUSDT", Decimal("100.10"), Decimal("100.00"))

    def test_calculate_spread_bps(self, analyzer):
        """Test spread calculation in basis points."""
        # Normal spread
        spread = analyzer.calculate_spread_bps(Decimal("100"), Decimal("100.10"))
        assert abs(spread - Decimal("10")) < Decimal("0.01")

        # Wider spread
        spread = analyzer.calculate_spread_bps(Decimal("100"), Decimal("101"))
        assert abs(spread - Decimal("100")) < Decimal("0.5")  # Allow for rounding in wider spreads

        # Very tight spread
        spread = analyzer.calculate_spread_bps(Decimal("100"), Decimal("100.01"))
        assert abs(spread - Decimal("1")) < Decimal("0.01")

        # Invalid inputs
        assert analyzer.calculate_spread_bps(Decimal("0"), Decimal("100")) == Decimal("0")
        assert analyzer.calculate_spread_bps(Decimal("100"), Decimal("0")) == Decimal("0")

    def test_spread_history_management(self, analyzer):
        """Test spread history tracking and limits."""
        symbol = "BTCUSDT"

        # Add spreads up to max history
        for i in range(150):  # More than max_history_size
            bid = Decimal("100") + Decimal(str(i * 0.01))
            ask = bid + Decimal("0.10")
            analyzer.track_spread(symbol, bid, ask)

        # Check history size is limited
        assert len(analyzer._spread_history[symbol]) <= analyzer.max_history_size

        # Verify metrics are updated
        metrics = analyzer.get_spread_metrics(symbol)
        assert metrics is not None
        assert metrics.symbol == symbol

    def test_calculate_spread_volatility(self, analyzer):
        """Test spread volatility calculation."""
        # No data
        assert analyzer.calculate_spread_volatility([]) == Decimal("0")

        # Single data point
        assert analyzer.calculate_spread_volatility([Decimal("10")]) == Decimal("0")

        # Constant spreads (no volatility)
        spreads = [Decimal("10")] * 10
        assert analyzer.calculate_spread_volatility(spreads) == Decimal("0")

        # Variable spreads
        spreads = [Decimal("10"), Decimal("15"), Decimal("8"), Decimal("12"), Decimal("20")]
        volatility = analyzer.calculate_spread_volatility(spreads)
        assert volatility > Decimal("0")
        # Approximate expected std dev
        assert Decimal("4") < volatility < Decimal("5")

    def test_calculate_order_imbalance(self, analyzer, sample_orderbook):
        """Test order book imbalance calculation."""
        imbalance = analyzer.calculate_order_imbalance(sample_orderbook)

        assert isinstance(imbalance, OrderImbalance)
        assert imbalance.bid_weight > Decimal("0")
        assert imbalance.ask_weight > Decimal("0")
        assert imbalance.ratio > Decimal("0")

        # Check significance detection
        # With equal-ish volumes, should not be significant
        assert not imbalance.is_significant

    def test_order_imbalance_significant(self, analyzer):
        """Test significant order imbalance detection."""
        # Heavy bid pressure
        orderbook_bid_heavy = {
            "bids": [["100", "100"], ["99", "200"]],
            "asks": [["101", "10"], ["102", "20"]]
        }
        imbalance = analyzer.calculate_order_imbalance(orderbook_bid_heavy)
        assert imbalance.ratio > Decimal("2")
        assert imbalance.is_significant

        # Heavy ask pressure
        orderbook_ask_heavy = {
            "bids": [["100", "10"], ["99", "20"]],
            "asks": [["101", "100"], ["102", "200"]]
        }
        imbalance = analyzer.calculate_order_imbalance(orderbook_ask_heavy)
        assert imbalance.ratio < Decimal("0.5")
        assert imbalance.is_significant

    def test_order_imbalance_empty_book(self, analyzer):
        """Test order imbalance with empty orderbook."""
        empty_book = {"bids": [], "asks": []}
        imbalance = analyzer.calculate_order_imbalance(empty_book)

        assert imbalance.ratio == Decimal("1.0")
        assert imbalance.bid_weight == Decimal("0")
        assert imbalance.ask_weight == Decimal("0")
        assert not imbalance.is_significant

    def test_detect_spread_compression(self, analyzer):
        """Test spread compression detection."""
        symbol = "BTCUSDT"

        # Build up history with normal spreads
        for _ in range(20):
            analyzer.track_spread(symbol, Decimal("100"), Decimal("100.20"))

        # No compression yet
        event = analyzer.detect_spread_compression(symbol)
        assert event is None

        # Add compressed spreads
        for _ in range(5):
            analyzer.track_spread(symbol, Decimal("100"), Decimal("100.05"))

        # Should detect compression
        event = analyzer.detect_spread_compression(symbol)
        assert isinstance(event, SpreadCompressionEvent)
        assert event.symbol == symbol
        assert event.current_spread < event.average_spread * Decimal("0.8")
        assert event.compression_ratio < Decimal("0.8")

    def test_compression_duration_tracking(self, analyzer):
        """Test tracking duration of spread compression."""
        symbol = "BTCUSDT"

        # No compression initially
        assert analyzer.get_compression_duration(symbol) is None

        # Build history and trigger compression
        for _ in range(20):
            analyzer.track_spread(symbol, Decimal("100"), Decimal("100.20"))
        for _ in range(5):
            analyzer.track_spread(symbol, Decimal("100"), Decimal("100.05"))

        # Detect compression
        analyzer.detect_spread_compression(symbol)

        # Check duration is tracked
        duration = analyzer.get_compression_duration(symbol)
        assert duration is not None
        assert duration >= 0

    def test_compression_recovery(self, analyzer):
        """Test spread compression recovery detection."""
        symbol = "BTCUSDT"

        # Build history with normal spreads
        for _ in range(20):
            analyzer.track_spread(symbol, Decimal("100"), Decimal("100.20"))

        # Trigger compression
        for _ in range(5):
            analyzer.track_spread(symbol, Decimal("100"), Decimal("100.05"))

        event = analyzer.detect_spread_compression(symbol)
        assert event is not None

        # Add recovering spreads
        for _ in range(5):
            analyzer.track_spread(symbol, Decimal("100"), Decimal("100.18"))

        # Should detect recovery
        event = analyzer.detect_spread_compression(symbol)
        assert event is None
        assert analyzer.get_compression_duration(symbol) is None

    def test_get_spread_metrics(self, analyzer):
        """Test retrieving spread metrics."""
        symbol = "BTCUSDT"

        # No metrics initially
        assert analyzer.get_spread_metrics(symbol) is None

        # Track some spreads
        analyzer.track_spread(symbol, Decimal("100"), Decimal("100.10"))

        # Get metrics
        metrics = analyzer.get_spread_metrics(symbol)
        assert metrics is not None
        assert metrics.symbol == symbol
        assert metrics.bid_price == Decimal("100")
        assert metrics.ask_price == Decimal("100.10")

    def test_get_all_metrics(self, analyzer):
        """Test retrieving all spread metrics."""
        # Initially empty
        assert analyzer.get_all_metrics() == {}

        # Track spreads for multiple symbols
        analyzer.track_spread("BTCUSDT", Decimal("100"), Decimal("100.10"))
        analyzer.track_spread("ETHUSDT", Decimal("50"), Decimal("50.05"))

        all_metrics = analyzer.get_all_metrics()
        assert len(all_metrics) == 2
        assert "BTCUSDT" in all_metrics
        assert "ETHUSDT" in all_metrics

    def test_clear_history(self, analyzer):
        """Test clearing spread history."""
        # Add data for multiple symbols
        analyzer.track_spread("BTCUSDT", Decimal("100"), Decimal("100.10"))
        analyzer.track_spread("ETHUSDT", Decimal("50"), Decimal("50.05"))

        # Clear specific symbol
        analyzer.clear_history("BTCUSDT")
        assert analyzer.get_spread_metrics("BTCUSDT") is None
        assert analyzer.get_spread_metrics("ETHUSDT") is not None

        # Clear all
        analyzer.track_spread("BTCUSDT", Decimal("100"), Decimal("100.10"))
        analyzer.clear_history()
        assert analyzer.get_all_metrics() == {}

    def test_spread_metrics_dataclass(self):
        """Test SpreadMetrics dataclass functionality."""
        metrics = SpreadMetrics(
            symbol="BTCUSDT",
            current_spread=Decimal("10"),
            avg_spread=Decimal("12"),
            volatility=Decimal("2"),
            bid_price=Decimal("100"),
            ask_price=Decimal("100.10"),
            bid_volume=Decimal("10"),
            ask_volume=Decimal("12")
        )

        # Check post_init calculation
        assert abs(metrics.spread_bps - Decimal("10")) < Decimal("0.01")

        # Test with zero prices
        metrics_zero = SpreadMetrics(
            symbol="TEST",
            current_spread=Decimal("0"),
            avg_spread=Decimal("0"),
            volatility=Decimal("0"),
            bid_price=Decimal("0"),
            ask_price=Decimal("0"),
            bid_volume=Decimal("0"),
            ask_volume=Decimal("0")
        )
        assert metrics_zero.spread_bps == Decimal("0")

    def test_order_imbalance_dataclass(self):
        """Test OrderImbalance dataclass functionality."""
        # Non-significant imbalance
        imbalance = OrderImbalance(
            ratio=Decimal("1.2"),
            bid_weight=Decimal("120"),
            ask_weight=Decimal("100")
        )
        assert not imbalance.is_significant

        # Significant bid pressure
        imbalance_bid = OrderImbalance(
            ratio=Decimal("2.5"),
            bid_weight=Decimal("250"),
            ask_weight=Decimal("100")
        )
        assert imbalance_bid.is_significant

        # Significant ask pressure
        imbalance_ask = OrderImbalance(
            ratio=Decimal("0.4"),
            bid_weight=Decimal("40"),
            ask_weight=Decimal("100")
        )
        assert imbalance_ask.is_significant

    def test_concurrent_tracking(self, analyzer):
        """Test tracking multiple symbols concurrently."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]

        # Track spreads for all symbols
        for i in range(10):
            for symbol in symbols:
                base_price = Decimal("100") + Decimal(str(hash(symbol) % 100))
                bid = base_price + Decimal(str(i * 0.01))
                ask = bid + Decimal("0.10")
                analyzer.track_spread(symbol, bid, ask)

        # Verify all symbols are tracked
        all_metrics = analyzer.get_all_metrics()
        assert len(all_metrics) == len(symbols)

        for symbol in symbols:
            assert symbol in all_metrics
            metrics = all_metrics[symbol]
            assert metrics.symbol == symbol
            assert len(analyzer._spread_history[symbol]) == 10

