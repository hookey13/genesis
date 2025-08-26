"""
Unit tests for tilt behavioral indicators.
"""

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

from genesis.tilt.indicators import (
    ClickSpeedIndicator,
    OrderFrequencyIndicator,
    PositionSizingIndicator,
    CancelRateIndicator
)


class TestClickSpeedIndicator:
    """Test click speed indicator."""
    
    @pytest.fixture
    def indicator(self):
        """Create click speed indicator."""
        return ClickSpeedIndicator(window_size=100)
    
    def test_record_market_update(self, indicator):
        """Test recording market update timestamp."""
        timestamp = datetime.now(timezone.utc)
        indicator.record_market_update(timestamp)
        
        assert indicator.last_market_update == timestamp
    
    def test_record_action_calculates_latency(self, indicator):
        """Test recording action calculates latency."""
        market_time = datetime.now(timezone.utc)
        action_time = market_time + timedelta(milliseconds=150)
        
        indicator.record_market_update(market_time)
        latency = indicator.record_action(action_time)
        
        assert latency is not None
        assert latency == Decimal("150")
        assert len(indicator.latencies_ms) == 1
    
    def test_negative_latency_ignored(self, indicator):
        """Test negative latency is ignored."""
        market_time = datetime.now(timezone.utc)
        action_time = market_time - timedelta(milliseconds=100)  # Before market update
        
        indicator.record_market_update(market_time)
        latency = indicator.record_action(action_time)
        
        assert latency is None
        assert len(indicator.latencies_ms) == 0
    
    def test_average_latency_calculation(self, indicator):
        """Test average latency calculation."""
        market_time = datetime.now(timezone.utc)
        indicator.record_market_update(market_time)
        
        # Record multiple actions
        latencies = [100, 150, 200, 120, 180]
        for ms in latencies:
            action_time = market_time + timedelta(milliseconds=ms)
            indicator.record_market_update(market_time)  # Reset for each
            indicator.record_action(action_time)
        
        avg = indicator.get_average_latency()
        expected = Decimal(str(sum(latencies) / len(latencies)))
        
        assert avg == expected
    
    def test_panic_clicking_detection(self, indicator):
        """Test panic clicking pattern detection."""
        market_time = datetime.now(timezone.utc)
        
        # Add rapid clicks (all under 100ms)
        for ms in [50, 60, 70]:
            indicator.latencies_ms.append(Decimal(str(ms)))
        
        assert indicator.detect_panic_clicking(threshold_ms=Decimal("100")) is True
        
        # Add slower click
        indicator.latencies_ms.append(Decimal("200"))
        
        assert indicator.detect_panic_clicking(threshold_ms=Decimal("100")) is False
    
    def test_recent_pattern_analysis(self, indicator):
        """Test recent pattern analysis."""
        # Add varied latencies
        latencies = [100, 150, 200, 250, 300, 120, 140, 160, 180, 200]
        for ms in latencies:
            indicator.latencies_ms.append(Decimal(str(ms)))
        
        pattern = indicator.get_recent_pattern(last_n=5)
        
        assert pattern["has_data"] is True
        assert pattern["sample_count"] == 5
        assert "average_ms" in pattern
        assert "trend" in pattern
        assert "is_erratic" in pattern
    
    def test_window_size_limit(self, indicator):
        """Test window size is respected."""
        # Add more than window size
        for i in range(150):
            indicator.latencies_ms.append(Decimal(str(i)))
        
        assert len(indicator.latencies_ms) <= indicator.window_size


class TestOrderFrequencyIndicator:
    """Test order frequency indicator."""
    
    @pytest.fixture
    def indicator(self):
        """Create order frequency indicator."""
        return OrderFrequencyIndicator(window_minutes=60)
    
    def test_record_order(self, indicator):
        """Test recording an order."""
        timestamp = datetime.now(timezone.utc)
        rate = indicator.record_order(timestamp)
        
        assert len(indicator.order_timestamps) == 1
        assert rate == Decimal("0")  # Single order, no rate yet
    
    def test_order_rate_calculation(self, indicator):
        """Test order rate calculation."""
        base_time = datetime.now(timezone.utc)
        
        # Add orders over 30 minutes
        for i in range(10):
            timestamp = base_time + timedelta(minutes=i * 3)
            indicator.record_order(timestamp)
        
        rate = indicator.get_current_rate()
        
        # 10 orders in ~30 minutes = ~20 orders/hour
        assert rate > Decimal("15")
        assert rate < Decimal("25")
    
    def test_old_orders_removed(self, indicator):
        """Test old orders are removed from window."""
        base_time = datetime.now(timezone.utc)
        
        # Add old order
        old_time = base_time - timedelta(minutes=120)
        indicator.record_order(old_time)
        
        # Add recent order
        indicator.record_order(base_time)
        
        # Old order should be removed
        assert len(indicator.order_timestamps) == 1
        assert indicator.order_timestamps[0] == base_time
    
    def test_overtrading_detection(self, indicator):
        """Test overtrading detection."""
        base_time = datetime.now(timezone.utc)
        
        # Add many orders quickly
        for i in range(20):
            timestamp = base_time + timedelta(seconds=i * 10)
            indicator.record_order(timestamp)
        
        # 20 orders in ~3 minutes = very high rate
        assert indicator.detect_overtrading(threshold=Decimal("50")) is True
        assert indicator.detect_overtrading(threshold=Decimal("500")) is False
    
    def test_pattern_analysis(self, indicator):
        """Test order frequency pattern analysis."""
        base_time = datetime.now(timezone.utc)
        
        # Add orders with increasing frequency
        for i in range(5):
            timestamp = base_time - timedelta(minutes=30-i*5)
            indicator.record_order(timestamp)
        
        for i in range(10):
            timestamp = base_time - timedelta(minutes=5-i*0.5)
            indicator.record_order(timestamp)
        
        pattern = indicator.get_pattern_analysis()
        
        assert pattern["has_data"] is True
        assert pattern["order_count"] > 0
        assert "current_rate_per_hour" in pattern
        assert "is_accelerating" in pattern
        assert "burst_detected" in pattern


class TestPositionSizingIndicator:
    """Test position sizing indicator."""
    
    @pytest.fixture
    def indicator(self):
        """Create position sizing indicator."""
        return PositionSizingIndicator(window_size=50)
    
    def test_record_position(self, indicator):
        """Test recording a position."""
        size = Decimal("1000")
        timestamp = datetime.now(timezone.utc)
        
        metrics = indicator.record_position(size, timestamp, "win")
        
        assert len(indicator.position_sizes) == 1
        assert indicator.position_sizes[0] == size
        assert metrics["has_data"] is True
    
    def test_variance_calculation(self, indicator):
        """Test position size variance calculation."""
        # Add positions with varying sizes
        sizes = [1000, 1100, 900, 1200, 800, 1000, 1050, 950]
        
        for size in sizes:
            indicator.record_position(
                Decimal(str(size)),
                datetime.now(timezone.utc)
            )
        
        metrics = indicator.calculate_variance()
        
        assert metrics["has_data"] is True
        assert metrics["sample_count"] == len(sizes)
        assert metrics["coefficient_of_variation"] > 0
        assert "mean_size" in metrics
        assert "std_dev" in metrics
    
    def test_martingale_detection(self, indicator):
        """Test martingale betting pattern detection."""
        timestamp = datetime.now(timezone.utc)
        
        # Normal position
        indicator.record_position(Decimal("1000"), timestamp, "win")
        
        # Loss followed by doubled position (martingale)
        indicator.record_position(Decimal("1000"), timestamp, "loss")
        indicator.record_position(Decimal("2000"), timestamp, "win")
        
        # Another martingale
        indicator.record_position(Decimal("1000"), timestamp, "loss")
        indicator.record_position(Decimal("1600"), timestamp, "win")
        
        metrics = indicator.calculate_variance()
        
        assert metrics["martingale_detected"] is True
    
    def test_drift_detection(self, indicator):
        """Test position size drift detection."""
        # Add positions with increasing trend
        base_size = 1000
        for i in range(20):
            size = base_size + (i * 50)  # Gradual increase
            indicator.record_position(
                Decimal(str(size)),
                datetime.now(timezone.utc)
            )
        
        metrics = indicator.calculate_variance()
        
        assert metrics["drift_direction"] == "increasing"
    
    def test_risk_score_calculation(self, indicator):
        """Test risk score calculation."""
        # Add risky pattern
        indicator.record_position(Decimal("1000"), datetime.now(timezone.utc), "loss")
        indicator.record_position(Decimal("2000"), datetime.now(timezone.utc), "loss")
        indicator.record_position(Decimal("4000"), datetime.now(timezone.utc), "win")
        
        score = indicator.get_risk_score()
        
        assert score >= Decimal("0")
        assert score <= Decimal("100")
        assert score > Decimal("30")  # Should be elevated due to martingale


class TestCancelRateIndicator:
    """Test cancel rate indicator."""
    
    @pytest.fixture
    def indicator(self):
        """Create cancel rate indicator."""
        return CancelRateIndicator(window_minutes=60)
    
    def test_record_order_events(self, indicator):
        """Test recording order events."""
        timestamp = datetime.now(timezone.utc)
        
        indicator.record_order_placed("order1", timestamp)
        indicator.record_order_cancelled("order1", timestamp + timedelta(seconds=5))
        indicator.record_order_filled("order2", timestamp + timedelta(seconds=10))
        
        assert len(indicator.order_events) == 3
    
    def test_cancel_rate_calculation(self, indicator):
        """Test cancel rate calculation."""
        base_time = datetime.now(timezone.utc)
        
        # Place and cancel some orders
        indicator.record_order_placed("order1", base_time)
        indicator.record_order_cancelled("order1", base_time + timedelta(seconds=5))
        
        indicator.record_order_placed("order2", base_time)
        indicator.record_order_filled("order2", base_time + timedelta(seconds=10))
        
        indicator.record_order_placed("order3", base_time)
        indicator.record_order_cancelled("order3", base_time + timedelta(seconds=3))
        
        rate = indicator.calculate_cancel_rate()
        
        # 2 cancels out of 3 orders = 0.667
        assert rate > Decimal("0.6")
        assert rate < Decimal("0.7")
    
    def test_rapid_cancel_detection(self, indicator):
        """Test rapid cancellation detection."""
        base_time = datetime.now(timezone.utc)
        
        # Add rapid cancels
        for i in range(3):
            order_id = f"order{i}"
            indicator.record_order_placed(order_id, base_time)
            indicator.record_order_cancelled(order_id, base_time + timedelta(seconds=2))
        
        pattern = indicator.get_pattern_analysis()
        
        assert pattern["rapid_cancels_detected"] is True
        assert pattern["rapid_cancel_count"] >= 3
    
    def test_indecision_pattern_detection(self, indicator):
        """Test indecision pattern detection."""
        base_time = datetime.now(timezone.utc)
        
        # High cancellation rate
        for i in range(10):
            order_id = f"order{i}"
            indicator.record_order_placed(order_id, base_time + timedelta(seconds=i*10))
            
            if i < 7:  # Cancel 70% of orders
                indicator.record_order_cancelled(order_id, base_time + timedelta(seconds=i*10+5))
            else:
                indicator.record_order_filled(order_id, base_time + timedelta(seconds=i*10+15))
        
        assert indicator.detect_indecision_pattern(threshold=Decimal("0.5")) is True
    
    def test_streak_analysis(self, indicator):
        """Test consecutive cancellation streak analysis."""
        base_time = datetime.now(timezone.utc)
        
        # Create a streak of cancellations
        for i in range(5):
            order_id = f"cancel{i}"
            indicator.record_order_placed(order_id, base_time + timedelta(seconds=i*10))
            indicator.record_order_cancelled(order_id, base_time + timedelta(seconds=i*10+5))
        
        # Then some fills
        for i in range(2):
            order_id = f"fill{i}"
            indicator.record_order_placed(order_id, base_time + timedelta(seconds=100+i*10))
            indicator.record_order_filled(order_id, base_time + timedelta(seconds=100+i*10+5))
        
        streak = indicator.get_streak_analysis()
        
        assert streak["has_data"] is True
        assert streak["max_cancel_streak"] >= 5
        assert streak["streak_warning"] is True