"""Unit tests for behavioral anomaly detection."""
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from genesis.tilt.baseline import BehavioralMetric
from genesis.tilt.indicators.mouse_patterns import MousePatternsIndicator
from genesis.tilt.indicators.revenge_trading import RevengeTradingDetector
from genesis.tilt.indicators.typing_speed import TypingSpeedIndicator


class TestRevengeTradingDetector:
    """Test revenge trading detection."""

    @pytest.fixture
    def detector(self):
        """Create revenge trading detector."""
        return RevengeTradingDetector(
            loss_streak_threshold=3,
            time_window_minutes=30,
            size_multiplier_threshold=Decimal('1.5')
        )

    def test_record_trade_result_loss(self, detector):
        """Test recording a trading loss."""
        profile_id = "test_profile"

        # Record a loss
        detector.record_trade_result(
            profile_id=profile_id,
            pnl=Decimal("-100"),
            symbol="BTC/USDT",
            position_size=Decimal("1000")
        )

        # Check loss was recorded
        assert detector.consecutive_losses[profile_id] == 1
        assert len(detector.loss_history[profile_id]) == 1
        assert detector.last_position_sizes[profile_id] == Decimal("1000")

    def test_record_trade_result_profit(self, detector):
        """Test recording a profitable trade resets streak."""
        profile_id = "test_profile"

        # Record losses first
        for i in range(3):
            detector.record_trade_result(
                profile_id=profile_id,
                pnl=Decimal("-100"),
                symbol="BTC/USDT",
                position_size=Decimal("1000")
            )

        assert detector.consecutive_losses[profile_id] == 3

        # Record a profit
        detector.record_trade_result(
            profile_id=profile_id,
            pnl=Decimal("100"),
            symbol="BTC/USDT",
            position_size=Decimal("1000")
        )

        # Streak should reset
        assert detector.consecutive_losses[profile_id] == 0

    def test_detect_revenge_pattern_size_increase(self, detector):
        """Test detection of revenge pattern through size increase."""
        profile_id = "test_profile"

        # Record 3 consecutive losses
        for i in range(3):
            detector.record_trade_result(
                profile_id=profile_id,
                pnl=Decimal("-100"),
                symbol="BTC/USDT",
                position_size=Decimal("1000")
            )

        # Create metric with increased position size
        metric = BehavioralMetric(
            metric_name="position_size",
            value=2000.0,  # Doubled position size
            timestamp=datetime.now(UTC),
            context={}
        )

        # Detect pattern
        result = detector.detect_revenge_pattern(profile_id, metric)

        # Assert pattern detected
        assert result is not None
        assert result['pattern'] == 'revenge_trading'
        assert result['consecutive_losses'] == 3
        assert result['position_size_increase'] == 2.0
        assert result['severity'] == 6  # min(3*2, 10)

    def test_detect_revenge_pattern_rapid_trading(self, detector):
        """Test detection of rapid trading after losses."""
        profile_id = "test_profile"
        now = datetime.now(UTC)

        # Record recent losses
        for i in range(3):
            detector.record_trade_result(
                profile_id=profile_id,
                pnl=Decimal("-100"),
                symbol="BTC/USDT",
                position_size=Decimal("1000"),
                timestamp=now - timedelta(seconds=30+i*10)  # Recent losses
            )

        # Create order frequency metric (rapid trading)
        metric = BehavioralMetric(
            metric_name="order_frequency",
            value=10.0,  # High frequency
            timestamp=now,
            context={}
        )

        # Detect pattern
        result = detector.detect_revenge_pattern(profile_id, metric)

        # Assert pattern detected
        assert result is not None
        assert result['pattern'] == 'revenge_trading_speed'
        assert result['consecutive_losses'] == 3
        assert result['severity'] == 7

    def test_no_revenge_pattern_below_threshold(self, detector):
        """Test no pattern detected below loss threshold."""
        profile_id = "test_profile"

        # Record only 2 losses (below threshold of 3)
        for i in range(2):
            detector.record_trade_result(
                profile_id=profile_id,
                pnl=Decimal("-100"),
                symbol="BTC/USDT",
                position_size=Decimal("1000")
            )

        # Create metric
        metric = BehavioralMetric(
            metric_name="position_size",
            value=2000.0,
            timestamp=datetime.now(UTC),
            context={}
        )

        # No pattern should be detected
        result = detector.detect_revenge_pattern(profile_id, metric)
        assert result is None

    def test_clean_old_losses(self, detector):
        """Test that old losses are cleaned up."""
        profile_id = "test_profile"
        now = datetime.now(UTC)

        # Record old and recent losses
        old_timestamp = now - timedelta(minutes=40)  # Outside 30-minute window
        recent_timestamp = now - timedelta(minutes=10)

        detector.record_trade_result(
            profile_id=profile_id,
            pnl=Decimal("-100"),
            symbol="BTC/USDT",
            position_size=Decimal("1000"),
            timestamp=old_timestamp
        )

        detector.record_trade_result(
            profile_id=profile_id,
            pnl=Decimal("-100"),
            symbol="BTC/USDT",
            position_size=Decimal("1000"),
            timestamp=recent_timestamp
        )

        # Old loss should be cleaned
        assert len(detector.loss_history[profile_id]) == 1
        assert detector.loss_history[profile_id][0].timestamp == recent_timestamp

    def test_reset_profile(self, detector):
        """Test resetting profile data."""
        profile_id = "test_profile"

        # Add some data
        detector.record_trade_result(
            profile_id=profile_id,
            pnl=Decimal("-100"),
            symbol="BTC/USDT",
            position_size=Decimal("1000")
        )

        # Reset
        detector.reset_profile(profile_id)

        # Check data is cleared
        assert len(detector.loss_history[profile_id]) == 0
        assert detector.consecutive_losses[profile_id] == 0
        assert profile_id not in detector.last_position_sizes


class TestTypingSpeedIndicator:
    """Test typing speed behavioral indicator."""

    @pytest.fixture
    def indicator(self):
        """Create typing speed indicator."""
        return TypingSpeedIndicator(
            window_size=50,
            burst_threshold_wpm=120,
            slow_threshold_wpm=20
        )

    def test_record_keystroke_event(self, indicator):
        """Test recording keystroke events."""
        # Record normal typing
        result = indicator.record_keystroke_event(
            key_count=50,
            duration_ms=3000  # 50 keys in 3 seconds
        )

        assert result['has_data']
        assert result['sample_count'] == 1
        assert len(indicator.keystroke_events) == 1

    def test_detect_typing_burst(self, indicator):
        """Test detection of typing bursts."""
        # Record a burst (fast typing)
        result = indicator.record_keystroke_event(
            key_count=100,
            duration_ms=2000  # 150 WPM (100 chars / 5 chars per word / (2/60) minutes)
        )

        assert indicator.burst_count == 1
        assert indicator.last_burst_time is not None

    def test_analyze_typing_patterns(self, indicator):
        """Test analysis of typing patterns."""
        # Record various typing speeds
        speeds = [
            (50, 3000),   # Normal
            (100, 2000),  # Fast
            (20, 4000),   # Slow
            (75, 3000),   # Normal-fast
        ]

        for key_count, duration in speeds:
            indicator.record_keystroke_event(key_count, duration)

        analysis = indicator.analyze_typing_patterns()

        assert analysis['has_data']
        assert analysis['sample_count'] == 4
        assert 'avg_wpm' in analysis
        assert 'max_wpm' in analysis
        assert 'min_wpm' in analysis
        assert analysis['burst_detected']  # One burst was recorded

    def test_detect_stress_typing(self, indicator):
        """Test detection of stress-induced typing patterns."""
        # Record erratic typing pattern
        for i in range(10):
            if i % 2 == 0:
                # Fast bursts
                indicator.record_keystroke_event(100, 1500)
            else:
                # Slow typing
                indicator.record_keystroke_event(20, 3000)

        # Check for stress detection
        stress = indicator.detect_stress_typing()

        assert stress is not None
        assert stress['stress_detected']
        assert 'rapid_bursts' in stress['indicators']
        assert stress['severity'] > 0

    def test_calculate_wpm(self, indicator):
        """Test WPM calculation."""
        # Test normal typing: 50 chars in 3 seconds
        # 50 chars / 5 chars per word = 10 words
        # 3000ms = 0.05 minutes
        # 10 words / 0.05 minutes = 200 WPM
        wpm = indicator._calculate_wpm(50, 3000)
        assert abs(wpm - 200) < 1

        # Test edge cases
        assert indicator._calculate_wpm(0, 1000) == 0
        assert indicator._calculate_wpm(10, 0) == 0

    def test_reset(self, indicator):
        """Test resetting indicator state."""
        # Add some data
        indicator.record_keystroke_event(50, 3000)
        indicator.burst_count = 5

        # Reset
        indicator.reset()

        # Check state is cleared
        assert len(indicator.keystroke_events) == 0
        assert indicator.burst_count == 0
        assert indicator.last_burst_time is None


class TestMousePatternsIndicator:
    """Test mouse pattern behavioral indicator."""

    @pytest.fixture
    def indicator(self):
        """Create mouse patterns indicator."""
        return MousePatternsIndicator(
            window_size=50,
            rapid_click_threshold_ms=200,
            jitter_threshold_pixels=5
        )

    def test_record_click(self, indicator):
        """Test recording mouse clicks."""
        result = indicator.record_click(
            position=(100, 200),
            duration_ms=100
        )

        assert result['has_data']
        assert result['click_count'] == 1
        assert len(indicator.mouse_events) == 1
        assert len(indicator.click_times) == 1

    def test_detect_rapid_clicking(self, indicator):
        """Test detection of rapid clicking."""
        now = datetime.now(UTC)

        # Record rapid clicks
        indicator.record_click((100, 200), 50, now)
        indicator.record_click((100, 200), 50, now + timedelta(milliseconds=150))

        assert indicator.rapid_click_count == 1

    def test_detect_double_click(self, indicator):
        """Test detection of double clicks."""
        now = datetime.now(UTC)

        # Record double click (within 500ms)
        indicator.record_click((100, 200), 50, now)
        indicator.record_click((100, 200), 50, now + timedelta(milliseconds=300))

        assert indicator.double_click_count == 1

    def test_record_movement(self, indicator):
        """Test recording mouse movements."""
        result = indicator.record_movement(
            start_pos=(100, 100),
            end_pos=(200, 200),
            duration_ms=500
        )

        assert result['has_data']
        assert result['movement_count'] == 1
        assert len(indicator.movement_velocities) == 1

    def test_detect_jitter(self, indicator):
        """Test detection of jittery movements."""
        # Record small, rapid movements (jitter)
        for i in range(10):
            indicator.record_movement(
                start_pos=(100, 100),
                end_pos=(102, 102),  # Small movement (< 5 pixels)
                duration_ms=50
            )

        assert indicator.jitter_count == 10

    def test_analyze_patterns(self, indicator):
        """Test comprehensive pattern analysis."""
        # Record various mouse activities
        now = datetime.now(UTC)

        # Clicks
        for i in range(5):
            indicator.record_click((100 + i*10, 200), 100, now + timedelta(seconds=i))

        # Movements
        for i in range(5):
            indicator.record_movement(
                start_pos=(100, 100),
                end_pos=(200 + i*50, 200),
                duration_ms=200
            )

        analysis = indicator.analyze_patterns()

        assert analysis['has_data']
        assert analysis['click_count'] == 5
        assert analysis['movement_count'] == 5
        assert 'click_rate_per_second' in analysis
        assert 'avg_velocity_pps' in analysis

    def test_detect_stress_patterns(self, indicator):
        """Test detection of stress-induced mouse patterns."""
        now = datetime.now(UTC)

        # Simulate stress pattern: rapid clicking + erratic movement
        for i in range(10):
            # Rapid clicks
            indicator.record_click((100, 200), 50, now + timedelta(milliseconds=i*100))

            # Erratic movements
            if i % 2 == 0:
                indicator.record_movement((100, 100), (500, 500), 100)  # Fast
            else:
                indicator.record_movement((100, 100), (110, 110), 500)  # Slow/jittery

        stress = indicator.detect_stress_patterns()

        assert stress is not None
        assert stress['stress_detected']
        assert 'rapid_clicking' in stress['indicators']
        assert stress['severity'] > 0

    def test_reset(self, indicator):
        """Test resetting indicator state."""
        # Add some data
        indicator.record_click((100, 200), 100)
        indicator.rapid_click_count = 5
        indicator.jitter_count = 10

        # Reset
        indicator.reset()

        # Check state is cleared
        assert len(indicator.mouse_events) == 0
        assert len(indicator.click_times) == 0
        assert indicator.rapid_click_count == 0
        assert indicator.jitter_count == 0
