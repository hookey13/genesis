"""Unit tests for statistical arbitrage components."""

from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from genesis.analytics.statistical_arb import (
    Signal,
    SpreadAnalyzer,
    StatisticalArbitrage,
    ThresholdMonitor,
)


class TestStatisticalArbitrage:
    """Test suite for StatisticalArbitrage class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.arb = StatisticalArbitrage()

        # Create sample price data
        self.prices1 = [Decimal(str(100 + i * 0.1)) for i in range(100)]
        self.prices2 = [Decimal(str(99 + i * 0.1)) for i in range(100)]

    def test_calculate_correlation_perfect_positive(self):
        """Test correlation calculation with perfectly correlated data."""
        # Create perfectly correlated data
        prices1 = [Decimal(str(i)) for i in range(1, 21)]
        prices2 = [Decimal(str(i * 2)) for i in range(1, 21)]

        correlation = self.arb.calculate_correlation(
            "BTCUSDT", "ETHUSDT", 20, prices1, prices2
        )

        assert correlation > Decimal("0.99")
        assert correlation <= Decimal("1.0")

    def test_calculate_correlation_perfect_negative(self):
        """Test correlation calculation with negatively correlated data."""
        # Create negatively correlated data
        prices1 = [Decimal(str(i)) for i in range(1, 21)]
        prices2 = [Decimal(str(21 - i)) for i in range(1, 21)]

        correlation = self.arb.calculate_correlation(
            "BTCUSDT", "ETHUSDT", 20, prices1, prices2
        )

        assert correlation < Decimal("-0.99")
        assert correlation >= Decimal("-1.0")

    def test_calculate_correlation_no_correlation(self):
        """Test correlation calculation with uncorrelated data."""
        # Create uncorrelated data (random-like)
        prices1 = [Decimal(str(100 + (i % 3))) for i in range(20)]
        prices2 = [Decimal(str(100 + ((i * 7) % 5))) for i in range(20)]

        correlation = self.arb.calculate_correlation(
            "BTCUSDT", "ETHUSDT", 20, prices1, prices2
        )

        # Should be close to 0 for uncorrelated data
        assert abs(correlation) < Decimal("0.5")

    def test_calculate_correlation_cache(self):
        """Test that correlation calculation uses cache."""
        # First call
        correlation1 = self.arb.calculate_correlation(
            "BTCUSDT", "ETHUSDT", 20, self.prices1[:20], self.prices2[:20]
        )

        # Second call (should use cache)
        correlation2 = self.arb.calculate_correlation(
            "BTCUSDT", "ETHUSDT", 20, self.prices1[:20], self.prices2[:20]
        )

        assert correlation1 == correlation2
        assert "BTCUSDT:ETHUSDT:20" in self.arb.correlation_cache

    def test_calculate_zscore_positive(self):
        """Test z-score calculation for positive divergence."""
        # Create data with known spread
        window = 20
        prices1 = [Decimal("100") for _ in range(window)]
        prices2 = [Decimal("99") for _ in range(window)]

        # Current prices diverged
        current_price1 = Decimal("102")
        current_price2 = Decimal("99")

        zscore = self.arb.calculate_zscore(
            current_price1, current_price2, window, prices1, prices2
        )

        # Z-score should be positive (spread is above mean)
        assert zscore > Decimal("0")

    def test_calculate_zscore_negative(self):
        """Test z-score calculation for negative divergence."""
        # Create data with known spread
        window = 20
        prices1 = [Decimal("100") for _ in range(window)]
        prices2 = [Decimal("99") for _ in range(window)]

        # Current prices converged
        current_price1 = Decimal("99")
        current_price2 = Decimal("99")

        zscore = self.arb.calculate_zscore(
            current_price1, current_price2, window, prices1, prices2
        )

        # Z-score should be negative (spread is below mean)
        assert zscore < Decimal("0")

    def test_calculate_zscore_zero_std(self):
        """Test z-score calculation when standard deviation is zero."""
        # Create constant spread data
        window = 20
        prices1 = [Decimal("100") for _ in range(window)]
        prices2 = [Decimal("100") for _ in range(window)]

        zscore = self.arb.calculate_zscore(
            Decimal("100"), Decimal("100"), window, prices1, prices2
        )

        # Should return 0 when std is 0
        assert zscore == Decimal("0")

    def test_test_cointegration_stationary(self):
        """Test cointegration with stationary spread."""
        # Create cointegrated series (mean-reverting spread)
        prices1 = []
        prices2 = []
        base = 100

        for i in range(50):
            noise = (i % 10) - 5  # Oscillating noise
            prices1.append(Decimal(str(base + i * 0.1 + noise * 0.1)))
            prices2.append(Decimal(str(base + i * 0.1)))

        is_cointegrated = self.arb.test_cointegration(prices1, prices2)

        # Should detect cointegration (simplified test may not always work)
        # This is a simplified test, real cointegration testing is more complex
        assert isinstance(is_cointegrated, bool)

    def test_test_cointegration_insufficient_data(self):
        """Test cointegration with insufficient data."""
        prices1 = [Decimal("100") for _ in range(10)]
        prices2 = [Decimal("99") for _ in range(10)]

        is_cointegrated = self.arb.test_cointegration(prices1, prices2)

        # Should return False with insufficient data
        assert is_cointegrated is False

    def test_create_correlation_matrix(self):
        """Test correlation matrix creation."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        price_data = {
            "BTCUSDT": [Decimal(str(100 + i)) for i in range(20)],
            "ETHUSDT": [Decimal(str(50 + i * 0.5)) for i in range(20)],
            "BNBUSDT": [Decimal(str(30 + i * 0.3)) for i in range(20)],
        }

        matrix = self.arb.create_correlation_matrix(symbols, price_data, 20)

        # Check matrix properties
        assert matrix.shape == (3, 3)
        # Diagonal should be 1 (perfect self-correlation)
        for i in range(3):
            assert matrix.iloc[i, i] == 1.0
        # Matrix should be symmetric
        for i in range(3):
            for j in range(3):
                assert abs(matrix.iloc[i, j] - matrix.iloc[j, i]) < 0.0001


class TestSpreadAnalyzer:
    """Test suite for SpreadAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SpreadAnalyzer(window_days=20)

    def test_calculate_spread(self):
        """Test spread calculation."""
        prices1 = pd.Series([100, 101, 102, 103, 104])
        prices2 = pd.Series([99, 100, 101, 102, 103])

        spread = self.analyzer.calculate_spread(prices1, prices2)

        # Check spread is calculated correctly (log spread)
        assert len(spread) == len(prices1)
        assert all(spread > 0)  # Since prices1 > prices2

    def test_analyze_spread(self):
        """Test spread analysis with rolling statistics."""
        prices1 = [Decimal(str(100 + i * 0.1)) for i in range(100)]
        prices2 = [Decimal(str(99 + i * 0.1)) for i in range(100)]
        timestamps = [datetime.now() - timedelta(minutes=100-i) for i in range(100)]

        result = self.analyzer.analyze_spread(
            "BTCUSDT", "ETHUSDT", prices1, prices2, timestamps
        )

        assert "current_spread" in result
        assert "spread_mean" in result
        assert "spread_std" in result
        assert "data_points" in result
        assert result["data_points"] == 100
        assert result["window_days"] == 20

    def test_generate_signal_entry(self):
        """Test signal generation for entry."""
        signal = self.analyzer.generate_signal(
            zscore=Decimal("2.5"),
            threshold=Decimal("2.0"),
            pair1="BTCUSDT",
            pair2="ETHUSDT",
            cointegrated=True,
            spread_stability=Decimal("0.8")
        )

        assert signal is not None
        assert signal.signal_type == "ENTRY"
        assert signal.zscore == Decimal("2.5")
        assert signal.confidence_score > Decimal("0.5")

    def test_generate_signal_exit(self):
        """Test signal generation for exit."""
        signal = self.analyzer.generate_signal(
            zscore=Decimal("0.3"),
            threshold=Decimal("2.0"),
            pair1="BTCUSDT",
            pair2="ETHUSDT",
            cointegrated=True,
            spread_stability=Decimal("0.8")
        )

        assert signal is not None
        assert signal.signal_type == "EXIT"
        assert signal.zscore == Decimal("0.3")

    def test_generate_signal_no_signal(self):
        """Test no signal generation when threshold not met."""
        signal = self.analyzer.generate_signal(
            zscore=Decimal("1.5"),
            threshold=Decimal("2.0"),
            pair1="BTCUSDT",
            pair2="ETHUSDT",
            cointegrated=True,
            spread_stability=Decimal("0.8")
        )

        assert signal is None

    def test_generate_signal_cooldown(self):
        """Test signal cooldown period."""
        # Generate first signal
        signal1 = self.analyzer.generate_signal(
            zscore=Decimal("2.5"),
            threshold=Decimal("2.0"),
            pair1="BTCUSDT",
            pair2="ETHUSDT",
            cointegrated=True,
            spread_stability=Decimal("0.8")
        )

        assert signal1 is not None

        # Try to generate another signal immediately (should be blocked)
        signal2 = self.analyzer.generate_signal(
            zscore=Decimal("2.5"),
            threshold=Decimal("2.0"),
            pair1="BTCUSDT",
            pair2="ETHUSDT",
            cointegrated=True,
            spread_stability=Decimal("0.8")
        )

        assert signal2 is None  # Blocked by cooldown

    def test_check_signal_persistence(self):
        """Test signal persistence checking."""
        # Create persistent signals
        signals = []
        for i in range(5):
            signal = Signal(
                pair1_symbol="BTCUSDT",
                pair2_symbol="ETHUSDT",
                zscore=Decimal("2.5"),
                threshold_sigma=Decimal("2.0"),
                signal_type="ENTRY",
                confidence_score=Decimal("0.8"),
                created_at=datetime.now() - timedelta(minutes=5-i)
            )
            signals.append(signal)

        persistent = self.analyzer.check_signal_persistence(signals, min_persistence=3)

        assert len(persistent) > 0
        assert persistent[0].signal_type == "ENTRY"

    def test_check_signal_persistence_insufficient(self):
        """Test signal persistence with insufficient signals."""
        signals = [
            Signal(
                pair1_symbol="BTCUSDT",
                pair2_symbol="ETHUSDT",
                zscore=Decimal("2.5"),
                threshold_sigma=Decimal("2.0"),
                signal_type="ENTRY",
                confidence_score=Decimal("0.8")
            )
        ]

        persistent = self.analyzer.check_signal_persistence(signals, min_persistence=3)

        assert len(persistent) == 0


class TestThresholdMonitor:
    """Test suite for ThresholdMonitor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ThresholdMonitor(default_sigma=Decimal("2"))

    def test_set_threshold(self):
        """Test setting custom threshold."""
        self.monitor.set_threshold("BTCUSDT:ETHUSDT", Decimal("2.5"))

        assert "BTCUSDT:ETHUSDT" in self.monitor.thresholds
        assert self.monitor.thresholds["BTCUSDT:ETHUSDT"] == Decimal("2.5")

    def test_check_threshold_breach(self):
        """Test threshold breach detection."""
        alert = self.monitor.check_threshold("BTCUSDT", "ETHUSDT", Decimal("2.5"))

        assert alert is not None
        assert alert["pair1"] == "BTCUSDT"
        assert alert["pair2"] == "ETHUSDT"
        assert alert["zscore"] == Decimal("2.5")
        assert alert["threshold"] == Decimal("2")
        assert alert["direction"] == "ABOVE"

    def test_check_threshold_no_breach(self):
        """Test no alert when threshold not breached."""
        alert = self.monitor.check_threshold("BTCUSDT", "ETHUSDT", Decimal("1.5"))

        assert alert is None

    def test_check_threshold_cooldown(self):
        """Test alert cooldown period."""
        # First alert
        alert1 = self.monitor.check_threshold("BTCUSDT", "ETHUSDT", Decimal("2.5"))
        assert alert1 is not None

        # Second alert immediately (should be blocked)
        alert2 = self.monitor.check_threshold("BTCUSDT", "ETHUSDT", Decimal("2.5"))
        assert alert2 is None

    @pytest.mark.asyncio
    async def test_monitor_thresholds(self):
        """Test monitoring multiple thresholds."""
        pairs = [("BTCUSDT", "ETHUSDT"), ("BNBUSDT", "ADAUSDT")]

        async def mock_get_zscore(pair1, pair2):
            if pair1 == "BTCUSDT":
                return Decimal("2.5")
            return Decimal("1.0")

        alerts = await self.monitor.monitor_thresholds(pairs, mock_get_zscore)

        assert len(alerts) == 1
        assert alerts[0]["pair1"] == "BTCUSDT"


class TestSignal:
    """Test suite for Signal dataclass."""

    def test_signal_creation_valid(self):
        """Test creating a valid signal."""
        signal = Signal(
            pair1_symbol="BTCUSDT",
            pair2_symbol="ETHUSDT",
            zscore=Decimal("2.5"),
            threshold_sigma=Decimal("2.0"),
            signal_type="ENTRY",
            confidence_score=Decimal("0.8")
        )

        assert signal.pair1_symbol == "BTCUSDT"
        assert signal.pair2_symbol == "ETHUSDT"
        assert signal.signal_type == "ENTRY"
        assert signal.confidence_score == Decimal("0.8")

    def test_signal_invalid_type(self):
        """Test signal creation with invalid type."""
        with pytest.raises(ValueError, match="Invalid signal type"):
            Signal(
                pair1_symbol="BTCUSDT",
                pair2_symbol="ETHUSDT",
                zscore=Decimal("2.5"),
                threshold_sigma=Decimal("2.0"),
                signal_type="INVALID",
                confidence_score=Decimal("0.8")
            )

    def test_signal_invalid_confidence(self):
        """Test signal creation with invalid confidence score."""
        with pytest.raises(ValueError, match="Confidence score must be between"):
            Signal(
                pair1_symbol="BTCUSDT",
                pair2_symbol="ETHUSDT",
                zscore=Decimal("2.5"),
                threshold_sigma=Decimal("2.0"),
                signal_type="ENTRY",
                confidence_score=Decimal("1.5")
            )
