"""
Unit tests for Risk Metrics Calculator.
"""

from decimal import Decimal

import pytest

from genesis.analytics.risk_metrics import RiskMetrics, RiskMetricsCalculator


class TestRiskMetrics:
    """Test RiskMetrics dataclass."""

    def test_risk_metrics_creation(self):
        """Test creating a RiskMetrics object."""
        metrics = RiskMetrics(
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("3.0"),
            max_drawdown=Decimal("0.15"),
            max_drawdown_duration_days=30,
            volatility=Decimal("0.2"),
            downside_deviation=Decimal("0.1"),
            value_at_risk_95=Decimal("0.05"),
            conditional_value_at_risk_95=Decimal("0.08"),
            beta=Decimal("1.2"),
            alpha=Decimal("0.05"),
        )

        assert metrics.sharpe_ratio == Decimal("1.5")
        assert metrics.sortino_ratio == Decimal("2.0")
        assert metrics.max_drawdown_duration_days == 30
        assert metrics.beta == Decimal("1.2")

    def test_risk_metrics_to_dict(self):
        """Test converting RiskMetrics to dictionary."""
        metrics = RiskMetrics(
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("3.0"),
            max_drawdown=Decimal("0.15"),
            max_drawdown_duration_days=30,
            volatility=Decimal("0.2"),
            downside_deviation=Decimal("0.1"),
            value_at_risk_95=Decimal("0.05"),
            conditional_value_at_risk_95=Decimal("0.08"),
        )

        result = metrics.to_dict()

        assert result["sharpe_ratio"] == "1.5"
        assert result["sortino_ratio"] == "2.0"
        assert result["max_drawdown_duration_days"] == 30
        assert result["beta"] is None  # Not set


class TestRiskMetricsCalculator:
    """Test RiskMetricsCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a RiskMetricsCalculator instance."""
        return RiskMetricsCalculator(risk_free_rate=Decimal("0.04"))

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns for testing."""
        # Mix of positive and negative returns
        returns = [
            Decimal("0.01"),  # 1% gain
            Decimal("0.02"),  # 2% gain
            Decimal("-0.015"),  # 1.5% loss
            Decimal("0.005"),  # 0.5% gain
            Decimal("-0.01"),  # 1% loss
            Decimal("0.03"),  # 3% gain
            Decimal("-0.005"),  # 0.5% loss
            Decimal("0.015"),  # 1.5% gain
            Decimal("0.01"),  # 1% gain
            Decimal("-0.02"),  # 2% loss
        ]
        return returns

    @pytest.fixture
    def positive_returns(self):
        """Create all positive returns for testing."""
        return [
            Decimal("0.01"),
            Decimal("0.02"),
            Decimal("0.015"),
            Decimal("0.025"),
            Decimal("0.01"),
        ]

    @pytest.fixture
    def benchmark_returns(self):
        """Create benchmark returns for testing."""
        returns = [
            Decimal("0.008"),  # 0.8% gain
            Decimal("0.015"),  # 1.5% gain
            Decimal("-0.01"),  # 1% loss
            Decimal("0.003"),  # 0.3% gain
            Decimal("-0.008"),  # 0.8% loss
            Decimal("0.02"),  # 2% gain
            Decimal("-0.003"),  # 0.3% loss
            Decimal("0.01"),  # 1% gain
            Decimal("0.008"),  # 0.8% gain
            Decimal("-0.015"),  # 1.5% loss
        ]
        return returns

    def test_calculate_mean(self, calculator):
        """Test mean calculation."""
        values = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        mean = calculator._calculate_mean(values)
        assert mean == Decimal("3")

    def test_calculate_mean_empty(self, calculator):
        """Test mean calculation with empty list."""
        mean = calculator._calculate_mean([])
        assert mean == Decimal("0")

    def test_calculate_variance(self, calculator):
        """Test variance calculation."""
        values = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        variance = calculator._calculate_variance(values)
        assert variance == Decimal("2")  # Variance of [1,2,3,4,5] is 2

    def test_calculate_volatility(self, calculator):
        """Test volatility (standard deviation) calculation."""
        values = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4"), Decimal("5")]
        volatility = calculator._calculate_volatility(values)
        # sqrt(2) ≈ 1.414
        assert abs(volatility - Decimal("1.414")) < Decimal("0.001")

    def test_calculate_sharpe_ratio(self, calculator, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = calculator.calculate_sharpe_ratio(sample_returns, periods_per_year=252)

        # Sharpe should be positive but modest given mixed returns
        assert sharpe > Decimal("0")
        assert sharpe < Decimal("5")  # Sanity check

    def test_calculate_sharpe_ratio_zero_volatility(self, calculator):
        """Test Sharpe ratio with zero volatility."""
        # All returns are the same (no volatility)
        returns = [Decimal("0.01")] * 10
        sharpe = calculator.calculate_sharpe_ratio(returns)

        # Should handle zero volatility gracefully
        assert sharpe == Decimal("0")

    def test_calculate_sortino_ratio(self, calculator, sample_returns):
        """Test Sortino ratio calculation."""
        sortino, downside_dev = calculator.calculate_sortino_ratio(sample_returns)

        # Sortino should typically be higher than Sharpe
        # since it only considers downside volatility
        assert sortino > Decimal("0")
        assert downside_dev > Decimal("0")

    def test_calculate_sortino_ratio_no_losses(self, calculator, positive_returns):
        """Test Sortino ratio with no negative returns."""
        sortino, downside_dev = calculator.calculate_sortino_ratio(positive_returns)

        # With no losses, Sortino should be capped at 999.99
        assert sortino == Decimal("999.99")
        assert downside_dev == Decimal("0")

    def test_negative_risk_free_rate(self):
        """Test calculator with negative risk-free rate."""
        # Negative rates are allowed (e.g., European bonds in recent years)
        calculator = RiskMetricsCalculator(risk_free_rate=Decimal("-0.02"))

        assert calculator.risk_free_rate == Decimal("-0.02")
        assert calculator._daily_risk_free == Decimal("-0.02") / Decimal("365")

        # Calculate Sharpe with negative risk-free rate
        returns = [Decimal("0.01"), Decimal("-0.005"), Decimal("0.02")]
        sharpe = calculator.calculate_sharpe_ratio(returns)

        # With negative risk-free rate, excess return is higher
        assert sharpe > Decimal("0")

    def test_extremely_negative_risk_free_rate_warning(self, caplog):
        """Test warning for extremely negative risk-free rate."""
        import structlog
        from structlog.testing import LogCapture
        
        # Configure structlog to use LogCapture for testing
        log_output = LogCapture()
        structlog.configure(
            processors=[log_output]
        )

        # This should trigger a warning
        calculator = RiskMetricsCalculator(risk_free_rate=Decimal("-0.15"))

        # Check that the warning was logged
        assert len(log_output.entries) == 1
        assert "Extremely negative risk-free rate" in str(log_output.entries[0]["event"])
        assert calculator.risk_free_rate == Decimal("-0.15")
        
        # Reset structlog configuration
        structlog.configure()

    def test_calculate_max_drawdown(self, calculator):
        """Test maximum drawdown calculation."""
        # Create returns that will produce a drawdown
        returns = [
            Decimal("0.1"),  # 10% gain (cum: 1.1)
            Decimal("0.05"),  # 5% gain (cum: 1.155)
            Decimal("-0.2"),  # 20% loss (cum: 0.924)
            Decimal("-0.1"),  # 10% loss (cum: 0.8316)
            Decimal("0.05"),  # 5% gain (cum: 0.87318)
        ]

        max_dd, duration = calculator.calculate_max_drawdown(returns)

        # Max drawdown from 1.155 to 0.8316 = ~28%
        assert max_dd > Decimal("0.25")
        assert max_dd < Decimal("0.30")
        assert duration > 0

    def test_calculate_max_drawdown_no_drawdown(self, calculator):
        """Test max drawdown with continuously rising returns."""
        returns = [Decimal("0.01"), Decimal("0.02"), Decimal("0.03")]
        max_dd, duration = calculator.calculate_max_drawdown(returns)

        assert max_dd == Decimal("0")
        assert duration == 0

    def test_calculate_calmar_ratio(self, calculator):
        """Test Calmar ratio calculation."""
        mean_return = Decimal("0.001")  # 0.1% daily return
        max_drawdown = Decimal("0.15")  # 15% max drawdown

        calmar = calculator.calculate_calmar_ratio(mean_return, max_drawdown)

        # Annual return ~25%, max DD 15%, so Calmar ~1.67
        assert calmar > Decimal("1.5")
        assert calmar < Decimal("2")

    def test_calculate_calmar_ratio_zero_drawdown(self, calculator):
        """Test Calmar ratio with zero drawdown."""
        mean_return = Decimal("0.001")
        max_drawdown = Decimal("0")

        calmar = calculator.calculate_calmar_ratio(mean_return, max_drawdown)

        assert calmar == Decimal("999.99")  # Capped value

    def test_calculate_value_at_risk(self, calculator, sample_returns):
        """Test Value at Risk calculation."""
        var_95 = calculator.calculate_value_at_risk(sample_returns, Decimal("0.95"))

        # VaR should be positive (representing potential loss)
        assert var_95 > Decimal("0")
        assert var_95 <= abs(min(sample_returns))  # Can't exceed worst return

    def test_calculate_conditional_value_at_risk(self, calculator, sample_returns):
        """Test Conditional Value at Risk (CVaR) calculation."""
        cvar_95 = calculator.calculate_conditional_value_at_risk(
            sample_returns, Decimal("0.95")
        )

        # CVaR should be >= VaR
        var_95 = calculator.calculate_value_at_risk(sample_returns, Decimal("0.95"))
        assert cvar_95 >= var_95

    def test_calculate_beta(self, calculator, sample_returns, benchmark_returns):
        """Test beta calculation."""
        beta = calculator.calculate_beta(sample_returns, benchmark_returns)

        # Beta should be around 1 for similar volatility
        assert beta > Decimal("0.5")
        assert beta < Decimal("2")

    def test_calculate_beta_mismatched_lengths(self, calculator):
        """Test beta with mismatched return lengths."""
        returns = [Decimal("0.01"), Decimal("0.02")]
        benchmark = [Decimal("0.01")]

        beta = calculator.calculate_beta(returns, benchmark)
        assert beta == Decimal("0")

    def test_calculate_alpha(self, calculator, sample_returns, benchmark_returns):
        """Test Jensen's alpha calculation."""
        beta = calculator.calculate_beta(sample_returns, benchmark_returns)
        alpha = calculator.calculate_alpha(sample_returns, benchmark_returns, beta)

        # Alpha represents excess return
        assert isinstance(alpha, Decimal)
        # Can be positive or negative
        assert abs(alpha) < Decimal("1")  # Sanity check

    def test_calculate_information_ratio(
        self, calculator, sample_returns, benchmark_returns
    ):
        """Test Information Ratio calculation."""
        info_ratio = calculator.calculate_information_ratio(
            sample_returns, benchmark_returns
        )

        # Information ratio measures active return vs tracking error
        assert isinstance(info_ratio, Decimal)
        assert abs(info_ratio) < Decimal("10")  # Sanity check

    def test_calculate_treynor_ratio(self, calculator):
        """Test Treynor Ratio calculation."""
        mean_return = Decimal("0.001")  # 0.1% daily
        beta = Decimal("1.2")

        treynor = calculator.calculate_treynor_ratio(mean_return, beta)

        # Treynor measures excess return per unit of systematic risk
        assert treynor > Decimal("0")

    def test_calculate_treynor_ratio_zero_beta(self, calculator):
        """Test Treynor ratio with zero beta."""
        mean_return = Decimal("0.001")
        beta = Decimal("0")

        treynor = calculator.calculate_treynor_ratio(mean_return, beta)
        assert treynor == Decimal("0")

    def test_calculate_metrics_comprehensive(
        self, calculator, sample_returns, benchmark_returns
    ):
        """Test comprehensive metrics calculation."""
        metrics = calculator.calculate_metrics(
            sample_returns, period="daily", benchmark_returns=benchmark_returns
        )

        # Check all metrics are calculated
        assert metrics.sharpe_ratio != Decimal("0")
        assert metrics.sortino_ratio != Decimal("0")
        assert metrics.calmar_ratio != Decimal("0")
        assert metrics.max_drawdown >= Decimal("0")
        assert metrics.volatility > Decimal("0")
        assert metrics.downside_deviation >= Decimal("0")
        assert metrics.value_at_risk_95 >= Decimal("0")
        assert metrics.conditional_value_at_risk_95 >= Decimal("0")
        assert metrics.beta is not None
        assert metrics.alpha is not None
        assert metrics.information_ratio is not None
        assert metrics.treynor_ratio is not None

    def test_calculate_metrics_no_benchmark(self, calculator, sample_returns):
        """Test metrics calculation without benchmark."""
        metrics = calculator.calculate_metrics(sample_returns, period="daily")

        # Basic metrics should be calculated
        assert metrics.sharpe_ratio != Decimal("0")
        assert metrics.sortino_ratio != Decimal("0")

        # Benchmark-relative metrics should be None
        assert metrics.beta is None
        assert metrics.alpha is None
        assert metrics.information_ratio is None
        assert metrics.treynor_ratio is None

    def test_calculate_metrics_empty_returns(self, calculator):
        """Test metrics with empty returns."""
        metrics = calculator.calculate_metrics([])

        # All metrics should be zero
        assert metrics.sharpe_ratio == Decimal("0")
        assert metrics.sortino_ratio == Decimal("0")
        assert metrics.calmar_ratio == Decimal("0")
        assert metrics.max_drawdown == Decimal("0")

    def test_calculate_rolling_metrics(self, calculator, sample_returns):
        """Test rolling window metrics calculation."""
        window_size = 5
        rolling = calculator.calculate_rolling_metrics(
            sample_returns, window_size, period="daily"
        )

        # Should have (len - window + 1) results
        expected_count = len(sample_returns) - window_size + 1
        assert len(rolling) == expected_count

        # Each result should be valid metrics
        for metrics in rolling:
            assert isinstance(metrics.sharpe_ratio, Decimal)
            assert isinstance(metrics.sortino_ratio, Decimal)

    def test_calculate_rolling_metrics_insufficient_data(self, calculator):
        """Test rolling metrics with insufficient data."""
        returns = [Decimal("0.01"), Decimal("0.02")]
        window_size = 5

        rolling = calculator.calculate_rolling_metrics(returns, window_size)

        assert len(rolling) == 0

    def test_get_periods_per_year(self, calculator):
        """Test period mapping for annualization."""
        assert calculator._get_periods_per_year("hourly") == 24 * 365
        assert calculator._get_periods_per_year("daily") == 252
        assert calculator._get_periods_per_year("weekly") == 52
        assert calculator._get_periods_per_year("monthly") == 12
        assert calculator._get_periods_per_year("quarterly") == 4
        assert calculator._get_periods_per_year("yearly") == 1
        assert calculator._get_periods_per_year("unknown") == 252  # Default
