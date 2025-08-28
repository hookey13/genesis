"""
Risk-adjusted metrics calculations for Project GENESIS.

This module implements Sharpe ratio, Sortino ratio, Calmar ratio,
and other risk-adjusted performance metrics.
"""

from dataclasses import dataclass
from decimal import Decimal

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RiskMetrics:
    """Container for risk-adjusted performance metrics."""

    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    calmar_ratio: Decimal
    max_drawdown: Decimal
    max_drawdown_duration_days: int
    volatility: Decimal
    downside_deviation: Decimal
    value_at_risk_95: Decimal
    conditional_value_at_risk_95: Decimal
    beta: Decimal | None = None
    alpha: Decimal | None = None
    information_ratio: Decimal | None = None
    treynor_ratio: Decimal | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "sharpe_ratio": str(self.sharpe_ratio),
            "sortino_ratio": str(self.sortino_ratio),
            "calmar_ratio": str(self.calmar_ratio),
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "volatility": str(self.volatility),
            "downside_deviation": str(self.downside_deviation),
            "value_at_risk_95": str(self.value_at_risk_95),
            "conditional_value_at_risk_95": str(self.conditional_value_at_risk_95),
            "beta": str(self.beta) if self.beta else None,
            "alpha": str(self.alpha) if self.alpha else None,
            "information_ratio": (
                str(self.information_ratio) if self.information_ratio else None
            ),
            "treynor_ratio": str(self.treynor_ratio) if self.treynor_ratio else None,
        }


class RiskMetricsCalculator:
    """Calculator for risk-adjusted performance metrics."""

    def __init__(self, risk_free_rate: Decimal = Decimal("0.04")):
        """
        Initialize the risk metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 4%)
                           Can be negative (e.g., during negative rate environments)
        """
        # Allow negative risk-free rates (e.g., negative interest rate environments)
        # but warn if extremely negative
        if risk_free_rate < Decimal("-0.1"):
            logger.warning(f"Extremely negative risk-free rate: {risk_free_rate}")

        self.risk_free_rate = risk_free_rate
        self._daily_risk_free = risk_free_rate / Decimal("365")

    def calculate_metrics(
        self,
        returns: list[Decimal],
        period: str = "daily",
        benchmark_returns: list[Decimal] | None = None,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics from returns series.

        Args:
            returns: List of periodic returns (as decimals, not percentages)
            period: Return period ('daily', 'hourly', 'weekly', 'monthly')
            benchmark_returns: Optional benchmark returns for relative metrics

        Returns:
            RiskMetrics object with calculated values
        """
        if not returns:
            return self._empty_metrics()

        # Convert period to annual scaling factor
        periods_per_year = self._get_periods_per_year(period)

        # Calculate basic statistics
        avg_return = self._calculate_mean(returns)
        volatility = self._calculate_volatility(returns)

        # Calculate Sharpe ratio
        sharpe_ratio = self.calculate_sharpe_ratio(
            returns, volatility, periods_per_year
        )

        # Calculate Sortino ratio (downside deviation)
        sortino_ratio, downside_dev = self.calculate_sortino_ratio(
            returns, periods_per_year
        )

        # Calculate maximum drawdown and Calmar ratio
        max_drawdown, dd_duration = self.calculate_max_drawdown(returns)
        calmar_ratio = self.calculate_calmar_ratio(
            avg_return, max_drawdown, periods_per_year
        )

        # Calculate Value at Risk and CVaR
        var_95 = self.calculate_value_at_risk(returns, Decimal("0.95"))
        cvar_95 = self.calculate_conditional_value_at_risk(returns, Decimal("0.95"))

        # Calculate relative metrics if benchmark provided
        beta = None
        alpha = None
        information_ratio = None
        treynor_ratio = None

        if benchmark_returns and len(benchmark_returns) == len(returns):
            beta = self.calculate_beta(returns, benchmark_returns)
            alpha = self.calculate_alpha(
                returns, benchmark_returns, beta, periods_per_year
            )
            information_ratio = self.calculate_information_ratio(
                returns, benchmark_returns, periods_per_year
            )
            if beta and beta != Decimal("0"):
                treynor_ratio = self.calculate_treynor_ratio(
                    avg_return, beta, periods_per_year
                )

        return RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration_days=dd_duration,
            volatility=volatility,
            downside_deviation=downside_dev,
            value_at_risk_95=var_95,
            conditional_value_at_risk_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
        )

    def calculate_sharpe_ratio(
        self,
        returns: list[Decimal],
        volatility: Decimal | None = None,
        periods_per_year: int = 252,
    ) -> Decimal:
        """
        Calculate Sharpe ratio.

        Sharpe = (Mean Return - Risk Free Rate) / Volatility

        Args:
            returns: List of periodic returns
            volatility: Pre-calculated volatility (optional)
            periods_per_year: Number of periods in a year

        Returns:
            Sharpe ratio
        """
        if not returns:
            return Decimal("0")

        if volatility is None:
            volatility = self._calculate_volatility(returns)

        if volatility == Decimal("0"):
            return Decimal("0")

        mean_return = self._calculate_mean(returns)

        # Annualize the metrics
        annual_return = mean_return * Decimal(str(periods_per_year))
        annual_volatility = volatility * Decimal(str(periods_per_year)).sqrt()

        # Calculate Sharpe ratio
        excess_return = annual_return - self.risk_free_rate
        sharpe = excess_return / annual_volatility

        return sharpe.quantize(Decimal("0.01"))

    def calculate_sortino_ratio(
        self, returns: list[Decimal], periods_per_year: int = 252
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate Sortino ratio using downside deviation.

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation

        Args:
            returns: List of periodic returns
            periods_per_year: Number of periods in a year

        Returns:
            Tuple of (Sortino ratio, downside deviation)
        """
        if not returns:
            return Decimal("0"), Decimal("0")

        mean_return = self._calculate_mean(returns)

        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < Decimal("0")]

        if not negative_returns:
            # No negative returns means infinite Sortino ratio (capped at 999.99)
            return Decimal("999.99"), Decimal("0")

        downside_deviation = self._calculate_volatility(negative_returns)

        if downside_deviation == Decimal("0"):
            return Decimal("999.99"), Decimal("0")

        # Annualize the metrics
        annual_return = mean_return * Decimal(str(periods_per_year))
        annual_downside_dev = downside_deviation * Decimal(str(periods_per_year)).sqrt()

        # Calculate Sortino ratio
        excess_return = annual_return - self.risk_free_rate
        sortino = excess_return / annual_downside_dev

        return sortino.quantize(Decimal("0.01")), downside_deviation

    def calculate_calmar_ratio(
        self, mean_return: Decimal, max_drawdown: Decimal, periods_per_year: int = 252
    ) -> Decimal:
        """
        Calculate Calmar ratio.

        Calmar = Annual Return / Maximum Drawdown

        Args:
            mean_return: Average periodic return
            max_drawdown: Maximum drawdown (as positive value)
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio
        """
        if max_drawdown == Decimal("0"):
            return Decimal("999.99")

        annual_return = mean_return * Decimal(str(periods_per_year))
        calmar = annual_return / max_drawdown

        return calmar.quantize(Decimal("0.01"))

    def calculate_max_drawdown(self, returns: list[Decimal]) -> tuple[Decimal, int]:
        """
        Calculate maximum drawdown and duration.

        Args:
            returns: List of periodic returns

        Returns:
            Tuple of (max drawdown as positive decimal, duration in periods)
        """
        if not returns:
            return Decimal("0"), 0

        # Calculate cumulative returns
        cumulative = []
        cum_return = Decimal("1")

        for ret in returns:
            cum_return *= Decimal("1") + ret
            cumulative.append(cum_return)

        # Find maximum drawdown
        max_dd = Decimal("0")
        max_dd_duration = 0
        current_dd_duration = 0
        running_max = cumulative[0]

        for i, cum_val in enumerate(cumulative):
            if cum_val >= running_max:
                running_max = cum_val
                current_dd_duration = 0
            else:
                current_dd_duration += 1
                drawdown = (running_max - cum_val) / running_max
                if drawdown > max_dd:
                    max_dd = drawdown
                    max_dd_duration = current_dd_duration

        return max_dd.quantize(Decimal("0.0001")), max_dd_duration

    def calculate_value_at_risk(
        self, returns: list[Decimal], confidence_level: Decimal = Decimal("0.95")
    ) -> Decimal:
        """
        Calculate Value at Risk (VaR) at given confidence level.

        Args:
            returns: List of periodic returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR (as positive value representing potential loss)
        """
        if not returns:
            return Decimal("0")

        sorted_returns = sorted(returns)

        # Calculate percentile index
        index = int((Decimal("1") - confidence_level) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))

        # Return absolute value of the percentile
        var = abs(sorted_returns[index])

        return var.quantize(Decimal("0.0001"))

    def calculate_conditional_value_at_risk(
        self, returns: list[Decimal], confidence_level: Decimal = Decimal("0.95")
    ) -> Decimal:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            returns: List of periodic returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            CVaR (average of returns below VaR threshold)
        """
        if not returns:
            return Decimal("0")

        sorted_returns = sorted(returns)

        # Calculate VaR threshold index
        index = int((Decimal("1") - confidence_level) * len(sorted_returns))
        index = max(1, min(index, len(sorted_returns)))

        # Calculate average of returns below VaR
        tail_returns = sorted_returns[:index]

        if not tail_returns:
            return Decimal("0")

        cvar = abs(self._calculate_mean(tail_returns))

        return cvar.quantize(Decimal("0.0001"))

    def calculate_beta(
        self, returns: list[Decimal], benchmark_returns: list[Decimal]
    ) -> Decimal:
        """
        Calculate beta relative to benchmark.

        Beta = Covariance(returns, benchmark) / Variance(benchmark)

        Args:
            returns: List of strategy returns
            benchmark_returns: List of benchmark returns

        Returns:
            Beta coefficient
        """
        if not returns or not benchmark_returns:
            return Decimal("0")

        if len(returns) != len(benchmark_returns):
            logger.warning("Returns and benchmark length mismatch")
            return Decimal("0")

        # Calculate covariance
        mean_returns = self._calculate_mean(returns)
        mean_benchmark = self._calculate_mean(benchmark_returns)

        covariance = Decimal("0")
        for ret, bench in zip(returns, benchmark_returns, strict=False):
            covariance += (ret - mean_returns) * (bench - mean_benchmark)
        covariance /= Decimal(str(len(returns)))

        # Calculate benchmark variance
        benchmark_variance = self._calculate_variance(benchmark_returns)

        if benchmark_variance == Decimal("0"):
            return Decimal("0")

        beta = covariance / benchmark_variance

        return beta.quantize(Decimal("0.01"))

    def calculate_alpha(
        self,
        returns: list[Decimal],
        benchmark_returns: list[Decimal],
        beta: Decimal,
        periods_per_year: int = 252,
    ) -> Decimal:
        """
        Calculate Jensen's alpha.

        Alpha = Portfolio Return - (Risk Free + Beta * (Market Return - Risk Free))

        Args:
            returns: List of strategy returns
            benchmark_returns: List of benchmark returns
            beta: Pre-calculated beta
            periods_per_year: Number of periods in a year

        Returns:
            Alpha (excess return)
        """
        if not returns or not benchmark_returns:
            return Decimal("0")

        # Annualize returns
        mean_return = self._calculate_mean(returns) * Decimal(str(periods_per_year))
        mean_benchmark = self._calculate_mean(benchmark_returns) * Decimal(
            str(periods_per_year)
        )

        # Calculate alpha
        expected_return = self.risk_free_rate + beta * (
            mean_benchmark - self.risk_free_rate
        )
        alpha = mean_return - expected_return

        return alpha.quantize(Decimal("0.0001"))

    def calculate_information_ratio(
        self,
        returns: list[Decimal],
        benchmark_returns: list[Decimal],
        periods_per_year: int = 252,
    ) -> Decimal:
        """
        Calculate Information Ratio.

        IR = (Portfolio Return - Benchmark Return) / Tracking Error

        Args:
            returns: List of strategy returns
            benchmark_returns: List of benchmark returns
            periods_per_year: Number of periods in a year

        Returns:
            Information ratio
        """
        if not returns or not benchmark_returns:
            return Decimal("0")

        if len(returns) != len(benchmark_returns):
            return Decimal("0")

        # Calculate excess returns
        excess_returns = [
            r - b for r, b in zip(returns, benchmark_returns, strict=False)
        ]

        if not excess_returns:
            return Decimal("0")

        # Calculate tracking error (std dev of excess returns)
        tracking_error = self._calculate_volatility(excess_returns)

        if tracking_error == Decimal("0"):
            return Decimal("0")

        # Annualize metrics
        mean_excess = self._calculate_mean(excess_returns) * Decimal(
            str(periods_per_year)
        )
        annual_tracking_error = tracking_error * Decimal(str(periods_per_year)).sqrt()

        info_ratio = mean_excess / annual_tracking_error

        return info_ratio.quantize(Decimal("0.01"))

    def calculate_treynor_ratio(
        self, mean_return: Decimal, beta: Decimal, periods_per_year: int = 252
    ) -> Decimal:
        """
        Calculate Treynor Ratio.

        Treynor = (Portfolio Return - Risk Free Rate) / Beta

        Args:
            mean_return: Average periodic return
            beta: Portfolio beta
            periods_per_year: Number of periods in a year

        Returns:
            Treynor ratio
        """
        if beta == Decimal("0"):
            return Decimal("0")

        # Annualize return
        annual_return = mean_return * Decimal(str(periods_per_year))

        # Calculate Treynor ratio
        excess_return = annual_return - self.risk_free_rate
        treynor = excess_return / beta

        return treynor.quantize(Decimal("0.01"))

    def calculate_rolling_metrics(
        self,
        returns: list[Decimal],
        window_size: int,
        period: str = "daily",
        benchmark_returns: list[Decimal] | None = None,
    ) -> list[RiskMetrics]:
        """
        Calculate rolling window risk metrics.

        Args:
            returns: List of periodic returns
            window_size: Size of rolling window
            period: Return period ('daily', 'hourly', 'weekly', 'monthly')
            benchmark_returns: Optional benchmark returns

        Returns:
            List of RiskMetrics for each window
        """
        if len(returns) < window_size:
            return []

        rolling_metrics = []

        for i in range(len(returns) - window_size + 1):
            window_returns = returns[i : i + window_size]
            window_benchmark = None

            if benchmark_returns:
                window_benchmark = benchmark_returns[i : i + window_size]

            metrics = self.calculate_metrics(window_returns, period, window_benchmark)
            rolling_metrics.append(metrics)

        return rolling_metrics

    def _calculate_mean(self, values: list[Decimal]) -> Decimal:
        """Calculate mean of values."""
        if not values:
            return Decimal("0")
        return sum(values) / Decimal(str(len(values)))

    def _calculate_variance(self, values: list[Decimal]) -> Decimal:
        """Calculate variance of values."""
        if not values:
            return Decimal("0")

        mean = self._calculate_mean(values)
        squared_diffs = [(v - mean) ** 2 for v in values]

        return self._calculate_mean(squared_diffs)

    def _calculate_volatility(self, values: list[Decimal]) -> Decimal:
        """Calculate standard deviation of values."""
        variance = self._calculate_variance(values)
        return variance.sqrt()

    def _get_periods_per_year(self, period: str) -> int:
        """Get number of periods per year for given period type."""
        period_map = {
            "hourly": 24 * 365,
            "daily": 252,  # Trading days
            "weekly": 52,
            "monthly": 12,
            "quarterly": 4,
            "yearly": 1,
        }
        return period_map.get(period, 252)

    def _empty_metrics(self) -> RiskMetrics:
        """Return empty risk metrics object."""
        return RiskMetrics(
            sharpe_ratio=Decimal("0"),
            sortino_ratio=Decimal("0"),
            calmar_ratio=Decimal("0"),
            max_drawdown=Decimal("0"),
            max_drawdown_duration_days=0,
            volatility=Decimal("0"),
            downside_deviation=Decimal("0"),
            value_at_risk_95=Decimal("0"),
            conditional_value_at_risk_95=Decimal("0"),
        )
