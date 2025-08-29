"""
Sharpe Ratio Calculator for Portfolio Optimization

Calculates risk-adjusted returns using the Sharpe ratio formula.
Supports multiple time periods and rolling window calculations.
Part of the Hunter+ tier portfolio optimization suite.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

import numpy as np
import structlog

from genesis.core.constants import TradingTier
from genesis.core.exceptions import (
    DataError as InvalidDataError,
)
from genesis.core.exceptions import (
    GenesisException as CalculationError,
)
from genesis.utils.decorators import requires_tier, with_timeout

logger = structlog.get_logger(__name__)


class TimePeriod(Enum):
    """Time period for Sharpe ratio calculation"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class SharpeRatioResult:
    """Result of Sharpe ratio calculation"""

    sharpe_ratio: Decimal
    mean_return: Decimal
    std_deviation: Decimal
    risk_free_rate: Decimal
    period: TimePeriod
    num_periods: int
    confidence_interval_lower: Decimal | None = None
    confidence_interval_upper: Decimal | None = None
    calculated_at: datetime = None

    def __post_init__(self):
        if self.calculated_at is None:
            self.calculated_at = datetime.now(UTC)


class SharpeRatioCalculator:
    """
    Calculates Sharpe ratio for strategy performance evaluation.

    Hunter+ tier feature for portfolio optimization.
    """

    # Annualization factors
    ANNUALIZATION_FACTORS = {
        TimePeriod.DAILY: Decimal("252"),  # Trading days per year
        TimePeriod.WEEKLY: Decimal("52"),  # Weeks per year
        TimePeriod.MONTHLY: Decimal("12"),  # Months per year
        TimePeriod.YEARLY: Decimal("1"),  # Already annualized
    }

    # Cache TTL in seconds
    CACHE_TTL_SECONDS = 3600  # 1 hour

    def __init__(self):
        """Initialize Sharpe ratio calculator"""
        self._cache: dict[str, tuple[SharpeRatioResult, datetime]] = {}
        self._cache_lock = asyncio.Lock()
        logger.info("sharpe_ratio_calculator_initialized")

    @requires_tier(TradingTier.HUNTER)
    @with_timeout(5.0)
    async def calculate_sharpe_ratio(
        self,
        returns: list[Decimal],
        risk_free_rate: Decimal = Decimal("0.02"),  # 2% default annual risk-free rate
        period: TimePeriod = TimePeriod.DAILY,
        confidence_level: float = 0.95,
    ) -> SharpeRatioResult:
        """
        Calculate Sharpe ratio for a series of returns.

        Args:
            returns: List of period returns (as decimals, e.g., 0.01 for 1%)
            risk_free_rate: Annual risk-free rate (default 2%)
            period: Time period of returns (daily, weekly, etc.)
            confidence_level: Confidence level for bootstrap intervals

        Returns:
            SharpeRatioResult with calculated metrics

        Raises:
            InvalidDataError: If returns data is invalid
            CalculationError: If calculation fails
        """
        try:
            # Validate inputs
            if not returns or len(returns) < 2:
                raise InvalidDataError(
                    "Need at least 2 return periods for Sharpe ratio"
                )

            # Check cache
            cache_key = self._get_cache_key(returns, risk_free_rate, period)
            cached = await self._get_cached_result(cache_key)
            if cached:
                logger.info("sharpe_ratio_cache_hit", cache_key=cache_key)
                return cached

            # Convert to numpy array for calculations
            returns_array = np.array([float(r) for r in returns])

            # Calculate basic statistics
            mean_return = Decimal(str(np.mean(returns_array)))
            std_deviation = Decimal(
                str(np.std(returns_array, ddof=1))
            )  # Sample std dev

            # Adjust risk-free rate to match period
            annualization_factor = self.ANNUALIZATION_FACTORS[period]
            period_risk_free = risk_free_rate / annualization_factor

            # Calculate excess return
            excess_return = mean_return - period_risk_free

            # Calculate Sharpe ratio (annualized)
            if std_deviation == Decimal("0"):
                # Handle zero volatility case
                sharpe_ratio = (
                    Decimal("0")
                    if excess_return == Decimal("0")
                    else (
                        Decimal("999")
                        if excess_return > Decimal("0")
                        else Decimal("-999")
                    )
                )
            else:
                # Sharpe = (excess_return / std_dev) * sqrt(annualization_factor)
                sharpe_ratio = (
                    excess_return / std_deviation
                ) * annualization_factor.sqrt()

            # Calculate confidence intervals using bootstrap
            ci_lower, ci_upper = None, None
            if len(returns) >= 30:  # Need sufficient data for bootstrap
                ci_lower, ci_upper = await self._bootstrap_confidence_interval(
                    returns_array,
                    period_risk_free,
                    annualization_factor,
                    confidence_level,
                )

            # Round results
            result = SharpeRatioResult(
                sharpe_ratio=sharpe_ratio.quantize(Decimal("0.0001"), ROUND_HALF_UP),
                mean_return=mean_return.quantize(Decimal("0.000001"), ROUND_HALF_UP),
                std_deviation=std_deviation.quantize(
                    Decimal("0.000001"), ROUND_HALF_UP
                ),
                risk_free_rate=risk_free_rate,
                period=period,
                num_periods=len(returns),
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
            )

            # Cache result
            await self._cache_result(cache_key, result)

            logger.info(
                "sharpe_ratio_calculated",
                sharpe_ratio=float(result.sharpe_ratio),
                mean_return=float(result.mean_return),
                std_dev=float(result.std_deviation),
                num_periods=result.num_periods,
                period=period.value,
            )

            return result

        except Exception as e:
            logger.error("sharpe_ratio_calculation_failed", error=str(e))
            raise CalculationError(f"Failed to calculate Sharpe ratio: {e}")

    @requires_tier(TradingTier.HUNTER)
    async def calculate_rolling_sharpe(
        self,
        returns: list[Decimal],
        window_size: int,
        risk_free_rate: Decimal = Decimal("0.02"),
        period: TimePeriod = TimePeriod.DAILY,
    ) -> list[SharpeRatioResult]:
        """
        Calculate rolling Sharpe ratio over a moving window.

        Args:
            returns: Full list of returns
            window_size: Size of rolling window
            risk_free_rate: Annual risk-free rate
            period: Time period of returns

        Returns:
            List of SharpeRatioResult for each window
        """
        if len(returns) < window_size:
            raise InvalidDataError(
                f"Need at least {window_size} periods for rolling calculation"
            )

        results = []
        for i in range(len(returns) - window_size + 1):
            window_returns = returns[i : i + window_size]
            result = await self.calculate_sharpe_ratio(
                window_returns,
                risk_free_rate,
                period,
                confidence_level=0,  # Skip CI for rolling calculations
            )
            results.append(result)

        logger.info(
            "rolling_sharpe_calculated",
            num_windows=len(results),
            window_size=window_size,
        )

        return results

    async def _bootstrap_confidence_interval(
        self,
        returns: np.ndarray,
        period_risk_free: Decimal,
        annualization_factor: Decimal,
        confidence_level: float,
        n_bootstrap: int = 1000,
    ) -> tuple[Decimal | None, Decimal | None]:
        """
        Calculate confidence intervals using bootstrap method.

        Args:
            returns: Array of returns
            period_risk_free: Risk-free rate adjusted for period
            annualization_factor: Factor to annualize Sharpe ratio
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound) or (None, None) if failed
        """
        try:
            bootstrap_sharpes = []
            n_samples = len(returns)

            for _ in range(n_bootstrap):
                # Resample with replacement
                sample_indices = np.random.choice(n_samples, n_samples, replace=True)
                sample_returns = returns[sample_indices]

                # Calculate Sharpe for this sample
                mean_return = np.mean(sample_returns)
                std_dev = np.std(sample_returns, ddof=1)

                if std_dev > 0:
                    excess_return = mean_return - float(period_risk_free)
                    sharpe = (excess_return / std_dev) * float(
                        annualization_factor.sqrt()
                    )
                    bootstrap_sharpes.append(sharpe)

            if bootstrap_sharpes:
                # Calculate percentiles
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100

                ci_lower = np.percentile(bootstrap_sharpes, lower_percentile)
                ci_upper = np.percentile(bootstrap_sharpes, upper_percentile)

                return (
                    Decimal(str(ci_lower)).quantize(Decimal("0.0001"), ROUND_HALF_UP),
                    Decimal(str(ci_upper)).quantize(Decimal("0.0001"), ROUND_HALF_UP),
                )

        except Exception as e:
            logger.warning("bootstrap_ci_failed", error=str(e))

        return None, None

    def _get_cache_key(
        self, returns: list[Decimal], risk_free_rate: Decimal, period: TimePeriod
    ) -> str:
        """Generate cache key for results"""
        # Use hash of returns + parameters
        returns_hash = hash(tuple(returns))
        return f"{returns_hash}_{risk_free_rate}_{period.value}"

    async def _get_cached_result(self, cache_key: str) -> SharpeRatioResult | None:
        """Get cached result if still valid"""
        async with self._cache_lock:
            if cache_key in self._cache:
                result, cached_at = self._cache[cache_key]
                age_seconds = (datetime.now(UTC) - cached_at).total_seconds()

                if age_seconds < self.CACHE_TTL_SECONDS:
                    return result
                else:
                    # Remove expired entry
                    del self._cache[cache_key]

        return None

    async def _cache_result(self, cache_key: str, result: SharpeRatioResult):
        """Cache calculation result"""
        async with self._cache_lock:
            self._cache[cache_key] = (result, datetime.now(UTC))

            # Limit cache size
            if len(self._cache) > 100:
                # Remove oldest entries
                sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_items[:20]:
                    del self._cache[key]

    async def clear_cache(self):
        """Clear all cached results"""
        async with self._cache_lock:
            self._cache.clear()
            logger.info("sharpe_ratio_cache_cleared")
