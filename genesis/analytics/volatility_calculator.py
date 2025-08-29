"""
Volatility Calculator Module

Provides comprehensive volatility calculations including ATR, realized volatility,
and volatility percentile rankings for market analysis.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import structlog

from genesis.core.exceptions import DataError

logger = structlog.get_logger(__name__)


@dataclass
class VolatilityMetrics:
    """Container for volatility metrics."""

    symbol: str
    atr: Decimal
    atr_percentage: Decimal  # ATR as percentage of price
    realized_vol: Decimal
    realized_vol_annualized: Decimal
    percentile: int  # Percentile rank vs history
    timestamp: datetime
    period: int  # Period used for calculation

    def to_dict(self) -> dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "atr": str(self.atr),
            "atr_percentage": str(self.atr_percentage),
            "realized_vol": str(self.realized_vol),
            "realized_vol_annualized": str(self.realized_vol_annualized),
            "percentile": self.percentile,
            "timestamp": self.timestamp.isoformat(),
            "period": self.period,
        }


class VolatilityCalculator:
    """
    Advanced volatility calculator for market analysis.

    Implements multiple volatility measures including ATR,
    realized volatility, and historical percentile rankings.
    """

    def __init__(self, cache_size: int = 1000):
        """
        Initialize the volatility calculator.

        Args:
            cache_size: Maximum cache size for calculations
        """
        self._cache: dict[str, VolatilityMetrics] = {}
        self._cache_size = cache_size
        self._historical_volatility: dict[str, pd.Series] = (
            {}
        )  # symbol -> historical volatilities

        logger.info(f"VolatilityCalculator initialized with cache_size={cache_size}")

    def calculate_atr(
        self,
        high_prices: list[Decimal],
        low_prices: list[Decimal],
        close_prices: list[Decimal],
        period: int = 14,
    ) -> Decimal:
        """
        Calculate Average True Range (ATR).

        Args:
            high_prices: List of high prices
            low_prices: List of low prices
            close_prices: List of close prices
            period: ATR period (default: 14)

        Returns:
            ATR value as Decimal
        """
        if len(high_prices) != len(low_prices) or len(high_prices) != len(close_prices):
            raise DataError("Price lists must have equal length")

        if len(high_prices) < period + 1:
            raise DataError(f"Insufficient data: {len(high_prices)} < {period + 1}")

        # Calculate True Range for each period
        true_ranges = []
        for i in range(1, len(high_prices)):
            # Use Decimal for all calculations
            high_low = high_prices[i] - low_prices[i]
            high_close = abs(high_prices[i] - close_prices[i - 1])
            low_close = abs(low_prices[i] - close_prices[i - 1])
            # Max works with Decimal values
            true_range = max(high_low, high_close, low_close)
            # Convert to float only for pandas compatibility
            true_ranges.append(float(true_range))

        # Convert to pandas for efficient calculation
        tr_series = pd.Series(true_ranges)

        # Calculate ATR using exponential moving average
        atr_series = tr_series.ewm(span=period, adjust=False).mean()

        # Return the most recent ATR value
        atr_value = Decimal(str(atr_series.iloc[-1]))

        return atr_value

    def calculate_atr_percentage(self, atr: Decimal, current_price: Decimal) -> Decimal:
        """
        Calculate ATR as percentage of current price.

        Args:
            atr: ATR value
            current_price: Current price

        Returns:
            ATR percentage
        """
        if current_price == 0:
            return Decimal("0")

        return (atr / current_price) * Decimal("100")

    def calculate_realized_volatility(
        self, prices: list[Decimal], window: int = 20
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate realized volatility using log returns.

        Args:
            prices: List of prices
            window: Rolling window size (default: 20)

        Returns:
            Tuple of (daily volatility, annualized volatility)
        """
        if len(prices) < window:
            raise DataError(f"Insufficient prices: {len(prices)} < {window}")

        # Convert to numpy array for efficient calculation
        # Convert to float array for numpy operations
        price_array = np.array([float(p) for p in prices])

        # Calculate log returns
        log_returns = np.log(price_array[1:] / price_array[:-1])

        # Use most recent window
        recent_returns = log_returns[-window:]

        # Calculate standard deviation (realized volatility)
        daily_vol = np.std(recent_returns, ddof=1)

        # Annualize (assuming 365 trading days)
        annualized_vol = daily_vol * np.sqrt(365)

        return Decimal(str(daily_vol)), Decimal(str(annualized_vol))

    def calculate_volatility_percentile(
        self,
        current_volatility: Decimal,
        historical_volatilities: list[Decimal],
        lookback_days: int = 30,
    ) -> int:
        """
        Calculate volatility percentile rank.

        Args:
            current_volatility: Current volatility value
            historical_volatilities: Historical volatility values
            lookback_days: Days to look back (default: 30)

        Returns:
            Percentile rank (0-100)
        """
        if not historical_volatilities:
            return 50  # Default to median if no history

        # Use most recent lookback_days values
        recent_history = historical_volatilities[-lookback_days:]

        # Convert to numpy for efficient calculation
        # Convert to float for numpy percentile calculation
        history_array = np.array([float(v) for v in recent_history])
        current_val = float(current_volatility)

        # Calculate percentile
        percentile = int(np.sum(history_array < current_val) / len(history_array) * 100)

        return percentile

    def calculate_rolling_volatility(
        self, prices: pd.Series, window: int = 20, min_periods: int | None = None
    ) -> pd.Series:
        """
        Calculate rolling volatility using pandas.

        Args:
            prices: Pandas Series of prices
            window: Rolling window size
            min_periods: Minimum periods required

        Returns:
            Series of rolling volatilities
        """
        if min_periods is None:
            min_periods = window

        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))

        # Calculate rolling standard deviation
        rolling_vol = log_returns.rolling(window=window, min_periods=min_periods).std()

        # Annualize
        rolling_vol_annualized = rolling_vol * np.sqrt(365)

        return rolling_vol_annualized

    async def calculate_comprehensive_volatility(
        self,
        symbol: str,
        high_prices: list[Decimal],
        low_prices: list[Decimal],
        close_prices: list[Decimal],
        atr_period: int = 14,
        realized_window: int = 20,
        use_cache: bool = True,
    ) -> VolatilityMetrics:
        """
        Calculate comprehensive volatility metrics.

        Args:
            symbol: Trading pair symbol
            high_prices: List of high prices
            low_prices: List of low prices
            close_prices: List of close prices
            atr_period: Period for ATR calculation
            realized_window: Window for realized volatility
            use_cache: Whether to use cached results

        Returns:
            Complete volatility metrics
        """
        # Check cache
        cache_key = f"{symbol}_{atr_period}_{realized_window}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            # Check if cache is fresh (within 1 minute)
            if (datetime.now(UTC) - cached.timestamp).total_seconds() < 60:
                logger.debug(f"Using cached volatility for {symbol}")
                return cached

        try:
            # Calculate ATR
            atr = self.calculate_atr(high_prices, low_prices, close_prices, atr_period)

            # Calculate ATR percentage
            current_price = close_prices[-1]
            atr_percentage = self.calculate_atr_percentage(atr, current_price)

            # Calculate realized volatility
            daily_vol, annualized_vol = self.calculate_realized_volatility(
                close_prices, realized_window
            )

            # Get historical volatilities for percentile calculation
            if symbol not in self._historical_volatility:
                self._historical_volatility[symbol] = pd.Series(dtype=float)

            # Add current volatility to history
            self._historical_volatility[symbol] = pd.concat(
                [
                    self._historical_volatility[symbol],
                    # Convert to float for pandas operations
                    pd.Series([float(annualized_vol)]),
                ]
            ).tail(
                100
            )  # Keep last 100 values

            # Calculate percentile
            historical_vols = [
                Decimal(str(v)) for v in self._historical_volatility[symbol].values
            ]
            percentile = self.calculate_volatility_percentile(
                annualized_vol, historical_vols, lookback_days=30
            )

            # Create metrics object
            metrics = VolatilityMetrics(
                symbol=symbol,
                atr=atr,
                atr_percentage=atr_percentage,
                realized_vol=daily_vol,
                realized_vol_annualized=annualized_vol,
                percentile=percentile,
                timestamp=datetime.now(UTC),
                period=atr_period,
            )

            # Update cache
            if use_cache:
                self._cache[cache_key] = metrics
                # Limit cache size
                if len(self._cache) > self._cache_size:
                    # Remove oldest entry
                    oldest_key = min(
                        self._cache.keys(), key=lambda k: self._cache[k].timestamp
                    )
                    del self._cache[oldest_key]

            logger.info(
                f"Calculated volatility for {symbol}: "
                f"ATR={atr:.4f} ({atr_percentage:.2f}%), "
                f"RealizedVol={annualized_vol:.2f}%, "
                f"Percentile={percentile}"
            )

            return metrics

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            raise DataError(f"Volatility calculation failed: {e}")

    def calculate_parkinson_volatility(
        self, high_prices: list[Decimal], low_prices: list[Decimal], window: int = 20
    ) -> Decimal:
        """
        Calculate Parkinson volatility (using high-low range).

        More efficient than close-to-close volatility as it uses
        intraday price information.

        Args:
            high_prices: List of high prices
            low_prices: List of low prices
            window: Calculation window

        Returns:
            Parkinson volatility
        """
        if len(high_prices) != len(low_prices):
            raise DataError("High and low price lists must have equal length")

        if len(high_prices) < window:
            raise DataError(f"Insufficient data: {len(high_prices)} < {window}")

        # Calculate log of high/low ratio
        log_hl_ratios = []
        for i in range(len(high_prices)):
            if low_prices[i] > 0:
                # Use Decimal division then convert for numpy
                ratio = float(high_prices[i] / low_prices[i])
                if ratio > 0:
                    log_hl_ratios.append(np.log(ratio))

        # Use most recent window
        recent_ratios = log_hl_ratios[-window:]

        # Parkinson volatility formula
        # σ = sqrt(1/(4*n*ln(2)) * Σ(ln(Hi/Li))^2)
        sum_squared = sum(r**2 for r in recent_ratios)
        parkinson_var = sum_squared / (4 * len(recent_ratios) * np.log(2))
        parkinson_vol = np.sqrt(parkinson_var)

        # Annualize
        annualized = parkinson_vol * np.sqrt(365)

        return Decimal(str(annualized))

    def calculate_garman_klass_volatility(
        self,
        open_prices: list[Decimal],
        high_prices: list[Decimal],
        low_prices: list[Decimal],
        close_prices: list[Decimal],
        window: int = 20,
    ) -> Decimal:
        """
        Calculate Garman-Klass volatility.

        Uses open, high, low, close prices for more accurate
        volatility estimation than simple close-to-close.

        Args:
            open_prices: List of open prices
            high_prices: List of high prices
            low_prices: List of low prices
            close_prices: List of close prices
            window: Calculation window

        Returns:
            Garman-Klass volatility
        """
        if not all(
            len(p) == len(open_prices) for p in [high_prices, low_prices, close_prices]
        ):
            raise DataError("All price lists must have equal length")

        if len(open_prices) < window:
            raise DataError(f"Insufficient data: {len(open_prices)} < {window}")

        gk_values = []
        for i in range(len(open_prices)):
            if open_prices[i] > 0 and low_prices[i] > 0:
                # First term: 0.5 * ln(H/L)^2
                # Convert ratios to float for numpy log operations
                hl_term = 0.5 * (np.log(float(high_prices[i] / low_prices[i]))) ** 2

                # Second term: -(2*ln(2)-1) * ln(C/O)^2
                co_term = (
                    -(2 * np.log(2) - 1)
                    * (np.log(float(close_prices[i] / open_prices[i]))) ** 2
                )

                gk_values.append(hl_term + co_term)

        # Use most recent window
        recent_values = gk_values[-window:]

        # Calculate average and take square root
        gk_var = sum(recent_values) / len(recent_values)
        gk_vol = np.sqrt(max(0, gk_var))  # Ensure non-negative

        # Annualize
        annualized = gk_vol * np.sqrt(365)

        return Decimal(str(annualized))

    def detect_volatility_regime(
        self,
        current_vol: Decimal,
        historical_vols: list[Decimal],
        low_threshold: int = 25,
        high_threshold: int = 75,
    ) -> str:
        """
        Detect volatility regime.

        Args:
            current_vol: Current volatility
            historical_vols: Historical volatilities
            low_threshold: Low volatility percentile threshold
            high_threshold: High volatility percentile threshold

        Returns:
            Volatility regime: 'LOW', 'NORMAL', or 'HIGH'
        """
        percentile = self.calculate_volatility_percentile(
            current_vol, historical_vols, lookback_days=30
        )

        if percentile < low_threshold:
            return "LOW"
        elif percentile > high_threshold:
            return "HIGH"
        else:
            return "NORMAL"

    def clear_cache(self, symbol: str | None = None):
        """
        Clear volatility cache.

        Args:
            symbol: Optional symbol to clear, or all if None
        """
        if symbol:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for {symbol}")
        else:
            self._cache.clear()
            logger.info("Cleared all volatility cache")
