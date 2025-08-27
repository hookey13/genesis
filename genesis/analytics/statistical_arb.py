"""Statistical Arbitrage Engine for detecting price divergences between correlated pairs."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog

# Set precision for financial calculations
getcontext().prec = 10

logger = structlog.get_logger(__name__)


@dataclass
class Signal:
    """Arbitrage signal with confidence score."""

    pair1_symbol: str
    pair2_symbol: str
    zscore: Decimal
    threshold_sigma: Decimal
    signal_type: str  # 'ENTRY' or 'EXIT'
    confidence_score: Decimal
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self):
        """Validate signal parameters."""
        if self.signal_type not in ["ENTRY", "EXIT"]:
            raise ValueError(f"Invalid signal type: {self.signal_type}")
        if not (Decimal("0") <= self.confidence_score <= Decimal("1")):
            raise ValueError(
                f"Confidence score must be between 0 and 1: {self.confidence_score}"
            )


class StatisticalArbitrage:
    """Core statistical arbitrage logic for pair correlation and divergence detection."""

    def __init__(self):
        """Initialize the statistical arbitrage engine."""
        self.correlation_cache: dict[str, tuple[Decimal, datetime]] = {}
        self.cache_ttl_seconds = 300  # 5 minutes cache

    def calculate_correlation(
        self,
        pair1: str,
        pair2: str,
        window: int,
        prices1: list[Decimal],
        prices2: list[Decimal],
    ) -> Decimal:
        """
        Calculate correlation between two pairs over a specified window.

        Args:
            pair1: First trading pair symbol
            pair2: Second trading pair symbol
            window: Rolling window size in data points
            prices1: Price history for pair1
            prices2: Price history for pair2

        Returns:
            Correlation coefficient between -1 and 1
        """
        cache_key = f"{pair1}:{pair2}:{window}"

        # Check cache
        if cache_key in self.correlation_cache:
            cached_corr, cached_time = self.correlation_cache[cache_key]
            if datetime.now(UTC) - cached_time < timedelta(
                seconds=self.cache_ttl_seconds
            ):
                logger.debug(
                    "Using cached correlation",
                    pair1=pair1,
                    pair2=pair2,
                    correlation=cached_corr,
                )
                return cached_corr

        # Validate input data
        if len(prices1) != len(prices2):
            raise ValueError(
                f"Price lists must have same length: {len(prices1)} vs {len(prices2)}"
            )
        if len(prices1) < window:
            raise ValueError(f"Insufficient data: {len(prices1)} < window {window}")

        # Convert to pandas for efficient calculation
        df = pd.DataFrame(
            {
                "price1": [float(p) for p in prices1[-window:]],
                "price2": [float(p) for p in prices2[-window:]],
            }
        )

        # Calculate correlation
        correlation = df["price1"].corr(df["price2"])

        # Handle NaN case
        if pd.isna(correlation):
            logger.warning(
                "Correlation calculation resulted in NaN", pair1=pair1, pair2=pair2
            )
            correlation = 0.0

        result = Decimal(str(correlation))

        # Update cache
        self.correlation_cache[cache_key] = (result, datetime.now(UTC))

        logger.info(
            "Calculated correlation",
            pair1=pair1,
            pair2=pair2,
            window=window,
            correlation=result,
        )

        return result

    def calculate_zscore(
        self,
        price1: Decimal,
        price2: Decimal,
        window: int,
        price_history1: list[Decimal],
        price_history2: list[Decimal],
    ) -> Decimal:
        """
        Calculate z-score for divergence detection.

        Args:
            price1: Current price of pair1
            price2: Current price of pair2
            window: Rolling window for spread calculation
            price_history1: Historical prices for pair1
            price_history2: Historical prices for pair2

        Returns:
            Z-score indicating standard deviations from mean spread
        """
        if len(price_history1) < window or len(price_history2) < window:
            raise ValueError(f"Insufficient history for window {window}")

        # Calculate spread ratio (log spread for better statistical properties)
        spreads = []
        for p1, p2 in zip(
            price_history1[-window:], price_history2[-window:], strict=False
        ):
            if p2 == 0:
                logger.error("Zero price detected", price2=p2)
                continue
            spread = float(np.log(float(p1) / float(p2)))
            spreads.append(spread)

        if not spreads:
            raise ValueError("No valid spreads calculated")

        # Calculate current spread
        if price2 == 0:
            raise ValueError("Current price2 is zero")
        current_spread = float(np.log(float(price1) / float(price2)))

        # Calculate mean and std
        spread_mean = np.mean(spreads)
        spread_std = np.std(spreads)

        # Handle zero std case
        if spread_std == 0:
            logger.warning(
                "Zero standard deviation in spread",
                pair1_price=price1,
                pair2_price=price2,
            )
            return Decimal("0")

        # Calculate z-score
        zscore = (current_spread - spread_mean) / spread_std

        result = Decimal(str(zscore))
        logger.debug(
            "Calculated z-score",
            price1=price1,
            price2=price2,
            zscore=result,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )

        return result

    def test_cointegration(
        self, pair1_prices: list[Decimal], pair2_prices: list[Decimal]
    ) -> bool:
        """
        Test for cointegration using Augmented Dickey-Fuller test.

        Args:
            pair1_prices: Price series for pair1
            pair2_prices: Price series for pair2

        Returns:
            True if pairs are cointegrated, False otherwise
        """
        if len(pair1_prices) != len(pair2_prices):
            raise ValueError("Price series must have same length")
        if len(pair1_prices) < 20:  # Minimum for meaningful test
            logger.warning(
                "Insufficient data for cointegration test", length=len(pair1_prices)
            )
            return False

        try:
            # Convert to numpy arrays
            prices1 = np.array([float(p) for p in pair1_prices])
            prices2 = np.array([float(p) for p in pair2_prices])

            # Calculate spread
            spread = prices1 - prices2

            # Perform ADF test using scipy (simplified version)
            # In production, would use statsmodels.tsa.stattools.adfuller
            # Here we use a simplified stationarity check

            # Check if spread is stationary (simplified)
            # Calculate rolling mean and std
            window = min(20, len(spread) // 4)
            rolling_mean = pd.Series(spread).rolling(window=window).mean()
            rolling_std = pd.Series(spread).rolling(window=window).std()

            # Check if rolling statistics are stable
            mean_stability = (
                rolling_mean.std() / rolling_mean.mean()
                if rolling_mean.mean() != 0
                else float("inf")
            )
            std_stability = (
                rolling_std.std() / rolling_std.mean()
                if rolling_std.mean() != 0
                else float("inf")
            )

            # Simplified cointegration check
            is_cointegrated = mean_stability < 0.1 and std_stability < 0.1

            logger.info(
                "Cointegration test result",
                cointegrated=is_cointegrated,
                mean_stability=mean_stability,
                std_stability=std_stability,
            )

            return is_cointegrated

        except Exception as e:
            logger.error("Cointegration test failed", error=str(e))
            return False

    def create_correlation_matrix(
        self, symbols: list[str], price_data: dict[str, list[Decimal]], window: int
    ) -> pd.DataFrame:
        """
        Create correlation matrix for multiple stablecoin pairs.

        Args:
            symbols: List of trading pair symbols
            price_data: Dictionary mapping symbols to price lists
            window: Rolling window for correlation calculation

        Returns:
            DataFrame with correlation matrix
        """
        n = len(symbols)
        matrix = np.zeros((n, n))

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i == j:
                    matrix[i, j] = 1.0
                elif sym1 in price_data and sym2 in price_data:
                    try:
                        corr = self.calculate_correlation(
                            sym1, sym2, window, price_data[sym1], price_data[sym2]
                        )
                        matrix[i, j] = float(corr)
                    except Exception as e:
                        logger.error(
                            "Failed to calculate correlation",
                            sym1=sym1,
                            sym2=sym2,
                            error=str(e),
                        )
                        matrix[i, j] = 0.0

        df = pd.DataFrame(matrix, index=symbols, columns=symbols)
        logger.info("Created correlation matrix", shape=df.shape)

        return df


class SpreadAnalyzer:
    """Analyze spreads between correlated pairs with rolling window calculations."""

    def __init__(self, window_days: int = 20):
        """
        Initialize spread analyzer.

        Args:
            window_days: Rolling window in days for spread analysis
        """
        self.window_days = window_days
        self.spread_history: dict[str, pd.DataFrame] = {}
        self.signal_cooldown: dict[str, datetime] = {}
        self.cooldown_minutes = 5

    def calculate_spread(
        self, pair1_prices: pd.Series, pair2_prices: pd.Series
    ) -> pd.Series:
        """
        Calculate spread between two price series.

        Args:
            pair1_prices: Price series for pair1
            pair2_prices: Price series for pair2

        Returns:
            Spread series
        """
        # Use log spread for better statistical properties
        spread = np.log(pair1_prices / pair2_prices)
        return spread

    def analyze_spread(
        self,
        pair1: str,
        pair2: str,
        prices1: list[Decimal],
        prices2: list[Decimal],
        timestamps: list[datetime],
    ) -> dict[str, Any]:
        """
        Analyze spread with rolling window statistics.

        Args:
            pair1: First pair symbol
            pair2: Second pair symbol
            prices1: Price history for pair1
            prices2: Price history for pair2
            timestamps: Timestamp for each price point

        Returns:
            Dictionary with spread analysis results
        """
        # Convert to pandas DataFrame
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price1": [float(p) for p in prices1],
                "price2": [float(p) for p in prices2],
            }
        )
        df.set_index("timestamp", inplace=True)

        # Calculate spread
        df["spread"] = self.calculate_spread(df["price1"], df["price2"])

        # Calculate rolling statistics (20-day window)
        window_size = (
            len(df) // self.window_days if len(df) > self.window_days else len(df)
        )
        df["spread_mean"] = df["spread"].rolling(window=window_size).mean()
        df["spread_std"] = df["spread"].rolling(window=window_size).std()

        # Store in history
        key = f"{pair1}:{pair2}"
        self.spread_history[key] = df

        # Get current statistics
        current_spread = df["spread"].iloc[-1]
        spread_mean = df["spread_mean"].iloc[-1]
        spread_std = df["spread_std"].iloc[-1]

        results = {
            "current_spread": Decimal(str(current_spread)),
            "spread_mean": Decimal(str(spread_mean)),
            "spread_std": Decimal(str(spread_std)),
            "data_points": len(df),
            "window_days": self.window_days,
        }

        logger.info("Spread analysis complete", pair1=pair1, pair2=pair2, **results)

        return results

    def generate_signal(
        self,
        zscore: Decimal,
        threshold: Decimal,
        pair1: str,
        pair2: str,
        cointegrated: bool,
        spread_stability: Decimal,
    ) -> Optional[Signal]:
        """
        Generate trading signal with confidence scoring.

        Args:
            zscore: Current z-score
            threshold: Sigma threshold for signal generation
            pair1: First pair symbol
            pair2: Second pair symbol
            cointegrated: Whether pairs are cointegrated
            spread_stability: Measure of spread stability (0-1)

        Returns:
            Signal object if threshold crossed, None otherwise
        """
        # Check cooldown
        key = f"{pair1}:{pair2}"
        if key in self.signal_cooldown:
            time_since_last = datetime.now(UTC) - self.signal_cooldown[key]
            if time_since_last < timedelta(minutes=self.cooldown_minutes):
                logger.debug(
                    "Signal in cooldown period",
                    pair1=pair1,
                    pair2=pair2,
                    remaining_seconds=(
                        timedelta(minutes=self.cooldown_minutes) - time_since_last
                    ).seconds,
                )
                return None

        # Determine signal type
        signal_type = None
        if abs(zscore) >= threshold:
            signal_type = "ENTRY"
        elif abs(zscore) <= Decimal("0.5"):  # Exit when returning to mean
            signal_type = "EXIT"

        if not signal_type:
            return None

        # Calculate confidence score
        confidence_components = []

        # Cointegration contributes 40% to confidence
        if cointegrated:
            confidence_components.append(Decimal("0.4"))
        else:
            confidence_components.append(Decimal("0.1"))

        # Spread stability contributes 30%
        confidence_components.append(spread_stability * Decimal("0.3"))

        # Z-score magnitude contributes 30%
        zscore_confidence = min(abs(zscore) / Decimal("4"), Decimal("1")) * Decimal(
            "0.3"
        )
        confidence_components.append(zscore_confidence)

        confidence_score = sum(confidence_components)
        confidence_score = min(confidence_score, Decimal("1"))  # Cap at 1

        # Create signal
        signal = Signal(
            pair1_symbol=pair1,
            pair2_symbol=pair2,
            zscore=zscore,
            threshold_sigma=threshold,
            signal_type=signal_type,
            confidence_score=confidence_score,
        )

        # Update cooldown
        self.signal_cooldown[key] = datetime.now(UTC)

        logger.info(
            "Generated signal",
            pair1=pair1,
            pair2=pair2,
            signal_type=signal_type,
            confidence=confidence_score,
            zscore=zscore,
        )

        return signal

    def check_signal_persistence(
        self, signals: list[Signal], min_persistence: int = 3
    ) -> list[Signal]:
        """
        Filter signals based on persistence to avoid false positives.

        Args:
            signals: List of recent signals
            min_persistence: Minimum number of consecutive signals required

        Returns:
            Filtered list of persistent signals
        """
        if len(signals) < min_persistence:
            return []

        # Group signals by pair
        pair_signals: dict[str, list[Signal]] = {}
        for signal in signals:
            key = f"{signal.pair1_symbol}:{signal.pair2_symbol}"
            if key not in pair_signals:
                pair_signals[key] = []
            pair_signals[key].append(signal)

        # Check persistence for each pair
        persistent_signals = []
        for key, sigs in pair_signals.items():
            if len(sigs) >= min_persistence:
                # Check if signals are consistent and recent
                latest_signal = sigs[-1]
                time_window = datetime.now(UTC) - timedelta(minutes=15)

                recent_consistent = [
                    s
                    for s in sigs
                    if s.created_at > time_window
                    and s.signal_type == latest_signal.signal_type
                ]

                if len(recent_consistent) >= min_persistence:
                    persistent_signals.append(latest_signal)
                    logger.info(
                        "Signal passed persistence check",
                        pair_key=key,
                        persistence_count=len(recent_consistent),
                    )

        return persistent_signals


class ThresholdMonitor:
    """Monitor z-score thresholds and generate alerts."""

    def __init__(self, default_sigma: Decimal = Decimal("2")):
        """
        Initialize threshold monitor.

        Args:
            default_sigma: Default sigma level for alerts
        """
        self.thresholds: dict[str, Decimal] = {}
        self.default_sigma = default_sigma
        self.alert_history: list[dict[str, Any]] = []
        self.last_alert_time: dict[str, datetime] = {}
        self.min_alert_interval = timedelta(minutes=5)

    def set_threshold(self, pair_key: str, sigma: Decimal):
        """
        Set custom threshold for a pair.

        Args:
            pair_key: Pair identifier (e.g., "BTCUSDT:ETHUSDT")
            sigma: Sigma threshold level
        """
        self.thresholds[pair_key] = sigma
        logger.info("Threshold set", pair_key=pair_key, sigma=sigma)

    def check_threshold(
        self, pair1: str, pair2: str, zscore: Decimal
    ) -> Optional[dict[str, Any]]:
        """
        Check if z-score breaches threshold.

        Args:
            pair1: First pair symbol
            pair2: Second pair symbol
            zscore: Current z-score

        Returns:
            Alert dictionary if threshold breached, None otherwise
        """
        pair_key = f"{pair1}:{pair2}"
        threshold = self.thresholds.get(pair_key, self.default_sigma)

        # Check if threshold is breached
        if abs(zscore) < threshold:
            return None

        # Check cooldown
        if pair_key in self.last_alert_time:
            time_since_last = datetime.now(UTC) - self.last_alert_time[pair_key]
            if time_since_last < self.min_alert_interval:
                return None

        # Create alert
        alert = {
            "pair1": pair1,
            "pair2": pair2,
            "zscore": zscore,
            "threshold": threshold,
            "direction": "ABOVE" if zscore > 0 else "BELOW",
            "timestamp": datetime.now(UTC),
        }

        # Update tracking
        self.alert_history.append(alert)
        self.last_alert_time[pair_key] = datetime.now(UTC)

        logger.warning(
            "Threshold breach detected",
            pair1=pair1,
            pair2=pair2,
            zscore=zscore,
            threshold=threshold,
        )

        return alert

    async def monitor_thresholds(
        self, pairs: list[tuple[str, str]], get_zscore_func
    ) -> list[dict[str, Any]]:
        """
        Monitor multiple pairs for threshold breaches.

        Args:
            pairs: List of pair tuples to monitor
            get_zscore_func: Async function to get current z-score for a pair

        Returns:
            List of alerts generated
        """
        alerts = []

        for pair1, pair2 in pairs:
            try:
                zscore = await get_zscore_func(pair1, pair2)
                alert = self.check_threshold(pair1, pair2, zscore)
                if alert:
                    alerts.append(alert)
            except Exception as e:
                logger.error(
                    "Failed to check threshold", pair1=pair1, pair2=pair2, error=str(e)
                )

        return alerts
