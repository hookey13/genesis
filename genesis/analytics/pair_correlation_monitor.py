from typing import Optional

"""Correlation monitoring system for multi-pair trading."""

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import structlog

from genesis.core.models import Position
from genesis.data.repository import Repository

logger = structlog.get_logger(__name__)


@dataclass
class CorrelationAlert:
    """Alert for high correlation detection."""

    alert_id: str
    symbol1: str
    symbol2: str
    correlation: Decimal
    threshold: Decimal
    timestamp: datetime
    message: str
    severity: str  # 'WARNING' or 'CRITICAL'


@dataclass
class PriceHistory:
    """Price history for correlation calculation."""

    symbol: str
    prices: deque[Decimal] = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque[datetime] = field(default_factory=lambda: deque(maxlen=1000))

    def add_price(self, price: Decimal, timestamp: datetime) -> None:
        """Add a price point to history."""
        self.prices.append(price)
        self.timestamps.append(timestamp)

    def get_returns(self, window_minutes: int) -> np.ndarray:
        """Calculate returns over specified window."""
        if len(self.prices) < 2:
            return np.array([])

        # Filter by time window
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        valid_indices = [i for i, ts in enumerate(self.timestamps) if ts >= cutoff_time]

        if len(valid_indices) < 2:
            return np.array([])

        # Calculate log returns
        prices = [float(self.prices[i]) for i in valid_indices]
        prices_array = np.array(prices)
        returns = np.log(prices_array[1:] / prices_array[:-1])

        return returns


class CorrelationMonitor:
    """Monitors and calculates correlations between trading pairs."""

    WARNING_THRESHOLD = Decimal("0.6")
    CRITICAL_THRESHOLD = Decimal("0.8")
    DEFAULT_WINDOW_MINUTES = 60
    MIN_DATA_POINTS = 20

    def __init__(self, repository: Repository):
        """Initialize correlation monitor.

        Args:
            repository: Data repository for persistence
        """
        self.repository = repository
        self._price_histories: dict[str, PriceHistory] = {}
        self._correlation_cache: dict[
            tuple[str, str, int], tuple[Decimal, datetime]
        ] = {}
        self._cache_ttl_seconds = 60  # Cache correlations for 1 minute
        self._alerts: list[CorrelationAlert] = []
        self._lock = asyncio.Lock()

    async def update_price(
        self, symbol: str, price: Decimal, timestamp: Optional[datetime] = None
    ) -> None:
        """Update price for a symbol.

        Args:
            symbol: Trading pair symbol
            price: Current price
            timestamp: Price timestamp (defaults to now)
        """
        async with self._lock:
            if symbol not in self._price_histories:
                self._price_histories[symbol] = PriceHistory(symbol=symbol)

            timestamp = timestamp or datetime.utcnow()
            self._price_histories[symbol].add_price(price, timestamp)

            logger.debug(
                "price_updated", symbol=symbol, price=price, timestamp=timestamp
            )

    async def calculate_pair_correlation(
        self, symbol1: str, symbol2: str, window_minutes: int = None
    ) -> Decimal:
        """Calculate correlation between two symbols.

        Args:
            symbol1: First trading pair
            symbol2: Second trading pair
            window_minutes: Time window for calculation

        Returns:
            Correlation coefficient between -1 and 1
        """
        window_minutes = window_minutes or self.DEFAULT_WINDOW_MINUTES

        # Check cache
        cache_key = (symbol1, symbol2, window_minutes)
        reverse_key = (symbol2, symbol1, window_minutes)

        async with self._lock:
            # Check both orderings in cache
            for key in [cache_key, reverse_key]:
                if key in self._correlation_cache:
                    cached_corr, cached_time = self._correlation_cache[key]
                    if (
                        datetime.utcnow() - cached_time
                    ).total_seconds() < self._cache_ttl_seconds:
                        return cached_corr

            # Load historical data if not in memory
            if symbol1 not in self._price_histories:
                await self._load_price_history(symbol1, window_minutes)
            if symbol2 not in self._price_histories:
                await self._load_price_history(symbol2, window_minutes)

            # Get price histories
            history1 = self._price_histories.get(symbol1)
            history2 = self._price_histories.get(symbol2)

            if not history1 or not history2:
                logger.warning(
                    "missing_price_history", symbol1=symbol1, symbol2=symbol2
                )
                return Decimal("0")

            # Calculate returns
            returns1 = history1.get_returns(window_minutes)
            returns2 = history2.get_returns(window_minutes)

            # Need minimum data points
            min_length = min(len(returns1), len(returns2))
            if min_length < self.MIN_DATA_POINTS:
                logger.debug(
                    "insufficient_data_for_correlation",
                    symbol1=symbol1,
                    symbol2=symbol2,
                    data_points=min_length,
                    required=self.MIN_DATA_POINTS,
                )
                return Decimal("0")

            # Align returns to same length
            returns1 = returns1[-min_length:]
            returns2 = returns2[-min_length:]

            # Calculate correlation
            try:
                correlation_matrix = np.corrcoef(returns1, returns2)
                correlation = correlation_matrix[0, 1]

                # Handle NaN (can occur with constant prices)
                if np.isnan(correlation):
                    correlation = 0.0

                correlation_decimal = Decimal(str(round(correlation, 4)))

                # Cache the result
                self._correlation_cache[cache_key] = (
                    correlation_decimal,
                    datetime.utcnow(),
                )

                # Check for alerts
                await self._check_correlation_alert(
                    symbol1, symbol2, correlation_decimal
                )

                # Store in database
                await self._store_correlation(
                    symbol1, symbol2, correlation_decimal, window_minutes
                )

                logger.info(
                    "correlation_calculated",
                    symbol1=symbol1,
                    symbol2=symbol2,
                    correlation=correlation_decimal,
                    window_minutes=window_minutes,
                    data_points=min_length,
                )

                return correlation_decimal

            except Exception as e:
                logger.error(
                    "correlation_calculation_error",
                    symbol1=symbol1,
                    symbol2=symbol2,
                    error=str(e),
                )
                return Decimal("0")

    async def get_correlation_matrix(
        self, symbols: Optional[list[str]] = None
    ) -> np.ndarray:
        """Get correlation matrix for specified symbols.

        Args:
            symbols: List of symbols (uses all available if None)

        Returns:
            Numpy array with correlation matrix
        """
        async with self._lock:
            if symbols is None:
                symbols = list(self._price_histories.keys())

            if len(symbols) < 2:
                return np.array([[1.0]])

            n = len(symbols)
            matrix = np.eye(n)  # Initialize with 1s on diagonal

            # Calculate all pairwise correlations
            for i in range(n):
                for j in range(i + 1, n):
                    correlation = await self.calculate_pair_correlation(
                        symbols[i], symbols[j]
                    )
                    matrix[i, j] = float(correlation)
                    matrix[j, i] = float(correlation)  # Symmetric

            return matrix

    async def get_highly_correlated_pairs(
        self, threshold: Optional[Decimal] = None
    ) -> list[tuple[str, str, Decimal]]:
        """Get pairs with correlation above threshold.

        Args:
            threshold: Correlation threshold (uses WARNING_THRESHOLD if None)

        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        threshold = threshold or self.WARNING_THRESHOLD
        high_correlations = []

        async with self._lock:
            symbols = list(self._price_histories.keys())

            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = await self.calculate_pair_correlation(
                        symbols[i], symbols[j]
                    )

                    if abs(correlation) >= threshold:
                        high_correlations.append((symbols[i], symbols[j], correlation))

            # Sort by correlation strength
            high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        return high_correlations

    async def analyze_portfolio_correlation(
        self, positions: list[Position]
    ) -> dict[str, any]:
        """Analyze correlation across portfolio positions.

        Args:
            positions: List of active positions

        Returns:
            Dictionary with correlation analysis
        """
        if len(positions) < 2:
            return {
                "max_correlation": Decimal("0"),
                "avg_correlation": Decimal("0"),
                "correlation_pairs": [],
                "risk_level": "LOW",
            }

        symbols = [p.symbol for p in positions]
        correlations = []
        correlation_pairs = []

        # Calculate all pairwise correlations
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = await self.calculate_pair_correlation(symbols[i], symbols[j])
                correlations.append(abs(corr))
                correlation_pairs.append((symbols[i], symbols[j], corr))

        max_correlation = max(correlations) if correlations else Decimal("0")
        avg_correlation = (
            sum(correlations) / len(correlations) if correlations else Decimal("0")
        )

        # Determine risk level
        if max_correlation >= self.CRITICAL_THRESHOLD:
            risk_level = "CRITICAL"
        elif max_correlation >= self.WARNING_THRESHOLD:
            risk_level = "WARNING"
        else:
            risk_level = "LOW"

        # Sort pairs by correlation
        correlation_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        return {
            "max_correlation": max_correlation,
            "avg_correlation": avg_correlation,
            "correlation_pairs": correlation_pairs[:5],  # Top 5 correlations
            "risk_level": risk_level,
            "positions_analyzed": len(positions),
            "threshold_warning": self.WARNING_THRESHOLD,
            "threshold_critical": self.CRITICAL_THRESHOLD,
        }

    async def get_recent_alerts(self, limit: int = 10) -> list[CorrelationAlert]:
        """Get recent correlation alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        async with self._lock:
            return self._alerts[-limit:]

    async def clear_alerts(self) -> None:
        """Clear all alerts."""
        async with self._lock:
            self._alerts.clear()
            logger.info("correlation_alerts_cleared")

    # Private methods

    async def _load_price_history(self, symbol: str, window_minutes: int) -> None:
        """Load price history from database.

        Args:
            symbol: Trading pair symbol
            window_minutes: Time window to load
        """
        try:
            # Load from repository
            start_time = datetime.utcnow() - timedelta(
                minutes=window_minutes * 2
            )  # Load extra for buffer
            price_data = await self.repository.get_price_history(
                symbol=symbol, start_time=start_time, end_time=datetime.utcnow()
            )

            if price_data:
                history = PriceHistory(symbol=symbol)
                for data_point in price_data:
                    history.add_price(data_point.price, data_point.timestamp)

                self._price_histories[symbol] = history

                logger.debug(
                    "price_history_loaded", symbol=symbol, data_points=len(price_data)
                )
        except Exception as e:
            logger.error("failed_to_load_price_history", symbol=symbol, error=str(e))

    async def _check_correlation_alert(
        self, symbol1: str, symbol2: str, correlation: Decimal
    ) -> None:
        """Check if correlation triggers an alert.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Correlation coefficient
        """
        abs_correlation = abs(correlation)

        if abs_correlation >= self.CRITICAL_THRESHOLD:
            severity = "CRITICAL"
            message = (
                f"Critical correlation detected between {symbol1} and {symbol2}: "
                f"{correlation:.2%}. Consider reducing exposure."
            )
        elif abs_correlation >= self.WARNING_THRESHOLD:
            severity = "WARNING"
            message = (
                f"High correlation detected between {symbol1} and {symbol2}: "
                f"{correlation:.2%}. Monitor closely."
            )
        else:
            return  # No alert needed

        alert = CorrelationAlert(
            alert_id=str(uuid.uuid4()),
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=correlation,
            threshold=(
                self.CRITICAL_THRESHOLD
                if severity == "CRITICAL"
                else self.WARNING_THRESHOLD
            ),
            timestamp=datetime.utcnow(),
            message=message,
            severity=severity,
        )

        self._alerts.append(alert)

        # Keep only recent alerts (last 100)
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

        logger.warning(
            "correlation_alert",
            severity=severity,
            symbol1=symbol1,
            symbol2=symbol2,
            correlation=correlation,
            message=message,
        )

    async def _store_correlation(
        self, symbol1: str, symbol2: str, correlation: Decimal, window_minutes: int
    ) -> None:
        """Store correlation in database.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            correlation: Correlation coefficient
            window_minutes: Calculation window
        """
        try:
            await self.repository.store_correlation(
                symbol1=symbol1,
                symbol2=symbol2,
                correlation=correlation,
                window_minutes=window_minutes,
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(
                "failed_to_store_correlation",
                symbol1=symbol1,
                symbol2=symbol2,
                error=str(e),
            )
