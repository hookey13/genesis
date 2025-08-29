"""
Correlation spike detector for multi-asset risk monitoring.

Calculates real-time correlation matrices and detects dangerous
correlation spikes that could amplify portfolio risk during
market stress events.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclass
class PricePoint:
    """Single price observation."""

    timestamp: datetime
    price: Decimal
    volume: Decimal


@dataclass
class CorrelationAlert:
    """Alert for correlation spike detection."""

    symbol1: str
    symbol2: str
    correlation: Decimal
    rolling_avg: Decimal
    spike_magnitude: Decimal  # How much above normal
    window_minutes: int
    detected_at: datetime


class CorrelationSpikeDetector:
    """
    Detects dangerous correlation spikes between trading pairs.

    High correlations during market stress can lead to:
    - Amplified losses across positions
    - Reduced diversification benefits
    - Systemic risk exposure
    """

    def __init__(
        self,
        event_bus: EventBus,
        window_minutes: int = 30,
        spike_threshold: Decimal = Decimal("0.80"),
        min_observations: int = 20,
    ):
        """
        Initialize correlation spike detector.

        Args:
            event_bus: Event bus for publishing alerts
            window_minutes: Rolling window for correlation calculation
            spike_threshold: Correlation threshold for alerts (0.80 = 80%)
            min_observations: Minimum data points for correlation calc
        """
        self.event_bus = event_bus
        self.window_minutes = window_minutes
        self.spike_threshold = spike_threshold
        self.min_observations = min_observations

        # Price history for each symbol
        self.price_history: dict[str, list[PricePoint]] = defaultdict(list)

        # Current correlation matrix
        self.correlation_matrix: dict[tuple[str, str], Decimal] = {}

        # Rolling correlation averages for spike detection
        self.correlation_history: dict[
            tuple[str, str], list[tuple[datetime, Decimal]]
        ] = defaultdict(list)

        # Active alerts
        self.active_alerts: dict[tuple[str, str], CorrelationAlert] = {}

        # Statistics
        self.calculations_performed = 0
        self.alerts_triggered = 0
        self.highest_correlation = Decimal("0")

        logger.info(
            "CorrelationSpikeDetector initialized",
            window_minutes=window_minutes,
            spike_threshold=float(spike_threshold),
        )

    def add_price_observation(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Add a price observation for correlation tracking.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Trading volume
            timestamp: Observation time (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Add to history
        self.price_history[symbol].append(
            PricePoint(timestamp=timestamp, price=price, volume=volume)
        )

        # Clean old data
        self._clean_old_observations(symbol)

    def _clean_old_observations(self, symbol: str) -> None:
        """Remove observations outside the analysis window."""
        cutoff = datetime.now(UTC) - timedelta(minutes=self.window_minutes * 2)

        self.price_history[symbol] = [
            point for point in self.price_history[symbol] if point.timestamp >= cutoff
        ]

    async def calculate_correlation_matrix(self) -> dict[tuple[str, str], Decimal]:
        """
        Calculate current correlation matrix for all tracked symbols.

        Returns:
            Dictionary mapping symbol pairs to correlation coefficients
        """
        self.calculations_performed += 1

        # Get symbols with sufficient data
        valid_symbols = [
            symbol
            for symbol, history in self.price_history.items()
            if len(history) >= self.min_observations
        ]

        if len(valid_symbols) < 2:
            return {}

        # Prepare data for correlation calculation
        cutoff = datetime.now(UTC) - timedelta(minutes=self.window_minutes)

        # Create price series for each symbol
        price_series = {}
        for symbol in valid_symbols:
            # Get recent prices
            recent_prices = [
                point
                for point in self.price_history[symbol]
                if point.timestamp >= cutoff
            ]

            if len(recent_prices) >= self.min_observations:
                # Convert to returns
                prices = [float(point.price) for point in recent_prices]
                returns = np.diff(np.log(prices))  # Log returns
                price_series[symbol] = returns

        # Calculate correlations
        new_matrix = {}

        for i, symbol1 in enumerate(valid_symbols):
            for symbol2 in valid_symbols[i + 1 :]:
                if symbol1 in price_series and symbol2 in price_series:
                    # Ensure same length
                    min_len = min(
                        len(price_series[symbol1]), len(price_series[symbol2])
                    )

                    if min_len >= self.min_observations - 1:  # -1 because of diff
                        series1 = price_series[symbol1][-min_len:]
                        series2 = price_series[symbol2][-min_len:]

                        # Calculate correlation
                        correlation = np.corrcoef(series1, series2)[0, 1]

                        # Handle NaN (can occur with constant prices)
                        if not np.isnan(correlation):
                            key = tuple(sorted([symbol1, symbol2]))
                            new_matrix[key] = Decimal(str(abs(correlation)))

                            # Track highest
                            if new_matrix[key] > self.highest_correlation:
                                self.highest_correlation = new_matrix[key]

                            # Update history for spike detection
                            now = datetime.now(UTC)
                            self.correlation_history[key].append((now, new_matrix[key]))

                            # Keep only recent history (last hour)
                            hour_ago = now - timedelta(hours=1)
                            self.correlation_history[key] = [
                                (ts, corr)
                                for ts, corr in self.correlation_history[key]
                                if ts >= hour_ago
                            ]

        self.correlation_matrix = new_matrix

        # Check for spikes
        await self._check_for_spikes(new_matrix)

        return new_matrix

    async def _check_for_spikes(
        self, current_matrix: dict[tuple[str, str], Decimal]
    ) -> None:
        """
        Check for correlation spikes compared to rolling average.

        Args:
            current_matrix: Current correlation matrix
        """
        for pair, correlation in current_matrix.items():
            # Check if above threshold
            if correlation >= self.spike_threshold:
                # Calculate rolling average
                history = self.correlation_history.get(pair, [])

                if len(history) >= 5:  # Need some history for comparison
                    # Calculate average correlation over past period
                    avg_correlation = sum(corr for _, corr in history[:-1]) / (
                        len(history) - 1
                    )

                    # Calculate spike magnitude
                    spike_magnitude = correlation - avg_correlation

                    # Check if this is a new spike or update
                    if (
                        pair not in self.active_alerts
                        or spike_magnitude > self.active_alerts[pair].spike_magnitude
                    ):
                        alert = CorrelationAlert(
                            symbol1=pair[0],
                            symbol2=pair[1],
                            correlation=correlation,
                            rolling_avg=avg_correlation,
                            spike_magnitude=spike_magnitude,
                            window_minutes=self.window_minutes,
                            detected_at=datetime.now(UTC),
                        )

                        self.active_alerts[pair] = alert
                        self.alerts_triggered += 1

                        # Publish alert
                        await self._publish_alert(alert)
                else:
                    # No history but above threshold - still alert
                    if (
                        correlation >= self.spike_threshold
                        and pair not in self.active_alerts
                    ):
                        alert = CorrelationAlert(
                            symbol1=pair[0],
                            symbol2=pair[1],
                            correlation=correlation,
                            rolling_avg=correlation,  # No history, use current
                            spike_magnitude=Decimal("0"),
                            window_minutes=self.window_minutes,
                            detected_at=datetime.now(UTC),
                        )

                        self.active_alerts[pair] = alert
                        self.alerts_triggered += 1

                        await self._publish_alert(alert)

            # Clear alert if correlation dropped
            elif (
                pair in self.active_alerts
                and correlation < self.spike_threshold * Decimal("0.95")
            ):
                # 5% buffer to prevent flapping
                del self.active_alerts[pair]

                # Publish recovery event
                await self.event_bus.publish(
                    Event(
                        event_type=EventType.CORRELATION_ALERT,
                        event_data={
                            "alert_type": "correlation_recovered",
                            "symbol1": pair[0],
                            "symbol2": pair[1],
                            "correlation": float(correlation),
                            "threshold": float(self.spike_threshold),
                        },
                    ),
                    priority=EventPriority.HIGH,
                )

    async def _publish_alert(self, alert: CorrelationAlert) -> None:
        """
        Publish correlation spike alert.

        Args:
            alert: Correlation alert to publish
        """
        logger.warning(
            "Correlation spike detected",
            symbol1=alert.symbol1,
            symbol2=alert.symbol2,
            correlation=float(alert.correlation),
            rolling_avg=float(alert.rolling_avg),
            spike_magnitude=float(alert.spike_magnitude),
        )

        await self.event_bus.publish(
            Event(
                event_type=EventType.CORRELATION_ALERT,
                event_data={
                    "alert_type": "correlation_spike",
                    "symbol1": alert.symbol1,
                    "symbol2": alert.symbol2,
                    "correlation": float(alert.correlation),
                    "rolling_avg": float(alert.rolling_avg),
                    "spike_magnitude": float(alert.spike_magnitude),
                    "window_minutes": alert.window_minutes,
                    "recommendation": self._get_recommendation(alert),
                },
            ),
            priority=EventPriority.CRITICAL,
        )

    def _get_recommendation(self, alert: CorrelationAlert) -> str:
        """
        Get risk management recommendation for correlation spike.

        Args:
            alert: Correlation alert

        Returns:
            Recommendation string
        """
        if alert.correlation >= Decimal("0.95"):
            return "CRITICAL: Reduce positions immediately - near-perfect correlation"
        elif alert.correlation >= Decimal("0.90"):
            return "HIGH: Consider closing one position - excessive correlation"
        elif alert.correlation >= Decimal("0.85"):
            return "MEDIUM: Reduce position sizes - high correlation risk"
        else:
            return (
                "WARNING: Monitor positions - correlation approaching dangerous levels"
            )

    def get_position_recommendations(
        self, positions: dict[str, Decimal]
    ) -> list[dict[str, Any]]:
        """
        Get position adjustment recommendations based on correlations.

        Args:
            positions: Current positions (symbol -> size)

        Returns:
            List of recommended adjustments
        """
        recommendations = []

        # Check each position pair
        symbols = list(positions.keys())

        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i + 1 :]:
                key = tuple(sorted([symbol1, symbol2]))

                if key in self.correlation_matrix:
                    correlation = self.correlation_matrix[key]

                    if correlation >= self.spike_threshold:
                        # Calculate combined exposure
                        combined_exposure = abs(positions[symbol1]) + abs(
                            positions[symbol2]
                        )

                        # Recommend reduction based on correlation level
                        if correlation >= Decimal("0.95"):
                            reduction_pct = Decimal("0.75")  # Reduce by 75%
                        elif correlation >= Decimal("0.90"):
                            reduction_pct = Decimal("0.50")  # Reduce by 50%
                        elif correlation >= Decimal("0.85"):
                            reduction_pct = Decimal("0.30")  # Reduce by 30%
                        else:
                            reduction_pct = Decimal("0.20")  # Reduce by 20%

                        recommendations.append(
                            {
                                "symbol1": symbol1,
                                "symbol2": symbol2,
                                "correlation": float(correlation),
                                "combined_exposure": float(combined_exposure),
                                "recommended_reduction": float(reduction_pct),
                                "priority": (
                                    "HIGH"
                                    if correlation >= Decimal("0.90")
                                    else "MEDIUM"
                                ),
                            }
                        )

        # Sort by priority and correlation
        recommendations.sort(
            key=lambda x: (x["priority"] == "HIGH", x["correlation"]), reverse=True
        )

        return recommendations

    def get_correlation_matrix_summary(self) -> dict[str, Any]:
        """
        Get summary of current correlation state.

        Returns:
            Summary dictionary
        """
        if not self.correlation_matrix:
            return {
                "matrix_size": 0,
                "highest_correlation": 0,
                "pairs_above_threshold": 0,
                "active_alerts": 0,
                "recommendations": "Insufficient data for correlation analysis",
            }

        pairs_above_threshold = sum(
            1
            for corr in self.correlation_matrix.values()
            if corr >= self.spike_threshold
        )

        return {
            "matrix_size": len(self.correlation_matrix),
            "highest_correlation": float(max(self.correlation_matrix.values())),
            "average_correlation": float(
                sum(self.correlation_matrix.values()) / len(self.correlation_matrix)
            ),
            "pairs_above_threshold": pairs_above_threshold,
            "active_alerts": len(self.active_alerts),
            "spike_threshold": float(self.spike_threshold),
            "window_minutes": self.window_minutes,
            "calculations_performed": self.calculations_performed,
            "alerts_triggered": self.alerts_triggered,
        }

    def reset(self) -> None:
        """Reset detector state (useful for testing)."""
        self.price_history.clear()
        self.correlation_matrix.clear()
        self.correlation_history.clear()
        self.active_alerts.clear()
        self.calculations_performed = 0
        self.alerts_triggered = 0
        self.highest_correlation = Decimal("0")

        logger.info("Correlation spike detector reset")
