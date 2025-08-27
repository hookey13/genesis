"""
Real-time correlation monitoring system for multi-strategy portfolios.

Monitors correlations between strategy returns, positions, and market factors
to detect concentration risk and correlation spikes.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import structlog

from genesis.core.events import Event, EventType
from genesis.core.models import PositionCorrelation
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclass
class CorrelationWindow:
    """Rolling window for correlation calculation."""
    window_size: int = 20  # Number of observations
    data: dict[str, list[float]] = field(default_factory=dict)
    timestamps: list[datetime] = field(default_factory=list)

    def add_observation(self, timestamp: datetime, values: dict[str, float]) -> None:
        """Add new observation to the window."""
        self.timestamps.append(timestamp)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        for key, value in values.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
            if len(self.data[key]) > self.window_size:
                self.data[key].pop(0)

    def is_ready(self) -> bool:
        """Check if window has enough data."""
        if not self.data:
            return False
        return all(len(values) >= self.window_size for values in self.data.values())

    def get_dataframe(self) -> pd.DataFrame:
        """Convert window data to DataFrame."""
        return pd.DataFrame(self.data, index=self.timestamps)


@dataclass
class CorrelationAlert:
    """Correlation alert information."""
    alert_id: str
    timestamp: datetime
    entity1: str
    entity2: str
    correlation: Decimal
    threshold: Decimal
    severity: str  # "warning", "critical"
    message: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "entity1": self.entity1,
            "entity2": self.entity2,
            "correlation": str(self.correlation),
            "threshold": str(self.threshold),
            "severity": self.severity,
            "message": self.message
        }


class CorrelationMonitor:
    """
    Monitors real-time correlations between strategies and positions.
    
    Tracks rolling correlations and generates alerts when thresholds are breached.
    """

    def __init__(
        self,
        event_bus: EventBus,
        window_size: int = 20,
        warning_threshold: Decimal = Decimal("0.6"),
        critical_threshold: Decimal = Decimal("0.8")
    ):
        """
        Initialize correlation monitor.
        
        Args:
            event_bus: Event bus for publishing alerts
            window_size: Rolling window size for correlation calculation
            warning_threshold: Correlation threshold for warnings
            critical_threshold: Correlation threshold for critical alerts
        """
        self.event_bus = event_bus
        self.window_size = window_size
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # Data storage
        self.strategy_returns: CorrelationWindow = CorrelationWindow(window_size)
        self.position_returns: CorrelationWindow = CorrelationWindow(window_size)
        self.correlation_matrix: pd.DataFrame | None = None
        self.correlation_history: list[tuple[datetime, pd.DataFrame]] = []
        self.active_alerts: dict[str, CorrelationAlert] = {}

        # Monitoring task
        self._monitor_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start correlation monitoring."""
        if not self._monitor_task:
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            logger.info("Correlation monitor started")

    async def stop(self) -> None:
        """Stop correlation monitoring."""
        self._shutdown_event.set()
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Correlation monitor stopped")

    async def add_strategy_returns(
        self,
        timestamp: datetime,
        returns: dict[str, float]
    ) -> None:
        """
        Add strategy return observations.
        
        Args:
            timestamp: Observation timestamp
            returns: Dictionary of strategy_id -> return percentage
        """
        self.strategy_returns.add_observation(timestamp, returns)

        if self.strategy_returns.is_ready():
            await self._calculate_correlations()

    async def add_position_returns(
        self,
        timestamp: datetime,
        returns: dict[str, float]
    ) -> None:
        """
        Add position return observations.
        
        Args:
            timestamp: Observation timestamp
            returns: Dictionary of position_id -> return percentage
        """
        self.position_returns.add_observation(timestamp, returns)

    async def _calculate_correlations(self) -> None:
        """Calculate correlation matrix from current window."""
        try:
            # Get strategy returns DataFrame
            df = self.strategy_returns.get_dataframe()

            if df.shape[1] < 2:
                return  # Need at least 2 strategies

            # Calculate correlation matrix
            corr_matrix = df.corr()
            self.correlation_matrix = corr_matrix

            # Store in history
            self.correlation_history.append((datetime.now(UTC), corr_matrix))
            if len(self.correlation_history) > 100:
                self.correlation_history.pop(0)

            # Check for alerts
            await self._check_correlation_alerts(corr_matrix)

        except Exception as e:
            logger.error(f"Failed to calculate correlations: {e}")

    async def _check_correlation_alerts(self, corr_matrix: pd.DataFrame) -> None:
        """
        Check correlation matrix for threshold breaches.
        
        Args:
            corr_matrix: Correlation matrix to check
        """
        alerts_to_publish = []

        # Check each pair
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                entity1 = corr_matrix.columns[i]
                entity2 = corr_matrix.columns[j]
                correlation = Decimal(str(abs(corr_matrix.iloc[i, j])))

                alert_key = f"{entity1}:{entity2}"

                # Check thresholds
                if correlation >= self.critical_threshold:
                    alert = await self._create_alert(
                        entity1, entity2, correlation,
                        self.critical_threshold, "critical"
                    )
                    self.active_alerts[alert_key] = alert
                    alerts_to_publish.append(alert)

                elif correlation >= self.warning_threshold:
                    # Only create warning if no critical alert exists
                    if alert_key not in self.active_alerts or \
                       self.active_alerts[alert_key].severity != "critical":
                        alert = await self._create_alert(
                            entity1, entity2, correlation,
                            self.warning_threshold, "warning"
                        )
                        self.active_alerts[alert_key] = alert
                        alerts_to_publish.append(alert)

                else:
                    # Clear alert if correlation dropped
                    if alert_key in self.active_alerts:
                        del self.active_alerts[alert_key]

        # Publish alerts
        for alert in alerts_to_publish:
            await self._publish_alert(alert)

    async def _create_alert(
        self,
        entity1: str,
        entity2: str,
        correlation: Decimal,
        threshold: Decimal,
        severity: str
    ) -> CorrelationAlert:
        """Create correlation alert."""
        from uuid import uuid4

        message = (
            f"{'Critical' if severity == 'critical' else 'Warning'}: "
            f"Correlation between {entity1} and {entity2} is {correlation:.2f} "
            f"(threshold: {threshold})"
        )

        return CorrelationAlert(
            alert_id=str(uuid4()),
            timestamp=datetime.now(UTC),
            entity1=entity1,
            entity2=entity2,
            correlation=correlation,
            threshold=threshold,
            severity=severity,
            message=message
        )

    async def _publish_alert(self, alert: CorrelationAlert) -> None:
        """Publish correlation alert event."""
        await self.event_bus.publish(Event(
            event_type=EventType.CORRELATION_ALERT,
            event_data=alert.to_dict()
        ))

        logger.warning(
            "Correlation alert",
            severity=alert.severity,
            entity1=alert.entity1,
            entity2=alert.entity2,
            correlation=float(alert.correlation)
        )

    async def calculate_position_correlations(
        self,
        positions: list[dict]
    ) -> list[PositionCorrelation]:
        """
        Calculate correlations between positions.
        
        Args:
            positions: List of position dictionaries with historical data
            
        Returns:
            List of PositionCorrelation objects
        """
        correlations = []

        if len(positions) < 2:
            return correlations

        try:
            # Create DataFrame from position data
            position_data = {}
            for pos in positions:
                if pos.get("returns"):
                    position_data[pos["position_id"]] = pos["returns"]

            if len(position_data) < 2:
                return correlations

            df = pd.DataFrame(position_data)
            corr_matrix = df.corr()

            # Create PositionCorrelation objects
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    pos1_id = corr_matrix.columns[i]
                    pos2_id = corr_matrix.columns[j]
                    corr_value = Decimal(str(corr_matrix.iloc[i, j]))

                    correlation = PositionCorrelation(
                        position_a_id=pos1_id,
                        position_b_id=pos2_id,
                        correlation_coefficient=corr_value,
                        alert_triggered=abs(corr_value) >= self.warning_threshold
                    )
                    correlations.append(correlation)

        except Exception as e:
            logger.error(f"Failed to calculate position correlations: {e}")

        return correlations

    def get_correlation_matrix(self) -> pd.DataFrame | None:
        """
        Get current correlation matrix.
        
        Returns:
            Current correlation matrix or None
        """
        return self.correlation_matrix

    def get_correlation_summary(self) -> dict:
        """
        Get summary of current correlations.
        
        Returns:
            Summary dictionary
        """
        if self.correlation_matrix is None:
            return {
                "status": "insufficient_data",
                "message": "Not enough data for correlation calculation"
            }

        corr_matrix = self.correlation_matrix

        # Extract upper triangle (excluding diagonal)
        upper_triangle = np.triu(corr_matrix.values, k=1)
        correlations = upper_triangle[upper_triangle != 0]

        if len(correlations) == 0:
            return {
                "status": "no_correlations",
                "message": "No correlations to report"
            }

        return {
            "status": "active",
            "num_pairs": len(correlations),
            "average_correlation": float(np.mean(np.abs(correlations))),
            "max_correlation": float(np.max(np.abs(correlations))),
            "min_correlation": float(np.min(np.abs(correlations))),
            "warning_breaches": len([c for c in correlations
                                    if abs(c) >= float(self.warning_threshold)]),
            "critical_breaches": len([c for c in correlations
                                     if abs(c) >= float(self.critical_threshold)]),
            "active_alerts": len(self.active_alerts)
        }

    def get_high_correlation_pairs(
        self,
        threshold: Decimal | None = None
    ) -> list[tuple[str, str, Decimal]]:
        """
        Get pairs with high correlations.
        
        Args:
            threshold: Correlation threshold (uses warning threshold if not specified)
            
        Returns:
            List of (entity1, entity2, correlation) tuples
        """
        if self.correlation_matrix is None:
            return []

        threshold = threshold or self.warning_threshold
        high_pairs = []

        corr_matrix = self.correlation_matrix

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlation = abs(corr_matrix.iloc[i, j])
                if correlation >= float(threshold):
                    high_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        Decimal(str(correlation))
                    ))

        return sorted(high_pairs, key=lambda x: x[2], reverse=True)

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                # Recalculate correlations periodically
                if self.strategy_returns.is_ready():
                    await self._calculate_correlations()

                # Check for correlation spikes
                if self.correlation_matrix is not None:
                    await self._check_for_correlation_crisis()

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _check_for_correlation_crisis(self) -> None:
        """Check for market-wide correlation spike."""
        if self.correlation_matrix is None:
            return

        # Calculate average absolute correlation
        upper_triangle = np.triu(self.correlation_matrix.values, k=1)
        correlations = upper_triangle[upper_triangle != 0]

        if len(correlations) > 0:
            avg_correlation = np.mean(np.abs(correlations))

            # Crisis threshold: average correlation > 0.9
            if avg_correlation > 0.9:
                await self.event_bus.publish(Event(
                    event_type=EventType.CORRELATION_ALERT,
                    event_data={
                        "type": "correlation_crisis",
                        "average_correlation": float(avg_correlation),
                        "message": "Market-wide correlation crisis detected",
                        "severity": "critical"
                    }
                ))

                logger.critical(
                    "Correlation crisis detected",
                    average_correlation=avg_correlation
                )
