"""Portfolio correlation monitoring and analysis module."""

from __future__ import annotations

from typing import Optional

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from genesis.core.constants import TradingTier
from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import Position
from genesis.engine.event_bus import EventBus
from genesis.utils.decorators import requires_tier, with_timeout

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"


@dataclass
class CorrelationAlert:
    """Alert for correlation threshold breach."""
    alert_id: UUID
    timestamp: datetime
    correlation_level: Decimal
    threshold: Decimal
    affected_positions: list[UUID]
    severity: str  # "warning" or "critical"
    message: str


@dataclass
class TradeSuggestion:
    """Decorrelation trade suggestion."""
    suggestion_id: UUID
    action: str  # "reduce", "close", "hedge"
    position_id: UUID
    suggested_quantity: Decimal
    expected_correlation_impact: Decimal
    rationale: str
    transaction_cost_estimate: Decimal


@dataclass
class CorrelationImpact:
    """Pre-trade correlation impact analysis."""
    current_correlation: Decimal
    projected_correlation: Decimal
    correlation_change: Decimal
    risk_assessment: str  # "low", "medium", "high"
    recommendation: str


@dataclass
class StressTestResult:
    """Correlation stress test results."""
    scenario: str
    correlation_spike: Decimal
    portfolio_impact: Decimal
    max_drawdown: Decimal
    positions_at_risk: list[UUID]
    timestamp: datetime


class CorrelationMonitor:
    """Monitor and analyze portfolio correlations."""

    def __init__(self, event_bus: Optional[EventBus] = None, config: Optional[dict] = None):
        """Initialize correlation monitor.

        Args:
            event_bus: Event bus for publishing alerts
            config: Configuration dictionary from trading_rules.yaml
        """
        self.event_bus = event_bus
        self.correlation_cache: dict[str, tuple[np.ndarray, datetime]] = {}
        self.historical_data: dict[UUID, pd.Series] = {}

        # Load configuration
        if config and 'correlation_monitoring' in config:
            corr_config = config['correlation_monitoring']
            self.cache_ttl = timedelta(seconds=corr_config['analysis']['cache_ttl_seconds'])
            self.warning_threshold = Decimal(str(corr_config['thresholds']['warning']))
            self.critical_threshold = Decimal(str(corr_config['thresholds']['critical']))
            self.alert_cooldown = timedelta(minutes=corr_config['alerting']['alert_cooldown_minutes'])
            self.max_alerts_per_day = corr_config['alerting']['max_alerts_per_day']
        else:
            # Default values
            self.cache_ttl = timedelta(seconds=5)
            self.warning_threshold = Decimal("0.6")
            self.critical_threshold = Decimal("0.8")
            self.alert_cooldown = timedelta(minutes=15)
            self.max_alerts_per_day = 50

        self.alert_history: dict[str, datetime] = {}  # Track last alert time per position pair
        self.daily_alert_count = 0
        self.alert_reset_time = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        # Set tier for decorator checks (default to HUNTER as this is a Hunter+ feature)
        self.tier = TradingTier.HUNTER

    @requires_tier(TradingTier.HUNTER)
    @with_timeout(500)  # 500ms timeout for correlation calculation
    async def calculate_correlation_matrix(self, positions: list[Position]) -> np.ndarray:
        """Calculate correlation matrix for positions.

        Args:
            positions: List of active positions

        Returns:
            Correlation matrix as numpy array
        """
        if not positions:
            return np.array([])

        if len(positions) == 1:
            return np.array([[1.0]])

        # Check cache
        cache_key = self._get_cache_key(positions)
        if cache_key in self.correlation_cache:
            cached_matrix, cached_time = self.correlation_cache[cache_key]
            if datetime.now(UTC) - cached_time < self.cache_ttl:
                return cached_matrix

        # Build price data matrix
        price_data = []
        for position in positions:
            # Simulate historical prices for now (would fetch from market data)
            prices = self._get_historical_prices(position)
            price_data.append(prices)

        # Calculate correlation matrix
        price_df = pd.DataFrame(price_data).T
        correlation_matrix = price_df.corr().values

        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        # Cache result
        self.correlation_cache[cache_key] = (
            correlation_matrix,
            datetime.now(UTC)
        )

        return correlation_matrix

    @requires_tier(TradingTier.HUNTER)
    async def track_correlation_history(self, positions: list[Position], window_days: int = 30) -> dict:
        """Track historical correlation over specified window.

        Args:
            positions: List of positions to analyze
            window_days: Historical window in days

        Returns:
            Dictionary with historical correlation data
        """
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=window_days)

        # Calculate daily correlations
        daily_correlations = []
        current_date = start_date

        while current_date <= end_date:
            # Simulate historical correlation calculation
            daily_matrix = await self.calculate_correlation_matrix(positions)
            if daily_matrix.size > 0:
                avg_correlation = self._calculate_average_correlation(daily_matrix)
                daily_correlations.append({
                    "date": current_date.isoformat(),
                    "average_correlation": float(avg_correlation),
                    "max_correlation": float(daily_matrix[daily_matrix < 1].max()) if daily_matrix.size > 1 else 0.0
                })
            current_date += timedelta(days=1)

        return {
            "window_days": window_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily_correlations": daily_correlations,
            "average_period_correlation": np.mean([d["average_correlation"] for d in daily_correlations]) if daily_correlations else 0.0
        }

    @requires_tier(TradingTier.HUNTER)
    async def analyze_by_market_regime(self, positions: list[Position], market_state: MarketState) -> dict:
        """Analyze correlation by market regime.

        Args:
            positions: List of positions
            market_state: Current market regime

        Returns:
            Correlation analysis by market regime
        """
        correlation_matrix = await self.calculate_correlation_matrix(positions)

        # Adjust correlation expectations based on market regime
        regime_multipliers = {
            MarketState.TRENDING_UP: 1.1,
            MarketState.TRENDING_DOWN: 1.2,
            MarketState.VOLATILE: 1.3,
            MarketState.RANGING: 0.9,
            MarketState.CALM: 0.8
        }

        multiplier = regime_multipliers.get(market_state, 1.0)
        adjusted_matrix = correlation_matrix * multiplier

        # Ensure correlations stay within [-1, 1]
        adjusted_matrix = np.clip(adjusted_matrix, -1.0, 1.0)

        return {
            "market_state": market_state.value,
            "base_correlation": float(self._calculate_average_correlation(correlation_matrix)),
            "adjusted_correlation": float(self._calculate_average_correlation(adjusted_matrix)),
            "regime_multiplier": multiplier,
            "risk_assessment": self._assess_regime_risk(adjusted_matrix, market_state)
        }

    @requires_tier(TradingTier.HUNTER)
    async def check_correlation_thresholds(self, positions: list[Position]) -> list[CorrelationAlert]:
        """Check correlation thresholds and generate alerts.

        Args:
            positions: List of positions to check

        Returns:
            List of correlation alerts
        """
        if len(positions) < 2:
            return []

        # Reset daily counter if needed
        current_time = datetime.now(UTC)
        if current_time.date() > self.alert_reset_time.date():
            self.daily_alert_count = 0
            self.alert_reset_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

        if self.daily_alert_count >= self.max_alerts_per_day:
            logger.warning(f"Daily alert limit reached ({self.max_alerts_per_day})")
            return []

        alerts = []
        correlation_matrix = await self.calculate_correlation_matrix(positions)

        # Check pairwise correlations
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                correlation = Decimal(str(abs(correlation_matrix[i][j])))

                # Check if correlation exceeds warning threshold
                if correlation > self.warning_threshold:
                    # Create unique key for this position pair
                    pair_key = "_".join(sorted([str(positions[i].position_id), str(positions[j].position_id)]))

                    # Check cooldown
                    if pair_key in self.alert_history:
                        last_alert = self.alert_history[pair_key]
                        if current_time - last_alert < self.alert_cooldown:
                            continue  # Skip this alert due to cooldown

                    # Determine severity
                    severity = "critical" if correlation > self.critical_threshold else "warning"
                    threshold = self.critical_threshold if severity == "critical" else self.warning_threshold

                    alert = CorrelationAlert(
                        alert_id=uuid4(),
                        timestamp=current_time,
                        correlation_level=correlation,
                        threshold=threshold,
                        affected_positions=[positions[i].position_id, positions[j].position_id],
                        severity=severity,
                        message=f"High correlation {correlation:.2%} between {positions[i].symbol} and {positions[j].symbol}"
                    )
                    alerts.append(alert)

                    # Update alert history
                    self.alert_history[pair_key] = current_time
                    self.daily_alert_count += 1

                    # Publish alert to event bus
                    if self.event_bus:
                        event = Event(
                            type=EventType.RISK_ALERT,
                            priority=EventPriority.HIGH if severity == "critical" else EventPriority.NORMAL,
                            data={"alert": alert}
                        )
                        await self.event_bus.publish(event)

        return alerts

    @requires_tier(TradingTier.HUNTER)
    async def suggest_decorrelation_trades(self, current_positions: list[Position]) -> list[TradeSuggestion]:
        """Suggest trades to reduce portfolio correlation.

        Args:
            current_positions: Current portfolio positions

        Returns:
            List of decorrelation trade suggestions
        """
        if len(current_positions) < 2:
            return []

        suggestions = []
        correlation_matrix = await self.calculate_correlation_matrix(current_positions)

        # Find highly correlated pairs
        for i in range(len(current_positions)):
            for j in range(i + 1, len(current_positions)):
                correlation = abs(correlation_matrix[i][j])

                if correlation > 0.7:  # High correlation threshold
                    # Suggest reducing the smaller position
                    smaller_position = current_positions[i] if current_positions[i].dollar_value < current_positions[j].dollar_value else current_positions[j]

                    suggestion = TradeSuggestion(
                        suggestion_id=uuid4(),
                        action="reduce",
                        position_id=smaller_position.position_id,
                        suggested_quantity=smaller_position.quantity * Decimal("0.5"),  # Reduce by 50%
                        expected_correlation_impact=Decimal(str(correlation * 0.3)),  # Estimate 30% reduction
                        rationale=f"High correlation ({correlation:.2f}) with other position. Reducing exposure.",
                        transaction_cost_estimate=smaller_position.dollar_value * Decimal("0.001")  # 0.1% estimate
                    )
                    suggestions.append(suggestion)

        # Apply efficient frontier optimization (simplified)
        suggestions = self._optimize_suggestions(suggestions, current_positions)

        return suggestions[:5]  # Return top 5 suggestions

    @requires_tier(TradingTier.HUNTER)
    async def calculate_correlation_impact(self, new_position: Position, existing_positions: list[Position]) -> CorrelationImpact:
        """Calculate correlation impact of new position.

        Args:
            new_position: Proposed new position
            existing_positions: Existing portfolio positions

        Returns:
            Correlation impact analysis
        """
        if not existing_positions:
            return CorrelationImpact(
                current_correlation=Decimal("0"),
                projected_correlation=Decimal("0"),
                correlation_change=Decimal("0"),
                risk_assessment="low",
                recommendation="No existing positions. Safe to proceed."
            )

        # Calculate current correlation
        current_matrix = await self.calculate_correlation_matrix(existing_positions)
        current_avg = Decimal(str(self._calculate_average_correlation(current_matrix)))

        # Calculate projected correlation with new position
        all_positions = existing_positions + [new_position]
        projected_matrix = await self.calculate_correlation_matrix(all_positions)
        projected_avg = Decimal(str(self._calculate_average_correlation(projected_matrix)))

        correlation_change = projected_avg - current_avg

        # Risk assessment
        if projected_avg > Decimal("0.8"):
            risk_assessment = "high"
            recommendation = "High correlation risk. Consider alternative positions."
        elif projected_avg > Decimal("0.6"):
            risk_assessment = "medium"
            recommendation = "Moderate correlation. Proceed with reduced size."
        else:
            risk_assessment = "low"
            recommendation = "Low correlation impact. Safe to proceed."

        return CorrelationImpact(
            current_correlation=current_avg,
            projected_correlation=projected_avg,
            correlation_change=correlation_change,
            risk_assessment=risk_assessment,
            recommendation=recommendation
        )

    @requires_tier(TradingTier.HUNTER)
    async def run_stress_test(self, positions: list[Position], correlation_spike: float = 0.8) -> StressTestResult:
        """Run correlation stress test.

        Args:
            positions: Positions to stress test
            correlation_spike: Correlation level to simulate

        Returns:
            Stress test results
        """
        if not positions:
            return StressTestResult(
                scenario="Empty portfolio",
                correlation_spike=Decimal(str(correlation_spike)),
                portfolio_impact=Decimal("0"),
                max_drawdown=Decimal("0"),
                positions_at_risk=[],
                timestamp=datetime.now(UTC)
            )

        # Create stressed correlation matrix
        n = len(positions)
        stressed_matrix = np.full((n, n), correlation_spike)
        np.fill_diagonal(stressed_matrix, 1.0)

        # Calculate portfolio impact
        portfolio_value = sum(p.dollar_value for p in positions)

        # Simulate drawdown (simplified model)
        # Higher correlation = higher potential drawdown
        correlation_factor = Decimal(str(correlation_spike))
        volatility_factor = Decimal("0.2")  # 20% volatility assumption
        max_drawdown = portfolio_value * correlation_factor * volatility_factor

        # Identify positions at risk
        positions_at_risk = []
        for position in positions:
            position_risk = position.dollar_value * correlation_factor * volatility_factor
            if position_risk > position.dollar_value * Decimal("0.15"):  # 15% threshold
                positions_at_risk.append(position.position_id)

        portfolio_impact = max_drawdown / portfolio_value if portfolio_value > 0 else Decimal("0")

        return StressTestResult(
            scenario=f"Correlation spike to {correlation_spike:.0%}",
            correlation_spike=Decimal(str(correlation_spike)),
            portfolio_impact=portfolio_impact,
            max_drawdown=max_drawdown,
            positions_at_risk=positions_at_risk,
            timestamp=datetime.now(UTC)
        )

    def _get_cache_key(self, positions: list[Position]) -> str:
        """Generate cache key for positions."""
        position_ids = sorted([str(p.position_id) for p in positions])
        return "_".join(position_ids)

    def _get_historical_prices(self, position: Position) -> np.ndarray:
        """Get historical prices for position (simulated)."""
        # In production, would fetch from market data service
        # Simulating with random walk for now
        np.random.seed(hash(str(position.position_id)) % 2**32)
        base_price = float(position.current_price)
        returns = np.random.normal(0, 0.02, 100)  # 2% volatility
        prices = base_price * np.exp(np.cumsum(returns))
        return prices

    def _calculate_average_correlation(self, correlation_matrix: np.ndarray) -> float:
        """Calculate average correlation excluding diagonal."""
        if correlation_matrix.size == 0:
            return 0.0
        if correlation_matrix.size == 1:
            return 0.0

        # Get upper triangle excluding diagonal
        upper_triangle = np.triu(correlation_matrix, k=1)
        non_zero_count = np.sum(upper_triangle != 0)

        if non_zero_count == 0:
            return 0.0

        return float(np.sum(np.abs(upper_triangle)) / non_zero_count)

    def _assess_regime_risk(self, correlation_matrix: np.ndarray, market_state: MarketState) -> str:
        """Assess risk based on correlation and market regime."""
        avg_correlation = self._calculate_average_correlation(correlation_matrix)

        if market_state in [MarketState.VOLATILE, MarketState.TRENDING_DOWN]:
            if avg_correlation > 0.7:
                return "Very High - Correlation amplifies regime risk"
            elif avg_correlation > 0.5:
                return "High - Significant regime correlation risk"
            else:
                return "Medium - Moderate regime risk"
        else:
            if avg_correlation > 0.8:
                return "High - Excessive correlation"
            elif avg_correlation > 0.6:
                return "Medium - Notable correlation"
            else:
                return "Low - Acceptable correlation"

    def _optimize_suggestions(self, suggestions: list[TradeSuggestion], positions: list[Position]) -> list[TradeSuggestion]:
        """Optimize suggestions using efficient frontier principles."""
        # Simplified optimization - sort by expected impact and cost
        if not suggestions:
            return suggestions

        # Score each suggestion
        for suggestion in suggestions:
            impact_score = float(suggestion.expected_correlation_impact)
            cost_score = float(suggestion.transaction_cost_estimate / positions[0].dollar_value) if positions else 0
            suggestion.score = impact_score - cost_score  # Higher is better

        # Sort by score
        suggestions.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)

        return suggestions
