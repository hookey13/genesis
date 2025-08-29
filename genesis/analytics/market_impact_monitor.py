"""
Market impact monitoring system for Project GENESIS.

This module tracks and analyzes the market impact of order executions,
particularly for iceberg orders, to optimize future execution strategies.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from genesis.core.models import TradingTier
from genesis.data.repository import Repository
from genesis.engine.executor.base import Order, OrderSide
from genesis.exchange.models import OrderBook
from genesis.utils.decorators import with_timeout

logger = structlog.get_logger(__name__)


class ImpactSeverity(str, Enum):
    """Market impact severity levels."""

    NEGLIGIBLE = "NEGLIGIBLE"  # < 0.1%
    LOW = "LOW"  # 0.1% - 0.3%
    MODERATE = "MODERATE"  # 0.3% - 0.5%
    HIGH = "HIGH"  # 0.5% - 1.0%
    SEVERE = "SEVERE"  # > 1.0%


@dataclass
class MarketImpactMetric:
    """Individual market impact measurement."""

    impact_id: str
    execution_id: str | None
    slice_id: str | None
    symbol: str
    side: OrderSide
    pre_price: Decimal
    post_price: Decimal
    price_impact_percent: Decimal
    volume_executed: Decimal
    order_book_depth_usdt: Decimal | None
    bid_ask_spread: Decimal | None
    liquidity_consumed_percent: Decimal | None
    market_depth_1pct: Decimal | None
    market_depth_2pct: Decimal | None
    cumulative_impact: Decimal | None
    severity: ImpactSeverity
    measured_at: datetime
    notes: str | None = None


@dataclass
class ImpactAnalysis:
    """Comprehensive impact analysis for an execution."""

    execution_id: str
    symbol: str
    total_volume: Decimal
    slice_count: int

    # Impact metrics
    total_impact_percent: Decimal
    average_impact_per_slice: Decimal
    max_slice_impact: Decimal
    min_slice_impact: Decimal
    impact_std_deviation: Decimal

    # Market metrics
    average_spread_during_execution: Decimal
    liquidity_consumed_total: Decimal
    market_depth_utilized_percent: Decimal

    # Time metrics
    execution_duration_seconds: float
    average_time_between_slices: float

    # Risk metrics
    severity_distribution: dict[ImpactSeverity, int]
    recovery_time_seconds: float | None  # Time for price to recover
    permanent_impact_percent: Decimal | None  # Impact after recovery period

    # Recommendations
    optimal_slice_size: Decimal
    recommended_delay_seconds: float
    max_safe_volume: Decimal

    timestamp: datetime = field(default_factory=datetime.now)


class MarketImpactMonitor:
    """
    Monitors and analyzes market impact of order executions.

    Tracks price movements before and after executions to measure impact,
    providing insights for optimizing future execution strategies.
    """

    # Configuration
    RECOVERY_MONITORING_PERIOD = timedelta(minutes=5)
    IMPACT_MEASUREMENT_DELAY = 1.0  # Seconds to wait after execution

    # Severity thresholds
    SEVERITY_THRESHOLDS = {
        ImpactSeverity.NEGLIGIBLE: Decimal("0.1"),
        ImpactSeverity.LOW: Decimal("0.3"),
        ImpactSeverity.MODERATE: Decimal("0.5"),
        ImpactSeverity.HIGH: Decimal("1.0"),
        ImpactSeverity.SEVERE: Decimal("999"),  # Anything above 1%
    }

    def __init__(
        self,
        gateway,
        repository: Repository | None = None,
        tier: TradingTier = TradingTier.HUNTER,
    ):
        """
        Initialize the market impact monitor.

        Args:
            gateway: Exchange gateway for market data
            repository: Data repository for persistence
            tier: Trading tier for feature gating
        """
        self.gateway = gateway
        self.repository = repository
        self.tier = tier

        # Track active monitoring
        self.active_monitors: dict[str, list[MarketImpactMetric]] = {}
        self.pre_execution_prices: dict[str, tuple[Decimal, Decimal]] = (
            {}
        )  # symbol -> (bid, ask)
        self.execution_start_times: dict[str, datetime] = {}

        logger.info("Market impact monitor initialized", tier=tier.value)

    async def measure_pre_execution_state(
        self, symbol: str, execution_id: str
    ) -> tuple[Decimal, Decimal, OrderBook]:
        """
        Capture market state before execution.

        Args:
            symbol: Trading symbol
            execution_id: Execution to track

        Returns:
            Pre-execution bid price, ask price, and order book
        """
        try:
            # Get current ticker
            ticker = await self.gateway.get_ticker(symbol)

            # Get order book snapshot
            order_book = await self.gateway.get_order_book(symbol, depth=50)

            # Store pre-execution prices
            self.pre_execution_prices[execution_id] = (
                ticker.bid_price,
                ticker.ask_price,
            )
            self.execution_start_times[execution_id] = datetime.now()

            # Initialize monitoring list
            if execution_id not in self.active_monitors:
                self.active_monitors[execution_id] = []

            logger.info(
                "Pre-execution state captured",
                execution_id=execution_id,
                symbol=symbol,
                bid_price=str(ticker.bid_price),
                ask_price=str(ticker.ask_price),
            )

            return ticker.bid_price, ticker.ask_price, order_book

        except Exception as e:
            logger.error(
                "Failed to capture pre-execution state",
                execution_id=execution_id,
                error=str(e),
            )
            raise

    @with_timeout(5000)  # 5 second timeout
    async def measure_slice_impact(
        self, execution_id: str, slice_order: Order, pre_price: Decimal, volume: Decimal
    ) -> MarketImpactMetric:
        """
        Measure impact of a single slice execution.

        Args:
            execution_id: Parent execution ID
            slice_order: The slice order executed
            pre_price: Price before slice execution
            volume: Volume executed in this slice

        Returns:
            Market impact metric for the slice
        """
        try:
            # Wait briefly for market to react
            await asyncio.sleep(self.IMPACT_MEASUREMENT_DELAY)

            # Get post-execution state
            ticker = await self.gateway.get_ticker(slice_order.symbol)
            order_book = await self.gateway.get_order_book(slice_order.symbol, depth=20)

            # Determine relevant post-price based on side
            if slice_order.side == OrderSide.BUY:
                post_price = ticker.ask_price
            else:
                post_price = ticker.bid_price

            # Calculate impact
            impact_percent = self.calculate_impact(pre_price, post_price, volume)

            # Calculate order book metrics
            spread = ticker.ask_price - ticker.bid_price
            depth_1pct = self._calculate_depth_at_level(
                order_book, Decimal("1.0"), slice_order.side
            )
            depth_2pct = self._calculate_depth_at_level(
                order_book, Decimal("2.0"), slice_order.side
            )

            # Determine severity
            severity = self._classify_impact_severity(abs(impact_percent))

            # Create metric
            metric = MarketImpactMetric(
                impact_id=str(uuid4()),
                execution_id=execution_id,
                slice_id=slice_order.order_id,
                symbol=slice_order.symbol,
                side=slice_order.side,
                pre_price=pre_price,
                post_price=post_price,
                price_impact_percent=impact_percent,
                volume_executed=volume,
                order_book_depth_usdt=depth_1pct,
                bid_ask_spread=spread,
                liquidity_consumed_percent=None,  # Calculate if needed
                market_depth_1pct=depth_1pct,
                market_depth_2pct=depth_2pct,
                cumulative_impact=None,  # Will be calculated later
                severity=severity,
                measured_at=datetime.now(),
                notes=f"Slice {slice_order.slice_number}/{slice_order.total_slices}",
            )

            # Store metric
            self.active_monitors[execution_id].append(metric)

            # Save to database
            if self.repository:
                await self._save_impact_metric(metric)

            logger.info(
                "Slice impact measured",
                execution_id=execution_id,
                slice_id=slice_order.order_id,
                impact_percent=str(impact_percent),
                severity=severity.value,
            )

            return metric

        except Exception as e:
            logger.error(
                "Failed to measure slice impact",
                execution_id=execution_id,
                slice_id=slice_order.order_id,
                error=str(e),
            )
            # Return a default metric on error
            return MarketImpactMetric(
                impact_id=str(uuid4()),
                execution_id=execution_id,
                slice_id=slice_order.order_id,
                symbol=slice_order.symbol,
                side=slice_order.side,
                pre_price=pre_price,
                post_price=pre_price,  # Assume no impact on error
                price_impact_percent=Decimal("0"),
                volume_executed=volume,
                order_book_depth_usdt=None,
                bid_ask_spread=None,
                liquidity_consumed_percent=None,
                market_depth_1pct=None,
                market_depth_2pct=None,
                cumulative_impact=None,
                severity=ImpactSeverity.NEGLIGIBLE,
                measured_at=datetime.now(),
                notes="Measurement failed",
            )

    def calculate_impact(
        self, pre_price: Decimal, post_price: Decimal, volume: Decimal
    ) -> Decimal:
        """
        Calculate price impact percentage.

        Args:
            pre_price: Price before execution
            post_price: Price after execution
            volume: Volume executed

        Returns:
            Impact percentage (positive = unfavorable move)
        """
        if pre_price == 0:
            return Decimal("0")

        # Calculate raw price change
        price_change = post_price - pre_price
        impact_percent = (price_change / pre_price) * Decimal("100")

        # Weight by volume (optional enhancement)
        # Could adjust impact based on volume relative to typical market volume

        return impact_percent.quantize(Decimal("0.0001"))

    async def analyze_execution_impact(
        self, execution_id: str, symbol: str, total_volume: Decimal, slice_count: int
    ) -> ImpactAnalysis:
        """
        Perform comprehensive analysis of execution impact.

        Args:
            execution_id: Execution to analyze
            symbol: Trading symbol
            total_volume: Total volume executed
            slice_count: Number of slices

        Returns:
            Comprehensive impact analysis
        """
        try:
            # Get all metrics for this execution
            metrics = self.active_monitors.get(execution_id, [])

            if not metrics:
                logger.warning(
                    "No metrics found for execution", execution_id=execution_id
                )
                return self._create_empty_analysis(
                    execution_id, symbol, total_volume, slice_count
                )

            # Calculate aggregate metrics
            total_impact = sum(m.price_impact_percent for m in metrics)
            avg_impact = total_impact / len(metrics) if metrics else Decimal("0")
            max_impact = (
                max(m.price_impact_percent for m in metrics)
                if metrics
                else Decimal("0")
            )
            min_impact = (
                min(m.price_impact_percent for m in metrics)
                if metrics
                else Decimal("0")
            )

            # Calculate standard deviation
            if len(metrics) > 1:
                mean = avg_impact
                variance = sum(
                    (m.price_impact_percent - mean) ** 2 for m in metrics
                ) / len(metrics)
                std_dev = variance.sqrt()
            else:
                std_dev = Decimal("0")

            # Calculate spread metrics
            avg_spread = (
                sum(m.bid_ask_spread for m in metrics if m.bid_ask_spread)
                / len([m for m in metrics if m.bid_ask_spread])
                if any(m.bid_ask_spread for m in metrics)
                else Decimal("0")
            )

            # Calculate liquidity consumption
            total_liquidity_consumed = sum(m.volume_executed for m in metrics)

            # Calculate time metrics
            if execution_id in self.execution_start_times:
                start_time = self.execution_start_times[execution_id]
                end_time = metrics[-1].measured_at if metrics else datetime.now()
                duration = (end_time - start_time).total_seconds()
                avg_time_between = duration / slice_count if slice_count > 1 else 0
            else:
                duration = 0
                avg_time_between = 0

            # Calculate severity distribution
            severity_dist = {}
            for severity in ImpactSeverity:
                severity_dist[severity] = sum(
                    1 for m in metrics if m.severity == severity
                )

            # Monitor for price recovery
            recovery_time = await self._monitor_price_recovery(execution_id, symbol)

            # Calculate permanent impact
            permanent_impact = await self._calculate_permanent_impact(
                execution_id, symbol
            )

            # Generate recommendations
            optimal_slice_size = self._recommend_optimal_slice_size(
                total_volume, slice_count, avg_impact
            )
            recommended_delay = self._recommend_delay(avg_impact, std_dev)
            max_safe_volume = self._calculate_max_safe_volume(metrics)

            analysis = ImpactAnalysis(
                execution_id=execution_id,
                symbol=symbol,
                total_volume=total_volume,
                slice_count=slice_count,
                total_impact_percent=total_impact,
                average_impact_per_slice=avg_impact,
                max_slice_impact=max_impact,
                min_slice_impact=min_impact,
                impact_std_deviation=std_dev,
                average_spread_during_execution=avg_spread,
                liquidity_consumed_total=total_liquidity_consumed,
                market_depth_utilized_percent=Decimal(
                    "0"
                ),  # Would need order book data
                execution_duration_seconds=duration,
                average_time_between_slices=avg_time_between,
                severity_distribution=severity_dist,
                recovery_time_seconds=recovery_time,
                permanent_impact_percent=permanent_impact,
                optimal_slice_size=optimal_slice_size,
                recommended_delay_seconds=recommended_delay,
                max_safe_volume=max_safe_volume,
            )

            # Save analysis
            if self.repository:
                await self._save_impact_analysis(analysis)

            # Clean up tracking
            if execution_id in self.active_monitors:
                del self.active_monitors[execution_id]
            if execution_id in self.pre_execution_prices:
                del self.pre_execution_prices[execution_id]
            if execution_id in self.execution_start_times:
                del self.execution_start_times[execution_id]

            logger.info(
                "Execution impact analysis complete",
                execution_id=execution_id,
                total_impact=str(total_impact),
                avg_impact=str(avg_impact),
                severity_distribution=severity_dist,
            )

            return analysis

        except Exception as e:
            logger.error(
                "Failed to analyze execution impact",
                execution_id=execution_id,
                error=str(e),
            )
            return self._create_empty_analysis(
                execution_id, symbol, total_volume, slice_count
            )

    def _classify_impact_severity(self, impact_percent: Decimal) -> ImpactSeverity:
        """Classify impact severity based on percentage."""
        impact_abs = abs(impact_percent)

        for severity, threshold in self.SEVERITY_THRESHOLDS.items():
            if impact_abs < threshold:
                return severity

        return ImpactSeverity.SEVERE

    def _calculate_depth_at_level(
        self, order_book: OrderBook, level_percent: Decimal, side: OrderSide
    ) -> Decimal:
        """Calculate order book depth at a price level."""
        if side == OrderSide.BUY:
            # For buys, look at asks
            if not order_book.asks:
                return Decimal("0")

            best_ask = Decimal(str(order_book.asks[0][0]))
            target_price = best_ask * (Decimal("1") + level_percent / Decimal("100"))

            total_volume = Decimal("0")
            for price, quantity in order_book.asks:
                if Decimal(str(price)) <= target_price:
                    total_volume += Decimal(str(price)) * Decimal(str(quantity))
                else:
                    break
        else:
            # For sells, look at bids
            if not order_book.bids:
                return Decimal("0")

            best_bid = Decimal(str(order_book.bids[0][0]))
            target_price = best_bid * (Decimal("1") - level_percent / Decimal("100"))

            total_volume = Decimal("0")
            for price, quantity in order_book.bids:
                if Decimal(str(price)) >= target_price:
                    total_volume += Decimal(str(price)) * Decimal(str(quantity))
                else:
                    break

        return total_volume

    async def _monitor_price_recovery(
        self, execution_id: str, symbol: str
    ) -> float | None:
        """Monitor how long it takes for price to recover."""
        if execution_id not in self.pre_execution_prices:
            return None

        pre_bid, pre_ask = self.pre_execution_prices[execution_id]
        start_time = datetime.now()

        # Monitor for up to 5 minutes
        while datetime.now() - start_time < self.RECOVERY_MONITORING_PERIOD:
            try:
                ticker = await self.gateway.get_ticker(symbol)

                # Check if price has recovered
                if ticker.bid_price >= pre_bid * Decimal(
                    "0.99"
                ) and ticker.ask_price <= pre_ask * Decimal("1.01"):
                    recovery_time = (datetime.now() - start_time).total_seconds()
                    logger.info(
                        "Price recovered",
                        execution_id=execution_id,
                        recovery_time_seconds=recovery_time,
                    )
                    return recovery_time

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error("Error monitoring recovery", error=str(e))
                break

        return None  # No recovery within monitoring period

    async def _calculate_permanent_impact(
        self, execution_id: str, symbol: str
    ) -> Decimal | None:
        """Calculate permanent price impact after recovery period."""
        if execution_id not in self.pre_execution_prices:
            return None

        try:
            # Wait for recovery period
            await asyncio.sleep(300)  # 5 minutes

            pre_bid, pre_ask = self.pre_execution_prices[execution_id]
            ticker = await self.gateway.get_ticker(symbol)

            # Calculate permanent impact
            bid_impact = ((ticker.bid_price - pre_bid) / pre_bid) * Decimal("100")
            ask_impact = ((ticker.ask_price - pre_ask) / pre_ask) * Decimal("100")

            permanent_impact = (bid_impact + ask_impact) / Decimal("2")

            return permanent_impact.quantize(Decimal("0.0001"))

        except Exception as e:
            logger.error("Failed to calculate permanent impact", error=str(e))
            return None

    def _recommend_optimal_slice_size(
        self, total_volume: Decimal, slice_count: int, avg_impact: Decimal
    ) -> Decimal:
        """Recommend optimal slice size based on impact."""
        current_slice_size = (
            total_volume / slice_count if slice_count > 0 else total_volume
        )

        # Adjust based on impact
        if abs(avg_impact) < Decimal("0.1"):
            # Very low impact, can increase slice size
            return current_slice_size * Decimal("1.2")
        elif abs(avg_impact) < Decimal("0.3"):
            # Acceptable impact, maintain size
            return current_slice_size
        elif abs(avg_impact) < Decimal("0.5"):
            # Moderate impact, reduce size slightly
            return current_slice_size * Decimal("0.9")
        else:
            # High impact, reduce size significantly
            return current_slice_size * Decimal("0.7")

    def _recommend_delay(self, avg_impact: Decimal, std_dev: Decimal) -> float:
        """Recommend delay between slices."""
        # Base delay on impact and volatility
        if abs(avg_impact) < Decimal("0.2") and std_dev < Decimal("0.1"):
            return 2.0  # Low impact and consistent, short delay
        elif abs(avg_impact) < Decimal("0.5"):
            return 3.5  # Moderate impact, medium delay
        else:
            return 5.0  # High impact, maximum delay

    def _calculate_max_safe_volume(self, metrics: list[MarketImpactMetric]) -> Decimal:
        """Calculate maximum safe volume based on historical impact."""
        if not metrics:
            return Decimal("0")

        # Find volume that kept impact below 0.3%
        safe_volumes = [
            m.volume_executed
            for m in metrics
            if abs(m.price_impact_percent) < Decimal("0.3")
        ]

        if safe_volumes:
            return max(safe_volumes)
        else:
            # No safe volumes found, use minimum
            return min(m.volume_executed for m in metrics) * Decimal("0.5")

    def _create_empty_analysis(
        self, execution_id: str, symbol: str, total_volume: Decimal, slice_count: int
    ) -> ImpactAnalysis:
        """Create empty analysis when no metrics available."""
        return ImpactAnalysis(
            execution_id=execution_id,
            symbol=symbol,
            total_volume=total_volume,
            slice_count=slice_count,
            total_impact_percent=Decimal("0"),
            average_impact_per_slice=Decimal("0"),
            max_slice_impact=Decimal("0"),
            min_slice_impact=Decimal("0"),
            impact_std_deviation=Decimal("0"),
            average_spread_during_execution=Decimal("0"),
            liquidity_consumed_total=Decimal("0"),
            market_depth_utilized_percent=Decimal("0"),
            execution_duration_seconds=0,
            average_time_between_slices=0,
            severity_distribution=dict.fromkeys(ImpactSeverity, 0),
            recovery_time_seconds=None,
            permanent_impact_percent=None,
            optimal_slice_size=(
                total_volume / slice_count if slice_count > 0 else total_volume
            ),
            recommended_delay_seconds=3.0,
            max_safe_volume=total_volume,
        )

    async def _save_impact_metric(self, metric: MarketImpactMetric) -> None:
        """Save impact metric to database."""
        if self.repository:
            await self.repository.save_market_impact_metric(metric)

    async def _save_impact_analysis(self, analysis: ImpactAnalysis) -> None:
        """Save impact analysis to database."""
        if self.repository:
            await self.repository.save_impact_analysis(analysis)

    async def create_impact_dashboard_widget(self) -> dict[str, Any]:
        """
        Create real-time impact dashboard widget data.

        Returns:
            Widget data for UI display
        """
        active_executions = []

        for execution_id, metrics in self.active_monitors.items():
            if metrics:
                latest_metric = metrics[-1]
                cumulative_impact = sum(m.price_impact_percent for m in metrics)

                active_executions.append(
                    {
                        "execution_id": execution_id,
                        "symbol": latest_metric.symbol,
                        "slices_completed": len(metrics),
                        "cumulative_impact": str(cumulative_impact),
                        "latest_impact": str(latest_metric.price_impact_percent),
                        "severity": latest_metric.severity.value,
                        "timestamp": latest_metric.measured_at.isoformat(),
                    }
                )

        return {
            "type": "market_impact",
            "active_executions": active_executions,
            "timestamp": datetime.now().isoformat(),
        }
