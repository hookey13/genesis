"""
Execution quality tracking and reporting for Project GENESIS.

This module tracks and analyzes order execution quality, including slippage,
fees, time to fill, and price improvement metrics.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum

import structlog

from genesis.core.models import Order, OrderSide, OrderType
from genesis.data.sqlite_repo import SQLiteRepository
from typing import Optional

logger = structlog.get_logger(__name__)


class ExecutionMetric(str, Enum):
    """Execution quality metrics."""
    SLIPPAGE = "SLIPPAGE"
    FEES = "FEES"
    TIME_TO_FILL = "TIME_TO_FILL"
    PRICE_IMPROVEMENT = "PRICE_IMPROVEMENT"
    REJECTION_RATE = "REJECTION_RATE"
    FILL_RATE = "FILL_RATE"


@dataclass
class ExecutionQuality:
    """Execution quality data for an order."""
    order_id: str
    symbol: str
    order_type: str
    routing_method: Optional[str]
    timestamp: datetime
    slippage_bps: Decimal  # Basis points
    total_fees: Decimal
    maker_fees: Decimal
    taker_fees: Decimal
    time_to_fill_ms: int
    fill_rate: Decimal  # Percentage filled
    price_improvement_bps: Decimal  # Positive = favorable
    execution_score: float
    market_conditions: Optional[str]  # JSON string of conditions


@dataclass
class ExecutionStats:
    """Aggregated execution statistics."""
    period: str  # e.g., "1h", "24h", "7d"
    total_orders: int
    avg_slippage_bps: Decimal
    total_fees: Decimal
    avg_maker_fees: Decimal
    avg_taker_fees: Decimal
    avg_time_to_fill_ms: int
    avg_fill_rate: Decimal
    price_improvement_rate: Decimal  # % of orders with positive improvement
    avg_execution_score: float
    best_execution_score: float
    worst_execution_score: float
    rejection_rate: Decimal
    orders_by_type: dict[str, int]
    orders_by_routing: dict[str, int]


class ExecutionScorer:
    """
    Scores execution quality based on configurable metrics.
    
    The scoring algorithm weights different factors to produce
    a quality score from 0-100.
    """

    # Weight configuration for scoring
    WEIGHTS = {
        ExecutionMetric.SLIPPAGE: 0.35,
        ExecutionMetric.FEES: 0.25,
        ExecutionMetric.TIME_TO_FILL: 0.20,
        ExecutionMetric.PRICE_IMPROVEMENT: 0.15,
        ExecutionMetric.FILL_RATE: 0.05
    }

    # Threshold values for scoring
    THRESHOLDS = {
        "slippage_excellent_bps": 5,  # < 5 bps is excellent
        "slippage_good_bps": 20,  # < 20 bps is good
        "slippage_poor_bps": 50,  # > 50 bps is poor
        "time_excellent_ms": 100,  # < 100ms is excellent
        "time_good_ms": 500,  # < 500ms is good
        "time_poor_ms": 2000,  # > 2000ms is poor
        "fee_rate_maker": 0.001,  # 0.1% maker fee
        "fee_rate_taker": 0.001,  # 0.1% taker fee
    }

    def __init__(self):
        """Initialize the execution scorer."""
        logger.info("Initialized ExecutionScorer")

    def calculate_score(
        self,
        order: Order,
        actual_price: Decimal,
        time_to_fill_ms: int,
        market_mid_price: Decimal
    ) -> tuple[float, ExecutionQuality]:
        """
        Calculate execution quality score for an order.
        
        Args:
            order: Completed order
            actual_price: Actual execution price
            time_to_fill_ms: Time from submission to fill
            market_mid_price: Market mid price at execution
            
        Returns:
            Tuple of (score, quality_details)
        """
        scores = {}

        # Calculate slippage score
        slippage_bps = self._calculate_slippage_bps(
            order.price or market_mid_price,
            actual_price,
            order.side
        )
        scores[ExecutionMetric.SLIPPAGE] = self._score_slippage(slippage_bps)

        # Calculate fee score
        total_fees = (order.maker_fee_paid or Decimal("0")) + (order.taker_fee_paid or Decimal("0"))
        fee_bps = (total_fees / (order.quantity * actual_price)) * Decimal("10000")
        scores[ExecutionMetric.FEES] = self._score_fees(fee_bps)

        # Calculate time to fill score
        scores[ExecutionMetric.TIME_TO_FILL] = self._score_time_to_fill(time_to_fill_ms)

        # Calculate price improvement
        price_improvement_bps = self._calculate_price_improvement_bps(
            order.price or market_mid_price,
            actual_price,
            market_mid_price,
            order.side
        )
        scores[ExecutionMetric.PRICE_IMPROVEMENT] = self._score_price_improvement(price_improvement_bps)

        # Calculate fill rate score
        fill_rate = (order.filled_quantity / order.quantity) * Decimal("100")
        scores[ExecutionMetric.FILL_RATE] = self._score_fill_rate(fill_rate)

        # Calculate weighted total score
        total_score = sum(
            scores[metric] * self.WEIGHTS[metric]
            for metric in scores
        )

        # Create quality record
        quality = ExecutionQuality(
            order_id=order.order_id,
            symbol=order.symbol,
            order_type=order.type.value if isinstance(order.type, OrderType) else order.type,
            routing_method=order.routing_method,
            timestamp=order.executed_at or datetime.now(UTC),
            slippage_bps=slippage_bps,
            total_fees=total_fees,
            maker_fees=order.maker_fee_paid or Decimal("0"),
            taker_fees=order.taker_fee_paid or Decimal("0"),
            time_to_fill_ms=time_to_fill_ms,
            fill_rate=fill_rate,
            price_improvement_bps=price_improvement_bps,
            execution_score=total_score,
            market_conditions=None  # Can be populated with JSON conditions
        )

        logger.info(
            "Calculated execution score",
            order_id=order.order_id,
            score=total_score,
            slippage_bps=float(slippage_bps),
            fee_bps=float(fee_bps),
            time_ms=time_to_fill_ms
        )

        return total_score, quality

    def _calculate_slippage_bps(
        self,
        expected_price: Decimal,
        actual_price: Decimal,
        side: OrderSide
    ) -> Decimal:
        """
        Calculate slippage in basis points.
        
        Args:
            expected_price: Expected execution price
            actual_price: Actual execution price
            side: Order side (BUY/SELL)
            
        Returns:
            Slippage in basis points (positive = unfavorable)
        """
        if expected_price == 0:
            return Decimal("0")

        if side == OrderSide.BUY:
            # For buys, higher actual price is unfavorable
            slippage = ((actual_price - expected_price) / expected_price)
        else:
            # For sells, lower actual price is unfavorable
            slippage = ((expected_price - actual_price) / expected_price)

        return slippage * Decimal("10000")  # Convert to basis points

    def _calculate_price_improvement_bps(
        self,
        expected_price: Decimal,
        actual_price: Decimal,
        market_mid_price: Decimal,
        side: OrderSide
    ) -> Decimal:
        """
        Calculate price improvement relative to market.
        
        Args:
            expected_price: Expected execution price
            actual_price: Actual execution price
            market_mid_price: Market mid price at execution
            side: Order side
            
        Returns:
            Price improvement in basis points (positive = favorable)
        """
        if market_mid_price == 0:
            return Decimal("0")

        if side == OrderSide.BUY:
            # For buys, lower actual than mid is favorable
            improvement = ((market_mid_price - actual_price) / market_mid_price)
        else:
            # For sells, higher actual than mid is favorable
            improvement = ((actual_price - market_mid_price) / market_mid_price)

        return improvement * Decimal("10000")

    def _score_slippage(self, slippage_bps: Decimal) -> float:
        """Score slippage (0-100)."""
        slippage = abs(float(slippage_bps))

        if slippage <= self.THRESHOLDS["slippage_excellent_bps"]:
            return 100.0
        elif slippage <= self.THRESHOLDS["slippage_good_bps"]:
            return 80.0 - (slippage - 5) * 2
        elif slippage <= self.THRESHOLDS["slippage_poor_bps"]:
            return 50.0 - (slippage - 20) * 1
        else:
            return max(0.0, 20.0 - (slippage - 50) * 0.5)

    def _score_fees(self, fee_bps: Decimal) -> float:
        """Score fees (0-100)."""
        fees = float(fee_bps)

        # Assume 10 bps is standard taker fee
        if fees <= 5:  # Maker fee level
            return 100.0
        elif fees <= 10:  # Standard taker fee
            return 80.0
        elif fees <= 15:
            return 60.0
        else:
            return max(0.0, 40.0 - (fees - 15) * 2)

    def _score_time_to_fill(self, time_ms: int) -> float:
        """Score time to fill (0-100)."""
        if time_ms <= self.THRESHOLDS["time_excellent_ms"]:
            return 100.0
        elif time_ms <= self.THRESHOLDS["time_good_ms"]:
            return 90.0 - (time_ms - 100) * 0.02
        elif time_ms <= self.THRESHOLDS["time_poor_ms"]:
            return 70.0 - (time_ms - 500) * 0.03
        else:
            return max(0.0, 40.0 - (time_ms - 2000) * 0.01)

    def _score_price_improvement(self, improvement_bps: Decimal) -> float:
        """Score price improvement (0-100)."""
        imp = float(improvement_bps)

        if imp > 10:  # Excellent improvement
            return 100.0
        elif imp > 0:  # Some improvement
            return 70.0 + imp * 3
        elif imp > -10:  # Slight negative
            return 50.0 + imp * 2
        else:  # Poor
            return max(0.0, 30.0 + imp * 0.5)

    def _score_fill_rate(self, fill_rate: Decimal) -> float:
        """Score fill rate (0-100)."""
        rate = float(fill_rate)

        if rate >= 100:
            return 100.0
        elif rate >= 95:
            return 90.0
        elif rate >= 90:
            return 80.0 - (95 - rate) * 2
        else:
            return max(0.0, 50.0 - (90 - rate) * 1)


class ExecutionQualityTracker:
    """
    Tracks and reports on execution quality metrics over time.
    """

    def __init__(self, repository: SQLiteRepository):
        """
        Initialize the tracker.
        
        Args:
            repository: Database repository for persistence
        """
        self.repository = repository
        self.scorer = ExecutionScorer()
        self._quality_cache: list[ExecutionQuality] = []
        logger.info("Initialized ExecutionQualityTracker")

    async def track_execution(
        self,
        order: Order,
        actual_price: Decimal,
        time_to_fill_ms: int,
        market_mid_price: Decimal
    ) -> float:
        """
        Track an order execution and return quality score.
        
        Args:
            order: Completed order
            actual_price: Actual execution price
            time_to_fill_ms: Time from submission to fill
            market_mid_price: Market mid price at execution
            
        Returns:
            Execution quality score
        """
        score, quality = self.scorer.calculate_score(
            order, actual_price, time_to_fill_ms, market_mid_price
        )

        # Cache quality record
        self._quality_cache.append(quality)

        # Persist to database
        await self._persist_quality(quality)

        # Update order with score
        order.execution_score = score

        return score

    async def get_statistics(
        self,
        period: str = "24h",
        symbol: Optional[str] = None
    ) -> ExecutionStats:
        """
        Get aggregated execution statistics.
        
        Args:
            period: Time period (1h, 24h, 7d)
            symbol: Optional symbol filter
            
        Returns:
            Aggregated statistics
        """
        # Calculate time range
        now = datetime.now(UTC)
        if period == "1h":
            start_time = now - timedelta(hours=1)
        elif period == "24h":
            start_time = now - timedelta(days=1)
        elif period == "7d":
            start_time = now - timedelta(days=7)
        else:
            start_time = now - timedelta(days=1)

        # Get quality records from cache and database
        records = await self._get_quality_records(start_time, symbol)

        if not records:
            # Return empty stats
            return ExecutionStats(
                period=period,
                total_orders=0,
                avg_slippage_bps=Decimal("0"),
                total_fees=Decimal("0"),
                avg_maker_fees=Decimal("0"),
                avg_taker_fees=Decimal("0"),
                avg_time_to_fill_ms=0,
                avg_fill_rate=Decimal("0"),
                price_improvement_rate=Decimal("0"),
                avg_execution_score=0.0,
                best_execution_score=0.0,
                worst_execution_score=0.0,
                rejection_rate=Decimal("0"),
                orders_by_type={},
                orders_by_routing={}
            )

        # Calculate aggregates
        total_orders = len(records)
        avg_slippage = sum(r.slippage_bps for r in records) / total_orders
        total_fees = sum(r.total_fees for r in records)
        avg_maker = sum(r.maker_fees for r in records) / total_orders
        avg_taker = sum(r.taker_fees for r in records) / total_orders
        avg_time = sum(r.time_to_fill_ms for r in records) / total_orders
        avg_fill = sum(r.fill_rate for r in records) / total_orders

        improvements = sum(1 for r in records if r.price_improvement_bps > 0)
        improvement_rate = (Decimal(improvements) / Decimal(total_orders)) * Decimal("100")

        scores = [r.execution_score for r in records]
        avg_score = sum(scores) / len(scores)
        best_score = max(scores)
        worst_score = min(scores)

        # Count by type and routing
        orders_by_type: dict[str, int] = {}
        orders_by_routing: dict[str, int] = {}

        for record in records:
            orders_by_type[record.order_type] = orders_by_type.get(record.order_type, 0) + 1
            if record.routing_method:
                orders_by_routing[record.routing_method] = orders_by_routing.get(record.routing_method, 0) + 1

        return ExecutionStats(
            period=period,
            total_orders=total_orders,
            avg_slippage_bps=avg_slippage,
            total_fees=total_fees,
            avg_maker_fees=avg_maker,
            avg_taker_fees=avg_taker,
            avg_time_to_fill_ms=int(avg_time),
            avg_fill_rate=avg_fill,
            price_improvement_rate=improvement_rate,
            avg_execution_score=avg_score,
            best_execution_score=best_score,
            worst_execution_score=worst_score,
            rejection_rate=Decimal("0"),  # TODO: Track rejections
            orders_by_type=orders_by_type,
            orders_by_routing=orders_by_routing
        )

    async def _persist_quality(self, quality: ExecutionQuality) -> None:
        """Persist quality record to database."""
        try:
            # Store execution quality data in the database
            await self.repository.save_execution_quality(quality)
            logger.debug(
                "Persisted execution quality",
                order_id=quality.order_id,
                score=quality.execution_score
            )
        except Exception as e:
            logger.error(
                "Failed to persist execution quality",
                order_id=quality.order_id,
                error=str(e)
            )

    async def _get_quality_records(
        self,
        start_time: datetime,
        symbol: Optional[str] = None
    ) -> list[ExecutionQuality]:
        """Get quality records from cache and database."""
        # Filter cache records
        records = [
            r for r in self._quality_cache
            if r.timestamp >= start_time and
            (symbol is None or r.symbol == symbol)
        ]

        # Also fetch from database for complete history
        try:
            db_records = await self.repository.get_execution_quality_records(
                start_time=start_time,
                symbol=symbol
            )
            # Combine with cache, avoiding duplicates
            seen_ids = {r.order_id for r in records}
            for db_record in db_records:
                if db_record.order_id not in seen_ids:
                    records.append(db_record)
        except Exception as e:
            logger.warning(
                "Failed to fetch execution quality from database",
                error=str(e)
            )

        return records

    def generate_report(self, stats: ExecutionStats) -> str:
        """
        Generate a human-readable execution quality report.
        
        Args:
            stats: Execution statistics
            
        Returns:
            Formatted report string
        """
        report = f"""
Execution Quality Report - Period: {stats.period}
{'=' * 50}

Summary:
- Total Orders: {stats.total_orders}
- Average Execution Score: {stats.avg_execution_score:.1f}/100
- Best Score: {stats.best_execution_score:.1f}
- Worst Score: {stats.worst_execution_score:.1f}

Slippage & Price Improvement:
- Average Slippage: {float(stats.avg_slippage_bps):.1f} bps
- Price Improvement Rate: {float(stats.price_improvement_rate):.1f}%

Fees:
- Total Fees Paid: {float(stats.total_fees):.4f}
- Average Maker Fees: {float(stats.avg_maker_fees):.4f}
- Average Taker Fees: {float(stats.avg_taker_fees):.4f}

Performance:
- Average Time to Fill: {stats.avg_time_to_fill_ms} ms
- Average Fill Rate: {float(stats.avg_fill_rate):.1f}%
- Rejection Rate: {float(stats.rejection_rate):.1f}%

Order Distribution:
"""

        # Add order type distribution
        if stats.orders_by_type:
            report += "\nBy Order Type:\n"
            for order_type, count in sorted(stats.orders_by_type.items()):
                percentage = (count / stats.total_orders) * 100
                report += f"  - {order_type}: {count} ({percentage:.1f}%)\n"

        # Add routing method distribution
        if stats.orders_by_routing:
            report += "\nBy Routing Method:\n"
            for method, count in sorted(stats.orders_by_routing.items()):
                percentage = (count / stats.total_orders) * 100
                report += f"  - {method}: {count} ({percentage:.1f}%)\n"

        return report
