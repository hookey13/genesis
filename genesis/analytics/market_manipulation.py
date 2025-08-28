"""Market Manipulation Detection - Spoofing and Layering."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Set
from collections import deque
from enum import Enum

import structlog

from genesis.engine.event_bus import EventBus
from genesis.core.events import Event
from genesis.exchange.order_book_manager import OrderBookSnapshot

logger = structlog.get_logger(__name__)


class ManipulationType(Enum):
    """Types of market manipulation."""

    SPOOFING = "spoofing"
    LAYERING = "layering"
    QUOTE_STUFFING = "quote_stuffing"
    MOMENTUM_IGNITION = "momentum_ignition"
    WASH_TRADING = "wash_trading"


@dataclass
class OrderActivity:
    """Order placement and cancellation activity."""

    order_id: str
    symbol: str
    timestamp: datetime
    price: Decimal
    quantity: Decimal
    side: str  # 'bid' or 'ask'
    action: str  # 'place' or 'cancel'
    levels_from_best: int  # Distance from best bid/ask

    @property
    def notional(self) -> Decimal:
        """Calculate notional value."""
        return self.price * self.quantity


@dataclass
class ManipulationPattern:
    """Detected manipulation pattern."""

    pattern_id: str
    symbol: str
    manipulation_type: ManipulationType
    start_time: datetime
    end_time: Optional[datetime]
    orders: List[OrderActivity] = field(default_factory=list)
    confidence: Decimal = Decimal("0.5")
    severity: str = "medium"  # 'low', 'medium', 'high'

    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate pattern duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def total_volume(self) -> Decimal:
        """Calculate total volume involved."""
        return sum(o.quantity for o in self.orders if o.action == "place")

    @property
    def cancellation_rate(self) -> Decimal:
        """Calculate order cancellation rate."""
        placements = sum(1 for o in self.orders if o.action == "place")
        cancellations = sum(1 for o in self.orders if o.action == "cancel")

        if placements > 0:
            return Decimal(str(cancellations)) / Decimal(str(placements))
        return Decimal("0")

    def is_significant(self) -> bool:
        """Check if pattern is significant."""
        return self.confidence >= Decimal("0.7") and self.severity in ["medium", "high"]


class MarketManipulationDetector:
    """Detects market manipulation patterns like spoofing and layering."""

    def __init__(
        self,
        event_bus: EventBus,
        cancellation_threshold: Decimal = Decimal("0.8"),
        time_window_seconds: int = 30,
        min_orders_for_pattern: int = 5,
    ):
        """Initialize manipulation detector.

        Args:
            event_bus: Event bus for publishing signals
            cancellation_threshold: Min cancellation rate for spoofing
            time_window_seconds: Time window for pattern detection
            min_orders_for_pattern: Minimum orders to form a pattern
        """
        self.event_bus = event_bus
        self.cancellation_threshold = cancellation_threshold
        self.time_window = timedelta(seconds=time_window_seconds)
        self.min_orders = min_orders_for_pattern

        # Order activity tracking
        self.order_history: Dict[str, deque] = {}
        self.active_orders: Dict[str, Dict[str, OrderActivity]] = {}

        # Detected patterns
        self.active_patterns: Dict[str, ManipulationPattern] = {}
        self.pattern_history: Dict[str, deque] = {}

        # Order book snapshots for context
        self.last_snapshots: Dict[str, OrderBookSnapshot] = {}

        # Statistics
        self.order_to_trade_ratios: Dict[str, deque] = {}

    async def track_order_placement(
        self,
        symbol: str,
        order_id: str,
        price: Decimal,
        quantity: Decimal,
        side: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Track order placement for manipulation detection.

        Args:
            symbol: Trading symbol
            order_id: Unique order ID
            price: Order price
            quantity: Order quantity
            side: Order side ('bid' or 'ask')
            timestamp: Order timestamp
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Calculate distance from best price
        levels_from_best = self._calculate_levels_from_best(symbol, price, side)

        # Create order activity record
        activity = OrderActivity(
            order_id=order_id,
            symbol=symbol,
            timestamp=timestamp,
            price=price,
            quantity=quantity,
            side=side,
            action="place",
            levels_from_best=levels_from_best,
        )

        # Store in active orders
        if symbol not in self.active_orders:
            self.active_orders[symbol] = {}
        self.active_orders[symbol][order_id] = activity

        # Add to history
        if symbol not in self.order_history:
            self.order_history[symbol] = deque(maxlen=1000)
        self.order_history[symbol].append(activity)

        # Check for layering pattern
        await self._detect_layering(symbol)

    async def track_order_cancellation(
        self, symbol: str, order_id: str, timestamp: Optional[datetime] = None
    ) -> None:
        """Track order cancellation for manipulation detection.

        Args:
            symbol: Trading symbol
            order_id: Order ID
            timestamp: Cancellation timestamp
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Find original order
        if (
            symbol not in self.active_orders
            or order_id not in self.active_orders[symbol]
        ):
            return

        original_order = self.active_orders[symbol][order_id]

        # Create cancellation record
        cancel_activity = OrderActivity(
            order_id=order_id,
            symbol=symbol,
            timestamp=timestamp,
            price=original_order.price,
            quantity=original_order.quantity,
            side=original_order.side,
            action="cancel",
            levels_from_best=original_order.levels_from_best,
        )

        # Add to history
        self.order_history[symbol].append(cancel_activity)

        # Remove from active orders
        del self.active_orders[symbol][order_id]

        # Check for spoofing pattern
        await self._detect_spoofing(symbol, original_order, cancel_activity)

    async def track_trade_execution(
        self,
        symbol: str,
        order_id: Optional[str],
        price: Decimal,
        quantity: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Track trade execution for ratio calculation.

        Args:
            symbol: Trading symbol
            order_id: Order ID if available
            price: Trade price
            quantity: Trade quantity
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Remove from active orders if matched
        if (
            order_id
            and symbol in self.active_orders
            and order_id in self.active_orders[symbol]
        ):
            del self.active_orders[symbol][order_id]

        # Update order-to-trade ratio
        await self._update_order_to_trade_ratio(symbol)

    def update_order_book(self, snapshot: OrderBookSnapshot) -> None:
        """Update order book snapshot for context.

        Args:
            snapshot: Order book snapshot
        """
        self.last_snapshots[snapshot.symbol] = snapshot

    def _calculate_levels_from_best(
        self, symbol: str, price: Decimal, side: str
    ) -> int:
        """Calculate how many levels away from best bid/ask.

        Args:
            symbol: Trading symbol
            price: Order price
            side: Order side

        Returns:
            Number of levels from best price
        """
        if symbol not in self.last_snapshots:
            return 0

        snapshot = self.last_snapshots[symbol]

        if side == "bid":
            if not snapshot.bids:
                return 0
            best = snapshot.best_bid
            if price >= best:
                return 0
            # Count levels
            for i, level in enumerate(snapshot.bids):
                if price >= level.price:
                    return i
            return len(snapshot.bids)
        else:  # ask
            if not snapshot.asks:
                return 0
            best = snapshot.best_ask
            if price <= best:
                return 0
            # Count levels
            for i, level in enumerate(snapshot.asks):
                if price <= level.price:
                    return i
            return len(snapshot.asks)

    async def _detect_spoofing(
        self, symbol: str, original_order: OrderActivity, cancel_activity: OrderActivity
    ) -> None:
        """Detect spoofing pattern.

        Args:
            symbol: Trading symbol
            original_order: Original order placement
            cancel_activity: Cancellation activity
        """
        # Calculate time between placement and cancellation
        duration = cancel_activity.timestamp - original_order.timestamp

        # Quick cancellation is suspicious
        if duration < timedelta(seconds=5):
            # Check recent cancellation rate
            recent_orders = self._get_recent_orders(symbol, timedelta(seconds=30))
            cancellation_rate = self._calculate_cancellation_rate(recent_orders)

            if cancellation_rate >= self.cancellation_threshold:
                # Check if orders were away from best price
                avg_levels = sum(o.levels_from_best for o in recent_orders) / len(
                    recent_orders
                )

                confidence = self._calculate_spoofing_confidence(
                    cancellation_rate, duration, avg_levels
                )

                if confidence >= Decimal("0.6"):
                    pattern = ManipulationPattern(
                        pattern_id=f"spoof_{symbol}_{datetime.now(timezone.utc).timestamp()}",
                        symbol=symbol,
                        manipulation_type=ManipulationType.SPOOFING,
                        start_time=original_order.timestamp,
                        end_time=cancel_activity.timestamp,
                        orders=recent_orders,
                        confidence=confidence,
                        severity=self._determine_severity(
                            confidence, original_order.notional
                        ),
                    )

                    await self._publish_manipulation_signal(pattern)

    async def _detect_layering(self, symbol: str) -> None:
        """Detect layering pattern.

        Args:
            symbol: Trading symbol
        """
        if symbol not in self.active_orders:
            return

        active = list(self.active_orders[symbol].values())

        # Group orders by side
        bid_orders = [o for o in active if o.side == "bid"]
        ask_orders = [o for o in active if o.side == "ask"]

        # Check for multiple orders at different levels
        for side_orders in [bid_orders, ask_orders]:
            if len(side_orders) >= self.min_orders:
                # Check if orders are at different price levels
                unique_prices = set(o.price for o in side_orders)

                if len(unique_prices) >= 3:  # Multiple price levels
                    # Check if placed within time window
                    time_span = max(o.timestamp for o in side_orders) - min(
                        o.timestamp for o in side_orders
                    )

                    if time_span <= self.time_window:
                        confidence = self._calculate_layering_confidence(side_orders)

                        if confidence >= Decimal("0.6"):
                            pattern = ManipulationPattern(
                                pattern_id=f"layer_{symbol}_{datetime.now(timezone.utc).timestamp()}",
                                symbol=symbol,
                                manipulation_type=ManipulationType.LAYERING,
                                start_time=min(o.timestamp for o in side_orders),
                                orders=side_orders,
                                confidence=confidence,
                                severity=self._determine_severity(
                                    confidence, sum(o.notional for o in side_orders)
                                ),
                            )

                            await self._publish_manipulation_signal(pattern)

    async def _detect_quote_stuffing(self, symbol: str) -> None:
        """Detect quote stuffing pattern.

        Args:
            symbol: Trading symbol
        """
        recent_orders = self._get_recent_orders(symbol, timedelta(seconds=1))

        # High frequency of order updates
        if len(recent_orders) >= 50:  # 50+ orders per second
            cancellation_rate = self._calculate_cancellation_rate(recent_orders)

            if cancellation_rate >= Decimal("0.9"):
                pattern = ManipulationPattern(
                    pattern_id=f"stuff_{symbol}_{datetime.now(timezone.utc).timestamp()}",
                    symbol=symbol,
                    manipulation_type=ManipulationType.QUOTE_STUFFING,
                    start_time=recent_orders[0].timestamp,
                    end_time=recent_orders[-1].timestamp,
                    orders=recent_orders,
                    confidence=Decimal("0.9"),
                    severity="high",
                )

                await self._publish_manipulation_signal(pattern)

    def _get_recent_orders(
        self, symbol: str, time_window: timedelta
    ) -> List[OrderActivity]:
        """Get recent orders within time window.

        Args:
            symbol: Trading symbol
            time_window: Time window

        Returns:
            List of recent orders
        """
        if symbol not in self.order_history:
            return []

        cutoff = datetime.now(timezone.utc) - time_window
        return [o for o in self.order_history[symbol] if o.timestamp >= cutoff]

    def _calculate_cancellation_rate(self, orders: List[OrderActivity]) -> Decimal:
        """Calculate cancellation rate for orders.

        Args:
            orders: List of orders

        Returns:
            Cancellation rate (0-1)
        """
        if not orders:
            return Decimal("0")

        placements = sum(1 for o in orders if o.action == "place")
        cancellations = sum(1 for o in orders if o.action == "cancel")

        if placements > 0:
            return Decimal(str(cancellations)) / Decimal(str(placements))
        return Decimal("0")

    def _calculate_spoofing_confidence(
        self, cancellation_rate: Decimal, duration: timedelta, avg_levels: float
    ) -> Decimal:
        """Calculate confidence in spoofing detection.

        Args:
            cancellation_rate: Order cancellation rate
            duration: Time between placement and cancellation
            avg_levels: Average levels from best price

        Returns:
            Confidence score (0-1)
        """
        confidence = Decimal("0")

        # High cancellation rate
        if cancellation_rate >= Decimal("0.9"):
            confidence += Decimal("0.4")
        elif cancellation_rate >= Decimal("0.8"):
            confidence += Decimal("0.3")

        # Quick cancellation
        if duration.total_seconds() < 2:
            confidence += Decimal("0.3")
        elif duration.total_seconds() < 5:
            confidence += Decimal("0.2")

        # Away from best price
        if avg_levels >= 3:
            confidence += Decimal("0.3")
        elif avg_levels >= 2:
            confidence += Decimal("0.2")

        return min(Decimal("1"), confidence)

    def _calculate_layering_confidence(self, orders: List[OrderActivity]) -> Decimal:
        """Calculate confidence in layering detection.

        Args:
            orders: List of orders at different levels

        Returns:
            Confidence score (0-1)
        """
        confidence = Decimal("0.5")  # Base confidence

        # Multiple price levels
        unique_prices = len(set(o.price for o in orders))
        if unique_prices >= 5:
            confidence += Decimal("0.3")
        elif unique_prices >= 3:
            confidence += Decimal("0.2")

        # Similar quantities (coordinated)
        quantities = [o.quantity for o in orders]
        avg_qty = sum(quantities) / len(quantities)
        variance = sum((q - avg_qty) ** 2 for q in quantities) / len(quantities)

        if variance < (avg_qty * Decimal("0.1")) ** 2:
            confidence += Decimal("0.2")

        return min(Decimal("1"), confidence)

    def _determine_severity(self, confidence: Decimal, notional: Decimal) -> str:
        """Determine manipulation severity.

        Args:
            confidence: Detection confidence
            notional: Total notional value

        Returns:
            Severity level
        """
        if confidence >= Decimal("0.9") or notional >= Decimal("1000000"):
            return "high"
        elif confidence >= Decimal("0.7") or notional >= Decimal("100000"):
            return "medium"
        else:
            return "low"

    async def _update_order_to_trade_ratio(self, symbol: str) -> None:
        """Update order-to-trade ratio statistics.

        Args:
            symbol: Trading symbol
        """
        recent_orders = self._get_recent_orders(symbol, timedelta(minutes=1))

        if symbol not in self.order_to_trade_ratios:
            self.order_to_trade_ratios[symbol] = deque(maxlen=100)

        # Simple ratio calculation (can be enhanced with actual trade data)
        placements = sum(1 for o in recent_orders if o.action == "place")
        # Assume trades = placements - cancellations
        trades = placements - sum(1 for o in recent_orders if o.action == "cancel")

        if trades > 0:
            ratio = Decimal(str(placements)) / Decimal(str(trades))
            self.order_to_trade_ratios[symbol].append(ratio)

            # High ratio might indicate manipulation
            if ratio > Decimal("10"):
                await self._detect_quote_stuffing(symbol)

    async def _publish_manipulation_signal(self, pattern: ManipulationPattern) -> None:
        """Publish manipulation detection signal.

        Args:
            pattern: Detected manipulation pattern
        """
        if pattern.is_significant():
            await self.event_bus.publish(
                Event(
                    type="market_manipulation_detected",
                    data={
                        "pattern_id": pattern.pattern_id,
                        "symbol": pattern.symbol,
                        "manipulation_type": pattern.manipulation_type.value,
                        "confidence": float(pattern.confidence),
                        "severity": pattern.severity,
                        "cancellation_rate": float(pattern.cancellation_rate),
                        "total_volume": float(pattern.total_volume),
                        "start_time": pattern.start_time.isoformat(),
                        "end_time": (
                            pattern.end_time.isoformat() if pattern.end_time else None
                        ),
                    },
                )
            )

            logger.warning(
                "manipulation_detected",
                symbol=pattern.symbol,
                type=pattern.manipulation_type.value,
                confidence=float(pattern.confidence),
                severity=pattern.severity,
            )

    def get_manipulation_statistics(self, symbol: str) -> Dict[str, any]:
        """Get manipulation statistics for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Statistics dictionary
        """
        recent_orders = self._get_recent_orders(symbol, timedelta(minutes=5))

        return {
            "cancellation_rate": float(
                self._calculate_cancellation_rate(recent_orders)
            ),
            "active_orders": len(self.active_orders.get(symbol, {})),
            "recent_order_count": len(recent_orders),
            "avg_order_to_trade_ratio": float(
                sum(self.order_to_trade_ratios.get(symbol, []))
                / len(self.order_to_trade_ratios.get(symbol, [1]))
            ),
            "active_patterns": len(
                [p for p in self.active_patterns.values() if p.symbol == symbol]
            ),
        }
