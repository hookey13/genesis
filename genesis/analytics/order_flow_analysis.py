"""Order Flow Imbalance Detection and Analysis."""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import structlog

from genesis.core.events import Event
from genesis.engine.event_bus import EventBus
from genesis.exchange.order_book_manager import OrderBookSnapshot

logger = structlog.get_logger(__name__)


@dataclass
class OrderFlowMetrics:
    """Metrics for order flow analysis."""

    symbol: str
    timestamp: datetime
    ofi: Decimal  # Order Flow Imbalance
    volume_ratio: Decimal  # Buy volume / Sell volume
    aggression_ratio: Decimal  # Aggressive orders / Passive orders
    net_flow: Decimal  # Buy flow - Sell flow
    flow_velocity: Decimal  # Rate of change in flow
    pressure_score: Decimal  # Overall buy/sell pressure (-100 to 100)
    confidence: Decimal  # Confidence in the metrics (0-1)

    def is_bullish(self) -> bool:
        """Check if flow indicates bullish pressure."""
        return self.pressure_score > Decimal("20")

    def is_bearish(self) -> bool:
        """Check if flow indicates bearish pressure."""
        return self.pressure_score < Decimal("-20")

    def is_significant(self) -> bool:
        """Check if the flow imbalance is significant."""
        return abs(self.ofi) > Decimal("0.3") and self.confidence > Decimal("0.7")


@dataclass
class TradeFlow:
    """Individual trade flow data."""

    timestamp: datetime
    price: Decimal
    quantity: Decimal
    side: str  # 'buy' or 'sell'
    is_aggressive: bool  # True if market order

    @property
    def notional(self) -> Decimal:
        """Calculate notional value."""
        return self.price * self.quantity


@dataclass
class FlowWindow:
    """Sliding window for flow analysis."""

    window_size: timedelta
    trades: deque = field(default_factory=deque)

    def add_trade(self, trade: TradeFlow) -> None:
        """Add trade to window and remove expired trades."""
        self.trades.append(trade)
        cutoff_time = datetime.now(UTC) - self.window_size

        # Remove old trades
        while self.trades and self.trades[0].timestamp < cutoff_time:
            self.trades.popleft()

    def get_buy_volume(self) -> Decimal:
        """Calculate total buy volume in window."""
        return sum(t.quantity for t in self.trades if t.side == "buy")

    def get_sell_volume(self) -> Decimal:
        """Calculate total sell volume in window."""
        return sum(t.quantity for t in self.trades if t.side == "sell")

    def get_aggressive_volume(self) -> Decimal:
        """Calculate aggressive order volume."""
        return sum(t.quantity for t in self.trades if t.is_aggressive)

    def get_passive_volume(self) -> Decimal:
        """Calculate passive order volume."""
        return sum(t.quantity for t in self.trades if not t.is_aggressive)


class OrderFlowAnalyzer:
    """Analyzes order flow imbalance and detects trading pressure."""

    def __init__(
        self,
        event_bus: EventBus,
        window_minutes: int = 5,
        sensitivity: Decimal = Decimal("0.3"),
    ):
        """Initialize order flow analyzer.

        Args:
            event_bus: Event bus for publishing signals
            window_minutes: Analysis window in minutes
            sensitivity: Sensitivity threshold for imbalance detection
        """
        self.event_bus = event_bus
        self.window_size = timedelta(minutes=window_minutes)
        self.sensitivity = sensitivity

        # Flow windows per symbol
        self.flow_windows: dict[str, FlowWindow] = {}

        # Historical metrics for trend analysis
        self.metrics_history: dict[str, deque] = {}
        self.history_size = 100

        # Order book snapshots for context
        self.last_snapshots: dict[str, OrderBookSnapshot] = {}

    async def analyze_trade(
        self, symbol: str, price: Decimal, quantity: Decimal, is_buyer_maker: bool
    ) -> OrderFlowMetrics | None:
        """Analyze individual trade for flow imbalance.

        Args:
            symbol: Trading symbol
            price: Trade price
            quantity: Trade quantity
            is_buyer_maker: True if buyer was maker (sell aggression)

        Returns:
            Order flow metrics if significant
        """
        # Determine trade side and aggression
        side = "sell" if is_buyer_maker else "buy"
        is_aggressive = True  # Taker is aggressive

        # Create trade flow record
        trade = TradeFlow(
            timestamp=datetime.now(UTC),
            price=price,
            quantity=quantity,
            side=side,
            is_aggressive=is_aggressive,
        )

        # Get or create flow window
        if symbol not in self.flow_windows:
            self.flow_windows[symbol] = FlowWindow(self.window_size)

        window = self.flow_windows[symbol]
        window.add_trade(trade)

        # Calculate metrics
        metrics = self._calculate_metrics(symbol, window)

        # Store metrics history
        if symbol not in self.metrics_history:
            self.metrics_history[symbol] = deque(maxlen=self.history_size)
        self.metrics_history[symbol].append(metrics)

        # Detect significant imbalances
        if metrics.is_significant():
            await self._publish_imbalance_signal(metrics)

        return metrics

    def update_order_book(self, snapshot: OrderBookSnapshot) -> None:
        """Update order book snapshot for context.

        Args:
            snapshot: Order book snapshot
        """
        self.last_snapshots[snapshot.symbol] = snapshot

    def _calculate_metrics(self, symbol: str, window: FlowWindow) -> OrderFlowMetrics:
        """Calculate order flow metrics from window.

        Args:
            symbol: Trading symbol
            window: Flow window with trades

        Returns:
            Calculated metrics
        """
        buy_volume = window.get_buy_volume()
        sell_volume = window.get_sell_volume()
        total_volume = buy_volume + sell_volume

        # Calculate OFI (Order Flow Imbalance)
        if total_volume > 0:
            ofi = (buy_volume - sell_volume) / total_volume
        else:
            ofi = Decimal("0")

        # Calculate volume ratio
        if sell_volume > 0:
            volume_ratio = buy_volume / sell_volume
        elif buy_volume > 0:
            volume_ratio = Decimal("999")  # Max ratio
        else:
            volume_ratio = Decimal("1")

        # Calculate aggression ratio
        aggressive_vol = window.get_aggressive_volume()
        passive_vol = window.get_passive_volume()

        if passive_vol > 0:
            aggression_ratio = aggressive_vol / passive_vol
        elif aggressive_vol > 0:
            aggression_ratio = Decimal("999")
        else:
            aggression_ratio = Decimal("1")

        # Calculate net flow
        net_flow = buy_volume - sell_volume

        # Calculate flow velocity (rate of change)
        flow_velocity = self._calculate_flow_velocity(symbol, net_flow)

        # Calculate pressure score (-100 to 100)
        pressure_score = self._calculate_pressure_score(
            ofi, volume_ratio, aggression_ratio, flow_velocity
        )

        # Calculate confidence based on volume and order book context
        confidence = self._calculate_confidence(symbol, total_volume, window)

        return OrderFlowMetrics(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            ofi=ofi,
            volume_ratio=volume_ratio,
            aggression_ratio=aggression_ratio,
            net_flow=net_flow,
            flow_velocity=flow_velocity,
            pressure_score=pressure_score,
            confidence=confidence,
        )

    def _calculate_flow_velocity(self, symbol: str, current_flow: Decimal) -> Decimal:
        """Calculate rate of change in flow.

        Args:
            symbol: Trading symbol
            current_flow: Current net flow

        Returns:
            Flow velocity
        """
        if symbol not in self.metrics_history or len(self.metrics_history[symbol]) < 2:
            return Decimal("0")

        previous_metrics = self.metrics_history[symbol][-1]
        time_diff = datetime.now(UTC) - previous_metrics.timestamp

        if time_diff.total_seconds() > 0:
            velocity = (current_flow - previous_metrics.net_flow) / Decimal(
                str(time_diff.total_seconds())
            )
        else:
            velocity = Decimal("0")

        return velocity

    def _calculate_pressure_score(
        self,
        ofi: Decimal,
        volume_ratio: Decimal,
        aggression_ratio: Decimal,
        flow_velocity: Decimal,
    ) -> Decimal:
        """Calculate overall pressure score.

        Args:
            ofi: Order flow imbalance
            volume_ratio: Buy/sell volume ratio
            aggression_ratio: Aggressive/passive ratio
            flow_velocity: Rate of change in flow

        Returns:
            Pressure score from -100 to 100
        """
        # Weight factors
        ofi_weight = Decimal("0.4")
        volume_weight = Decimal("0.3")
        aggression_weight = Decimal("0.2")
        velocity_weight = Decimal("0.1")

        # Normalize inputs
        ofi_score = ofi * Decimal("100")  # Already -1 to 1

        # Volume ratio score (log scale)
        if volume_ratio > 1:
            volume_score = min(
                Decimal("100"),
                Decimal(str(np.log(float(volume_ratio)))) * Decimal("30"),
            )
        elif volume_ratio < 1 and volume_ratio > 0:
            volume_score = max(
                Decimal("-100"),
                Decimal(str(np.log(float(volume_ratio)))) * Decimal("30"),
            )
        else:
            volume_score = Decimal("0")

        # Aggression score
        aggression_score = min(
            Decimal("100"), (aggression_ratio - Decimal("1")) * Decimal("20")
        )

        # Velocity score (normalized)
        velocity_normalized = max(
            Decimal("-100"), min(Decimal("100"), flow_velocity * Decimal("10"))
        )

        # Weighted combination
        pressure = (
            ofi_score * ofi_weight
            + volume_score * volume_weight
            + aggression_score * aggression_weight
            + velocity_normalized * velocity_weight
        )

        # Clamp to range
        return max(Decimal("-100"), min(Decimal("100"), pressure))

    def _calculate_confidence(
        self, symbol: str, total_volume: Decimal, window: FlowWindow
    ) -> Decimal:
        """Calculate confidence in the metrics.

        Args:
            symbol: Trading symbol
            total_volume: Total volume in window
            window: Flow window

        Returns:
            Confidence score (0-1)
        """
        confidence = Decimal("0.5")  # Base confidence

        # More trades = higher confidence
        num_trades = len(window.trades)
        if num_trades > 100:
            confidence += Decimal("0.2")
        elif num_trades > 50:
            confidence += Decimal("0.1")

        # Higher volume = higher confidence
        if total_volume > Decimal("100"):
            confidence += Decimal("0.2")
        elif total_volume > Decimal("50"):
            confidence += Decimal("0.1")

        # Order book context
        if symbol in self.last_snapshots:
            snapshot = self.last_snapshots[symbol]
            if snapshot.spread_bps and snapshot.spread_bps < 10:
                confidence += Decimal("0.1")  # Tight spread = good liquidity

        return min(Decimal("1"), confidence)

    async def _publish_imbalance_signal(self, metrics: OrderFlowMetrics) -> None:
        """Publish order flow imbalance signal.

        Args:
            metrics: Order flow metrics
        """
        direction = "buy" if metrics.is_bullish() else "sell"

        await self.event_bus.publish(
            Event(
                type="order_flow_imbalance",
                data={
                    "symbol": metrics.symbol,
                    "ofi": float(metrics.ofi),
                    "volume_ratio": float(metrics.volume_ratio),
                    "pressure_score": float(metrics.pressure_score),
                    "direction": direction,
                    "confidence": float(metrics.confidence),
                    "timestamp": metrics.timestamp.isoformat(),
                },
            )
        )

        logger.info(
            "order_flow_imbalance_detected",
            symbol=metrics.symbol,
            ofi=float(metrics.ofi),
            pressure_score=float(metrics.pressure_score),
            direction=direction,
        )

    def get_flow_trend(self, symbol: str, periods: int = 10) -> str | None:
        """Get order flow trend over recent periods.

        Args:
            symbol: Trading symbol
            periods: Number of periods to analyze

        Returns:
            Trend direction ('bullish', 'bearish', 'neutral') or None
        """
        if symbol not in self.metrics_history:
            return None

        history = list(self.metrics_history[symbol])[-periods:]
        if len(history) < 3:
            return None

        # Calculate average pressure
        avg_pressure = sum(m.pressure_score for m in history) / len(history)

        # Check trend consistency
        bullish_count = sum(1 for m in history if m.is_bullish())
        bearish_count = sum(1 for m in history if m.is_bearish())

        if avg_pressure > Decimal("20") and bullish_count > len(history) * 0.6:
            return "bullish"
        elif avg_pressure < Decimal("-20") and bearish_count > len(history) * 0.6:
            return "bearish"
        else:
            return "neutral"

    def get_cumulative_flow(self, symbol: str) -> Decimal | None:
        """Get cumulative net flow for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Cumulative net flow or None
        """
        if symbol not in self.flow_windows:
            return None

        window = self.flow_windows[symbol]
        return window.get_buy_volume() - window.get_sell_volume()

    def detect_flow_divergence(self, symbol: str, price_trend: str) -> str | None:
        """Detect divergence between price and flow.

        Args:
            symbol: Trading symbol
            price_trend: Current price trend ('up', 'down', 'sideways')

        Returns:
            Divergence type ('bullish', 'bearish') or None
        """
        flow_trend = self.get_flow_trend(symbol)

        if not flow_trend:
            return None

        # Bullish divergence: price down but flow bullish
        if price_trend == "down" and flow_trend == "bullish":
            return "bullish"

        # Bearish divergence: price up but flow bearish
        if price_trend == "up" and flow_trend == "bearish":
            return "bearish"

        return None
