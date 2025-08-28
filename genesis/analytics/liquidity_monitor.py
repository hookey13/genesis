"""
Liquidity monitor for market depth analysis.

Monitors order book depth, spread widths, and volume patterns
to detect liquidity crises that could impact execution quality
and increase slippage risk.
"""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class LiquidityState(str, Enum):
    """Market liquidity states."""

    HEALTHY = "healthy"  # Normal liquidity
    DEGRADED = "degraded"  # Reduced liquidity
    CRISIS = "crisis"  # Severe liquidity shortage
    FROZEN = "frozen"  # No liquidity available


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state."""

    timestamp: datetime
    symbol: str
    best_bid: Decimal
    best_ask: Decimal
    spread: Decimal
    spread_bps: int  # Spread in basis points
    bid_depth_1pct: Decimal  # Total bids within 1% of best bid
    ask_depth_1pct: Decimal  # Total asks within 1% of best ask
    total_bid_volume: Decimal
    total_ask_volume: Decimal
    imbalance_ratio: Decimal  # Bid volume / Ask volume


@dataclass
class LiquidityCrisis:
    """Record of a liquidity crisis event."""

    symbol: str
    state: LiquidityState
    liquidity_score: Decimal  # 0-1 score
    spread_bps: int
    depth_reduction_pct: Decimal
    volume_reduction_pct: Decimal
    detected_at: datetime
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL


class LiquidityMonitor:
    """
    Monitors market liquidity conditions across trading pairs.

    Detects:
    - Sudden spread widening
    - Order book depth reduction
    - Volume dry-ups
    - One-sided markets
    """

    def __init__(
        self,
        event_bus: EventBus,
        depth_window_minutes: int = 5,
        volume_window_minutes: int = 30,
        crisis_threshold: Decimal = Decimal("0.30"),  # Below 30% normal liquidity
        spread_alert_bps: int = 50,  # Alert if spread > 50 basis points
    ):
        """
        Initialize liquidity monitor.

        Args:
            event_bus: Event bus for publishing alerts
            depth_window_minutes: Window for depth analysis
            volume_window_minutes: Window for volume analysis
            crisis_threshold: Liquidity score threshold for crisis
            spread_alert_bps: Spread threshold in basis points
        """
        self.event_bus = event_bus
        self.depth_window_minutes = depth_window_minutes
        self.volume_window_minutes = volume_window_minutes
        self.crisis_threshold = crisis_threshold
        self.spread_alert_bps = spread_alert_bps

        # Order book history
        self.order_book_history: dict[str, deque[OrderBookSnapshot]] = {}

        # Current liquidity state
        self.liquidity_states: dict[str, LiquidityState] = {}
        self.liquidity_scores: dict[str, Decimal] = {}

        # Baseline liquidity metrics (for comparison)
        self.baseline_depth: dict[str, Decimal] = {}
        self.baseline_volume: dict[str, Decimal] = {}
        self.baseline_spread: dict[str, int] = {}

        # Active crises
        self.active_crises: dict[str, LiquidityCrisis] = {}

        # Statistics
        self.snapshots_processed = 0
        self.crises_detected = 0
        self.worst_liquidity_score = Decimal("1.0")

        logger.info(
            "LiquidityMonitor initialized",
            depth_window=depth_window_minutes,
            volume_window=volume_window_minutes,
            crisis_threshold=float(crisis_threshold),
        )

    def process_order_book(
        self,
        symbol: str,
        bids: list[tuple[Decimal, Decimal]],  # [(price, volume), ...]
        asks: list[tuple[Decimal, Decimal]],
        timestamp: Optional[datetime] = None,
    ) -> OrderBookSnapshot:
        """
        Process order book data and calculate liquidity metrics.

        Args:
            symbol: Trading symbol
            bids: List of bid levels (price, volume)
            asks: List of ask levels (price, volume)
            timestamp: Snapshot time

        Returns:
            Order book snapshot with calculated metrics
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        if not bids or not asks:
            logger.warning(
                "Empty order book received",
                symbol=symbol,
                has_bids=bool(bids),
                has_asks=bool(asks),
            )
            # Return minimal snapshot
            return OrderBookSnapshot(
                timestamp=timestamp,
                symbol=symbol,
                best_bid=Decimal("0"),
                best_ask=Decimal("0"),
                spread=Decimal("999999"),
                spread_bps=999999,
                bid_depth_1pct=Decimal("0"),
                ask_depth_1pct=Decimal("0"),
                total_bid_volume=Decimal("0"),
                total_ask_volume=Decimal("0"),
                imbalance_ratio=Decimal("0"),
            )

        # Calculate basic metrics
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid

        # Calculate spread in basis points
        if best_bid > 0:
            spread_bps = int((spread / best_bid) * 10000)
        else:
            spread_bps = 999999

        # Calculate depth within 1% of best prices
        bid_depth_1pct = Decimal("0")
        bid_threshold = best_bid * Decimal("0.99")  # 1% below best bid

        for price, volume in bids:
            if price >= bid_threshold:
                bid_depth_1pct += volume
            else:
                break  # Assuming sorted order

        ask_depth_1pct = Decimal("0")
        ask_threshold = best_ask * Decimal("1.01")  # 1% above best ask

        for price, volume in asks:
            if price <= ask_threshold:
                ask_depth_1pct += volume
            else:
                break

        # Calculate total volumes
        total_bid_volume = sum(volume for _, volume in bids)
        total_ask_volume = sum(volume for _, volume in asks)

        # Calculate imbalance ratio
        if total_ask_volume > 0:
            imbalance_ratio = total_bid_volume / total_ask_volume
        else:
            imbalance_ratio = Decimal("999") if total_bid_volume > 0 else Decimal("1")

        # Create snapshot
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            spread_bps=spread_bps,
            bid_depth_1pct=bid_depth_1pct,
            ask_depth_1pct=ask_depth_1pct,
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            imbalance_ratio=imbalance_ratio,
        )

        # Store in history
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = deque(maxlen=1000)

        self.order_book_history[symbol].append(snapshot)
        self.snapshots_processed += 1

        # Update baseline if needed
        self._update_baseline(symbol, snapshot)

        return snapshot

    def _update_baseline(self, symbol: str, snapshot: OrderBookSnapshot) -> None:
        """
        Update baseline liquidity metrics.

        Args:
            symbol: Trading symbol
            snapshot: Current order book snapshot
        """
        # Simple exponential moving average for baseline
        alpha = Decimal("0.02")  # Smoothing factor

        total_depth = snapshot.bid_depth_1pct + snapshot.ask_depth_1pct
        total_volume = snapshot.total_bid_volume + snapshot.total_ask_volume

        if symbol not in self.baseline_depth:
            # Initialize baselines
            self.baseline_depth[symbol] = total_depth
            self.baseline_volume[symbol] = total_volume
            self.baseline_spread[symbol] = snapshot.spread_bps
        else:
            # Update with EMA
            self.baseline_depth[symbol] = (
                alpha * total_depth
                + (Decimal("1") - alpha) * self.baseline_depth[symbol]
            )
            self.baseline_volume[symbol] = (
                alpha * total_volume
                + (Decimal("1") - alpha) * self.baseline_volume[symbol]
            )
            self.baseline_spread[symbol] = int(
                float(alpha) * snapshot.spread_bps
                + (1 - float(alpha)) * self.baseline_spread[symbol]
            )

    async def calculate_liquidity_score(self, symbol: str) -> Decimal:
        """
        Calculate comprehensive liquidity score for a symbol.

        Score components:
        - Spread tightness (25%)
        - Order book depth (35%)
        - Volume consistency (25%)
        - Market balance (15%)

        Returns:
            Liquidity score from 0 (no liquidity) to 1 (perfect liquidity)
        """
        if symbol not in self.order_book_history:
            return Decimal("1.0")  # No data, assume healthy

        history = list(self.order_book_history[symbol])
        if not history:
            return Decimal("1.0")

        # Get recent snapshot
        latest = history[-1]

        # Component 1: Spread tightness (lower is better)
        if self.baseline_spread.get(symbol, 0) > 0:
            spread_ratio = min(
                Decimal(str(self.baseline_spread[symbol] / max(1, latest.spread_bps))),
                Decimal("1.0"),
            )
        else:
            spread_ratio = Decimal("0.5") if latest.spread_bps < 100 else Decimal("0.2")

        spread_score = spread_ratio * Decimal("0.25")

        # Component 2: Order book depth
        if symbol in self.baseline_depth and self.baseline_depth[symbol] > 0:
            current_depth = latest.bid_depth_1pct + latest.ask_depth_1pct
            depth_ratio = min(
                current_depth / self.baseline_depth[symbol], Decimal("1.0")
            )
        else:
            depth_ratio = Decimal("0.5")  # No baseline

        depth_score = depth_ratio * Decimal("0.35")

        # Component 3: Volume consistency
        if symbol in self.baseline_volume and self.baseline_volume[symbol] > 0:
            current_volume = latest.total_bid_volume + latest.total_ask_volume
            volume_ratio = min(
                current_volume / self.baseline_volume[symbol], Decimal("1.0")
            )
        else:
            volume_ratio = Decimal("0.5")

        volume_score = volume_ratio * Decimal("0.25")

        # Component 4: Market balance (imbalance is bad)
        if latest.imbalance_ratio > Decimal("0"):
            # Perfect balance = 1.0, convert to score
            if latest.imbalance_ratio > Decimal("1"):
                balance_ratio = Decimal("1") / latest.imbalance_ratio
            else:
                balance_ratio = latest.imbalance_ratio

            # Penalize extreme imbalances
            if balance_ratio < Decimal("0.5"):
                balance_ratio = balance_ratio * Decimal("0.5")
        else:
            balance_ratio = Decimal("0")

        balance_score = balance_ratio * Decimal("0.15")

        # Calculate total score
        liquidity_score = spread_score + depth_score + volume_score + balance_score

        # Store score
        self.liquidity_scores[symbol] = liquidity_score

        # Track worst score
        if liquidity_score < self.worst_liquidity_score:
            self.worst_liquidity_score = liquidity_score

        # Check for crisis
        await self._check_for_crisis(symbol, liquidity_score, latest)

        return liquidity_score

    async def _check_for_crisis(
        self, symbol: str, liquidity_score: Decimal, snapshot: OrderBookSnapshot
    ) -> None:
        """
        Check if liquidity crisis conditions are met.

        Args:
            symbol: Trading symbol
            liquidity_score: Current liquidity score
            snapshot: Latest order book snapshot
        """
        # Determine state based on score
        if liquidity_score < Decimal("0.10"):
            new_state = LiquidityState.FROZEN
            severity = "CRITICAL"
        elif liquidity_score < Decimal("0.20"):
            new_state = LiquidityState.CRISIS
            severity = "HIGH"
        elif liquidity_score < Decimal("0.40"):
            new_state = LiquidityState.DEGRADED
            severity = "MEDIUM"
        else:
            new_state = LiquidityState.HEALTHY
            severity = "LOW"

        # Check for state change or worsening
        old_state = self.liquidity_states.get(symbol, LiquidityState.HEALTHY)
        self.liquidity_states[symbol] = new_state

        # Alert on crisis conditions
        if new_state in [LiquidityState.CRISIS, LiquidityState.FROZEN]:
            # Calculate reduction percentages
            depth_reduction = Decimal("0")
            volume_reduction = Decimal("0")

            if symbol in self.baseline_depth and self.baseline_depth[symbol] > 0:
                current_depth = snapshot.bid_depth_1pct + snapshot.ask_depth_1pct
                depth_reduction = (
                    (self.baseline_depth[symbol] - current_depth)
                    / self.baseline_depth[symbol]
                ) * Decimal("100")

            if symbol in self.baseline_volume and self.baseline_volume[symbol] > 0:
                current_volume = snapshot.total_bid_volume + snapshot.total_ask_volume
                volume_reduction = (
                    (self.baseline_volume[symbol] - current_volume)
                    / self.baseline_volume[symbol]
                ) * Decimal("100")

            crisis = LiquidityCrisis(
                symbol=symbol,
                state=new_state,
                liquidity_score=liquidity_score,
                spread_bps=snapshot.spread_bps,
                depth_reduction_pct=max(depth_reduction, Decimal("0")),
                volume_reduction_pct=max(volume_reduction, Decimal("0")),
                detected_at=datetime.now(UTC),
                severity=severity,
            )

            # Store or update crisis
            if (
                symbol not in self.active_crises
                or crisis.severity != self.active_crises[symbol].severity
            ):
                self.active_crises[symbol] = crisis
                self.crises_detected += 1

                # Publish alert
                await self._publish_crisis_alert(crisis)

        # Clear crisis if recovered
        elif symbol in self.active_crises and new_state == LiquidityState.HEALTHY:
            del self.active_crises[symbol]

            # Publish recovery
            await self.event_bus.publish(
                Event(
                    event_type=EventType.MARKET_STATE_CHANGE,
                    aggregate_id=symbol,
                    event_data={
                        "state": "liquidity_recovered",
                        "symbol": symbol,
                        "liquidity_score": float(liquidity_score),
                        "spread_bps": snapshot.spread_bps,
                    },
                ),
                priority=EventPriority.HIGH,
            )

    async def _publish_crisis_alert(self, crisis: LiquidityCrisis) -> None:
        """
        Publish liquidity crisis alert.

        Args:
            crisis: Liquidity crisis details
        """
        logger.critical(
            "Liquidity crisis detected",
            symbol=crisis.symbol,
            state=crisis.state.value,
            liquidity_score=float(crisis.liquidity_score),
            severity=crisis.severity,
        )

        await self.event_bus.publish(
            Event(
                event_type=EventType.MARKET_STATE_CHANGE,
                aggregate_id=crisis.symbol,
                event_data={
                    "state": "liquidity_crisis",
                    "symbol": crisis.symbol,
                    "crisis_state": crisis.state.value,
                    "liquidity_score": float(crisis.liquidity_score),
                    "spread_bps": crisis.spread_bps,
                    "depth_reduction_pct": float(crisis.depth_reduction_pct),
                    "volume_reduction_pct": float(crisis.volume_reduction_pct),
                    "severity": crisis.severity,
                    "recommendation": self._get_recommendation(crisis),
                },
            ),
            priority=EventPriority.CRITICAL,
        )

    def _get_recommendation(self, crisis: LiquidityCrisis) -> str:
        """
        Get trading recommendation for liquidity crisis.

        Args:
            crisis: Liquidity crisis details

        Returns:
            Trading recommendation
        """
        if crisis.state == LiquidityState.FROZEN:
            return "HALT TRADING: No liquidity available - close positions if possible"
        elif crisis.state == LiquidityState.CRISIS:
            return "EMERGENCY: Use market orders only - expect high slippage"
        elif crisis.state == LiquidityState.DEGRADED:
            return "CAUTION: Reduce order sizes - limited liquidity available"
        else:
            return "MONITOR: Liquidity conditions improving"

    def get_liquidity_summary(self) -> dict[str, Any]:
        """
        Get summary of current liquidity conditions.

        Returns:
            Summary dictionary
        """
        healthy_count = sum(
            1
            for state in self.liquidity_states.values()
            if state == LiquidityState.HEALTHY
        )

        degraded_count = sum(
            1
            for state in self.liquidity_states.values()
            if state == LiquidityState.DEGRADED
        )

        crisis_count = sum(
            1
            for state in self.liquidity_states.values()
            if state in [LiquidityState.CRISIS, LiquidityState.FROZEN]
        )

        return {
            "total_symbols": len(self.liquidity_states),
            "healthy_symbols": healthy_count,
            "degraded_symbols": degraded_count,
            "crisis_symbols": crisis_count,
            "active_crises": len(self.active_crises),
            "worst_liquidity_score": float(self.worst_liquidity_score),
            "average_liquidity_score": (
                float(sum(self.liquidity_scores.values()) / len(self.liquidity_scores))
                if self.liquidity_scores
                else 1.0
            ),
            "snapshots_processed": self.snapshots_processed,
            "crises_detected": self.crises_detected,
        }

    def get_symbol_status(self, symbol: str) -> dict[str, Any]:
        """
        Get detailed liquidity status for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Status dictionary
        """
        status = {
            "symbol": symbol,
            "state": self.liquidity_states.get(symbol, LiquidityState.HEALTHY).value,
            "liquidity_score": float(self.liquidity_scores.get(symbol, Decimal("1.0"))),
            "has_crisis": symbol in self.active_crises,
        }

        if symbol in self.active_crises:
            crisis = self.active_crises[symbol]
            status.update(
                {
                    "crisis_severity": crisis.severity,
                    "spread_bps": crisis.spread_bps,
                    "depth_reduction_pct": float(crisis.depth_reduction_pct),
                    "volume_reduction_pct": float(crisis.volume_reduction_pct),
                    "crisis_duration_seconds": (
                        datetime.now(UTC) - crisis.detected_at
                    ).total_seconds(),
                }
            )

        if self.order_book_history.get(symbol):
            latest = self.order_book_history[symbol][-1]
            status.update(
                {
                    "latest_spread_bps": latest.spread_bps,
                    "imbalance_ratio": float(latest.imbalance_ratio),
                    "bid_depth": float(latest.bid_depth_1pct),
                    "ask_depth": float(latest.ask_depth_1pct),
                }
            )

        return status

    def reset(self) -> None:
        """Reset monitor state (useful for testing)."""
        self.order_book_history.clear()
        self.liquidity_states.clear()
        self.liquidity_scores.clear()
        self.baseline_depth.clear()
        self.baseline_volume.clear()
        self.baseline_spread.clear()
        self.active_crises.clear()
        self.snapshots_processed = 0
        self.crises_detected = 0
        self.worst_liquidity_score = Decimal("1.0")

        logger.info("Liquidity monitor reset")
