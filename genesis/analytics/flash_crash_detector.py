"""
Flash crash detector for rapid price movement monitoring.

Detects sudden price drops that occur within very short timeframes,
enabling automatic protective actions like order cancellation to
prevent catastrophic losses during market disruptions.
"""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional

import structlog

from genesis.core.events import Event, EventPriority, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


@dataclass
class PriceSnapshot:
    """Single price observation with context."""

    timestamp: datetime
    symbol: str
    price: Decimal
    volume: Decimal
    trades_count: int


@dataclass
class FlashCrashEvent:
    """Record of a flash crash detection."""

    symbol: str
    start_price: Decimal
    end_price: Decimal
    drop_percentage: Decimal
    duration_seconds: float
    start_time: datetime
    end_time: datetime
    max_price: Decimal
    min_price: Decimal
    volume_during_crash: Decimal
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL


class FlashCrashDetector:
    """
    Detects flash crashes - rapid price drops within short time windows.

    Flash crashes can occur due to:
    - Large market sell orders
    - Algorithmic trading errors
    - Liquidity evaporation
    - Cascade of stop-loss orders
    """

    def __init__(
        self,
        event_bus: EventBus,
        window_seconds: int = 60,
        crash_threshold: Decimal = Decimal("0.10"),  # 10% drop
        recovery_window_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize flash crash detector.

        Args:
            event_bus: Event bus for publishing alerts
            window_seconds: Time window for crash detection
            crash_threshold: Percentage drop to trigger alert (0.10 = 10%)
            recovery_window_seconds: Time before resetting after crash
        """
        self.event_bus = event_bus
        self.window_seconds = window_seconds
        self.crash_threshold = crash_threshold
        self.recovery_window_seconds = recovery_window_seconds

        # Price history for each symbol
        # Using deque for efficient removal of old data
        self.price_history: dict[str, deque[PriceSnapshot]] = {}

        # Active flash crashes
        self.active_crashes: dict[str, FlashCrashEvent] = {}

        # Crash history
        self.crash_history: list[FlashCrashEvent] = []

        # Statistics
        self.prices_processed = 0
        self.crashes_detected = 0
        self.false_positives = 0
        self.worst_crash_pct = Decimal("0")
        self.fastest_crash_seconds = float('inf')

        # Order cancellation tracking
        self.symbols_with_cancelled_orders: set[str] = set()

        logger.info(
            "FlashCrashDetector initialized",
            window_seconds=window_seconds,
            crash_threshold=float(crash_threshold),
            recovery_window=recovery_window_seconds
        )

    async def process_price(
        self,
        symbol: str,
        price: Decimal,
        volume: Decimal = Decimal("0"),
        trades_count: int = 0,
        timestamp: Optional[datetime] = None
    ) -> Optional[FlashCrashEvent]:
        """
        Process a new price and check for flash crash.

        Args:
            symbol: Trading symbol
            price: Current price
            volume: Volume at this price (optional)
            trades_count: Number of trades (optional)
            timestamp: Price timestamp

        Returns:
            FlashCrashEvent if crash detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        self.prices_processed += 1

        # Create snapshot
        snapshot = PriceSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            price=price,
            volume=volume,
            trades_count=trades_count
        )

        # Initialize history if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=10000)

        # Add to history
        self.price_history[symbol].append(snapshot)

        # Clean old data
        self._clean_old_prices(symbol)

        # Check for flash crash
        crash_event = await self._detect_flash_crash(symbol)

        return crash_event

    def _clean_old_prices(self, symbol: str) -> None:
        """
        Remove prices outside the analysis window.

        Args:
            symbol: Trading symbol
        """
        if symbol not in self.price_history:
            return

        # Keep data for crash window plus recovery window
        cutoff = datetime.now(UTC) - timedelta(
            seconds=self.window_seconds + self.recovery_window_seconds
        )

        # Remove old prices from front of deque
        while (self.price_history[symbol] and
               self.price_history[symbol][0].timestamp < cutoff):
            self.price_history[symbol].popleft()

    async def _detect_flash_crash(self, symbol: str) -> Optional[FlashCrashEvent]:
        """
        Detect flash crash in recent price history.

        Args:
            symbol: Trading symbol

        Returns:
            FlashCrashEvent if detected, None otherwise
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return None

        # Check if already in active crash
        if symbol in self.active_crashes:
            # Check if crash has recovered
            crash = self.active_crashes[symbol]
            time_since_crash = (datetime.now(UTC) - crash.end_time).total_seconds()

            if time_since_crash > self.recovery_window_seconds:
                # Recovery period over, can detect new crashes
                del self.active_crashes[symbol]
                if symbol in self.symbols_with_cancelled_orders:
                    self.symbols_with_cancelled_orders.remove(symbol)
            else:
                # Still in recovery period
                return None

        # Get prices within detection window
        now = datetime.now(UTC)
        window_start = now - timedelta(seconds=self.window_seconds)

        window_prices = [
            s for s in self.price_history[symbol]
            if s.timestamp >= window_start
        ]

        if len(window_prices) < 2:
            return None

        # Find max and min prices in window
        max_price = max(s.price for s in window_prices)
        min_price = min(s.price for s in window_prices)

        # Find when they occurred
        max_snapshot = next(s for s in window_prices if s.price == max_price)
        min_snapshot = next(s for s in window_prices if s.price == min_price)

        # Check if min came after max (price dropped)
        if min_snapshot.timestamp <= max_snapshot.timestamp:
            return None  # Price went up or sideways, not a crash

        # Calculate drop percentage
        if max_price > 0:
            drop_pct = (max_price - min_price) / max_price
        else:
            return None

        # Check if exceeds threshold
        if drop_pct >= self.crash_threshold:
            # Calculate crash duration
            duration_seconds = (min_snapshot.timestamp - max_snapshot.timestamp).total_seconds()

            # Calculate volume during crash
            crash_volume = sum(
                s.volume for s in window_prices
                if max_snapshot.timestamp <= s.timestamp <= min_snapshot.timestamp
            )

            # Determine severity
            if drop_pct >= Decimal("0.30"):
                severity = "CRITICAL"
            elif drop_pct >= Decimal("0.20"):
                severity = "HIGH"
            elif drop_pct >= Decimal("0.15"):
                severity = "MEDIUM"
            else:
                severity = "LOW"

            # Create crash event
            crash_event = FlashCrashEvent(
                symbol=symbol,
                start_price=max_price,
                end_price=min_price,
                drop_percentage=drop_pct,
                duration_seconds=duration_seconds,
                start_time=max_snapshot.timestamp,
                end_time=min_snapshot.timestamp,
                max_price=max_price,
                min_price=min_price,
                volume_during_crash=crash_volume,
                severity=severity
            )

            # Store crash
            self.active_crashes[symbol] = crash_event
            self.crash_history.append(crash_event)
            self.crashes_detected += 1

            # Update statistics
            if drop_pct > self.worst_crash_pct:
                self.worst_crash_pct = drop_pct

            if duration_seconds < self.fastest_crash_seconds:
                self.fastest_crash_seconds = duration_seconds

            # Publish alert
            await self._publish_crash_alert(crash_event)

            # Trigger protective actions
            await self._trigger_protective_actions(crash_event)

            return crash_event

        return None

    async def _publish_crash_alert(self, crash: FlashCrashEvent) -> None:
        """
        Publish flash crash alert.

        Args:
            crash: Flash crash event details
        """
        logger.critical(
            "FLASH CRASH DETECTED",
            symbol=crash.symbol,
            drop_percentage=float(crash.drop_percentage),
            duration_seconds=crash.duration_seconds,
            severity=crash.severity
        )

        await self.event_bus.publish(
            Event(
                event_type=EventType.MARKET_STATE_CHANGE,
                aggregate_id=crash.symbol,
                event_data={
                    "alert_type": "flash_crash",
                    "symbol": crash.symbol,
                    "start_price": float(crash.start_price),
                    "end_price": float(crash.end_price),
                    "drop_percentage": float(crash.drop_percentage),
                    "duration_seconds": crash.duration_seconds,
                    "severity": crash.severity,
                    "volume_during_crash": float(crash.volume_during_crash),
                    "action": "cancel_all_orders",
                    "recommendation": self._get_recommendation(crash)
                }
            ),
            priority=EventPriority.CRITICAL
        )

    async def _trigger_protective_actions(self, crash: FlashCrashEvent) -> None:
        """
        Trigger protective actions for flash crash.

        Args:
            crash: Flash crash event
        """
        # Mark symbol for order cancellation
        self.symbols_with_cancelled_orders.add(crash.symbol)

        # Publish order cancellation event
        await self.event_bus.publish(
            Event(
                event_type=EventType.ORDER_CANCELLED,
                aggregate_id=crash.symbol,
                event_data={
                    "reason": "flash_crash_protection",
                    "symbol": crash.symbol,
                    "action": "cancel_all_open_orders",
                    "severity": crash.severity
                }
            ),
            priority=EventPriority.CRITICAL
        )

        logger.info(
            "Protective actions triggered",
            symbol=crash.symbol,
            action="cancel_all_orders"
        )

    def _get_recommendation(self, crash: FlashCrashEvent) -> str:
        """
        Get trading recommendation for flash crash.

        Args:
            crash: Flash crash event

        Returns:
            Trading recommendation
        """
        if crash.severity == "CRITICAL":
            return "HALT ALL TRADING: Extreme volatility - wait for market stability"
        elif crash.severity == "HIGH":
            return "CANCEL ALL ORDERS: High volatility - reassess positions"
        elif crash.severity == "MEDIUM":
            return "REDUCE EXPOSURE: Moderate volatility - tighten stops"
        else:
            return "MONITOR CLOSELY: Mild volatility detected"

    async def cancel_all_orders(self, symbol: str) -> dict[str, Any]:
        """
        Cancel all open orders for a symbol (to be implemented with exchange).

        Args:
            symbol: Trading symbol

        Returns:
            Cancellation result
        """
        # This would integrate with the exchange gateway
        # For now, return a mock response

        result = {
            "symbol": symbol,
            "orders_cancelled": 0,  # Would be actual count
            "success": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": "flash_crash_protection"
        }

        logger.info(
            "Order cancellation requested",
            symbol=symbol,
            reason="flash_crash"
        )

        return result

    def get_crash_history(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Get crash history, optionally filtered by symbol.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of crash events
        """
        crashes = self.crash_history

        if symbol:
            crashes = [c for c in crashes if c.symbol == symbol]

        return [
            {
                "symbol": c.symbol,
                "drop_percentage": float(c.drop_percentage),
                "duration_seconds": c.duration_seconds,
                "start_time": c.start_time.isoformat(),
                "end_time": c.end_time.isoformat(),
                "severity": c.severity,
                "start_price": float(c.start_price),
                "end_price": float(c.end_price),
                "volume": float(c.volume_during_crash)
            }
            for c in crashes
        ]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get detector statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "prices_processed": self.prices_processed,
            "crashes_detected": self.crashes_detected,
            "false_positives": self.false_positives,
            "worst_crash_percentage": float(self.worst_crash_pct),
            "fastest_crash_seconds": self.fastest_crash_seconds if self.fastest_crash_seconds != float('inf') else None,
            "active_crashes": len(self.active_crashes),
            "symbols_monitored": len(self.price_history),
            "symbols_with_cancelled_orders": list(self.symbols_with_cancelled_orders),
            "detection_window_seconds": self.window_seconds,
            "crash_threshold": float(self.crash_threshold),
            "recovery_window_seconds": self.recovery_window_seconds
        }

    def is_in_crash_recovery(self, symbol: str) -> bool:
        """
        Check if symbol is in crash recovery period.

        Args:
            symbol: Trading symbol

        Returns:
            True if in recovery period
        """
        if symbol not in self.active_crashes:
            return False

        crash = self.active_crashes[symbol]
        time_since_crash = (datetime.now(UTC) - crash.end_time).total_seconds()

        return time_since_crash <= self.recovery_window_seconds

    def reset(self) -> None:
        """Reset detector state (useful for testing)."""
        self.price_history.clear()
        self.active_crashes.clear()
        self.crash_history.clear()
        self.symbols_with_cancelled_orders.clear()
        self.prices_processed = 0
        self.crashes_detected = 0
        self.false_positives = 0
        self.worst_crash_pct = Decimal("0")
        self.fastest_crash_seconds = float('inf')

        logger.info("Flash crash detector reset")
