"""
Spread Analytics Module

Provides comprehensive spread analysis, tracking, and metrics calculation
for identifying profitable trading opportunities.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

import structlog

from genesis.core.exceptions import ValidationError
from typing import Optional

logger = structlog.get_logger(__name__)


@dataclass
class SpreadMetrics:
    """Spread metrics for a trading pair"""

    symbol: str
    current_spread: Decimal
    avg_spread: Decimal
    volatility: Decimal
    bid_price: Decimal
    ask_price: Decimal
    bid_volume: Decimal
    ask_volume: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    spread_bps: Decimal = field(init=False)

    def __post_init__(self) -> None:
        """Calculate spread in basis points"""
        if self.ask_price > 0 and self.bid_price > 0:
            mid_price = (self.ask_price + self.bid_price) / Decimal("2")
            self.spread_bps = ((self.ask_price - self.bid_price) / mid_price) * Decimal(
                "10000"
            )
        else:
            self.spread_bps = Decimal("0")


@dataclass
class OrderImbalance:
    """Order book imbalance metrics"""

    ratio: Decimal
    bid_weight: Decimal
    ask_weight: Decimal
    is_significant: bool = field(init=False)

    def __post_init__(self) -> None:
        """Determine if imbalance is significant"""
        self.is_significant = self.ratio > Decimal("2.0") or self.ratio < Decimal("0.5")


@dataclass
class SpreadCompressionEvent:
    """Event for spread compression detection"""

    symbol: str
    current_spread: Decimal
    average_spread: Decimal
    compression_ratio: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class SpreadAnalyzer:
    """
    Main spread analytics engine for tracking and analyzing spreads
    across multiple trading pairs
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Initialize spread analyzer

        Args:
            max_history_size: Maximum number of spreads to keep in memory per pair
        """
        self.max_history_size = max_history_size
        self._spread_history: dict[str, deque[Decimal]] = {}
        self._metrics_cache: dict[str, SpreadMetrics] = {}
        self._moving_averages: dict[str, Decimal] = {}
        self._compression_tracking: dict[str, datetime] = {}
        self._logger = logger.bind(component="SpreadAnalyzer")

    def track_spread(self, symbol: str, bid: Decimal, ask: Decimal) -> SpreadMetrics:
        """
        Track spread for a trading pair

        Args:
            symbol: Trading pair symbol
            bid: Best bid price
            ask: Best ask price

        Returns:
            SpreadMetrics object with current spread data
        """
        if bid <= 0 or ask <= 0:
            raise ValidationError(f"Invalid prices: bid={bid}, ask={ask}")

        if ask <= bid:
            raise ValidationError(
                f"Ask price must be greater than bid: bid={bid}, ask={ask}"
            )

        # Calculate spread in basis points
        spread_bps = self.calculate_spread_bps(bid, ask)

        # Update history
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=self.max_history_size)

        self._spread_history[symbol].append(spread_bps)

        # Calculate metrics
        spreads = list(self._spread_history[symbol])
        avg_spread = sum(spreads) / len(spreads) if spreads else spread_bps
        volatility = self.calculate_spread_volatility(spreads)

        # Create metrics object
        metrics = SpreadMetrics(
            symbol=symbol,
            current_spread=spread_bps,
            avg_spread=avg_spread,
            volatility=volatility,
            bid_price=bid,
            ask_price=ask,
            bid_volume=Decimal("0"),  # Will be updated when orderbook data available
            ask_volume=Decimal("0"),
        )

        self._metrics_cache[symbol] = metrics

        self._logger.debug(
            "Spread tracked",
            symbol=symbol,
            spread_bps=float(spread_bps),
            avg_spread=float(avg_spread),
            volatility=float(volatility),
        )

        return metrics

    def calculate_spread_bps(self, bid: Decimal, ask: Decimal) -> Decimal:
        """
        Calculate spread in basis points

        Args:
            bid: Best bid price
            ask: Best ask price

        Returns:
            Spread in basis points
        """
        if bid <= 0 or ask <= 0:
            return Decimal("0")

        mid_price = (ask + bid) / Decimal("2")
        spread_bps = ((ask - bid) / mid_price) * Decimal("10000")
        return spread_bps

    def calculate_spread_volatility(self, spreads: list[Decimal]) -> Decimal:
        """
        Calculate spread volatility (standard deviation)

        Args:
            spreads: List of spread values

        Returns:
            Volatility as standard deviation
        """
        if not spreads or len(spreads) < 2:
            return Decimal("0")

        mean = sum(spreads) / len(spreads)
        variance = sum((x - mean) ** 2 for x in spreads) / len(spreads)
        std_dev = variance.sqrt()

        return std_dev

    def get_spread_metrics(self, symbol: str) -> Optional[SpreadMetrics]:
        """
        Get current spread metrics for a symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            SpreadMetrics or None if not tracked
        """
        return self._metrics_cache.get(symbol)

    def get_all_metrics(self) -> dict[str, SpreadMetrics]:
        """
        Get spread metrics for all tracked symbols

        Returns:
            Dictionary of symbol to SpreadMetrics
        """
        return self._metrics_cache.copy()

    def calculate_order_imbalance(self, orderbook: dict) -> OrderImbalance:
        """
        Calculate order book imbalance

        Args:
            orderbook: Order book data with bids and asks

        Returns:
            OrderImbalance metrics
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return OrderImbalance(
                ratio=Decimal("1.0"), bid_weight=Decimal("0"), ask_weight=Decimal("0")
            )

        # Calculate weighted volumes
        bid_weight = Decimal("0")
        ask_weight = Decimal("0")

        # Weight by price level (closer to mid = higher weight)
        for i, (_price, volume) in enumerate(bids[:10]):
            weight = Decimal(1) / Decimal(i + 1)
            bid_weight += Decimal(str(volume)) * weight

        for i, (_price, volume) in enumerate(asks[:10]):
            weight = Decimal(1) / Decimal(i + 1)
            ask_weight += Decimal(str(volume)) * weight

        # Calculate ratio
        ratio = bid_weight / ask_weight if ask_weight > 0 else Decimal("0")

        return OrderImbalance(ratio=ratio, bid_weight=bid_weight, ask_weight=ask_weight)

    def detect_spread_compression(self, symbol: str) -> Optional[SpreadCompressionEvent]:
        """
        Detect spread compression for a symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            SpreadCompressionEvent if compression detected, None otherwise
        """
        if symbol not in self._spread_history:
            return None

        spreads = list(self._spread_history[symbol])
        if len(spreads) < 20:
            return None  # Need at least 20 periods for moving average

        # Calculate 20-period moving average
        ma_20 = sum(spreads[-20:]) / Decimal("20")
        current_spread = spreads[-1]

        # Update moving average cache
        self._moving_averages[symbol] = ma_20

        # Check for compression (current < 80% of average)
        compression_threshold = ma_20 * Decimal("0.8")

        if current_spread < compression_threshold:
            compression_ratio = current_spread / ma_20

            # Track compression start time
            if symbol not in self._compression_tracking:
                self._compression_tracking[symbol] = datetime.now(UTC)

            event = SpreadCompressionEvent(
                symbol=symbol,
                current_spread=current_spread,
                average_spread=ma_20,
                compression_ratio=compression_ratio,
            )

            self._logger.info(
                "Spread compression detected",
                symbol=symbol,
                current_spread=float(current_spread),
                average_spread=float(ma_20),
                compression_ratio=float(compression_ratio),
            )

            return event
        else:
            # Clear compression tracking if spread recovered
            if symbol in self._compression_tracking:
                recovery_threshold = ma_20 * Decimal("0.9")
                if current_spread > recovery_threshold:
                    del self._compression_tracking[symbol]
                    self._logger.info(
                        "Spread compression recovered",
                        symbol=symbol,
                        current_spread=float(current_spread),
                        average_spread=float(ma_20),
                    )

        return None

    def get_compression_duration(self, symbol: str) -> Optional[float]:
        """
        Get duration of current spread compression

        Args:
            symbol: Trading pair symbol

        Returns:
            Duration in seconds or None if not compressed
        """
        if symbol not in self._compression_tracking:
            return None

        duration = (
            datetime.now(UTC) - self._compression_tracking[symbol]
        ).total_seconds()
        return duration

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """
        Clear spread history

        Args:
            symbol: Symbol to clear, or None to clear all
        """
        if symbol:
            if symbol in self._spread_history:
                del self._spread_history[symbol]
            if symbol in self._metrics_cache:
                del self._metrics_cache[symbol]
            if symbol in self._moving_averages:
                del self._moving_averages[symbol]
            if symbol in self._compression_tracking:
                del self._compression_tracking[symbol]
        else:
            self._spread_history.clear()
            self._metrics_cache.clear()
            self._moving_averages.clear()
            self._compression_tracking.clear()

        self._logger.info("Spread history cleared", symbol=symbol)
