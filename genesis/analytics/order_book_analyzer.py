"""
Order book depth analyzer for Project GENESIS.

This module analyzes order book liquidity to determine optimal slicing
strategies for large orders, minimizing market impact.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

import structlog

from genesis.engine.executor.base import OrderSide
from genesis.exchange.models import OrderBook

logger = structlog.get_logger(__name__)


class LiquidityLevel(str, Enum):
    """Liquidity level classification."""

    DEEP = "DEEP"  # Can absorb large orders
    MODERATE = "MODERATE"  # Requires careful slicing
    SHALLOW = "SHALLOW"  # Very limited liquidity
    CRITICAL = "CRITICAL"  # Dangerous to trade


@dataclass
class LiquidityProfile:
    """Comprehensive liquidity analysis result."""

    symbol: str
    timestamp: datetime

    # Volume metrics
    total_bid_volume: Decimal
    total_ask_volume: Decimal
    imbalance_ratio: Decimal  # (ask - bid) / (ask + bid)

    # Depth metrics (volume to move price by X%)
    bid_depth_0_5pct: Decimal
    bid_depth_1pct: Decimal
    bid_depth_2pct: Decimal
    ask_depth_0_5pct: Decimal
    ask_depth_1pct: Decimal
    ask_depth_2pct: Decimal

    # Spread metrics
    best_bid: Decimal
    best_ask: Decimal
    spread_absolute: Decimal
    spread_percent: Decimal

    # Liquidity classification
    liquidity_level: LiquidityLevel

    # Slicing recommendations
    optimal_slice_count: int
    max_safe_order_size: Decimal
    expected_slippage_1x: Decimal  # For 1x max_safe_order_size
    expected_slippage_2x: Decimal  # For 2x max_safe_order_size

    # Risk indicators
    concentration_risk: Decimal  # How much liquidity is at best prices
    depth_consistency: Decimal  # How evenly distributed is liquidity


class OrderBookAnalyzer:
    """
    Analyzes order book depth and liquidity for optimal order execution.

    Provides intelligent slicing recommendations based on available liquidity,
    minimizing market impact while ensuring execution efficiency.
    """

    # Configuration constants
    MIN_SLICES = 3
    MAX_SLICES = 10
    LIQUIDITY_CACHE_TTL = timedelta(seconds=5)

    # Liquidity thresholds (as percentage of total volume)
    DEEP_LIQUIDITY_THRESHOLD = Decimal("0.01")  # Order < 1% of volume
    MODERATE_LIQUIDITY_THRESHOLD = Decimal("0.05")  # Order < 5% of volume
    SHALLOW_LIQUIDITY_THRESHOLD = Decimal("0.10")  # Order < 10% of volume

    def __init__(self):
        """Initialize the order book analyzer."""
        self._cache: dict[str, tuple[LiquidityProfile, datetime]] = {}
        logger.info("Order book analyzer initialized")

    def analyze_liquidity_depth(
        self, order_book: OrderBook, side: OrderSide | None = None
    ) -> LiquidityProfile:
        """
        Perform comprehensive liquidity analysis on order book.

        Args:
            order_book: Current order book snapshot
            side: Optional side to focus analysis on

        Returns:
            Detailed liquidity profile
        """
        symbol = order_book.symbol

        # Check cache
        if symbol in self._cache:
            cached_profile, cache_time = self._cache[symbol]
            if datetime.now() - cache_time < self.LIQUIDITY_CACHE_TTL:
                logger.debug("Using cached liquidity profile", symbol=symbol)
                return cached_profile

        # Calculate basic metrics
        best_bid = (
            Decimal(str(order_book.bids[0][0])) if order_book.bids else Decimal("0")
        )
        best_ask = (
            Decimal(str(order_book.asks[0][0])) if order_book.asks else Decimal("0")
        )
        spread_absolute = best_ask - best_bid
        spread_percent = (
            (spread_absolute / best_bid * Decimal("100"))
            if best_bid > 0
            else Decimal("0")
        )

        # Calculate volume metrics
        total_bid_volume = self._calculate_total_volume(order_book.bids, best_bid)
        total_ask_volume = self._calculate_total_volume(order_book.asks, best_ask)

        imbalance_ratio = self._calculate_imbalance_ratio(
            total_bid_volume, total_ask_volume
        )

        # Calculate depth metrics
        bid_depths = self._calculate_depth_levels(
            order_book.bids, best_bid, is_bid=True
        )
        ask_depths = self._calculate_depth_levels(
            order_book.asks, best_ask, is_bid=False
        )

        # Analyze liquidity distribution
        concentration_risk = self._calculate_concentration_risk(
            order_book.bids, order_book.asks
        )

        depth_consistency = self._calculate_depth_consistency(
            order_book.bids, order_book.asks
        )

        # Classify liquidity level
        liquidity_level = self._classify_liquidity_level(
            total_bid_volume, total_ask_volume, spread_percent, depth_consistency
        )

        # Calculate safe order size
        max_safe_order_size = self._calculate_max_safe_order_size(
            bid_depths[1] if side == OrderSide.SELL else ask_depths[1], liquidity_level
        )

        # Calculate slicing recommendations based on liquidity
        if liquidity_level == LiquidityLevel.DEEP:
            optimal_slice_count = self.MIN_SLICES
        elif liquidity_level == LiquidityLevel.MODERATE:
            optimal_slice_count = 5
        elif liquidity_level == LiquidityLevel.SHALLOW:
            optimal_slice_count = 7
        else:  # CRITICAL
            optimal_slice_count = self.MAX_SLICES

        # Estimate slippage
        expected_slippage_1x = self._estimate_slippage(
            max_safe_order_size,
            order_book.asks if side == OrderSide.BUY else order_book.bids,
            best_ask if side == OrderSide.BUY else best_bid,
        )

        expected_slippage_2x = self._estimate_slippage(
            max_safe_order_size * Decimal("2"),
            order_book.asks if side == OrderSide.BUY else order_book.bids,
            best_ask if side == OrderSide.BUY else best_bid,
        )

        # Create profile
        profile = LiquidityProfile(
            symbol=symbol,
            timestamp=datetime.now(),
            total_bid_volume=total_bid_volume,
            total_ask_volume=total_ask_volume,
            imbalance_ratio=imbalance_ratio,
            bid_depth_0_5pct=bid_depths[0],
            bid_depth_1pct=bid_depths[1],
            bid_depth_2pct=bid_depths[2],
            ask_depth_0_5pct=ask_depths[0],
            ask_depth_1pct=ask_depths[1],
            ask_depth_2pct=ask_depths[2],
            best_bid=best_bid,
            best_ask=best_ask,
            spread_absolute=spread_absolute,
            spread_percent=spread_percent,
            liquidity_level=liquidity_level,
            optimal_slice_count=optimal_slice_count,
            max_safe_order_size=max_safe_order_size,
            expected_slippage_1x=expected_slippage_1x,
            expected_slippage_2x=expected_slippage_2x,
            concentration_risk=concentration_risk,
            depth_consistency=depth_consistency,
        )

        # Cache the profile
        self._cache[symbol] = (profile, datetime.now())

        logger.info(
            "Liquidity analysis complete",
            symbol=symbol,
            liquidity_level=liquidity_level.value,
            optimal_slices=optimal_slice_count,
            max_safe_size=str(max_safe_order_size),
        )

        return profile

    def calculate_optimal_slice_count(
        self, order_value: Decimal, liquidity_profile: LiquidityProfile
    ) -> int:
        """
        Calculate optimal number of slices for an order.

        Args:
            order_value: Total order value in USDT
            liquidity_profile: Current liquidity analysis

        Returns:
            Optimal number of slices (minimum 3)
        """
        # Quick return for small orders
        if order_value <= liquidity_profile.max_safe_order_size:
            return self.MIN_SLICES

        # Calculate based on order size relative to safe size
        size_ratio = order_value / liquidity_profile.max_safe_order_size

        # Adjust based on liquidity level
        if liquidity_profile.liquidity_level == LiquidityLevel.DEEP:
            # Deep liquidity: fewer slices needed
            slices = self.MIN_SLICES + int(size_ratio - 1)
        elif liquidity_profile.liquidity_level == LiquidityLevel.MODERATE:
            # Moderate liquidity: standard slicing
            slices = self.MIN_SLICES + int(size_ratio * 1.5)
        elif liquidity_profile.liquidity_level == LiquidityLevel.SHALLOW:
            # Shallow liquidity: more slices needed
            slices = self.MIN_SLICES + int(size_ratio * 2)
        else:  # CRITICAL
            # Critical liquidity: maximum slicing
            slices = self.MIN_SLICES + int(size_ratio * 3)

        # Apply bounds
        slices = max(self.MIN_SLICES, min(slices, self.MAX_SLICES))

        # Adjust for depth consistency
        if liquidity_profile.depth_consistency < Decimal("0.5"):
            # Poor consistency: add more slices
            slices = min(slices + 1, self.MAX_SLICES)

        logger.debug(
            "Calculated optimal slice count",
            order_value=str(order_value),
            safe_size=str(liquidity_profile.max_safe_order_size),
            liquidity_level=liquidity_profile.liquidity_level.value,
            slices=slices,
        )

        return slices

    def _calculate_total_volume(
        self, levels: list[list[float]], reference_price: Decimal
    ) -> Decimal:
        """Calculate total volume in USDT."""
        total = Decimal("0")
        for price, quantity in levels:
            total += Decimal(str(price)) * Decimal(str(quantity))
        return total

    def _calculate_imbalance_ratio(
        self, bid_volume: Decimal, ask_volume: Decimal
    ) -> Decimal:
        """Calculate order book imbalance ratio."""
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return Decimal("0")

        imbalance = (ask_volume - bid_volume) / total_volume
        return imbalance.quantize(Decimal("0.0001"))

    def _calculate_depth_levels(
        self, levels: list[list[float]], reference_price: Decimal, is_bid: bool
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Calculate volume to move price by 0.5%, 1%, and 2%."""
        if not levels or reference_price == 0:
            return (Decimal("0"), Decimal("0"), Decimal("0"))

        # Calculate target prices
        if is_bid:
            target_0_5 = reference_price * Decimal("0.995")
            target_1 = reference_price * Decimal("0.99")
            target_2 = reference_price * Decimal("0.98")
        else:
            target_0_5 = reference_price * Decimal("1.005")
            target_1 = reference_price * Decimal("1.01")
            target_2 = reference_price * Decimal("1.02")

        depth_0_5 = Decimal("0")
        depth_1 = Decimal("0")
        depth_2 = Decimal("0")

        for price, quantity in levels:
            price_decimal = Decimal(str(price))
            volume = price_decimal * Decimal(str(quantity))

            if is_bid:
                if price_decimal >= target_0_5:
                    depth_0_5 += volume
                if price_decimal >= target_1:
                    depth_1 += volume
                if price_decimal >= target_2:
                    depth_2 += volume
            else:
                if price_decimal <= target_0_5:
                    depth_0_5 += volume
                if price_decimal <= target_1:
                    depth_1 += volume
                if price_decimal <= target_2:
                    depth_2 += volume

        return (depth_0_5, depth_1, depth_2)

    def _calculate_concentration_risk(
        self, bids: list[list[float]], asks: list[list[float]]
    ) -> Decimal:
        """Calculate concentration risk (liquidity at best prices)."""
        if not bids or not asks:
            return Decimal("1")  # Maximum risk

        # Calculate volume at best 3 levels
        best_bid_volume = sum(Decimal(str(p)) * Decimal(str(q)) for p, q in bids[:3])
        best_ask_volume = sum(Decimal(str(p)) * Decimal(str(q)) for p, q in asks[:3])

        total_bid_volume = sum(Decimal(str(p)) * Decimal(str(q)) for p, q in bids)
        total_ask_volume = sum(Decimal(str(p)) * Decimal(str(q)) for p, q in asks)

        total_volume = total_bid_volume + total_ask_volume
        if total_volume == 0:
            return Decimal("1")

        concentration = (best_bid_volume + best_ask_volume) / total_volume
        return concentration.quantize(Decimal("0.0001"))

    def _calculate_depth_consistency(
        self, bids: list[list[float]], asks: list[list[float]]
    ) -> Decimal:
        """Calculate how evenly distributed liquidity is."""
        if len(bids) < 5 or len(asks) < 5:
            return Decimal("0")  # Poor consistency

        # Calculate volume variance across levels
        bid_volumes = [Decimal(str(p)) * Decimal(str(q)) for p, q in bids[:10]]
        ask_volumes = [Decimal(str(p)) * Decimal(str(q)) for p, q in asks[:10]]

        all_volumes = bid_volumes + ask_volumes
        if not all_volumes:
            return Decimal("0")

        avg_volume = sum(all_volumes) / len(all_volumes)
        if avg_volume == 0:
            return Decimal("0")

        # Calculate coefficient of variation
        variance = sum((v - avg_volume) ** 2 for v in all_volumes) / len(all_volumes)
        std_dev = variance.sqrt()
        cv = std_dev / avg_volume

        # Convert to consistency score (0-1, higher is better)
        consistency = max(Decimal("0"), Decimal("1") - cv)
        return consistency.quantize(Decimal("0.0001"))

    def _classify_liquidity_level(
        self,
        bid_volume: Decimal,
        ask_volume: Decimal,
        spread_percent: Decimal,
        depth_consistency: Decimal,
    ) -> LiquidityLevel:
        """Classify overall liquidity level."""
        # Check for critical conditions
        if bid_volume == 0 or ask_volume == 0:
            return LiquidityLevel.CRITICAL

        if spread_percent > Decimal("1"):
            return LiquidityLevel.CRITICAL

        # Calculate average volume
        avg_volume = (bid_volume + ask_volume) / Decimal("2")

        # Classify based on volume and consistency
        if depth_consistency > Decimal("0.7") and avg_volume > Decimal("100000"):
            return LiquidityLevel.DEEP
        elif depth_consistency > Decimal("0.5") and avg_volume > Decimal("50000"):
            return LiquidityLevel.MODERATE
        elif depth_consistency > Decimal("0.3") and avg_volume > Decimal("10000"):
            return LiquidityLevel.SHALLOW
        else:
            return LiquidityLevel.CRITICAL

    def _calculate_max_safe_order_size(
        self, depth_1pct: Decimal, liquidity_level: LiquidityLevel
    ) -> Decimal:
        """Calculate maximum safe order size."""
        if liquidity_level == LiquidityLevel.DEEP:
            # Can safely use 50% of 1% depth
            return depth_1pct * Decimal("0.5")
        elif liquidity_level == LiquidityLevel.MODERATE:
            # Use 30% of 1% depth
            return depth_1pct * Decimal("0.3")
        elif liquidity_level == LiquidityLevel.SHALLOW:
            # Use 20% of 1% depth
            return depth_1pct * Decimal("0.2")
        else:  # CRITICAL
            # Use only 10% of 1% depth
            return depth_1pct * Decimal("0.1")

    def _estimate_slippage(
        self, order_size: Decimal, levels: list[list[float]], start_price: Decimal
    ) -> Decimal:
        """Estimate slippage for a given order size."""
        if not levels or order_size == 0 or start_price == 0:
            return Decimal("0")

        remaining_size = order_size
        total_cost = Decimal("0")
        total_quantity = Decimal("0")

        for price, quantity in levels:
            price_decimal = Decimal(str(price))
            quantity_decimal = Decimal(str(quantity))
            level_value = price_decimal * quantity_decimal

            if level_value >= remaining_size:
                # This level can fulfill the remainder
                needed_quantity = remaining_size / price_decimal
                total_cost += remaining_size
                total_quantity += needed_quantity
                break
            else:
                # Consume entire level
                total_cost += level_value
                total_quantity += quantity_decimal
                remaining_size -= level_value

        if total_quantity == 0:
            return Decimal("100")  # Maximum slippage

        avg_price = total_cost / total_quantity
        slippage = abs((avg_price - start_price) / start_price * Decimal("100"))

        return slippage.quantize(Decimal("0.0001"))

    def clear_cache(self, symbol: str | None = None) -> None:
        """
        Clear cached liquidity profiles.

        Args:
            symbol: Optional specific symbol to clear, otherwise clear all
        """
        if symbol:
            if symbol in self._cache:
                del self._cache[symbol]
                logger.debug("Cleared cache for symbol", symbol=symbol)
        else:
            self._cache.clear()
            logger.debug("Cleared all cached profiles")
