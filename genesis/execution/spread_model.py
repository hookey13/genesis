"""Dynamic spread calculation model for market making."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SpreadFactors:
    """Factors affecting spread calculation."""

    base_spread: Decimal
    volatility_factor: Decimal
    inventory_factor: Decimal
    competition_factor: Decimal
    adverse_selection_factor: Decimal
    fee_adjustment: Decimal

    def calculate_total(self) -> Decimal:
        """Calculate total spread considering all factors."""
        total = (
            self.base_spread
            * self.volatility_factor
            * self.inventory_factor
            * self.competition_factor
            * self.adverse_selection_factor
        )
        # Add fee adjustment (not multiplicative)
        total += self.fee_adjustment
        return total


@dataclass
class MarketConditions:
    """Current market conditions for spread calculation."""

    current_price: Decimal
    bid_price: Decimal
    ask_price: Decimal
    volatility: Decimal
    volume_24h: Decimal
    order_book_depth: Dict[str, List[Tuple[Decimal, Decimal]]]
    recent_trades: List[Dict]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    @property
    def current_spread_bps(self) -> Decimal:
        """Calculate current market spread in basis points."""
        if self.bid_price <= 0:
            return Decimal("0")
        return (self.ask_price - self.bid_price) / self.bid_price * Decimal("10000")


class SpreadModel:
    """Dynamic spread calculation model with multi-factor adjustments."""

    def __init__(
        self,
        base_spread_bps: Decimal = Decimal("10"),
        min_spread_bps: Decimal = Decimal("5"),
        max_spread_bps: Decimal = Decimal("50"),
        maker_fee_bps: Decimal = Decimal("-2.5"),
        taker_fee_bps: Decimal = Decimal("4.5"),
        min_profit_bps: Decimal = Decimal("1"),
    ):
        """Initialize spread model with parameters."""
        self.base_spread_bps = base_spread_bps
        self.min_spread_bps = min_spread_bps
        self.max_spread_bps = max_spread_bps
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.min_profit_bps = min_profit_bps

        # Volatility parameters
        self.volatility_low_threshold = Decimal("0.001")  # 0.1%
        self.volatility_high_threshold = Decimal("0.01")  # 1%
        self.volatility_multiplier_min = Decimal("0.5")
        self.volatility_multiplier_max = Decimal("3.0")

        # Competition parameters
        self.competition_distance_bps = Decimal("2")
        self.competition_tightening_bps = Decimal("1")

        # Historical data
        self.spread_history: List[Tuple[datetime, Decimal]] = []
        self.adjustment_history: List[SpreadFactors] = []

    def calculate_spread(
        self,
        market_conditions: MarketConditions,
        inventory_skew: Decimal = Decimal("0"),
        toxic_flow_detected: bool = False,
        competitor_spread_bps: Optional[Decimal] = None,
    ) -> Tuple[Decimal, SpreadFactors]:
        """
        Calculate optimal spread based on market conditions.

        Args:
            market_conditions: Current market state
            inventory_skew: Current inventory skew (-1 to 1)
            toxic_flow_detected: Whether toxic flow is detected
            competitor_spread_bps: Competitor's spread in basis points

        Returns:
            Tuple of (spread_bps, factors used in calculation)
        """
        # Initialize factors
        factors = SpreadFactors(
            base_spread=self.base_spread_bps,
            volatility_factor=self._calculate_volatility_factor(
                market_conditions.volatility
            ),
            inventory_factor=self._calculate_inventory_factor(inventory_skew),
            competition_factor=self._calculate_competition_factor(
                market_conditions.current_spread_bps, competitor_spread_bps
            ),
            adverse_selection_factor=self._calculate_adverse_factor(
                toxic_flow_detected
            ),
            fee_adjustment=self._calculate_fee_adjustment(),
        )

        # Calculate total spread
        spread_bps = factors.calculate_total()

        # Apply bounds
        spread_bps = self._apply_bounds(spread_bps)

        # Store in history
        now = datetime.now(UTC)
        self.spread_history.append((now, spread_bps))
        self.adjustment_history.append(factors)

        # Clean old history (keep last hour)
        cutoff = now - timedelta(hours=1)
        self.spread_history = [(t, s) for t, s in self.spread_history if t > cutoff]

        logger.debug(
            "Calculated spread",
            spread_bps=float(spread_bps),
            base=float(factors.base_spread),
            volatility_factor=float(factors.volatility_factor),
            inventory_factor=float(factors.inventory_factor),
            competition_factor=float(factors.competition_factor),
            adverse_factor=float(factors.adverse_selection_factor),
        )

        return spread_bps, factors

    def calculate_quote_prices(
        self, mid_price: Decimal, spread_bps: Decimal, skew_bps: Decimal = Decimal("0")
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate bid and ask prices from mid price and spread.

        Args:
            mid_price: Current mid market price
            spread_bps: Total spread in basis points
            skew_bps: Skew adjustment in basis points (positive = higher asks)

        Returns:
            Tuple of (bid_price, ask_price)
        """
        half_spread = spread_bps / Decimal("2") / Decimal("10000")
        skew = skew_bps / Decimal("10000")

        # Apply skew to move both quotes in the same direction
        bid_price = mid_price * (Decimal("1") - half_spread - skew)
        ask_price = mid_price * (Decimal("1") + half_spread - skew)

        return bid_price, ask_price

    def calculate_layered_spreads(
        self,
        base_spread_bps: Decimal,
        num_layers: int = 3,
        layer_multiplier: Decimal = Decimal("2"),
    ) -> List[Decimal]:
        """
        Calculate spreads for multiple quote layers.

        Args:
            base_spread_bps: Base spread for first layer
            num_layers: Number of quote layers
            layer_multiplier: Multiplier for each successive layer

        Returns:
            List of spread values for each layer
        """
        spreads = []
        current_spread = base_spread_bps

        for i in range(num_layers):
            spreads.append(current_spread)
            current_spread = current_spread * layer_multiplier

        return spreads

    def _calculate_volatility_factor(self, volatility: Decimal) -> Decimal:
        """Calculate volatility adjustment factor."""
        if volatility <= self.volatility_low_threshold:
            # Low volatility: tighten spreads
            return self.volatility_multiplier_min
        elif volatility >= self.volatility_high_threshold:
            # High volatility: widen spreads
            return self.volatility_multiplier_max
        else:
            # Linear interpolation
            range_vol = self.volatility_high_threshold - self.volatility_low_threshold
            range_mult = self.volatility_multiplier_max - self.volatility_multiplier_min

            factor = self.volatility_multiplier_min + (
                (volatility - self.volatility_low_threshold) / range_vol * range_mult
            )
            return factor

    def _calculate_inventory_factor(self, inventory_skew: Decimal) -> Decimal:
        """
        Calculate inventory adjustment factor.

        Inventory skew affects spread width:
        - High absolute skew -> wider spreads (more risk)
        - Low skew -> normal spreads
        """
        abs_skew = abs(inventory_skew)

        if abs_skew < Decimal("0.3"):
            # Low inventory: normal spreads
            return Decimal("1.0")
        elif abs_skew < Decimal("0.7"):
            # Medium inventory: slightly wider
            return Decimal("1.2")
        else:
            # High inventory: much wider
            return Decimal("1.5")

    def _calculate_competition_factor(
        self, market_spread_bps: Decimal, competitor_spread_bps: Optional[Decimal]
    ) -> Decimal:
        """Calculate competition adjustment factor."""
        if competitor_spread_bps is None:
            return Decimal("1.0")

        # If competitors have tighter spreads, we need to compete
        if competitor_spread_bps < self.base_spread_bps:
            # Tighten by up to 20%
            tightening_pct = min(
                (self.base_spread_bps - competitor_spread_bps) / self.base_spread_bps,
                Decimal("0.2"),
            )
            return Decimal("1") - tightening_pct

        return Decimal("1.0")

    def _calculate_adverse_factor(self, toxic_flow_detected: bool) -> Decimal:
        """Calculate adverse selection adjustment factor."""
        if toxic_flow_detected:
            # Widen spreads significantly
            return Decimal("2.0")
        return Decimal("1.0")

    def _calculate_fee_adjustment(self) -> Decimal:
        """Calculate fee-based spread adjustment."""
        # Ensure we cover fees and make minimum profit
        # Need to cover potential taker fee on adverse selection
        fee_coverage = abs(self.maker_fee_bps) + self.min_profit_bps
        return fee_coverage

    def _apply_bounds(self, spread_bps: Decimal) -> Decimal:
        """Apply minimum and maximum spread bounds."""
        return max(self.min_spread_bps, min(spread_bps, self.max_spread_bps))

    def get_optimal_refresh_interval(
        self, volatility: Decimal, toxic_flow: bool
    ) -> int:
        """
        Calculate optimal quote refresh interval.

        Args:
            volatility: Current market volatility
            toxic_flow: Whether toxic flow is detected

        Returns:
            Refresh interval in seconds
        """
        base_interval = 5  # 5 seconds base

        if toxic_flow:
            # Faster refresh during toxic flow
            return 2

        if volatility > self.volatility_high_threshold:
            # Fast refresh in high volatility
            return 3
        elif volatility < self.volatility_low_threshold:
            # Slower refresh in low volatility
            return 10

        return base_interval

    def analyze_spread_efficiency(self) -> Dict[str, Decimal]:
        """Analyze historical spread efficiency."""
        if not self.spread_history:
            return {}

        spreads = [s for _, s in self.spread_history]

        return {
            "average_spread_bps": sum(spreads) / len(spreads),
            "min_spread_bps": min(spreads),
            "max_spread_bps": max(spreads),
            "spread_volatility": self._calculate_spread_volatility(spreads),
        }

    def _calculate_spread_volatility(self, spreads: List[Decimal]) -> Decimal:
        """Calculate volatility of spreads."""
        if len(spreads) < 2:
            return Decimal("0")

        mean = sum(spreads) / len(spreads)
        variance = sum((s - mean) ** 2 for s in spreads) / len(spreads)
        return variance.sqrt() if variance > 0 else Decimal("0")
