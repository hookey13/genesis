"""Inventory risk management for market making strategies."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class InventoryZone(Enum):
    """Inventory risk zones."""

    GREEN = "GREEN"  # Safe zone: 0-30%
    YELLOW = "YELLOW"  # Caution zone: 30-70%
    RED = "RED"  # Danger zone: 70-100%


@dataclass
class InventoryPosition:
    """Track inventory position for a symbol."""

    symbol: str
    quantity: Decimal
    average_price: Decimal
    dollar_value: Decimal
    last_update: datetime
    max_position_size: Decimal

    @property
    def utilization_pct(self) -> Decimal:
        """Calculate position utilization percentage."""
        if self.max_position_size <= 0:
            return Decimal("0")
        return (
            abs(self.quantity * self.average_price)
            / self.max_position_size
            * Decimal("100")
        )

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == 0


@dataclass
class InventoryMetrics:
    """Inventory risk and performance metrics."""

    total_inventory_value: Decimal
    total_max_position: Decimal
    utilization_pct: Decimal
    zone: InventoryZone
    skew: Decimal
    positions_count: int
    long_positions: int
    short_positions: int
    largest_position_pct: Decimal
    concentration_risk: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class InventoryLimits:
    """Risk limits for inventory management."""

    max_position_size_usdt: Decimal = Decimal("10000")
    max_total_inventory_usdt: Decimal = Decimal("50000")
    max_position_pct: Decimal = Decimal("0.10")  # 10% of capital
    max_concentration_pct: Decimal = Decimal("0.30")  # 30% in one symbol

    # Zone thresholds
    green_zone_max: Decimal = Decimal("0.30")  # 0-30%
    yellow_zone_max: Decimal = Decimal("0.70")  # 30-70%

    # Skew limits
    max_skew: Decimal = Decimal("0.50")  # Maximum 50% skew
    target_skew: Decimal = Decimal("0")  # Target neutral

    # Time limits
    max_position_age_hours: int = 24  # Maximum position age
    stale_position_hours: int = 4  # Position considered stale


class InventoryManager:
    """Manage inventory positions and risk for market making."""

    def __init__(self, limits: Optional[InventoryLimits] = None):
        """Initialize inventory manager with risk limits."""
        self.limits = limits or InventoryLimits()
        self.positions: Dict[str, InventoryPosition] = {}
        self.position_history: List[Tuple[datetime, InventoryMetrics]] = []
        self.pnl_history: List[Tuple[datetime, Decimal]] = []

        # Risk tracking
        self.current_zone = InventoryZone.GREEN
        self.zone_changes: List[Tuple[datetime, InventoryZone, InventoryZone]] = []

        # Performance tracking
        self.total_pnl = Decimal("0")
        self.realized_pnl = Decimal("0")
        self.unrealized_pnl = Decimal("0")

    def update_position(
        self,
        symbol: str,
        quantity_change: Decimal,
        price: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> InventoryPosition:
        """
        Update inventory position for a symbol.

        Args:
            symbol: Trading symbol
            quantity_change: Change in quantity (positive for buy, negative for sell)
            price: Execution price
            timestamp: Time of update

        Returns:
            Updated inventory position
        """
        timestamp = timestamp or datetime.now(UTC)

        if symbol not in self.positions:
            # New position
            self.positions[symbol] = InventoryPosition(
                symbol=symbol,
                quantity=quantity_change,
                average_price=price,
                dollar_value=quantity_change * price,
                last_update=timestamp,
                max_position_size=self.limits.max_position_size_usdt,
            )
        else:
            # Update existing position
            pos = self.positions[symbol]
            old_value = pos.quantity * pos.average_price
            new_value = quantity_change * price

            # Update quantity
            new_quantity = pos.quantity + quantity_change

            # Calculate new average price (weighted average)
            if new_quantity != 0:
                if (pos.quantity > 0 and quantity_change > 0) or (
                    pos.quantity < 0 and quantity_change < 0
                ):
                    # Adding to position
                    new_avg_price = (old_value + new_value) / new_quantity
                else:
                    # Reducing position or crossing zero
                    if abs(new_quantity) > 0:
                        new_avg_price = price
                    else:
                        new_avg_price = Decimal("0")
            else:
                new_avg_price = Decimal("0")

            # Calculate realized PnL if reducing position
            if (pos.quantity > 0 and quantity_change < 0) or (
                pos.quantity < 0 and quantity_change > 0
            ):
                # Closing or reducing position
                closed_qty = min(abs(quantity_change), abs(pos.quantity))
                if pos.quantity > 0:
                    # Long position being reduced
                    realized = closed_qty * (price - pos.average_price)
                else:
                    # Short position being reduced
                    realized = closed_qty * (pos.average_price - price)

                self.realized_pnl += realized
                self.total_pnl += realized

                logger.info(
                    "Position reduced",
                    symbol=symbol,
                    closed_qty=float(closed_qty),
                    realized_pnl=float(realized),
                )

            # Update position
            pos.quantity = new_quantity
            pos.average_price = (
                new_avg_price if new_avg_price > 0 else pos.average_price
            )
            pos.dollar_value = new_quantity * pos.average_price
            pos.last_update = timestamp

        # Update metrics
        self._update_metrics()

        return self.positions[symbol]

    def get_position(self, symbol: str) -> Optional[InventoryPosition]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)

    def get_total_inventory_value(self) -> Decimal:
        """Calculate total inventory value across all positions."""
        return sum(abs(pos.dollar_value) for pos in self.positions.values())

    def calculate_skew(self) -> Decimal:
        """
        Calculate overall inventory skew.

        Returns:
            Skew from -1 (fully short) to 1 (fully long)
        """
        long_value = sum(
            pos.dollar_value for pos in self.positions.values() if pos.is_long
        )
        short_value = abs(
            sum(pos.dollar_value for pos in self.positions.values() if pos.is_short)
        )

        total_value = long_value + short_value
        if total_value == 0:
            return Decimal("0")

        skew = (long_value - short_value) / total_value
        return skew

    def get_inventory_zone(self) -> InventoryZone:
        """Determine current inventory zone based on utilization."""
        total_value = self.get_total_inventory_value()
        utilization = total_value / self.limits.max_total_inventory_usdt

        if utilization <= self.limits.green_zone_max:
            return InventoryZone.GREEN
        elif utilization <= self.limits.yellow_zone_max:
            return InventoryZone.YELLOW
        else:
            return InventoryZone.RED

    def get_position_adjustments(self, symbol: str) -> Dict[str, Decimal]:
        """
        Get recommended position adjustments for a symbol.

        Returns:
            Dictionary with adjustment recommendations
        """
        pos = self.positions.get(symbol)
        if not pos:
            return {
                "size_multiplier": Decimal("1.0"),
                "skew_adjustment": Decimal("0"),
                "urgency": Decimal("0"),
            }

        zone = self.get_inventory_zone()
        skew = self.calculate_skew()
        utilization = pos.utilization_pct / Decimal("100")

        adjustments = {
            "size_multiplier": Decimal("1.0"),
            "skew_adjustment": Decimal("0"),
            "urgency": Decimal("0"),
        }

        # Zone-based adjustments
        if zone == InventoryZone.YELLOW:
            # Reduce new position sizes
            adjustments["size_multiplier"] = Decimal("0.5")
            adjustments["urgency"] = Decimal("0.5")
        elif zone == InventoryZone.RED:
            # Only allow position reduction
            adjustments["size_multiplier"] = Decimal("0")
            adjustments["urgency"] = Decimal("1.0")

        # Skew-based adjustments (in basis points)
        if abs(skew) > Decimal("0.3"):
            # Significant skew, adjust prices to reduce it
            adjustments["skew_adjustment"] = skew * Decimal("20")  # Max 20 bps

        # Position age adjustments
        age_hours = (datetime.now(UTC) - pos.last_update).total_seconds() / 3600
        if age_hours > self.limits.stale_position_hours:
            adjustments["urgency"] = min(
                adjustments["urgency"] + Decimal("0.2"), Decimal("1.0")
            )

        return adjustments

    def should_accept_order(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal
    ) -> Tuple[bool, str]:
        """
        Check if an order should be accepted based on inventory limits.

        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            quantity: Order quantity
            price: Order price

        Returns:
            Tuple of (should_accept, reason)
        """
        # Calculate potential new position
        quantity_change = quantity if side == "BUY" else -quantity
        potential_value = abs(quantity * price)

        # Get current position
        current_pos = self.positions.get(symbol)
        current_value = abs(current_pos.dollar_value) if current_pos else Decimal("0")

        # Check single position limit
        new_position_value = current_value + potential_value
        if new_position_value > self.limits.max_position_size_usdt:
            return (
                False,
                f"Would exceed position limit ({float(new_position_value)} > {float(self.limits.max_position_size_usdt)})",
            )

        # Check total inventory limit
        total_inventory = self.get_total_inventory_value()
        new_total = total_inventory + potential_value
        if new_total > self.limits.max_total_inventory_usdt:
            return (
                False,
                f"Would exceed total inventory limit ({float(new_total)} > {float(self.limits.max_total_inventory_usdt)})",
            )

        # Check concentration risk
        concentration = (
            new_position_value / new_total if new_total > 0 else Decimal("0")
        )
        if concentration > self.limits.max_concentration_pct:
            return (
                False,
                f"Would exceed concentration limit ({float(concentration * 100)}% > {float(self.limits.max_concentration_pct * 100)}%)",
            )

        # Check if we're in RED zone
        zone = self.get_inventory_zone()
        if zone == InventoryZone.RED:
            # Only allow position reduction in RED zone
            if current_pos:
                if (current_pos.is_long and side == "BUY") or (
                    current_pos.is_short and side == "SELL"
                ):
                    return False, "In RED zone, only position reduction allowed"

        return True, "OK"

    def get_exit_signals(self) -> List[Dict]:
        """
        Generate exit signals for positions that need to be reduced.

        Returns:
            List of exit signal dictionaries
        """
        signals = []
        zone = self.get_inventory_zone()

        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue

            # Check if position is too old
            age_hours = (datetime.now(UTC) - pos.last_update).total_seconds() / 3600
            should_exit = False
            urgency = Decimal("0.5")

            if zone == InventoryZone.RED:
                should_exit = True
                urgency = Decimal("1.0")
            elif age_hours > self.limits.max_position_age_hours:
                should_exit = True
                urgency = Decimal("0.8")
            elif zone == InventoryZone.YELLOW and pos.utilization_pct > 50:
                should_exit = True
                urgency = Decimal("0.6")

            if should_exit:
                signals.append(
                    {
                        "symbol": symbol,
                        "side": "SELL" if pos.is_long else "BUY",
                        "quantity": abs(pos.quantity)
                        * urgency,  # Partial exit based on urgency
                        "urgency": float(urgency),
                        "reason": f"Zone: {zone.value}, Age: {age_hours:.1f}h",
                    }
                )

        return signals

    def _update_metrics(self) -> None:
        """Update inventory metrics and check zone changes."""
        # Calculate current metrics
        metrics = InventoryMetrics(
            total_inventory_value=self.get_total_inventory_value(),
            total_max_position=self.limits.max_total_inventory_usdt,
            utilization_pct=(
                self.get_total_inventory_value()
                / self.limits.max_total_inventory_usdt
                * Decimal("100")
                if self.limits.max_total_inventory_usdt > 0
                else Decimal("0")
            ),
            zone=self.get_inventory_zone(),
            skew=self.calculate_skew(),
            positions_count=len(self.positions),
            long_positions=sum(1 for p in self.positions.values() if p.is_long),
            short_positions=sum(1 for p in self.positions.values() if p.is_short),
            largest_position_pct=max(
                (p.utilization_pct for p in self.positions.values()),
                default=Decimal("0"),
            ),
            concentration_risk=self._calculate_concentration_risk(),
        )

        # Check for zone change
        old_zone = self.current_zone
        new_zone = metrics.zone
        if old_zone != new_zone:
            self.zone_changes.append((datetime.now(UTC), old_zone, new_zone))
            self.current_zone = new_zone

            logger.warning(
                "Inventory zone changed",
                old_zone=old_zone.value,
                new_zone=new_zone.value,
                utilization=float(metrics.utilization_pct),
            )

        # Store metrics history
        self.position_history.append((datetime.now(UTC), metrics))

        # Clean old history (keep last 24 hours)
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        self.position_history = [(t, m) for t, m in self.position_history if t > cutoff]

    def _calculate_concentration_risk(self) -> Decimal:
        """Calculate concentration risk (Herfindahl index)."""
        total = self.get_total_inventory_value()
        if total == 0:
            return Decimal("0")

        hhi = sum(
            (abs(pos.dollar_value) / total) ** 2 for pos in self.positions.values()
        )

        return hhi

    def get_metrics_summary(self) -> Dict:
        """Get summary of current inventory metrics."""
        metrics = InventoryMetrics(
            total_inventory_value=self.get_total_inventory_value(),
            total_max_position=self.limits.max_total_inventory_usdt,
            utilization_pct=(
                self.get_total_inventory_value()
                / self.limits.max_total_inventory_usdt
                * Decimal("100")
                if self.limits.max_total_inventory_usdt > 0
                else Decimal("0")
            ),
            zone=self.get_inventory_zone(),
            skew=self.calculate_skew(),
            positions_count=len(self.positions),
            long_positions=sum(1 for p in self.positions.values() if p.is_long),
            short_positions=sum(1 for p in self.positions.values() if p.is_short),
            largest_position_pct=max(
                (p.utilization_pct for p in self.positions.values()),
                default=Decimal("0"),
            ),
            concentration_risk=self._calculate_concentration_risk(),
        )

        return {
            "total_inventory_value": float(metrics.total_inventory_value),
            "utilization_pct": float(metrics.utilization_pct),
            "zone": metrics.zone.value,
            "skew": float(metrics.skew),
            "positions_count": metrics.positions_count,
            "long_positions": metrics.long_positions,
            "short_positions": metrics.short_positions,
            "largest_position_pct": float(metrics.largest_position_pct),
            "concentration_risk": float(metrics.concentration_risk),
            "total_pnl": float(self.total_pnl),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl),
        }
