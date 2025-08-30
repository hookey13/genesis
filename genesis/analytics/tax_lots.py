"""Tax lot tracking system with FIFO/LIFO calculations."""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)

CostBasisMethod = Literal["FIFO", "LIFO", "HIFO", "SPECIFIC"]


class LotStatus(Enum):
    """Tax lot status."""
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"


@dataclass
class TaxLot:
    """Tax lot for tracking cost basis."""

    lot_id: str
    symbol: str
    quantity: Decimal
    remaining_quantity: Decimal
    cost_per_unit: Decimal
    acquired_at: datetime
    account_id: str
    order_id: str
    status: LotStatus = LotStatus.OPEN
    closed_quantity: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def total_cost(self) -> Decimal:
        """Calculate total cost of the lot."""
        return self.quantity * self.cost_per_unit

    @property
    def remaining_cost(self) -> Decimal:
        """Calculate remaining cost basis."""
        return self.remaining_quantity * self.cost_per_unit

    def close_partial(self, quantity: Decimal, sale_price: Decimal) -> Decimal:
        """Close partial quantity from this lot."""
        if quantity > self.remaining_quantity:
            raise ValueError(
                f"Cannot close {quantity} from lot with {self.remaining_quantity} remaining"
            )

        # Calculate P&L for this portion
        cost_basis = quantity * self.cost_per_unit
        sale_proceeds = quantity * sale_price
        pnl = sale_proceeds - cost_basis

        # Update lot
        self.remaining_quantity -= quantity
        self.closed_quantity += quantity
        self.realized_pnl += pnl

        # Update status
        if self.remaining_quantity == Decimal("0"):
            self.status = LotStatus.CLOSED
        else:
            self.status = LotStatus.PARTIAL

        return pnl


@dataclass
class LotAssignment:
    """Assignment of a sale to specific tax lots."""

    sale_id: str
    symbol: str
    sale_quantity: Decimal
    sale_price: Decimal
    sale_date: datetime
    lot_assignments: list[tuple[str, Decimal, Decimal]] = field(default_factory=list)
    total_cost_basis: Decimal = Decimal("0")
    total_proceeds: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    method_used: CostBasisMethod | None = None

    def add_lot_assignment(
        self,
        lot_id: str,
        quantity: Decimal,
        cost_per_unit: Decimal
    ) -> None:
        """Add a lot assignment to this sale."""
        self.lot_assignments.append((lot_id, quantity, cost_per_unit))
        cost_basis = quantity * cost_per_unit
        self.total_cost_basis += cost_basis

    def calculate_pnl(self) -> None:
        """Calculate realized P&L."""
        self.total_proceeds = self.sale_quantity * self.sale_price
        self.realized_pnl = self.total_proceeds - self.total_cost_basis


class TaxLotTracker:
    """Tax lot tracking system for cost basis calculation."""

    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.lots: dict[str, list[TaxLot]] = {}  # symbol -> list of lots
        self.lot_by_id: dict[str, TaxLot] = {}  # lot_id -> lot
        self.assignments: list[LotAssignment] = []

    def add_lot(self, lot: TaxLot) -> None:
        """Add a new tax lot."""
        if lot.symbol not in self.lots:
            self.lots[lot.symbol] = []

        self.lots[lot.symbol].append(lot)
        self.lot_by_id[lot.lot_id] = lot

        self.logger.info(
            "tax_lot_added",
            lot_id=lot.lot_id,
            symbol=lot.symbol,
            quantity=str(lot.quantity),
            cost_per_unit=str(lot.cost_per_unit)
        )

    def get_open_lots(
        self,
        symbol: str,
        method: CostBasisMethod = "FIFO"
    ) -> list[TaxLot]:
        """Get open lots for a symbol, sorted by method."""
        if symbol not in self.lots:
            return []

        open_lots = [
            lot for lot in self.lots[symbol]
            if lot.status != LotStatus.CLOSED
        ]

        if method == "FIFO":
            # First In First Out - oldest first
            return sorted(open_lots, key=lambda x: x.acquired_at)
        elif method == "LIFO":
            # Last In First Out - newest first
            return sorted(open_lots, key=lambda x: x.acquired_at, reverse=True)
        elif method == "HIFO":
            # Highest In First Out - highest cost first
            return sorted(open_lots, key=lambda x: x.cost_per_unit, reverse=True)
        else:
            return open_lots

    def process_sale(
        self,
        sale_id: str,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        sale_date: datetime,
        method: CostBasisMethod = "FIFO"
    ) -> LotAssignment:
        """Process a sale and assign to tax lots."""
        assignment = LotAssignment(
            sale_id=sale_id,
            symbol=symbol,
            sale_quantity=quantity,
            sale_price=price,
            sale_date=sale_date,
            method_used=method
        )

        remaining_quantity = quantity
        open_lots = self.get_open_lots(symbol, method)

        for lot in open_lots:
            if remaining_quantity <= Decimal("0"):
                break

            # Determine how much to take from this lot
            take_quantity = min(remaining_quantity, lot.remaining_quantity)

            # Close partial lot and get P&L
            lot_pnl = lot.close_partial(take_quantity, price)

            # Record assignment
            assignment.add_lot_assignment(
                lot.lot_id,
                take_quantity,
                lot.cost_per_unit
            )

            remaining_quantity -= take_quantity

            self.logger.info(
                "lot_assigned",
                sale_id=sale_id,
                lot_id=lot.lot_id,
                quantity=str(take_quantity),
                pnl=str(lot_pnl)
            )

        if remaining_quantity > Decimal("0"):
            self.logger.warning(
                "insufficient_lots",
                sale_id=sale_id,
                symbol=symbol,
                unassigned_quantity=str(remaining_quantity)
            )

        # Calculate total P&L
        assignment.calculate_pnl()
        self.assignments.append(assignment)

        self.logger.info(
            "sale_processed",
            sale_id=sale_id,
            symbol=symbol,
            quantity=str(quantity),
            realized_pnl=str(assignment.realized_pnl),
            method=method
        )

        return assignment

    def get_position_summary(self, symbol: str) -> dict[str, Any]:
        """Get summary of position including unrealized P&L."""
        if symbol not in self.lots:
            return {
                "symbol": symbol,
                "total_quantity": "0",
                "average_cost": "0",
                "total_cost_basis": "0",
                "realized_pnl": "0",
                "open_lots": 0
            }

        symbol_lots = self.lots[symbol]
        open_lots = [lot for lot in symbol_lots if lot.status != LotStatus.CLOSED]

        total_quantity = sum(lot.remaining_quantity for lot in open_lots)
        total_cost = sum(lot.remaining_cost for lot in open_lots)
        average_cost = total_cost / total_quantity if total_quantity > 0 else Decimal("0")

        realized_pnl = sum(lot.realized_pnl for lot in symbol_lots)

        return {
            "symbol": symbol,
            "total_quantity": str(total_quantity),
            "average_cost": str(average_cost),
            "total_cost_basis": str(total_cost),
            "realized_pnl": str(realized_pnl),
            "open_lots": len(open_lots)
        }

    def calculate_unrealized_pnl(
        self,
        symbol: str,
        current_price: Decimal
    ) -> Decimal:
        """Calculate unrealized P&L for a position."""
        if symbol not in self.lots:
            return Decimal("0")

        open_lots = self.get_open_lots(symbol)

        unrealized_pnl = Decimal("0")
        for lot in open_lots:
            market_value = lot.remaining_quantity * current_price
            cost_basis = lot.remaining_cost
            unrealized_pnl += market_value - cost_basis

        return unrealized_pnl

    def get_tax_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> dict[str, Any]:
        """Generate tax report for date range."""
        period_assignments = [
            a for a in self.assignments
            if start_date <= a.sale_date <= end_date
        ]

        total_proceeds = sum(a.total_proceeds for a in period_assignments)
        total_cost_basis = sum(a.total_cost_basis for a in period_assignments)
        total_realized_pnl = sum(a.realized_pnl for a in period_assignments)

        # Group by symbol
        by_symbol = {}
        for assignment in period_assignments:
            if assignment.symbol not in by_symbol:
                by_symbol[assignment.symbol] = {
                    "proceeds": Decimal("0"),
                    "cost_basis": Decimal("0"),
                    "realized_pnl": Decimal("0"),
                    "num_sales": 0
                }

            by_symbol[assignment.symbol]["proceeds"] += assignment.total_proceeds
            by_symbol[assignment.symbol]["cost_basis"] += assignment.total_cost_basis
            by_symbol[assignment.symbol]["realized_pnl"] += assignment.realized_pnl
            by_symbol[assignment.symbol]["num_sales"] += 1

        # Convert Decimals to strings
        by_symbol_str = {
            symbol: {
                "proceeds": str(data["proceeds"]),
                "cost_basis": str(data["cost_basis"]),
                "realized_pnl": str(data["realized_pnl"]),
                "num_sales": data["num_sales"]
            }
            for symbol, data in by_symbol.items()
        }

        return {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_proceeds": str(total_proceeds),
            "total_cost_basis": str(total_cost_basis),
            "total_realized_pnl": str(total_realized_pnl),
            "num_sales": len(period_assignments),
            "by_symbol": by_symbol_str
        }

    def get_lot_details(self, lot_id: str) -> dict[str, Any] | None:
        """Get detailed information about a specific lot."""
        if lot_id not in self.lot_by_id:
            return None

        lot = self.lot_by_id[lot_id]

        return {
            "lot_id": lot.lot_id,
            "symbol": lot.symbol,
            "quantity": str(lot.quantity),
            "remaining_quantity": str(lot.remaining_quantity),
            "closed_quantity": str(lot.closed_quantity),
            "cost_per_unit": str(lot.cost_per_unit),
            "total_cost": str(lot.total_cost),
            "remaining_cost": str(lot.remaining_cost),
            "acquired_at": lot.acquired_at.isoformat(),
            "account_id": lot.account_id,
            "order_id": lot.order_id,
            "status": lot.status.value,
            "realized_pnl": str(lot.realized_pnl)
        }
