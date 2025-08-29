"""
Domain services for complex business logic.

PositionService handles FIFO position management and PnL calculation.
All values normalized to quote currency (USDT).
"""

import logging
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy.orm import Session as DBSession

from genesis.data.models import PnLLedger, Position, Trade
from genesis.data.repositories import PositionRepository

logger = logging.getLogger(__name__)


class PositionService:
    """
    Service for position management and PnL calculation.

    Uses FIFO (First In First Out) for position averaging
    and realizes PnL on reducing/closing trades.
    """

    def __init__(self, db_session: DBSession):
        self.db_session = db_session
        self.position_repo = PositionRepository(db_session)

    def apply_trade(self, trade: Trade) -> tuple[Position, Decimal]:
        """
        Apply trade to position and calculate realized PnL.

        Args:
            trade: Trade to apply

        Returns:
            Tuple of (updated position, realized PnL in quote currency)
        """
        position = self.position_repo.get_or_create(trade.symbol)

        # Determine if trade increases or reduces position
        trade_qty_signed = trade.qty if trade.side == "buy" else -trade.qty
        old_qty_signed = position.qty
        new_qty_signed = old_qty_signed + trade_qty_signed

        realized_pnl = Decimal("0")
        new_avg_price = position.avg_entry_price

        # Case 1: Opening or increasing position (same direction)
        if self._same_sign(old_qty_signed, trade_qty_signed):
            # Weighted average for new position price
            if abs(new_qty_signed) > 0:
                old_value = abs(old_qty_signed) * position.avg_entry_price
                new_value = trade.qty * trade.price
                new_avg_price = (old_value + new_value) / abs(new_qty_signed)
            logger.info(
                f"Position increased: {trade.symbol} {old_qty_signed} -> {new_qty_signed}"
            )

        # Case 2: Reducing position (opposite direction)
        elif abs(old_qty_signed) > abs(trade_qty_signed):
            # Partial close - realize PnL on closed portion
            closed_qty = trade.qty
            if old_qty_signed > 0:  # Long position being reduced by sell
                realized_pnl = closed_qty * (trade.price - position.avg_entry_price)
            else:  # Short position being reduced by buy
                realized_pnl = closed_qty * (position.avg_entry_price - trade.price)

            # Position avg price unchanged (FIFO)
            logger.info(
                f"Position reduced: {trade.symbol} {old_qty_signed} -> {new_qty_signed}, PnL={realized_pnl}"
            )

        # Case 3: Closing and reversing position
        else:
            # First close existing position
            closed_qty = abs(old_qty_signed)
            if old_qty_signed > 0:  # Long closed by sell
                realized_pnl = closed_qty * (trade.price - position.avg_entry_price)
            elif old_qty_signed < 0:  # Short closed by buy
                realized_pnl = closed_qty * (position.avg_entry_price - trade.price)

            # Then open new position with remaining quantity
            remaining_qty = trade.qty - closed_qty
            if remaining_qty > 0:
                new_avg_price = trade.price

            logger.info(
                f"Position reversed: {trade.symbol} {old_qty_signed} -> {new_qty_signed}, PnL={realized_pnl}"
            )

        # Subtract fees from PnL (always a cost)
        realized_pnl -= trade.fee_amount

        # Update position
        self.position_repo.update(
            symbol=trade.symbol,
            qty=new_qty_signed,
            avg_entry_price=new_avg_price if new_qty_signed != 0 else Decimal("0"),
            realised_pnl_delta=realized_pnl,
        )

        # Return updated position
        return self.position_repo.get_or_create(trade.symbol), realized_pnl

    def calculate_unrealized_pnl(self, symbol: str, current_price: Decimal) -> Decimal:
        """
        Calculate unrealized PnL for a position.

        Args:
            symbol: Symbol to calculate for
            current_price: Current market price

        Returns:
            Unrealized PnL in quote currency
        """
        position = self.position_repo.get_or_create(symbol)

        if position.qty == 0:
            return Decimal("0")

        if position.qty > 0:  # Long position
            return position.qty * (current_price - position.avg_entry_price)
        else:  # Short position
            return abs(position.qty) * (position.avg_entry_price - current_price)

    def get_position_value(self, symbol: str, current_price: Decimal) -> Decimal:
        """Get current position value in quote currency."""
        position = self.position_repo.get_or_create(symbol)
        return abs(position.qty) * current_price

    def close_all_positions(
        self, session_id: str, prices: dict[str, Decimal]
    ) -> Decimal:
        """
        Mark all positions closed at given prices and calculate total PnL.

        Args:
            session_id: Trading session ID
            prices: Current prices for each symbol

        Returns:
            Total realized PnL from closing all positions
        """
        total_pnl = Decimal("0")

        for symbol, price in prices.items():
            position = self.position_repo.get_or_create(symbol)

            if position.qty != 0:
                # Calculate PnL from closing
                if position.qty > 0:
                    pnl = position.qty * (price - position.avg_entry_price)
                else:
                    pnl = abs(position.qty) * (position.avg_entry_price - price)

                # Update position to flat
                self.position_repo.update(
                    symbol=symbol,
                    qty=Decimal("0"),
                    avg_entry_price=Decimal("0"),
                    realised_pnl_delta=pnl,
                )

                # Record in ledger
                ledger_entry = PnLLedger(
                    session_id=session_id,
                    event_type="adjustment",
                    symbol=symbol,
                    amount_quote=pnl,
                    ref_type=None,
                    ref_id=None,
                )
                self.db_session.add(ledger_entry)

                total_pnl += pnl
                logger.info(f"Closed position {symbol}: PnL={pnl}")

        self.db_session.commit()
        return total_pnl

    def _same_sign(self, a: Decimal, b: Decimal) -> bool:
        """Check if two numbers have the same sign."""
        return (a >= 0 and b >= 0) or (a < 0 and b < 0)

    def get_position_summary(
        self, symbol: str, current_price: Decimal | None = None
    ) -> dict:
        """
        Get comprehensive position summary.

        Returns:
            Dict with position details including realized and unrealized PnL
        """
        position = self.position_repo.get_or_create(symbol)

        summary = {
            "symbol": symbol,
            "quantity": position.qty,
            "avg_entry_price": position.avg_entry_price,
            "realized_pnl": position.realised_pnl,
            "unrealized_pnl": Decimal("0"),
            "total_pnl": position.realised_pnl,
            "position_value": Decimal("0"),
            "is_long": position.qty > 0,
            "is_short": position.qty < 0,
            "is_flat": position.qty == 0,
        }

        if current_price and position.qty != 0:
            summary["unrealized_pnl"] = self.calculate_unrealized_pnl(
                symbol, current_price
            )
            summary["total_pnl"] = summary["realized_pnl"] + summary["unrealized_pnl"]
            summary["position_value"] = abs(position.qty) * current_price

        return summary


class RiskService:
    """Service for risk calculations and position sizing."""

    @staticmethod
    def calculate_position_size(
        account_balance: Decimal,
        risk_percent: Decimal,
        stop_distance: Decimal,
        price: Decimal,
        lot_step: Decimal,
    ) -> Decimal:
        """
        Calculate position size based on risk parameters.

        Args:
            account_balance: Total account balance in quote currency
            risk_percent: Percentage of account to risk (e.g., 2 for 2%)
            stop_distance: Distance to stop loss in price units
            price: Current asset price
            lot_step: Minimum lot size increment

        Returns:
            Position size rounded to lot_step
        """
        if stop_distance <= 0:
            raise ValueError("Stop distance must be positive")

        # Calculate risk amount in quote currency
        risk_amount = account_balance * (risk_percent / Decimal("100"))

        # Calculate position size
        position_size = risk_amount / stop_distance

        # Round to lot step
        if lot_step > 0:
            position_size = (position_size / lot_step).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            ) * lot_step

        return position_size

    @staticmethod
    def calculate_kelly_fraction(
        win_probability: Decimal,
        avg_win: Decimal,
        avg_loss: Decimal,
        kelly_multiplier: Decimal = Decimal("0.25"),
    ) -> Decimal:
        """
        Calculate Kelly Criterion fraction for position sizing.

        Args:
            win_probability: Probability of winning (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount (positive value)
            kelly_multiplier: Safety multiplier (e.g., 0.25 for quarter Kelly)

        Returns:
            Fraction of capital to risk
        """
        if avg_loss <= 0:
            return Decimal("0")

        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        q = Decimal("1") - win_probability
        b = avg_win / avg_loss

        kelly_fraction = (win_probability * b - q) / b

        # Apply safety multiplier and bounds
        kelly_fraction = kelly_fraction * kelly_multiplier
        kelly_fraction = max(Decimal("0"), min(kelly_fraction, Decimal("0.25")))

        return kelly_fraction
