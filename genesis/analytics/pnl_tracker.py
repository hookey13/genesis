"""
P&L tracking and aggregation system.

Provides accurate P&L calculation for positions and sessions,
with precision to 2 decimal places as required.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import structlog

from genesis.core.models import Position, PositionSide, TradingSession
from genesis.data.models_db import TradingSessionDB, get_session

logger = structlog.get_logger(__name__)


@dataclass
class PnLSnapshot:
    """Point-in-time P&L snapshot."""
    
    timestamp: datetime
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    open_positions: int
    closed_positions: int
    win_rate: Decimal
    average_win: Decimal
    average_loss: Decimal
    max_drawdown: Decimal
    current_balance: Decimal


class PnLTracker:
    """
    Tracks and calculates P&L with 2 decimal place accuracy.
    
    Handles both realized and unrealized P&L for positions and sessions.
    """
    
    def __init__(self, session_id: str | None = None):
        """
        Initialize P&L tracker.
        
        Args:
            session_id: Optional session ID for database storage
        """
        self.session_id = session_id
        self.db_session = get_session() if session_id else None
        
        # In-memory tracking
        self.positions: dict[str, Position] = {}
        self.closed_positions: list[Position] = []
        self.realized_pnl = Decimal("0")
        self.fees_paid = Decimal("0")
        
        # Metrics
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.max_drawdown = Decimal("0")
        self.peak_balance = Decimal("0")
        
    def add_position(self, position: Position) -> None:
        """
        Add a position to track.
        
        Args:
            position: Position to track
        """
        self.positions[position.position_id] = position
        logger.debug(
            "Position added to P&L tracker",
            position_id=position.position_id,
            symbol=position.symbol,
            entry_price=str(position.entry_price),
        )
        
    def update_position_price(self, position_id: str, current_price: Decimal) -> Decimal:
        """
        Update position with current market price.
        
        Args:
            position_id: Position to update
            current_price: Current market price
            
        Returns:
            Updated unrealized P&L (2 decimal precision)
        """
        if position_id not in self.positions:
            logger.warning("Position not found for P&L update", position_id=position_id)
            return Decimal("0")
            
        position = self.positions[position_id]
        position.update_pnl(current_price)
        
        # Round to 2 decimal places
        pnl_rounded = position.pnl_dollars.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        position.pnl_dollars = pnl_rounded
        
        logger.debug(
            "Position P&L updated",
            position_id=position_id,
            current_price=str(current_price),
            pnl=str(pnl_rounded),
        )
        
        return pnl_rounded
        
    def close_position(
        self,
        position_id: str,
        exit_price: Decimal,
        fees: Decimal = Decimal("0"),
        close_reason: str = "manual",
    ) -> dict[str, Any]:
        """
        Close a position and calculate realized P&L.
        
        Args:
            position_id: Position to close
            exit_price: Exit price
            fees: Trading fees incurred
            close_reason: Reason for closing
            
        Returns:
            Dict with P&L details
        """
        if position_id not in self.positions:
            logger.warning("Position not found for closing", position_id=position_id)
            return {"error": "Position not found"}
            
        position = self.positions[position_id]
        
        # Calculate final P&L
        if position.side == PositionSide.LONG:
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:  # SHORT
            gross_pnl = (position.entry_price - exit_price) * position.quantity
            
        # Apply fees
        net_pnl = gross_pnl - fees
        
        # Round to 2 decimal places
        net_pnl_rounded = net_pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
        # Update position
        position.pnl_dollars = net_pnl_rounded
        position.close_reason = close_reason
        position.updated_at = datetime.now()
        
        # Update tracking
        self.realized_pnl += net_pnl_rounded
        self.fees_paid += fees
        self.total_trades += 1
        
        if net_pnl_rounded > 0:
            self.winning_trades += 1
        elif net_pnl_rounded < 0:
            self.losing_trades += 1
            
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_id]
        
        # Update database if session exists
        if self.db_session and self.session_id:
            self._update_session_metrics()
            
        logger.info(
            "Position closed",
            position_id=position_id,
            symbol=position.symbol,
            entry_price=str(position.entry_price),
            exit_price=str(exit_price),
            gross_pnl=str(gross_pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
            fees=str(fees),
            net_pnl=str(net_pnl_rounded),
            close_reason=close_reason,
        )
        
        return {
            "position_id": position_id,
            "symbol": position.symbol,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.quantity,
            "gross_pnl": gross_pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            "fees": fees,
            "net_pnl": net_pnl_rounded,
            "close_reason": close_reason,
        }
        
    def get_unrealized_pnl(self) -> Decimal:
        """
        Calculate total unrealized P&L for open positions.
        
        Returns:
            Total unrealized P&L (2 decimal precision)
        """
        total_unrealized = sum(
            position.pnl_dollars for position in self.positions.values()
        )
        return total_unrealized.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
    def get_total_pnl(self) -> Decimal:
        """
        Calculate total P&L (realized + unrealized).
        
        Returns:
            Total P&L (2 decimal precision)
        """
        total = self.realized_pnl + self.get_unrealized_pnl()
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
    def get_win_rate(self) -> Decimal:
        """
        Calculate win rate percentage.
        
        Returns:
            Win rate as percentage (2 decimal precision)
        """
        if self.total_trades == 0:
            return Decimal("0")
            
        win_rate = (Decimal(self.winning_trades) / Decimal(self.total_trades)) * Decimal("100")
        return win_rate.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        
    def calculate_drawdown(self, current_balance: Decimal) -> Decimal:
        """
        Calculate maximum drawdown.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            Maximum drawdown (2 decimal precision)
        """
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            
        if self.peak_balance > 0:
            drawdown = ((self.peak_balance - current_balance) / self.peak_balance) * Decimal("100")
            drawdown_rounded = drawdown.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            
            if drawdown_rounded > self.max_drawdown:
                self.max_drawdown = drawdown_rounded
                
            return drawdown_rounded
            
        return Decimal("0")
        
    def get_average_win_loss(self) -> dict[str, Decimal]:
        """
        Calculate average win and average loss.
        
        Returns:
            Dict with average_win and average_loss (2 decimal precision)
        """
        wins = [p.pnl_dollars for p in self.closed_positions if p.pnl_dollars > 0]
        losses = [p.pnl_dollars for p in self.closed_positions if p.pnl_dollars < 0]
        
        avg_win = Decimal("0")
        avg_loss = Decimal("0")
        
        if wins:
            avg_win = (sum(wins) / len(wins)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            
        if losses:
            avg_loss = (sum(losses) / len(losses)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            
        return {"average_win": avg_win, "average_loss": avg_loss}
        
    def get_snapshot(self, current_balance: Decimal) -> PnLSnapshot:
        """
        Get current P&L snapshot.
        
        Args:
            current_balance: Current account balance
            
        Returns:
            P&L snapshot with all metrics
        """
        avg_metrics = self.get_average_win_loss()
        
        return PnLSnapshot(
            timestamp=datetime.now(),
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.get_unrealized_pnl(),
            total_pnl=self.get_total_pnl(),
            open_positions=len(self.positions),
            closed_positions=len(self.closed_positions),
            win_rate=self.get_win_rate(),
            average_win=avg_metrics["average_win"],
            average_loss=avg_metrics["average_loss"],
            max_drawdown=self.calculate_drawdown(current_balance),
            current_balance=current_balance,
        )
        
    def _update_session_metrics(self) -> None:
        """Update session metrics in database."""
        if not self.db_session or not self.session_id:
            return
            
        try:
            session = (
                self.db_session.query(TradingSessionDB)
                .filter_by(session_id=self.session_id)
                .first()
            )
            
            if session:
                session.realized_pnl = str(self.realized_pnl)
                session.total_trades = self.total_trades
                session.winning_trades = self.winning_trades
                session.losing_trades = self.losing_trades
                session.win_rate = str(self.get_win_rate())
                session.max_drawdown = str(self.max_drawdown)
                
                self.db_session.commit()
                
        except Exception as e:
            logger.error("Failed to update session metrics", error=str(e))
            self.db_session.rollback()
            
    def export_metrics(self) -> dict[str, Any]:
        """
        Export all P&L metrics.
        
        Returns:
            Complete metrics dictionary
        """
        avg_metrics = self.get_average_win_loss()
        
        return {
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.get_unrealized_pnl()),
            "total_pnl": str(self.get_total_pnl()),
            "fees_paid": str(self.fees_paid),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": str(self.get_win_rate()),
            "average_win": str(avg_metrics["average_win"]),
            "average_loss": str(avg_metrics["average_loss"]),
            "max_drawdown": str(self.max_drawdown),
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
        }