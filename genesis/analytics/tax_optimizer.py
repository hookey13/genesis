"""
Tax lot optimization module for Project GENESIS.

Provides FIFO/LIFO/HIFO lot selection methods, tax-aware position closing,
and year-end tax reporting for institutional-grade tax optimization.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import structlog

from genesis.core.constants import TradingTier
from genesis.core.models import Position, TaxMethod
from genesis.data.repository import Repository
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class TaxOptimizer:
    """Optimizes tax treatment of trades through lot selection strategies."""

    def __init__(self, repository: Repository):
        """Initialize TaxOptimizer."""
        self.repository = repository
        logger.info("tax_optimizer_initialized")

    @requires_tier(TradingTier.STRATEGIST)
    async def select_tax_lots(
        self,
        account_id: str,
        symbol: str,
        quantity: Decimal,
        method: TaxMethod = TaxMethod.FIFO,
    ) -> List[Dict[str, Any]]:
        """
        Select tax lots for position closure.
        
        Args:
            account_id: Account ID
            symbol: Symbol to close
            quantity: Quantity to close
            method: Tax lot selection method
            
        Returns:
            List of selected tax lots with quantities
        """
        try:
            # Get all positions for symbol
            positions = await self.repository.get_positions_by_account(
                account_id, status="open"
            )
            symbol_positions = [
                p for p in positions 
                if p.symbol == symbol and p.tax_lot_id is not None
            ]
            
            if not symbol_positions:
                return []
            
            # Sort based on method
            if method == TaxMethod.FIFO:
                # First In First Out - oldest first
                symbol_positions.sort(key=lambda p: p.acquisition_date or p.created_at)
            elif method == TaxMethod.LIFO:
                # Last In First Out - newest first
                symbol_positions.sort(
                    key=lambda p: p.acquisition_date or p.created_at, 
                    reverse=True
                )
            elif method == TaxMethod.HIFO:
                # Highest In First Out - highest cost basis first (best for tax)
                symbol_positions.sort(
                    key=lambda p: p.cost_basis or p.entry_price,
                    reverse=True
                )
            
            # Select lots
            selected_lots = []
            remaining_quantity = quantity
            
            for position in symbol_positions:
                if remaining_quantity <= 0:
                    break
                
                lot_quantity = min(position.quantity, remaining_quantity)
                selected_lots.append({
                    "tax_lot_id": position.tax_lot_id,
                    "position_id": position.position_id,
                    "quantity": str(lot_quantity),
                    "cost_basis": str(position.cost_basis or position.entry_price),
                    "acquisition_date": (
                        position.acquisition_date.isoformat() 
                        if position.acquisition_date else position.created_at.isoformat()
                    ),
                })
                
                remaining_quantity -= lot_quantity
            
            logger.info(
                "tax_lots_selected",
                account_id=account_id,
                symbol=symbol,
                method=method.value,
                lots_selected=len(selected_lots),
            )
            
            return selected_lots
            
        except Exception as e:
            logger.error("tax_lot_selection_failed", error=str(e))
            raise

    @requires_tier(TradingTier.STRATEGIST)
    async def calculate_tax_impact(
        self,
        position: Position,
        exit_price: Decimal,
        tax_rate: Decimal = Decimal("0.20"),
    ) -> Dict[str, Decimal]:
        """
        Calculate tax impact of closing position.
        
        Args:
            position: Position to close
            exit_price: Exit price
            tax_rate: Applicable tax rate
            
        Returns:
            Tax impact calculations
        """
        cost_basis = position.cost_basis or position.entry_price
        proceeds = exit_price * position.quantity
        gain_loss = proceeds - (cost_basis * position.quantity)
        
        # Check holding period for long/short term
        if position.acquisition_date:
            holding_period = datetime.now(timezone.utc) - position.acquisition_date
            is_long_term = holding_period.days > 365
        else:
            holding_period = datetime.now(timezone.utc) - position.created_at
            is_long_term = holding_period.days > 365
        
        # Apply different rates for long/short term
        if is_long_term:
            effective_rate = tax_rate * Decimal("0.5")  # Simplified long-term rate
        else:
            effective_rate = tax_rate
        
        tax_liability = max(Decimal("0"), gain_loss * effective_rate)
        
        return {
            "proceeds": proceeds,
            "cost_basis": cost_basis * position.quantity,
            "gain_loss": gain_loss,
            "is_long_term": is_long_term,
            "holding_days": holding_period.days,
            "tax_rate": effective_rate,
            "tax_liability": tax_liability,
            "net_proceeds": proceeds - tax_liability,
        }

    @requires_tier(TradingTier.STRATEGIST)
    async def generate_tax_report(
        self,
        account_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Generate year-end tax report.
        
        Args:
            account_id: Account ID
            year: Tax year
            
        Returns:
            Tax report with gains/losses
        """
        start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        
        # Get all closed positions for the year
        trades = await self.repository.get_trades_by_account(
            account_id, start_date, end_date
        )
        
        # Calculate totals
        short_term_gains = Decimal("0")
        short_term_losses = Decimal("0")
        long_term_gains = Decimal("0")
        long_term_losses = Decimal("0")
        
        trade_details = []
        
        for trade in trades:
            # Determine if long or short term based on timestamps
            if hasattr(trade, "position_id"):
                position = await self.repository.get_position(trade.position_id)
                if position and position.acquisition_date:
                    holding_period = trade.timestamp - position.acquisition_date
                    is_long_term = holding_period.days > 365
                else:
                    is_long_term = False
            else:
                is_long_term = False
            
            pnl = trade.pnl_dollars
            
            if is_long_term:
                if pnl > 0:
                    long_term_gains += pnl
                else:
                    long_term_losses += abs(pnl)
            else:
                if pnl > 0:
                    short_term_gains += pnl
                else:
                    short_term_losses += abs(pnl)
            
            trade_details.append({
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "quantity": str(trade.quantity),
                "entry_price": str(trade.entry_price),
                "exit_price": str(trade.exit_price),
                "pnl": str(pnl),
                "is_long_term": is_long_term,
                "date": trade.timestamp.isoformat(),
            })
        
        # Calculate net amounts
        net_short_term = short_term_gains - short_term_losses
        net_long_term = long_term_gains - long_term_losses
        total_net = net_short_term + net_long_term
        
        report = {
            "report_id": str(uuid4()),
            "account_id": account_id,
            "tax_year": year,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "short_term_gains": str(short_term_gains),
                "short_term_losses": str(short_term_losses),
                "net_short_term": str(net_short_term),
                "long_term_gains": str(long_term_gains),
                "long_term_losses": str(long_term_losses),
                "net_long_term": str(net_long_term),
                "total_net_gain_loss": str(total_net),
            },
            "trade_count": len(trades),
            "trades": trade_details,
        }
        
        logger.info(
            "tax_report_generated",
            account_id=account_id,
            year=year,
            total_net=str(total_net),
        )
        
        return report

    @requires_tier(TradingTier.STRATEGIST)
    async def optimize_year_end_positions(
        self,
        account_id: str,
        target_net_gain: Decimal = Decimal("0"),
    ) -> List[Dict[str, Any]]:
        """
        Suggest positions to close for tax optimization.
        
        Args:
            account_id: Account ID
            target_net_gain: Target net gain for year
            
        Returns:
            List of suggested position closures
        """
        # Get current year's realized gains
        current_year = datetime.now(timezone.utc).year
        tax_report = await self.generate_tax_report(account_id, current_year)
        current_net = Decimal(tax_report["summary"]["total_net_gain_loss"])
        
        # Get open positions
        positions = await self.repository.get_positions_by_account(
            account_id, status="open"
        )
        
        suggestions = []
        
        # If we have gains, look for losses to harvest
        if current_net > target_net_gain:
            loss_positions = [
                p for p in positions 
                if p.pnl_dollars < 0
            ]
            loss_positions.sort(key=lambda p: p.pnl_dollars)
            
            for position in loss_positions:
                suggestions.append({
                    "position_id": position.position_id,
                    "symbol": position.symbol,
                    "current_pnl": str(position.pnl_dollars),
                    "action": "harvest_loss",
                    "tax_benefit": str(abs(position.pnl_dollars) * Decimal("0.20")),
                })
        
        # If we have losses, look for gains to realize
        elif current_net < target_net_gain:
            gain_positions = [
                p for p in positions
                if p.pnl_dollars > 0
            ]
            gain_positions.sort(key=lambda p: p.pnl_dollars, reverse=True)
            
            for position in gain_positions:
                # Check if long-term for better tax treatment
                if position.acquisition_date:
                    holding_period = datetime.now(timezone.utc) - position.acquisition_date
                    is_long_term = holding_period.days > 365
                else:
                    is_long_term = False
                
                if is_long_term:
                    suggestions.append({
                        "position_id": position.position_id,
                        "symbol": position.symbol,
                        "current_pnl": str(position.pnl_dollars),
                        "action": "realize_long_term_gain",
                        "tax_impact": str(position.pnl_dollars * Decimal("0.10")),
                    })
        
        logger.info(
            "year_end_optimization_complete",
            account_id=account_id,
            suggestions_count=len(suggestions),
        )
        
        return suggestions