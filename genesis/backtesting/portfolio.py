"""
Portfolio Management for Backtesting

Tracks positions, P&L, and portfolio metrics during backtests.
"""

import copy
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any

import structlog

logger = structlog.get_logger()


@dataclass
class Position:
    """Represents a position in a single asset."""
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    realized_pnl: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    total_fees: Decimal = Decimal('0')
    entry_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, new_price: Decimal) -> None:
        """Update position with new market price."""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        self.last_update = datetime.now()
    
    def add_fill(self, fill: Any) -> None:
        """Add a fill to the position."""
        from genesis.backtesting.execution_simulator import OrderSide
        
        if fill.side == OrderSide.BUY:
            # Average up the entry price
            total_value = (self.quantity * self.entry_price) + (fill.quantity * fill.price)
            self.quantity += fill.quantity
            self.entry_price = total_value / self.quantity if self.quantity != 0 else Decimal('0')
        else:  # SELL
            # Realize P&L on the sold portion
            if self.quantity > 0:
                realized = (fill.price - self.entry_price) * min(fill.quantity, self.quantity)
                self.realized_pnl += realized
            self.quantity -= fill.quantity
        
        self.total_fees += fill.fee
        self.last_update = fill.timestamp


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""
    timestamp: datetime
    cash: Decimal
    positions: Dict[str, Position]
    total_equity: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    drawdown: Decimal
    margin_used: Decimal = Decimal('0')
    leverage: Decimal = Decimal('1')


class Portfolio:
    """
    Portfolio tracker for backtesting.
    
    Manages cash, positions, P&L, and risk metrics.
    """
    
    def __init__(
        self,
        initial_capital: Decimal = Decimal('1000'),
        enable_shorting: bool = False,
        use_leverage: bool = False,
        max_leverage: Decimal = Decimal('1.0'),
        max_position_size: Optional[Decimal] = None
    ):
        """Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            enable_shorting: Allow short positions
            use_leverage: Allow leveraged trading
            max_leverage: Maximum leverage ratio
            max_position_size: Maximum position size per symbol
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.enable_shorting = enable_shorting
        self.use_leverage = use_leverage
        self.max_leverage = max_leverage
        self.max_position_size = max_position_size
        
        # Tracking
        self.history: List[PortfolioSnapshot] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self.peak_equity = initial_capital
        self.current_drawdown = Decimal('0')
        self.max_drawdown = Decimal('0')
        
        # Metrics
        self.total_fees = Decimal('0')
        self.total_slippage = Decimal('0')
        self.trade_count = 0
        
        # Take initial snapshot
        self._take_snapshot(datetime.now())
    
    @property
    def total_equity(self) -> Decimal:
        """Calculate total portfolio value."""
        positions_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def available_capital(self) -> Decimal:
        """Calculate available capital for new trades."""
        if self.use_leverage:
            return self.cash * self.max_leverage
        return self.cash
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return sum(
            pos.unrealized_pnl 
            for pos in self.positions.values()
        )
    
    @property
    def realized_pnl(self) -> Decimal:
        """Calculate total realized P&L."""
        return sum(
            trade['pnl'] 
            for trade in self.closed_trades
        )
    
    async def mark_to_market(self, market_snapshot: Any) -> None:
        """
        Update portfolio with current market prices.
        
        Args:
            market_snapshot: Current market data
        """
        symbol = market_snapshot.symbol
        
        if symbol in self.positions:
            position = self.positions[symbol]
            position.update_price(market_snapshot.close)
            
            # Update drawdown
            current_equity = self.total_equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
                self.current_drawdown = Decimal('0')
            else:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
            
            # Take snapshot periodically (every 100th update)
            if len(self.history) % 100 == 0:
                self._take_snapshot(market_snapshot.timestamp)
    
    async def process_fill(self, fill: Any) -> bool:
        """
        Process an order fill.
        
        Args:
            fill: Fill details from execution simulator
            
        Returns:
            True if processed successfully
        """
        from genesis.backtesting.execution_simulator import OrderSide
        
        try:
            symbol = fill.symbol
            
            # Update cash
            if fill.side == OrderSide.BUY:
                cost = fill.value + fill.fee
                if cost > self.available_capital:
                    logger.error(
                        "insufficient_funds",
                        required=float(cost),
                        available=float(self.available_capital)
                    )
                    return False
                self.cash -= cost
            else:  # SELL
                self.cash += fill.value - fill.fee
            
            # Update or create position
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=Decimal('0'),
                    entry_price=fill.price,
                    current_price=fill.price,
                    entry_time=fill.timestamp
                )
            
            position = self.positions[symbol]
            old_quantity = position.quantity
            position.add_fill(fill)
            
            # Check if position closed
            if old_quantity != 0 and position.quantity == 0:
                # Position closed, record trade
                self.closed_trades.append({
                    'symbol': symbol,
                    'entry_time': position.entry_time,
                    'exit_time': fill.timestamp,
                    'entry_price': float(position.entry_price),
                    'exit_price': float(fill.price),
                    'quantity': float(old_quantity),
                    'pnl': float(position.realized_pnl),
                    'fees': float(position.total_fees),
                    'metadata': position.metadata
                })
                # Remove closed position
                del self.positions[symbol]
            
            # Check for short positions if not allowed
            if not self.enable_shorting and position.quantity < 0:
                logger.error(
                    "short_not_allowed",
                    symbol=symbol,
                    quantity=float(position.quantity)
                )
                return False
            
            # Update tracking
            self.total_fees += fill.fee
            self.total_slippage += abs(fill.slippage)
            self.trade_count += 1
            
            # Take snapshot
            self._take_snapshot(fill.timestamp)
            
            logger.info(
                "fill_processed",
                symbol=symbol,
                side=fill.side.value,
                quantity=float(fill.quantity),
                price=float(fill.price),
                cash=float(self.cash),
                equity=float(self.total_equity)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "fill_processing_failed",
                error=str(e),
                fill=fill
            )
            return False
    
    def get_position(self, symbol: str) -> Decimal:
        """Get position size for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position quantity (negative for short)
        """
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return Decimal('0')
    
    def _take_snapshot(self, timestamp: datetime) -> None:
        """Take a snapshot of current portfolio state."""
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=copy.deepcopy(self.positions),
            total_equity=self.total_equity,
            unrealized_pnl=self.unrealized_pnl,
            realized_pnl=self.realized_pnl,
            drawdown=self.current_drawdown,
            margin_used=self._calculate_margin_used(),
            leverage=self._calculate_leverage()
        )
        self.history.append(snapshot.__dict__)
    
    def _calculate_margin_used(self) -> Decimal:
        """Calculate margin used for leveraged positions."""
        if not self.use_leverage:
            return Decimal('0')
        
        total_exposure = sum(
            abs(pos.quantity * pos.current_price)
            for pos in self.positions.values()
        )
        return total_exposure / self.max_leverage if self.max_leverage > 0 else Decimal('0')
    
    def _calculate_leverage(self) -> Decimal:
        """Calculate current leverage ratio."""
        if self.total_equity == 0:
            return Decimal('0')
        
        total_exposure = sum(
            abs(pos.quantity * pos.current_price)
            for pos in self.positions.values()
        )
        return total_exposure / self.total_equity
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.history:
            return {}
        
        # Calculate returns series
        returns = []
        for i in range(1, len(self.history)):
            prev_equity = self.history[i-1]['total_equity']
            curr_equity = self.history[i]['total_equity']
            if prev_equity != 0:
                ret = (curr_equity - prev_equity) / prev_equity
                returns.append(float(ret))
        
        # Calculate Sharpe ratio (assuming 0 risk-free rate)
        if returns:
            import numpy as np
            returns_array = np.array(returns)
            avg_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            
            # Annualize (assuming minute bars, 252 trading days)
            periods_per_year = 252 * 24 * 60  # Minutes in a trading year
            sharpe = (avg_return * np.sqrt(periods_per_year)) / std_return if std_return > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns_array[returns_array < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino = (avg_return * np.sqrt(periods_per_year)) / downside_std if downside_std > 0 else 0
            
            # Calmar ratio
            annual_return = avg_return * periods_per_year
            calmar = annual_return / float(self.max_drawdown) if self.max_drawdown > 0 else 0
        else:
            sharpe = sortino = calmar = 0
        
        return {
            'initial_capital': float(self.initial_capital),
            'final_equity': float(self.total_equity),
            'total_return': float((self.total_equity - self.initial_capital) / self.initial_capital),
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_fees': float(self.total_fees),
            'total_slippage': float(self.total_slippage),
            'max_drawdown': float(self.max_drawdown),
            'current_drawdown': float(self.current_drawdown),
            'trade_count': self.trade_count,
            'closed_trades': len(self.closed_trades),
            'open_positions': len(self.positions),
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'peak_equity': float(self.peak_equity)
        }