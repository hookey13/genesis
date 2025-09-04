"""Virtual Portfolio for paper trading tracking."""

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Position:
    """Represents a position in the virtual portfolio."""

    symbol: str
    quantity: Decimal
    average_price: Decimal
    side: str
    opened_at: datetime
    last_price: Decimal | None = None
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")


    def update_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L based on current price.

        Args:
            current_price: Current market price
        """
        self.last_price = current_price
        if self.side == "long":
            self.unrealized_pnl = (current_price - self.average_price) * self.quantity
        else:
            self.unrealized_pnl = (self.average_price - current_price) * self.quantity


@dataclass
class Trade:
    """Represents a completed trade."""

    order_id: str
    timestamp: datetime
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    value: Decimal
    pnl: Decimal
    balance_before: Decimal
    balance_after: Decimal


class VirtualPortfolio:
    """Virtual portfolio for tracking paper trading performance."""

    def __init__(self, strategy_id: str, initial_balance: Decimal):
        """Initialize virtual portfolio.

        Args:
            strategy_id: Unique identifier for the strategy
            initial_balance: Initial balance for the portfolio
        """
        self.strategy_id = strategy_id
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.balance = initial_balance  # Add balance property for compatibility
        self.positions: dict[str, Position] = {}
        self.trades: list[dict[str, Any]] = []
        self.daily_returns: deque = deque(maxlen=365)
        self.peak_balance = initial_balance
        self.max_drawdown = Decimal("0")
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = Decimal("0")
        self.total_loss = Decimal("0")
        self.start_time = datetime.now()
        self.last_trade_time = None
        self._lock = asyncio.Lock()

    async def process_fill(self, order: Any) -> None:
        """Process a filled order and update portfolio.

        Args:
            order: Filled order object
        """
        async with self._lock:
            trade_value = order.filled_quantity * order.average_fill_price

            trade = {
                "order_id": order.order_id,
                "timestamp": order.fill_timestamp,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": str(order.filled_quantity),
                "price": str(order.average_fill_price),
                "value": str(trade_value),
                "balance_before": str(self.current_balance),
            }

            if order.side == "buy":
                await self._process_buy(order)
            else:
                await self._process_sell(order)

            trade["balance_after"] = str(self.current_balance)
            trade["pnl"] = str(
                Decimal(trade["balance_after"]) - Decimal(trade["balance_before"])
            )

            self.trades.append(trade)
            self.total_trades += 1
            self.last_trade_time = order.fill_timestamp

            self._update_performance_metrics(trade)

            logger.info(
                "Trade processed in virtual portfolio",
                strategy_id=self.strategy_id,
                order_id=order.order_id,
                pnl=trade["pnl"],
            )

    async def _process_buy(self, order: Any) -> None:
        """Process a buy order.

        Args:
            order: Buy order to process
        """
        symbol = order.symbol
        quantity = order.filled_quantity
        price = order.average_fill_price

        if symbol in self.positions:
            position = self.positions[symbol]
            total_quantity = position.quantity + quantity
            total_value = (position.quantity * position.average_price) + (
                quantity * price
            )
            position.average_price = total_value / total_quantity
            position.quantity = total_quantity
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                side="long",
                opened_at=order.fill_timestamp,
            )

        self.current_balance -= quantity * price

    async def _process_sell(self, order: Any) -> None:
        """Process a sell order.

        Args:
            order: Sell order to process
        """
        symbol = order.symbol
        quantity = order.filled_quantity
        price = order.average_fill_price

        if symbol in self.positions:
            position = self.positions[symbol]

            if position.quantity >= quantity:
                realized_pnl = (price - position.average_price) * quantity
                position.realized_pnl += realized_pnl
                position.quantity -= quantity

                if position.quantity == Decimal("0"):
                    del self.positions[symbol]

                self.current_balance += quantity * price

                if realized_pnl > 0:
                    self.winning_trades += 1
                    self.total_profit += realized_pnl
                else:
                    self.losing_trades += 1
                    self.total_loss += abs(realized_pnl)
            else:
                logger.warning(
                    "Insufficient position for sell order",
                    symbol=symbol,
                    position_qty=str(position.quantity),
                    order_qty=str(quantity),
                )
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                side="short",
                opened_at=order.fill_timestamp,
            )
            self.current_balance += quantity * price

    def _update_performance_metrics(self, trade: dict[str, Any]) -> None:
        """Update performance metrics after a trade.

        Args:
            trade: Trade details
        """
        _ = trade  # Will be used for detailed metrics in future
        total_value = self.current_balance + sum(
            pos.quantity * (pos.last_price or pos.average_price)
            for pos in self.positions.values()
        )

        if total_value > self.peak_balance:
            self.peak_balance = total_value

        drawdown = (self.peak_balance - total_value) / self.peak_balance
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        daily_return = (total_value - self.initial_balance) / self.initial_balance
        self.daily_returns.append(float(daily_return))

    def get_metrics(self) -> dict[str, Any]:
        """Get portfolio performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        total_value = self.current_balance + sum(
            pos.quantity * (pos.last_price or pos.average_price)
            for pos in self.positions.values()
        )

        total_pnl = total_value - self.initial_balance
        pnl_percentage = (total_pnl / self.initial_balance) * Decimal("100")

        win_rate = (
            Decimal(str(self.winning_trades / self.total_trades))
            if self.total_trades > 0
            else Decimal("0")
        )

        profit_factor = (
            self.total_profit / self.total_loss if self.total_loss > 0 else Decimal("0")
        )

        sharpe_ratio = self._calculate_sharpe_ratio()

        days_running = (datetime.now() - self.start_time).days or 1

        return {
            "strategy_id": self.strategy_id,
            "initial_balance": str(self.initial_balance),
            "current_balance": str(self.current_balance),
            "total_value": str(total_value),
            "total_pnl": str(total_pnl),
            "pnl_percentage": float(pnl_percentage),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": float(self.max_drawdown),
            "days_running": days_running,
            "positions": len(self.positions),
            "last_trade_time": (
                self.last_trade_time.isoformat() if self.last_trade_time else None
            ),
            "start_time": self.start_time.isoformat(),
        }

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from daily returns.

        Returns:
            Sharpe ratio
        """
        if len(self.daily_returns) < 2:
            return 0.0

        returns_array = np.array(self.daily_returns)

        if returns_array.std() == 0:
            return 0.0

        risk_free_rate = 0.02 / 365
        excess_returns = returns_array - risk_free_rate

        sharpe = np.sqrt(365) * excess_returns.mean() / returns_array.std()

        return float(sharpe)

    async def update_market_prices(self, prices: dict[str, Decimal]) -> None:
        """Update market prices for positions.

        Args:
            prices: Dictionary of symbol to current price
        """
        async with self._lock:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].update_pnl(price)

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position object or None
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all positions.

        Returns:
            Dictionary of all positions
        """
        return self.positions.copy()

    async def reset(self) -> None:
        """Reset portfolio to initial state."""
        async with self._lock:
            self.current_balance = self.initial_balance
            self.positions.clear()
            self.trades.clear()
            self.daily_returns.clear()
            self.peak_balance = self.initial_balance
            self.max_drawdown = Decimal("0")
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_profit = Decimal("0")
            self.total_loss = Decimal("0")
            self.start_time = datetime.now()
            self.last_trade_time = None

            logger.info("Virtual portfolio reset", strategy_id=self.strategy_id)
