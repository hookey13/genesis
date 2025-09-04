"""Paper Trading Simulator - Parallel execution alongside live trading."""

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

from genesis.paper_trading.persistence import PersistenceConfig, StatePersistence
from genesis.paper_trading.validation_criteria import ValidationCriteria
from genesis.paper_trading.virtual_portfolio import VirtualPortfolio

logger = structlog.get_logger(__name__)


class SimulationMode(Enum):
    """Paper trading simulation modes."""

    REALISTIC = "realistic"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"


@dataclass
class SimulationConfig:
    """Configuration for paper trading simulation."""

    mode: SimulationMode = SimulationMode.REALISTIC
    base_latency_ms: float = 10.0
    latency_std_ms: float = 2.0
    base_slippage_bps: float = 2.0
    slippage_std_bps: float = 1.0
    partial_fill_threshold: Decimal = Decimal("10000")
    max_fill_ratio: float = 0.5
    enabled: bool = True


@dataclass
class SimulatedOrder:
    """Represents a simulated order."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    price: Decimal | None
    timestamp: datetime
    status: str = "pending"
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None
    fill_timestamp: datetime | None = None
    slippage: Decimal | None = None
    latency_ms: float | None = None


class PaperTradingSimulator:
    """Main paper trading simulator for strategy validation."""

    def __init__(
        self,
        config: SimulationConfig,
        validation_criteria: ValidationCriteria,
        persistence_config: PersistenceConfig = None,
    ):
        """Initialize paper trading simulator.

        Args:
            config: Simulation configuration
            validation_criteria: Validation criteria for strategy promotion
            persistence_config: Configuration for state persistence
        """
        self.config = config
        self.validation_criteria = validation_criteria
        self.portfolios: dict[str, VirtualPortfolio] = {}
        self.orders: dict[str, SimulatedOrder] = {}
        self.order_counter = 0
        self.running = False
        self._tasks: list[asyncio.Task] = []
        self.persistence = StatePersistence(persistence_config)
        self._save_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the paper trading simulator."""
        if self.running:
            logger.warning("Paper trading simulator already running")
            return

        self.running = True

        # Start auto-save task
        if self.persistence.config.auto_save_interval_seconds > 0:
            self._save_task = asyncio.create_task(self._auto_save_loop())

        logger.info("Paper trading simulator started", config=self.config)

    async def stop(self) -> None:
        """Stop the paper trading simulator."""
        if not self.running:
            return

        self.running = False

        # Stop auto-save task
        if self._save_task:
            self._save_task.cancel()
            await asyncio.gather(self._save_task, return_exceptions=True)

        # Save final state
        await self._save_all_portfolios()

        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("Paper trading simulator stopped")

    def create_portfolio(
        self, strategy_id: str, initial_balance: Decimal
    ) -> VirtualPortfolio:
        """Create a virtual portfolio for a strategy.

        Args:
            strategy_id: Unique identifier for the strategy
            initial_balance: Initial balance for the portfolio

        Returns:
            Created virtual portfolio
        """
        if strategy_id in self.portfolios:
            logger.warning("Portfolio already exists", strategy_id=strategy_id)
            return self.portfolios[strategy_id]

        portfolio = VirtualPortfolio(strategy_id, initial_balance)
        self.portfolios[strategy_id] = portfolio

        logger.info(
            "Virtual portfolio created",
            strategy_id=strategy_id,
            initial_balance=str(initial_balance),
        )

        return portfolio

    async def submit_order(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> SimulatedOrder:
        """Submit a simulated order.

        Args:
            strategy_id: Strategy submitting the order
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit)
            quantity: Order quantity
            price: Order price (for limit orders)

        Returns:
            Simulated order object
        """
        if not self.running:
            raise RuntimeError("Paper trading simulator not running")

        if strategy_id not in self.portfolios:
            raise ValueError(f"Portfolio not found for strategy {strategy_id}")

        self.order_counter += 1
        order_id = f"SIM-{self.order_counter:08d}"

        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
        )

        self.orders[order_id] = order

        task = asyncio.create_task(self._execute_order(strategy_id, order))
        self._tasks.append(task)

        logger.info(
            "Simulated order submitted",
            order_id=order_id,
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            quantity=str(quantity),
        )

        return order

    async def _execute_order(self, strategy_id: str, order: SimulatedOrder) -> None:
        """Execute a simulated order with realistic fills.

        Args:
            strategy_id: Strategy that submitted the order
            order: Order to execute
        """
        try:
            latency_ms = self._calculate_latency()
            await asyncio.sleep(latency_ms / 1000.0)

            order.latency_ms = latency_ms

            slippage = self._calculate_slippage()
            order.slippage = slippage

            if order.quantity > self.config.partial_fill_threshold:
                fill_ratio = min(random.uniform(0.3, 1.0), self.config.max_fill_ratio)
                order.filled_quantity = order.quantity * Decimal(str(fill_ratio))
            else:
                order.filled_quantity = order.quantity

            if order.order_type == "market":
                order.average_fill_price = self._apply_slippage(
                    order.price or Decimal("1"), slippage, order.side
                )
            else:
                order.average_fill_price = order.price

            order.status = "filled"
            order.fill_timestamp = datetime.now()

            portfolio = self.portfolios[strategy_id]
            await portfolio.process_fill(order)

            # Save order to persistence
            self.persistence.save_order(order, strategy_id)

            logger.info(
                "Simulated order filled",
                order_id=order.order_id,
                filled_quantity=str(order.filled_quantity),
                fill_price=str(order.average_fill_price),
                latency_ms=latency_ms,
                slippage_bps=float(slippage),
            )

        except Exception as e:
            logger.error(
                "Error executing simulated order", order_id=order.order_id, error=str(e)
            )
            order.status = "failed"

    def _calculate_latency(self) -> float:
        """Calculate simulated latency based on configuration.

        Returns:
            Latency in milliseconds
        """
        if self.config.mode == SimulationMode.OPTIMISTIC:
            return max(1.0, random.gauss(5.0, 1.0))
        elif self.config.mode == SimulationMode.PESSIMISTIC:
            return max(5.0, random.gauss(20.0, 5.0))
        else:
            return max(
                1.0,
                random.gauss(self.config.base_latency_ms, self.config.latency_std_ms),
            )

    def _calculate_slippage(self) -> Decimal:
        """Calculate simulated slippage based on configuration.

        Returns:
            Slippage in basis points
        """
        if self.config.mode == SimulationMode.OPTIMISTIC:
            slippage_bps = max(0.0, random.gauss(1.0, 0.5))
        elif self.config.mode == SimulationMode.PESSIMISTIC:
            slippage_bps = max(2.0, random.gauss(5.0, 2.0))
        else:
            slippage_bps = max(
                0.0,
                random.gauss(
                    self.config.base_slippage_bps, self.config.slippage_std_bps
                ),
            )

        return Decimal(str(slippage_bps))

    def _apply_slippage(
        self, base_price: Decimal, slippage_bps: Decimal, side: str
    ) -> Decimal:
        """Apply slippage to a price.

        Args:
            base_price: Base execution price
            slippage_bps: Slippage in basis points
            side: Order side (buy/sell)

        Returns:
            Price with slippage applied
        """
        slippage_factor = slippage_bps / Decimal("10000")

        if side == "buy":
            return base_price * (Decimal("1") + slippage_factor)
        else:
            return base_price * (Decimal("1") - slippage_factor)

    def _get_current_price(self, symbol: str) -> Decimal:  # noqa: ARG002
        """Get current price for a symbol (placeholder for testing).

        Args:
            symbol: Trading symbol

        Returns:
            Current price
        """
        # This is a placeholder for testing - in production would connect to market data
        # The symbol parameter will be used in production
        return Decimal("50000")

    def get_portfolio_metrics(self, strategy_id: str) -> dict[str, Any]:
        """Get performance metrics for a portfolio.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Portfolio performance metrics
        """
        if strategy_id not in self.portfolios:
            raise ValueError(f"Portfolio not found for strategy {strategy_id}")

        return self.portfolios[strategy_id].get_metrics()

    def check_promotion_eligibility(self, strategy_id: str) -> bool:
        """Check if a strategy is eligible for promotion to live trading.

        Args:
            strategy_id: Strategy to check

        Returns:
            True if eligible for promotion
        """
        if strategy_id not in self.portfolios:
            return False

        portfolio = self.portfolios[strategy_id]
        metrics = portfolio.get_metrics()

        return self.validation_criteria.is_eligible(metrics)

    async def export_audit_trail(self, strategy_id: str) -> dict[str, Any]:
        """Export audit trail for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Audit trail data
        """
        if strategy_id not in self.portfolios:
            raise ValueError(f"Portfolio not found for strategy {strategy_id}")

        portfolio = self.portfolios[strategy_id]
        strategy_orders = [
            order
            for order in self.orders.values()
            if any(trade["order_id"] == order.order_id for trade in portfolio.trades)
        ]

        return {
            "strategy_id": strategy_id,
            "portfolio_metrics": portfolio.get_metrics(),
            "trades": portfolio.trades,
            "orders": [
                {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": str(order.quantity),
                    "filled_quantity": str(order.filled_quantity),
                    "average_fill_price": (
                        str(order.average_fill_price)
                        if order.average_fill_price
                        else None
                    ),
                    "timestamp": order.timestamp.isoformat(),
                    "fill_timestamp": (
                        order.fill_timestamp.isoformat()
                        if order.fill_timestamp
                        else None
                    ),
                    "latency_ms": order.latency_ms,
                    "slippage_bps": float(order.slippage) if order.slippage else None,
                }
                for order in strategy_orders
            ],
            "validation_criteria": self.validation_criteria.to_dict(),
            "promotion_eligible": self.check_promotion_eligibility(strategy_id),
            "export_timestamp": datetime.now().isoformat(),
        }

    async def _auto_save_loop(self) -> None:
        """Auto-save portfolios periodically."""
        try:
            while self.running:
                await asyncio.sleep(self.persistence.config.auto_save_interval_seconds)
                await self._save_all_portfolios()

                # Backup database periodically
                self.persistence.backup_database()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in auto-save loop", error=str(e))

    async def _save_all_portfolios(self) -> None:
        """Save all portfolio states to persistence."""
        for strategy_id, portfolio in self.portfolios.items():
            try:
                metrics = portfolio.get_metrics()
                positions = {
                    symbol: {
                        "quantity": str(pos.quantity),
                        "average_price": str(pos.average_price),
                        "side": pos.side,
                        "opened_at": pos.opened_at.isoformat(),
                        "last_price": str(pos.last_price) if pos.last_price else None,
                        "realized_pnl": str(pos.realized_pnl),
                        "unrealized_pnl": str(pos.unrealized_pnl),
                    }
                    for symbol, pos in portfolio.positions.items()
                }

                self.persistence.save_portfolio_state(
                    strategy_id=strategy_id,
                    current_balance=portfolio.current_balance,
                    positions=positions,
                    trades=portfolio.trades,
                    metrics=metrics,
                )

            except Exception as e:
                logger.error(
                    "Error saving portfolio state",
                    strategy_id=strategy_id,
                    error=str(e)
                )

    def load_portfolio(self, strategy_id: str) -> VirtualPortfolio | None:
        """Load portfolio from persistence.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Loaded portfolio or None if not found
        """
        state = self.persistence.load_portfolio_state(strategy_id)
        if not state:
            return None

        balance, positions_data, trades, metrics = state

        # Recreate portfolio with saved state
        portfolio = VirtualPortfolio(strategy_id, balance)
        portfolio.trades = trades

        # Restore positions
        from genesis.paper_trading.virtual_portfolio import Position
        for symbol, pos_data in positions_data.items():
            portfolio.positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal(pos_data["quantity"]),
                average_price=Decimal(pos_data["average_price"]),
                side=pos_data["side"],
                opened_at=datetime.fromisoformat(pos_data["opened_at"]),
                last_price=Decimal(pos_data["last_price"]) if pos_data["last_price"] else None,
                realized_pnl=Decimal(pos_data["realized_pnl"]),
                unrealized_pnl=Decimal(pos_data["unrealized_pnl"]),
            )

        self.portfolios[strategy_id] = portfolio
        logger.info("Portfolio loaded from persistence", strategy_id=strategy_id)

        return portfolio
