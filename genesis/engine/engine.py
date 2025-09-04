"""
Trading Engine wrapper for backwards compatibility with test suite.

This module provides the TradingEngine class expected by the test suite,
wrapping the actual implementation in TradingLoop and other components.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog

from genesis.core.models import Order, Position, Signal
from genesis.engine.event_bus import EventBus
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.state_machine import TierStateMachine
from genesis.engine.trading_loop import TradingLoop
from genesis.exchange.gateway import ExchangeGateway
from genesis.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)


class TradingEngine:
    """
    Main trading engine interface for test compatibility.
    
    Wraps TradingLoop and provides the interface expected by the test suite.
    """

    def __init__(
        self,
        exchange_gateway: ExchangeGateway,
        risk_engine: RiskEngine,
        state_machine: TierStateMachine,
        event_bus: Optional[EventBus] = None
    ):
        """Initialize the trading engine."""
        self.exchange_gateway = exchange_gateway
        self.risk_engine = risk_engine
        self.state_machine = state_machine
        self.event_bus = event_bus or EventBus()
        
        self.trading_loop = TradingLoop(
            gateway=exchange_gateway,
            risk_engine=risk_engine,
            state_machine=state_machine,
            event_bus=self.event_bus
        )
        
        self._strategies: List[BaseStrategy] = []
        self._positions: Dict[str, Position] = {}
        self._pending_orders: List[Order] = []
        self._backup_gateway: Optional[ExchangeGateway] = None
        self._database = None
        self._cache = None
        self._monitors = []
        self._non_critical_services = {}
        self._partition_detector = None
        self._load_balancer = None
        self._is_running = False

    async def start(self) -> None:
        """Start the trading engine."""
        await self.trading_loop.start()
        self._is_running = True
        self.state_machine.current_state = "RUNNING"
        logger.info("Trading engine started")

    async def stop(self) -> None:
        """Stop the trading engine."""
        await self.trading_loop.stop()
        self._is_running = False
        self.state_machine.current_state = "STOPPED"
        logger.info("Trading engine stopped")

    async def pause(self) -> None:
        """Pause the trading engine."""
        self._is_running = False
        self.state_machine.current_state = "PAUSED"
        logger.info("Trading engine paused")

    async def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict]:
        """Process a trading signal."""
        try:
            order = await self.trading_loop.process_signal(Signal(**signal))
            if order:
                return {
                    "order_id": order.order_id,
                    "status": order.status.value,
                    "filled_qty": order.filled_quantity,
                    "avg_price": order.average_price
                }
        except Exception as e:
            logger.error(f"Signal processing failed: {e}")
            return None

    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute an order."""
        try:
            result = await self.exchange_gateway.place_order(order)
            if result and result.get("order_id"):
                self._pending_orders.append(order)
            return result
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {"status": "failed", "error": str(e)}

    async def submit_order(self, order: Order) -> None:
        """Submit an order to the queue."""
        self._pending_orders.append(order)

    async def open_position(self, position: Position) -> None:
        """Open a new position."""
        position.is_open = True
        self._positions[position.symbol] = position
        logger.info(f"Position opened: {position.symbol}")

    async def close_position(self, position: Position) -> None:
        """Close an existing position."""
        position.is_open = False
        position.realized_pnl = position.unrealized_pnl
        logger.info(f"Position closed: {position.symbol}")

    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        return list(self._positions.values())

    async def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        return self._pending_orders

    async def get_all_orders(self) -> List[Order]:
        """Get all orders."""
        return self._pending_orders

    def get_pending_orders_sync(self) -> List[Order]:
        """Get pending orders synchronously."""
        return self._pending_orders

    async def register_strategy(self, strategy: BaseStrategy) -> None:
        """Register a strategy."""
        if strategy not in self._strategies:
            self._strategies.append(strategy)
            logger.info(f"Strategy registered: {strategy.__class__.__name__}")

    async def unregister_strategy(self, strategy: BaseStrategy) -> None:
        """Unregister a strategy."""
        if strategy in self._strategies:
            self._strategies.remove(strategy)
            logger.info(f"Strategy unregistered: {strategy.__class__.__name__}")

    def get_active_strategies(self) -> List[BaseStrategy]:
        """Get active strategies."""
        return self._strategies

    async def process_market_data(self, market_data: Dict[str, Any]) -> None:
        """Process market data update."""
        for strategy in self._strategies:
            try:
                await strategy.on_market_data(market_data)
            except Exception as e:
                logger.error(f"Strategy market data processing failed: {e}")

    async def get_state(self) -> Dict[str, Any]:
        """Get current engine state."""
        return {
            "positions": self._positions,
            "pending_orders": self._pending_orders,
            "strategies": [s.__class__.__name__ for s in self._strategies],
            "is_running": self._is_running
        }

    async def restore_from_state(self, state: Dict[str, Any]) -> None:
        """Restore engine from saved state."""
        self._positions = state.get("positions", {})
        self._pending_orders = state.get("pending_orders", [])
        self._is_running = state.get("is_running", False)
        logger.info("Engine restored from state")

    async def graceful_shutdown(self) -> None:
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown")
        self._is_running = False
        await self.stop()

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data with failover."""
        try:
            return await self.exchange_gateway.get_ticker(symbol)
        except Exception as e:
            if self._backup_gateway:
                logger.warning(f"Primary gateway failed, using backup: {e}")
                return await self._backup_gateway.get_ticker(symbol)
            raise

    async def save_order(self, order_data: Dict) -> bool:
        """Save order to database with failover."""
        if self._database:
            try:
                return await self._database.save(order_data)
            except Exception as e:
                logger.error(f"Database save failed: {e}")
                if hasattr(self._database, 'replica'):
                    return await self._database.replica.save(order_data)
        return False

    async def cache_position(self, position: Dict) -> None:
        """Cache position data."""
        if self._cache:
            try:
                await self._cache.set(f"position:{position['symbol']}", position)
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")

    async def get_cached_position(self, symbol: str) -> Optional[Dict]:
        """Get cached position."""
        if self._cache:
            try:
                return await self._cache.get(f"position:{symbol}")
            except Exception:
                return None
        return None

    async def get_position_from_db(self, symbol: str) -> Optional[Dict]:
        """Get position from database."""
        if self._database:
            return await self._database.get_position(symbol)
        return None

    async def record_metric(self, metric: Dict) -> None:
        """Record metric with failover."""
        for monitor in self._monitors:
            try:
                await monitor.record(metric)
                break
            except Exception as e:
                logger.warning(f"Monitor failed: {e}")
                continue

    async def health_check(self) -> bool:
        """Check engine health."""
        return self._is_running

    async def is_trading_safe(self) -> bool:
        """Check if trading is safe."""
        if self._partition_detector:
            if self._partition_detector.is_partitioned("exchange"):
                return False
        return self._is_running

    async def save_configuration(self, config: Dict) -> None:
        """Save configuration."""
        if self._database:
            await self._database.save_config(config)

    async def load_configuration(self) -> Dict:
        """Load configuration."""
        if self._database:
            return await self._database.load_config()
        return {}

    async def restore_positions_from_db(self) -> List[Position]:
        """Restore positions from database."""
        if self._database:
            positions_data = await self._database.get_all_positions()
            positions = []
            for data in positions_data:
                try:
                    position = Position(**data)
                    self._positions[position.symbol] = position
                    positions.append(position)
                except Exception as e:
                    logger.error(f"Failed to restore position: {e}")
            return positions
        return []

    async def restore_orders_from_db(self) -> None:
        """Restore orders from database."""
        if self._database:
            orders_data = await self._database.get_all_orders()
            for data in orders_data:
                try:
                    order = Order(**data)
                    self._pending_orders.append(order)
                except Exception as e:
                    logger.error(f"Failed to restore order: {e}")

    def set_backup_gateway(self, gateway: ExchangeGateway) -> None:
        """Set backup exchange gateway."""
        self._backup_gateway = gateway

    def set_database_connections(self, primary, replica) -> None:
        """Set database connections."""
        self._database = primary
        self._database.replica = replica

    def set_cache(self, cache) -> None:
        """Set cache manager."""
        self._cache = cache

    def set_monitors(self, primary, backup) -> None:
        """Set monitoring systems."""
        self._monitors = [primary, backup]

    def set_non_critical_services(self, services: Dict) -> None:
        """Set non-critical services."""
        self._non_critical_services = services

    def set_partition_detector(self, detector) -> None:
        """Set partition detector."""
        self._partition_detector = detector

    def set_load_balancer(self, balancer) -> None:
        """Set load balancer."""
        self._load_balancer = balancer

    def simulate_failure(self) -> None:
        """Simulate engine failure for testing."""
        self._is_running = False
        logger.error("Engine failure simulated")

    async def sync_state(self, state: Dict) -> None:
        """Sync state from another engine."""
        await self.restore_from_state(state)

    async def promote_to_primary(self) -> None:
        """Promote to primary engine."""
        self._is_running = True
        logger.info("Promoted to primary engine")

    async def is_primary(self) -> bool:
        """Check if this is the primary engine."""
        return self._is_running