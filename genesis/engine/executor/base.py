"""
Abstract base class for order executors in Project GENESIS.

This module defines the interface that all order executors must implement,
ensuring consistent behavior across different execution strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import uuid4

import structlog

from genesis.core.models import TradingTier

logger = structlog.get_logger(__name__)


class OrderType(str, Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    FOK = "FOK"  # Fill or Kill
    IOC = "IOC"  # Immediate or Cancel
    POST_ONLY = "POST_ONLY"  # Maker-only order
    LIMIT_MAKER = "LIMIT_MAKER"  # Binance-specific post-only


class OrderSide(str, Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order status states."""
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    position_id: str | None
    client_order_id: str
    symbol: str
    type: OrderType
    side: OrderSide
    price: Decimal | None
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    status: OrderStatus = OrderStatus.PENDING
    slice_number: int | None = None
    total_slices: int | None = None
    latency_ms: int | None = None
    slippage_percent: Decimal | None = None
    created_at: datetime = None
    executed_at: datetime | None = None
    exchange_order_id: str | None = None
    routing_method: str | None = None
    maker_fee_paid: Decimal | None = None
    taker_fee_paid: Decimal | None = None
    execution_score: float | None = None

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if not self.client_order_id:
            self.client_order_id = str(uuid4())


@dataclass
class ExecutionResult:
    """Result of an order execution."""
    success: bool
    order: Order
    message: str
    actual_price: Decimal | None = None
    slippage_percent: Decimal | None = None
    latency_ms: int | None = None
    error: str | None = None


class ExecutionStrategy(str, Enum):
    """Execution strategy types."""
    MARKET = "MARKET"
    ICEBERG = "ICEBERG"
    VWAP = "VWAP"
    TWAP = "TWAP"
    SMART = "SMART"  # Smart order routing


class OrderExecutor(ABC):
    """
    Abstract base class for order execution strategies.
    
    This class defines the interface that all concrete executors must implement.
    Each tier has its own executor with appropriate features.
    """

    def __init__(self, tier: TradingTier):
        """
        Initialize the order executor.
        
        Args:
            tier: Current trading tier
        """
        self.tier = tier
        self.iceberg_executor = None  # Will be set by strategy engine
        self.smart_router = None  # Will be set by strategy engine
        logger.info("Initializing order executor", tier=tier.value)

    @abstractmethod
    async def execute_market_order(self, order: Order, confirmation_required: bool = True) -> ExecutionResult:
        """
        Execute a market order.
        
        Args:
            order: Order to execute
            confirmation_required: Whether to require confirmation before execution
            
        Returns:
            ExecutionResult with execution details
        """
        pass

    async def execute_order(self, order: Order, strategy: ExecutionStrategy = ExecutionStrategy.MARKET) -> ExecutionResult:
        """
        Execute an order with the specified strategy.
        
        Args:
            order: Order to execute
            strategy: Execution strategy to use
            
        Returns:
            ExecutionResult with execution details
        """
        if strategy == ExecutionStrategy.ICEBERG:
            if self.iceberg_executor is None:
                raise ValueError("Iceberg executor not configured")
            return await self.iceberg_executor.execute_iceberg_order(order)
        elif strategy == ExecutionStrategy.SMART:
            if self.smart_router is None:
                raise ValueError("Smart router not configured")
            return await self.smart_router.execute_routed_order(order)
        elif strategy == ExecutionStrategy.MARKET:
            return await self.execute_market_order(order)
        else:
            raise NotImplementedError(f"Strategy {strategy} not yet implemented")

    async def route_order(self, order: Order) -> "RoutedOrder":
        """
        Route an order using smart routing logic.
        
        Args:
            order: Order to route
            
        Returns:
            RoutedOrder with routing decision
        """
        if self.smart_router is None:
            raise ValueError("Smart router not configured for this tier")
        
        from genesis.engine.executor.smart_router import UrgencyLevel
        
        # Determine urgency based on order characteristics
        urgency = UrgencyLevel.NORMAL
        if hasattr(order, 'urgency'):
            urgency = order.urgency
        
        return await self.smart_router.route_order(order, urgency)
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: Exchange order ID to cancel
            symbol: Trading symbol
            
        Returns:
            True if cancellation successful
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """
        Emergency cancel all open orders.
        
        Args:
            symbol: Optional symbol to filter cancellations
            
        Returns:
            Number of orders cancelled
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Order:
        """
        Get current status of an order.
        
        Args:
            order_id: Order ID to check
            symbol: Trading symbol
            
        Returns:
            Order with current status
        """
        pass

    def generate_client_order_id(self) -> str:
        """
        Generate a unique client order ID for idempotency.
        
        Returns:
            Unique client order ID
        """
        return str(uuid4())

    def calculate_slippage(self, expected_price: Decimal, actual_price: Decimal, side: OrderSide) -> Decimal:
        """
        Calculate slippage percentage between expected and actual price.
        
        Args:
            expected_price: Expected execution price
            actual_price: Actual execution price
            side: Order side (buy/sell)
            
        Returns:
            Slippage percentage (positive = unfavorable, negative = favorable)
        """
        if expected_price == 0:
            return Decimal("0")

        if side == OrderSide.BUY:
            # For buys, higher actual price is unfavorable
            slippage = ((actual_price - expected_price) / expected_price) * Decimal("100")
        else:
            # For sells, lower actual price is unfavorable
            slippage = ((expected_price - actual_price) / expected_price) * Decimal("100")

        return slippage.quantize(Decimal("0.0001"))

    def validate_order(self, order: Order) -> None:
        """
        Validate order parameters.
        
        Args:
            order: Order to validate
            
        Raises:
            ValueError: If order parameters are invalid
        """
        if order.quantity <= 0:
            raise ValueError(f"Order quantity must be positive: {order.quantity}")

        if order.type == OrderType.LIMIT and order.price is None:
            raise ValueError("Limit orders require a price")

        if order.type == OrderType.LIMIT and order.price <= 0:
            raise ValueError(f"Order price must be positive: {order.price}")

        if not order.symbol:
            raise ValueError("Order symbol is required")

        if not order.client_order_id:
            raise ValueError("Client order ID is required for idempotency")
