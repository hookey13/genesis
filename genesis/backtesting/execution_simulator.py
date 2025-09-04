"""
Execution Simulator for Backtesting

Simulates realistic order execution with slippage and fees.
"""

import random
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any
from enum import Enum

import structlog

logger = structlog.get_logger()


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Side of the order."""
    BUY = "buy"
    SELL = "sell"


class FillStatus(Enum):
    """Status of order fill."""
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"


@dataclass
class Signal:
    """Trading signal from strategy."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    metadata: Dict[str, Any] = None


@dataclass  
class Fill:
    """Execution fill details."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    slippage: Decimal
    status: FillStatus
    order_id: str
    value: Decimal  # quantity * price
    metadata: Dict[str, Any] = None


class SlippageModel:
    """Base class for slippage models."""
    
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: OrderSide,
        market_data: Any
    ) -> Decimal:
        """Calculate slippage amount."""
        raise NotImplementedError


class LinearSlippageModel(SlippageModel):
    """Linear slippage based on order size."""
    
    def __init__(self, slippage_pct: Decimal = Decimal("0.001")):
        """Initialize with slippage percentage."""
        self.slippage_pct = slippage_pct
    
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: OrderSide,
        market_data: Any
    ) -> Decimal:
        """Calculate linear slippage.
        
        Slippage increases linearly with order size relative to volume.
        """
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            # Slippage based on order size relative to market volume
            volume_impact = quantity / market_data.volume
            slippage_multiplier = 1 + min(volume_impact * 10, Decimal('0.05'))  # Cap at 5%
        else:
            slippage_multiplier = 1
        
        base_slippage = price * self.slippage_pct * slippage_multiplier
        
        # Adverse selection: buys pay more, sells receive less
        if side == OrderSide.BUY:
            return base_slippage
        else:
            return -base_slippage


class SquareRootSlippageModel(SlippageModel):
    """Square-root market impact model."""
    
    def __init__(self, impact_coeff: Decimal = Decimal("0.0001")):
        """Initialize with impact coefficient."""
        self.impact_coeff = impact_coeff
    
    def calculate(
        self,
        price: Decimal,
        quantity: Decimal,
        side: OrderSide,
        market_data: Any
    ) -> Decimal:
        """Calculate square-root market impact.
        
        Slippage follows square root of order size for more realistic impact.
        """
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            # Average daily volume proxy
            adv = market_data.volume * 1440  # Assume 1m bars, scale to daily
            
            # Square root market impact
            import math
            impact = self.impact_coeff * Decimal(str(math.sqrt(float(quantity / adv))))
            slippage = price * impact
        else:
            slippage = price * self.impact_coeff
        
        # Adverse selection
        if side == OrderSide.BUY:
            return slippage
        else:
            return -slippage


class FeeModel:
    """Base class for fee models."""
    
    def calculate(self, quantity: Decimal, price: Decimal, side: OrderSide) -> Decimal:
        """Calculate trading fees."""
        raise NotImplementedError


class BinanceFeeModel(FeeModel):
    """Binance fee structure."""
    
    def __init__(
        self,
        maker_fee: Decimal = Decimal("0.001"),
        taker_fee: Decimal = Decimal("0.001"),
        use_bnb: bool = False
    ):
        """Initialize Binance fee model.
        
        Args:
            maker_fee: Maker fee rate (0.1% default)
            taker_fee: Taker fee rate (0.1% default)
            use_bnb: Whether BNB discount is applied (25% off)
        """
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.bnb_discount = Decimal("0.75") if use_bnb else Decimal("1.0")
    
    def calculate(
        self,
        quantity: Decimal,
        price: Decimal,
        side: OrderSide,
        is_maker: bool = False
    ) -> Decimal:
        """Calculate Binance trading fees."""
        notional = quantity * price
        base_fee = self.maker_fee if is_maker else self.taker_fee
        return notional * base_fee * self.bnb_discount


class ExecutionSimulator:
    """
    Simulates order execution with realistic fills.
    
    Handles slippage, fees, partial fills, and rejections.
    """
    
    def __init__(
        self,
        slippage_model: str = "linear",
        fee_model: str = "binance",
        rejection_rate: float = 0.001,
        partial_fill_rate: float = 0.05
    ):
        """Initialize execution simulator.
        
        Args:
            slippage_model: Type of slippage model to use
            fee_model: Type of fee model to use
            rejection_rate: Probability of order rejection
            partial_fill_rate: Probability of partial fill
        """
        self.slippage_model = self._create_slippage_model(slippage_model)
        self.fee_model = self._create_fee_model(fee_model)
        self.rejection_rate = rejection_rate
        self.partial_fill_rate = partial_fill_rate
        self.order_counter = 0
    
    def _create_slippage_model(self, model_type: str) -> SlippageModel:
        """Create slippage model instance."""
        if model_type == "linear":
            return LinearSlippageModel()
        elif model_type == "square_root":
            return SquareRootSlippageModel()
        else:
            logger.warning(f"Unknown slippage model: {model_type}, using linear")
            return LinearSlippageModel()
    
    def _create_fee_model(self, model_type: str) -> FeeModel:
        """Create fee model instance."""
        if model_type == "binance":
            return BinanceFeeModel()
        else:
            logger.warning(f"Unknown fee model: {model_type}, using binance")
            return BinanceFeeModel()
    
    async def simulate_fill(
        self,
        signal: Signal,
        market_data: Any,
        portfolio: Any = None
    ) -> Optional[Fill]:
        """
        Simulate order execution.
        
        Args:
            signal: Trading signal to execute
            market_data: Current market snapshot
            portfolio: Portfolio for position/margin checks
            
        Returns:
            Fill object if executed, None if rejected
        """
        self.order_counter += 1
        order_id = f"BT-{self.order_counter:08d}"
        
        # Check for rejection
        if random.random() < self.rejection_rate:
            logger.warning(
                "order_rejected",
                order_id=order_id,
                reason="random_rejection",
                symbol=signal.symbol
            )
            return None
        
        # Determine fill price based on order type
        fill_price = await self._determine_fill_price(
            signal, market_data
        )
        
        if not fill_price:
            logger.warning(
                "order_rejected",
                order_id=order_id,
                reason="no_fill_price",
                symbol=signal.symbol
            )
            return None
        
        # Check portfolio constraints
        if portfolio:
            if not await self._check_portfolio_constraints(
                signal, fill_price, portfolio
            ):
                logger.warning(
                    "order_rejected",
                    order_id=order_id,
                    reason="portfolio_constraints",
                    symbol=signal.symbol
                )
                return None
        
        # Calculate slippage
        slippage = self.slippage_model.calculate(
            price=fill_price,
            quantity=signal.quantity,
            side=signal.side,
            market_data=market_data
        )
        
        # Adjust fill price for slippage
        executed_price = fill_price + slippage
        
        # Ensure price is positive
        executed_price = max(executed_price, Decimal("0.00000001"))
        
        # Determine fill quantity (potential partial fill)
        fill_quantity = signal.quantity
        fill_status = FillStatus.FILLED
        
        if random.random() < self.partial_fill_rate:
            # Partial fill between 50% and 90% of requested
            fill_pct = Decimal(str(random.uniform(0.5, 0.9)))
            fill_quantity = signal.quantity * fill_pct
            fill_status = FillStatus.PARTIALLY_FILLED
            logger.info(
                "partial_fill",
                order_id=order_id,
                requested=float(signal.quantity),
                filled=float(fill_quantity)
            )
        
        # Calculate fees
        is_maker = signal.order_type == OrderType.LIMIT
        fee = self.fee_model.calculate(
            quantity=fill_quantity,
            price=executed_price,
            side=signal.side,
            is_maker=is_maker
        )
        
        # Create fill object
        fill = Fill(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=signal.side,
            quantity=fill_quantity,
            price=executed_price,
            fee=fee,
            slippage=slippage,
            status=fill_status,
            order_id=order_id,
            value=fill_quantity * executed_price,
            metadata={
                'signal_metadata': signal.metadata,
                'market_price': float(fill_price),
                'requested_quantity': float(signal.quantity)
            }
        )
        
        logger.info(
            "order_filled",
            order_id=order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=float(fill.quantity),
            price=float(fill.price),
            slippage=float(fill.slippage),
            fee=float(fill.fee),
            status=fill.status.value
        )
        
        return fill
    
    async def _determine_fill_price(
        self,
        signal: Signal,
        market_data: Any
    ) -> Optional[Decimal]:
        """
        Determine the base fill price before slippage.
        
        Args:
            signal: Trading signal
            market_data: Market snapshot
            
        Returns:
            Base fill price or None if cannot execute
        """
        if signal.order_type == OrderType.MARKET:
            # Use bid/ask if available, otherwise use close
            if signal.side == OrderSide.BUY:
                if hasattr(market_data, 'ask_price') and market_data.ask_price:
                    return market_data.ask_price
                else:
                    return market_data.close
            else:  # SELL
                if hasattr(market_data, 'bid_price') and market_data.bid_price:
                    return market_data.bid_price
                else:
                    return market_data.close
        
        elif signal.order_type == OrderType.LIMIT:
            # Check if limit price would fill
            if signal.side == OrderSide.BUY:
                market_price = market_data.ask_price if hasattr(market_data, 'ask_price') else market_data.close
                if signal.limit_price >= market_price:
                    return signal.limit_price
            else:  # SELL
                market_price = market_data.bid_price if hasattr(market_data, 'bid_price') else market_data.close
                if signal.limit_price <= market_price:
                    return signal.limit_price
            return None  # Limit order wouldn't fill
        
        elif signal.order_type == OrderType.STOP:
            # Check if stop triggered
            if signal.side == OrderSide.BUY:
                if market_data.close >= signal.stop_price:
                    return market_data.close
            else:  # SELL
                if market_data.close <= signal.stop_price:
                    return market_data.close
            return None  # Stop not triggered
        
        else:
            logger.warning(f"Unsupported order type: {signal.order_type}")
            return None
    
    async def _check_portfolio_constraints(
        self,
        signal: Signal,
        fill_price: Decimal,
        portfolio: Any
    ) -> bool:
        """
        Check if portfolio can support the order.
        
        Args:
            signal: Trading signal
            fill_price: Expected fill price
            portfolio: Portfolio object
            
        Returns:
            True if order can be executed
        """
        required_capital = signal.quantity * fill_price
        
        # Add estimated fees
        fee = self.fee_model.calculate(
            quantity=signal.quantity,
            price=fill_price,
            side=signal.side,
            is_maker=False  # Assume taker for conservative estimate
        )
        
        total_required = required_capital + fee
        
        # Check available capital
        if hasattr(portfolio, 'available_capital'):
            if portfolio.available_capital < total_required:
                logger.warning(
                    "insufficient_capital",
                    required=float(total_required),
                    available=float(portfolio.available_capital)
                )
                return False
        
        # Check position limits
        if hasattr(portfolio, 'max_position_size'):
            current_position = portfolio.get_position(signal.symbol) if hasattr(portfolio, 'get_position') else 0
            new_position = current_position + (signal.quantity if signal.side == OrderSide.BUY else -signal.quantity)
            
            if abs(new_position) > portfolio.max_position_size:
                logger.warning(
                    "position_limit_exceeded",
                    current=float(current_position),
                    requested=float(signal.quantity),
                    max_allowed=float(portfolio.max_position_size)
                )
                return False
        
        return True