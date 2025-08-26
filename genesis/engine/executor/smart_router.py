"""
Smart Order Router for intelligent order type selection and execution.

This module implements intelligent routing logic that automatically selects
the most appropriate order type based on market conditions, liquidity depth,
and fee optimization strategies.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

import structlog

from genesis.core.models import TradingTier
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
)
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


class ExtendedOrderType(str, Enum):
    """Extended order types for smart routing."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    FOK = "FOK"  # Fill or Kill
    IOC = "IOC"  # Immediate or Cancel
    POST_ONLY = "POST_ONLY"  # Maker-only order
    LIMIT_MAKER = "LIMIT_MAKER"  # Binance-specific post-only


class LiquidityLevel(str, Enum):
    """Liquidity depth assessment levels."""
    DEEP = "DEEP"  # > 100x order size available
    NORMAL = "NORMAL"  # 10-100x order size available
    SHALLOW = "SHALLOW"  # 2-10x order size available
    THIN = "THIN"  # < 2x order size available


class TimeFactor(str, Enum):
    """Time of day trading factors."""
    ASIA_OPEN = "ASIA_OPEN"  # 00:00-08:00 UTC
    EUROPE_OPEN = "EUROPE_OPEN"  # 08:00-14:00 UTC
    US_OPEN = "US_OPEN"  # 14:00-21:00 UTC
    QUIET = "QUIET"  # 21:00-00:00 UTC


class UrgencyLevel(str, Enum):
    """Order urgency levels."""
    HIGH = "HIGH"  # Execute immediately
    NORMAL = "NORMAL"  # Can wait for better price
    LOW = "LOW"  # Passive execution preferred


@dataclass
class MarketConditions:
    """Current market conditions assessment."""
    spread_percent: Decimal
    bid_liquidity: Decimal
    ask_liquidity: Decimal
    liquidity_level: LiquidityLevel
    time_factor: TimeFactor
    volatility: Decimal  # 24hr volatility percentage
    order_book_imbalance: Decimal  # -1 to 1 (negative = sell pressure)
    timestamp: datetime


@dataclass
class RoutedOrder:
    """Order with routing decision metadata."""
    order: Order
    selected_type: ExtendedOrderType
    routing_reason: str
    expected_fee_rate: Decimal
    market_conditions: MarketConditions
    post_only_retry_count: int = 0


@dataclass
class OrderBook:
    """Simplified order book representation."""
    symbol: str
    bids: list[tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    asks: list[tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    timestamp: datetime


class SmartRouter:
    """
    Smart order router for intelligent order type selection.
    
    Analyzes market conditions and automatically selects the most
    appropriate order type to minimize costs and slippage.
    """

    # Configuration thresholds
    TIGHT_SPREAD_THRESHOLD = Decimal("0.0005")  # 0.05%
    WIDE_SPREAD_THRESHOLD = Decimal("0.002")  # 0.2%
    POST_ONLY_SPREAD_MIN = Decimal("0.0005")  # Minimum spread for post-only
    HIGH_VOLATILITY_THRESHOLD = Decimal("0.05")  # 5% daily volatility

    # Fee rates (from Binance)
    MAKER_FEE_RATE = Decimal("0.001")  # 0.1%
    TAKER_FEE_RATE = Decimal("0.001")  # 0.1%

    # Cache settings
    CACHE_TTL_SECONDS = 5

    REQUIRED_TIER = TradingTier.HUNTER  # Minimum tier requirement

    def __init__(self, exchange_gateway):
        """
        Initialize the smart router.
        
        Args:
            exchange_gateway: Exchange gateway for market data
        """
        self.exchange_gateway = exchange_gateway
        self._conditions_cache: dict[str, tuple[MarketConditions, datetime]] = {}
        logger.info("Initialized SmartRouter")

    @requires_tier(TradingTier.HUNTER)
    async def analyze_market_conditions(self, symbol: str) -> MarketConditions:
        """
        Analyze current market conditions for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            MarketConditions with current market state
        """
        # Check cache first
        if symbol in self._conditions_cache:
            conditions, cached_at = self._conditions_cache[symbol]
            age = (datetime.now(UTC) - cached_at).total_seconds()
            if age < self.CACHE_TTL_SECONDS:
                logger.debug("Using cached market conditions",
                           symbol=symbol, age_seconds=age)
                return conditions

        try:
            # Fetch order book and ticker data
            order_book = await self.exchange_gateway.get_order_book(symbol)
            ticker = await self.exchange_gateway.get_ticker(symbol)

            # Calculate spread
            best_bid = Decimal(str(order_book['bids'][0][0])) if order_book['bids'] else Decimal("0")
            best_ask = Decimal(str(order_book['asks'][0][0])) if order_book['asks'] else Decimal("0")
            mid_price = (best_bid + best_ask) / Decimal("2")

            spread_percent = Decimal("0")
            if mid_price > 0:
                spread_percent = ((best_ask - best_bid) / mid_price) * Decimal("100")

            # Calculate liquidity depth
            bid_liquidity = sum(Decimal(str(bid[1])) for bid in order_book['bids'][:20])
            ask_liquidity = sum(Decimal(str(ask[1])) for ask in order_book['asks'][:20])

            # Assess liquidity level (simplified)
            total_liquidity = bid_liquidity + ask_liquidity
            liquidity_level = self._assess_liquidity_level(total_liquidity)

            # Determine time factor
            time_factor = self._get_time_factor()

            # Calculate order book imbalance
            imbalance = Decimal("0")
            if bid_liquidity + ask_liquidity > 0:
                imbalance = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity)

            # Get volatility from ticker
            volatility = Decimal(str(ticker.get('priceChangePercent', 0))) / Decimal("100")

            conditions = MarketConditions(
                spread_percent=spread_percent,
                bid_liquidity=bid_liquidity,
                ask_liquidity=ask_liquidity,
                liquidity_level=liquidity_level,
                time_factor=time_factor,
                volatility=abs(volatility),
                order_book_imbalance=imbalance,
                timestamp=datetime.now(UTC)
            )

            # Cache the conditions
            self._conditions_cache[symbol] = (conditions, datetime.now(UTC))

            logger.info("Market conditions analyzed",
                       symbol=symbol,
                       spread_percent=float(spread_percent),
                       liquidity_level=liquidity_level.value,
                       volatility=float(volatility))

            return conditions

        except Exception as e:
            logger.error("Failed to analyze market conditions",
                        symbol=symbol, error=str(e))
            # Return conservative defaults on error
            return MarketConditions(
                spread_percent=Decimal("0.001"),
                bid_liquidity=Decimal("0"),
                ask_liquidity=Decimal("0"),
                liquidity_level=LiquidityLevel.THIN,
                time_factor=self._get_time_factor(),
                volatility=Decimal("0.01"),
                order_book_imbalance=Decimal("0"),
                timestamp=datetime.now(UTC)
            )

    @requires_tier(TradingTier.HUNTER)
    def select_order_type(
        self,
        order: Order,
        conditions: MarketConditions,
        urgency: UrgencyLevel = UrgencyLevel.NORMAL
    ) -> ExtendedOrderType:
        """
        Select the most appropriate order type based on conditions.
        
        Args:
            order: Order to route
            conditions: Current market conditions
            urgency: Order urgency level
            
        Returns:
            Selected order type
        """
        # High urgency always uses market orders
        if urgency == UrgencyLevel.HIGH:
            logger.debug("High urgency - selecting MARKET order",
                        order_id=order.order_id)
            return ExtendedOrderType.MARKET

        # Check spread conditions
        if conditions.spread_percent <= self.TIGHT_SPREAD_THRESHOLD:
            # Tight spread - market orders are acceptable
            if conditions.liquidity_level in [LiquidityLevel.DEEP, LiquidityLevel.NORMAL]:
                logger.debug("Tight spread with good liquidity - MARKET order",
                           order_id=order.order_id)
                return ExtendedOrderType.MARKET
            else:
                # Low liquidity - use IOC to avoid slippage
                logger.debug("Tight spread but low liquidity - IOC order",
                           order_id=order.order_id)
                return ExtendedOrderType.IOC

        elif conditions.spread_percent >= self.WIDE_SPREAD_THRESHOLD:
            # Wide spread - try to capture spread with post-only
            if urgency == UrgencyLevel.LOW:
                logger.debug("Wide spread, low urgency - POST_ONLY order",
                           order_id=order.order_id)
                return ExtendedOrderType.POST_ONLY
            else:
                # Normal urgency with wide spread - use limit order
                logger.debug("Wide spread, normal urgency - LIMIT order",
                           order_id=order.order_id)
                return ExtendedOrderType.LIMIT

        # Medium spread - decision based on other factors
        if conditions.volatility > self.HIGH_VOLATILITY_THRESHOLD:
            # High volatility - use FOK to ensure full fill or nothing
            if conditions.liquidity_level == LiquidityLevel.THIN:
                logger.debug("High volatility, thin liquidity - FOK order",
                           order_id=order.order_id)
                return ExtendedOrderType.FOK
            else:
                # Good liquidity - IOC acceptable
                logger.debug("High volatility, good liquidity - IOC order",
                           order_id=order.order_id)
                return ExtendedOrderType.IOC

        # Default case - optimize for fees
        if urgency == UrgencyLevel.LOW and conditions.spread_percent > self.POST_ONLY_SPREAD_MIN:
            logger.debug("Low urgency, optimizing fees - POST_ONLY order",
                       order_id=order.order_id)
            return ExtendedOrderType.POST_ONLY

        logger.debug("Default selection - LIMIT order",
                   order_id=order.order_id)
        return ExtendedOrderType.LIMIT

    @requires_tier(TradingTier.HUNTER)
    def calculate_execution_score(
        self,
        order: Order,
        result: ExecutionResult,
        market_conditions: MarketConditions
    ) -> float:
        """
        Calculate execution quality score (0-100).
        
        Args:
            order: Original order
            result: Execution result
            market_conditions: Market conditions at execution
            
        Returns:
            Execution quality score
        """
        score = 100.0

        # Penalize for slippage (max -30 points)
        if result.slippage_percent:
            slippage_penalty = min(float(abs(result.slippage_percent)) * 10, 30)
            score -= slippage_penalty

        # Penalize for high fees (max -20 points)
        if hasattr(order, 'taker_fee_paid') and order.taker_fee_paid:
            # Taker fee paid - penalize
            fee_penalty = float(order.taker_fee_paid / order.quantity) * 2000
            score -= min(fee_penalty, 20)
        elif hasattr(order, 'maker_fee_paid') and order.maker_fee_paid:
            # Maker fee paid - smaller penalty
            fee_penalty = float(order.maker_fee_paid / order.quantity) * 1000
            score -= min(fee_penalty, 10)

        # Penalize for high latency (max -20 points)
        if result.latency_ms:
            if result.latency_ms > 1000:
                score -= 20
            elif result.latency_ms > 500:
                score -= 10
            elif result.latency_ms > 200:
                score -= 5

        # Bonus for favorable execution in volatile markets (+10 points)
        if market_conditions.volatility > self.HIGH_VOLATILITY_THRESHOLD:
            if result.slippage_percent and result.slippage_percent < 0:
                # Negative slippage is favorable
                score += 10

        # Bonus for successful post-only execution (+5 points)
        if hasattr(order, 'routing_method') and order.routing_method == 'POST_ONLY':
            if result.success and not hasattr(order, 'taker_fee_paid'):
                score += 5

        return max(0.0, min(100.0, score))

    @requires_tier(TradingTier.HUNTER)
    async def route_order(
        self,
        order: Order,
        urgency: UrgencyLevel = UrgencyLevel.NORMAL
    ) -> RoutedOrder:
        """
        Route an order with intelligent type selection.
        
        Args:
            order: Order to route
            urgency: Order urgency level
            
        Returns:
            RoutedOrder with routing decision
        """
        # Analyze market conditions
        conditions = await self.analyze_market_conditions(order.symbol)

        # Select order type
        selected_type = self.select_order_type(order, conditions, urgency)

        # Determine expected fees
        expected_fee_rate = self.TAKER_FEE_RATE
        if selected_type in [ExtendedOrderType.POST_ONLY, ExtendedOrderType.LIMIT_MAKER]:
            expected_fee_rate = self.MAKER_FEE_RATE
        elif selected_type == ExtendedOrderType.LIMIT:
            # Limit orders might become maker or taker
            if conditions.spread_percent > self.POST_ONLY_SPREAD_MIN:
                expected_fee_rate = self.MAKER_FEE_RATE

        # Create routing reason
        routing_reason = self._generate_routing_reason(
            selected_type, conditions, urgency
        )

        # Add routing metadata to order
        order.routing_method = selected_type.value

        routed_order = RoutedOrder(
            order=order,
            selected_type=selected_type,
            routing_reason=routing_reason,
            expected_fee_rate=expected_fee_rate,
            market_conditions=conditions
        )

        logger.info("Order routed",
                   order_id=order.order_id,
                   selected_type=selected_type.value,
                   reason=routing_reason,
                   expected_fee_rate=float(expected_fee_rate))

        return routed_order

    def calculate_spread_percentage(self, order_book: OrderBook) -> Decimal:
        """
        Calculate spread as a percentage of mid price.
        
        Args:
            order_book: Order book data
            
        Returns:
            Spread percentage
        """
        if not order_book.bids or not order_book.asks:
            return Decimal("999")  # No market

        best_bid = order_book.bids[0][0]
        best_ask = order_book.asks[0][0]
        mid_price = (best_bid + best_ask) / Decimal("2")

        if mid_price == 0:
            return Decimal("999")

        spread = best_ask - best_bid
        spread_percent = (spread / mid_price) * Decimal("100")

        return spread_percent.quantize(Decimal("0.0001"))

    def assess_liquidity_depth(
        self,
        order_book: OrderBook,
        size: Decimal
    ) -> LiquidityLevel:
        """
        Assess liquidity depth relative to order size.
        
        Args:
            order_book: Order book data
            size: Order size to evaluate
            
        Returns:
            Liquidity level assessment
        """
        # Calculate available liquidity at different price levels
        bid_liquidity = sum(qty for _, qty in order_book.bids[:10])
        ask_liquidity = sum(qty for _, qty in order_book.asks[:10])

        # Use the relevant side's liquidity
        available_liquidity = min(bid_liquidity, ask_liquidity)

        # Compare to order size
        ratio = available_liquidity / size if size > 0 else Decimal("999")

        if ratio > 100:
            return LiquidityLevel.DEEP
        elif ratio > 10:
            return LiquidityLevel.NORMAL
        elif ratio > 2:
            return LiquidityLevel.SHALLOW
        else:
            return LiquidityLevel.THIN

    def get_time_of_day_factor(self) -> TimeFactor:
        """
        Get current time of day trading factor.
        
        Returns:
            Time factor based on UTC hour
        """
        current_hour = datetime.now(UTC).hour

        if 0 <= current_hour < 8:
            return TimeFactor.ASIA_OPEN
        elif 8 <= current_hour < 14:
            return TimeFactor.EUROPE_OPEN
        elif 14 <= current_hour < 21:
            return TimeFactor.US_OPEN
        else:
            return TimeFactor.QUIET

    def estimate_market_impact(
        self,
        size: Decimal,
        liquidity: LiquidityLevel
    ) -> Decimal:
        """
        Estimate market impact of an order.
        
        Args:
            size: Order size
            liquidity: Current liquidity level
            
        Returns:
            Estimated price impact percentage
        """
        # Simplified impact model
        base_impact = Decimal("0")

        if liquidity == LiquidityLevel.DEEP:
            base_impact = size * Decimal("0.00001")  # 0.001% per unit
        elif liquidity == LiquidityLevel.NORMAL:
            base_impact = size * Decimal("0.00005")  # 0.005% per unit
        elif liquidity == LiquidityLevel.SHALLOW:
            base_impact = size * Decimal("0.0002")  # 0.02% per unit
        else:  # THIN
            base_impact = size * Decimal("0.001")  # 0.1% per unit

        return min(base_impact, Decimal("5"))  # Cap at 5% impact

    def _assess_liquidity_level(self, total_liquidity: Decimal) -> LiquidityLevel:
        """
        Assess overall liquidity level.
        
        Args:
            total_liquidity: Total available liquidity
            
        Returns:
            Liquidity level classification
        """
        # These thresholds would be calibrated per trading pair
        if total_liquidity > 10000:
            return LiquidityLevel.DEEP
        elif total_liquidity > 1000:
            return LiquidityLevel.NORMAL
        elif total_liquidity > 100:
            return LiquidityLevel.SHALLOW
        else:
            return LiquidityLevel.THIN

    def _get_time_factor(self) -> TimeFactor:
        """
        Get current time factor.
        
        Returns:
            Current time factor
        """
        return self.get_time_of_day_factor()

    def _generate_routing_reason(
        self,
        selected_type: ExtendedOrderType,
        conditions: MarketConditions,
        urgency: UrgencyLevel
    ) -> str:
        """
        Generate human-readable routing reason.
        
        Args:
            selected_type: Selected order type
            conditions: Market conditions
            urgency: Order urgency
            
        Returns:
            Routing reason string
        """
        reasons = []

        if urgency == UrgencyLevel.HIGH:
            reasons.append("high urgency")
        elif urgency == UrgencyLevel.LOW:
            reasons.append("low urgency")

        if conditions.spread_percent <= self.TIGHT_SPREAD_THRESHOLD:
            reasons.append("tight spread")
        elif conditions.spread_percent >= self.WIDE_SPREAD_THRESHOLD:
            reasons.append("wide spread")

        if conditions.liquidity_level == LiquidityLevel.THIN:
            reasons.append("thin liquidity")
        elif conditions.liquidity_level == LiquidityLevel.DEEP:
            reasons.append("deep liquidity")

        if conditions.volatility > self.HIGH_VOLATILITY_THRESHOLD:
            reasons.append("high volatility")

        if selected_type in [ExtendedOrderType.POST_ONLY, ExtendedOrderType.LIMIT_MAKER]:
            reasons.append("optimizing maker fees")

        return f"{selected_type.value} selected due to: {', '.join(reasons)}"

    @requires_tier(TradingTier.HUNTER)
    async def execute_routed_order(self, order: Order) -> "ExecutionResult":
        """
        Execute an order using smart routing.
        
        Args:
            order: Order to execute
            
        Returns:
            ExecutionResult with execution details
        """
        from genesis.engine.executor.base import ExecutionResult
        from genesis.engine.executor.smart_router import UrgencyLevel

        # Route the order
        routed = await self.route_order(order, UrgencyLevel.NORMAL)

        # Here you would execute the order based on routing decision
        # For now, return a mock result
        return ExecutionResult(
            success=True,
            order=order,
            message=f"Order executed via {routed.selected_type.value}",
            actual_price=order.price,
            slippage_percent=Decimal("0"),
            latency_ms=100
        )
