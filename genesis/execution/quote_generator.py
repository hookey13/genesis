"""Quote generation logic for market making strategies."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import structlog

from genesis.core.models import Order, OrderSide, OrderType

logger = structlog.get_logger(__name__)


@dataclass
class QuoteLevel:
    """Single quote level with price and size."""

    price: Decimal
    quantity: Decimal
    side: OrderSide
    layer: int
    spread_bps: Decimal
    distance_bps: Decimal
    order_id: UUID = field(default_factory=uuid4)

    @property
    def dollar_value(self) -> Decimal:
        """Calculate dollar value of quote."""
        return self.price * self.quantity


@dataclass
class QuoteSet:
    """Set of quotes for both sides of the market."""

    symbol: str
    mid_price: Decimal
    bid_quotes: List[QuoteLevel]
    ask_quotes: List[QuoteLevel]
    total_bid_value: Decimal
    total_ask_value: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict = field(default_factory=dict)

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        return self.bid_quotes[0].price if self.bid_quotes else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price."""
        return self.ask_quotes[0].price if self.ask_quotes else None

    @property
    def quoted_spread_bps(self) -> Optional[Decimal]:
        """Calculate quoted spread in basis points."""
        if self.best_bid and self.best_ask and self.best_bid > 0:
            return (self.best_ask - self.best_bid) / self.best_bid * Decimal("10000")
        return None


@dataclass
class QuoteParameters:
    """Parameters for quote generation."""

    num_layers: int = 3
    layer_spacing_multiplier: Decimal = Decimal("2")
    base_quote_size_usdt: Decimal = Decimal("1000")
    min_quote_size_usdt: Decimal = Decimal("100")
    max_quote_size_usdt: Decimal = Decimal("5000")

    # Size distribution across layers (must sum to 1.0)
    layer_size_distribution: List[Decimal] = field(
        default_factory=lambda: [Decimal("0.40"), Decimal("0.35"), Decimal("0.25")]
    )

    # Post-only orders to ensure maker fees
    post_only: bool = True

    # Price improvement
    price_improvement_bps: Decimal = Decimal("0.5")  # Improve by 0.5 bps inside spread


class QuoteGenerator:
    """Generate quotes for market making."""

    def __init__(self, params: Optional[QuoteParameters] = None):
        """Initialize quote generator with parameters."""
        self.params = params or QuoteParameters()
        self.active_quotes: Dict[str, QuoteSet] = {}
        self.quote_history: List[QuoteSet] = []

        # Performance tracking
        self.total_quotes_generated = 0
        self.total_quote_value = Decimal("0")

    def generate_quotes(
        self,
        symbol: str,
        mid_price: Decimal,
        spread_bps: Decimal,
        inventory_skew: Decimal = Decimal("0"),
        size_multiplier: Decimal = Decimal("1"),
        bid_size_adjustment: Decimal = Decimal("1"),
        ask_size_adjustment: Decimal = Decimal("1"),
    ) -> QuoteSet:
        """
        Generate a set of quotes for market making.

        Args:
            symbol: Trading symbol
            mid_price: Current mid-market price
            spread_bps: Target spread in basis points
            inventory_skew: Inventory skew for price adjustment (-1 to 1)
            size_multiplier: Overall size multiplier
            bid_size_adjustment: Bid-specific size adjustment
            ask_size_adjustment: Ask-specific size adjustment

        Returns:
            QuoteSet with bid and ask quotes
        """
        bid_quotes = []
        ask_quotes = []

        # Calculate skew adjustment in basis points
        skew_adjustment_bps = inventory_skew * Decimal("20")  # Max 20 bps skew

        # Generate quotes for each layer
        for layer in range(self.params.num_layers):
            # Calculate layer spread
            layer_multiplier = self.params.layer_spacing_multiplier**layer
            layer_spread_bps = spread_bps * layer_multiplier

            # Calculate bid and ask prices with skew
            bid_price, ask_price = self._calculate_quote_prices(
                mid_price, layer_spread_bps, skew_adjustment_bps
            )

            # Calculate sizes for this layer
            layer_size_pct = self._get_layer_size_distribution(layer)
            base_size = (
                self.params.base_quote_size_usdt * size_multiplier * layer_size_pct
            )

            # Apply side-specific adjustments
            bid_size = base_size * bid_size_adjustment
            ask_size = base_size * ask_size_adjustment

            # Create bid quote if size is sufficient
            if bid_size >= self.params.min_quote_size_usdt:
                bid_quantity = bid_size / bid_price
                bid_quote = QuoteLevel(
                    price=bid_price,
                    quantity=bid_quantity,
                    side=OrderSide.BUY,
                    layer=layer,
                    spread_bps=layer_spread_bps,
                    distance_bps=self._calculate_distance_bps(mid_price, bid_price),
                )
                bid_quotes.append(bid_quote)

            # Create ask quote if size is sufficient
            if ask_size >= self.params.min_quote_size_usdt:
                ask_quantity = ask_size / ask_price
                ask_quote = QuoteLevel(
                    price=ask_price,
                    quantity=ask_quantity,
                    side=OrderSide.SELL,
                    layer=layer,
                    spread_bps=layer_spread_bps,
                    distance_bps=self._calculate_distance_bps(mid_price, ask_price),
                )
                ask_quotes.append(ask_quote)

        # Calculate total values
        total_bid_value = sum(q.dollar_value for q in bid_quotes)
        total_ask_value = sum(q.dollar_value for q in ask_quotes)

        # Create quote set
        quote_set = QuoteSet(
            symbol=symbol,
            mid_price=mid_price,
            bid_quotes=bid_quotes,
            ask_quotes=ask_quotes,
            total_bid_value=total_bid_value,
            total_ask_value=total_ask_value,
            metadata={
                "spread_bps": float(spread_bps),
                "skew": float(inventory_skew),
                "size_multiplier": float(size_multiplier),
            },
        )

        # Update tracking
        self.active_quotes[symbol] = quote_set
        self.quote_history.append(quote_set)
        self.total_quotes_generated += len(bid_quotes) + len(ask_quotes)
        self.total_quote_value += total_bid_value + total_ask_value

        logger.info(
            "Generated quotes",
            symbol=symbol,
            bid_levels=len(bid_quotes),
            ask_levels=len(ask_quotes),
            total_bid_value=float(total_bid_value),
            total_ask_value=float(total_ask_value),
            spread_bps=float(spread_bps),
        )

        return quote_set

    def generate_aggressive_quotes(
        self,
        symbol: str,
        mid_price: Decimal,
        side: OrderSide,
        target_size_usdt: Decimal,
        max_price_impact_bps: Decimal = Decimal("50"),
    ) -> List[QuoteLevel]:
        """
        Generate aggressive quotes for position reduction.

        Args:
            symbol: Trading symbol
            mid_price: Current mid-market price
            side: Side to quote on (BUY to reduce short, SELL to reduce long)
            target_size_usdt: Target size to quote
            max_price_impact_bps: Maximum price impact allowed

        Returns:
            List of aggressive quote levels
        """
        quotes = []
        remaining_size = target_size_usdt
        current_distance_bps = Decimal("5")  # Start 5 bps from mid

        while (
            remaining_size > self.params.min_quote_size_usdt
            and current_distance_bps <= max_price_impact_bps
        ):
            # Calculate price for this level
            if side == OrderSide.BUY:
                # Aggressive buy (higher price)
                price = mid_price * (
                    Decimal("1") + current_distance_bps / Decimal("10000")
                )
            else:
                # Aggressive sell (lower price)
                price = mid_price * (
                    Decimal("1") - current_distance_bps / Decimal("10000")
                )

            # Calculate size for this level (use more aggressive at each level)
            level_size = min(
                remaining_size * Decimal("0.5"),  # 50% of remaining
                self.params.max_quote_size_usdt,
            )

            if level_size >= self.params.min_quote_size_usdt:
                quantity = level_size / price
                quote = QuoteLevel(
                    price=price,
                    quantity=quantity,
                    side=side,
                    layer=len(quotes),
                    spread_bps=current_distance_bps * 2,  # Effective spread
                    distance_bps=current_distance_bps,
                )
                quotes.append(quote)
                remaining_size -= level_size

            # Increase distance for next level
            current_distance_bps *= Decimal("1.5")

        logger.info(
            "Generated aggressive quotes",
            symbol=symbol,
            side=side.value,
            levels=len(quotes),
            total_size=float(target_size_usdt - remaining_size),
        )

        return quotes

    def convert_to_orders(self, quote_set: QuoteSet) -> List[Order]:
        """
        Convert quote set to order objects.

        Args:
            quote_set: Set of quotes to convert

        Returns:
            List of Order objects
        """
        orders = []

        # Convert bid quotes
        for quote in quote_set.bid_quotes:
            order = Order(
                order_id=quote.order_id,
                symbol=quote_set.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT_MAKER
                if self.params.post_only
                else OrderType.LIMIT,
                price=quote.price,
                quantity=quote.quantity,
                status="NEW",
                metadata={
                    "layer": quote.layer,
                    "quote_type": "market_making",
                    "post_only": self.params.post_only,
                },
            )
            orders.append(order)

        # Convert ask quotes
        for quote in quote_set.ask_quotes:
            order = Order(
                order_id=quote.order_id,
                symbol=quote_set.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT_MAKER
                if self.params.post_only
                else OrderType.LIMIT,
                price=quote.price,
                quantity=quote.quantity,
                status="NEW",
                metadata={
                    "layer": quote.layer,
                    "quote_type": "market_making",
                    "post_only": self.params.post_only,
                },
            )
            orders.append(order)

        return orders

    def adjust_for_competition(
        self,
        quote_set: QuoteSet,
        best_bid: Decimal,
        best_ask: Decimal,
        improvement_bps: Optional[Decimal] = None,
    ) -> QuoteSet:
        """
        Adjust quotes to stay competitive with market.

        Args:
            quote_set: Current quote set
            best_bid: Best bid in market
            best_ask: Best ask in market
            improvement_bps: Price improvement in basis points

        Returns:
            Adjusted quote set
        """
        improvement_bps = improvement_bps or self.params.price_improvement_bps

        # Adjust bid quotes
        for quote in quote_set.bid_quotes:
            if quote.layer == 0:  # Only adjust first layer
                # Improve on best bid
                improved_price = best_bid * (
                    Decimal("1") + improvement_bps / Decimal("10000")
                )
                # Don't cross the mid
                quote.price = min(
                    improved_price, quote_set.mid_price * Decimal("0.9999")
                )

        # Adjust ask quotes
        for quote in quote_set.ask_quotes:
            if quote.layer == 0:  # Only adjust first layer
                # Improve on best ask
                improved_price = best_ask * (
                    Decimal("1") - improvement_bps / Decimal("10000")
                )
                # Don't cross the mid
                quote.price = max(
                    improved_price, quote_set.mid_price * Decimal("1.0001")
                )

        return quote_set

    def validate_quotes(self, quote_set: QuoteSet) -> Tuple[bool, List[str]]:
        """
        Validate quote set for consistency and risk limits.

        Args:
            quote_set: Quote set to validate

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Check for crossed quotes
        if quote_set.best_bid and quote_set.best_ask:
            if quote_set.best_bid >= quote_set.best_ask:
                errors.append(
                    f"Crossed quotes: bid {quote_set.best_bid} >= ask {quote_set.best_ask}"
                )

        # Check minimum sizes
        for quote in quote_set.bid_quotes + quote_set.ask_quotes:
            if quote.dollar_value < self.params.min_quote_size_usdt:
                errors.append(
                    f"Quote below minimum size: {quote.dollar_value} < {self.params.min_quote_size_usdt}"
                )

        # Check maximum sizes
        for quote in quote_set.bid_quotes + quote_set.ask_quotes:
            if quote.dollar_value > self.params.max_quote_size_usdt:
                errors.append(
                    f"Quote above maximum size: {quote.dollar_value} > {self.params.max_quote_size_usdt}"
                )

        # Check price sanity (within 10% of mid)
        for quote in quote_set.bid_quotes:
            if abs(quote.price - quote_set.mid_price) / quote_set.mid_price > Decimal(
                "0.1"
            ):
                errors.append(
                    f"Bid price too far from mid: {quote.price} vs {quote_set.mid_price}"
                )

        for quote in quote_set.ask_quotes:
            if abs(quote.price - quote_set.mid_price) / quote_set.mid_price > Decimal(
                "0.1"
            ):
                errors.append(
                    f"Ask price too far from mid: {quote.price} vs {quote_set.mid_price}"
                )

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(
                "Quote validation failed", symbol=quote_set.symbol, errors=errors
            )

        return is_valid, errors

    def _calculate_quote_prices(
        self, mid_price: Decimal, spread_bps: Decimal, skew_bps: Decimal
    ) -> Tuple[Decimal, Decimal]:
        """Calculate bid and ask prices from mid, spread, and skew."""
        half_spread = spread_bps / Decimal("2") / Decimal("10000")
        skew = skew_bps / Decimal("10000")

        # Positive skew means we want to sell more (higher asks, lower bids)
        # Negative skew means we want to buy more (higher bids, lower asks)
        bid_price = mid_price * (Decimal("1") - half_spread - skew)
        ask_price = mid_price * (Decimal("1") + half_spread - skew)

        return bid_price, ask_price

    def _calculate_distance_bps(
        self, mid_price: Decimal, quote_price: Decimal
    ) -> Decimal:
        """Calculate distance from mid price in basis points."""
        if mid_price == 0:
            return Decimal("0")
        return abs(quote_price - mid_price) / mid_price * Decimal("10000")

    def _get_layer_size_distribution(self, layer: int) -> Decimal:
        """Get size distribution percentage for a layer."""
        if layer < len(self.params.layer_size_distribution):
            return self.params.layer_size_distribution[layer]
        # Default distribution for extra layers
        return Decimal("0.20")

    def get_active_quote_summary(self) -> Dict:
        """Get summary of active quotes."""
        total_bid_value = Decimal("0")
        total_ask_value = Decimal("0")
        total_quotes = 0

        for symbol, quote_set in self.active_quotes.items():
            total_bid_value += quote_set.total_bid_value
            total_ask_value += quote_set.total_ask_value
            total_quotes += len(quote_set.bid_quotes) + len(quote_set.ask_quotes)

        return {
            "symbols": list(self.active_quotes.keys()),
            "total_quotes": total_quotes,
            "total_bid_value": float(total_bid_value),
            "total_ask_value": float(total_ask_value),
            "total_value": float(total_bid_value + total_ask_value),
            "quotes_generated": self.total_quotes_generated,
            "total_quote_value": float(self.total_quote_value),
        }
