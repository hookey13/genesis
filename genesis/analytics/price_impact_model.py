"""Price Impact Model for trade execution analysis."""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import structlog

from genesis.exchange.order_book_manager import OrderBookSnapshot

logger = structlog.get_logger(__name__)


@dataclass
class PriceImpactEstimate:
    """Price impact estimation for an order."""

    symbol: str
    side: str
    quantity: Decimal
    temporary_impact: Decimal  # Temporary price movement
    permanent_impact: Decimal  # Permanent price change
    total_impact: Decimal  # Total expected impact
    expected_price: Decimal  # Expected execution price
    slippage_bps: int  # Slippage in basis points
    kyle_lambda: Decimal  # Kyle's lambda coefficient
    market_depth: Decimal  # Estimated market depth
    confidence: Decimal  # Confidence in estimate

    def is_acceptable(self, max_slippage_bps: int = 50) -> bool:
        """Check if impact is within acceptable limits."""
        return self.slippage_bps <= max_slippage_bps


class PriceImpactModel:
    """Models price impact for order execution."""

    def __init__(self, alpha: Decimal = Decimal("0.1"), beta: Decimal = Decimal("0.5")):
        """Initialize price impact model.

        Args:
            alpha: Temporary impact coefficient
            beta: Permanent impact coefficient
        """
        self.alpha = alpha
        self.beta = beta
        self.market_depth_estimates: dict[str, Decimal] = {}
        self.kyle_lambda_estimates: dict[str, Decimal] = {}
        self.execution_history: dict[str, list[tuple[Decimal, Decimal, Decimal]]] = {}

    def estimate_impact(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_book: OrderBookSnapshot | None = None,
        participation_rate: Decimal = Decimal("0.1"),
    ) -> PriceImpactEstimate:
        """Estimate price impact for an order.

        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            order_book: Current order book
            participation_rate: Expected participation rate

        Returns:
            Price impact estimate
        """
        # Get current mid price
        if order_book and order_book.mid_price:
            mid_price = order_book.mid_price
        else:
            mid_price = Decimal("1")  # Default

        # Estimate market depth
        market_depth = self._estimate_market_depth(symbol, order_book)

        # Calculate Kyle's lambda
        kyle_lambda = self._calculate_kyle_lambda(symbol, quantity, market_depth)

        # Linear impact model
        temporary_impact = self.alpha * quantity / market_depth
        permanent_impact = self.beta * kyle_lambda * quantity

        # Total impact
        total_impact = temporary_impact + permanent_impact

        # Expected execution price
        if side == "buy":
            expected_price = mid_price * (Decimal("1") + total_impact)
        else:
            expected_price = mid_price * (Decimal("1") - total_impact)

        # Calculate slippage
        slippage = abs(expected_price - mid_price) / mid_price
        slippage_bps = int(slippage * Decimal("10000"))

        # Confidence based on data availability
        confidence = self._calculate_confidence(symbol, order_book)

        return PriceImpactEstimate(
            symbol=symbol,
            side=side,
            quantity=quantity,
            temporary_impact=temporary_impact,
            permanent_impact=permanent_impact,
            total_impact=total_impact,
            expected_price=expected_price,
            slippage_bps=slippage_bps,
            kyle_lambda=kyle_lambda,
            market_depth=market_depth,
            confidence=confidence,
        )

    def _estimate_market_depth(
        self, symbol: str, order_book: OrderBookSnapshot | None
    ) -> Decimal:
        """Estimate market depth from order book or history.

        Args:
            symbol: Trading symbol
            order_book: Current order book

        Returns:
            Estimated market depth
        """
        if order_book:
            # Calculate depth from order book
            bid_depth = sum(level.quantity for level in order_book.bids[:10])
            ask_depth = sum(level.quantity for level in order_book.asks[:10])
            depth = (bid_depth + ask_depth) / Decimal("2")

            # Update estimate
            self.market_depth_estimates[symbol] = depth
            return depth

        # Use historical estimate
        return self.market_depth_estimates.get(symbol, Decimal("100"))

    def _calculate_kyle_lambda(
        self, symbol: str, quantity: Decimal, market_depth: Decimal
    ) -> Decimal:
        """Calculate Kyle's lambda (price impact coefficient).

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            market_depth: Market depth

        Returns:
            Kyle's lambda
        """
        if symbol in self.kyle_lambda_estimates:
            # Use historical estimate
            base_lambda = self.kyle_lambda_estimates[symbol]
        else:
            # Default based on market depth
            base_lambda = Decimal("1") / (market_depth * Decimal("100"))

        # Adjust for order size
        size_adjustment = min(quantity / market_depth, Decimal("1"))
        kyle_lambda = base_lambda * (Decimal("1") + size_adjustment)

        return kyle_lambda

    def _calculate_confidence(
        self, symbol: str, order_book: OrderBookSnapshot | None
    ) -> Decimal:
        """Calculate confidence in impact estimate.

        Args:
            symbol: Trading symbol
            order_book: Order book data

        Returns:
            Confidence score (0-1)
        """
        confidence = Decimal("0.5")

        # Order book available
        if order_book:
            confidence += Decimal("0.2")
            # Good depth
            if len(order_book.bids) >= 10 and len(order_book.asks) >= 10:
                confidence += Decimal("0.1")

        # Historical data available
        if symbol in self.execution_history:
            history_size = len(self.execution_history[symbol])
            if history_size >= 100:
                confidence += Decimal("0.2")
            elif history_size >= 10:
                confidence += Decimal("0.1")

        return min(confidence, Decimal("1"))

    def update_from_execution(
        self,
        symbol: str,
        quantity: Decimal,
        expected_price: Decimal,
        actual_price: Decimal,
    ) -> None:
        """Update model from actual execution.

        Args:
            symbol: Trading symbol
            quantity: Executed quantity
            expected_price: Expected price
            actual_price: Actual execution price
        """
        # Store execution data
        if symbol not in self.execution_history:
            self.execution_history[symbol] = []

        impact = abs(actual_price - expected_price) / expected_price
        self.execution_history[symbol].append((quantity, expected_price, actual_price))

        # Update Kyle's lambda estimate
        if len(self.execution_history[symbol]) >= 10:
            self._recalibrate_kyle_lambda(symbol)

    def _recalibrate_kyle_lambda(self, symbol: str) -> None:
        """Recalibrate Kyle's lambda from execution history.

        Args:
            symbol: Trading symbol
        """
        history = self.execution_history[symbol][-100:]  # Last 100 trades

        impacts = []
        quantities = []

        for qty, expected, actual in history:
            impact = abs(actual - expected) / expected
            impacts.append(float(impact))
            quantities.append(float(qty))

        # Simple linear regression
        if len(impacts) >= 10:
            # Kyle's lambda = impact / quantity
            lambdas = [i / q for i, q in zip(impacts, quantities, strict=False) if q > 0]
            if lambdas:
                self.kyle_lambda_estimates[symbol] = Decimal(str(np.median(lambdas)))

    def get_optimal_slice_size(
        self, symbol: str, total_quantity: Decimal, max_impact_bps: int = 10
    ) -> Decimal:
        """Calculate optimal slice size for large orders.

        Args:
            symbol: Trading symbol
            total_quantity: Total order size
            max_impact_bps: Maximum acceptable impact per slice

        Returns:
            Optimal slice size
        """
        market_depth = self.market_depth_estimates.get(symbol, Decimal("100"))

        # Calculate slice size for target impact
        max_impact = Decimal(str(max_impact_bps)) / Decimal("10000")

        # Rearrange impact formula: impact = alpha * quantity / depth
        # quantity = impact * depth / alpha
        optimal_size = max_impact * market_depth / self.alpha

        # Don't exceed 10% of total
        max_slice = total_quantity * Decimal("0.1")

        return min(optimal_size, max_slice)
