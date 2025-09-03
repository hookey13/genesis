"""Market making strategy for two-sided liquidity provision."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import structlog

from genesis.core.models import Order, OrderSide, OrderType, OrderStatus, Signal, SignalType
from genesis.strategies.base import BaseStrategy, StrategyConfig

logger = structlog.get_logger(__name__)


@dataclass
class MarketMakerConfig(StrategyConfig):
    """Configuration for market making strategy."""

    # Spread parameters
    base_spread_bps: Decimal = Decimal("10")  # 10 basis points (0.1%)
    min_spread_bps: Decimal = Decimal("5")  # Minimum 5 bps
    max_spread_bps: Decimal = Decimal("50")  # Maximum 50 bps

    # Quote parameters
    quote_layers: int = 3  # Number of quote layers
    layer_spacing_multiplier: Decimal = Decimal("2")  # 1x, 2x, 4x spacing
    quote_refresh_seconds: int = 5  # Refresh every 5 seconds
    price_move_threshold_bps: Decimal = Decimal(
        "10"
    )  # 0.1% price move triggers refresh

    # Size distribution (must sum to 1.0)
    layer_size_distribution: List[Decimal] = field(
        default_factory=lambda: [Decimal("0.40"), Decimal("0.35"), Decimal("0.25")]
    )

    # Inventory management
    max_inventory_pct: Decimal = Decimal("0.10")  # 10% of capital per symbol
    inventory_zones: Dict[str, Tuple[Decimal, Decimal]] = field(
        default_factory=lambda: {
            "GREEN": (Decimal("0"), Decimal("0.30")),  # 0-30%: Normal
            "YELLOW": (Decimal("0.30"), Decimal("0.70")),  # 30-70%: Reduce
            "RED": (Decimal("0.70"), Decimal("1.00")),  # 70-100%: Exit only
        }
    )

    # Skew parameters
    max_skew_bps: Decimal = Decimal("20")  # Maximum 20 bps skew
    skew_normalization_rate: Decimal = Decimal("0.01")  # 1% per update

    # Adverse selection
    toxic_flow_threshold: Decimal = Decimal("0.80")  # 80% one-sided fills
    adverse_spread_multiplier: Decimal = Decimal("2.0")  # 2x spread on toxic flow
    toxic_size_reduction: Decimal = Decimal("0.50")  # 50% size reduction
    recovery_fills_required: int = 100  # Fills to recover

    # Risk management
    max_daily_loss_pct: Decimal = Decimal("0.01")  # 1% daily loss limit
    max_drawdown_pct: Decimal = Decimal("0.02")  # 2% max drawdown
    min_quote_size_usdt: Decimal = Decimal("100")  # Minimum $100 quotes

    # Fee optimization
    maker_fee_bps: Decimal = Decimal("-2.5")  # -0.025% maker rebate
    taker_fee_bps: Decimal = Decimal("4.5")  # 0.045% taker fee
    min_profit_bps: Decimal = Decimal("1")  # Minimum 1 bp profit after fees

    # Volatility adjustment
    volatility_window_seconds: int = 300  # 5 minute window
    volatility_multiplier_min: Decimal = Decimal("0.5")  # 0.5x in low vol
    volatility_multiplier_max: Decimal = Decimal("3.0")  # 3x in high vol

    # Competition adjustment
    competition_check_interval: int = 10  # Check every 10 seconds
    competition_tightening_bps: Decimal = Decimal("1")  # Tighten by 1 bp
    competition_distance_bps: Decimal = Decimal("2")  # Stay within 2 bps


@dataclass
class InventoryState:
    """Track inventory positions and metrics."""

    current_position: Decimal = Decimal("0")
    max_position: Decimal = Decimal("0")
    zone: str = "GREEN"
    skew: Decimal = Decimal("0")
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_zone(self, zones: Dict[str, Tuple[Decimal, Decimal]]) -> None:
        """Update inventory zone based on current position."""
        pct = (
            abs(self.current_position) / self.max_position
            if self.max_position > 0
            else Decimal("0")
        )

        for zone_name, (low, high) in zones.items():
            if low <= pct <= high:
                self.zone = zone_name
                break

    def calculate_skew(self) -> Decimal:
        """Calculate inventory skew for price adjustment."""
        if self.max_position == 0:
            return Decimal("0")

        # Positive skew when long (lower bids, higher asks)
        # Negative skew when short (higher bids, lower asks)
        self.skew = self.current_position / self.max_position
        return self.skew


@dataclass
class AdverseSelectionTracker:
    """Track adverse selection metrics."""

    total_fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0
    toxic_flow_detected: bool = False
    recovery_fills: int = 0
    immediate_losses: int = 0
    last_check: datetime = field(default_factory=lambda: datetime.now(UTC))

    def update_fill(self, side: OrderSide, immediate_loss: bool = False) -> None:
        """Update fill statistics."""
        self.total_fills += 1
        if side == OrderSide.BUY:
            self.buy_fills += 1
        else:
            self.sell_fills += 1

        if immediate_loss:
            self.immediate_losses += 1

        # Check for recovery
        if self.toxic_flow_detected:
            self.recovery_fills += 1

    def check_toxic_flow(self, threshold: Decimal) -> bool:
        """Check if flow is toxic based on one-sided fills."""
        if self.total_fills < 10:  # Need minimum sample
            return False

        buy_ratio = Decimal(self.buy_fills) / Decimal(self.total_fills)
        sell_ratio = Decimal(self.sell_fills) / Decimal(self.total_fills)

        self.toxic_flow_detected = max(buy_ratio, sell_ratio) >= threshold
        return self.toxic_flow_detected

    def should_recover(self, required_fills: int) -> bool:
        """Check if we should recover from toxic flow."""
        if not self.toxic_flow_detected:
            return False

        if self.recovery_fills >= required_fills:
            self.reset()
            return True

        return False

    def reset(self) -> None:
        """Reset tracker after recovery."""
        self.toxic_flow_detected = False
        self.recovery_fills = 0
        self.immediate_losses = 0
        # Keep some history
        self.total_fills = max(0, self.total_fills - 100)
        self.buy_fills = max(0, self.buy_fills - 50)
        self.sell_fills = max(0, self.sell_fills - 50)


class MarketMakingStrategy(BaseStrategy):
    """Two-sided market making strategy with inventory management."""

    def __init__(self, config: MarketMakerConfig | None = None):
        """Initialize market making strategy."""
        super().__init__(config or MarketMakerConfig())
        self.config: MarketMakerConfig = self.config  # Type hint

        # Initialize components
        self.inventory = InventoryState()
        self.adverse_tracker = AdverseSelectionTracker()

        # Market data
        self.current_mid_price: Decimal = Decimal("0")
        self.current_bid: Decimal = Decimal("0")
        self.current_ask: Decimal = Decimal("0")
        self.last_refresh_time = datetime.now(UTC)
        self.last_mid_price: Decimal = Decimal("0")

        # Active quotes
        self.active_buy_orders: List[Order] = []
        self.active_sell_orders: List[Order] = []

        # Performance tracking
        self.daily_pnl: Decimal = Decimal("0")
        self.session_start = datetime.now(UTC)

        # Volatility tracking
        self.price_history: List[Tuple[datetime, Decimal]] = []
        self.current_volatility: Decimal = Decimal("1")

        # Competition tracking
        self.competitor_spread: Optional[Decimal] = None
        self.last_competition_check = datetime.now(UTC)

    async def generate_signals(self) -> List[Signal]:
        """Generate market making signals."""
        signals = []

        # Check if we should refresh quotes
        if self._should_refresh_quotes():
            # Cancel stale orders first
            cancel_signals = await self._cancel_stale_orders()
            signals.extend(cancel_signals)

            # Generate new quotes
            quote_signals = await self._generate_quotes()
            signals.extend(quote_signals)

            self.last_refresh_time = datetime.now(UTC)
            self.last_mid_price = self.current_mid_price

        # Manage inventory if needed
        inventory_signals = await self.manage_positions()
        signals.extend(inventory_signals)

        return signals

    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """Analyze market data and update internal state."""
        # Update market prices
        self.current_bid = Decimal(str(market_data.get("bid", "0")))
        self.current_ask = Decimal(str(market_data.get("ask", "0")))
        self.current_mid_price = (self.current_bid + self.current_ask) / 2

        # Track price history for volatility
        now = datetime.now(UTC)
        self.price_history.append((now, self.current_mid_price))

        # Clean old price history
        cutoff = now - timedelta(seconds=self.config.volatility_window_seconds)
        self.price_history = [(t, p) for t, p in self.price_history if t > cutoff]

        # Calculate current volatility
        self._calculate_volatility()

        # Check competition
        if (
            now - self.last_competition_check
        ).seconds > self.config.competition_check_interval:
            await self._check_competition(market_data)
            self.last_competition_check = now

        # Check adverse selection
        self.adverse_tracker.check_toxic_flow(self.config.toxic_flow_threshold)

        # No immediate signal, signals generated in generate_signals
        return None

    async def manage_positions(self) -> List[Signal]:
        """Manage inventory and generate position reduction signals if needed."""
        signals = []

        # Update inventory zone
        self.inventory.update_zone(self.config.inventory_zones)

        # If in RED zone, generate exit signals
        if self.inventory.zone == "RED":
            if self.inventory.current_position > 0:
                # Long position, need to sell
                signal = Signal(
                    strategy_id=self.config.name,
                    signal_type=SignalType.EXIT_LONG,
                    symbol=self.config.symbol,
                    strength=Decimal("1.0"),
                    entry_price=self.current_bid,
                    stop_loss=None,
                    take_profit=None,
                    position_size=abs(self.inventory.current_position)
                    * Decimal("0.5"),  # Exit half
                    metadata={"reason": "inventory_reduction", "zone": "RED"},
                )
                signals.append(signal)
            elif self.inventory.current_position < 0:
                # Short position, need to buy
                signal = Signal(
                    strategy_id=self.config.name,
                    signal_type=SignalType.EXIT_SHORT,
                    symbol=self.config.symbol,
                    strength=Decimal("1.0"),
                    entry_price=self.current_ask,
                    stop_loss=None,
                    take_profit=None,
                    position_size=abs(self.inventory.current_position)
                    * Decimal("0.5"),  # Cover half
                    metadata={"reason": "inventory_reduction", "zone": "RED"},
                )
                signals.append(signal)

        return signals

    async def on_order_filled(self, order: Order) -> None:
        """Handle order fill and update inventory."""
        # Update inventory
        if order.side == OrderSide.BUY:
            self.inventory.current_position += order.quantity
            self.adverse_tracker.update_fill(OrderSide.BUY)
        else:
            self.inventory.current_position -= order.quantity
            self.adverse_tracker.update_fill(OrderSide.SELL)

        # Calculate inventory skew
        self.inventory.calculate_skew()
        self.inventory.last_update = datetime.now(UTC)

        # Update performance metrics
        # This is simplified - real implementation would track entry prices
        fee = abs(
            order.quantity * order.price * self.config.maker_fee_bps / Decimal("10000")
        )
        self.update_performance_metrics(fee, True)  # Assume maker rebate is profit

        # Remove from active orders
        if order.side == OrderSide.BUY:
            self.active_buy_orders = [
                o for o in self.active_buy_orders if o.order_id != order.order_id
            ]
        else:
            self.active_sell_orders = [
                o for o in self.active_sell_orders if o.order_id != order.order_id
            ]

        logger.info(
            "Order filled",
            order_id=str(order.order_id),
            side=order.side.value,
            price=float(order.price),
            quantity=float(order.quantity),
            inventory=float(self.inventory.current_position),
            zone=self.inventory.zone,
        )

    async def on_position_closed(self, position) -> None:
        """Handle position close event."""
        # Update PnL
        self.daily_pnl += (
            position.realized_pnl if hasattr(position, "realized_pnl") else Decimal("0")
        )

        # Check daily loss limit
        if (
            abs(self.daily_pnl)
            > self.config.max_position_usdt * self.config.max_daily_loss_pct
        ):
            logger.warning("Daily loss limit reached, pausing strategy")
            await self.pause()

    def _should_refresh_quotes(self) -> bool:
        """Check if quotes should be refreshed."""
        now = datetime.now(UTC)

        # Time-based refresh
        if (now - self.last_refresh_time).seconds >= self.config.quote_refresh_seconds:
            return True

        # Price-based refresh
        if self.last_mid_price > 0:
            price_change_bps = abs(
                (self.current_mid_price - self.last_mid_price)
                / self.last_mid_price
                * Decimal("10000")
            )
            if price_change_bps >= self.config.price_move_threshold_bps:
                return True

        # Toxic flow detected
        if self.adverse_tracker.toxic_flow_detected:
            return True

        return False

    async def _cancel_stale_orders(self) -> List[Signal]:
        """Generate cancel signals for stale orders."""
        signals = []

        for order in self.active_buy_orders + self.active_sell_orders:
            signal = Signal(
                strategy_id=self.config.name,
                signal_type=SignalType.CLOSE,
                symbol=self.config.symbol,
                confidence=Decimal("1.0"),
                price_target=order.price,
                metadata={"order_id": str(order.order_id), "reason": "refresh"},
            )
            signals.append(signal)

        # Clear active orders (will be repopulated with new quotes)
        self.active_buy_orders.clear()
        self.active_sell_orders.clear()

        return signals

    async def _generate_quotes(self) -> List[Signal]:
        """Generate new quote orders on both sides."""
        signals = []

        # Calculate effective spread
        spread = self._calculate_effective_spread()

        # Generate quotes for each layer
        for layer in range(self.config.quote_layers):
            # Calculate layer distance
            layer_multiplier = self.config.layer_spacing_multiplier**layer
            layer_spread = spread * layer_multiplier

            # Calculate prices with inventory skew
            skew_adjustment = (
                self.inventory.skew * self.config.max_skew_bps / Decimal("10000")
            )

            # Buy quote (bid)
            buy_price = self.current_mid_price * (
                Decimal("1") - layer_spread / Decimal("10000") - skew_adjustment
            )

            # Sell quote (ask)
            sell_price = self.current_mid_price * (
                Decimal("1") + layer_spread / Decimal("10000") - skew_adjustment
            )

            # Calculate size for this layer
            if layer < len(self.config.layer_size_distribution):
                size_pct = self.config.layer_size_distribution[layer]
            else:
                size_pct = Decimal("0.20")  # Default 20%

            base_size = self.config.max_position_usdt * Decimal(
                "0.1"
            )  # 10% of max per side
            layer_size = base_size * size_pct

            # Apply adverse selection adjustments
            if self.adverse_tracker.toxic_flow_detected:
                layer_size *= self.config.toxic_size_reduction

            # Apply inventory zone adjustments
            if self.inventory.zone == "YELLOW":
                # Reduce size in yellow zone
                if self.inventory.current_position > 0:
                    # Long inventory, reduce buy size
                    buy_size = layer_size * Decimal("0.5")
                    sell_size = layer_size
                else:
                    # Short inventory, reduce sell size
                    buy_size = layer_size
                    sell_size = layer_size * Decimal("0.5")
            elif self.inventory.zone == "RED":
                # Only quote to reduce position in red zone
                if self.inventory.current_position > 0:
                    # Long inventory, only sell
                    buy_size = Decimal("0")
                    sell_size = layer_size * Decimal("1.5")
                else:
                    # Short inventory, only buy
                    buy_size = layer_size * Decimal("1.5")
                    sell_size = Decimal("0")
            else:
                # Green zone, normal sizing
                buy_size = sell_size = layer_size

            # Generate buy signal if size > minimum
            if buy_size >= self.config.min_quote_size_usdt:
                buy_quantity = buy_size / buy_price
                buy_signal = Signal(
                    strategy_id=self.config.name,
                    signal_type=SignalType.BUY,
                    symbol=self.config.symbol,
                    confidence=Decimal(
                        "0.8"
                    ),  # Market making signals are medium strength
                    price_target=buy_price,
                    quantity=buy_quantity,
                    metadata={
                        "order_type": "LIMIT",
                        "post_only": True,  # Ensure maker fee
                        "layer": layer,
                        "side": "BUY",
                    },
                )
                signals.append(buy_signal)

                # Track active order
                buy_order = Order(
                    order_id=str(uuid4()),
                    symbol=self.config.symbol,
                    side=OrderSide.BUY,
                    type=OrderType.LIMIT,
                    price=buy_price,
                    quantity=buy_quantity,
                    status=OrderStatus.PENDING,
                )
                self.active_buy_orders.append(buy_order)

            # Generate sell signal if size > minimum
            if sell_size >= self.config.min_quote_size_usdt:
                sell_quantity = sell_size / sell_price
                sell_signal = Signal(
                    strategy_id=self.config.name,
                    signal_type=SignalType.SELL,
                    symbol=self.config.symbol,
                    confidence=Decimal("0.8"),
                    price_target=sell_price,
                    quantity=sell_quantity,
                    metadata={
                        "order_type": "LIMIT",
                        "post_only": True,
                        "layer": layer,
                        "side": "SELL",
                    },
                )
                signals.append(sell_signal)

                # Track active order
                sell_order = Order(
                    order_id=str(uuid4()),
                    symbol=self.config.symbol,
                    side=OrderSide.SELL,
                    type=OrderType.LIMIT,
                    price=sell_price,
                    quantity=sell_quantity,
                    status=OrderStatus.PENDING,
                )
                self.active_sell_orders.append(sell_order)

        logger.info(
            "Generated quotes",
            buy_orders=len(self.active_buy_orders),
            sell_orders=len(self.active_sell_orders),
            spread_bps=float(spread),
            inventory_skew=float(self.inventory.skew),
        )

        return signals

    def _calculate_effective_spread(self) -> Decimal:
        """Calculate effective spread considering all factors."""
        # Start with base spread
        spread = self.config.base_spread_bps

        # Apply volatility adjustment
        spread *= self.current_volatility

        # Apply adverse selection adjustment
        if self.adverse_tracker.toxic_flow_detected:
            spread *= self.config.adverse_spread_multiplier

        # Apply competition adjustment
        if self.competitor_spread and self.competitor_spread < spread:
            # Tighten spread to stay competitive
            spread = max(
                self.competitor_spread - self.config.competition_tightening_bps,
                self.config.min_spread_bps,
            )

        # Ensure spread is profitable after fees
        min_profitable_spread = (
            self.config.min_profit_bps + abs(self.config.maker_fee_bps) * 2
        )
        spread = max(spread, min_profitable_spread)

        # Apply min/max bounds
        spread = min(
            max(spread, self.config.min_spread_bps), self.config.max_spread_bps
        )

        return spread

    def _calculate_volatility(self) -> None:
        """Calculate current volatility from price history."""
        if len(self.price_history) < 2:
            self.current_volatility = Decimal("1")
            return

        # Calculate returns
        returns = []
        for i in range(1, len(self.price_history)):
            prev_price = self.price_history[i - 1][1]
            curr_price = self.price_history[i][1]
            if prev_price > 0:
                ret = (curr_price - prev_price) / prev_price
                returns.append(ret)

        if not returns:
            self.current_volatility = Decimal("1")
            return

        # Calculate standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance.sqrt() if variance > 0 else Decimal("0")

        # Convert to volatility multiplier
        # Higher volatility -> wider spreads
        if std_dev > Decimal("0.01"):  # High volatility (>1%)
            self.current_volatility = min(
                Decimal("1") + std_dev * Decimal("100"),
                self.config.volatility_multiplier_max,
            )
        elif std_dev < Decimal("0.001"):  # Low volatility (<0.1%)
            self.current_volatility = self.config.volatility_multiplier_min
        else:
            # Normal volatility
            self.current_volatility = Decimal("1")

    async def _check_competition(self, market_data: Dict[str, Any]) -> None:
        """Check competitor spreads in the order book."""
        # Get order book data
        bids = market_data.get("bids", [])
        asks = market_data.get("asks", [])

        if not bids or not asks:
            return

        # Calculate best bid-ask spread (excluding our orders)
        # In real implementation, would filter out our own orders
        best_bid = Decimal(str(bids[0][0])) if bids else self.current_bid
        best_ask = Decimal(str(asks[0][0])) if asks else self.current_ask

        if best_bid > 0:
            competitor_spread_bps = (best_ask - best_bid) / best_bid * Decimal("10000")
            self.competitor_spread = competitor_spread_bps
