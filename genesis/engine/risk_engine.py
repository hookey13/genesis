"""
Risk management engine for Project GENESIS.

This module implements position sizing, risk limits, stop-loss calculations,
and P&L tracking with strict adherence to tier-based limits.
"""

from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Optional, ClassVar

import structlog

from genesis.analytics.kelly_sizing import KellyCalculator
from genesis.analytics.strategy_metrics import StrategyPerformanceTracker
from genesis.core.exceptions import (
    DailyLossLimitReached,
    InsufficientBalance,
    MinimumPositionSize,
    RiskLimitExceeded,
)
from genesis.core.models import (
    Account,
    ConvictionLevel,
    Position,
    PositionSide,
    Trade,
    TradingSession,
    TradingTier,
)
from genesis.utils.decorators import requires_tier

logger = structlog.get_logger(__name__)


@dataclass
class RiskDecision:
    """Result of a risk check."""
    approved: bool
    reason: Optional[str] = None
    adjusted_quantity: Optional[Decimal] = None
    warnings: list[str] = None


class RiskEngine:
    """
    Core risk management engine.

    Handles position sizing, risk limits, P&L calculations,
    and tier-based restrictions.
    """

    # Risk parameters by tier
    TIER_LIMITS: ClassVar[dict] = {
        TradingTier.SNIPER: {
            "daily_loss_limit": Decimal("25"),
            "position_risk_percent": Decimal("5"),
            "max_positions": 1,
            "stop_loss_percent": Decimal("2")
        },
        TradingTier.HUNTER: {
            "daily_loss_limit": Decimal("100"),
            "position_risk_percent": Decimal("5"),
            "max_positions": 3,
            "stop_loss_percent": Decimal("2")
        },
        TradingTier.STRATEGIST: {
            "daily_loss_limit": Decimal("500"),
            "position_risk_percent": Decimal("5"),
            "max_positions": 5,
            "stop_loss_percent": Decimal("2")
        },
        TradingTier.ARCHITECT: {
            "daily_loss_limit": Decimal("1000"),
            "position_risk_percent": Decimal("5"),
            "max_positions": 10,
            "stop_loss_percent": Decimal("2")
        }
    }

    MINIMUM_POSITION_SIZE = Decimal("10")  # $10 minimum

    def __init__(self, account: Account, session: Optional[TradingSession] = None,
                 use_kelly_sizing: bool = True):
        """
        Initialize risk engine with account and session.

        Args:
            account: Trading account
            session: Current trading session (optional)
            use_kelly_sizing: Whether to use Kelly sizing (Hunter+ feature)
        """
        self.account = account
        self.session = session
        self.tier_limits = self.TIER_LIMITS[account.tier]
        self.positions: dict[str, Position] = {}
        self.use_kelly_sizing = use_kelly_sizing and account.tier >= TradingTier.HUNTER

        # Initialize Kelly calculator and performance tracker if enabled
        if self.use_kelly_sizing:
            self.kelly_calculator = KellyCalculator(
                default_fraction=Decimal("0.25"),
                min_trades=20,
                lookback_days=30,
                max_kelly=Decimal("0.5")
            )
            self.performance_tracker = StrategyPerformanceTracker()
        else:
            self.kelly_calculator = None
            self.performance_tracker = None

        logger.info(
            "Risk engine initialized",
            account_id=account.account_id,
            tier=account.tier.value,
            balance=str(account.balance_usdt),
            kelly_sizing_enabled=self.use_kelly_sizing
        )

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: Decimal,
        stop_loss_price: Optional[Decimal] = None,
        custom_risk_percent: Optional[Decimal] = None,
        strategy_id: Optional[str] = None,
        conviction: ConvictionLevel = ConvictionLevel.MEDIUM,
        use_volatility_adjustment: bool = True
    ) -> Decimal:
        """
        Calculate position size based on risk parameters.

        Uses Kelly Criterion for Hunter+ tiers, or the 5% rule (or custom percentage)
        for Sniper tier to determine maximum position size based on account balance and stop loss.

        Args:
            symbol: Trading symbol
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price (optional, will calculate if not provided)
            custom_risk_percent: Custom risk percentage (optional, defaults to tier limit)
            strategy_id: Strategy identifier for Kelly sizing (Hunter+ feature)
            conviction: Conviction level for position sizing override (Strategist+ feature)
            use_volatility_adjustment: Whether to adjust for volatility (Hunter+ feature)

        Returns:
            Position size in base currency units

        Raises:
            InsufficientBalance: If account balance is too low
            MinimumPositionSize: If calculated size is below minimum
        """
        # Check for zero or insufficient balance early
        if self.account.balance_usdt <= 0:
            raise InsufficientBalance(
                "Account balance is zero or negative",
                required_amount=self.MINIMUM_POSITION_SIZE,
                available_amount=self.account.balance_usdt
            )

        # Check if balance is too low to meet minimum position size
        if self.account.balance_usdt < self.MINIMUM_POSITION_SIZE:
            raise MinimumPositionSize(
                f"Account balance ${self.account.balance_usdt:.2f} is below minimum position size ${self.MINIMUM_POSITION_SIZE}",
                position_size=self.account.balance_usdt,
                minimum_size=self.MINIMUM_POSITION_SIZE
            )

        # Try Kelly sizing first if enabled and strategy provided
        if self.use_kelly_sizing and strategy_id and self.kelly_calculator:
            try:
                # Get strategy performance metrics
                edge_metrics = self.performance_tracker.calculate_strategy_edge(strategy_id)

                # Check if we have enough data for Kelly
                if edge_metrics["sample_size"] >= self.kelly_calculator.min_trades:
                    # Calculate Kelly fraction
                    kelly_f = self.kelly_calculator.calculate_kelly_fraction(
                        edge_metrics["win_rate"],
                        edge_metrics["win_loss_ratio"]
                    )

                    # Get recent trades for performance adjustment
                    recent_trades = self.performance_tracker.get_recent_trades(strategy_id)
                    if recent_trades:
                        kelly_f = self.kelly_calculator.adjust_kelly_for_performance(
                            kelly_f, recent_trades
                        )

                    # Calculate base Kelly position size
                    kelly_size = self.kelly_calculator.calculate_position_size(
                        kelly_f, self.account.balance_usdt
                    )

                    # Apply conviction multiplier if Strategist tier
                    if self.account.tier >= TradingTier.STRATEGIST:
                        kelly_size = self.kelly_calculator.apply_conviction_multiplier(
                            kelly_size, conviction
                        )

                    # Apply volatility adjustment if enabled
                    if use_volatility_adjustment and recent_trades:
                        returns = [float(t.pnl_percent) for t in recent_trades[-14:]]
                        if len(returns) >= 14:
                            vol_multiplier, _ = self.kelly_calculator.calculate_volatility_multiplier(returns)
                            kelly_size = kelly_size * vol_multiplier

                    # Enforce position boundaries
                    kelly_size = self.kelly_calculator.enforce_position_boundaries(
                        kelly_size, self.account.balance_usdt, self.account.tier
                    )

                    # Convert to quantity
                    quantity = (kelly_size / entry_price).quantize(
                        Decimal("0.00000001"), rounding=ROUND_DOWN
                    )

                    logger.info(
                        "Kelly position size calculated",
                        symbol=symbol,
                        strategy_id=strategy_id,
                        kelly_fraction=str(kelly_f),
                        position_value=str(kelly_size),
                        quantity=str(quantity),
                        conviction=conviction.value
                    )

                    return quantity
                else:
                    logger.info(
                        "Insufficient trade history for Kelly sizing",
                        strategy_id=strategy_id,
                        sample_size=edge_metrics["sample_size"],
                        min_required=self.kelly_calculator.min_trades
                    )
            except Exception as e:
                logger.warning(
                    "Kelly sizing failed, falling back to fixed percentage",
                    error=str(e),
                    strategy_id=strategy_id
                )

        # Fall back to fixed percentage sizing
        risk_percent = custom_risk_percent or self.tier_limits["position_risk_percent"]
        risk_amount = (self.account.balance_usdt * risk_percent) / Decimal("100")

        # Calculate stop loss if not provided
        if stop_loss_price is None:
            stop_loss_price = self.calculate_stop_loss(entry_price, PositionSide.LONG)

        # Calculate position size based on stop loss
        price_risk = abs(entry_price - stop_loss_price)
        if price_risk == 0:
            # If no price risk, use full risk amount for position
            position_value = risk_amount
            quantity = (position_value / entry_price).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
        else:
            # Position size = Risk Amount / Price Risk per unit
            # This gives us the number of units we can buy while risking only the risk amount
            quantity = (risk_amount / price_risk).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
            position_value = quantity * entry_price

        # Ensure position value doesn't exceed account balance
        if position_value > self.account.balance_usdt:
            # If position value would exceed balance, recalculate quantity
            quantity = (self.account.balance_usdt / entry_price).quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
            position_value = quantity * entry_price

        # Ensure position meets minimum size
        if position_value < self.MINIMUM_POSITION_SIZE:
            # If the calculated position is below minimum, try to meet minimum if possible
            if self.account.balance_usdt >= self.MINIMUM_POSITION_SIZE:
                # Use minimum position size if we have enough balance
                quantity = (self.MINIMUM_POSITION_SIZE / entry_price).quantize(
                    Decimal("0.00000001"), rounding=ROUND_UP
                )
                position_value = quantity * entry_price
                # Final check that we don't exceed balance
                if position_value > self.account.balance_usdt:
                    raise MinimumPositionSize(
                        "Cannot meet minimum position size with available balance",
                        position_size=position_value,
                        minimum_size=self.MINIMUM_POSITION_SIZE
                    )
            else:
                raise MinimumPositionSize(
                    f"Position size ${position_value:.2f} is below minimum ${self.MINIMUM_POSITION_SIZE}",
                    position_size=position_value,
                    minimum_size=self.MINIMUM_POSITION_SIZE
                )

        logger.info(
            "Position size calculated",
            symbol=symbol,
            entry_price=str(entry_price),
            stop_loss=str(stop_loss_price),
            risk_percent=str(risk_percent),
            position_value=str(position_value),
            quantity=str(quantity)
        )

        return quantity

    def calculate_stop_loss(
        self,
        entry_price: Decimal,
        side: PositionSide,
        stop_loss_percent: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate stop loss price based on entry and percentage.

        Args:
            entry_price: Entry price for the position
            side: Position side (LONG or SHORT)
            stop_loss_percent: Stop loss percentage (optional, defaults to tier limit)

        Returns:
            Stop loss price
        """
        sl_percent = stop_loss_percent or self.tier_limits["stop_loss_percent"]

        if side == PositionSide.LONG:
            # For long positions, stop loss is below entry
            stop_loss = entry_price * (Decimal("1") - sl_percent / Decimal("100"))
        else:
            # For short positions, stop loss is above entry
            stop_loss = entry_price * (Decimal("1") + sl_percent / Decimal("100"))

        # Round to 8 decimal places for crypto
        stop_loss = stop_loss.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)

        logger.debug(
            "Stop loss calculated",
            entry_price=str(entry_price),
            side=side.value,
            stop_loss_percent=str(sl_percent),
            stop_loss_price=str(stop_loss)
        )

        return stop_loss

    def calculate_pnl(self, position: Position, current_price: Decimal) -> dict[str, Decimal]:
        """
        Calculate P&L for a position.

        Args:
            position: Position to calculate P&L for
            current_price: Current market price

        Returns:
            Dictionary with pnl_dollars and pnl_percent
        """
        if position.side == PositionSide.LONG:
            price_change = current_price - position.entry_price
        else:  # SHORT
            price_change = position.entry_price - current_price

        pnl_dollars = (price_change * position.quantity).quantize(
            Decimal("0.01"), rounding=ROUND_DOWN
        )
        pnl_percent = ((price_change / position.entry_price) * Decimal("100")).quantize(
            Decimal("0.0001"), rounding=ROUND_DOWN
        )

        return {
            "pnl_dollars": pnl_dollars,
            "pnl_percent": pnl_percent
        }

    async def check_risk_limits(self, order_params: dict) -> RiskDecision:
        """
        Check if an order meets risk requirements.
        
        Args:
            order_params: Dictionary with symbol, side, quantity
            
        Returns:
            RiskDecision with approval status and details
        """
        try:
            symbol = order_params.get("symbol")
            side = order_params.get("side")
            quantity = order_params.get("quantity")

            # Validate basic parameters
            if not all([symbol, side, quantity]):
                return RiskDecision(
                    approved=False,
                    reason="Missing required order parameters"
                )

            # Check daily loss limit
            if self.session and self.session.total_pnl < -self.tier_limits["max_daily_loss"]:
                return RiskDecision(
                    approved=False,
                    reason="Daily loss limit reached"
                )

            # Check position limit
            position_value = quantity * Decimal("50000")  # Approximate value
            if position_value > self.tier_limits["max_position_value"]:
                return RiskDecision(
                    approved=False,
                    reason=f"Position size exceeds tier limit of ${self.tier_limits['max_position_value']}"
                )

            # Check balance
            if self.account.balance_usdt < position_value * Decimal("0.01"):  # 1% margin
                return RiskDecision(
                    approved=False,
                    reason="Insufficient balance for margin"
                )

            return RiskDecision(
                approved=True,
                adjusted_quantity=quantity
            )

        except Exception as e:
            logger.error("Risk check failed", error=str(e))
            return RiskDecision(
                approved=False,
                reason=f"Risk check error: {e}"
            )

    def validate_order_risk(
        self,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        entry_price: Decimal,
        is_iceberg: bool = False
    ) -> None:
        """
        Validate an order against risk limits.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            entry_price: Entry price
            is_iceberg: Whether this is an iceberg order

        Raises:
            RiskLimitExceeded: If order would exceed risk limits
            DailyLossLimitReached: If daily loss limit has been reached
            InsufficientBalance: If insufficient balance
        """
        position_value = quantity * entry_price

        # For iceberg orders, validate total order value
        if is_iceberg:
            logger.info(
                "Validating iceberg order risk",
                symbol=symbol,
                total_value=str(position_value),
                quantity=str(quantity)
            )

        # Check minimum position size
        if position_value < self.MINIMUM_POSITION_SIZE:
            raise MinimumPositionSize(
                f"Position size ${position_value:.2f} is below minimum",
                position_size=position_value,
                minimum_size=self.MINIMUM_POSITION_SIZE
            )

        # Check account balance
        if position_value > self.account.balance_usdt:
            raise InsufficientBalance(
                "Insufficient balance for position",
                required_amount=position_value,
                available_amount=self.account.balance_usdt
            )

        # Check position risk percentage
        risk_percent = (position_value / self.account.balance_usdt) * Decimal("100")
        max_risk = self.tier_limits["position_risk_percent"]

        if risk_percent > max_risk:
            raise RiskLimitExceeded(
                f"Position risk {risk_percent:.2f}% exceeds maximum {max_risk}%",
                limit_type="position_risk",
                current_value=risk_percent,
                limit_value=max_risk
            )

        # Check daily loss limit if session exists
        if self.session is not None and self.session.is_daily_limit_reached():
            raise DailyLossLimitReached(
                f"Daily loss limit of ${self.tier_limits['daily_loss_limit']} reached",
                current_loss=abs(self.session.realized_pnl),
                daily_limit=self.tier_limits["daily_loss_limit"]
            )

        # Check maximum positions
        if len(self.positions) >= self.tier_limits["max_positions"]:
            raise RiskLimitExceeded(
                f"Maximum positions ({self.tier_limits['max_positions']}) reached for {self.account.tier.value} tier",
                limit_type="max_positions",
                current_value=Decimal(len(self.positions)),
                limit_value=Decimal(self.tier_limits["max_positions"])
            )

        logger.info(
            "Order risk validation passed",
            symbol=symbol,
            side=side.value,
            quantity=str(quantity),
            entry_price=str(entry_price),
            position_value=str(position_value),
            risk_percent=str(risk_percent)
        )

    def prevent_exceeding_limits(self) -> bool:
        """
        Check if any risk limits would be exceeded.

        Returns:
            True if within limits, False otherwise
        """
        # Check daily loss limit
        if self.session is not None and self.session.is_daily_limit_reached():
            logger.warning(
                "Daily loss limit reached",
                current_loss=str(abs(self.session.realized_pnl)),
                limit=str(self.tier_limits["daily_loss_limit"])
            )
            return False

        # Check position count
        if len(self.positions) >= self.tier_limits["max_positions"]:
            logger.warning(
                "Maximum positions reached",
                current_positions=len(self.positions),
                max_positions=self.tier_limits["max_positions"]
            )
            return False

        return True

    def add_position(self, position: Position) -> None:
        """Add a position to tracking."""
        self.positions[position.position_id] = position
        logger.info(
            "Position added to risk engine",
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side.value
        )

    def remove_position(self, position_id: str) -> None:
        """Remove a position from tracking."""
        if position_id in self.positions:
            del self.positions[position_id]
            logger.info("Position removed from risk engine", position_id=position_id)

    def update_all_pnl(self, price_updates: dict[str, Decimal]) -> None:
        """
        Update P&L for all positions with new prices.

        Args:
            price_updates: Dictionary of symbol -> current_price
        """
        for position in self.positions.values():
            if position.symbol in price_updates:
                position.update_pnl(price_updates[position.symbol])

    def get_total_exposure(self) -> Decimal:
        """Calculate total exposure across all positions."""
        total = sum(p.dollar_value for p in self.positions.values())
        return total.quantize(Decimal("0.01"), rounding=ROUND_UP)

    def get_total_pnl(self) -> dict[str, Decimal]:
        """Calculate total P&L across all positions."""
        total_dollars = sum(p.pnl_dollars for p in self.positions.values())
        total_value = sum(p.dollar_value for p in self.positions.values())

        if total_value > 0:
            total_percent = (total_dollars / total_value) * Decimal("100")
        else:
            total_percent = Decimal("0")

        return {
            "total_pnl_dollars": total_dollars.quantize(Decimal("0.01"), rounding=ROUND_DOWN),
            "total_pnl_percent": total_percent.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        }

    @requires_tier(TradingTier.HUNTER)
    async def calculate_position_correlations(self) -> list[tuple]:
        """
        Calculate correlations between positions (Hunter+ feature).

        Returns:
            List of (position_a, position_b, correlation) tuples
        """
        # This is a placeholder for correlation calculation
        # Actual implementation would use historical price data
        correlations = []
        positions_list = list(self.positions.values())

        for i, pos_a in enumerate(positions_list):
            for pos_b in positions_list[i+1:]:
                # Simplified correlation based on symbol similarity
                if pos_a.symbol[:3] == pos_b.symbol[:3]:
                    correlation = Decimal("0.8")
                else:
                    correlation = Decimal("0.2")

                correlations.append((pos_a, pos_b, correlation))

        return correlations

    def record_trade_result(self, strategy_id: str, trade: Trade) -> None:
        """
        Record a completed trade for Kelly sizing calculations.
        
        Args:
            strategy_id: Strategy identifier
            trade: Completed trade result
        """
        if self.performance_tracker:
            self.performance_tracker.record_trade(strategy_id, trade)
            logger.info(
                "Trade recorded for Kelly sizing",
                strategy_id=strategy_id,
                trade_id=trade.trade_id,
                pnl_dollars=str(trade.pnl_dollars)
            )

    def validate_portfolio_risk(self, positions: list[Position]) -> dict:
        """
        Validate portfolio-level risk for multi-pair trading.
        
        Args:
            positions: List of positions to validate
            
        Returns:
            RiskDecision dictionary with validation results
        """
        risk_decision = {
            "approved": True,
            "warnings": [],
            "rejections": [],
            "portfolio_exposure": Decimal("0"),
            "correlation_risk": Decimal("0"),
            "adjusted_limits": {}
        }

        # Calculate total exposure
        total_exposure = sum(p.dollar_value for p in positions)
        risk_decision["portfolio_exposure"] = total_exposure

        # Check against tier limits
        max_exposure = self.account.balance_usdt * Decimal("0.9")  # 90% max exposure
        if total_exposure > max_exposure:
            risk_decision["approved"] = False
            risk_decision["rejections"].append(
                f"Portfolio exposure ${total_exposure:.2f} exceeds limit ${max_exposure:.2f}"
            )

        # Check position count
        if len(positions) > self.tier_limits["max_positions"]:
            risk_decision["approved"] = False
            risk_decision["rejections"].append(
                f"Position count {len(positions)} exceeds tier limit {self.tier_limits['max_positions']}"
            )

        # Check for concentrated positions
        for position in positions:
            position_weight = position.dollar_value / total_exposure if total_exposure > 0 else Decimal("0")
            if position_weight > Decimal("0.4"):  # 40% concentration warning
                risk_decision["warnings"].append(
                    f"High concentration in {position.symbol}: {position_weight:.1%}"
                )

        # Check daily P&L against limits
        if self.session:
            current_pnl = self.session.realized_pnl + self.session.unrealized_pnl
            if abs(current_pnl) > self.tier_limits["daily_loss_limit"] * Decimal("0.8"):
                risk_decision["warnings"].append(
                    f"Approaching daily loss limit: ${abs(current_pnl):.2f} of ${self.tier_limits['daily_loss_limit']:.2f}"
                )

            if abs(current_pnl) >= self.tier_limits["daily_loss_limit"]:
                risk_decision["approved"] = False
                risk_decision["rejections"].append(
                    f"Daily loss limit reached: ${abs(current_pnl):.2f}"
                )

        logger.info(
            "Portfolio risk validated",
            approved=risk_decision["approved"],
            exposure=str(total_exposure),
            position_count=len(positions),
            warnings=len(risk_decision["warnings"]),
            rejections=len(risk_decision["rejections"])
        )

        return risk_decision
