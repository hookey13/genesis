"""Simple arbitrage strategy for Sniper tier ($500-$2k)."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import structlog

from genesis.analytics.opportunity_models import (
    DirectArbitrageOpportunity,
    OpportunityStatus,
)
from genesis.core.models import Signal, SignalType
from genesis.strategies.base import BaseStrategy, StrategyConfig

logger = structlog.get_logger(__name__)


class SniperArbitrageStrategy(BaseStrategy):
    """Simple arbitrage strategy for capturing spread opportunities."""

    def __init__(self, config: StrategyConfig | None = None):
        """Initialize the strategy with configuration."""
        if config is None:
            config = StrategyConfig(
                name="SniperArbitrageStrategy",
                symbol="BTCUSDT",
                max_position_usdt=Decimal("1000"),
                risk_limit=Decimal("0.02"),  # 2% max risk per position
                tier_required="SNIPER",
                metadata={
                    "min_confidence": 0.6,
                    "min_profit_pct": 0.3,
                    "stop_loss_pct": 1.0,
                    "take_profit_pct": 0.5,
                    "position_timeout_minutes": 5,
                }
            )
        super().__init__(config)

        # Position tracking
        self.active_positions: dict[str, dict[str, Any]] = {}

        # Performance tracking
        self.performance_tracker = PerformanceTracker()

        # Kelly criterion position sizer
        self.position_sizer = KellySizer(
            max_risk_pct=config.risk_limit,
            confidence_threshold=config.metadata.get("min_confidence", 0.6)
        )

    async def analyze(self, market_data: dict[str, Any]) -> Signal | None:
        """Analyze market data for arbitrage opportunities.
        
        Args:
            market_data: Market data containing arbitrage opportunities.
            
        Returns:
            Trading signal if opportunity meets criteria, None otherwise.
        """
        try:
            # Extract opportunities from market data
            opportunities = market_data.get("arbitrage_opportunities", [])

            if not opportunities:
                return None

            # Select best opportunity based on profit and confidence
            best_opportunity = self._select_best_opportunity(opportunities)

            if not best_opportunity:
                return None

            # Validate minimum profit threshold
            min_profit = Decimal(str(self.config.metadata.get("min_profit_pct", 0.3)))
            if best_opportunity.profit_pct < min_profit:
                logger.debug(
                    f"Opportunity profit {best_opportunity.profit_pct}% "
                    f"below threshold {min_profit}%"
                )
                return None

            # Calculate position size using Kelly criterion
            position_size = await self.position_sizer.calculate_size(
                opportunity=best_opportunity,
                account_balance=market_data.get("account_balance", Decimal("1000")),
                existing_positions=self.active_positions
            )

            if position_size is None or position_size <= 0:
                logger.debug(f"Position size too small or None: {position_size}, skipping opportunity")
                return None

            # Generate trading signal
            signal = Signal(
                signal_id=str(uuid4()),
                strategy_id=str(self.config.strategy_id),
                symbol=best_opportunity.symbol,
                signal_type=SignalType.BUY,
                confidence=Decimal(str(best_opportunity.confidence_score)),
                price_target=best_opportunity.buy_price,
                stop_loss=best_opportunity.buy_price * (
                    Decimal("1") - Decimal(str(self.config.metadata.get("stop_loss_pct", 1.0))) / Decimal("100")
                ),
                take_profit=best_opportunity.buy_price * (
                    Decimal("1") + Decimal(str(self.config.metadata.get("take_profit_pct", 0.5))) / Decimal("100")
                ),
                quantity=position_size,
                metadata={
                    "opportunity_id": best_opportunity.id,
                    "profit_pct": float(best_opportunity.profit_pct),
                    "buy_exchange": best_opportunity.buy_exchange,
                    "sell_exchange": best_opportunity.sell_exchange,
                    "entry_time": datetime.now(UTC).isoformat()
                },
                timestamp=datetime.now(UTC)
            )

            # Track position
            self._track_position(signal, best_opportunity)

            # Record signal for performance tracking
            self.performance_tracker.record_signal(signal)

            logger.info(
                f"Generated arbitrage signal: {signal.symbol} "
                f"profit={best_opportunity.profit_pct}% "
                f"confidence={best_opportunity.confidence_score}"
            )

            return signal

        except Exception as e:
            logger.error(f"Error analyzing market data: {e}")
            return None

    async def manage_positions(self) -> list[Signal]:
        """Manage existing positions and generate exit signals.
        
        Returns:
            List of exit signals for position management.
        """
        exit_signals = []
        current_time = datetime.now(UTC)

        for position_id, position in list(self.active_positions.items()):
            # Check stop loss
            if self._should_stop_loss(position):
                exit_signal = self._create_exit_signal(
                    position,
                    reason="stop_loss",
                    signal_type=SignalType.SELL
                )
                exit_signals.append(exit_signal)
                self._close_position(position_id, "stop_loss")
                continue

            # Check take profit
            if self._should_take_profit(position):
                exit_signal = self._create_exit_signal(
                    position,
                    reason="take_profit",
                    signal_type=SignalType.SELL
                )
                exit_signals.append(exit_signal)
                self._close_position(position_id, "take_profit")
                continue

            # Check position timeout
            timeout_minutes = self.config.metadata.get("position_timeout_minutes", 5)
            entry_time = datetime.fromisoformat(position["entry_time"])
            if current_time - entry_time > timedelta(minutes=timeout_minutes):
                exit_signal = self._create_exit_signal(
                    position,
                    reason="timeout",
                    signal_type=SignalType.CLOSE
                )
                exit_signals.append(exit_signal)
                self._close_position(position_id, "timeout")

        return exit_signals

    async def generate_signals(self) -> list[Signal]:
        """Generate trading signals (required by BaseStrategy).
        
        Returns:
            List of trading signals.
        """
        # This method is required by BaseStrategy but we use analyze() and manage_positions()
        # for the actual signal generation in this implementation
        return await self.manage_positions()

    async def on_order_filled(self, order) -> None:
        """Handle order fill event.
        
        Args:
            order: The filled order.
        """
        logger.info(f"Order filled: {order}")
        # Update position tracking based on order

    async def on_position_closed(self, position) -> None:
        """Handle position close event.
        
        Args:
            position: The closed position.
        """
        logger.info(f"Position closed: {position}")
        # Update performance metrics

    def _select_best_opportunity(
        self, opportunities: list[DirectArbitrageOpportunity]
    ) -> DirectArbitrageOpportunity | None:
        """Select the best arbitrage opportunity.
        
        Args:
            opportunities: List of available opportunities.
            
        Returns:
            Best opportunity or None if none meet criteria.
        """
        min_confidence = self.config.metadata.get("min_confidence", 0.6)

        # Filter by confidence and active status
        valid_opportunities = [
            opp for opp in opportunities
            if opp.confidence_score >= min_confidence
            and opp.status == OpportunityStatus.ACTIVE
        ]

        if not valid_opportunities:
            return None

        # Sort by profit percentage and confidence score
        valid_opportunities.sort(
            key=lambda x: (x.profit_pct, x.confidence_score),
            reverse=True
        )

        return valid_opportunities[0]

    def _track_position(self, signal: Signal, opportunity: DirectArbitrageOpportunity) -> None:
        """Track a new position.
        
        Args:
            signal: The trading signal.
            opportunity: The arbitrage opportunity.
        """
        position_id = signal.signal_id
        self.active_positions[position_id] = {
            "signal_id": position_id,
            "symbol": signal.symbol,
            "entry_price": float(signal.price_target),
            "quantity": float(signal.quantity),
            "stop_loss": float(signal.stop_loss),
            "take_profit": float(signal.take_profit),
            "entry_time": signal.metadata["entry_time"],
            "opportunity_id": opportunity.id,
            "current_price": float(signal.price_target),
            "unrealized_pnl": 0.0
        }

    def _should_stop_loss(self, position: dict[str, Any]) -> bool:
        """Check if stop loss should trigger.
        
        Args:
            position: Position details.
            
        Returns:
            True if stop loss should trigger.
        """
        current_price = position.get("current_price", position["entry_price"])
        return current_price <= position["stop_loss"]

    def _should_take_profit(self, position: dict[str, Any]) -> bool:
        """Check if take profit should trigger.
        
        Args:
            position: Position details.
            
        Returns:
            True if take profit should trigger.
        """
        current_price = position.get("current_price", position["entry_price"])
        return current_price >= position["take_profit"]

    def _create_exit_signal(
        self,
        position: dict[str, Any],
        reason: str,
        signal_type: SignalType
    ) -> Signal:
        """Create an exit signal for a position.
        
        Args:
            position: Position to exit.
            reason: Reason for exit.
            signal_type: Type of exit signal.
            
        Returns:
            Exit signal.
        """
        return Signal(
            signal_id=str(uuid4()),
            strategy_id=str(self.config.strategy_id),
            symbol=position["symbol"],
            signal_type=signal_type,
            confidence=Decimal("1.0"),
            quantity=Decimal(str(position["quantity"])),
            metadata={
                "position_id": position["signal_id"],
                "exit_reason": reason,
                "entry_price": position["entry_price"],
                "exit_price": position.get("current_price", position["entry_price"]),
                "pnl": position.get("unrealized_pnl", 0.0)
            },
            timestamp=datetime.now(UTC)
        )

    def _close_position(self, position_id: str, reason: str) -> None:
        """Close a position and update tracking.
        
        Args:
            position_id: ID of position to close.
            reason: Reason for closing.
        """
        if position_id in self.active_positions:
            position = self.active_positions[position_id]

            # Calculate P&L
            entry_price = position["entry_price"]
            exit_price = position.get("current_price", entry_price)
            quantity = position["quantity"]
            pnl = (exit_price - entry_price) * quantity

            # Update performance metrics
            is_win = pnl > 0
            self.update_performance_metrics(Decimal(str(pnl)), is_win)
            self.performance_tracker.record_trade(position, pnl, is_win)

            # Remove from active positions
            del self.active_positions[position_id]

            logger.info(
                f"Closed position {position_id}: reason={reason}, "
                f"pnl={pnl:.2f}, win={is_win}"
            )

    async def save_state(self) -> dict[str, Any]:
        """Save strategy state for persistence.
        
        Returns:
            Dictionary containing the strategy state.
        """
        base_state = await super().save_state()
        base_state["active_positions"] = self.active_positions
        base_state["performance_metrics"] = self.performance_tracker.get_metrics()
        return base_state

    async def load_state(self, state_data: dict[str, Any]) -> None:
        """Load strategy state from persistence.
        
        Args:
            state_data: Dictionary containing the strategy state.
        """
        await super().load_state(state_data)
        self.active_positions = state_data.get("active_positions", {})

        if "performance_metrics" in state_data:
            self.performance_tracker.load_metrics(state_data["performance_metrics"])


class KellySizer:
    """Kelly criterion position sizing calculator."""

    def __init__(self, max_risk_pct: Decimal, confidence_threshold: float):
        """Initialize the Kelly sizer.
        
        Args:
            max_risk_pct: Maximum risk percentage per position.
            confidence_threshold: Minimum confidence for position sizing.
        """
        self.max_risk_pct = max_risk_pct
        self.confidence_threshold = confidence_threshold

    async def calculate_size(
        self,
        opportunity: DirectArbitrageOpportunity,
        account_balance: Decimal,
        existing_positions: dict[str, Any]
    ) -> Decimal:
        """Calculate position size using Kelly criterion.
        
        Kelly formula: f = (p * b - q) / b
        Where:
        - f = fraction of capital to wager
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = odds received on the wager
        
        Args:
            opportunity: The trading opportunity.
            account_balance: Current account balance.
            existing_positions: Dictionary of existing positions.
            
        Returns:
            Position size in base currency.
        """
        try:
            # Use confidence score as win probability
            p = Decimal(str(opportunity.confidence_score))
            q = Decimal("1") - p

            # Calculate odds from profit percentage
            # For arbitrage, the "odds" are the profit ratio
            # If we make 0.4% profit, we get 1.004 back for every 1 wagered
            # So b = 0.004 (the profit portion)
            profit_ratio = opportunity.profit_pct / Decimal("100")

            # Kelly formula: f = p - q/b where b is the odds
            # For arbitrage with high confidence and small profits,
            # we use a simplified approach
            if profit_ratio > 0 and p > q:
                # Standard Kelly: f = (p*b - q) / b
                # But for arbitrage, we're more interested in edge/odds
                # f = edge / odds = (p * profit - (1-p) * loss) / profit
                # Since arbitrage has minimal loss risk, we simplify
                kelly_fraction = p * profit_ratio * Decimal("10")  # Scale up for small profits
                # Apply Kelly fraction scaling for conservative sizing (half-Kelly)
                kelly_fraction = kelly_fraction * Decimal("0.5")
            else:
                kelly_fraction = Decimal("0")

            # Cap at maximum risk percentage (2%)
            kelly_fraction = min(kelly_fraction, self.max_risk_pct)

            # Ensure non-negative
            kelly_fraction = max(kelly_fraction, Decimal("0"))

            # Apply minimum fraction for viable positions
            if kelly_fraction > 0 and kelly_fraction < Decimal("0.001"):
                kelly_fraction = Decimal("0.001")  # Minimum 0.1% position

            # Calculate position size
            position_value = account_balance * kelly_fraction

            # Adjust for existing positions
            existing_exposure = sum(
                Decimal(str(pos.get("quantity", 0))) * Decimal(str(pos.get("entry_price", 0)))
                for pos in existing_positions.values()
            )

            available_capital = account_balance - existing_exposure
            position_value = min(position_value, available_capital * self.max_risk_pct)

            # Ensure minimum position size (avoid dust)
            min_position = Decimal("10")  # $10 minimum
            if position_value < min_position and kelly_fraction > 0:
                # If Kelly suggests a position but it's too small, use minimum
                position_value = min_position

            # Calculate quantity based on opportunity price
            if opportunity.buy_price > 0:
                quantity = position_value / opportunity.buy_price
            else:
                quantity = Decimal("0")

            return quantity

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return Decimal("0")


class PerformanceTracker:
    """Track strategy performance metrics."""

    def __init__(self):
        """Initialize the performance tracker."""
        self.trades: list[dict[str, Any]] = []
        self.signals: list[dict[str, Any]] = []
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "gross_profit": Decimal("0"),
            "gross_loss": Decimal("0"),
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": Decimal("0"),
            "recovery_time": timedelta(0)
        }

    def record_signal(self, signal: Signal) -> None:
        """Record a trading signal.
        
        Args:
            signal: The trading signal to record.
        """
        self.signals.append({
            "signal_id": signal.signal_id,
            "timestamp": signal.timestamp.isoformat(),
            "symbol": signal.symbol,
            "type": signal.signal_type.value,
            "confidence": float(signal.confidence),
            "quantity": float(signal.quantity) if signal.quantity else 0
        })

    def record_trade(self, position: dict[str, Any], pnl: float, is_win: bool) -> None:
        """Record a completed trade.
        
        Args:
            position: The position details.
            pnl: Profit/loss amount.
            is_win: Whether the trade was profitable.
        """
        self.trades.append({
            "position_id": position["signal_id"],
            "symbol": position["symbol"],
            "entry_price": position["entry_price"],
            "exit_price": position.get("current_price", position["entry_price"]),
            "quantity": position["quantity"],
            "pnl": pnl,
            "is_win": is_win,
            "timestamp": datetime.now(UTC).isoformat()
        })

        # Update metrics
        self.metrics["total_trades"] += 1

        if is_win:
            self.metrics["winning_trades"] += 1
            self.metrics["gross_profit"] += Decimal(str(abs(pnl)))
        else:
            self.metrics["losing_trades"] += 1
            self.metrics["gross_loss"] += Decimal(str(abs(pnl)))

        # Calculate win rate
        if self.metrics["total_trades"] > 0:
            self.metrics["win_rate"] = (
                self.metrics["winning_trades"] / self.metrics["total_trades"]
            )

        # Calculate profit factor
        if self.metrics["gross_loss"] > 0:
            self.metrics["profit_factor"] = float(
                self.metrics["gross_profit"] / self.metrics["gross_loss"]
            )

        # Update max drawdown
        if pnl < 0 and abs(pnl) > float(self.metrics["max_drawdown"]):
            self.metrics["max_drawdown"] = Decimal(str(abs(pnl)))

        # Calculate Sharpe ratio (simplified)
        self._calculate_sharpe_ratio()

    def _calculate_sharpe_ratio(self) -> None:
        """Calculate Sharpe ratio from trade history."""
        if len(self.trades) < 2:
            return

        returns = [trade["pnl"] for trade in self.trades]
        if not returns:
            return

        # Calculate average return
        avg_return = sum(returns) / len(returns)

        # Calculate standard deviation
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5

        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        if std_dev > 0:
            self.metrics["sharpe_ratio"] = avg_return / std_dev

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary of performance metrics.
        """
        return {
            **self.metrics,
            "gross_profit": float(self.metrics["gross_profit"]),
            "gross_loss": float(self.metrics["gross_loss"]),
            "max_drawdown": float(self.metrics["max_drawdown"]),
            "recovery_time": str(self.metrics["recovery_time"])
        }

    def load_metrics(self, metrics_data: dict[str, Any]) -> None:
        """Load metrics from saved state.
        
        Args:
            metrics_data: Dictionary of saved metrics.
        """
        self.metrics.update({
            "total_trades": metrics_data.get("total_trades", 0),
            "winning_trades": metrics_data.get("winning_trades", 0),
            "losing_trades": metrics_data.get("losing_trades", 0),
            "gross_profit": Decimal(str(metrics_data.get("gross_profit", 0))),
            "gross_loss": Decimal(str(metrics_data.get("gross_loss", 0))),
            "win_rate": metrics_data.get("win_rate", 0.0),
            "profit_factor": metrics_data.get("profit_factor", 0.0),
            "sharpe_ratio": metrics_data.get("sharpe_ratio", 0.0),
            "max_drawdown": Decimal(str(metrics_data.get("max_drawdown", 0)))
        })

