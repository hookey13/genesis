"""Mean Reversion Strategy for Hunter Tier.

This strategy implements mean reversion trading using Bollinger Bands and RSI,
designed for the Hunter tier ($2k-$10k) with multi-pair support and 
sophisticated risk management.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import structlog

from genesis.core.constants import TradingTier
from genesis.core.models import Order, Position, Signal, SignalType
from genesis.strategies.base import BaseStrategy, StrategyConfig

logger = structlog.get_logger(__name__)


@dataclass
class MeanReversionConfig(StrategyConfig):
    """Configuration for Mean Reversion Strategy."""

    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std_dev: float = 2.0

    # RSI parameters
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Market regime detection
    atr_period: int = 14
    regime_lookback_periods: int = 100

    # Risk management
    max_concurrent_positions: int = 5
    max_correlation: float = 0.7
    stop_loss_atr_multiplier: float = 2.0
    max_position_percent: Decimal = Decimal("0.05")  # 5% max per position

    # Multi-pair support
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])

    # Cache settings
    cache_ttl_seconds: int = 300  # 5 minutes

    # Performance targets
    target_sharpe_ratio: Decimal = Decimal("1.5")
    max_drawdown_percent: Decimal = Decimal("0.05")  # 5%

    def __post_init__(self):
        """Set Hunter tier requirements."""
        self.tier_required = TradingTier.HUNTER.value
        self.name = self.name or "MeanReversionStrategy"


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy implementation for Hunter tier."""

    def __init__(self, config: MeanReversionConfig | None = None):
        """Initialize the Mean Reversion Strategy."""
        self.config = config or MeanReversionConfig()
        super().__init__(self.config)

        # Market data storage
        self.price_history: dict[str, pd.DataFrame] = {}
        self.indicators_cache: dict[str, dict[str, Any]] = {}
        self.cache_timestamps: dict[str, datetime] = {}

        # Portfolio tracking
        self.symbol_positions: dict[str, Position] = {}
        self.correlation_matrix: pd.DataFrame | None = None

        # Performance tracking
        self.entry_signals: dict[str, Signal] = {}
        self.realized_pnl: list[Decimal] = []
        self.trade_history: list[dict[str, Any]] = []

        logger.info(
            "MeanReversionStrategy initialized",
            strategy_id=str(self.strategy_id),
            symbols=self.config.symbols,
            tier=self.config.tier_required
        )

    async def generate_signals(self) -> list[Signal]:
        """Generate trading signals based on mean reversion logic.
        
        Returns:
            List of signals for all monitored symbols.
        """
        signals = []

        try:
            # Generate signals for each symbol
            for symbol in self.config.symbols:
                # Check if we have enough data
                if not self._has_sufficient_data(symbol):
                    continue

                # Get or calculate indicators
                indicators = await self._get_indicators(symbol)
                if not indicators:
                    continue

                # Check market regime
                if not await self._is_ranging_market(symbol, indicators):
                    logger.debug(f"Skipping {symbol} - trending market detected")
                    continue

                # Generate entry signal
                signal = await self._generate_entry_signal(symbol, indicators)
                if signal and await self._validate_signal(signal):
                    signals.append(signal)

            # Manage existing positions
            exit_signals = await self.manage_positions()
            signals.extend(exit_signals)

        except Exception as e:
            logger.error(f"Error generating signals: {e}", exc_info=True)

        return signals

    async def analyze(self, market_data: dict[str, Any]) -> Signal | None:
        """Analyze market data and generate trading signal.
        
        Args:
            market_data: Market data containing price, volume, etc.
            
        Returns:
            Trading signal or None if no opportunity.
        """
        try:
            symbol = market_data.get("symbol")
            if not symbol or symbol not in self.config.symbols:
                return None

            # Update price history
            await self._update_price_history(symbol, market_data)

            # Check if we need to generate a signal
            if not self._has_sufficient_data(symbol):
                return None

            # Get indicators
            indicators = await self._get_indicators(symbol)
            if not indicators:
                return None

            # Check market regime
            if not await self._is_ranging_market(symbol, indicators):
                return None

            # Generate and validate signal
            signal = await self._generate_entry_signal(symbol, indicators)
            if signal and await self._validate_signal(signal):
                return signal

        except Exception as e:
            logger.error(f"Error analyzing market data: {e}", exc_info=True)

        return None

    async def manage_positions(self) -> list[Signal]:
        """Manage existing positions and generate exit signals.
        
        Returns:
            List of exit signals for position management.
        """
        exit_signals = []

        try:
            for symbol, position in self.symbol_positions.items():
                # Get current indicators
                indicators = await self._get_indicators(symbol)
                if not indicators:
                    continue

                # Check exit conditions
                exit_signal = await self._check_exit_conditions(position, indicators)
                if exit_signal:
                    exit_signals.append(exit_signal)

        except Exception as e:
            logger.error(f"Error managing positions: {e}", exc_info=True)

        return exit_signals

    async def on_order_filled(self, order: Order) -> None:
        """Handle order fill event.
        
        Args:
            order: The filled order.
        """
        try:
            symbol = order.symbol

            # Update position tracking
            if order.side == "BUY":
                # Create or update position
                position = Position(
                    position_id=str(order.order_id),
                    account_id=str(self.strategy_id),
                    symbol=symbol,
                    side="LONG",
                    entry_price=order.filled_price,
                    quantity=order.filled_quantity,
                    dollar_value=order.filled_price * order.filled_quantity,
                    created_at=datetime.now(UTC)
                )
                self.symbol_positions[symbol] = position
                self.state.positions.append(position)

                logger.info(
                    f"Position opened for {symbol}",
                    position_id=position.position_id,
                    entry_price=float(position.entry_price),
                    quantity=float(position.quantity)
                )

            elif order.side == "SELL":
                # Close position
                if symbol in self.symbol_positions:
                    position = self.symbol_positions[symbol]

                    # Calculate P&L
                    pnl = (order.filled_price - position.entry_price) * position.quantity
                    is_win = pnl > 0

                    # Update performance metrics
                    self.update_performance_metrics(pnl, is_win)
                    self.realized_pnl.append(pnl)

                    # Record trade
                    self.trade_history.append({
                        "symbol": symbol,
                        "entry_price": float(position.entry_price),
                        "exit_price": float(order.filled_price),
                        "quantity": float(position.quantity),
                        "pnl": float(pnl),
                        "is_win": is_win,
                        "duration": (datetime.now(UTC) - position.created_at).total_seconds(),
                        "timestamp": datetime.now(UTC).isoformat()
                    })

                    # Remove position
                    del self.symbol_positions[symbol]
                    self.state.positions = [p for p in self.state.positions if p.symbol != symbol]

                    logger.info(
                        f"Position closed for {symbol}",
                        pnl=float(pnl),
                        is_win=is_win,
                        exit_price=float(order.filled_price)
                    )

        except Exception as e:
            logger.error(f"Error handling order fill: {e}", exc_info=True)

    async def on_position_closed(self, position: Position) -> None:
        """Handle position close event.
        
        Args:
            position: The closed position.
        """
        try:
            symbol = position.symbol

            # Remove from tracking
            if symbol in self.symbol_positions:
                del self.symbol_positions[symbol]

            # Update state
            self.state.positions = [p for p in self.state.positions if p.position_id != position.position_id]

            logger.info(
                "Position closed event handled",
                symbol=symbol,
                position_id=position.position_id
            )

        except Exception as e:
            logger.error(f"Error handling position close: {e}", exc_info=True)

    async def _update_price_history(self, symbol: str, market_data: dict[str, Any]) -> None:
        """Update price history with new market data."""
        try:
            # Extract price data
            price = Decimal(str(market_data.get("price", 0)))
            volume = Decimal(str(market_data.get("volume", 0)))
            timestamp = market_data.get("timestamp", datetime.now(UTC))

            # Create new row
            new_row = pd.DataFrame({
                "timestamp": [timestamp],
                "price": [float(price)],
                "volume": [float(volume)],
                "high": [float(market_data.get("high", price))],
                "low": [float(market_data.get("low", price))],
                "close": [float(price)]
            })

            # Update or create price history
            if symbol not in self.price_history:
                self.price_history[symbol] = new_row
            else:
                self.price_history[symbol] = pd.concat([
                    self.price_history[symbol],
                    new_row
                ], ignore_index=True)

                # Keep only recent data (e.g., last 1000 candles)
                if len(self.price_history[symbol]) > 1000:
                    self.price_history[symbol] = self.price_history[symbol].iloc[-1000:]

        except Exception as e:
            logger.error(f"Error updating price history for {symbol}: {e}")

    async def _get_indicators(self, symbol: str) -> dict[str, Any] | None:
        """Get or calculate technical indicators for a symbol."""
        try:
            # Check cache
            if symbol in self.indicators_cache:
                cache_time = self.cache_timestamps.get(symbol)
                if cache_time and (datetime.now(UTC) - cache_time).seconds < self.config.cache_ttl_seconds:
                    return self.indicators_cache[symbol]

            # Get price data
            if symbol not in self.price_history:
                return None

            df = self.price_history[symbol]
            if len(df) < max(self.config.bb_period, self.config.rsi_period, self.config.atr_period):
                return None

            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df["close"])

            # Calculate RSI
            rsi = self._calculate_rsi(df["close"])

            # Calculate ATR
            atr = self._calculate_atr(df)

            # Create indicators dict
            indicators = {
                "bb_upper": bb_upper.iloc[-1] if len(bb_upper) > 0 else None,
                "bb_middle": bb_middle.iloc[-1] if len(bb_middle) > 0 else None,
                "bb_lower": bb_lower.iloc[-1] if len(bb_lower) > 0 else None,
                "rsi": rsi.iloc[-1] if len(rsi) > 0 else None,
                "atr": atr.iloc[-1] if len(atr) > 0 else None,
                "current_price": df["close"].iloc[-1],
                "timestamp": datetime.now(UTC)
            }

            # Update cache
            self.indicators_cache[symbol] = indicators
            self.cache_timestamps[symbol] = datetime.now(UTC)

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None

    def _calculate_bollinger_bands(self, prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=self.config.bb_period).mean()
        std = prices.rolling(window=self.config.bb_period).std()
        upper = middle + (std * self.config.bb_std_dev)
        lower = middle - (std * self.config.bb_std_dev)
        return upper, middle, lower

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_period).mean()

        # Avoid division by zero
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.config.atr_period).mean()
        return atr

    async def _is_ranging_market(self, symbol: str, indicators: dict[str, Any]) -> bool:
        """Check if market is in ranging (non-trending) regime."""
        try:
            if symbol not in self.price_history:
                return False

            df = self.price_history[symbol]
            if len(df) < self.config.regime_lookback_periods:
                return True  # Default to ranging if insufficient data

            # Calculate ADX for trend strength (simplified version)
            # ADX > 25 indicates trending market
            prices = df["close"].tail(self.config.regime_lookback_periods)

            # Simple trend detection using price range vs volatility
            price_range = prices.max() - prices.min()
            avg_price = prices.mean()
            range_percent = (price_range / avg_price) * 100

            # If price range is less than 15% over lookback period, consider it ranging
            # (More lenient for backtesting purposes)
            is_ranging = range_percent < 15

            # Ensure we return a Python bool, not numpy.bool_
            return bool(is_ranging)

        except Exception as e:
            logger.error(f"Error checking market regime for {symbol}: {e}")
            return False

    async def _generate_entry_signal(self, symbol: str, indicators: dict[str, Any]) -> Signal | None:
        """Generate entry signal based on indicators."""
        try:
            current_price = Decimal(str(indicators["current_price"]))
            bb_upper = indicators["bb_upper"]
            bb_lower = indicators["bb_lower"]
            bb_middle = indicators["bb_middle"]
            rsi = indicators["rsi"]
            atr = indicators["atr"]

            # Check if we already have a position
            if symbol in self.symbol_positions:
                return None

            signal = None

            # Buy signal: Price near lower band and RSI oversold
            if current_price <= Decimal(str(bb_lower * 1.01)) and rsi < self.config.rsi_oversold:
                # Calculate position size based on volatility
                position_size = await self._calculate_position_size(symbol, current_price, Decimal(str(atr)))

                if position_size > 0:
                    signal = Signal(
                        strategy_id=str(self.strategy_id),
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        confidence=Decimal(str((self.config.rsi_oversold - rsi) / self.config.rsi_oversold)),
                        price_target=Decimal(str(bb_middle)),
                        stop_loss=current_price - (Decimal(str(atr)) * Decimal(str(self.config.stop_loss_atr_multiplier))),
                        take_profit=Decimal(str(bb_middle)),
                        quantity=position_size,
                        metadata={
                            "strategy": "mean_reversion",
                            "rsi": float(rsi),
                            "bb_position": "lower",
                            "atr": float(atr)
                        }
                    )

            # Short signal (if supported): Price near upper band and RSI overbought
            elif current_price >= Decimal(str(bb_upper * 0.99)) and rsi > self.config.rsi_overbought:
                # For now, we'll generate a HOLD signal as shorting requires additional setup
                logger.debug(f"Potential short signal for {symbol}, but shorting not implemented")

            return signal

        except Exception as e:
            logger.error(f"Error generating entry signal for {symbol}: {e}")
            return None

    async def _check_exit_conditions(self, position: Position, indicators: dict[str, Any]) -> Signal | None:
        """Check if position should be exited."""
        try:
            current_price = Decimal(str(indicators["current_price"]))
            bb_middle = Decimal(str(indicators["bb_middle"]))
            atr = Decimal(str(indicators["atr"]))

            # Calculate P&L
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_percent = (current_price - position.entry_price) / position.entry_price

            # Exit conditions
            should_exit = False
            exit_reason = ""

            # 1. Mean reversion achieved (price returned to middle band)
            if abs(current_price - bb_middle) / bb_middle < Decimal("0.01"):  # Within 1% of middle band
                should_exit = True
                exit_reason = "mean_reversion_target"

            # 2. Stop loss hit
            stop_loss = position.entry_price - (atr * Decimal(str(self.config.stop_loss_atr_multiplier)))
            if current_price <= stop_loss:
                should_exit = True
                exit_reason = "stop_loss"

            # 3. Time-based exit (position held too long)
            position_age = datetime.now(UTC) - position.created_at
            if position_age > timedelta(days=3):  # Exit after 3 days
                should_exit = True
                exit_reason = "time_exit"

            # 4. Profit target hit (alternative to mean reversion)
            if pnl_percent > Decimal("0.02"):  # 2% profit
                should_exit = True
                exit_reason = "profit_target"

            if should_exit:
                return Signal(
                    strategy_id=str(self.strategy_id),
                    symbol=position.symbol,
                    signal_type=SignalType.CLOSE,
                    confidence=Decimal("0.9"),
                    quantity=position.quantity,
                    metadata={
                        "strategy": "mean_reversion",
                        "exit_reason": exit_reason,
                        "pnl": float(pnl),
                        "pnl_percent": float(pnl_percent),
                        "position_id": position.position_id
                    }
                )

            return None

        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return None

    async def _calculate_position_size(self, symbol: str, price: Decimal, atr: Decimal) -> Decimal:
        """Calculate position size based on volatility and risk limits."""
        try:
            # Get account balance (would come from account service in production)
            account_balance = self.config.max_position_usdt * Decimal("20")  # Assume 20x max position as total

            # Calculate position value based on Kelly criterion (simplified)
            # Using fixed fraction for now
            position_value = account_balance * self.config.max_position_percent

            # Adjust for volatility (lower size for higher volatility)
            volatility_adjustment = Decimal("1") / (Decimal("1") + (atr / price))
            position_value *= volatility_adjustment

            # Calculate quantity
            quantity = position_value / price

            # Apply limits
            max_position_value = self.config.max_position_usdt
            if position_value > max_position_value:
                quantity = max_position_value / price

            return quantity.quantize(Decimal("0.001"))  # Round to 3 decimal places

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return Decimal("0")

    async def _validate_signal(self, signal: Signal) -> bool:
        """Validate signal against risk limits and portfolio constraints."""
        try:
            # Check maximum concurrent positions
            if len(self.symbol_positions) >= self.config.max_concurrent_positions:
                logger.debug(f"Max concurrent positions reached ({self.config.max_concurrent_positions})")
                return False

            # Check correlation limits (simplified - would use actual correlation matrix in production)
            # For now, just limit one position per symbol
            if signal.symbol in self.symbol_positions:
                logger.debug(f"Already have position in {signal.symbol}")
                return False

            # Check drawdown limit
            if self.state.max_drawdown > self.config.max_drawdown_percent * self.config.max_position_usdt:
                logger.warning("Max drawdown limit reached")
                return False

            # Signal is valid
            return True

        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False

    def _has_sufficient_data(self, symbol: str) -> bool:
        """Check if we have enough historical data for analysis."""
        if symbol not in self.price_history:
            return False

        min_periods = max(
            self.config.bb_period,
            self.config.rsi_period,
            self.config.atr_period,
            self.config.regime_lookback_periods
        )

        return len(self.price_history[symbol]) >= min_periods

    async def calculate_sharpe_ratio(self) -> Decimal:
        """Calculate Sharpe ratio for performance tracking."""
        try:
            if len(self.realized_pnl) < 2:
                return Decimal("0")

            returns = pd.Series([float(pnl) for pnl in self.realized_pnl])

            # Assuming daily returns and 252 trading days
            mean_return = returns.mean()
            std_return = returns.std()

            if std_return == 0:
                return Decimal("0")

            sharpe = Decimal(str(mean_return / std_return * np.sqrt(252)))
            return sharpe.quantize(Decimal("0.01"))

        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return Decimal("0")
