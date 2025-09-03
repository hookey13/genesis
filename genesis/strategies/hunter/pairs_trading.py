"""Statistical pairs trading strategy for Hunter tier."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID, uuid4

import pandas as pd
import structlog

from genesis.analytics.cointegration import CointegrationTester
from genesis.analytics.spread_calculator import SpreadCalculator
from genesis.core.models import Order, Position, Signal
from genesis.strategies.base import BaseStrategy, StrategyConfig, StrategyState

logger = structlog.get_logger(__name__)


@dataclass
class TradingPair:
    """Represents a cointegrated trading pair."""

    pair_id: UUID = field(default_factory=uuid4)
    symbol1: str = ""
    symbol2: str = ""
    correlation: Decimal = Decimal("0")
    cointegration_pvalue: Decimal = Decimal("1")
    hedge_ratio: Decimal = Decimal("1")
    spread_mean: Decimal = Decimal("0")
    spread_std: Decimal = Decimal("0")
    current_zscore: Decimal = Decimal("0")
    last_calibration: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = False
    position_size: Decimal = Decimal("0")
    entry_zscore: Decimal = Decimal("0")
    entry_time: datetime | None = None

    def is_cointegrated(self) -> bool:
        """Check if pair is still cointegrated."""
        return self.cointegration_pvalue < Decimal("0.05")

    def needs_recalibration(self, recalibration_hours: int = 168) -> bool:
        """Check if pair needs recalibration (default weekly)."""
        time_since_calibration = datetime.now(UTC) - self.last_calibration
        return time_since_calibration > timedelta(hours=recalibration_hours)


@dataclass
class PairsTradingConfig(StrategyConfig):
    """Configuration for pairs trading strategy."""

    max_pairs: int = 5
    correlation_threshold: Decimal = Decimal("0.8")
    cointegration_pvalue_threshold: Decimal = Decimal("0.05")
    entry_zscore: Decimal = Decimal("2.0")
    exit_zscore: Decimal = Decimal("0.5")
    stop_loss_zscore: Decimal = Decimal("3.0")
    lookback_window: int = 100
    recalibration_frequency_hours: int = 168  # Weekly
    max_holding_period_days: int = 30
    position_size_per_pair: Decimal = Decimal("0.02")  # 2% per pair
    max_correlation_between_pairs: Decimal = Decimal("0.5")
    spread_half_life_min: int = 5
    spread_half_life_max: int = 50
    min_volume_filter: Decimal = Decimal("100000")  # Min daily volume in USDT


@dataclass
class PairsTradingState(StrategyState):
    """Runtime state for pairs trading strategy."""

    active_pairs: list[TradingPair] = field(default_factory=list)
    candidate_pairs: list[TradingPair] = field(default_factory=list)
    pair_positions: dict[str, Position] = field(default_factory=dict)
    spread_history: dict[str, list[Decimal]] = field(default_factory=dict)
    pair_performance: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_scan_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    total_pairs_traded: int = 0
    successful_pairs: int = 0


class PairsTradingStrategy(BaseStrategy):
    """Statistical pairs trading strategy using cointegration."""

    def __init__(self, config: PairsTradingConfig | None = None):
        """Initialize pairs trading strategy."""
        config = config or PairsTradingConfig()
        super().__init__(config)
        self.config: PairsTradingConfig = config
        self.state: PairsTradingState = PairsTradingState()
        self.market_data_cache: dict[str, pd.DataFrame] = {}
        self.correlation_matrix: pd.DataFrame | None = None
        self.cointegration_tester = CointegrationTester(confidence_level=0.95)
        self.spread_calculator = SpreadCalculator(lookback_window=config.lookback_window)

    async def generate_signals(self) -> list[Signal]:
        """Generate trading signals for all active pairs.
        
        Returns:
            List of signals for pair trades.
        """
        signals = []

        # Scan for new pairs if needed
        if await self._should_scan_for_pairs():
            await self._scan_for_pairs()

        # Recalibrate existing pairs if needed
        for pair in self.state.active_pairs:
            if pair.needs_recalibration(self.config.recalibration_frequency_hours):
                await self._recalibrate_pair(pair)

        # Generate signals for active pairs
        for pair in self.state.active_pairs:
            signal = await self._generate_pair_signal(pair)
            if signal:
                signals.append(signal)

        # Manage existing positions
        exit_signals = await self.manage_positions()
        signals.extend(exit_signals)

        return signals

    async def analyze(self, market_data: dict[str, Any]) -> Signal | None:
        """Analyze market data for pair trading opportunities.
        
        Args:
            market_data: Current market data.
            
        Returns:
            Trading signal or None.
        """
        # Update market data cache
        symbol = market_data.get("symbol", "")
        if symbol:
            await self._update_market_data(symbol, market_data)

        # Check if we have enough data for analysis
        if len(self.market_data_cache) < 2:
            return None

        # Look for immediate pair trading opportunities
        for pair in self.state.active_pairs:
            if pair.symbol1 in self.market_data_cache and pair.symbol2 in self.market_data_cache:
                signal = await self._check_pair_entry(pair, market_data)
                if signal:
                    return signal

        return None

    async def manage_positions(self) -> list[Signal]:
        """Manage existing pair positions.
        
        Returns:
            List of exit signals.
        """
        exit_signals = []

        for pair in self.state.active_pairs:
            if not pair.is_active:
                continue

            # Check exit conditions
            exit_signal = await self._check_pair_exit(pair)
            if exit_signal:
                exit_signals.append(exit_signal)

            # Check stop loss
            stop_signal = await self._check_stop_loss(pair)
            if stop_signal:
                exit_signals.append(stop_signal)

            # Check max holding period
            if pair.entry_time:
                holding_period = datetime.now(UTC) - pair.entry_time
                if holding_period.days > self.config.max_holding_period_days:
                    timeout_signal = await self._create_timeout_exit(pair)
                    if timeout_signal:
                        exit_signals.append(timeout_signal)

        return exit_signals

    async def on_order_filled(self, order: Order) -> None:
        """Handle order fill event.
        
        Args:
            order: The filled order.
        """
        logger.info(f"Pairs trading order filled: {order}")

        # Update pair position tracking
        # Order may not have metadata, try to match by symbol
        pair_symbol = order.symbol
        for pair in self.state.active_pairs:
            # Check if the order symbol matches the pair
            if pair_symbol and (f"{pair.symbol1}/{pair.symbol2}" == pair_symbol or 
                               f"{pair.symbol2}/{pair.symbol1}" == pair_symbol):
                    pair.is_active = True
                    pair.entry_time = datetime.now(UTC)
                    pair.entry_zscore = pair.current_zscore
                    break

        # Update performance metrics
        self.state.last_update = datetime.now(UTC)

    async def on_position_closed(self, position: Position) -> None:
        """Handle position close event.
        
        Args:
            position: The closed position.
        """
        logger.info(f"Pairs trading position closed: {position}")

        # Update pair status
        # Position may not have metadata, try to match by symbol
        position_symbol = position.symbol
        for pair in self.state.active_pairs:
            # Check if the position symbol matches the pair
            if position_symbol and (f"{pair.symbol1}/{pair.symbol2}" == position_symbol or 
                                   f"{pair.symbol2}/{pair.symbol1}" == position_symbol):
                    pair.is_active = False
                    pair.entry_time = None

                    # Update performance tracking
                    self._update_pair_performance(pair, position)
                    break

        # Update strategy metrics
        is_win = position.pnl_dollars > 0 if position.pnl_dollars else False
        self.update_performance_metrics(
            position.pnl_dollars or Decimal("0"),
            is_win
        )

    def _extract_price_data(self, data: Any, symbol: str) -> Optional[Any]:
        """Safely extract price data from various data structures.
        
        Args:
            data: Market data (dict, array-like, or other structure)
            symbol: Symbol name for logging
            
        Returns:
            Price data or None if extraction fails
        """
        try:
            # Handle dict structure
            if isinstance(data, dict):
                if 'close' not in data:
                    logger.warning(f"Missing 'close' key for {symbol}")
                    return None
                return data['close']
            # Assume array-like structure
            return data
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to extract price data for {symbol}: {e}")
            return None
    
    async def _should_scan_for_pairs(self) -> bool:
        """Check if we should scan for new pairs.
        
        Returns:
            True if scan is needed.
        """
        # Scan if we have fewer than max pairs
        if len(self.state.active_pairs) < self.config.max_pairs:
            return True

        # Scan periodically (daily)
        time_since_scan = datetime.now(UTC) - self.state.last_scan_time
        if time_since_scan > timedelta(hours=24):
            return True

        return False

    async def _scan_for_pairs(self) -> None:
        """Scan market for cointegrated pairs."""
        logger.info("Scanning for cointegrated pairs")

        self.state.last_scan_time = datetime.now(UTC)

        # Get available symbols from market data cache
        symbols = list(self.market_data_cache.keys())

        if len(symbols) < 2:
            logger.warning("Not enough symbols for pair scanning")
            return

        candidate_pairs = []

        # Test all combinations
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1 = symbols[i]
                symbol2 = symbols[j]

                # Get price data
                data1 = self.market_data_cache.get(symbol1)
                data2 = self.market_data_cache.get(symbol2)

                if data1 is None or data2 is None:
                    continue

                # Extract price data safely using helper method
                price1 = self._extract_price_data(data1, symbol1)
                price2 = self._extract_price_data(data2, symbol2)
                
                if price1 is None or price2 is None:
                    continue
                
                # Validate price data has sufficient length
                try:
                    if len(price1) < self.config.lookback_window or len(price2) < self.config.lookback_window:
                        logger.debug(f"Insufficient data for pair {symbol1}/{symbol2}")
                        continue
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Invalid price data for pair {symbol1}/{symbol2}: {e}")
                    continue
                
                # Calculate correlation with validated data
                corr_metrics = self.spread_calculator.calculate_correlation(price1, price2)

                # Check correlation threshold
                if corr_metrics.pearson_correlation < self.config.correlation_threshold:
                    continue

                # Test cointegration
                coint_result = self.cointegration_tester.test_engle_granger(
                    price1,
                    price2
                )

                # Check if cointegrated
                if not coint_result.is_cointegrated:
                    continue

                # Calculate spread metrics
                spread_metrics = self.spread_calculator.calculate_spread(
                    price1,
                    price2,
                    coint_result.hedge_ratio or Decimal("1"),
                    spread_type="log"
                )

                # Check spread quality
                quality = self.spread_calculator.analyze_spread_quality(
                    spread_metrics,
                    self.config.spread_half_life_min,
                    self.config.spread_half_life_max
                )

                if quality['is_tradeable']:
                    # Create trading pair
                    pair = TradingPair(
                        symbol1=symbol1,
                        symbol2=symbol2,
                        correlation=corr_metrics.pearson_correlation,
                        cointegration_pvalue=coint_result.p_value,
                        hedge_ratio=coint_result.hedge_ratio or Decimal("1"),
                        spread_mean=spread_metrics.mean,
                        spread_std=spread_metrics.std_dev,
                        current_zscore=spread_metrics.current_zscore,
                        last_calibration=datetime.now(UTC)
                    )
                    candidate_pairs.append((quality['quality_score'], pair))

        # Sort by quality score and select best pairs
        candidate_pairs.sort(key=lambda x: x[0], reverse=True)

        # Add pairs up to max limit
        for score, pair in candidate_pairs:
            if len(self.state.active_pairs) >= self.config.max_pairs:
                break

            # Check correlation with existing pairs
            if self._check_pair_independence(pair):
                self.state.active_pairs.append(pair)
                logger.info(f"Added pair {pair.symbol1}/{pair.symbol2} with quality score {score}")

    async def _recalibrate_pair(self, pair: TradingPair) -> None:
        """Recalibrate pair parameters.
        
        Args:
            pair: The pair to recalibrate.
        """
        logger.info(f"Recalibrating pair {pair.symbol1}/{pair.symbol2}")

        # Get current data
        data1 = self.market_data_cache.get(pair.symbol1)
        data2 = self.market_data_cache.get(pair.symbol2)

        if data1 is None or data2 is None:
            logger.warning(f"Missing data for pair {pair.symbol1}/{pair.symbol2}")
            return

        # Extract price data safely
        price1 = self._extract_price_data(data1, pair.symbol1)
        price2 = self._extract_price_data(data2, pair.symbol2)
        
        if price1 is None or price2 is None:
            logger.warning(f"Failed to extract price data for pair {pair.symbol1}/{pair.symbol2}")
            return pair  # Return unchanged pair
        
        # Update correlation
        corr_metrics = self.spread_calculator.calculate_correlation(
            price1,
            price2
        )
        pair.correlation = corr_metrics.pearson_correlation

        # Re-test cointegration
        coint_result = self.cointegration_tester.test_engle_granger(
            price1,
            price2
        )
        pair.cointegration_pvalue = coint_result.p_value

        # Update hedge ratio if still cointegrated
        if coint_result.is_cointegrated and coint_result.hedge_ratio:
            pair.hedge_ratio = coint_result.hedge_ratio

            # Update spread statistics
            spread_metrics = self.spread_calculator.calculate_spread(
                price1,
                price2,
                pair.hedge_ratio,
                spread_type="log"
            )
            pair.spread_mean = spread_metrics.mean
            pair.spread_std = spread_metrics.std_dev
            pair.current_zscore = spread_metrics.current_zscore

        pair.last_calibration = datetime.now(UTC)

        # Remove pair if no longer cointegrated
        if not pair.is_cointegrated():
            logger.warning(f"Pair {pair.symbol1}/{pair.symbol2} no longer cointegrated")
            self.state.active_pairs.remove(pair)

    async def _generate_pair_signal(self, pair: TradingPair) -> Signal | None:
        """Generate signal for a specific pair.
        
        Args:
            pair: The trading pair.
            
        Returns:
            Trading signal or None.
        """
        # Skip if pair already has active position
        if pair.is_active:
            return None

        # Check entry conditions
        if abs(pair.current_zscore) >= self.config.entry_zscore:
            return await self._create_entry_signal(pair)

        return None

    async def _check_pair_entry(self, pair: TradingPair, market_data: dict[str, Any]) -> Signal | None:
        """Check if pair should be entered.
        
        Args:
            pair: The trading pair.
            market_data: Current market data.
            
        Returns:
            Entry signal or None.
        """
        # Update current z-score
        await self._update_pair_zscore(pair)

        # Check entry conditions
        if not pair.is_active and abs(pair.current_zscore) >= self.config.entry_zscore:
            return await self._create_entry_signal(pair)

        return None

    async def _check_pair_exit(self, pair: TradingPair) -> Signal | None:
        """Check if pair position should be exited.
        
        Args:
            pair: The trading pair.
            
        Returns:
            Exit signal or None.
        """
        # Update current z-score
        await self._update_pair_zscore(pair)

        # Check exit conditions (mean reversion)
        if abs(pair.current_zscore) <= self.config.exit_zscore:
            return await self._create_exit_signal(pair, "MEAN_REVERSION")

        return None

    async def _check_stop_loss(self, pair: TradingPair) -> Signal | None:
        """Check if stop loss is triggered.
        
        Args:
            pair: The trading pair.
            
        Returns:
            Stop loss signal or None.
        """
        # Check if spread has diverged too much
        if abs(pair.current_zscore) >= self.config.stop_loss_zscore:
            return await self._create_exit_signal(pair, "STOP_LOSS")

        return None

    async def _create_entry_signal(self, pair: TradingPair) -> Signal:
        """Create entry signal for pair trade.
        
        Args:
            pair: The trading pair.
            
        Returns:
            Entry signal.
        """
        # Determine direction based on z-score
        if pair.current_zscore > self.config.entry_zscore:
            # Spread is too high - short spread (buy symbol2, sell symbol1)
            direction = "SHORT_SPREAD"
        else:
            # Spread is too low - long spread (buy symbol1, sell symbol2)
            direction = "LONG_SPREAD"

        # Determine signal type based on direction
        if direction == "LONG_SPREAD":
            signal_type = "BUY"  # Buy the spread
        else:
            signal_type = "SELL"  # Sell the spread
        
        signal = Signal(
            signal_id=str(uuid4()),
            strategy_id=str(self.config.strategy_id),
            symbol=f"{pair.symbol1}/{pair.symbol2}",
            signal_type=signal_type,
            confidence=min(abs(pair.current_zscore) / self.config.entry_zscore, Decimal("1")),
            price_target=None,  # Will be set by executor
            stop_loss=None,  # Managed by z-score
            take_profit=None,  # Managed by z-score
            quantity=self.config.position_size_per_pair,
            metadata={
                "pair_id": str(pair.pair_id),
                "hedge_ratio": str(pair.hedge_ratio),
                "current_zscore": str(pair.current_zscore),
                "correlation": str(pair.correlation),
                "direction": direction,
                "signal_category": "PAIRS_ENTRY"
            }
        )

        return signal

    async def _create_exit_signal(self, pair: TradingPair, reason: str) -> Signal:
        """Create exit signal for pair trade.
        
        Args:
            pair: The trading pair.
            reason: Exit reason.
            
        Returns:
            Exit signal.
        """
        signal = Signal(
            signal_id=str(uuid4()),
            strategy_id=str(self.config.strategy_id),
            symbol=f"{pair.symbol1}/{pair.symbol2}",
            signal_type="CLOSE",
            confidence=Decimal("1"),
            price_target=None,
            stop_loss=None,
            take_profit=None,
            quantity=None,  # Close full position
            metadata={
                "pair_id": str(pair.pair_id),
                "exit_reason": reason,
                "exit_zscore": str(pair.current_zscore),
                "signal_category": "PAIRS_EXIT"
            }
        )

        return signal

    async def _create_timeout_exit(self, pair: TradingPair) -> Signal:
        """Create exit signal for max holding period.
        
        Args:
            pair: The trading pair.
            
        Returns:
            Timeout exit signal.
        """
        return await self._create_exit_signal(pair, "MAX_HOLDING_PERIOD")

    async def _update_market_data(self, symbol: str, data: dict[str, Any]) -> None:
        """Update market data cache.
        
        Args:
            symbol: The symbol.
            data: Market data.
        """
        # Initialize if needed
        if symbol not in self.market_data_cache:
            self.market_data_cache[symbol] = pd.DataFrame()

        # Create new row from market data
        new_row = pd.DataFrame([{
            'timestamp': data.get('timestamp', datetime.now()),
            'open': data.get('open', 0),
            'high': data.get('high', 0),
            'low': data.get('low', 0),
            'close': data.get('close', data.get('price', 0)),
            'volume': data.get('volume', 0)
        }])

        # Append to existing data
        self.market_data_cache[symbol] = pd.concat(
            [self.market_data_cache[symbol], new_row],
            ignore_index=True
        )

        # Keep only recent data (lookback_window * 2)
        max_rows = self.config.lookback_window * 2
        if len(self.market_data_cache[symbol]) > max_rows:
            self.market_data_cache[symbol] = self.market_data_cache[symbol].iloc[-max_rows:]

    async def _update_pair_zscore(self, pair: TradingPair) -> None:
        """Update current z-score for a pair.
        
        Args:
            pair: The trading pair.
        """
        # Get current data
        data1 = self.market_data_cache.get(pair.symbol1)
        data2 = self.market_data_cache.get(pair.symbol2)

        if data1 is None or data2 is None:
            logger.warning(f"Missing data for z-score update: {pair.symbol1}/{pair.symbol2}")
            return

        # Extract price data safely
        price1 = self._extract_price_data(data1, pair.symbol1)
        price2 = self._extract_price_data(data2, pair.symbol2)
        
        if price1 is None or price2 is None:
            logger.warning(f"Failed to extract price data for signal generation: {pair.symbol1}/{pair.symbol2}")
            return None
        
        # Calculate current spread
        spread_metrics = self.spread_calculator.calculate_spread(
            price1,
            price2,
            pair.hedge_ratio,
            spread_type="log"
        )

        # Update pair z-score
        pair.current_zscore = spread_metrics.current_zscore

    def _update_pair_performance(self, pair: TradingPair, position: Position) -> None:
        """Update performance metrics for a pair.
        
        Args:
            pair: The trading pair.
            position: The closed position.
        """
        pair_key = f"{pair.symbol1}/{pair.symbol2}"

        if pair_key not in self.state.pair_performance:
            self.state.pair_performance[pair_key] = {
                "trades": 0,
                "wins": 0,
                "total_pnl": Decimal("0"),
                "avg_holding_time": timedelta(),
                "best_trade": Decimal("0"),
                "worst_trade": Decimal("0")
            }

        perf = self.state.pair_performance[pair_key]
        perf["trades"] += 1

        # Use pnl_dollars instead of realized_pnl
        if position.pnl_dollars and position.pnl_dollars > 0:
            perf["wins"] += 1

        perf["total_pnl"] += position.pnl_dollars or Decimal("0")

        if position.pnl_dollars:
            if position.pnl_dollars > perf["best_trade"]:
                perf["best_trade"] = position.pnl_dollars
            if position.pnl_dollars < perf["worst_trade"]:
                perf["worst_trade"] = position.pnl_dollars

        self.state.total_pairs_traded += 1
        if position.pnl_dollars and position.pnl_dollars > 0:
            self.state.successful_pairs += 1

    def get_pair_statistics(self) -> dict[str, Any]:
        """Get statistics for all pairs.
        
        Returns:
            Dictionary of pair statistics.
        """
        stats = {
            "active_pairs": len(self.state.active_pairs),
            "candidate_pairs": len(self.state.candidate_pairs),
            "total_pairs_traded": self.state.total_pairs_traded,
            "successful_pairs": self.state.successful_pairs,
            "pair_performance": {}
        }

        for pair_key, perf in self.state.pair_performance.items():
            win_rate = perf["wins"] / perf["trades"] if perf["trades"] > 0 else 0
            stats["pair_performance"][pair_key] = {
                "trades": perf["trades"],
                "win_rate": float(win_rate),
                "total_pnl": float(perf["total_pnl"]),
                "best_trade": float(perf["best_trade"]),
                "worst_trade": float(perf["worst_trade"])
            }

        return stats

    def _check_pair_independence(self, new_pair: TradingPair) -> bool:
        """Check if new pair is independent enough from existing pairs.
        
        Args:
            new_pair: The pair to check.
            
        Returns:
            True if pair is independent enough.
        """
        for existing_pair in self.state.active_pairs:
            # Check if pairs share a symbol
            shared_symbols = (
                new_pair.symbol1 in (existing_pair.symbol1, existing_pair.symbol2) or
                new_pair.symbol2 in (existing_pair.symbol1, existing_pair.symbol2)
            )

            if shared_symbols:
                # If they share a symbol, they're correlated
                return False

        return True
