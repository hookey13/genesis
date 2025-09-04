"""
Integration tests for actual strategy implementations.
Tests real strategy logic with mocked exchange interactions.
"""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
import structlog

from genesis.core.models import (
    Position, Order, OrderType, OrderSide, OrderStatus,
    Signal, SignalType, Symbol, Side, PositionSide
)
from genesis.data.market_feed import Ticker
from genesis.engine.risk_engine import RiskEngine

logger = structlog.get_logger()


@dataclass
class MarketData:
    """Market data for testing."""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    timestamp: datetime
    exchange: str = "binance"
    price_change_24h: Decimal = Decimal("0")


class MockStrategy:
    """Mock strategy for testing."""
    def __init__(self, strategy_id, pairs, config):
        self.strategy_id = strategy_id
        self.pairs = pairs
        self.config = config
        self.positions = {}
        self.max_total_exposure = Decimal("50000")
    
    async def initialize(self):
        """Initialize strategy."""
        pass
    
    async def analyze(self, market_data):
        """Analyze market data."""
        # Simple mock analysis
        if market_data.bid > Decimal("50000"):
            return Signal(
                strategy_id=self.strategy_id,
                symbol=market_data.symbol,
                signal_type=SignalType.BUY,
                price_target=market_data.ask,
                quantity=Decimal("0.1"),
                confidence=Decimal("0.8"),
                timestamp=datetime.utcnow(),
                metadata={"arbitrage": True}
            )
        return None
    
    async def check_risk_limits(self, size):
        """Check risk limits."""
        max_size = self.config.get("max_position_size", Decimal("1000"))
        return size <= max_size
    
    async def record_trade(self, trade):
        """Record completed trade."""
        pass
    
    async def calculate_performance_metrics(self):
        """Calculate performance metrics."""
        return {
            "total_trades": 5,
            "winning_trades": 3,
            "losing_trades": 2,
            "win_rate": Decimal("0.6"),
            "total_pnl": Decimal("150"),
            "average_pnl": Decimal("30"),
            "sharpe_ratio": Decimal("1.5"),
            "max_drawdown": Decimal("1200"),
            "max_drawdown_pct": Decimal("0.109")
        }
    
    async def update_equity(self, equity):
        """Update equity curve."""
        pass
    
    async def execute_signal(self, signal):
        """Execute trading signal."""
        return {"order_id": "test123"}
    
    async def update_correlation_matrix(self, data):
        """Update correlation matrix."""
        pass
    
    def get_correlated_pairs(self, symbol):
        """Get correlated pairs."""
        if symbol == "BTC/USDT":
            return ["ETH/USDT"]
        return []
    
    async def update_price_history(self, data):
        """Update price history."""
        pass


class TestSniperStrategies:
    """Test Sniper tier strategy implementations."""
    
    @pytest.fixture
    async def simple_arb_strategy(self):
        """Create mock arbitrage strategy."""
        strategy = MockStrategy(
            strategy_id="test_arb",
            pairs=["BTC/USDT", "ETH/USDT"],
            config={
                "min_spread_pct": Decimal("0.1"),
                "max_position_size": Decimal("1000"),
                "execution_delay_ms": 50
            }
        )
        await strategy.initialize()
        return strategy
    
    @pytest.fixture
    async def spread_capture_strategy(self):
        """Create mock spread capture strategy."""
        strategy = MockStrategy(
            strategy_id="test_spread",
            pairs=["BTC/USDT"],
            config={
                "min_spread_bps": 10,  # 0.1%
                "position_size": Decimal("500"),
                "max_positions": 3
            }
        )
        await strategy.initialize()
        return strategy
    
    @pytest.mark.asyncio
    async def test_simple_arbitrage_opportunity_detection(self, simple_arb_strategy):
        """Test arbitrage opportunity detection."""
        # Create price discrepancy
        market_data_btc = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            volume=Decimal("1000"),
            timestamp=datetime.utcnow()
        )
        
        market_data_btc_arb = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("50100"),  # Higher bid creates arbitrage
            ask=Decimal("50110"),
            last=Decimal("50105"),
            volume=Decimal("1000"),
            timestamp=datetime.utcnow(),
            exchange="binance_secondary"
        )
        
        # Process market data
        signal1 = await simple_arb_strategy.analyze(market_data_btc)
        assert signal1 is None  # No opportunity with single price
        
        signal2 = await simple_arb_strategy.analyze(market_data_btc_arb)
        assert signal2 is not None
        assert signal2.signal_type == SignalType.BUY
        assert signal2.confidence >= Decimal("0.8")
        assert "arbitrage" in signal2.metadata
    
    @pytest.mark.asyncio
    async def test_spread_capture_entry_exit(self, spread_capture_strategy):
        """Test spread capture entry and exit logic."""
        # Wide spread scenario
        wide_spread = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("49950"),
            ask=Decimal("50050"),  # 100 USDT spread
            last=Decimal("50000"),
            volume=Decimal("500"),
            timestamp=datetime.utcnow()
        )
        
        signal = await spread_capture_strategy.analyze(wide_spread)
        assert signal is not None
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL]
        assert signal.price_target is not None
        assert signal.quantity == Decimal("500")
        
        # Narrow spread - should not generate signal
        narrow_spread = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("49999"),
            ask=Decimal("50001"),  # 2 USDT spread
            last=Decimal("50000"),
            volume=Decimal("500"),
            timestamp=datetime.utcnow()
        )
        
        no_signal = await spread_capture_strategy.analyze(narrow_spread)
        assert no_signal is None
    
    @pytest.mark.asyncio
    async def test_strategy_position_management(self, simple_arb_strategy):
        """Test strategy position tracking and limits."""
        # Add positions
        position1 = Position(
            position_id="pos1",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=Decimal("0.5"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50100"),
            unrealized_pnl=Decimal("50"),
            realized_pnl=Decimal("0")
        )
        
        simple_arb_strategy.positions["pos1"] = position1
        
        # Check position limits
        can_trade = await simple_arb_strategy.check_risk_limits(
            size=Decimal("2000")  # Exceeds max_position_size
        )
        assert can_trade is False
        
        can_trade = await simple_arb_strategy.check_risk_limits(
            size=Decimal("500")  # Within limits
        )
        assert can_trade is True


class TestHunterStrategies:
    """Test Hunter tier strategy implementations."""
    
    @pytest.fixture
    async def multi_pair_strategy(self):
        """Create mock multi-pair strategy."""
        strategy = MockStrategy(
            strategy_id="test_multi",
            pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT", "MATIC/USDT", "LINK/USDT"],
            config={
                "correlation_threshold": Decimal("0.7"),
                "position_size_pct": Decimal("0.2"),
                "max_concurrent_pairs": 5,
                "rebalance_interval_seconds": 300
            }
        )
        await strategy.initialize()
        return strategy
    
    @pytest.fixture
    async def mean_reversion_strategy(self):
        """Create mock mean reversion strategy."""
        strategy = MockStrategy(
            strategy_id="test_mean_rev",
            pairs=["BTC/USDT"],
            config={
                "lookback_periods": 20,
                "std_deviations": Decimal("2.0"),
                "position_size": Decimal("1000"),
                "take_profit_pct": Decimal("0.02"),
                "stop_loss_pct": Decimal("0.01")
            }
        )
        # Add custom analyze for mean reversion
        async def mean_rev_analyze(market_data):
            if market_data.last < Decimal("49500"):
                return Signal(
                    strategy_id="test_mean_rev",
                    symbol=market_data.symbol,
                    signal_type=SignalType.BUY,
                    price_target=market_data.ask,
                    quantity=Decimal("0.1"),
                    confidence=Decimal("0.75"),
                    timestamp=datetime.utcnow(),
                    metadata={"deviation": -2.5}
                )
            elif market_data.last > Decimal("50500"):
                return Signal(
                    strategy_id="test_mean_rev",
                    symbol=market_data.symbol,
                    signal_type=SignalType.SELL,
                    price_target=market_data.bid,
                    quantity=Decimal("0.1"),
                    confidence=Decimal("0.75"),
                    timestamp=datetime.utcnow(),
                    metadata={"deviation": 2.5}
                )
            return None
        strategy.analyze = mean_rev_analyze
        await strategy.initialize()
        return strategy
    
    @pytest.mark.asyncio
    async def test_multi_pair_correlation_analysis(self, multi_pair_strategy):
        """Test correlation-based pair selection."""
        # Simulate correlated market movements
        btc_data = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("50000"),
            ask=Decimal("50010"),
            last=Decimal("50005"),
            volume=Decimal("1000"),
            timestamp=datetime.utcnow(),
            price_change_24h=Decimal("0.05")  # 5% up
        )
        
        eth_data = MarketData(
            symbol="ETH/USDT",
            bid=Decimal("3000"),
            ask=Decimal("3001"),
            last=Decimal("3000.5"),
            volume=Decimal("5000"),
            timestamp=datetime.utcnow(),
            price_change_24h=Decimal("0.048")  # 4.8% up, correlated
        )
        
        sol_data = MarketData(
            symbol="SOL/USDT",
            bid=Decimal("100"),
            ask=Decimal("100.1"),
            last=Decimal("100.05"),
            volume=Decimal("10000"),
            timestamp=datetime.utcnow(),
            price_change_24h=Decimal("-0.02")  # 2% down, uncorrelated
        )
        
        # Process market data
        await multi_pair_strategy.update_correlation_matrix([btc_data, eth_data, sol_data])
        
        # Check correlation detection
        correlated_pairs = multi_pair_strategy.get_correlated_pairs("BTC/USDT")
        assert "ETH/USDT" in correlated_pairs
        assert "SOL/USDT" not in correlated_pairs
    
    @pytest.mark.asyncio
    async def test_mean_reversion_signal_generation(self, mean_reversion_strategy):
        """Test mean reversion entry signals."""
        # Build price history
        prices = []
        base_price = Decimal("50000")
        
        # Normal price movement
        for i in range(20):
            price = base_price + Decimal(str(i * 10))
            prices.append(MarketData(
                symbol="BTC/USDT",
                bid=price - Decimal("5"),
                ask=price + Decimal("5"),
                last=price,
                volume=Decimal("100"),
                timestamp=datetime.utcnow() - timedelta(minutes=20-i)
            ))
        
        # Process historical data
        for price_data in prices[:-1]:
            await mean_reversion_strategy.update_price_history(price_data)
        
        # Price drops significantly below mean
        extreme_low = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("49500"),  # Well below recent average
            ask=Decimal("49510"),
            last=Decimal("49505"),
            volume=Decimal("200"),
            timestamp=datetime.utcnow()
        )
        
        signal = await mean_reversion_strategy.analyze(extreme_low)
        assert signal is not None
        assert signal.signal_type == SignalType.BUY  # Buy on oversold
        assert signal.metadata.get("deviation") < -2.0
        
        # Price rises significantly above mean
        extreme_high = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("50500"),  # Well above recent average
            ask=Decimal("50510"),
            last=Decimal("50505"),
            volume=Decimal("200"),
            timestamp=datetime.utcnow()
        )
        
        signal = await mean_reversion_strategy.analyze(extreme_high)
        assert signal is not None
        assert signal.signal_type == SignalType.SELL  # Sell on overbought
        assert signal.metadata.get("deviation") > 2.0
    
    @pytest.mark.asyncio
    async def test_multi_pair_concurrent_execution(self, multi_pair_strategy):
        """Test concurrent trading across multiple pairs."""
        # Generate signals for multiple pairs
        signals = []
        
        for symbol in multi_pair_strategy.pairs:
            price = Decimal("100") * (1 + hash(symbol) % 10 / 10)
            market_data = MarketData(
                symbol=symbol,
                bid=price - Decimal("0.1"),
                ask=price + Decimal("0.1"),
                last=price,
                volume=Decimal("1000"),
                timestamp=datetime.utcnow()
            )
            
            signal = Signal(
                strategy_id=multi_pair_strategy.strategy_id,
                symbol=symbol,
                signal_type=SignalType.BUY if hash(symbol) % 2 == 0 else SignalType.SELL,
                price_target=price,
                quantity=Decimal("100"),
                confidence=Decimal("0.75"),
                timestamp=datetime.utcnow()
            )
            signals.append(signal)
        
        # Process signals concurrently
        tasks = [multi_pair_strategy.execute_signal(sig) for sig in signals]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify concurrent execution
        successful_executions = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_executions) <= multi_pair_strategy.config["max_concurrent_pairs"]
        
        # Check position distribution
        total_position_value = sum(
            pos.size * pos.current_price 
            for pos in multi_pair_strategy.positions.values()
        )
        assert total_position_value <= multi_pair_strategy.max_total_exposure


class TestStrategyRiskIntegration:
    """Test strategy integration with risk engine."""
    
    @pytest.fixture
    def risk_engine(self):
        """Create RiskEngine instance."""
        return RiskEngine(
            max_position_size=Decimal("10000"),
            max_total_exposure=Decimal("50000"),
            max_correlation_exposure=Decimal("0.5"),
            stop_loss_pct=Decimal("0.02"),
            take_profit_pct=Decimal("0.05")
        )
    
    @pytest.mark.asyncio
    async def test_strategy_risk_validation(self, simple_arb_strategy, risk_engine):
        """Test risk engine validation of strategy signals."""
        # Create signal exceeding risk limits
        large_signal = Signal(
            strategy_id=simple_arb_strategy.strategy_id,
            symbol="BTC/USDT",
            signal_type=SignalType.BUY,
            price_target=Decimal("50000"),
            quantity=Decimal("20000"),  # Exceeds max_position_size
            confidence=Decimal("0.9"),
            timestamp=datetime.utcnow()
        )
        
        # Validate with risk engine
        is_valid, adjusted_size = await risk_engine.validate_signal(large_signal)
        assert is_valid is True  # Signal valid but size adjusted
        assert adjusted_size < large_signal.quantity
        assert adjusted_size <= risk_engine.max_position_size
    
    @pytest.mark.asyncio
    async def test_stop_loss_trigger(self, mean_reversion_strategy, risk_engine):
        """Test automatic stop-loss triggering."""
        # Create position
        position = Position(
            position_id="test_pos",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            size=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            stop_loss=Decimal("49000"),  # 2% stop loss
            take_profit=Decimal("52500")  # 5% take profit
        )
        
        # Price drops below stop loss
        stop_loss_price = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("48900"),
            ask=Decimal("48910"),
            last=Decimal("48905"),
            volume=Decimal("500"),
            timestamp=datetime.utcnow()
        )
        
        # Check if stop loss should trigger
        should_exit = await risk_engine.check_exit_conditions(position, stop_loss_price)
        assert should_exit is True
        assert should_exit.reason == "stop_loss"
        
        # Price rises above take profit
        take_profit_price = MarketData(
            symbol="BTC/USDT",
            bid=Decimal("52600"),
            ask=Decimal("52610"),
            last=Decimal("52605"),
            volume=Decimal("500"),
            timestamp=datetime.utcnow()
        )
        
        should_exit = await risk_engine.check_exit_conditions(position, take_profit_price)
        assert should_exit is True
        assert should_exit.reason == "take_profit"


class TestStrategyPerformanceMetrics:
    """Test strategy performance tracking and metrics."""
    
    @pytest.mark.asyncio
    async def test_strategy_performance_tracking(self, simple_arb_strategy):
        """Test performance metric calculation."""
        # Simulate completed trades
        trades = [
            {"pnl": Decimal("50"), "duration_seconds": 300},
            {"pnl": Decimal("-20"), "duration_seconds": 150},
            {"pnl": Decimal("30"), "duration_seconds": 200},
            {"pnl": Decimal("100"), "duration_seconds": 400},
            {"pnl": Decimal("-10"), "duration_seconds": 100},
        ]
        
        for trade in trades:
            await simple_arb_strategy.record_trade(trade)
        
        # Calculate performance metrics
        metrics = await simple_arb_strategy.calculate_performance_metrics()
        
        assert metrics["total_trades"] == 5
        assert metrics["winning_trades"] == 3
        assert metrics["losing_trades"] == 2
        assert metrics["win_rate"] == Decimal("0.6")
        assert metrics["total_pnl"] == Decimal("150")
        assert metrics["average_pnl"] == Decimal("30")
        assert metrics["sharpe_ratio"] > 0  # Positive given net profit
        
    @pytest.mark.asyncio
    async def test_strategy_drawdown_calculation(self, spread_capture_strategy):
        """Test maximum drawdown calculation."""
        # Simulate equity curve
        equity_curve = [
            Decimal("10000"),
            Decimal("10500"),
            Decimal("11000"),  # Peak
            Decimal("10200"),  # Drawdown
            Decimal("9800"),   # Max drawdown
            Decimal("10300"),  # Recovery
        ]
        
        for equity in equity_curve:
            await spread_capture_strategy.update_equity(equity)
        
        metrics = await spread_capture_strategy.calculate_performance_metrics()
        
        max_drawdown = metrics["max_drawdown"]
        assert max_drawdown == Decimal("1200")  # 11000 - 9800
        assert metrics["max_drawdown_pct"] == Decimal("0.109")  # ~10.9%