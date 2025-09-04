"""Strategy fixtures for testing."""

from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock
import structlog
from datetime import datetime

from genesis.strategies.base import BaseStrategy
from genesis.core.models import OrderSide

logger = structlog.get_logger(__name__)


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name: str = "mock_strategy"):
        super().__init__()
        self.name = name
        self.signals_generated = 0
        self.positions_opened = 0
        self.total_pnl = Decimal("0")
        self.should_generate_signal = True
        self.signal_side = OrderSide.BUY
        
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze market data and generate signal."""
        if not self.should_generate_signal:
            return None
        
        self.signals_generated += 1
        
        return {
            "strategy": self.name,
            "symbol": "BTC/USDT",
            "side": self.signal_side,
            "quantity": Decimal("0.1"),
            "signal_strength": Decimal("0.8"),
            "timestamp": datetime.now()
        }
    
    async def on_position_opened(self, position: Dict[str, Any]):
        """Handle position opened event."""
        self.positions_opened += 1
        
    async def on_position_closed(self, position: Dict[str, Any]):
        """Handle position closed event."""
        self.total_pnl += position.get("realized_pnl", Decimal("0"))
    
    async def get_state(self) -> Dict[str, Any]:
        """Get strategy state."""
        return {
            "name": self.name,
            "signals_generated": self.signals_generated,
            "positions_opened": self.positions_opened,
            "total_pnl": self.total_pnl,
            "is_valid": True
        }
    
    async def save_state(self, state: Dict[str, Any]):
        """Save strategy state."""
        self.signals_generated = state.get("signals_generated", 0)
        self.positions_opened = state.get("positions_opened", 0)
        self.total_pnl = state.get("total_pnl", Decimal("0"))
    
    async def load_state(self) -> Dict[str, Any]:
        """Load strategy state."""
        return await self.get_state()


class AlwaysBuyStrategy(MockStrategy):
    """Strategy that always generates buy signals."""
    
    def __init__(self):
        super().__init__("always_buy")
        self.signal_side = OrderSide.BUY


class AlwaysSellStrategy(MockStrategy):
    """Strategy that always generates sell signals."""
    
    def __init__(self):
        super().__init__("always_sell")
        self.signal_side = OrderSide.SELL


class ConditionalStrategy(MockStrategy):
    """Strategy with conditional signal generation."""
    
    def __init__(self, condition_func=None):
        super().__init__("conditional")
        self.condition_func = condition_func or (lambda x: True)
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate signal based on condition."""
        if self.condition_func(market_data):
            return await super().analyze(market_data)
        return None


class SlowStrategy(MockStrategy):
    """Strategy that simulates slow processing."""
    
    def __init__(self, delay_seconds: float = 1.0):
        super().__init__("slow_strategy")
        self.delay_seconds = delay_seconds
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze with delay."""
        import asyncio
        await asyncio.sleep(self.delay_seconds)
        return await super().analyze(market_data)


class ErrorProneStrategy(MockStrategy):
    """Strategy that occasionally throws errors."""
    
    def __init__(self, error_rate: float = 0.1):
        super().__init__("error_prone")
        self.error_rate = error_rate
        self.attempt_count = 0
    
    async def analyze(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze with potential errors."""
        import random
        self.attempt_count += 1
        
        if random.random() < self.error_rate:
            raise Exception(f"Strategy error on attempt {self.attempt_count}")
        
        return await super().analyze(market_data)


class BacktestStrategy(MockStrategy):
    """Strategy for backtesting."""
    
    def __init__(self):
        super().__init__("backtest")
        self.historical_signals = []
        self.win_rate = Decimal("0")
        self.profit_factor = Decimal("0")
        self.max_drawdown = Decimal("0")
        
    async def backtest(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run backtest on historical data."""
        signals = []
        trades = []
        
        for data_point in historical_data:
            signal = await self.analyze(data_point)
            if signal:
                signals.append(signal)
                
                # Simulate trade execution
                trade = {
                    "signal": signal,
                    "entry_price": data_point.get("price"),
                    "exit_price": None,
                    "pnl": Decimal("0")
                }
                trades.append(trade)
        
        # Calculate metrics
        winning_trades = [t for t in trades if t["pnl"] > 0]
        self.win_rate = Decimal(len(winning_trades)) / Decimal(len(trades)) if trades else Decimal("0")
        
        return {
            "total_signals": len(signals),
            "total_trades": len(trades),
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown
        }


def create_strategy_suite() -> List[BaseStrategy]:
    """Create a suite of strategies for testing."""
    return [
        AlwaysBuyStrategy(),
        AlwaysSellStrategy(),
        ConditionalStrategy(lambda x: x.get("BTC/USDT", {}).get("bid", 0) > Decimal("50000")),
        SlowStrategy(0.1),
        BacktestStrategy()
    ]


def create_tiered_strategies() -> Dict[str, List[BaseStrategy]]:
    """Create strategies organized by tier."""
    return {
        "sniper": [
            MockStrategy("sniper_arb"),
            MockStrategy("sniper_spread")
        ],
        "hunter": [
            MockStrategy("hunter_mean_reversion"),
            MockStrategy("hunter_pairs"),
            MockStrategy("hunter_momentum")
        ],
        "strategist": [
            MockStrategy("strategist_statistical"),
            MockStrategy("strategist_market_making"),
            MockStrategy("strategist_options")
        ]
    }


def create_performance_test_strategies(count: int = 100) -> List[BaseStrategy]:
    """Create multiple strategies for performance testing."""
    strategies = []
    
    for i in range(count):
        strategy = MockStrategy(f"perf_test_{i}")
        strategy.should_generate_signal = i % 10 == 0  # Only 10% generate signals
        strategies.append(strategy)
    
    return strategies


class StrategyTestHarness:
    """Test harness for strategy testing."""
    
    def __init__(self):
        self.strategies = []
        self.market_data_feed = []
        self.executed_signals = []
        self.performance_metrics = {}
        
    def add_strategy(self, strategy: BaseStrategy):
        """Add strategy to test harness."""
        self.strategies.append(strategy)
        
    def add_market_data(self, data: Dict[str, Any]):
        """Add market data to feed."""
        self.market_data_feed.append(data)
        
    async def run_test(self, duration_seconds: int = 60):
        """Run strategy test."""
        import asyncio
        import time
        
        start_time = time.time()
        signals_generated = 0
        
        while time.time() - start_time < duration_seconds:
            for data in self.market_data_feed:
                for strategy in self.strategies:
                    signal = await strategy.analyze(data)
                    if signal:
                        signals_generated += 1
                        self.executed_signals.append(signal)
                
                await asyncio.sleep(0.1)
        
        self.performance_metrics = {
            "duration": duration_seconds,
            "total_signals": signals_generated,
            "signals_per_second": signals_generated / duration_seconds,
            "strategies_tested": len(self.strategies)
        }
        
        return self.performance_metrics
    
    def get_results(self) -> Dict[str, Any]:
        """Get test results."""
        return {
            "metrics": self.performance_metrics,
            "signals": self.executed_signals,
            "strategy_count": len(self.strategies)
        }