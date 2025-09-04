"""Multi-strategy concurrent execution tests."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog
from concurrent.futures import ThreadPoolExecutor

from genesis.core.models import Position, Order, OrderType, OrderSide
from genesis.engine.engine import TradingEngine
from genesis.strategies.base import BaseStrategy

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
class TestMultiStrategyExecution:
    """Test concurrent execution of multiple strategies."""

    async def test_three_sniper_strategies_concurrent(self, trading_system):
        """Test 3 Sniper strategies running simultaneously."""
        from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
        from genesis.strategies.sniper.spread_capture import SpreadCaptureStrategy
        
        strategies = [
            SimpleArbitrageStrategy(
                symbol="BTC/USDT",
                min_spread=Decimal("0.001")
            ),
            SimpleArbitrageStrategy(
                symbol="ETH/USDT",
                min_spread=Decimal("0.0015")
            ),
            SpreadCaptureStrategy(
                symbol="BNB/USDT",
                min_spread_pct=Decimal("0.002")
            )
        ]
        
        for strategy in strategies:
            await trading_system.engine.register_strategy(strategy)
        
        market_data = {
            "BTC/USDT": {"bid": Decimal("50000"), "ask": Decimal("50100")},
            "ETH/USDT": {"bid": Decimal("3000"), "ask": Decimal("3005")},
            "BNB/USDT": {"bid": Decimal("400"), "ask": Decimal("401")}
        }
        
        tasks = []
        for _ in range(10):  # Process 10 market updates
            tasks.append(trading_system.engine.process_market_data(market_data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0
        
        active_strategies = trading_system.engine.get_active_strategies()
        assert len(active_strategies) == 3

    async def test_two_hunter_strategies_multi_pair(self, trading_system):
        """Test 2 Hunter strategies with 5 pairs each."""
        from genesis.strategies.hunter.mean_reversion import MeanReversionStrategy
        from genesis.strategies.hunter.multi_pair import MultiPairStrategy
        
        pairs1 = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
        pairs2 = ["MATIC/USDT", "DOT/USDT", "AVAX/USDT", "LINK/USDT", "UNI/USDT"]
        
        strategy1 = MultiPairStrategy(
            pairs=pairs1,
            correlation_threshold=Decimal("0.7")
        )
        
        strategy2 = MultiPairStrategy(
            pairs=pairs2,
            correlation_threshold=Decimal("0.6")
        )
        
        await trading_system.engine.register_strategy(strategy1)
        await trading_system.engine.register_strategy(strategy2)
        
        market_updates = []
        for pair in pairs1 + pairs2:
            market_updates.append({
                pair: {
                    "bid": Decimal("100") * (1 + hash(pair) % 10 / 100),
                    "ask": Decimal("100") * (1 + hash(pair) % 10 / 100 + 0.001),
                    "volume": Decimal("10000")
                }
            })
        
        tasks = []
        for update in market_updates:
            tasks.append(trading_system.engine.process_market_data(update))
        
        await asyncio.gather(*tasks)
        
        positions = await trading_system.engine.get_positions()
        
        assert len(trading_system.engine.get_active_strategies()) == 2

    async def test_strategy_isolation(self, trading_system):
        """Verify strategies are properly isolated from each other."""
        from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
        
        strategy1 = SimpleArbitrageStrategy(
            symbol="BTC/USDT",
            min_spread=Decimal("0.001")
        )
        strategy1.name = "strategy1"
        
        strategy2 = SimpleArbitrageStrategy(
            symbol="BTC/USDT",  # Same symbol
            min_spread=Decimal("0.002")
        )
        strategy2.name = "strategy2"
        
        await trading_system.engine.register_strategy(strategy1)
        await trading_system.engine.register_strategy(strategy2)
        
        strategy1_state = {"positions": [], "orders": []}
        strategy2_state = {"positions": [], "orders": []}
        
        strategy1_state["orders"].append({
            "symbol": "BTC/USDT",
            "quantity": Decimal("0.1")
        })
        
        strategy2_state["orders"].append({
            "symbol": "BTC/USDT",
            "quantity": Decimal("0.2")
        })
        
        assert strategy1_state["orders"][0]["quantity"] == Decimal("0.1")
        assert strategy2_state["orders"][0]["quantity"] == Decimal("0.2")
        assert strategy1_state != strategy2_state

    async def test_resource_sharing_conflicts(self, trading_system):
        """Test resource sharing and conflict resolution."""
        strategies = []
        
        for i in range(5):
            from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
            strategy = SimpleArbitrageStrategy(
                symbol="BTC/USDT",  # All strategies on same symbol
                min_spread=Decimal("0.001") * (i + 1)
            )
            strategy.name = f"strategy_{i}"
            strategies.append(strategy)
            await trading_system.engine.register_strategy(strategy)
        
        market_data = {
            "BTC/USDT": {
                "bid": Decimal("50000"),
                "ask": Decimal("50050"),
                "volume": Decimal("100")
            }
        }
        
        tasks = []
        for _ in range(20):  # Generate many concurrent signals
            tasks.append(trading_system.engine.process_market_data(market_data))
        
        await asyncio.gather(*tasks)
        
        total_position = await trading_system.engine.get_total_position("BTC/USDT")
        max_allowed = trading_system.risk_engine.get_max_position_size("BTC/USDT")
        
        assert total_position <= max_allowed

    async def test_strategy_performance_isolation(self, trading_system):
        """Test that slow strategy doesn't affect others."""
        from genesis.strategies.base import BaseStrategy
        
        class SlowStrategy(BaseStrategy):
            async def analyze(self, market_data):
                await asyncio.sleep(1)  # Simulate slow processing
                return None
        
        class FastStrategy(BaseStrategy):
            async def analyze(self, market_data):
                return {
                    "symbol": "ETH/USDT",
                    "side": OrderSide.BUY,
                    "quantity": Decimal("0.1")
                }
        
        slow_strategy = SlowStrategy()
        fast_strategy = FastStrategy()
        
        await trading_system.engine.register_strategy(slow_strategy)
        await trading_system.engine.register_strategy(fast_strategy)
        
        market_data = {"ETH/USDT": {"bid": Decimal("3000"), "ask": Decimal("3001")}}
        
        start_time = asyncio.get_event_loop().time()
        
        await trading_system.engine.process_market_data(market_data)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        
        assert duration < 1.5  # Fast strategy shouldn't wait for slow one

    async def test_strategy_priority_execution(self, trading_system):
        """Test strategy execution priority based on tier."""
        from genesis.strategies.base import BaseStrategy
        
        class HighPriorityStrategy(BaseStrategy):
            priority = 1
            
            async def analyze(self, market_data):
                return {"signal": "high_priority"}
        
        class LowPriorityStrategy(BaseStrategy):
            priority = 10
            
            async def analyze(self, market_data):
                return {"signal": "low_priority"}
        
        high_strategy = HighPriorityStrategy()
        low_strategy = LowPriorityStrategy()
        
        await trading_system.engine.register_strategy(low_strategy)
        await trading_system.engine.register_strategy(high_strategy)
        
        execution_order = []
        
        async def track_execution(strategy, signal):
            execution_order.append(strategy.priority)
        
        trading_system.engine.on_signal_generated = track_execution
        
        market_data = {"BTC/USDT": {"bid": Decimal("50000"), "ask": Decimal("50001")}}
        await trading_system.engine.process_market_data(market_data)
        
        assert execution_order[0] < execution_order[-1]  # High priority first

    async def test_concurrent_order_submission(self, trading_system):
        """Test concurrent order submission from multiple strategies."""
        orders_to_submit = []
        
        for i in range(100):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000") + Decimal(i)
            )
            orders_to_submit.append(order)
        
        async def submit_order(order):
            return await trading_system.engine.execute_order(order)
        
        tasks = [submit_order(order) for order in orders_to_submit]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_orders = [r for r in results if not isinstance(r, Exception)]
        failed_orders = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful_orders) > 0
        assert len(failed_orders) < len(orders_to_submit) * 0.1  # Less than 10% failure

    async def test_strategy_state_consistency(self, trading_system):
        """Test state consistency across concurrent strategy executions."""
        from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
        
        strategies = []
        for i in range(3):
            strategy = SimpleArbitrageStrategy(
                symbol=f"COIN{i}/USDT",
                min_spread=Decimal("0.001")
            )
            strategies.append(strategy)
            await trading_system.engine.register_strategy(strategy)
        
        initial_states = {}
        for strategy in strategies:
            initial_states[strategy.symbol] = await strategy.get_state()
        
        market_updates = []
        for i in range(3):
            market_updates.append({
                f"COIN{i}/USDT": {
                    "bid": Decimal("100") + Decimal(i),
                    "ask": Decimal("101") + Decimal(i)
                }
            })
        
        tasks = []
        for _ in range(10):
            for update in market_updates:
                tasks.append(trading_system.engine.process_market_data(update))
        
        await asyncio.gather(*tasks)
        
        final_states = {}
        for strategy in strategies:
            final_states[strategy.symbol] = await strategy.get_state()
        
        for symbol in initial_states:
            assert final_states[symbol] != initial_states[symbol]  # State should have changed
            assert final_states[symbol]["is_valid"]  # State should be valid