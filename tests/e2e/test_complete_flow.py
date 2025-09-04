"""Complete flow tests for end-to-end trading scenarios."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog
from datetime import datetime, timedelta

from genesis.core.models import Position, Order, OrderType, OrderSide
from genesis.engine.engine import TradingEngine
from genesis.monitoring.strategy_monitor import StrategyMonitor
from genesis.monitoring.risk_metrics import RiskMetrics

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
class TestCompleteScenarios:
    """Test complete trading scenarios from start to finish."""

    async def test_full_trading_day_simulation(self, trading_system):
        """Simulate a complete trading day with multiple strategies."""
        start_balance = await trading_system.exchange_gateway.get_balance()
        initial_usdt = start_balance["USDT"]
        
        await trading_system.engine.start()
        
        trades_executed = []
        for hour in range(24):
            market_data = {
                "BTC/USDT": {
                    "bid": Decimal("50000") + Decimal(hour * 100),
                    "ask": Decimal("50001") + Decimal(hour * 100),
                    "volume": Decimal("1000")
                }
            }
            
            await trading_system.engine.process_market_data(market_data)
            
            if hour % 4 == 0:  # Execute trades every 4 hours
                signal = {
                    "strategy": "simple_arb",
                    "symbol": "BTC/USDT",
                    "side": OrderSide.BUY if hour % 8 == 0 else OrderSide.SELL,
                    "quantity": Decimal("0.01"),
                    "signal_strength": Decimal("0.7")
                }
                
                trade = await trading_system.engine.process_signal(signal)
                if trade:
                    trades_executed.append(trade)
        
        await trading_system.engine.stop()
        
        final_balance = await trading_system.exchange_gateway.get_balance()
        final_usdt = final_balance["USDT"]
        
        assert len(trades_executed) > 0
        assert final_usdt != initial_usdt  # Balance should have changed

    async def test_market_crash_scenario(self, trading_system):
        """Test system behavior during market crash."""
        await trading_system.engine.start()
        
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("50000.00")
        )
        
        await trading_system.engine.open_position(position)
        
        crash_prices = [Decimal("48000"), Decimal("45000"), Decimal("42000")]
        
        for price in crash_prices:
            market_data = {
                "BTC/USDT": {
                    "bid": price - Decimal("10"),
                    "ask": price,
                    "last": price
                }
            }
            
            await trading_system.engine.process_market_data(market_data)
            
            position.current_price = price
            should_exit = await trading_system.risk_engine.check_stop_loss(position)
            
            if should_exit:
                await trading_system.engine.emergency_close_position(position)
                break
        
        assert not position.is_open
        assert position.realized_pnl < 0  # Should have loss but limited by stop-loss

    async def test_high_volatility_handling(self, trading_system):
        """Test system during high volatility periods."""
        await trading_system.engine.start()
        
        volatility_events = []
        base_price = Decimal("50000")
        
        for i in range(100):
            variation = Decimal((-1) ** i * 500 * (i % 10))
            price = base_price + variation
            
            market_data = {
                "BTC/USDT": {
                    "bid": price - Decimal("10"),
                    "ask": price + Decimal("10"),
                    "last": price,
                    "volume": Decimal("10000")
                }
            }
            
            await trading_system.engine.process_market_data(market_data)
            
            if abs(variation) > Decimal("2000"):
                volatility_events.append({
                    "timestamp": datetime.now(),
                    "price": price,
                    "variation": variation
                })
        
        circuit_breaker_triggered = trading_system.engine.is_circuit_breaker_active()
        
        if len(volatility_events) > 20:
            assert circuit_breaker_triggered

    async def test_strategy_rotation(self, trading_system):
        """Test switching between strategies based on market conditions."""
        await trading_system.engine.start()
        
        market_conditions = [
            {"trend": "bullish", "volatility": "low"},
            {"trend": "bearish", "volatility": "high"},
            {"trend": "sideways", "volatility": "medium"}
        ]
        
        active_strategies = []
        
        for condition in market_conditions:
            selected_strategy = await trading_system.engine.select_strategy(condition)
            
            if selected_strategy:
                await trading_system.engine.activate_strategy(selected_strategy)
                active_strategies.append(selected_strategy)
                
                await asyncio.sleep(1)  # Run strategy
                
                await trading_system.engine.deactivate_strategy(selected_strategy)
        
        assert len(active_strategies) == len(market_conditions)
        assert all(s is not None for s in active_strategies)


@pytest.mark.asyncio
class TestDataIntegrity:
    """Test data consistency and integrity across the system."""

    async def test_order_audit_trail(self, trading_system):
        """Test complete audit trail for orders."""
        order_id = "audit_test_123"
        
        order = Order(
            order_id=order_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("49900.00")
        )
        
        await trading_system.engine.submit_order(order)
        
        await asyncio.sleep(0.1)  # Let order process
        
        audit_trail = await trading_system.engine.get_order_audit_trail(order_id)
        
        assert len(audit_trail) > 0
        assert audit_trail[0]["event"] == "order_submitted"
        assert audit_trail[-1]["event"] in ["order_filled", "order_cancelled", "order_rejected"]

    async def test_position_reconciliation(self, trading_system):
        """Test position reconciliation across components."""
        positions_engine = await trading_system.engine.get_positions()
        positions_risk = await trading_system.risk_engine.get_positions()
        positions_db = await trading_system.engine.load_positions_from_db()
        
        assert positions_engine == positions_risk
        assert positions_engine == positions_db

    async def test_balance_consistency(self, trading_system):
        """Test balance consistency after trades."""
        initial_balance = await trading_system.exchange_gateway.get_balance()
        initial_usdt = initial_balance["USDT"]
        initial_btc = initial_balance.get("BTC", Decimal("0"))
        
        buy_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01")
        )
        
        await trading_system.engine.execute_order(buy_order)
        
        final_balance = await trading_system.exchange_gateway.get_balance()
        final_usdt = final_balance["USDT"]
        final_btc = final_balance.get("BTC", Decimal("0"))
        
        expected_cost = Decimal("0.01") * Decimal("50000")  # Approximate
        
        assert final_usdt < initial_usdt
        assert final_btc > initial_btc
        assert abs((initial_usdt - final_usdt) - expected_cost) < Decimal("100")  # Allow for fees

    async def test_transaction_atomicity(self, trading_system):
        """Test atomicity of multi-step transactions."""
        from genesis.data.repository import TradeRepository
        
        repo = TradeRepository()
        
        async with repo.transaction() as tx:
            trade1 = {"order_id": "tx_1", "amount": Decimal("100")}
            trade2 = {"order_id": "tx_2", "amount": Decimal("200")}
            
            await tx.save_trade(trade1)
            await tx.save_trade(trade2)
            
            saved_trades = await tx.get_trades_by_ids(["tx_1", "tx_2"])
            assert len(saved_trades) == 2
        
        final_trades = await repo.get_trades_by_ids(["tx_1", "tx_2"])
        assert len(final_trades) == 2


@pytest.mark.asyncio
class TestPerformanceMetrics:
    """Test system performance metrics collection and analysis."""

    async def test_latency_measurement(self, trading_system):
        """Test end-to-end latency measurement."""
        latencies = []
        
        for _ in range(100):
            start_time = asyncio.get_event_loop().time()
            
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001")
            )
            
            await trading_system.engine.execute_order(order)
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 50  # Average should be under 50ms
        assert max_latency < 100  # Max should be under 100ms

    async def test_throughput_measurement(self, trading_system):
        """Test order throughput capacity."""
        orders_processed = 0
        start_time = asyncio.get_event_loop().time()
        
        async def submit_order():
            nonlocal orders_processed
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001")
            )
            await trading_system.engine.execute_order(order)
            orders_processed += 1
        
        tasks = [submit_order() for _ in range(1000)]
        await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time
        throughput = orders_processed / duration
        
        assert throughput >= 1000  # Should handle at least 1000 orders/second

    async def test_memory_usage(self, trading_system):
        """Test memory usage under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        for _ in range(10000):
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.001")
            )
            await trading_system.engine.execute_order(order)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 500  # Should not increase by more than 500MB

    async def test_cpu_utilization(self, trading_system):
        """Test CPU utilization with multiple strategies."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        for strategy in trading_system.strategies:
            await trading_system.engine.register_strategy(strategy)
        
        cpu_samples = []
        
        for _ in range(60):  # Sample for 60 seconds
            cpu_percent = process.cpu_percent(interval=1)
            cpu_samples.append(cpu_percent)
        
        avg_cpu = sum(cpu_samples) / len(cpu_samples)
        max_cpu = max(cpu_samples)
        
        assert avg_cpu < 50  # Average CPU should be under 50%
        assert max_cpu < 80  # Max CPU should be under 80%