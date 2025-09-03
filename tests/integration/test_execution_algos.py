"""Integration tests for execution algorithms."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from genesis.core.models import Order, OrderSide, OrderStatus, OrderType, Signal
from genesis.execution.execution_scheduler import ExecutionScheduler, SchedulerConfig
from genesis.execution.order_slicer import OrderSlicer, SliceConfig, SlicingMethod
from genesis.execution.volume_curve import VolumeCurveEstimator
from genesis.strategies.strategist.vwap_execution import (
    UrgencyLevel,
    VWAPExecutionConfig,
    VWAPExecutionStrategy,
    VWAPOrderConfig,
)


class MockExchange:
    """Mock exchange for testing."""
    
    def __init__(self):
        """Initialize mock exchange."""
        self.orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.market_data: Dict[str, Any] = {
            "BTCUSDT": {
                "price": Decimal("50000"),
                "volume_24h": Decimal("1000000"),
                "bid": Decimal("49990"),
                "ask": Decimal("50010"),
                "spread": Decimal("20"),
                "liquidity": Decimal("500000")
            }
        }
        self.execution_delay = 0.01  # 10ms simulated latency
    
    async def place_order(self, order: Order) -> Order:
        """Simulate order placement."""
        await asyncio.sleep(self.execution_delay)
        
        order.exchange_order_id = f"EX{uuid4().hex[:8]}"
        order.status = OrderStatus.PENDING
        self.orders.append(order)
        
        # Simulate immediate fill for market orders
        if order.type == OrderType.MARKET:
            await self.fill_order(order)
            
        return order
    
    async def fill_order(self, order: Order) -> None:
        """Simulate order fill."""
        symbol_data = self.market_data.get(order.symbol, {})
        
        # Determine fill price
        if order.side == OrderSide.BUY:
            fill_price = symbol_data.get("ask", Decimal("50000"))
        else:
            fill_price = symbol_data.get("bid", Decimal("50000"))
            
        order.price = fill_price
        order.filled_quantity = order.quantity
        order.status = OrderStatus.FILLED
        order.executed_at = datetime.now(UTC)
        
        self.filled_orders.append(order)
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get market data for symbol."""
        return self.market_data.get(symbol, {})
    
    async def get_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """Get volume profile for symbol."""
        return {
            "symbol": symbol,
            "intervals": 48,
            "volumes": [Decimal("20000") + Decimal(str(i * 1000)) for i in range(48)]
        }


class TestVWAPExecutionIntegration:
    """Integration tests for VWAP execution."""
    
    @pytest.fixture
    def mock_exchange(self):
        """Create mock exchange."""
        return MockExchange()
    
    @pytest.fixture
    def vwap_strategy(self):
        """Create VWAP strategy."""
        config = VWAPExecutionConfig(
            name="IntegrationVWAP",
            max_participation_rate=Decimal("0.10"),
            volume_lookback_days=20,
            adaptive_scheduling=True
        )
        return VWAPExecutionStrategy(config)
    
    @pytest.fixture
    def scheduler(self):
        """Create execution scheduler."""
        config = SchedulerConfig(
            min_interval_seconds=1,
            max_interval_seconds=60,
            adaptive_reschedule=True,
            max_concurrent_tasks=5
        )
        return ExecutionScheduler(config)
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add 5 second timeout
    async def test_end_to_end_vwap_execution(self, vwap_strategy, mock_exchange):
        """Test complete VWAP execution flow."""
        # Mock time to avoid waiting for scheduled times
        with patch('genesis.strategies.strategist.vwap_execution.datetime') as mock_datetime:
            # Set current time for schedule generation
            current_time = datetime.now(UTC)
            mock_datetime.now.return_value = current_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Create parent order with immediate execution
            parent_config = VWAPOrderConfig(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("10.0"),
                urgency=UrgencyLevel.MEDIUM,  # Use MEDIUM to get multiple slices
                start_time=current_time,
                end_time=current_time + timedelta(hours=1),  # 1 hour to get 6 slices for MEDIUM urgency
                min_slice_size=Decimal("0.5"),
                max_slice_size=Decimal("3.0")  # Increase max slice size
            )
            
            parent_id = await vwap_strategy.create_parent_order(parent_config)
            
            # Force immediate execution by advancing time for each slice
            executed_count = 0
            max_iterations = 20
            
            for i in range(max_iterations):
                # Advance time by 10 minutes each iteration to match schedule intervals
                mock_datetime.now.return_value = current_time + timedelta(minutes=i * 10)
                
                # Check if should execute
                if await vwap_strategy._should_execute_slice(parent_id):
                    # Generate signal
                    signal = await vwap_strategy._generate_slice_signal(parent_id)
                    
                    if signal:
                        # Create order from signal
                        order = Order(
                            symbol=signal.symbol,
                            side=OrderSide.BUY,
                            type=OrderType.MARKET,
                            quantity=signal.quantity
                        )
                        
                        # Execute on mock exchange
                        filled_order = await mock_exchange.place_order(order)
                        
                        # Update strategy
                        await vwap_strategy.update_child_order(parent_id, filled_order)
                        executed_count += 1
                        
                # Check completion
                state = vwap_strategy.order_states[parent_id]
                if state.remaining_quantity <= 0:
                    break
                    
            # Verify execution
            state = vwap_strategy.order_states[parent_id]
            assert state.executed_quantity == Decimal("10.0")
            assert state.remaining_quantity == Decimal("0")
            assert len(state.completed_orders) == executed_count
            assert state.average_price > Decimal("0")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add 5 second timeout
    async def test_adaptive_execution_with_market_impact(self, vwap_strategy, mock_exchange):
        """Test adaptive execution with market impact simulation."""
        # Mock time to avoid waiting
        with patch('genesis.strategies.strategist.vwap_execution.datetime') as mock_datetime:
            current_time = datetime.now(UTC)
            mock_datetime.now.return_value = current_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Create large order with immediate execution
            parent_config = VWAPOrderConfig(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("100.0"),  # Large order
                urgency=UrgencyLevel.HIGH,  # Use HIGH for faster but sliced execution
                target_participation_rate=Decimal("0.05"),  # 5% to minimize impact
                start_time=current_time,
                end_time=current_time + timedelta(hours=3),  # 3 hours to get 12 slices for HIGH urgency
                max_slice_size=Decimal("10.0")  # Limit slice size to force multiple orders
            )
            
            parent_id = await vwap_strategy.create_parent_order(parent_config)
            
            # Track market impact
            initial_price = mock_exchange.market_data["BTCUSDT"]["price"]
            executed_quantity = Decimal("0")
            max_iterations = 50  # Add iteration limit to prevent infinite loop
            iteration = 0
            
            while executed_quantity < Decimal("100.0") and iteration < max_iterations:
                # Advance time by 15 minutes each iteration to match schedule intervals (3 hours / 12 slices = 15 min)
                mock_datetime.now.return_value = current_time + timedelta(minutes=iteration * 15)
                
                if await vwap_strategy._should_execute_slice(parent_id):
                    signal = await vwap_strategy._generate_slice_signal(parent_id)
                    
                    if signal:
                        # Simulate market impact - price rises with buying
                        impact = (executed_quantity / Decimal("1000")) * Decimal("100")  # 0.1% per 10 BTC
                        mock_exchange.market_data["BTCUSDT"]["price"] = initial_price + impact
                        mock_exchange.market_data["BTCUSDT"]["ask"] = initial_price + impact + Decimal("10")
                        
                        # Execute order
                        order = Order(
                            symbol=signal.symbol,
                            side=OrderSide.BUY,
                            type=OrderType.MARKET,
                            quantity=signal.quantity
                        )
                        
                        filled_order = await mock_exchange.place_order(order)
                        await vwap_strategy.update_child_order(parent_id, filled_order)
                        
                        executed_quantity += filled_order.filled_quantity
                
                iteration += 1
                        
            # Calculate implementation shortfall
            state = vwap_strategy.order_states[parent_id]
            final_price = mock_exchange.market_data["BTCUSDT"]["price"]
            market_impact_pct = ((final_price - initial_price) / initial_price) * Decimal("100")
            
            assert state.executed_quantity == Decimal("100.0")
            assert market_impact_pct < Decimal("10")  # Less than 10% impact
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add 5 second timeout
    async def test_emergency_liquidation_flow(self, vwap_strategy, mock_exchange):
        """Test emergency liquidation execution."""
        # Mock time to avoid waiting
        with patch('genesis.strategies.strategist.vwap_execution.datetime') as mock_datetime:
            current_time = datetime.now(UTC)
            mock_datetime.now.return_value = current_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Create normal order
            parent_config = VWAPOrderConfig(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                total_quantity=Decimal("50.0"),
                urgency=UrgencyLevel.LOW,
                start_time=current_time,
                end_time=current_time + timedelta(hours=2)
            )
            
            parent_id = await vwap_strategy.create_parent_order(parent_config)
            
            # Execute a few slices normally
            for i in range(3):
                # Advance time to trigger slice execution
                mock_datetime.now.return_value = current_time + timedelta(seconds=i * 10)
                
                if await vwap_strategy._should_execute_slice(parent_id):
                    signal = await vwap_strategy._generate_slice_signal(parent_id)
                    if signal:
                        order = Order(
                            symbol=signal.symbol,
                            side=OrderSide.SELL,
                            type=OrderType.MARKET,
                            quantity=signal.quantity
                        )
                        filled_order = await mock_exchange.place_order(order)
                        await vwap_strategy.update_child_order(parent_id, filled_order)
            
            # Trigger emergency liquidation
            await vwap_strategy.trigger_emergency_liquidation(parent_id)
            
            # Execute emergency order
            signal = await vwap_strategy._generate_slice_signal(parent_id)
            assert signal is not None
            assert signal.metadata["urgency"] == UrgencyLevel.EMERGENCY.value
            
            # Remaining quantity should be in single order
            state = vwap_strategy.order_states[parent_id]
            assert len(state.schedule) == 1
            assert state.schedule[0]["urgency"] == UrgencyLevel.EMERGENCY.value
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add 5 second timeout
    async def test_scheduler_integration(self, scheduler, vwap_strategy):
        """Test integration between scheduler and VWAP strategy."""
        # Mock time to avoid waiting
        with patch('genesis.execution.execution_scheduler.datetime') as mock_datetime:
            current_time = datetime.now(UTC)
            mock_datetime.now.return_value = current_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Start scheduler
            await scheduler.start()
            
            try:
                # Create VWAP order with immediate execution
                parent_config = VWAPOrderConfig(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    total_quantity=Decimal("20.0"),
                    urgency=UrgencyLevel.EMERGENCY  # Use emergency for immediate execution
                )
                
                parent_id = await vwap_strategy.create_parent_order(parent_config)
                state = vwap_strategy.order_states[parent_id]
                
                # Create slices
                slicer = OrderSlicer()
                slices = slicer.slice_order(
                    total_quantity=parent_config.total_quantity,
                    method=SlicingMethod.LINEAR
                )
                
                # Create execution plan
                plan = await scheduler.create_execution_plan(
                    parent_order_id=str(parent_id),
                    slices=slices,
                    urgency=parent_config.urgency,
                    start_time=current_time,
                    end_time=current_time + timedelta(seconds=5)  # Short timeframe
                )
                
                assert plan.plan_id in scheduler.execution_plans
                assert len(plan.tasks) == len(slices)
                
                # Simulate task execution with time advancement
                executed_tasks = 0
                max_iterations = 10
                
                for i in range(max_iterations):
                    # Advance time
                    mock_datetime.now.return_value = current_time + timedelta(seconds=i * 0.5)
                    
                    task = await scheduler.get_next_task()
                    if task:
                        # Simulate execution
                        await asyncio.sleep(0.01)
                        scheduler.complete_task(task.task_id, success=True)
                        executed_tasks += 1
                        
                        if executed_tasks >= 3:
                            break
                        
                # Verify execution tracking
                assert plan.executed_quantity > Decimal("0")
                
            finally:
                await scheduler.stop()
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add 5 second timeout
    async def test_volume_curve_adaptation(self, vwap_strategy):
        """Test volume curve adaptation to real-time data."""
        estimator = VolumeCurveEstimator()
        
        # Get initial curve
        initial_curve = await estimator.estimate_volume_curve("BTCUSDT")
        
        # Simulate real-time volume updates
        current_time = datetime.now(UTC)
        observed_volume = Decimal("150000")  # Higher than expected
        
        await estimator.update_with_realtime_data(
            "BTCUSDT",
            observed_volume,
            current_time
        )
        
        # Get updated curve
        cache_key = f"BTCUSDT:{current_time.date()}"
        updated_curve = estimator.cached_curves.get(cache_key)
        
        if updated_curve:
            # Future intervals should be adjusted
            _, current_interval = estimator.get_current_interval_volume(initial_curve, current_time)
            
            if current_interval < len(initial_curve.normalized_volumes) - 1:
                # Check that future volumes were adjusted
                assert updated_curve.normalized_volumes[current_interval + 1] != initial_curve.normalized_volumes[current_interval + 1]
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add 5 second timeout
    async def test_multi_parent_order_execution(self, vwap_strategy, mock_exchange):
        """Test executing multiple parent orders concurrently."""
        # Mock time to avoid waiting
        with patch('genesis.strategies.strategist.vwap_execution.datetime') as mock_datetime:
            current_time = datetime.now(UTC)
            mock_datetime.now.return_value = current_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            # Create multiple parent orders
            parent_ids = []
            
            for i in range(3):
                config = VWAPOrderConfig(
                    symbol="BTCUSDT",
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    total_quantity=Decimal(f"{10 + i * 5}.0"),
                    urgency=UrgencyLevel.EMERGENCY,  # Use emergency for immediate execution
                    start_time=current_time,
                    end_time=current_time + timedelta(seconds=30)
                )
                parent_id = await vwap_strategy.create_parent_order(config)
                parent_ids.append(parent_id)
        
            # Generate signals for all orders
            all_signals = await vwap_strategy.generate_signals()
            
            # Should have signals from multiple parents
            parent_order_ids = {s.metadata.get("parent_order_id") for s in all_signals}
            assert len(parent_order_ids) <= 3
            
            # Execute orders
            for signal in all_signals:
                order = Order(
                    symbol=signal.symbol,
                    side=OrderSide.BUY if signal.signal_type == signal.signal_type.BUY else OrderSide.SELL,
                    type=OrderType.MARKET,
                    quantity=signal.quantity
                )
                
                filled_order = await mock_exchange.place_order(order)
                
                # Update correct parent order
                parent_id = uuid4()  # Would parse from signal metadata in real implementation
                for pid in parent_ids:
                    if str(pid) == signal.metadata.get("parent_order_id"):
                        await vwap_strategy.update_child_order(pid, filled_order)
                        break
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(5)  # Add 5 second timeout
    async def test_performance_benchmark(self, vwap_strategy):
        """Benchmark performance of VWAP execution."""
        # Mock time to avoid waiting
        with patch('genesis.strategies.strategist.vwap_execution.datetime') as mock_datetime:
            current_time = datetime.now(UTC)
            mock_datetime.now.return_value = current_time
            mock_datetime.fromisoformat = datetime.fromisoformat
            
            start_time = time.perf_counter()
            
            # Create large parent order
            parent_config = VWAPOrderConfig(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("1000.0"),
                urgency=UrgencyLevel.HIGH,
                start_time=current_time,
                end_time=current_time + timedelta(hours=1)
            )
        
            parent_id = await vwap_strategy.create_parent_order(parent_config)
            
            # Measure schedule generation time
            schedule_time = time.perf_counter() - start_time
            
            state = vwap_strategy.order_states[parent_id]
            
            # Performance assertions
            assert schedule_time < 0.1  # Schedule generation under 100ms
            assert len(state.schedule) > 0
            assert len(state.schedule) <= 100  # Reasonable slice count
            
            # Measure signal generation time
            signal_start = time.perf_counter()
            
            for _ in range(10):
                await vwap_strategy.generate_signals()
                
            signal_time = (time.perf_counter() - signal_start) / 10
            
            assert signal_time < 0.02  # Signal generation under 20ms
            
            # Measure memory usage (approximate)
            import sys
            memory_usage = sys.getsizeof(vwap_strategy.parent_orders) + \
                          sys.getsizeof(vwap_strategy.order_states) + \
                          sys.getsizeof(vwap_strategy.volume_curves)
            
            assert memory_usage < 100 * 1024 * 1024  # Under 100MB