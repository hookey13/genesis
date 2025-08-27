"""Integration tests for complete VWAP execution workflow."""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pandas as pd

from genesis.engine.executor.vwap import VWAPExecutor, ExecutionMode
from genesis.engine.executor.base import (
    Order, OrderSide, OrderType, OrderStatus, ExecutionResult
)
from genesis.core.constants import TradingTier
from genesis.core.models import Symbol
from genesis.core.events import Event, EventType, EventPriority
from genesis.exchange.gateway import BinanceGateway as ExchangeGateway
from genesis.analytics.volume_analyzer import VolumeAnalyzer, VolumeProfile, VolumePrediction
from genesis.analytics.vwap_tracker import VWAPTracker, Trade, VWAPMetrics
from genesis.engine.event_bus import EventBus


@pytest.fixture
def mock_exchange_gateway():
    """Create a mock exchange gateway with realistic behavior."""
    gateway = Mock(spec=ExchangeGateway)
    
    # Mock successful order placement
    async def mock_place_order(order):
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.executed_at = datetime.now(timezone.utc)
        return {
            'order_id': order.order_id,
            'exchange_order_id': f'binance_{order.order_id}',
            'status': 'FILLED',
            'filled_quantity': str(order.quantity),
            'average_price': '100.00'
        }
    
    gateway.place_order = AsyncMock(side_effect=mock_place_order)
    gateway.cancel_order = AsyncMock(return_value=True)
    gateway.get_order_status = AsyncMock()
    gateway.get_klines = AsyncMock()
    
    return gateway


@pytest.fixture
def integration_config():
    """Create integration test configuration."""
    return {
        'vwap_participation_rate_percent': 10.0,
        'vwap_min_slice_size_usd': 50.00,
        'vwap_max_slices': 10,  # Smaller for testing
        'vwap_time_window_minutes': 60,  # Shorter for testing
        'vwap_aggressive_threshold_percent': 5.0
    }


@pytest.fixture
async def vwap_system(mock_exchange_gateway, integration_config):
    """Create complete VWAP system with all components."""
    # Create real components
    event_bus = EventBus()
    volume_analyzer = VolumeAnalyzer(mock_exchange_gateway)
    vwap_tracker = VWAPTracker(event_bus, window_minutes=30)
    
    # Create VWAP executor
    executor = VWAPExecutor(
        tier=TradingTier.STRATEGIST,
        exchange_gateway=mock_exchange_gateway,
        volume_analyzer=volume_analyzer,
        vwap_tracker=vwap_tracker,
        event_bus=event_bus,
        config=integration_config
    )
    
    # Start tracker
    await vwap_tracker.start()
    
    yield {
        'executor': executor,
        'tracker': vwap_tracker,
        'analyzer': volume_analyzer,
        'event_bus': event_bus,
        'gateway': mock_exchange_gateway
    }
    
    # Cleanup
    await vwap_tracker.stop()


class TestVWAPWorkflowIntegration:
    """Test complete VWAP execution workflow."""
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_end_to_end_vwap_execution(self, vwap_system):
        """Test complete VWAP execution from order to completion."""
        executor = vwap_system['executor']
        tracker = vwap_system['tracker']
        analyzer = vwap_system['analyzer']
        gateway = vwap_system['gateway']
        
        # Setup mock historical data
        mock_klines = []
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        for i in range(336):  # 7 days * 48 half-hour buckets
            mock_klines.append([
                int((base_time + timedelta(minutes=i*30)).timestamp() * 1000),
                100, 101, 99, 100.5,
                1000 + (i % 48) * 100,  # Volume pattern
                0, 0, 0, 0, 0, 0
            ])
        gateway.get_klines.return_value = mock_klines
        
        # Create order
        order = Order(
            order_id='integration_test_order',
            position_id='pos_123',
            client_order_id='client_123',
            symbol='BTC/USDT',
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal('500'),  # Smaller for testing
            filled_quantity=Decimal('0'),
            status=OrderStatus.PENDING
        )
        
        # Add some market trades to tracker
        symbol = Symbol('BTC/USDT')
        for i in range(10):
            trade = Trade(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                price=Decimal('100') + Decimal(str(i * 0.1)),
                volume=Decimal('50')
            )
            tracker.add_trade(symbol, trade)
        
        # Execute VWAP order with shorter time horizon
        with patch.object(executor, '_execute_slices') as mock_execute:
            # Mock successful slice execution
            mock_execute.return_value = ExecutionResult(
                success=True,
                order=order,
                message="VWAP execution completed",
                actual_price=Decimal('100.25'),
                slippage_percent=Decimal('0.25')
            )
            
            result = await executor.execute_vwap(
                order,
                mode=ExecutionMode.NORMAL,
                time_horizon_minutes=30,  # Short for testing
                participation_rate=Decimal('0.1')
            )
            
            # Verify execution succeeded
            assert result.success is True
            assert "VWAP execution completed" in result.message
            
            # Verify slices were calculated
            assert order.order_id in executor._active_executions
            slices = executor._active_executions[order.order_id]
            assert len(slices) > 0
            assert len(slices) <= executor.max_slices
            
            # Verify total slice quantity matches order
            total_quantity = sum(s.quantity for s in slices)
            assert abs(total_quantity - order.quantity) < Decimal('1')
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_vwap_with_volume_spike_detection(self, vwap_system):
        """Test VWAP execution with volume spike detection and adaptation."""
        executor = vwap_system['executor']
        analyzer = vwap_system['analyzer']
        gateway = vwap_system['gateway']
        
        # Setup normal volume profile
        normal_volume = 1000
        spike_volume = 5000  # 5x normal
        
        mock_klines = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=2)
        for i in range(4):  # 2 hours of 30-min data
            volume = spike_volume if i == 2 else normal_volume  # Spike in 3rd bucket
            mock_klines.append([
                int((base_time + timedelta(minutes=i*30)).timestamp() * 1000),
                100, 101, 99, 100.5,
                volume,
                0, 0, 0, 0, 0, 0
            ])
        gateway.get_klines.return_value = mock_klines
        
        # Create order
        order = Order(
            order_id='spike_test_order',
            position_id='pos_456',
            client_order_id='client_456',
            symbol='BTC/USDT',
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            price=None,
            quantity=Decimal('200'),
            filled_quantity=Decimal('0'),
            status=OrderStatus.PENDING
        )
        
        # Get volume prediction
        symbol = Symbol('BTC/USDT')
        prediction = await analyzer.predict_intraday_volume(
            symbol,
            datetime.now(timezone.utc),
            horizon_hours=1
        )
        
        # Calculate slices
        slices = await executor._calculate_slices(
            order,
            prediction,
            Decimal('0.1'),
            ExecutionMode.NORMAL,
            60
        )
        
        # Verify slices adapt to volume pattern
        assert len(slices) > 0
        
        # Check for volume spike
        is_spike, deviation = await analyzer.analyze_volume_spike(
            symbol,
            Decimal(str(spike_volume)),
            30
        )
        
        # Should detect spike
        assert is_spike is True
        assert deviation > Decimal('2')
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_vwap_execution_mode_switching(self, vwap_system):
        """Test dynamic switching between execution modes."""
        executor = vwap_system['executor']
        
        # Create order
        order = Order(
            order_id='mode_test_order',
            position_id='pos_789',
            client_order_id='client_789',
            symbol='BTC/USDT',
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal('1000'),
            filled_quantity=Decimal('200'),  # 20% filled
            status=OrderStatus.PARTIAL
        )
        
        # Create slices with varied completion
        from genesis.engine.executor.vwap import VWAPSlice
        slices = []
        for i in range(5):
            slice_obj = VWAPSlice(
                slice_id=f'slice_{i}',
                parent_order_id=order.order_id,
                symbol=Symbol('BTC/USDT'),
                side=OrderSide.BUY,
                quantity=Decimal('200'),
                target_price=None,
                scheduled_time=datetime.now(timezone.utc) + timedelta(minutes=i*10),
                bucket_minute=i * 30
            )
            # First two slices are filled
            if i < 2:
                slice_obj.status = OrderStatus.FILLED
                slice_obj.executed_quantity = Decimal('100')
            else:
                slice_obj.status = OrderStatus.PENDING
            slices.append(slice_obj)
        
        # Test switching logic
        # 20% filled (200/1000) but 40% time passed (2/5 slices) 
        should_switch = await executor._should_switch_to_aggressive(order, slices)
        assert should_switch is True
        
        # Update to better progress
        order.filled_quantity = Decimal('450')  # 45% filled
        should_switch = await executor._should_switch_to_aggressive(order, slices)
        assert should_switch is False
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_vwap_performance_tracking(self, vwap_system):
        """Test VWAP execution performance tracking and reporting."""
        executor = vwap_system['executor']
        tracker = vwap_system['tracker']
        
        # Start execution tracking
        symbol = Symbol('BTC/USDT')
        execution_id = 'perf_test_exec'
        target_volume = Decimal('1000')
        
        # Add market trades for VWAP calculation
        for i in range(20):
            trade = Trade(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                price=Decimal('100') + Decimal(str(i * 0.05)),
                volume=Decimal('100')
            )
            tracker.add_trade(symbol, trade)
        
        # Start tracking
        performance = tracker.start_execution_tracking(
            symbol, execution_id, target_volume
        )
        
        # Simulate execution fills
        fills = [
            (Decimal('100.10'), Decimal('200')),
            (Decimal('100.20'), Decimal('300')),
            (Decimal('100.15'), Decimal('250')),
            (Decimal('100.25'), Decimal('250'))
        ]
        
        for price, volume in fills:
            tracker.update_execution(execution_id, price, volume)
        
        # Complete execution
        final_performance = tracker.complete_execution(execution_id, target_volume)
        
        assert final_performance is not None
        assert final_performance.executed_volume == Decimal('1000')
        assert final_performance.fill_rate == Decimal('100')
        assert final_performance.trades_executed == 4
        
        # Calculate expected execution VWAP
        total_value = sum(p * v for p, v in fills)
        expected_vwap = total_value / Decimal('1000')
        assert abs(final_performance.execution_vwap - expected_vwap) < Decimal('0.01')
        
        # Get performance statistics
        stats = tracker.get_performance_stats(symbol=symbol, hours=1)
        assert stats['executions'] == 1
        assert stats['total_volume'] == '1000'
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_vwap_with_iceberg_orders(self, vwap_system):
        """Test VWAP execution using iceberg orders for dark pool simulation."""
        executor = vwap_system['executor']
        
        order = Order(
            order_id='iceberg_test_order',
            position_id='pos_ice',
            client_order_id='client_ice',
            symbol='BTC/USDT',
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal('500'),
            filled_quantity=Decimal('0'),
            status=OrderStatus.PENDING
        )
        
        # Mock iceberg execution
        with patch.object(executor, '_execute_iceberg_slice') as mock_iceberg:
            mock_iceberg.return_value = ExecutionResult(
                success=True,
                order=Mock(filled_quantity=Decimal('100')),
                message="Iceberg executed",
                actual_price=Decimal('100')
            )
            
            # Create a slice
            from genesis.engine.executor.vwap import VWAPSlice
            slice_obj = VWAPSlice(
                slice_id='slice_1',
                parent_order_id=order.order_id,
                symbol=Symbol('BTC/USDT'),
                side=OrderSide.BUY,
                quantity=Decimal('100'),
                target_price=None,
                scheduled_time=datetime.now(timezone.utc),
                bucket_minute=0
            )
            
            # Execute with iceberg
            await executor._execute_single_slice(
                slice_obj, order, ExecutionMode.NORMAL, use_iceberg=True
            )
            
            # Verify iceberg was used
            mock_iceberg.assert_called()
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_vwap_error_recovery(self, vwap_system):
        """Test VWAP execution error handling and recovery."""
        executor = vwap_system['executor']
        gateway = vwap_system['gateway']
        
        # Setup to fail first attempt, succeed on retry
        call_count = 0
        async def mock_place_with_retry(order):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            # Success on retry
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            return {'status': 'FILLED'}
        
        gateway.place_order = AsyncMock(side_effect=mock_place_with_retry)
        
        order = Order(
            order_id='error_test_order',
            position_id='pos_err',
            client_order_id='client_err',
            symbol='BTC/USDT',
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            price=None,
            quantity=Decimal('100'),
            filled_quantity=Decimal('0'),
            status=OrderStatus.PENDING
        )
        
        # Execute market order with retry logic
        result = await executor.execute_market_order(order)
        
        # Should succeed after retry
        assert result.success is True
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_vwap_concurrent_executions(self, vwap_system):
        """Test handling multiple concurrent VWAP executions."""
        executor = vwap_system['executor']
        
        # Create multiple orders
        orders = []
        for i in range(3):
            order = Order(
                order_id=f'concurrent_order_{i}',
                position_id=f'pos_{i}',
                client_order_id=f'client_{i}',
                symbol='BTC/USDT' if i < 2 else 'ETH/USDT',
                type=OrderType.MARKET,
                side=OrderSide.BUY,
                price=None,
                quantity=Decimal('100'),
                filled_quantity=Decimal('0'),
                status=OrderStatus.PENDING
            )
            orders.append(order)
        
        # Mock volume predictions
        mock_prediction = VolumePrediction(
            symbol=Symbol('BTC/USDT'),
            prediction_time=datetime.now(timezone.utc),
            predicted_buckets={0: Decimal('10000')},
            confidence_scores={0: Decimal('0.9')},
            total_predicted=Decimal('10000'),
            model_accuracy=Decimal('0.85')
        )
        
        with patch.object(executor.volume_analyzer, 'predict_intraday_volume', 
                         return_value=mock_prediction):
            with patch.object(executor.volume_analyzer, 'get_optimal_participation_rate',
                            return_value={0: Decimal('0.1')}):
                # Start executions concurrently
                tasks = []
                for order in orders:
                    with patch.object(executor, '_execute_slices',
                                    return_value=ExecutionResult(
                                        success=True,
                                        order=order,
                                        message="Completed",
                                        actual_price=Decimal('100')
                                    )):
                        task = asyncio.create_task(executor.execute_vwap(order))
                        tasks.append(task)
                
                # Wait for all to complete
                results = await asyncio.gather(*tasks)
                
                # Verify all succeeded
                assert all(r.success for r in results)
                assert len(executor._active_executions) == 3
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_vwap_performance_requirements(self, vwap_system):
        """Test that VWAP calculations meet performance requirements."""
        tracker = vwap_system['tracker']
        
        # Add 1000 trades (stress test)
        symbol = Symbol('BTC/USDT')
        trades = []
        base_time = datetime.now(timezone.utc)
        
        for i in range(1000):
            trade = Trade(
                timestamp=base_time - timedelta(seconds=i),
                price=Decimal('100') + Decimal(str(i * 0.001)),
                volume=Decimal('10')
            )
            trades.append(trade)
        
        # Measure VWAP calculation time
        import time
        start = time.time()
        vwap = await tracker.calculate_real_time_vwap(trades)
        elapsed_ms = (time.time() - start) * 1000
        
        # Should complete within 100ms as per requirements
        assert elapsed_ms < 100
        assert vwap > Decimal('0')
        
        # Calculate expected VWAP for verification
        total_value = sum(t.value for t in trades)
        total_volume = sum(t.volume for t in trades)
        expected_vwap = total_value / total_volume
        assert abs(vwap - expected_vwap) < Decimal('0.01')