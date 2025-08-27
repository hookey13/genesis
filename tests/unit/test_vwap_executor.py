"""Unit tests for VWAPExecutor."""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

from genesis.engine.executor.vwap import (
    VWAPExecutor, ExecutionMode, VWAPSlice
)
from genesis.engine.executor.base import (
    Order, OrderSide, OrderType, OrderStatus, ExecutionResult
)
from genesis.core.constants import TradingTier
from genesis.core.models import Symbol
from genesis.core.exceptions import TierViolation
from genesis.exchange.gateway import BinanceGateway as ExchangeGateway
from genesis.analytics.volume_analyzer import VolumeAnalyzer, VolumePrediction
from genesis.analytics.vwap_tracker import VWAPTracker
from genesis.engine.event_bus import EventBus


@pytest.fixture
def mock_exchange_gateway():
    """Create a mock exchange gateway."""
    gateway = Mock(spec=ExchangeGateway)
    gateway.place_order = AsyncMock()
    gateway.cancel_order = AsyncMock()
    gateway.get_order_status = AsyncMock()
    return gateway


@pytest.fixture
def mock_volume_analyzer():
    """Create a mock volume analyzer."""
    analyzer = Mock(spec=VolumeAnalyzer)
    analyzer.predict_intraday_volume = AsyncMock()
    analyzer.get_optimal_participation_rate = Mock()
    return analyzer


@pytest.fixture
def mock_vwap_tracker():
    """Create a mock VWAP tracker."""
    tracker = Mock(spec=VWAPTracker)
    tracker.start_execution_tracking = Mock()
    tracker.update_execution = Mock()
    tracker.complete_execution = Mock()
    tracker.get_current_vwap = Mock(return_value=Decimal('100'))
    return tracker


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    event_bus = Mock(spec=EventBus)
    event_bus.emit = AsyncMock()
    return event_bus


@pytest.fixture
def vwap_config():
    """Create VWAP configuration."""
    return {
        'vwap_participation_rate_percent': 10.0,
        'vwap_min_slice_size_usd': 50.00,
        'vwap_max_slices': 100,
        'vwap_time_window_minutes': 240,
        'vwap_aggressive_threshold_percent': 5.0
    }


@pytest.fixture
def vwap_executor(mock_exchange_gateway, mock_volume_analyzer, mock_vwap_tracker, mock_event_bus, vwap_config):
    """Create a VWAPExecutor instance with mock dependencies."""
    return VWAPExecutor(
        tier=TradingTier.STRATEGIST,
        exchange_gateway=mock_exchange_gateway,
        volume_analyzer=mock_volume_analyzer,
        vwap_tracker=mock_vwap_tracker,
        event_bus=mock_event_bus,
        config=vwap_config
    )


@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id='test_order_123',
        position_id='pos_123',
        client_order_id=str(uuid4()),
        symbol='BTC/USDT',
        type=OrderType.MARKET,
        side=OrderSide.BUY,
        price=None,
        quantity=Decimal('1000'),
        filled_quantity=Decimal('0'),
        status=OrderStatus.PENDING
    )


@pytest.fixture
def sample_volume_prediction():
    """Create a sample volume prediction."""
    return VolumePrediction(
        symbol=Symbol('BTC/USDT'),
        prediction_time=datetime.now(timezone.utc),
        predicted_buckets={
            0: Decimal('10000'),
            30: Decimal('15000'),
            60: Decimal('20000'),
            90: Decimal('12000')
        },
        confidence_scores={
            0: Decimal('0.9'),
            30: Decimal('0.85'),
            60: Decimal('0.8'),
            90: Decimal('0.75')
        },
        total_predicted=Decimal('57000'),
        model_accuracy=Decimal('0.85')
    )


class TestVWAPSlice:
    """Test VWAPSlice class."""
    
    def test_slice_initialization(self):
        """Test VWAPSlice initialization."""
        slice_obj = VWAPSlice(
            slice_id='slice_1',
            parent_order_id='order_123',
            symbol=Symbol('BTC/USDT'),
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            target_price=Decimal('50000'),
            scheduled_time=datetime.now(timezone.utc),
            bucket_minute=30
        )
        
        assert slice_obj.slice_id == 'slice_1'
        assert slice_obj.parent_order_id == 'order_123'
        assert slice_obj.symbol == Symbol('BTC/USDT')
        assert slice_obj.side == OrderSide.BUY
        assert slice_obj.quantity == Decimal('100')
        assert slice_obj.target_price == Decimal('50000')
        assert slice_obj.bucket_minute == 30
        assert slice_obj.executed_quantity == Decimal('0')
        assert slice_obj.executed_value == Decimal('0')
        assert slice_obj.status == OrderStatus.PENDING
        assert slice_obj.order is None
        assert slice_obj.attempts == 0
        assert slice_obj.last_error is None


class TestVWAPExecutor:
    """Test VWAPExecutor class."""
    
    def test_initialization(self, vwap_executor, vwap_config):
        """Test VWAPExecutor initialization."""
        assert vwap_executor.tier == TradingTier.STRATEGIST
        assert vwap_executor.default_participation_rate == Decimal('0.1')
        assert vwap_executor.min_slice_size_usd == Decimal('50')
        assert vwap_executor.max_slices == 100
        assert vwap_executor.time_window_minutes == 240
        assert vwap_executor.aggressive_threshold == Decimal('0.05')
    
    @pytest.mark.asyncio
    async def test_execute_vwap_tier_restriction(self, mock_exchange_gateway, mock_volume_analyzer, 
                                                 mock_vwap_tracker, mock_event_bus, vwap_config, sample_order):
        """Test that VWAP execution requires Strategist tier."""
        # Create executor with Hunter tier (below Strategist)
        executor = VWAPExecutor(
            tier=TradingTier.HUNTER,
            exchange_gateway=mock_exchange_gateway,
            volume_analyzer=mock_volume_analyzer,
            vwap_tracker=mock_vwap_tracker,
            event_bus=mock_event_bus,
            config=vwap_config
        )
        
        # Should raise TierViolation
        with pytest.raises(TierViolation):
            await executor.execute_vwap(sample_order)
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_execute_vwap_success(self, vwap_executor, mock_volume_analyzer, mock_vwap_tracker,
                                       sample_order, sample_volume_prediction):
        """Test successful VWAP execution."""
        # Setup mocks
        mock_volume_analyzer.predict_intraday_volume.return_value = sample_volume_prediction
        mock_volume_analyzer.get_optimal_participation_rate.return_value = {
            0: Decimal('0.1'),
            30: Decimal('0.1'),
            60: Decimal('0.1'),
            90: Decimal('0.1')
        }
        
        # Mock the slice execution
        with patch.object(vwap_executor, '_execute_slices', return_value=ExecutionResult(
            success=True,
            order=sample_order,
            message="VWAP execution completed",
            actual_price=Decimal('100'),
            slippage_percent=Decimal('0.5')
        )) as mock_execute:
            
            result = await vwap_executor.execute_vwap(sample_order)
            
            assert result.success is True
            assert "VWAP execution completed" in result.message
            
            # Verify tracking started
            mock_vwap_tracker.start_execution_tracking.assert_called_once()
            
            # Verify prediction was called
            mock_volume_analyzer.predict_intraday_volume.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_calculate_slices(self, vwap_executor, sample_order, sample_volume_prediction):
        """Test slice calculation logic."""
        vwap_executor.volume_analyzer.get_optimal_participation_rate = Mock(return_value={
            0: Decimal('0.1'),
            30: Decimal('0.1'),
            60: Decimal('0.1'),
            90: Decimal('0.1')
        })
        
        slices = await vwap_executor._calculate_slices(
            sample_order,
            sample_volume_prediction,
            Decimal('0.1'),
            ExecutionMode.NORMAL,
            240
        )
        
        assert len(slices) > 0
        assert all(isinstance(s, VWAPSlice) for s in slices)
        
        # Verify total quantity matches order
        total_quantity = sum(s.quantity for s in slices)
        assert abs(total_quantity - sample_order.quantity) < Decimal('0.01')
        
        # Verify slice sizes respect minimum
        for slice_obj in slices:
            assert slice_obj.quantity >= vwap_executor.min_slice_size_usd / Decimal('100')
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_calculate_slices_aggressive_mode(self, vwap_executor, sample_order, sample_volume_prediction):
        """Test slice calculation in aggressive mode."""
        vwap_executor.volume_analyzer.get_optimal_participation_rate = Mock(return_value={
            0: Decimal('0.1'),
            30: Decimal('0.1')
        })
        
        slices_normal = await vwap_executor._calculate_slices(
            sample_order,
            sample_volume_prediction,
            Decimal('0.1'),
            ExecutionMode.NORMAL,
            240
        )
        
        slices_aggressive = await vwap_executor._calculate_slices(
            sample_order,
            sample_volume_prediction,
            Decimal('0.1'),
            ExecutionMode.AGGRESSIVE,
            240
        )
        
        # Aggressive mode should front-load execution
        if len(slices_aggressive) > 0 and len(slices_normal) > 0:
            # First slice in aggressive mode should be larger
            assert slices_aggressive[0].quantity >= slices_normal[0].quantity
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_calculate_slices_passive_mode(self, vwap_executor, sample_order, sample_volume_prediction):
        """Test slice calculation in passive mode."""
        vwap_executor.volume_analyzer.get_optimal_participation_rate = Mock(return_value={
            0: Decimal('0.1'),
            30: Decimal('0.1')
        })
        
        slices_normal = await vwap_executor._calculate_slices(
            sample_order,
            sample_volume_prediction,
            Decimal('0.1'),
            ExecutionMode.NORMAL,
            240
        )
        
        slices_passive = await vwap_executor._calculate_slices(
            sample_order,
            sample_volume_prediction,
            Decimal('0.1'),
            ExecutionMode.PASSIVE,
            240
        )
        
        # Passive mode should spread out execution more
        if len(slices_passive) > 0 and len(slices_normal) > 0:
            # Slices should be smaller in passive mode
            assert slices_passive[0].quantity <= slices_normal[0].quantity * Decimal('1.2')
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_execute_single_slice(self, vwap_executor, mock_vwap_tracker):
        """Test execution of a single slice."""
        parent_order = Order(
            order_id='parent_123',
            position_id='pos_123',
            client_order_id=str(uuid4()),
            symbol='BTC/USDT',
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal('1000')
        )
        
        slice_obj = VWAPSlice(
            slice_id='slice_1',
            parent_order_id='parent_123',
            symbol=Symbol('BTC/USDT'),
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            target_price=None,
            scheduled_time=datetime.now(timezone.utc),
            bucket_minute=30
        )
        
        # Mock successful market order execution
        with patch.object(vwap_executor, 'execute_market_order', return_value=ExecutionResult(
            success=True,
            order=Mock(filled_quantity=Decimal('100')),
            message="Executed",
            actual_price=Decimal('50000')
        )):
            await vwap_executor._execute_single_slice(
                slice_obj, parent_order, ExecutionMode.NORMAL, False
            )
            
            assert slice_obj.status == OrderStatus.FILLED
            assert slice_obj.executed_quantity == Decimal('100')
            assert slice_obj.executed_value == Decimal('100') * Decimal('50000')
            
            # Verify VWAP tracker was updated
            mock_vwap_tracker.update_execution.assert_called()
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_execute_single_slice_failure_retry(self, vwap_executor):
        """Test slice execution failure and retry."""
        parent_order = Mock()
        slice_obj = VWAPSlice(
            slice_id='slice_1',
            parent_order_id='parent_123',
            symbol=Symbol('BTC/USDT'),
            side=OrderSide.BUY,
            quantity=Decimal('100'),
            target_price=None,
            scheduled_time=datetime.now(timezone.utc),
            bucket_minute=30
        )
        
        # Mock failed then successful execution
        exec_results = [
            ExecutionResult(success=False, order=Mock(), message="Failed", error="Network error"),
            ExecutionResult(success=True, order=Mock(filled_quantity=Decimal('100')), 
                          message="Executed", actual_price=Decimal('50000'))
        ]
        
        with patch.object(vwap_executor, 'execute_market_order', side_effect=exec_results):
            await vwap_executor._execute_single_slice(
                slice_obj, parent_order, ExecutionMode.NORMAL, False
            )
            
            # Should retry and eventually succeed
            assert slice_obj.attempts == 1
            assert slice_obj.status == OrderStatus.FILLED
    
    def test_get_order_type(self, vwap_executor):
        """Test order type selection based on mode."""
        assert vwap_executor._get_order_type(ExecutionMode.PASSIVE) == OrderType.LIMIT_MAKER
        assert vwap_executor._get_order_type(ExecutionMode.AGGRESSIVE) == OrderType.MARKET
        assert vwap_executor._get_order_type(ExecutionMode.NORMAL) == OrderType.LIMIT
    
    @pytest.mark.asyncio
    @patch.dict('os.environ', {'PYTEST_CURRENT_TEST': 'test'})
    async def test_should_switch_to_aggressive(self, vwap_executor):
        """Test logic for switching to aggressive mode."""
        order = Mock(quantity=Decimal('1000'), filled_quantity=Decimal('200'))
        
        # Create slices with different completion states
        slices = [
            Mock(status=OrderStatus.FILLED),
            Mock(status=OrderStatus.FILLED),
            Mock(status=OrderStatus.PENDING),
            Mock(status=OrderStatus.PENDING),
            Mock(status=OrderStatus.PENDING)
        ]
        
        # 20% filled but 40% of time passed - should switch
        should_switch = await vwap_executor._should_switch_to_aggressive(order, slices)
        assert should_switch is True
        
        # Update to better progress
        order.filled_quantity = Decimal('450')
        should_switch = await vwap_executor._should_switch_to_aggressive(order, slices)
        assert should_switch is False
    
    def test_build_final_result(self, vwap_executor, sample_order):
        """Test building final execution result."""
        slices = [
            Mock(executed_quantity=Decimal('300'), executed_value=Decimal('30000')),
            Mock(executed_quantity=Decimal('400'), executed_value=Decimal('40400')),
            Mock(executed_quantity=Decimal('300'), executed_value=Decimal('29700'))
        ]
        
        result = vwap_executor._build_final_result(sample_order, slices, "Test complete")
        
        assert result.success is True
        assert result.order.filled_quantity == Decimal('1000')
        assert result.order.status == OrderStatus.FILLED
        assert result.actual_price == Decimal('100.1')  # (30000+40400+29700)/1000
        assert result.message == "Test complete"
    
    def test_build_final_result_partial_fill(self, vwap_executor, sample_order):
        """Test building result for partial fill."""
        slices = [
            Mock(executed_quantity=Decimal('300'), executed_value=Decimal('30000')),
            Mock(executed_quantity=Decimal('200'), executed_value=Decimal('20000'))
        ]
        
        result = vwap_executor._build_final_result(sample_order, slices, "Partial fill")
        
        assert result.success is False
        assert result.order.filled_quantity == Decimal('500')
        assert result.order.status == OrderStatus.PARTIAL
    
    @pytest.mark.asyncio
    async def test_cleanup_execution(self, vwap_executor):
        """Test cleanup of execution tracking."""
        order_id = 'test_order'
        vwap_executor._active_executions[order_id] = []
        
        # Create a mock task
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        vwap_executor._execution_tasks[order_id] = mock_task
        
        await vwap_executor._cleanup_execution(order_id)
        
        assert order_id not in vwap_executor._active_executions
        assert order_id not in vwap_executor._execution_tasks
        mock_task.cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, vwap_executor):
        """Test order cancellation."""
        result = await vwap_executor.cancel_order('order_123', 'BTC/USDT')
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, vwap_executor):
        """Test cancelling all orders."""
        # Add some active executions
        vwap_executor._active_executions['order_1'] = [
            Mock(symbol=Symbol('BTC/USDT')),
            Mock(symbol=Symbol('BTC/USDT'))
        ]
        vwap_executor._active_executions['order_2'] = [
            Mock(symbol=Symbol('ETH/USDT'))
        ]
        
        # Cancel all
        count = await vwap_executor.cancel_all_orders()
        assert count == 3
        assert len(vwap_executor._active_executions) == 0
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders_filtered(self, vwap_executor):
        """Test cancelling orders filtered by symbol."""
        vwap_executor._active_executions['order_1'] = [
            Mock(symbol=Symbol('BTC/USDT')),
            Mock(symbol=Symbol('BTC/USDT'))
        ]
        vwap_executor._active_executions['order_2'] = [
            Mock(symbol=Symbol('ETH/USDT'))
        ]
        
        # Cancel only BTC/USDT
        count = await vwap_executor.cancel_all_orders('BTC/USDT')
        assert count == 2
        assert 'order_2' in vwap_executor._active_executions
    
    @pytest.mark.asyncio
    async def test_get_order_status(self, vwap_executor):
        """Test getting order status."""
        # Test active VWAP execution
        vwap_executor._active_executions['order_123'] = [
            Mock(quantity=Decimal('100'), executed_quantity=Decimal('50'), side=OrderSide.BUY),
            Mock(quantity=Decimal('100'), executed_quantity=Decimal('30'), side=OrderSide.BUY)
        ]
        
        status = await vwap_executor.get_order_status('order_123', 'BTC/USDT')
        
        assert status.order_id == 'order_123'
        assert status.quantity == Decimal('200')
        assert status.filled_quantity == Decimal('80')
        assert status.status == OrderStatus.PARTIAL
        
        # Test non-existent order
        status = await vwap_executor.get_order_status('nonexistent', 'BTC/USDT')
        assert status.order_id == 'nonexistent'
        assert status.status == OrderStatus.PENDING