"""Unit tests for VWAP execution strategy."""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from genesis.core.models import Order, OrderSide, OrderStatus, Signal, SignalType
from genesis.execution.execution_scheduler import (
    ExecutionPlan,
    ExecutionScheduler,
    ExecutionTask,
    ScheduleType,
    SchedulerConfig,
)
from genesis.execution.order_slicer import OrderSlice, OrderSlicer, SliceConfig, SlicingMethod
from genesis.execution.volume_curve import VolumeCurveEstimator, VolumeProfile
from genesis.strategies.strategist.vwap_execution import (
    UrgencyLevel,
    VWAPExecutionConfig,
    VWAPExecutionState,
    VWAPExecutionStrategy,
    VWAPOrderConfig,
)


class TestVWAPExecutionStrategy:
    """Test VWAP execution strategy."""
    
    @pytest.fixture
    def strategy(self):
        """Create VWAP strategy instance."""
        config = VWAPExecutionConfig(
            name="TestVWAP",
            max_participation_rate=Decimal("0.10"),
            default_urgency=UrgencyLevel.MEDIUM,
            volume_lookback_days=20,
            volume_curve_intervals=48
        )
        return VWAPExecutionStrategy(config)
    
    @pytest.fixture
    def parent_order_config(self):
        """Create parent order configuration."""
        return VWAPOrderConfig(
            parent_order_id=uuid4(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            total_quantity=Decimal("10.0"),
            target_participation_rate=Decimal("0.10"),
            urgency=UrgencyLevel.MEDIUM,
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(hours=1),
            min_slice_size=Decimal("0.1"),
            max_slice_size=Decimal("2.0")
        )
    
    @pytest.mark.asyncio
    async def test_create_parent_order(self, strategy, parent_order_config):
        """Test creating parent order."""
        parent_id = await strategy.create_parent_order(parent_order_config)
        
        assert parent_id == parent_order_config.parent_order_id
        assert parent_id in strategy.parent_orders
        assert parent_id in strategy.order_states
        
        state = strategy.order_states[parent_id]
        assert state.remaining_quantity == Decimal("10.0")
        assert state.executed_quantity == Decimal("0")
        assert len(state.schedule) > 0
    
    @pytest.mark.asyncio
    async def test_generate_signals(self, strategy, parent_order_config):
        """Test signal generation."""
        # Create parent order
        parent_id = await strategy.create_parent_order(parent_order_config)
        
        # Mock should_execute_slice to return True
        strategy._should_execute_slice = AsyncMock(return_value=True)
        
        # Generate signals
        signals = await strategy.generate_signals()
        
        assert len(signals) > 0
        signal = signals[0]
        assert signal.signal_type == SignalType.BUY
        assert signal.symbol == "BTCUSDT"
        assert signal.metadata["parent_order_id"] == str(parent_id)
    
    @pytest.mark.asyncio
    async def test_urgency_levels(self, strategy):
        """Test different urgency levels."""
        # Test LOW urgency
        config_low = VWAPOrderConfig(
            total_quantity=Decimal("10.0"),
            urgency=UrgencyLevel.LOW,
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(hours=2)
        )
        parent_id_low = await strategy.create_parent_order(config_low)
        state_low = strategy.order_states[parent_id_low]
        
        # Test EMERGENCY urgency
        config_emergency = VWAPOrderConfig(
            total_quantity=Decimal("10.0"),
            urgency=UrgencyLevel.EMERGENCY,
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(hours=2)
        )
        parent_id_emergency = await strategy.create_parent_order(config_emergency)
        state_emergency = strategy.order_states[parent_id_emergency]
        
        # Emergency should have fewer slices (immediate execution)
        assert len(state_emergency.schedule) < len(state_low.schedule)
        assert state_emergency.total_slices == 1  # Single immediate execution
    
    @pytest.mark.asyncio
    async def test_participation_limit_enforcement(self, strategy):
        """Test participation rate limit enforcement."""
        quantity = Decimal("1000")
        market_volume = Decimal("5000")
        
        # Should limit to 10% of market volume
        limited = strategy.enforce_participation_limits(quantity, market_volume)
        assert limited == Decimal("500")  # 10% of 5000
        
        # Small quantity should pass through
        small_quantity = Decimal("100")
        limited_small = strategy.enforce_participation_limits(small_quantity, market_volume)
        assert limited_small == small_quantity
    
    @pytest.mark.asyncio
    async def test_emergency_liquidation(self, strategy, parent_order_config):
        """Test emergency liquidation mode."""
        parent_id = await strategy.create_parent_order(parent_order_config)
        
        # Trigger emergency liquidation
        await strategy.trigger_emergency_liquidation(parent_id)
        
        config = strategy.parent_orders[parent_id]
        state = strategy.order_states[parent_id]
        
        assert config.urgency == UrgencyLevel.EMERGENCY
        assert len(state.schedule) == 1
        assert state.total_slices == 1
    
    @pytest.mark.asyncio
    async def test_child_order_update(self, strategy, parent_order_config):
        """Test updating child order status."""
        parent_id = await strategy.create_parent_order(parent_order_config)
        
        # Create mock child order
        child_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            filled_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED
        )
        
        await strategy.update_child_order(parent_id, child_order)
        
        state = strategy.order_states[parent_id]
        assert state.executed_quantity == Decimal("1.0")
        assert state.remaining_quantity == Decimal("9.0")
        assert len(state.completed_orders) == 1
        assert state.average_price == Decimal("50000")
    
    @pytest.mark.asyncio
    async def test_implementation_shortfall_calculation(self, strategy, parent_order_config):
        """Test implementation shortfall calculation."""
        parent_id = await strategy.create_parent_order(parent_order_config)
        state = strategy.order_states[parent_id]
        
        # Set execution data
        state.executed_quantity = Decimal("10.0")
        state.average_price = Decimal("50500")
        
        # Mock benchmark VWAP
        strategy.calculate_vwap_benchmark = AsyncMock(return_value=Decimal("50000"))
        
        shortfall = await strategy.calculate_implementation_shortfall(parent_id)
        
        # (50500 - 50000) / 50000 * 10000 = 100 bps
        assert shortfall == Decimal("100")
        assert state.implementation_shortfall == Decimal("100")


class TestVolumeCurveEstimator:
    """Test volume curve estimator."""
    
    @pytest.fixture
    def estimator(self):
        """Create volume curve estimator."""
        return VolumeCurveEstimator(lookback_days=20, intervals_per_day=48)
    
    @pytest.mark.asyncio
    async def test_estimate_volume_curve(self, estimator):
        """Test volume curve estimation."""
        profile = await estimator.estimate_volume_curve("BTCUSDT")
        
        assert profile.symbol == "BTCUSDT"
        assert len(profile.intervals) == 48
        assert len(profile.normalized_volumes) == 48
        assert abs(sum(profile.normalized_volumes) - Decimal("1.0")) < Decimal("0.01")
    
    @pytest.mark.asyncio
    async def test_u_shaped_distribution(self, estimator):
        """Test U-shaped volume distribution."""
        profile = await estimator.estimate_volume_curve("BTCUSDT")
        
        # Early morning should have higher volume than mid-day
        morning_volume = profile.normalized_volumes[2]  # ~1 AM
        midday_volume = profile.normalized_volumes[24]  # ~12 PM
        
        assert morning_volume > midday_volume
    
    @pytest.mark.asyncio
    async def test_special_events_adjustment(self, estimator):
        """Test adjustment for special events."""
        base_profile = await estimator.estimate_volume_curve("BTCUSDT")
        
        # Get profile with earnings event
        special_profile = await estimator.estimate_volume_curve(
            "BTCUSDT",
            special_events=["earnings"]
        )
        
        # Volume should be higher with special event
        assert special_profile.normalized_volumes[0] > base_profile.normalized_volumes[0]
    
    def test_get_current_interval_volume(self, estimator):
        """Test getting volume for current interval."""
        # Create mock profile
        intervals = [datetime.now(UTC) + timedelta(minutes=i*30) for i in range(48)]
        normalized_volumes = [Decimal("0.02") for _ in range(48)]
        
        profile = VolumeProfile(
            symbol="BTCUSDT",
            intervals=intervals,
            volumes=[Decimal("1000") for _ in range(48)],
            normalized_volumes=normalized_volumes,
            total_volume=Decimal("48000"),
            date=datetime.now(UTC)
        )
        
        # Get volume for current time
        current_time = intervals[5] + timedelta(minutes=15)
        volume, idx = estimator.get_current_interval_volume(profile, current_time)
        
        assert idx == 5
        assert volume == Decimal("0.02")


class TestOrderSlicer:
    """Test order slicer."""
    
    @pytest.fixture
    def slicer(self):
        """Create order slicer."""
        config = SliceConfig(
            method=SlicingMethod.VOLUME_WEIGHTED,
            min_slice_size=Decimal("0.1"),
            max_slice_size=Decimal("2.0"),
            max_slices=100,
            randomize_sizes=False  # Disable for deterministic tests
        )
        return OrderSlicer(config)
    
    def test_linear_slicing(self, slicer):
        """Test linear slicing method."""
        slices = slicer.slice_order(
            total_quantity=Decimal("10.0"),
            method=SlicingMethod.LINEAR
        )
        
        assert len(slices) > 0
        assert sum(s.quantity for s in slices) == Decimal("10.0")
        
        # Should be roughly equal sized
        sizes = [s.quantity for s in slices]
        assert max(sizes) - min(sizes) < Decimal("1.0")
    
    def test_iceberg_slicing(self, slicer):
        """Test iceberg slicing method."""
        slices = slicer.slice_order(
            total_quantity=Decimal("10.0"),
            method=SlicingMethod.ICEBERG
        )
        
        assert len(slices) > 0
        assert sum(s.quantity for s in slices) == Decimal("10.0")
        
        # Each slice should have visible quantity
        for slice in slices:
            assert slice.visible_quantity is not None
            assert slice.visible_quantity <= slice.quantity
    
    @pytest.mark.asyncio
    async def test_volume_weighted_slicing(self, slicer):
        """Test volume-weighted slicing."""
        # Create mock volume profile
        estimator = VolumeCurveEstimator()
        profile = await estimator.estimate_volume_curve("BTCUSDT")
        
        slices = slicer.slice_order(
            total_quantity=Decimal("10.0"),
            method=SlicingMethod.VOLUME_WEIGHTED,
            volume_profile=profile
        )
        
        assert len(slices) > 0
        assert sum(s.quantity for s in slices) == Decimal("10.0")
        
        # Slices should have execution times
        for slice in slices[:-1]:  # Except potential residual
            if not slice.metadata.get("residual"):
                assert slice.execution_time is not None
    
    def test_adaptive_slicing(self, slicer):
        """Test adaptive slicing based on market conditions."""
        market_conditions = {
            "liquidity": 1000000,
            "volatility": 0.03,
            "spread": 0.002,
            "volume": 10000000
        }
        
        slices = slicer.slice_order(
            total_quantity=Decimal("10.0"),
            method=SlicingMethod.ADAPTIVE,
            market_conditions=market_conditions
        )
        
        assert len(slices) > 0
        assert sum(s.quantity for s in slices) == Decimal("10.0")
        
        # Should have metadata about conditions
        assert slices[0].metadata.get("liquidity") is not None
    
    def test_generate_child_orders(self, slicer):
        """Test generating child orders from slices."""
        slices = slicer.slice_order(
            total_quantity=Decimal("10.0"),
            method=SlicingMethod.LINEAR
        )
        
        parent_id = str(uuid4())
        child_orders = slicer.generate_child_orders(
            parent_order_id=parent_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            slices=slices
        )
        
        assert len(child_orders) == len(slices)
        assert all(o.symbol == "BTCUSDT" for o in child_orders)
        assert all(o.side == OrderSide.BUY for o in child_orders)
        assert sum(o.quantity for o in child_orders) == Decimal("10.0")


class TestExecutionScheduler:
    """Test execution scheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Create execution scheduler."""
        config = SchedulerConfig(
            min_interval_seconds=5,
            max_interval_seconds=300,
            adaptive_reschedule=True
        )
        return ExecutionScheduler(config)
    
    @pytest.fixture
    def slices(self):
        """Create test slices."""
        return [
            OrderSlice(
                slice_number=i+1,
                total_slices=5,
                quantity=Decimal("2.0"),
                priority=i
            )
            for i in range(5)
        ]
    
    @pytest.mark.asyncio
    async def test_create_execution_plan(self, scheduler, slices):
        """Test creating execution plan."""
        plan = await scheduler.create_execution_plan(
            parent_order_id="test_order",
            slices=slices,
            urgency=UrgencyLevel.MEDIUM,
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(hours=1)
        )
        
        assert plan.plan_id == "plan_test_order"
        assert len(plan.tasks) == len(slices)
        assert plan.total_quantity == Decimal("10.0")
        assert plan.schedule_type == ScheduleType.SCHEDULED
    
    @pytest.mark.asyncio
    async def test_emergency_schedule_type(self, scheduler, slices):
        """Test emergency schedule type."""
        plan = await scheduler.create_execution_plan(
            parent_order_id="emergency_order",
            slices=slices,
            urgency=UrgencyLevel.EMERGENCY
        )
        
        assert plan.schedule_type == ScheduleType.IMMEDIATE
        
        # All tasks should be scheduled immediately
        now = datetime.now(UTC)
        for task in plan.tasks:
            assert (task.scheduled_time - now).total_seconds() < 60
    
    @pytest.mark.asyncio
    async def test_scheduler_lifecycle(self, scheduler):
        """Test scheduler start/stop lifecycle."""
        await scheduler.start()
        assert scheduler.is_running
        
        await asyncio.sleep(0.1)  # Let it run briefly
        
        await scheduler.stop()
        assert not scheduler.is_running
    
    @pytest.mark.asyncio
    async def test_task_completion(self, scheduler, slices):
        """Test task completion tracking."""
        plan = await scheduler.create_execution_plan(
            parent_order_id="test_order",
            slices=slices,
            urgency=UrgencyLevel.MEDIUM
        )
        
        # Complete first task
        task = plan.tasks[0]
        scheduler.active_tasks[task.task_id] = task
        scheduler.complete_task(task.task_id, success=True)
        
        assert task.status == "COMPLETED"
        assert task.task_id not in scheduler.active_tasks
        assert plan.executed_quantity == Decimal("2.0")
    
    @pytest.mark.asyncio
    async def test_adaptive_rescheduling(self, scheduler, slices):
        """Test adaptive rescheduling."""
        plan = await scheduler.create_execution_plan(
            parent_order_id="adaptive_order",
            slices=slices,
            urgency=UrgencyLevel.HIGH,
            start_time=datetime.now(UTC) - timedelta(minutes=30),
            end_time=datetime.now(UTC) + timedelta(minutes=30)
        )
        
        # Set plan to adaptive type
        plan.schedule_type = ScheduleType.ADAPTIVE
        
        # Simulate being behind schedule
        plan.executed_quantity = Decimal("2.0")  # Only 20% done
        
        # Trigger acceleration
        await scheduler._accelerate_schedule(plan, datetime.now(UTC))
        
        # Check that pending tasks are moved earlier
        pending_tasks = [t for t in plan.tasks if t.status == "PENDING"]
        assert len(pending_tasks) > 0