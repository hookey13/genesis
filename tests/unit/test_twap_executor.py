"""Unit tests for TWAP executor."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from genesis.core.exceptions import OrderExecutionError, ValidationError
from genesis.core.models import Account, TradingTier
from genesis.data.market_data_service import VolumeProfile
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
    OrderSide,
    OrderType,
)
from genesis.engine.executor.twap import (
    MAX_DURATION_MINUTES,
    MAX_PARTICIPATION_RATE,
    MIN_DURATION_MINUTES,
    TimeSlice,
    TwapExecution,
    TwapExecutor,
)
from genesis.engine.risk_engine import RiskDecision


class TestTwapExecutor:
    """Test TWAP executor functionality."""

    @pytest.fixture
    def mock_gateway(self):
        """Create mock gateway."""
        gateway = AsyncMock()
        gateway.get_ticker = AsyncMock(
            return_value=MagicMock(last_price=Decimal("50000"), volume=Decimal("1000"))
        )
        return gateway

    @pytest.fixture
    def mock_account(self):
        """Create mock account with Strategist tier."""
        return Account(
            account_id="test-account",
            tier=TradingTier.STRATEGIST,
            balance_usdt=Decimal("100000"),
        )

    @pytest.fixture
    def mock_market_executor(self):
        """Create mock market executor."""
        executor = AsyncMock()
        executor.generate_client_order_id = MagicMock(return_value=str(uuid4()))
        executor.execute_market_order = AsyncMock(
            return_value=ExecutionResult(
                success=True,
                order=MagicMock(),
                message="Order executed",
                actual_price=Decimal("50100"),
                slippage_percent=Decimal("0.1"),
            )
        )
        return executor

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        repo = AsyncMock()
        repo.save_twap_execution = AsyncMock()
        repo.save_twap_slice = AsyncMock()
        repo.update_twap_execution = AsyncMock()
        return repo

    @pytest.fixture
    def mock_market_data_service(self):
        """Create mock market data service."""
        service = AsyncMock()
        service.get_current_price = AsyncMock(return_value=Decimal("50000"))
        service.is_volume_anomaly = AsyncMock(return_value=False)

        # Mock volume profile
        volume_profile = MagicMock(spec=VolumeProfile)
        volume_profile.get_hourly_volumes = MagicMock(
            return_value={i: Decimal("100") for i in range(24)}
        )
        service.get_volume_profile = AsyncMock(return_value=volume_profile)

        return service

    @pytest.fixture
    def mock_risk_engine(self):
        """Create mock risk engine."""
        engine = AsyncMock()
        engine.check_risk_limits = AsyncMock(
            return_value=RiskDecision(approved=True, reason=None)
        )
        return engine

    @pytest.fixture
    def mock_event_bus(self):
        """Create mock event bus."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        return bus

    @pytest.fixture
    def twap_executor(
        self,
        mock_gateway,
        mock_account,
        mock_market_executor,
        mock_repository,
        mock_market_data_service,
        mock_risk_engine,
        mock_event_bus,
    ):
        """Create TWAP executor instance."""
        return TwapExecutor(
            gateway=mock_gateway,
            account=mock_account,
            market_executor=mock_market_executor,
            repository=mock_repository,
            market_data_service=mock_market_data_service,
            risk_engine=mock_risk_engine,
            event_bus=mock_event_bus,
        )

    @pytest.fixture
    def sample_order(self):
        """Create sample order."""
        return Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("1.0"),
        )

    @pytest.mark.asyncio
    async def test_init_requires_strategist_tier(
        self,
        mock_gateway,
        mock_market_executor,
        mock_repository,
        mock_market_data_service,
        mock_risk_engine,
        mock_event_bus,
    ):
        """Test TWAP executor requires Strategist tier."""
        # Create account with insufficient tier
        low_tier_account = Account(
            account_id="test-account",
            tier=TradingTier.HUNTER,
            balance_usdt=Decimal("5000"),
        )

        with pytest.raises(OrderExecutionError, match="STRATEGIST"):
            TwapExecutor(
                gateway=mock_gateway,
                account=low_tier_account,
                market_executor=mock_market_executor,
                repository=mock_repository,
                market_data_service=mock_market_data_service,
                risk_engine=mock_risk_engine,
                event_bus=mock_event_bus,
            )

    @pytest.mark.asyncio
    async def test_validate_duration(self, twap_executor):
        """Test duration validation."""
        # Valid duration
        twap_executor._validate_duration(15)  # Should not raise

        # Too short
        with pytest.raises(ValidationError, match=f"at least {MIN_DURATION_MINUTES}"):
            twap_executor._validate_duration(3)

        # Too long
        with pytest.raises(
            ValidationError, match=f"cannot exceed {MAX_DURATION_MINUTES}"
        ):
            twap_executor._validate_duration(45)

    @pytest.mark.asyncio
    async def test_calculate_time_slices(self, twap_executor):
        """Test time slice calculation with adaptive timing."""
        volume_profile = MagicMock(spec=VolumeProfile)
        volume_profile.get_hourly_volumes = MagicMock(
            return_value={
                i: Decimal("100") if i in range(9, 17) else Decimal("50")
                for i in range(24)
            }
        )

        slices = await twap_executor.calculate_time_slices(
            duration_minutes=10,
            volume_profile=volume_profile,
            total_quantity=Decimal("10.0"),
        )

        assert len(slices) > 0
        assert all(isinstance(s, TimeSlice) for s in slices)
        assert sum(s.target_quantity for s in slices) == Decimal("10.0")
        assert all(s.participation_rate <= MAX_PARTICIPATION_RATE for s in slices)

    @pytest.mark.asyncio
    async def test_track_arrival_price(self, twap_executor, mock_market_data_service):
        """Test arrival price tracking."""
        mock_market_data_service.get_current_price = AsyncMock(
            return_value=Decimal("51000")
        )

        arrival_price = await twap_executor.track_arrival_price("BTC/USDT")

        assert arrival_price == Decimal("51000")
        mock_market_data_service.get_current_price.assert_called_once_with("BTC/USDT")

    @pytest.mark.asyncio
    async def test_check_early_completion_buy(
        self, twap_executor, mock_market_data_service
    ):
        """Test early completion check for buy orders."""
        # Price dropped sufficiently for early completion
        mock_market_data_service.get_current_price = AsyncMock(
            return_value=Decimal("49800")
        )

        should_complete = await twap_executor.check_early_completion(
            symbol="BTC/USDT", side=OrderSide.BUY, target_price=Decimal("50000")
        )

        assert should_complete is True

        # Price not favorable enough
        mock_market_data_service.get_current_price = AsyncMock(
            return_value=Decimal("49950")
        )

        should_complete = await twap_executor.check_early_completion(
            symbol="BTC/USDT", side=OrderSide.BUY, target_price=Decimal("50000")
        )

        assert should_complete is False

    @pytest.mark.asyncio
    async def test_check_early_completion_sell(
        self, twap_executor, mock_market_data_service
    ):
        """Test early completion check for sell orders."""
        # Price increased sufficiently for early completion
        mock_market_data_service.get_current_price = AsyncMock(
            return_value=Decimal("50200")
        )

        should_complete = await twap_executor.check_early_completion(
            symbol="BTC/USDT", side=OrderSide.SELL, target_price=Decimal("50000")
        )

        assert should_complete is True

    @pytest.mark.asyncio
    async def test_enforce_participation_limit(self, twap_executor, mock_gateway):
        """Test participation rate limiting."""
        mock_gateway.get_ticker = AsyncMock(
            return_value=MagicMock(volume=Decimal("14400"))  # 10 per minute average
        )

        # Normal conditions
        adjusted = await twap_executor.enforce_participation_limit(
            slice_size=Decimal("2.0"),
            symbol="BTC/USDT",
            max_participation=Decimal("0.10"),
        )

        # Should be limited to 10% of volume per minute (1.0)
        assert adjusted == Decimal("1.0")

        # During volume anomaly
        twap_executor.market_data_service.is_volume_anomaly = AsyncMock(
            return_value=True
        )

        adjusted = await twap_executor.enforce_participation_limit(
            slice_size=Decimal("2.0"),
            symbol="BTC/USDT",
            max_participation=Decimal("0.10"),
        )

        # Should be further reduced by 50%
        assert adjusted == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_calculate_twap_price(self, twap_executor):
        """Test TWAP price calculation."""
        slices = [
            {"executed_quantity": "1.0", "execution_price": "50000"},
            {"executed_quantity": "1.0", "execution_price": "50100"},
            {"executed_quantity": "1.0", "execution_price": "50200"},
        ]

        twap_price = twap_executor.calculate_twap_price(slices)

        assert twap_price == Decimal("50100")

    @pytest.mark.asyncio
    async def test_calculate_implementation_shortfall(self, twap_executor):
        """Test implementation shortfall calculation."""
        # Positive shortfall (unfavorable)
        shortfall = twap_executor.calculate_implementation_shortfall(
            arrival_price=Decimal("50000"), execution_price=Decimal("50100")
        )

        assert shortfall == Decimal("0.2000")

        # Negative shortfall (favorable)
        shortfall = twap_executor.calculate_implementation_shortfall(
            arrival_price=Decimal("50000"), execution_price=Decimal("49900")
        )

        assert shortfall == Decimal("-0.2000")

    @pytest.mark.asyncio
    async def test_pause_resume_execution(self, twap_executor):
        """Test pausing and resuming TWAP execution."""
        # Create active execution
        execution = TwapExecution(
            execution_id="test-exec",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=Decimal("10.0"),
            duration_minutes=15,
            slices=[],
            arrival_price=Decimal("50000"),
            remaining_quantity=Decimal("10.0"),
        )
        twap_executor.active_executions["test-exec"] = execution

        # Test pause
        success = await twap_executor.pause("test-exec")
        assert success is True
        assert execution.status == "PAUSED"
        assert execution.paused_at is not None

        # Test resume
        success = await twap_executor.resume("test-exec")
        assert success is True
        assert execution.status == "ACTIVE"
        assert execution.resumed_at is not None

        # Test pause non-existent execution
        with pytest.raises(ValidationError):
            await twap_executor.pause("non-existent")

        # Test resume non-paused execution
        with pytest.raises(ValidationError, match="Cannot resume"):
            await twap_executor.resume("test-exec")

    @pytest.mark.asyncio
    async def test_execute_twap_basic_flow(
        self,
        twap_executor,
        sample_order,
        mock_market_executor,
        mock_market_data_service,
        mock_risk_engine,
        mock_event_bus,
    ):
        """Test basic TWAP execution flow."""
        # Mock successful slice executions
        mock_market_executor.execute_market_order = AsyncMock(
            side_effect=[
                ExecutionResult(
                    success=True,
                    order=MagicMock(filled_quantity=Decimal("0.2")),
                    message="Slice 1 executed",
                    actual_price=Decimal("50000"),
                    slippage_percent=Decimal("0.0"),
                ),
                ExecutionResult(
                    success=True,
                    order=MagicMock(filled_quantity=Decimal("0.3")),
                    message="Slice 2 executed",
                    actual_price=Decimal("50100"),
                    slippage_percent=Decimal("0.1"),
                ),
                ExecutionResult(
                    success=True,
                    order=MagicMock(filled_quantity=Decimal("0.5")),
                    message="Slice 3 executed",
                    actual_price=Decimal("50050"),
                    slippage_percent=Decimal("0.05"),
                ),
            ]
        )

        # Speed up execution for testing
        with patch.object(asyncio, "sleep", new_callable=AsyncMock):
            result = await twap_executor.execute_twap(sample_order, duration_minutes=5)

        assert result.success is True
        assert "TWAP execution" in result.message
        assert sample_order.filled_quantity > 0

        # Verify events were published
        assert (
            mock_event_bus.publish.call_count >= 2
        )  # At least start and complete events

    @pytest.mark.asyncio
    async def test_execute_twap_with_risk_rejection(
        self, twap_executor, sample_order, mock_risk_engine
    ):
        """Test TWAP execution when risk engine rejects slices."""
        # Mock risk rejection
        mock_risk_engine.check_risk_limits = AsyncMock(
            return_value=RiskDecision(approved=False, reason="Position limit exceeded")
        )

        with patch.object(asyncio, "sleep", new_callable=AsyncMock):
            result = await twap_executor.execute_twap(sample_order, duration_minutes=5)

        # Execution should complete but with no filled quantity
        assert sample_order.filled_quantity == Decimal("0")

    @pytest.mark.asyncio
    async def test_execute_twap_with_early_completion(
        self, twap_executor, sample_order, mock_market_data_service
    ):
        """Test TWAP execution with early completion triggered."""
        # Mock favorable price for early completion
        mock_market_data_service.get_current_price = AsyncMock(
            side_effect=[
                Decimal("50000"),  # Arrival price
                Decimal("49800"),  # Favorable price for buy
            ]
        )

        with patch.object(asyncio, "sleep", new_callable=AsyncMock):
            result = await twap_executor.execute_twap(sample_order, duration_minutes=10)

        # Check that early completion was noted
        execution_id = (
            list(twap_executor.active_executions.keys())[0]
            if twap_executor.active_executions
            else None
        )
        if execution_id:
            execution = twap_executor.active_executions[execution_id]
            assert execution.early_completion is True

    @pytest.mark.asyncio
    async def test_cancel_order(self, twap_executor, mock_market_executor):
        """Test order cancellation."""
        # Create active execution
        execution = TwapExecution(
            execution_id="test-exec",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=Decimal("10.0"),
            duration_minutes=15,
            slices=[],
            arrival_price=Decimal("50000"),
            remaining_quantity=Decimal("10.0"),
            background_task=MagicMock(),
        )
        execution.background_task.cancel = MagicMock()
        twap_executor.active_executions["test-exec"] = execution

        # Cancel order
        success = await twap_executor.cancel_order("some-order", "BTC/USDT")

        assert success is True
        assert execution.status == "CANCELLED"
        execution.background_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, twap_executor, mock_market_executor):
        """Test cancelling all orders."""
        # Create multiple active executions
        for i in range(3):
            execution = TwapExecution(
                execution_id=f"test-exec-{i}",
                symbol="BTC/USDT" if i < 2 else "ETH/USDT",
                side=OrderSide.BUY,
                total_quantity=Decimal("10.0"),
                duration_minutes=15,
                slices=[],
                arrival_price=Decimal("50000"),
                remaining_quantity=Decimal("10.0"),
                background_task=MagicMock(),
            )
            execution.background_task.cancel = MagicMock()
            twap_executor.active_executions[f"test-exec-{i}"] = execution

        # Cancel all BTC/USDT orders
        mock_market_executor.cancel_all_orders = AsyncMock(return_value=2)
        cancelled_count = await twap_executor.cancel_all_orders("BTC/USDT")

        assert cancelled_count == 4  # 2 TWAP + 2 market
        assert twap_executor.active_executions["test-exec-0"].status == "CANCELLED"
        assert twap_executor.active_executions["test-exec-1"].status == "CANCELLED"
        assert (
            twap_executor.active_executions["test-exec-2"].status == "ACTIVE"
        )  # Different symbol

    @pytest.mark.asyncio
    async def test_execute_market_order(self, twap_executor, sample_order):
        """Test that execute_market_order routes to TWAP."""
        with patch.object(twap_executor, "execute_twap") as mock_execute:
            mock_execute.return_value = ExecutionResult(
                success=True, order=sample_order, message="TWAP completed"
            )

            result = await twap_executor.execute_market_order(sample_order)

            mock_execute.assert_called_once_with(sample_order, 15)  # Default duration
            assert result.success is True
