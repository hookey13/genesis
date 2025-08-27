"""Integration tests for TWAP execution workflow."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from genesis.analytics.twap_analyzer import TwapAnalyzer
from genesis.core.events import EventType
from genesis.core.models import Account, TradingTier
from genesis.data.market_data_service import MarketDataService, VolumeProfile
from genesis.data.repository import Repository
from genesis.engine.event_bus import EventBus
from genesis.engine.executor.base import (
    ExecutionResult,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.engine.executor.twap import TwapExecutor
from genesis.engine.risk_engine import RiskDecision, RiskEngine
from genesis.exchange.gateway import BinanceGateway
from genesis.ui.widgets.twap_progress import TwapProgressWidget


class TestTwapWorkflow:
    """Test complete TWAP execution workflow."""

    @pytest.fixture
    async def setup_components(self):
        """Set up all components for TWAP workflow."""
        # Mock gateway
        gateway = AsyncMock(spec=BinanceGateway)
        gateway.get_ticker = AsyncMock(return_value=MagicMock(
            last_price=Decimal("50000"),
            volume=Decimal("10000")
        ))
        gateway.get_order_book = AsyncMock()

        # Account with Strategist tier
        account = Account(
            account_id="test-account",
            tier=TradingTier.STRATEGIST,
            balance_usdt=Decimal("100000")
        )

        # Mock market executor
        market_executor = AsyncMock(spec=MarketOrderExecutor)
        market_executor.generate_client_order_id = MagicMock(return_value=str(uuid4()))

        # Mock repository
        repository = AsyncMock(spec=Repository)
        repository.save_twap_execution = AsyncMock()
        repository.save_twap_slice = AsyncMock()
        repository.update_twap_execution = AsyncMock()
        repository.get_twap_execution = AsyncMock()
        repository.get_twap_slices = AsyncMock()
        repository.save_twap_analysis = AsyncMock()

        # Mock market data service
        market_data_service = AsyncMock(spec=MarketDataService)
        market_data_service.get_current_price = AsyncMock(return_value=Decimal("50000"))
        market_data_service.is_volume_anomaly = AsyncMock(return_value=False)

        volume_profile = MagicMock(spec=VolumeProfile)
        volume_profile.get_hourly_volumes = MagicMock(return_value={
            i: Decimal("100") for i in range(24)
        })
        market_data_service.get_volume_profile = AsyncMock(return_value=volume_profile)

        # Mock risk engine
        risk_engine = AsyncMock(spec=RiskEngine)
        risk_engine.check_risk_limits = AsyncMock(return_value=RiskDecision(
            approved=True,
            reason=None
        ))

        # Event bus
        event_bus = AsyncMock(spec=EventBus)
        event_bus.publish = AsyncMock()

        # Create TWAP executor
        twap_executor = TwapExecutor(
            gateway=gateway,
            account=account,
            market_executor=market_executor,
            repository=repository,
            market_data_service=market_data_service,
            risk_engine=risk_engine,
            event_bus=event_bus
        )

        # Create analyzer
        analyzer = TwapAnalyzer(repository=repository)

        # Create progress widget
        progress_widget = TwapProgressWidget()

        return {
            'gateway': gateway,
            'account': account,
            'market_executor': market_executor,
            'repository': repository,
            'market_data_service': market_data_service,
            'risk_engine': risk_engine,
            'event_bus': event_bus,
            'twap_executor': twap_executor,
            'analyzer': analyzer,
            'progress_widget': progress_widget
        }

    @pytest.mark.asyncio
    async def test_complete_twap_execution(self, setup_components):
        """Test complete TWAP execution from start to finish."""
        components = await setup_components
        twap_executor = components['twap_executor']
        market_executor = components['market_executor']
        event_bus = components['event_bus']

        # Create order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("1.0")
        )

        # Mock successful slice executions
        slice_results = []
        for i in range(5):
            slice_results.append(ExecutionResult(
                success=True,
                order=MagicMock(filled_quantity=Decimal("0.2")),
                message=f"Slice {i+1} executed",
                actual_price=Decimal(str(50000 + i * 10)),
                slippage_percent=Decimal("0.05")
            ))

        market_executor.execute_market_order = AsyncMock(side_effect=slice_results)

        # Execute TWAP with short duration for testing
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
            result = await twap_executor.execute_twap(order, duration_minutes=5)

        # Verify execution completed
        assert result.success is True
        assert order.filled_quantity == Decimal("1.0")
        assert order.status == OrderStatus.FILLED

        # Verify events published
        assert event_bus.publish.call_count >= 2  # Start and complete events
        start_event = event_bus.publish.call_args_list[0][0][0]
        assert start_event.type == EventType.ORDER_PLACED
        assert start_event.data['type'] == 'TWAP_STARTED'

    @pytest.mark.asyncio
    async def test_twap_with_pause_resume(self, setup_components):
        """Test TWAP execution with pause and resume."""
        components = await setup_components
        twap_executor = components['twap_executor']
        market_executor = components['market_executor']

        # Create order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("1.0")
        )

        # Set up slower slice execution
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(0.1)
            return ExecutionResult(
                success=True,
                order=MagicMock(filled_quantity=Decimal("0.1")),
                message="Slice executed",
                actual_price=Decimal("50000")
            )

        market_executor.execute_market_order = slow_execution

        # Start TWAP execution
        task = asyncio.create_task(twap_executor.execute_twap(order, duration_minutes=5))

        # Give it time to start
        await asyncio.sleep(0.05)

        # Get execution ID and pause
        execution_id = list(twap_executor.active_executions.keys())[0]
        await twap_executor.pause(execution_id)

        # Verify paused
        assert twap_executor.active_executions[execution_id].status == "PAUSED"

        # Resume
        await twap_executor.resume(execution_id)
        assert twap_executor.active_executions[execution_id].status == "ACTIVE"

        # Cancel to clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_twap_early_completion(self, setup_components):
        """Test TWAP with early completion on favorable price."""
        components = await setup_components
        twap_executor = components['twap_executor']
        market_executor = components['market_executor']
        market_data_service = components['market_data_service']

        # Create buy order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("1.0")
        )

        # Mock price drop for early completion
        market_data_service.get_current_price = AsyncMock(
            side_effect=[
                Decimal("50000"),  # Arrival price
                Decimal("49800"),  # Favorable price (0.4% better)
                Decimal("49800"),
            ]
        )

        # Mock successful execution
        market_executor.execute_market_order = AsyncMock(return_value=ExecutionResult(
            success=True,
            order=MagicMock(filled_quantity=Decimal("1.0")),
            message="Order executed",
            actual_price=Decimal("49800")
        ))

        # Execute with mocked sleep
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
            result = await twap_executor.execute_twap(order, duration_minutes=10)

        # Should complete early
        assert result.success is True
        # Check that not all slices were executed (early completion)
        assert market_executor.execute_market_order.call_count < 10

    @pytest.mark.asyncio
    async def test_twap_with_risk_rejection(self, setup_components):
        """Test TWAP when risk engine rejects some slices."""
        components = await setup_components
        twap_executor = components['twap_executor']
        risk_engine = components['risk_engine']
        market_executor = components['market_executor']

        # Create order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("1.0")
        )

        # Mock risk rejection for some slices
        risk_decisions = [
            RiskDecision(approved=True, reason=None),
            RiskDecision(approved=False, reason="Position limit"),
            RiskDecision(approved=True, reason=None),
            RiskDecision(approved=False, reason="Daily loss limit"),
            RiskDecision(approved=True, reason=None),
        ]
        risk_engine.check_risk_limits = AsyncMock(side_effect=risk_decisions)

        # Mock successful execution for approved slices
        market_executor.execute_market_order = AsyncMock(return_value=ExecutionResult(
            success=True,
            order=MagicMock(filled_quantity=Decimal("0.33")),
            message="Slice executed",
            actual_price=Decimal("50000")
        ))

        # Execute with mocked sleep
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
            result = await twap_executor.execute_twap(order, duration_minutes=5)

        # Should have partial fill (3 out of 5 slices approved)
        assert order.status == OrderStatus.PARTIAL
        assert market_executor.execute_market_order.call_count == 3

    @pytest.mark.asyncio
    async def test_twap_with_volume_anomaly(self, setup_components):
        """Test TWAP execution during volume anomaly."""
        components = await setup_components
        twap_executor = components['twap_executor']
        market_data_service = components['market_data_service']
        market_executor = components['market_executor']
        gateway = components['gateway']

        # Create order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("1.0")
        )

        # Mock volume anomaly detection
        market_data_service.is_volume_anomaly = AsyncMock(return_value=True)

        # Mock lower volume during anomaly
        gateway.get_ticker = AsyncMock(return_value=MagicMock(
            last_price=Decimal("50000"),
            volume=Decimal("5000")  # Lower volume
        ))

        # Mock execution with reduced size
        execution_count = 0

        async def execute_with_count(*args, **kwargs):
            nonlocal execution_count
            execution_count += 1
            # Reduced quantity due to participation limit
            return ExecutionResult(
                success=True,
                order=MagicMock(filled_quantity=Decimal("0.05")),  # Smaller slices
                message="Slice executed",
                actual_price=Decimal("50000")
            )

        market_executor.execute_market_order = execute_with_count

        # Execute with mocked sleep
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
            result = await twap_executor.execute_twap(order, duration_minutes=5)

        # Should have more slices due to reduced participation
        assert execution_count > 5

    @pytest.mark.asyncio
    async def test_twap_analysis_workflow(self, setup_components):
        """Test complete workflow including post-trade analysis."""
        components = await setup_components
        twap_executor = components['twap_executor']
        analyzer = components['analyzer']
        repository = components['repository']
        market_executor = components['market_executor']

        # Create and execute order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("1.0")
        )

        # Mock successful execution
        market_executor.execute_market_order = AsyncMock(return_value=ExecutionResult(
            success=True,
            order=MagicMock(filled_quantity=Decimal("0.2")),
            message="Slice executed",
            actual_price=Decimal("50050"),
            slippage_percent=Decimal("0.1")
        ))

        # Execute TWAP
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
            result = await twap_executor.execute_twap(order, duration_minutes=5)

        assert result.success is True

        # Mock repository data for analysis
        execution_data = {
            'execution_id': 'test-exec',
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'total_quantity': '1.0',
            'executed_quantity': '1.0',
            'duration_minutes': 5,
            'slice_count': 5,
            'arrival_price': '50000',
            'early_completion': False,
            'started_at': datetime.now() - timedelta(minutes=5),
            'completed_at': datetime.now(),
            'remaining_quantity': '0'
        }

        slice_history = [
            {
                'slice_number': i + 1,
                'executed_quantity': '0.2',
                'execution_price': str(50000 + i * 10),
                'market_price': str(50000 + i * 8),
                'slippage_bps': str(5 + i),
                'participation_rate': str(7),
                'volume_at_execution': '100',
                'status': 'EXECUTED',
                'executed_at': datetime.now()
            }
            for i in range(5)
        ]

        repository.get_twap_execution = AsyncMock(return_value=execution_data)
        repository.get_twap_slices = AsyncMock(return_value=slice_history)

        # Generate analysis report
        report = await analyzer.generate_execution_report('test-exec')

        assert report is not None
        assert report.execution_id == 'test-exec'
        assert report.total_quantity == Decimal('1.0')
        assert report.executed_quantity == Decimal('1.0')
        assert report.timing_score >= 0
        assert report.execution_risk_score >= 0
        assert len(report.improvement_opportunities) >= 0

    @pytest.mark.asyncio
    async def test_twap_progress_widget_integration(self, setup_components):
        """Test TWAP progress widget updates during execution."""
        components = await setup_components
        progress_widget = components['progress_widget']

        # Simulate execution data updates
        execution_data = {
            'execution_id': 'test-exec-123',
            'symbol': 'BTC/USDT',
            'side': 'BUY',
            'total_quantity': '10.0',
            'executed_quantity': '0',
            'remaining_quantity': '10.0',
            'slice_count': 10,
            'completed_slices': 0,
            'arrival_price': '50000',
            'current_price': '50000',
            'twap_price': '0',
            'participation_rate': '0',
            'implementation_shortfall': '0',
            'status': 'ACTIVE',
            'started_at': datetime.now()
        }

        # Update widget with initial data
        progress_widget.update_execution(execution_data)
        assert progress_widget.status == 'ACTIVE'
        assert progress_widget.executed_quantity == Decimal('0')

        # Simulate slice executions
        for i in range(5):
            slice_data = {
                'slice_number': i + 1,
                'executed_at': datetime.now().isoformat(),
                'executed_quantity': '2.0',
                'execution_price': str(50000 + i * 10),
                'participation_rate': '7.5',
                'slippage_bps': '10',
                'status': 'EXECUTED'
            }
            progress_widget.add_slice_execution(slice_data)

        # Verify updates
        assert progress_widget.executed_quantity == Decimal('10.0')
        assert progress_widget.remaining_quantity == Decimal('0')
        assert progress_widget.completed_slices == 5
        assert len(progress_widget.slice_history) == 5
        assert progress_widget.twap_price > 0

        # Complete execution
        final_metrics = {
            'twap_price': '50020',
            'implementation_shortfall': '0.04'
        }
        progress_widget.complete_execution(final_metrics)

        assert progress_widget.status == 'COMPLETED'
        assert progress_widget.twap_price == Decimal('50020')
        assert progress_widget.implementation_shortfall == Decimal('0.04')

    @pytest.mark.asyncio
    async def test_twap_cancellation(self, setup_components):
        """Test TWAP order cancellation."""
        components = await setup_components
        twap_executor = components['twap_executor']
        market_executor = components['market_executor']

        # Create order
        order = Order(
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            client_order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side=OrderSide.BUY,
            price=None,
            quantity=Decimal("1.0")
        )

        # Set up slow execution
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(0.1)
            return ExecutionResult(
                success=True,
                order=MagicMock(filled_quantity=Decimal("0.1")),
                message="Slice executed",
                actual_price=Decimal("50000")
            )

        market_executor.execute_market_order = slow_execution

        # Start TWAP execution
        task = asyncio.create_task(twap_executor.execute_twap(order, duration_minutes=5))

        # Give it time to start
        await asyncio.sleep(0.05)

        # Cancel all orders
        cancelled = await twap_executor.cancel_all_orders("BTC/USDT")
        assert cancelled > 0

        # Verify execution marked as cancelled
        for execution in twap_executor.active_executions.values():
            if execution.symbol == "BTC/USDT":
                assert execution.status == "CANCELLED"

        # Clean up task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_multiple_concurrent_twap(self, setup_components):
        """Test multiple concurrent TWAP executions."""
        components = await setup_components
        twap_executor = components['twap_executor']
        market_executor = components['market_executor']

        # Create multiple orders
        orders = []
        for i in range(3):
            order = Order(
                order_id=str(uuid4()),
                position_id=str(uuid4()),
                client_order_id=str(uuid4()),
                symbol=f"{'BTC' if i < 2 else 'ETH'}/USDT",
                type=OrderType.MARKET,
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                price=None,
                quantity=Decimal(str(i + 1))
            )
            orders.append(order)

        # Mock execution
        market_executor.execute_market_order = AsyncMock(return_value=ExecutionResult(
            success=True,
            order=MagicMock(filled_quantity=Decimal("0.5")),
            message="Slice executed",
            actual_price=Decimal("50000")
        ))

        # Start concurrent executions
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock):
            tasks = [
                asyncio.create_task(twap_executor.execute_twap(order, duration_minutes=5))
                for order in orders
            ]

            # Wait for all to complete
            results = await asyncio.gather(*tasks)

        # Verify all completed
        assert all(r.success for r in results)
        assert len(twap_executor.active_executions) == 0  # All cleaned up
