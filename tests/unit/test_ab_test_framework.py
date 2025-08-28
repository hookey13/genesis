"""Unit tests for A/B Testing Framework"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from genesis.analytics.strategy_performance import StrategyPerformanceTracker
from genesis.core.events import EventType
from genesis.engine.ab_test_framework import (
    ABTest,
    ABTestFramework,
    AllocationMethod,
    TestStatus,
    TestVariant,
)
from genesis.engine.event_bus import EventBus


@pytest.fixture
def mock_event_bus():
    """Create mock event bus"""
    event_bus = Mock(spec=EventBus)
    event_bus.publish = AsyncMock()
    return event_bus


@pytest.fixture
def mock_performance_tracker():
    """Create mock performance tracker"""
    return Mock(spec=StrategyPerformanceTracker)


@pytest.fixture
def framework(mock_event_bus, mock_performance_tracker, tmp_path):
    """Create test framework instance"""
    return ABTestFramework(
        event_bus=mock_event_bus,
        performance_tracker=mock_performance_tracker,
        storage_path=str(tmp_path / "ab_tests"),
    )


@pytest.fixture
def variant_a():
    """Create test variant A"""
    return TestVariant(
        variant_id="variant_a",
        strategy_name="strategy_a",
        strategy_params={"param1": "value1"},
        weight=Decimal("0.5"),
    )


@pytest.fixture
def variant_b():
    """Create test variant B"""
    return TestVariant(
        variant_id="variant_b",
        strategy_name="strategy_b",
        strategy_params={"param2": "value2"},
        weight=Decimal("0.5"),
    )


class TestABTestFramework:
    """Test A/B testing framework functionality"""

    @pytest.mark.asyncio
    async def test_create_test(self, framework, variant_a, variant_b, mock_event_bus):
        """Test creating a new A/B test"""
        test = await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
            min_trades=50,
            confidence_level=Decimal("0.95"),
            allocation_method=AllocationMethod.RANDOM,
        )

        assert test.test_id == "test_1"
        assert test.name == "Test 1"
        assert test.variant_a == variant_a
        assert test.variant_b == variant_b
        assert test.min_trades_per_variant == 50
        assert test.confidence_level == Decimal("0.95")
        assert test.status == TestStatus.PENDING
        assert "test_1" in framework.active_tests

        # Verify event published
        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.event_type == EventType.AB_TEST_CREATED

    @pytest.mark.asyncio
    async def test_create_duplicate_test(self, framework, variant_a, variant_b):
        """Test creating duplicate test raises error"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )

        with pytest.raises(ValueError, match="already exists"):
            await framework.create_test(
                test_id="test_1",
                name="Test 1",
                description="Test description",
                variant_a=variant_a,
                variant_b=variant_b,
            )

    @pytest.mark.asyncio
    async def test_start_test(self, framework, variant_a, variant_b, mock_event_bus):
        """Test starting an A/B test"""
        test = await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )

        await framework.start_test("test_1")

        assert test.status == TestStatus.RUNNING
        assert test.start_time is not None

        # Verify event published
        assert mock_event_bus.publish.call_count == 2  # create + start
        event = mock_event_bus.publish.call_args[0][0]
        assert event.event_type == EventType.AB_TEST_STARTED

    @pytest.mark.asyncio
    async def test_start_nonexistent_test(self, framework):
        """Test starting nonexistent test raises error"""
        with pytest.raises(ValueError, match="not found"):
            await framework.start_test("nonexistent")

    @pytest.mark.asyncio
    async def test_start_already_running_test(self, framework, variant_a, variant_b):
        """Test starting already running test raises error"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )
        await framework.start_test("test_1")

        with pytest.raises(ValueError, match="not in PENDING status"):
            await framework.start_test("test_1")

    @pytest.mark.asyncio
    async def test_allocate_variant_random(self, framework, variant_a, variant_b):
        """Test random variant allocation"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
            allocation_method=AllocationMethod.RANDOM,
        )
        await framework.start_test("test_1")

        # Test allocation distribution
        allocations = []
        with patch("numpy.random.random") as mock_random:
            # Test variant A allocation
            mock_random.return_value = 0.3
            allocations.append(framework.allocate_variant("test_1"))

            # Test variant B allocation
            mock_random.return_value = 0.7
            allocations.append(framework.allocate_variant("test_1"))

        assert allocations[0] == "variant_a"
        assert allocations[1] == "variant_b"

    @pytest.mark.asyncio
    async def test_allocate_variant_round_robin(self, framework, variant_a, variant_b):
        """Test round-robin variant allocation"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
            allocation_method=AllocationMethod.ROUND_ROBIN,
        )
        await framework.start_test("test_1")

        # Test alternating allocation
        allocations = []
        for _ in range(4):
            allocations.append(framework.allocate_variant("test_1"))

        assert allocations == ["variant_a", "variant_b", "variant_a", "variant_b"]

    @pytest.mark.asyncio
    async def test_allocate_variant_weighted(self, framework):
        """Test weighted variant allocation"""
        variant_a = TestVariant(
            variant_id="variant_a",
            strategy_name="strategy_a",
            strategy_params={},
            weight=Decimal("0.7"),
        )
        variant_b = TestVariant(
            variant_id="variant_b",
            strategy_name="strategy_b",
            strategy_params={},
            weight=Decimal("0.3"),
        )

        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
            allocation_method=AllocationMethod.WEIGHTED,
        )
        await framework.start_test("test_1")

        with patch("numpy.random.random") as mock_random:
            # Test weighted allocation
            mock_random.return_value = 0.6  # Should allocate to variant A (0.6 < 0.7)
            assert framework.allocate_variant("test_1") == "variant_a"

            mock_random.return_value = 0.8  # Should allocate to variant B (0.8 > 0.7)
            assert framework.allocate_variant("test_1") == "variant_b"

    @pytest.mark.asyncio
    async def test_allocate_variant_not_running(self, framework, variant_a, variant_b):
        """Test allocating variant for non-running test raises error"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )

        with pytest.raises(ValueError, match="not running"):
            framework.allocate_variant("test_1")

    @pytest.mark.asyncio
    async def test_record_trade_result(self, framework, variant_a, variant_b):
        """Test recording trade results"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
            min_trades=2,
        )
        await framework.start_test("test_1")

        # Record trades for variant A
        await framework.record_trade_result(
            "test_1", "variant_a", Decimal("100"), datetime.now(UTC)
        )
        await framework.record_trade_result(
            "test_1", "variant_a", Decimal("-50"), datetime.now(UTC)
        )

        test = framework.active_tests.get("test_1")
        assert test.variant_a.trades_executed == 2
        assert test.variant_a.total_pnl_usdt == Decimal("50")
        assert test.variant_a.win_rate == Decimal("0.5")
        assert len(test.variant_a.returns) == 2

    @pytest.mark.asyncio
    async def test_record_trade_invalid_variant(self, framework, variant_a, variant_b):
        """Test recording trade for invalid variant raises error"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )
        await framework.start_test("test_1")

        with pytest.raises(ValueError, match="not found"):
            await framework.record_trade_result(
                "test_1", "invalid_variant", Decimal("100"), datetime.now(UTC)
            )

    @pytest.mark.asyncio
    async def test_complete_test(self, framework, variant_a, variant_b, mock_event_bus):
        """Test completing an A/B test"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
            min_trades=2,
        )
        await framework.start_test("test_1")

        # Add sufficient trades
        for _ in range(35):
            await framework.record_trade_result(
                "test_1",
                "variant_a",
                Decimal(str(np.random.normal(10, 5))),
                datetime.now(UTC),
            )
            await framework.record_trade_result(
                "test_1",
                "variant_b",
                Decimal(str(np.random.normal(5, 5))),
                datetime.now(UTC),
            )

        # Complete test
        test = await framework.complete_test("test_1")

        assert test.status == TestStatus.COMPLETED
        assert test.end_time is not None
        assert test.p_value is not None
        assert test.confidence_interval is not None
        assert "test_1" not in framework.active_tests
        assert test in framework.completed_tests

        # Verify event published
        event = mock_event_bus.publish.call_args[0][0]
        assert event.event_type == EventType.AB_TEST_COMPLETED

    @pytest.mark.asyncio
    async def test_abort_test(self, framework, variant_a, variant_b, mock_event_bus):
        """Test aborting an A/B test"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )
        await framework.start_test("test_1")

        await framework.abort_test("test_1", "Test reason")

        assert "test_1" not in framework.active_tests
        assert len(framework.completed_tests) == 1
        assert framework.completed_tests[0].status == TestStatus.ABORTED

        # Verify event published
        event = mock_event_bus.publish.call_args[0][0]
        assert event.event_type == EventType.AB_TEST_ABORTED
        assert event.event_data["reason"] == "Test reason"

    def test_calculate_p_value(self, framework):
        """Test p-value calculation"""
        returns_a = [Decimal("10"), Decimal("15"), Decimal("5")] * 20
        returns_b = [Decimal("5"), Decimal("8"), Decimal("3")] * 20

        p_value = framework._calculate_p_value(returns_a, returns_b)
        assert p_value > Decimal("0")
        assert p_value < Decimal("1")

    def test_calculate_confidence_interval(self, framework):
        """Test confidence interval calculation"""
        returns_a = [Decimal("10"), Decimal("15"), Decimal("5")] * 20
        returns_b = [Decimal("5"), Decimal("8"), Decimal("3")] * 20

        lower, upper = framework._calculate_confidence_interval(
            returns_a, returns_b, Decimal("0.95")
        )

        assert lower < upper
        assert isinstance(lower, Decimal)
        assert isinstance(upper, Decimal)

    def test_calculate_sharpe_ratio(self, framework):
        """Test Sharpe ratio calculation"""
        returns = [Decimal("10"), Decimal("-5"), Decimal("15"), Decimal("0")] * 10

        sharpe = framework._calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, Decimal)
        assert sharpe != Decimal("0")

        # Test empty returns
        assert framework._calculate_sharpe_ratio([]) == Decimal("0")

        # Test zero std returns
        assert framework._calculate_sharpe_ratio([Decimal("10")] * 10) == Decimal("0")

    def test_calculate_max_drawdown(self, framework):
        """Test maximum drawdown calculation"""
        returns = [
            Decimal("100"),
            Decimal("50"),
            Decimal("-75"),
            Decimal("25"),
            Decimal("-100"),
            Decimal("50"),
        ]

        max_dd = framework._calculate_max_drawdown(returns)
        assert isinstance(max_dd, Decimal)
        assert max_dd > Decimal("0")

        # Test empty returns
        assert framework._calculate_max_drawdown([]) == Decimal("0")

        # Test only positive returns
        assert framework._calculate_max_drawdown([Decimal("10")] * 10) == Decimal("0")

    @pytest.mark.asyncio
    async def test_get_test_results(self, framework, variant_a, variant_b):
        """Test retrieving test results"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )

        # Get active test
        test = await framework.get_test_results("test_1")
        assert test is not None
        assert test.test_id == "test_1"

        # Get nonexistent test
        test = await framework.get_test_results("nonexistent")
        assert test is None

    @pytest.mark.asyncio
    async def test_get_all_tests(self, framework, variant_a, variant_b):
        """Test retrieving all tests"""
        await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )
        await framework.create_test(
            test_id="test_2",
            name="Test 2",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )

        all_tests = await framework.get_all_tests()
        assert len(all_tests) == 2
        assert all(isinstance(test, ABTest) for test in all_tests)

    @pytest.mark.asyncio
    async def test_save_and_load_test_results(
        self, framework, variant_a, variant_b, tmp_path
    ):
        """Test saving and loading test results"""
        test = await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )
        test.status = TestStatus.COMPLETED
        test.winner = "variant_a"
        test.p_value = Decimal("0.03")
        test.confidence_interval = (Decimal("0.1"), Decimal("0.5"))

        # Save test
        await framework._save_test_results(test)

        # Verify file created
        filepath = tmp_path / "ab_tests" / "test_1.json"
        assert filepath.exists()

        # Load test
        loaded_test = await framework._load_test_results("test_1")
        assert loaded_test is not None
        assert loaded_test.test_id == "test_1"
        assert loaded_test.winner == "variant_a"
        assert loaded_test.p_value == Decimal("0.03")

    @pytest.mark.asyncio
    async def test_generate_report(self, framework, variant_a, variant_b):
        """Test generating test report"""
        test = await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
        )
        await framework.start_test("test_1")

        # Add some trades
        for _ in range(10):
            await framework.record_trade_result(
                "test_1", "variant_a", Decimal("100"), datetime.now(UTC)
            )

        report = await framework.generate_report("test_1")

        assert "A/B Test Report: Test 1" in report
        assert "Test ID: test_1" in report
        assert "Status: running" in report
        assert "Variant A" in report
        assert "Variant B" in report
        assert "Trades: 10" in report

    @pytest.mark.asyncio
    async def test_check_test_completion_min_trades(
        self, framework, variant_a, variant_b
    ):
        """Test completion check with minimum trades requirement"""
        test = await framework.create_test(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
            min_trades=10,
        )

        # Not enough trades
        test.variant_a.trades_executed = 5
        test.variant_b.trades_executed = 5
        assert not await framework._check_test_completion(test)

        # Enough trades
        test.variant_a.trades_executed = 10
        test.variant_b.trades_executed = 10
        assert not await framework._check_test_completion(
            test
        )  # Still need statistical significance

    @pytest.mark.asyncio
    async def test_check_test_completion_statistical_significance(self, framework):
        """Test completion check with statistical significance"""
        variant_a = TestVariant(
            variant_id="variant_a",
            strategy_name="strategy_a",
            strategy_params={},
            returns=[Decimal("10")] * 35,
        )
        variant_b = TestVariant(
            variant_id="variant_b",
            strategy_name="strategy_b",
            strategy_params={},
            returns=[Decimal("5")] * 35,
        )

        test = ABTest(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=variant_b,
            min_trades_per_variant=10,
            confidence_level=Decimal("0.95"),
        )
        test.variant_a.trades_executed = 35
        test.variant_b.trades_executed = 35

        # Should complete with statistical significance
        with patch.object(
            framework, "_calculate_p_value", return_value=Decimal("0.01")
        ):
            assert await framework._check_test_completion(test)

    def test_test_variant_to_dict(self, variant_a):
        """Test TestVariant conversion in ABTest.to_dict"""
        test = ABTest(
            test_id="test_1",
            name="Test 1",
            description="Test description",
            variant_a=variant_a,
            variant_b=TestVariant(
                variant_id="variant_b",
                strategy_name="strategy_b",
                strategy_params={"param": "value"},
            ),
        )

        data = test.to_dict()
        assert data["test_id"] == "test_1"
        assert data["variant_a"]["variant_id"] == "variant_a"
        assert data["variant_b"]["variant_id"] == "variant_b"
        assert data["status"] == "pending"
        assert data["allocation_method"] == "random"
