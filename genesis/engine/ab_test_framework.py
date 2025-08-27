"""A/B Testing Framework for Strategy Variations

This module implements a comprehensive A/B testing framework for comparing
strategy variations and determining statistical significance.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

import numpy as np
import structlog
from scipy import stats

from genesis.analytics.strategy_performance import StrategyPerformanceTracker
from genesis.core.events import Event, EventType
from genesis.engine.event_bus import EventBus

logger = structlog.get_logger(__name__)


class TestStatus(Enum):
    """Status of an A/B test"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"


class AllocationMethod(Enum):
    """Method for allocating traffic to variants"""
    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"


@dataclass
class TestVariant:
    """Represents a single variant in an A/B test"""
    variant_id: str
    strategy_name: str
    strategy_params: dict
    weight: Decimal = Decimal("0.5")

    # Performance metrics
    trades_executed: int = 0
    total_pnl_usdt: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")

    # Statistical metrics
    returns: list[Decimal] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)


@dataclass
class ABTest:
    """Represents an A/B test configuration and results"""
    test_id: str
    name: str
    description: str
    variant_a: TestVariant
    variant_b: TestVariant

    # Test configuration
    min_trades_per_variant: int = 100
    confidence_level: Decimal = Decimal("0.95")
    allocation_method: AllocationMethod = AllocationMethod.RANDOM

    # Test state
    status: TestStatus = TestStatus.PENDING
    start_time: datetime | None = None
    end_time: datetime | None = None

    # Results
    winner: str | None = None
    p_value: Decimal | None = None
    confidence_interval: tuple[Decimal, Decimal] | None = None
    statistical_significance: bool = False

    def to_dict(self) -> dict:
        """Convert test to dictionary for storage"""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "variant_a": {
                "variant_id": self.variant_a.variant_id,
                "strategy_name": self.variant_a.strategy_name,
                "strategy_params": self.variant_a.strategy_params,
                "weight": str(self.variant_a.weight),
                "trades_executed": self.variant_a.trades_executed,
                "total_pnl_usdt": str(self.variant_a.total_pnl_usdt),
                "win_rate": str(self.variant_a.win_rate),
                "sharpe_ratio": str(self.variant_a.sharpe_ratio),
                "max_drawdown": str(self.variant_a.max_drawdown)
            },
            "variant_b": {
                "variant_id": self.variant_b.variant_id,
                "strategy_name": self.variant_b.strategy_name,
                "strategy_params": self.variant_b.strategy_params,
                "weight": str(self.variant_b.weight),
                "trades_executed": self.variant_b.trades_executed,
                "total_pnl_usdt": str(self.variant_b.total_pnl_usdt),
                "win_rate": str(self.variant_b.win_rate),
                "sharpe_ratio": str(self.variant_b.sharpe_ratio),
                "max_drawdown": str(self.variant_b.max_drawdown)
            },
            "min_trades_per_variant": self.min_trades_per_variant,
            "confidence_level": str(self.confidence_level),
            "allocation_method": self.allocation_method.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "winner": self.winner,
            "p_value": str(self.p_value) if self.p_value else None,
            "confidence_interval": [str(ci) for ci in self.confidence_interval] if self.confidence_interval else None,
            "statistical_significance": self.statistical_significance
        }


class ABTestFramework:
    """Framework for running A/B tests on strategy variations"""

    def __init__(self,
                 event_bus: EventBus,
                 performance_tracker: StrategyPerformanceTracker,
                 storage_path: str = ".genesis/data/ab_tests"):
        """Initialize A/B test framework

        Args:
            event_bus: Event bus for publishing test events
            performance_tracker: Strategy performance tracking
            storage_path: Path for storing test results
        """
        self.event_bus = event_bus
        self.performance_tracker = performance_tracker
        self.storage_path = storage_path

        self.active_tests: dict[str, ABTest] = {}
        self.completed_tests: list[ABTest] = []

        # Allocation state
        self._round_robin_counters: dict[str, int] = {}

        logger.info("A/B test framework initialized", storage_path=storage_path)

    async def create_test(self,
                         test_id: str,
                         name: str,
                         description: str,
                         variant_a: TestVariant,
                         variant_b: TestVariant,
                         min_trades: int = 100,
                         confidence_level: Decimal = Decimal("0.95"),
                         allocation_method: AllocationMethod = AllocationMethod.RANDOM) -> ABTest:
        """Create a new A/B test

        Args:
            test_id: Unique test identifier
            name: Test name
            description: Test description
            variant_a: First variant configuration
            variant_b: Second variant configuration
            min_trades: Minimum trades per variant
            confidence_level: Required confidence level
            allocation_method: Traffic allocation method

        Returns:
            Created A/B test
        """
        if test_id in self.active_tests:
            raise ValueError(f"Test {test_id} already exists")

        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            variant_a=variant_a,
            variant_b=variant_b,
            min_trades_per_variant=min_trades,
            confidence_level=confidence_level,
            allocation_method=allocation_method
        )

        self.active_tests[test_id] = test

        # Initialize round-robin counter if needed
        if allocation_method == AllocationMethod.ROUND_ROBIN:
            self._round_robin_counters[test_id] = 0

        logger.info("A/B test created",
                   test_id=test_id,
                   name=name,
                   variant_a=variant_a.variant_id,
                   variant_b=variant_b.variant_id)

        # Publish test creation event
        await self.event_bus.publish(Event(
            event_type=EventType.AB_TEST_CREATED,
            event_data={
                "test_id": test_id,
                "name": name,
                "variants": [variant_a.variant_id, variant_b.variant_id]
            }
        ))

        return test

    async def start_test(self, test_id: str) -> None:
        """Start an A/B test

        Args:
            test_id: Test identifier
        """
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        if test.status != TestStatus.PENDING:
            raise ValueError(f"Test {test_id} is not in PENDING status")

        test.status = TestStatus.RUNNING
        test.start_time = datetime.now(UTC)

        logger.info("A/B test started", test_id=test_id, start_time=test.start_time)

        # Publish test start event
        await self.event_bus.publish(Event(
            event_type=EventType.AB_TEST_STARTED,
            event_data={
                "test_id": test_id,
                "start_time": test.start_time.isoformat()
            }
        ))

    def allocate_variant(self, test_id: str) -> str:
        """Allocate traffic to a variant based on test configuration

        Args:
            test_id: Test identifier

        Returns:
            Variant ID to use
        """
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        if test.status != TestStatus.RUNNING:
            raise ValueError(f"Test {test_id} is not running")

        if test.allocation_method == AllocationMethod.RANDOM:
            # Random allocation based on weights
            rand = np.random.random()
            if rand < float(test.variant_a.weight):
                return test.variant_a.variant_id
            else:
                return test.variant_b.variant_id

        elif test.allocation_method == AllocationMethod.ROUND_ROBIN:
            # Alternate between variants
            counter = self._round_robin_counters.get(test_id, 0)
            variant = test.variant_a.variant_id if counter % 2 == 0 else test.variant_b.variant_id
            self._round_robin_counters[test_id] = counter + 1
            return variant

        elif test.allocation_method == AllocationMethod.WEIGHTED:
            # Weighted random allocation
            total_weight = test.variant_a.weight + test.variant_b.weight
            rand = Decimal(str(np.random.random())) * total_weight
            if rand < test.variant_a.weight:
                return test.variant_a.variant_id
            else:
                return test.variant_b.variant_id

        else:
            raise ValueError(f"Unknown allocation method: {test.allocation_method}")

    async def record_trade_result(self,
                                 test_id: str,
                                 variant_id: str,
                                 pnl_usdt: Decimal,
                                 timestamp: datetime) -> None:
        """Record a trade result for a variant

        Args:
            test_id: Test identifier
            variant_id: Variant identifier
            pnl_usdt: Trade P&L in USDT
            timestamp: Trade timestamp
        """
        test = self.active_tests.get(test_id)
        if not test:
            # Check if test was already completed
            for completed in self.completed_tests:
                if completed.test_id == test_id:
                    logger.debug(f"Test {test_id} already completed, ignoring trade result")
                    return
            raise ValueError(f"Test {test_id} not found")

        # Find the variant
        variant = test.variant_a if test.variant_a.variant_id == variant_id else test.variant_b
        if variant.variant_id != variant_id:
            raise ValueError(f"Variant {variant_id} not found in test {test_id}")

        # Update variant metrics
        variant.trades_executed += 1
        variant.total_pnl_usdt += pnl_usdt
        variant.returns.append(pnl_usdt)
        variant.timestamps.append(timestamp)

        # Update win rate
        wins = sum(1 for r in variant.returns if r > 0)
        variant.win_rate = Decimal(str(wins)) / Decimal(str(len(variant.returns)))

        logger.debug("Trade result recorded",
                    test_id=test_id,
                    variant_id=variant_id,
                    trade_count=variant.trades_executed,
                    pnl_usdt=pnl_usdt)

        # Check if test is complete
        if await self._check_test_completion(test):
            await self.complete_test(test_id)

    async def _check_test_completion(self, test: ABTest) -> bool:
        """Check if test has enough data for completion

        Args:
            test: A/B test to check

        Returns:
            True if test should be completed
        """
        # Check minimum trades requirement
        if test.variant_a.trades_executed < test.min_trades_per_variant:
            return False
        if test.variant_b.trades_executed < test.min_trades_per_variant:
            return False

        # Check for statistical significance
        if len(test.variant_a.returns) > 30 and len(test.variant_b.returns) > 30:
            p_value = self._calculate_p_value(test.variant_a.returns, test.variant_b.returns)
            if p_value < (Decimal("1") - test.confidence_level):
                return True

        return False

    def _calculate_p_value(self, returns_a: list[Decimal], returns_b: list[Decimal]) -> Decimal:
        """Calculate p-value using t-test

        Args:
            returns_a: Returns for variant A
            returns_b: Returns for variant B

        Returns:
            P-value from t-test
        """
        # Convert to numpy arrays
        a = np.array([float(r) for r in returns_a])
        b = np.array([float(r) for r in returns_b])

        # Perform t-test
        _, p_value = stats.ttest_ind(a, b)

        return Decimal(str(p_value))

    def _calculate_confidence_interval(self,
                                      returns_a: list[Decimal],
                                      returns_b: list[Decimal],
                                      confidence_level: Decimal) -> tuple[Decimal, Decimal]:
        """Calculate confidence interval for difference in means

        Args:
            returns_a: Returns for variant A
            returns_b: Returns for variant B
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            Confidence interval (lower, upper)
        """
        # Convert to numpy arrays
        a = np.array([float(r) for r in returns_a])
        b = np.array([float(r) for r in returns_b])

        # Calculate difference in means
        diff_mean = np.mean(a) - np.mean(b)

        # Calculate standard error
        se_a = np.std(a, ddof=1) / np.sqrt(len(a))
        se_b = np.std(b, ddof=1) / np.sqrt(len(b))
        se_diff = np.sqrt(se_a**2 + se_b**2)

        # Calculate confidence interval
        alpha = 1 - float(confidence_level)
        t_critical = stats.t.ppf(1 - alpha/2, len(a) + len(b) - 2)
        margin_error = t_critical * se_diff

        lower = Decimal(str(diff_mean - margin_error))
        upper = Decimal(str(diff_mean + margin_error))

        return (lower, upper)

    def _calculate_sharpe_ratio(self, returns: list[Decimal]) -> Decimal:
        """Calculate Sharpe ratio for returns

        Args:
            returns: List of returns

        Returns:
            Sharpe ratio
        """
        if not returns:
            return Decimal("0")

        returns_array = np.array([float(r) for r in returns])

        # Assume risk-free rate of 0 for simplicity
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)

        if std_return == 0:
            return Decimal("0")

        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
        return Decimal(str(sharpe))

    def _calculate_max_drawdown(self, returns: list[Decimal]) -> Decimal:
        """Calculate maximum drawdown

        Args:
            returns: List of returns

        Returns:
            Maximum drawdown percentage
        """
        if not returns:
            return Decimal("0")

        # Calculate cumulative returns
        cumulative = []
        total = Decimal("0")
        for r in returns:
            total += r
            cumulative.append(total)

        # Calculate drawdowns
        peak = cumulative[0]
        max_dd = Decimal("0")

        for value in cumulative[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak if peak != 0 else Decimal("0")
                max_dd = max(max_dd, dd)

        return max_dd * Decimal("100")  # Return as percentage

    async def complete_test(self, test_id: str) -> ABTest:
        """Complete an A/B test and calculate results

        Args:
            test_id: Test identifier

        Returns:
            Completed test with results
        """
        test = self.active_tests.get(test_id)
        if not test:
            # Check if already completed
            for completed in self.completed_tests:
                if completed.test_id == test_id:
                    logger.debug(f"Test {test_id} already completed")
                    return completed
            raise ValueError(f"Test {test_id} not found")

        test.status = TestStatus.COMPLETED
        test.end_time = datetime.now(UTC)

        # Calculate final metrics
        for variant in [test.variant_a, test.variant_b]:
            variant.sharpe_ratio = self._calculate_sharpe_ratio(variant.returns)
            variant.max_drawdown = self._calculate_max_drawdown(variant.returns)

        # Statistical analysis
        test.p_value = self._calculate_p_value(test.variant_a.returns, test.variant_b.returns)
        test.confidence_interval = self._calculate_confidence_interval(
            test.variant_a.returns,
            test.variant_b.returns,
            test.confidence_level
        )

        # Determine winner
        test.statistical_significance = test.p_value < (Decimal("1") - test.confidence_level)

        if test.statistical_significance:
            # Compare average returns
            avg_a = sum(test.variant_a.returns) / len(test.variant_a.returns) if test.variant_a.returns else Decimal("0")
            avg_b = sum(test.variant_b.returns) / len(test.variant_b.returns) if test.variant_b.returns else Decimal("0")

            if avg_a > avg_b:
                test.winner = test.variant_a.variant_id
            else:
                test.winner = test.variant_b.variant_id

        # Move to completed tests
        self.completed_tests.append(test)
        del self.active_tests[test_id]

        # Save results
        await self._save_test_results(test)

        logger.info("A/B test completed",
                   test_id=test_id,
                   winner=test.winner,
                   p_value=test.p_value,
                   statistical_significance=test.statistical_significance)

        # Publish test completion event
        await self.event_bus.publish(Event(
            event_type=EventType.AB_TEST_COMPLETED,
            event_data={
                "test_id": test_id,
                "winner": test.winner,
                "p_value": str(test.p_value),
                "statistical_significance": test.statistical_significance
            }
        ))

        return test

    async def abort_test(self, test_id: str, reason: str = "") -> None:
        """Abort an A/B test

        Args:
            test_id: Test identifier
            reason: Reason for aborting
        """
        test = self.active_tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        test.status = TestStatus.ABORTED
        test.end_time = datetime.now(UTC)

        # Move to completed tests
        self.completed_tests.append(test)
        del self.active_tests[test_id]

        # Save results
        await self._save_test_results(test)

        logger.info("A/B test aborted", test_id=test_id, reason=reason)

        # Publish test abortion event
        await self.event_bus.publish(Event(
            event_type=EventType.AB_TEST_ABORTED,
            event_data={
                "test_id": test_id,
                "reason": reason
            }
        ))

    async def get_test_results(self, test_id: str) -> ABTest | None:
        """Get results for a test

        Args:
            test_id: Test identifier

        Returns:
            Test results or None if not found
        """
        # Check active tests
        if test_id in self.active_tests:
            return self.active_tests[test_id]

        # Check completed tests
        for test in self.completed_tests:
            if test.test_id == test_id:
                return test

        # Try loading from storage
        return await self._load_test_results(test_id)

    async def get_all_tests(self) -> list[ABTest]:
        """Get all tests (active and completed)

        Returns:
            List of all tests
        """
        all_tests = list(self.active_tests.values()) + self.completed_tests
        return all_tests

    async def _save_test_results(self, test: ABTest) -> None:
        """Save test results to storage

        Args:
            test: Test to save
        """
        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)

            filepath = os.path.join(self.storage_path, f"{test.test_id}.json")
            with open(filepath, 'w') as f:
                json.dump(test.to_dict(), f, indent=2)

            logger.debug("Test results saved", test_id=test.test_id, filepath=filepath)

        except Exception as e:
            logger.error("Failed to save test results", test_id=test.test_id, error=str(e))

    async def _load_test_results(self, test_id: str) -> ABTest | None:
        """Load test results from storage

        Args:
            test_id: Test identifier

        Returns:
            Loaded test or None if not found
        """
        try:
            import os
            filepath = os.path.join(self.storage_path, f"{test_id}.json")

            if not os.path.exists(filepath):
                return None

            with open(filepath) as f:
                data = json.load(f)

            # Reconstruct test from data
            variant_a = TestVariant(
                variant_id=data["variant_a"]["variant_id"],
                strategy_name=data["variant_a"]["strategy_name"],
                strategy_params=data["variant_a"]["strategy_params"],
                weight=Decimal(data["variant_a"]["weight"]),
                trades_executed=data["variant_a"]["trades_executed"],
                total_pnl_usdt=Decimal(data["variant_a"]["total_pnl_usdt"]),
                win_rate=Decimal(data["variant_a"]["win_rate"]),
                sharpe_ratio=Decimal(data["variant_a"]["sharpe_ratio"]),
                max_drawdown=Decimal(data["variant_a"]["max_drawdown"])
            )

            variant_b = TestVariant(
                variant_id=data["variant_b"]["variant_id"],
                strategy_name=data["variant_b"]["strategy_name"],
                strategy_params=data["variant_b"]["strategy_params"],
                weight=Decimal(data["variant_b"]["weight"]),
                trades_executed=data["variant_b"]["trades_executed"],
                total_pnl_usdt=Decimal(data["variant_b"]["total_pnl_usdt"]),
                win_rate=Decimal(data["variant_b"]["win_rate"]),
                sharpe_ratio=Decimal(data["variant_b"]["sharpe_ratio"]),
                max_drawdown=Decimal(data["variant_b"]["max_drawdown"])
            )

            test = ABTest(
                test_id=data["test_id"],
                name=data["name"],
                description=data["description"],
                variant_a=variant_a,
                variant_b=variant_b,
                min_trades_per_variant=data["min_trades_per_variant"],
                confidence_level=Decimal(data["confidence_level"]),
                allocation_method=AllocationMethod(data["allocation_method"]),
                status=TestStatus(data["status"]),
                start_time=datetime.fromisoformat(data["start_time"]) if data["start_time"] else None,
                end_time=datetime.fromisoformat(data["end_time"]) if data["end_time"] else None,
                winner=data["winner"],
                p_value=Decimal(data["p_value"]) if data["p_value"] else None,
                confidence_interval=tuple(Decimal(ci) for ci in data["confidence_interval"]) if data["confidence_interval"] else None,
                statistical_significance=data["statistical_significance"]
            )

            logger.debug("Test results loaded", test_id=test_id, filepath=filepath)
            return test

        except Exception as e:
            logger.error("Failed to load test results", test_id=test_id, error=str(e))
            return None

    async def generate_report(self, test_id: str) -> str:
        """Generate a detailed report for a test

        Args:
            test_id: Test identifier

        Returns:
            Formatted report string
        """
        test = await self.get_test_results(test_id)
        if not test:
            return f"Test {test_id} not found"

        report = []
        report.append(f"# A/B Test Report: {test.name}")
        report.append(f"Test ID: {test.test_id}")
        report.append(f"Description: {test.description}")
        report.append(f"Status: {test.status.value}")
        report.append("")

        if test.start_time:
            report.append(f"Start Time: {test.start_time.isoformat()}")
        if test.end_time:
            report.append(f"End Time: {test.end_time.isoformat()}")
            duration = test.end_time - test.start_time
            report.append(f"Duration: {duration}")
        report.append("")

        report.append("## Variant A")
        report.append(f"- Strategy: {test.variant_a.strategy_name}")
        report.append(f"- Trades: {test.variant_a.trades_executed}")
        report.append(f"- Total P&L: {test.variant_a.total_pnl_usdt} USDT")
        report.append(f"- Win Rate: {test.variant_a.win_rate * 100:.2f}%")
        report.append(f"- Sharpe Ratio: {test.variant_a.sharpe_ratio:.3f}")
        report.append(f"- Max Drawdown: {test.variant_a.max_drawdown:.2f}%")
        report.append("")

        report.append("## Variant B")
        report.append(f"- Strategy: {test.variant_b.strategy_name}")
        report.append(f"- Trades: {test.variant_b.trades_executed}")
        report.append(f"- Total P&L: {test.variant_b.total_pnl_usdt} USDT")
        report.append(f"- Win Rate: {test.variant_b.win_rate * 100:.2f}%")
        report.append(f"- Sharpe Ratio: {test.variant_b.sharpe_ratio:.3f}")
        report.append(f"- Max Drawdown: {test.variant_b.max_drawdown:.2f}%")
        report.append("")

        if test.status == TestStatus.COMPLETED:
            report.append("## Statistical Analysis")
            report.append(f"- P-Value: {test.p_value:.6f}")
            report.append(f"- Confidence Level: {test.confidence_level * 100:.1f}%")
            if test.confidence_interval:
                report.append(f"- Confidence Interval: [{test.confidence_interval[0]:.4f}, {test.confidence_interval[1]:.4f}]")
            report.append(f"- Statistical Significance: {test.statistical_significance}")

            if test.winner:
                report.append("")
                report.append(f"**Winner: {test.winner}**")

        return "\n".join(report)
