"""
Iceberg execution tracking and reporting system for Project GENESIS.

This module provides comprehensive tracking and reporting of iceberg executions,
generating detailed analysis reports for performance optimization.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog

from genesis.analytics.market_impact_monitor import ImpactSeverity
from genesis.data.repository import Repository
from genesis.engine.executor.base import OrderSide, OrderStatus

logger = structlog.get_logger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status states."""
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass
class SliceExecutionRecord:
    """Record of a single slice execution."""
    slice_id: str
    slice_number: int
    total_slices: int
    quantity: Decimal
    value_usdt: Decimal
    expected_price: Decimal | None
    actual_price: Decimal | None
    slippage_percent: Decimal | None
    delay_seconds: float
    status: OrderStatus
    fees_usdt: Decimal | None
    latency_ms: int | None
    submitted_at: datetime
    filled_at: datetime | None
    error_message: str | None = None


@dataclass
class ExecutionProgress:
    """Real-time execution progress tracking."""
    execution_id: str
    symbol: str
    side: OrderSide
    total_quantity: Decimal
    total_value_usdt: Decimal
    slice_count: int

    # Progress metrics
    slices_completed: int
    slices_failed: int
    quantity_filled: Decimal
    value_filled_usdt: Decimal
    completion_percent: Decimal

    # Performance metrics
    average_slippage: Decimal
    cumulative_slippage: Decimal
    max_slice_slippage: Decimal
    total_fees_usdt: Decimal

    # Time metrics
    started_at: datetime
    estimated_completion: datetime | None
    elapsed_seconds: float
    average_slice_time: float

    # Status
    status: ExecutionStatus
    abort_reason: str | None = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionReport:
    """Comprehensive post-execution analysis report."""
    execution_id: str
    symbol: str
    side: OrderSide

    # Order details
    total_quantity_ordered: Decimal
    total_quantity_filled: Decimal
    fill_rate: Decimal
    total_value_usdt: Decimal

    # Slicing details
    slice_count: int
    slices_completed: int
    slices_failed: int
    slice_size_avg: Decimal
    slice_size_std_dev: Decimal

    # Performance metrics
    total_slippage_percent: Decimal
    average_slippage_percent: Decimal
    max_slice_slippage: Decimal
    min_slice_slippage: Decimal
    slippage_std_dev: Decimal

    # Market impact
    total_market_impact: Decimal
    average_impact_per_slice: Decimal
    impact_severity_distribution: dict[ImpactSeverity, int]

    # Cost analysis
    total_fees_usdt: Decimal
    average_fees_per_slice: Decimal
    slippage_cost_usdt: Decimal
    total_cost_usdt: Decimal
    cost_per_unit: Decimal

    # Time analysis
    execution_duration_seconds: float
    average_slice_duration: float
    fastest_slice_seconds: float
    slowest_slice_seconds: float
    average_delay_between_slices: float

    # Quality metrics
    execution_quality_score: Decimal  # 0-100 score
    recommendations: list[str]

    # Metadata
    started_at: datetime
    completed_at: datetime
    status: ExecutionStatus
    abort_reason: str | None
    generated_at: datetime = field(default_factory=datetime.now)


class IcebergReportGenerator:
    """
    Generates comprehensive reports for iceberg executions.
    
    Tracks execution progress in real-time and produces detailed
    post-execution analysis reports for performance optimization.
    """

    def __init__(self, repository: Repository | None = None):
        """
        Initialize the report generator.
        
        Args:
            repository: Data repository for persistence
        """
        self.repository = repository
        self.active_executions: dict[str, ExecutionProgress] = {}
        self.slice_records: dict[str, list[SliceExecutionRecord]] = {}

        logger.info("Iceberg report generator initialized")

    def start_tracking_execution(
        self,
        execution_id: str,
        symbol: str,
        side: OrderSide,
        total_quantity: Decimal,
        total_value_usdt: Decimal,
        slice_count: int
    ) -> ExecutionProgress:
        """
        Start tracking a new iceberg execution.
        
        Args:
            execution_id: Unique execution identifier
            symbol: Trading symbol
            side: Order side
            total_quantity: Total quantity to execute
            total_value_usdt: Total value in USDT
            slice_count: Number of slices
            
        Returns:
            ExecutionProgress tracker
        """
        progress = ExecutionProgress(
            execution_id=execution_id,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            total_value_usdt=total_value_usdt,
            slice_count=slice_count,
            slices_completed=0,
            slices_failed=0,
            quantity_filled=Decimal("0"),
            value_filled_usdt=Decimal("0"),
            completion_percent=Decimal("0"),
            average_slippage=Decimal("0"),
            cumulative_slippage=Decimal("0"),
            max_slice_slippage=Decimal("0"),
            total_fees_usdt=Decimal("0"),
            started_at=datetime.now(),
            estimated_completion=None,
            elapsed_seconds=0,
            average_slice_time=0,
            status=ExecutionStatus.PENDING
        )

        self.active_executions[execution_id] = progress
        self.slice_records[execution_id] = []

        logger.info(
            "Started tracking execution",
            execution_id=execution_id,
            symbol=symbol,
            slice_count=slice_count
        )

        return progress

    def update_slice_execution(
        self,
        execution_id: str,
        slice_record: SliceExecutionRecord
    ) -> ExecutionProgress:
        """
        Update execution progress with slice result.
        
        Args:
            execution_id: Execution to update
            slice_record: Slice execution record
            
        Returns:
            Updated execution progress
        """
        if execution_id not in self.active_executions:
            raise ValueError(f"Execution {execution_id} not being tracked")

        progress = self.active_executions[execution_id]
        self.slice_records[execution_id].append(slice_record)

        # Update progress metrics
        if slice_record.status == OrderStatus.FILLED:
            progress.slices_completed += 1
            progress.quantity_filled += slice_record.quantity
            progress.value_filled_usdt += slice_record.value_usdt

            if slice_record.slippage_percent:
                progress.cumulative_slippage += slice_record.slippage_percent
                progress.max_slice_slippage = max(
                    progress.max_slice_slippage,
                    abs(slice_record.slippage_percent)
                )

            if slice_record.fees_usdt:
                progress.total_fees_usdt += slice_record.fees_usdt
        else:
            progress.slices_failed += 1

        # Calculate averages
        if progress.slices_completed > 0:
            progress.average_slippage = (
                progress.cumulative_slippage / progress.slices_completed
            )

        # Calculate completion percentage
        progress.completion_percent = (
            (progress.slices_completed / progress.slice_count) * Decimal("100")
            if progress.slice_count > 0 else Decimal("0")
        )

        # Update time metrics
        progress.elapsed_seconds = (datetime.now() - progress.started_at).total_seconds()

        if progress.slices_completed > 0:
            progress.average_slice_time = (
                progress.elapsed_seconds / progress.slices_completed
            )

            # Estimate completion time
            remaining_slices = progress.slice_count - progress.slices_completed
            estimated_remaining_seconds = remaining_slices * progress.average_slice_time
            progress.estimated_completion = (
                datetime.now() + timedelta(seconds=estimated_remaining_seconds)
            )

        # Update status
        if progress.slices_completed + progress.slices_failed >= progress.slice_count:
            if progress.slices_failed == 0:
                progress.status = ExecutionStatus.COMPLETED
            elif progress.slices_completed == 0:
                progress.status = ExecutionStatus.FAILED
            else:
                progress.status = ExecutionStatus.COMPLETED  # Partial success
        else:
            progress.status = ExecutionStatus.EXECUTING

        progress.last_updated = datetime.now()

        logger.debug(
            "Updated execution progress",
            execution_id=execution_id,
            completed=progress.slices_completed,
            failed=progress.slices_failed,
            completion_percent=str(progress.completion_percent)
        )

        return progress

    def abort_execution(
        self,
        execution_id: str,
        reason: str
    ) -> None:
        """
        Mark execution as aborted.
        
        Args:
            execution_id: Execution to abort
            reason: Abort reason
        """
        if execution_id in self.active_executions:
            progress = self.active_executions[execution_id]
            progress.status = ExecutionStatus.ABORTED
            progress.abort_reason = reason
            progress.last_updated = datetime.now()

            logger.warning(
                "Execution aborted",
                execution_id=execution_id,
                reason=reason
            )

    async def generate_execution_report(
        self,
        execution_id: str
    ) -> ExecutionReport:
        """
        Generate comprehensive execution report.
        
        Args:
            execution_id: Execution to report on
            
        Returns:
            Detailed execution report
        """
        if execution_id not in self.active_executions:
            raise ValueError(f"Execution {execution_id} not found")

        progress = self.active_executions[execution_id]
        slices = self.slice_records[execution_id]

        # Calculate fill rate
        fill_rate = (
            progress.quantity_filled / progress.total_quantity
            if progress.total_quantity > 0 else Decimal("0")
        )

        # Calculate slice size statistics
        slice_sizes = [s.quantity for s in slices if s.status == OrderStatus.FILLED]
        if slice_sizes:
            avg_size = sum(slice_sizes) / len(slice_sizes)
            if len(slice_sizes) > 1:
                variance = sum((s - avg_size) ** 2 for s in slice_sizes) / len(slice_sizes)
                std_dev = variance.sqrt()
            else:
                std_dev = Decimal("0")
        else:
            avg_size = Decimal("0")
            std_dev = Decimal("0")

        # Calculate slippage statistics
        slippages = [
            s.slippage_percent for s in slices
            if s.slippage_percent is not None and s.status == OrderStatus.FILLED
        ]

        if slippages:
            total_slippage = sum(slippages)
            avg_slippage = total_slippage / len(slippages)
            max_slippage = max(slippages, key=abs)
            min_slippage = min(slippages, key=abs)

            if len(slippages) > 1:
                slip_variance = sum((s - avg_slippage) ** 2 for s in slippages) / len(slippages)
                slip_std_dev = slip_variance.sqrt()
            else:
                slip_std_dev = Decimal("0")
        else:
            total_slippage = avg_slippage = max_slippage = min_slippage = slip_std_dev = Decimal("0")

        # Calculate time statistics
        slice_times = []
        for i, slice_rec in enumerate(slices):
            if slice_rec.filled_at and slice_rec.submitted_at:
                duration = (slice_rec.filled_at - slice_rec.submitted_at).total_seconds()
                slice_times.append(duration)

        if slice_times:
            avg_slice_duration = sum(slice_times) / len(slice_times)
            fastest_slice = min(slice_times)
            slowest_slice = max(slice_times)
        else:
            avg_slice_duration = fastest_slice = slowest_slice = 0

        # Calculate delays between slices
        delays = [s.delay_seconds for s in slices if s.delay_seconds > 0]
        avg_delay = sum(delays) / len(delays) if delays else 0

        # Calculate costs
        slippage_cost = abs(progress.cumulative_slippage) * progress.value_filled_usdt / Decimal("100")
        total_cost = progress.total_fees_usdt + slippage_cost
        cost_per_unit = (
            total_cost / progress.quantity_filled
            if progress.quantity_filled > 0 else Decimal("0")
        )

        # Calculate quality score (0-100)
        quality_score = self._calculate_quality_score(
            fill_rate,
            avg_slippage,
            progress.slices_failed,
            progress.slice_count
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_slippage,
            slip_std_dev,
            progress.slices_failed,
            progress.slice_count,
            avg_delay
        )

        # Create severity distribution (would need impact data)
        severity_dist = dict.fromkeys(ImpactSeverity, 0)

        report = ExecutionReport(
            execution_id=execution_id,
            symbol=progress.symbol,
            side=progress.side,
            total_quantity_ordered=progress.total_quantity,
            total_quantity_filled=progress.quantity_filled,
            fill_rate=fill_rate,
            total_value_usdt=progress.total_value_usdt,
            slice_count=progress.slice_count,
            slices_completed=progress.slices_completed,
            slices_failed=progress.slices_failed,
            slice_size_avg=avg_size,
            slice_size_std_dev=std_dev,
            total_slippage_percent=total_slippage,
            average_slippage_percent=avg_slippage,
            max_slice_slippage=max_slippage,
            min_slice_slippage=min_slippage,
            slippage_std_dev=slip_std_dev,
            total_market_impact=Decimal("0"),  # Would need impact monitor data
            average_impact_per_slice=Decimal("0"),
            impact_severity_distribution=severity_dist,
            total_fees_usdt=progress.total_fees_usdt,
            average_fees_per_slice=(
                progress.total_fees_usdt / progress.slices_completed
                if progress.slices_completed > 0 else Decimal("0")
            ),
            slippage_cost_usdt=slippage_cost,
            total_cost_usdt=total_cost,
            cost_per_unit=cost_per_unit,
            execution_duration_seconds=progress.elapsed_seconds,
            average_slice_duration=avg_slice_duration,
            fastest_slice_seconds=fastest_slice,
            slowest_slice_seconds=slowest_slice,
            average_delay_between_slices=avg_delay,
            execution_quality_score=quality_score,
            recommendations=recommendations,
            started_at=progress.started_at,
            completed_at=datetime.now(),
            status=progress.status,
            abort_reason=progress.abort_reason
        )

        # Save report
        if self.repository:
            await self._save_report(report)

        # Clean up tracking
        del self.active_executions[execution_id]
        del self.slice_records[execution_id]

        logger.info(
            "Generated execution report",
            execution_id=execution_id,
            quality_score=str(quality_score),
            fill_rate=str(fill_rate),
            avg_slippage=str(avg_slippage)
        )

        return report

    def _calculate_quality_score(
        self,
        fill_rate: Decimal,
        avg_slippage: Decimal,
        failures: int,
        total_slices: int
    ) -> Decimal:
        """Calculate execution quality score (0-100)."""
        # Start with perfect score
        score = Decimal("100")

        # Deduct for incomplete fills
        score -= (Decimal("1") - fill_rate) * Decimal("20")

        # Deduct for slippage
        score -= min(abs(avg_slippage) * Decimal("10"), Decimal("30"))

        # Deduct for failures
        if total_slices > 0:
            failure_rate = Decimal(str(failures)) / Decimal(str(total_slices))
            score -= failure_rate * Decimal("20")

        # Ensure score is between 0 and 100
        score = max(Decimal("0"), min(score, Decimal("100")))

        return score.quantize(Decimal("0.01"))

    def _generate_recommendations(
        self,
        avg_slippage: Decimal,
        slip_std_dev: Decimal,
        failures: int,
        total_slices: int,
        avg_delay: float
    ) -> list[str]:
        """Generate execution improvement recommendations."""
        recommendations = []

        # Slippage recommendations
        if abs(avg_slippage) > Decimal("0.5"):
            recommendations.append(
                "High slippage detected. Consider reducing slice sizes or increasing delays."
            )

        if slip_std_dev > Decimal("0.3"):
            recommendations.append(
                "Inconsistent slippage across slices. Market conditions may be volatile."
            )

        # Failure recommendations
        if failures > 0:
            failure_rate = failures / total_slices if total_slices > 0 else 0
            if failure_rate > 0.1:
                recommendations.append(
                    f"{failures} slices failed. Check network stability and order validation."
                )

        # Timing recommendations
        if avg_delay < 2:
            recommendations.append(
                "Consider increasing delays between slices to reduce market impact."
            )
        elif avg_delay > 4:
            recommendations.append(
                "Delays may be excessive. Consider reducing for faster execution."
            )

        # General recommendations
        if abs(avg_slippage) < Decimal("0.1") and failures == 0:
            recommendations.append(
                "Excellent execution quality. Current settings are optimal."
            )

        return recommendations

    async def _save_report(self, report: ExecutionReport) -> None:
        """Save report to database."""
        if self.repository:
            await self.repository.save_execution_report(report)

    def create_progress_widget(self, execution_id: str) -> dict[str, Any]:
        """
        Create real-time progress widget data.
        
        Args:
            execution_id: Execution to display
            
        Returns:
            Widget data for UI display
        """
        if execution_id not in self.active_executions:
            return {"error": "Execution not found"}

        progress = self.active_executions[execution_id]

        return {
            "type": "iceberg_progress",
            "execution_id": execution_id,
            "symbol": progress.symbol,
            "side": progress.side.value,
            "progress": {
                "completed": progress.slices_completed,
                "failed": progress.slices_failed,
                "total": progress.slice_count,
                "percent": str(progress.completion_percent)
            },
            "performance": {
                "avg_slippage": str(progress.average_slippage),
                "max_slippage": str(progress.max_slice_slippage),
                "fees": str(progress.total_fees_usdt)
            },
            "time": {
                "elapsed": progress.elapsed_seconds,
                "estimated_completion": (
                    progress.estimated_completion.isoformat()
                    if progress.estimated_completion else None
                )
            },
            "status": progress.status.value,
            "last_updated": progress.last_updated.isoformat()
        }

    def export_report_json(self, report: ExecutionReport) -> str:
        """
        Export report as JSON.
        
        Args:
            report: Report to export
            
        Returns:
            JSON string representation
        """
        report_dict = {
            "execution_id": report.execution_id,
            "symbol": report.symbol,
            "side": report.side.value,
            "summary": {
                "quantity_filled": str(report.total_quantity_filled),
                "fill_rate": str(report.fill_rate),
                "total_value": str(report.total_value_usdt),
                "quality_score": str(report.execution_quality_score)
            },
            "slicing": {
                "slice_count": report.slice_count,
                "completed": report.slices_completed,
                "failed": report.slices_failed,
                "avg_size": str(report.slice_size_avg)
            },
            "performance": {
                "avg_slippage": str(report.average_slippage_percent),
                "max_slippage": str(report.max_slice_slippage),
                "total_fees": str(report.total_fees_usdt),
                "total_cost": str(report.total_cost_usdt)
            },
            "timing": {
                "duration_seconds": report.execution_duration_seconds,
                "avg_slice_time": report.average_slice_duration,
                "avg_delay": report.average_delay_between_slices
            },
            "recommendations": report.recommendations,
            "status": report.status.value,
            "generated_at": report.generated_at.isoformat()
        }

        return json.dumps(report_dict, indent=2)
