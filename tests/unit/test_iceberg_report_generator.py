"""Unit tests for Iceberg Report Generator."""

import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest

from genesis.analytics.iceberg_report_generator import (
    ExecutionProgress,
    ExecutionReport,
    ExecutionStatus,
    IcebergReportGenerator,
    SliceExecutionRecord,
)
from genesis.core.models import OrderSide


class TestIcebergReportGenerator:
    """Test suite for IcebergReportGenerator."""

    @pytest.fixture
    def report_generator(self):
        """Create report generator instance."""
        mock_repo = Mock()
        mock_repo.save_iceberg_report = AsyncMock()
        return IcebergReportGenerator(repository=mock_repo)

    @pytest.fixture
    def sample_execution_report(self):
        """Create sample execution report."""
        return ExecutionReport(
            report_id=str(uuid.uuid4()),
            execution_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            original_quantity=Decimal("1.0"),
            filled_quantity=Decimal("1.0"),
            total_slices=5,
            completed_slices=5,
            status=ExecutionStatus.COMPLETED,
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now(),
            average_price=Decimal("50000"),
            total_fees=Decimal("50"),
            total_slippage=Decimal("0.15"),
            average_impact=Decimal("0.08"),
            max_slice_impact=Decimal("0.12"),
            execution_time_seconds=300,
            slices=[
                SliceExecutionRecord(
                    slice_number=i,
                    quantity=Decimal("0.2"),
                    price=Decimal("50000") + Decimal(i * 10),
                    slippage=Decimal("0.03"),
                    impact=Decimal("0.02"),
                    execution_time_ms=500 + i * 100,
                    timestamp=datetime.now() - timedelta(minutes=5 - i),
                )
                for i in range(1, 6)
            ],
            abort_reason=None,
            recommendations=["Consider smaller slice sizes for better execution"],
            quality_score=85,
        )

    def test_create_progress_widget(self, report_generator):
        """Test creating progress widget data."""
        execution_id = str(uuid.uuid4())

        # Track execution progress
        report_generator.executions[execution_id] = ExecutionProgress(
            execution_id=execution_id,
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            total_quantity=Decimal("1.0"),
            total_value_usdt=Decimal("50000"),
            slice_count=10,
            slices_completed=4,
            slices_failed=0,
            quantity_filled=Decimal("0.4"),
            value_filled_usdt=Decimal("20000"),
            completion_percent=Decimal("40"),
        )

        widget_data = report_generator.create_progress_widget(execution_id)

        assert widget_data["execution_id"] == execution_id
        assert widget_data["symbol"] == "BTC/USDT"
        assert "progress" in widget_data
        assert "slices" in widget_data
        assert "filled" in widget_data

    def test_export_report_json(self, report_generator, sample_execution_report):
        """Test exporting report as JSON."""
        json_output = report_generator.export_report_json(sample_execution_report)

        assert isinstance(json_output, str)

        # Parse JSON to verify structure
        data = json.loads(json_output)
        assert data["execution_id"] == sample_execution_report.execution_id
        assert data["symbol"] == "BTC/USDT"
        assert data["status"] == "COMPLETED"
        assert data["total_slices"] == 5
        assert data["quality_score"] == 85
        assert len(data["slices"]) == 5

    @pytest.mark.asyncio
    async def test_save_report(self, report_generator, sample_execution_report):
        """Test saving report to repository."""
        await report_generator._save_report(sample_execution_report)

        report_generator.repository.save_iceberg_report.assert_called_once()
        call_args = report_generator.repository.save_iceberg_report.call_args[0]
        assert call_args[0] == sample_execution_report

    def test_execution_status_enum(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.PENDING == "PENDING"
        assert ExecutionStatus.EXECUTING == "EXECUTING"
        assert ExecutionStatus.COMPLETED == "COMPLETED"
        assert ExecutionStatus.ABORTED == "ABORTED"

    def test_slice_execution_record(self):
        """Test SliceExecutionRecord dataclass."""
        slice_record = SliceExecutionRecord(
            slice_number=1,
            quantity=Decimal("0.2"),
            price=Decimal("50000"),
            slippage=Decimal("0.05"),
            impact=Decimal("0.03"),
            execution_time_ms=500,
            timestamp=datetime.now(),
        )

        assert slice_record.slice_number == 1
        assert slice_record.quantity == Decimal("0.2")
        assert slice_record.price == Decimal("50000")
        assert slice_record.slippage == Decimal("0.05")
        assert slice_record.impact == Decimal("0.03")
        assert slice_record.execution_time_ms == 500
        assert isinstance(slice_record.timestamp, datetime)

    def test_execution_progress_tracking(self, report_generator):
        """Test tracking execution progress."""
        execution_id = str(uuid.uuid4())

        # Create initial progress
        progress = ExecutionProgress(
            execution_id=execution_id,
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            total_quantity=Decimal("2.0"),
            total_value_usdt=Decimal("4000"),
            slice_count=8,
            slices_completed=0,
            slices_failed=0,
            quantity_filled=Decimal("0"),
            value_filled_usdt=Decimal("0"),
            completion_percent=Decimal("0"),
        )

        report_generator.executions[execution_id] = progress

        # Update progress
        progress.slices_completed = 3
        progress.completion_percent = Decimal("37.5")
        progress.quantity_filled = Decimal("0.75")
        progress.value_filled_usdt = Decimal("1500")

        assert report_generator.executions[execution_id].slices_completed == 3
        assert report_generator.executions[execution_id].completion_percent == Decimal(
            "37.5"
        )

    def test_execution_report_with_abort(self):
        """Test execution report for aborted execution."""
        report = ExecutionReport(
            report_id=str(uuid.uuid4()),
            execution_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            original_quantity=Decimal("1.0"),
            filled_quantity=Decimal("0.4"),  # Only 2 of 5 slices filled
            total_slices=5,
            completed_slices=2,
            status=ExecutionStatus.ABORTED,
            start_time=datetime.now() - timedelta(minutes=2),
            end_time=datetime.now() - timedelta(minutes=1),
            average_price=Decimal("50100"),
            total_fees=Decimal("20"),
            total_slippage=Decimal("0.55"),  # High slippage caused abort
            average_impact=Decimal("0.30"),
            max_slice_impact=Decimal("0.60"),
            execution_time_seconds=60,
            slices=[
                SliceExecutionRecord(
                    slice_number=1,
                    quantity=Decimal("0.2"),
                    price=Decimal("50050"),
                    slippage=Decimal("0.10"),
                    impact=Decimal("0.05"),
                    execution_time_ms=500,
                    timestamp=datetime.now() - timedelta(minutes=2),
                ),
                SliceExecutionRecord(
                    slice_number=2,
                    quantity=Decimal("0.2"),
                    price=Decimal("50300"),  # High slippage
                    slippage=Decimal("0.60"),
                    impact=Decimal("0.55"),
                    execution_time_ms=600,
                    timestamp=datetime.now() - timedelta(minutes=1, seconds=30),
                ),
            ],
            abort_reason="Slippage exceeded 0.5% threshold",
            recommendations=[
                "Market conditions too volatile",
                "Consider smaller order size",
                "Wait for better liquidity",
            ],
            quality_score=25,  # Low score due to abort
        )

        assert report.status == ExecutionStatus.ABORTED
        assert report.completed_slices < report.total_slices
        assert report.filled_quantity < report.original_quantity
        assert report.abort_reason is not None
        assert "slippage" in report.abort_reason.lower()
        assert report.quality_score < 50  # Poor quality due to abort

    def test_multiple_execution_tracking(self, report_generator):
        """Test tracking multiple concurrent executions."""
        execution_ids = [str(uuid.uuid4()) for _ in range(3)]

        # Create multiple executions
        for i, exec_id in enumerate(execution_ids):
            report_generator.executions[exec_id] = ExecutionProgress(
                execution_id=exec_id,
                symbol=f"PAIR{i}/USDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                total_quantity=Decimal("1.0"),
                total_value_usdt=Decimal("50000"),
                slice_count=5 + i,
                slices_completed=i,
                slices_failed=0,
                quantity_filled=Decimal(str(i * 0.2)),
                value_filled_usdt=Decimal(str(i * 10000)),
                completion_percent=Decimal(str((i / (5 + i)) * 100)),
            )

        assert len(report_generator.executions) == 3

        # Verify each execution is tracked independently
        for i, exec_id in enumerate(execution_ids):
            execution = report_generator.executions[exec_id]
            assert execution.symbol == f"PAIR{i}/USDT"
            assert execution.slice_count == 5 + i
            assert execution.slices_completed == i

    def test_quality_score_calculation(self, sample_execution_report):
        """Test quality score ranges."""
        # Good execution
        good_report = sample_execution_report
        assert good_report.quality_score >= 80

        # Poor execution (high slippage)
        poor_report = ExecutionReport(
            report_id=str(uuid.uuid4()),
            execution_id=str(uuid.uuid4()),
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            original_quantity=Decimal("1.0"),
            filled_quantity=Decimal("1.0"),
            total_slices=5,
            completed_slices=5,
            status=ExecutionStatus.COMPLETED,
            start_time=datetime.now() - timedelta(minutes=10),
            end_time=datetime.now(),
            average_price=Decimal("50500"),
            total_fees=Decimal("100"),
            total_slippage=Decimal("1.0"),  # High slippage
            average_impact=Decimal("0.5"),  # High impact
            max_slice_impact=Decimal("0.8"),
            execution_time_seconds=600,  # Slow execution
            slices=[],
            abort_reason=None,
            recommendations=["Execution quality was poor"],
            quality_score=35,  # Low score
        )

        assert poor_report.quality_score < 50
        assert poor_report.total_slippage > Decimal("0.5")
        assert poor_report.average_impact > Decimal("0.3")
