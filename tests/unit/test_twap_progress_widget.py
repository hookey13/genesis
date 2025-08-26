"""Unit tests for TWAP progress widget."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from textual.app import App
from textual.widgets import Static

from genesis.ui.widgets.twap_progress import TwapProgressWidget


class TestTwapProgressWidget:
    """Test TWAP progress widget functionality."""

    @pytest.fixture
    def widget(self):
        """Create TWAP progress widget instance."""
        return TwapProgressWidget()

    @pytest.fixture
    def sample_execution_data(self):
        """Create sample execution data."""
        return {
            "execution_id": "test-exec-123",
            "symbol": "BTC/USDT",
            "side": "BUY",
            "total_quantity": "10.0",
            "executed_quantity": "3.0",
            "remaining_quantity": "7.0",
            "slice_count": 10,
            "completed_slices": 3,
            "arrival_price": "50000",
            "current_price": "50100",
            "twap_price": "50050",
            "participation_rate": "8.5",
            "implementation_shortfall": "0.1",
            "status": "ACTIVE",
            "started_at": datetime.now()
        }

    @pytest.fixture
    def sample_slice_data(self):
        """Create sample slice execution data."""
        return {
            "slice_number": 4,
            "executed_at": datetime.now().isoformat(),
            "executed_quantity": "1.0",
            "execution_price": "50075",
            "participation_rate": "9.0",
            "slippage_bps": "15",
            "status": "EXECUTED"
        }

    def test_init(self, widget):
        """Test widget initialization."""
        assert widget.execution_id is None
        assert widget.symbol == ""
        assert widget.total_quantity == Decimal("0")
        assert widget.executed_quantity == Decimal("0")
        assert widget.status == "INACTIVE"
        assert widget.slice_history == []

    def test_update_execution(self, widget, sample_execution_data):
        """Test updating execution data."""
        widget.update_execution(sample_execution_data)

        assert widget.execution_id == "test-exec-123"
        assert widget.symbol == "BTC/USDT"
        assert widget.side == "BUY"
        assert widget.total_quantity == Decimal("10.0")
        assert widget.executed_quantity == Decimal("3.0")
        assert widget.remaining_quantity == Decimal("7.0")
        assert widget.slice_count == 10
        assert widget.completed_slices == 3
        assert widget.arrival_price == Decimal("50000")
        assert widget.current_price == Decimal("50100")
        assert widget.twap_price == Decimal("50050")
        assert widget.avg_participation_rate == Decimal("8.5")
        assert widget.implementation_shortfall == Decimal("0.1")
        assert widget.status == "ACTIVE"

    def test_add_slice_execution(self, widget, sample_execution_data, sample_slice_data):
        """Test adding slice execution to history."""
        # Initialize widget with execution data
        widget.update_execution(sample_execution_data)
        initial_executed = widget.executed_quantity
        initial_slices = widget.completed_slices

        # Add slice
        widget.add_slice_execution(sample_slice_data)

        assert len(widget.slice_history) == 1
        assert widget.slice_history[0] == sample_slice_data
        assert widget.completed_slices == initial_slices + 1
        assert widget.executed_quantity == initial_executed + Decimal("1.0")
        assert widget.remaining_quantity == widget.total_quantity - widget.executed_quantity

    def test_twap_price_calculation(self, widget):
        """Test TWAP price calculation from slice history."""
        widget.total_quantity = Decimal("10.0")
        
        # Add multiple slices
        slices = [
            {"executed_quantity": "2.0", "execution_price": "50000"},
            {"executed_quantity": "3.0", "execution_price": "50100"},
            {"executed_quantity": "2.0", "execution_price": "50200"},
        ]
        
        for slice_data in slices:
            widget.add_slice_execution(slice_data)

        # Calculate expected TWAP: (2*50000 + 3*50100 + 2*50200) / 7
        expected_twap = (Decimal("100000") + Decimal("150300") + Decimal("100400")) / Decimal("7")
        assert abs(widget.twap_price - expected_twap) < Decimal("0.01")

    def test_update_time_remaining(self, widget, sample_execution_data):
        """Test time remaining calculation."""
        # Set up execution with some progress
        sample_execution_data["started_at"] = datetime.now()
        widget.update_execution(sample_execution_data)
        
        with patch('genesis.ui.widgets.twap_progress.datetime') as mock_datetime:
            # Mock 30 seconds elapsed
            mock_datetime.now.return_value = widget.started_at.replace(
                second=widget.started_at.second + 30
            )
            
            widget.update_time_remaining()
            
            # With 3 slices done in 30 seconds, expect ~70 seconds remaining for 7 slices
            assert widget.time_remaining_seconds > 0

    def test_pause_execution(self, widget, sample_execution_data):
        """Test pausing execution."""
        widget.update_execution(sample_execution_data)
        assert widget.status == "ACTIVE"

        widget.pause_execution()
        assert widget.status == "PAUSED"

    def test_resume_execution(self, widget, sample_execution_data):
        """Test resuming execution."""
        sample_execution_data["status"] = "PAUSED"
        widget.update_execution(sample_execution_data)
        assert widget.status == "PAUSED"

        widget.resume_execution()
        assert widget.status == "ACTIVE"

    def test_complete_execution(self, widget, sample_execution_data):
        """Test completing execution."""
        widget.update_execution(sample_execution_data)

        final_metrics = {
            "twap_price": "50080",
            "implementation_shortfall": "0.16"
        }

        widget.complete_execution(final_metrics)

        assert widget.status == "COMPLETED"
        assert widget.twap_price == Decimal("50080")
        assert widget.implementation_shortfall == Decimal("0.16")

    def test_clear_execution(self, widget, sample_execution_data):
        """Test clearing execution data."""
        widget.update_execution(sample_execution_data)
        assert widget.execution_id is not None

        widget.clear_execution()

        assert widget.execution_id is None
        assert widget.symbol == ""
        assert widget.total_quantity == Decimal("0")
        assert widget.slice_history == []
        assert widget.status == "INACTIVE"

    def test_create_header_no_execution(self, widget):
        """Test header creation with no active execution."""
        with patch.object(widget, 'query_one', return_value=MagicMock(spec=Static)):
            header = widget._create_header()
            assert header is not None
            # Should show "No active TWAP execution"

    def test_create_header_with_execution(self, widget, sample_execution_data):
        """Test header creation with active execution."""
        widget.update_execution(sample_execution_data)
        
        with patch.object(widget, 'query_one', return_value=MagicMock(spec=Static)):
            header = widget._create_header()
            assert header is not None
            # Should show execution details

    def test_create_metrics_table(self, widget, sample_execution_data):
        """Test metrics table creation."""
        widget.update_execution(sample_execution_data)
        
        with patch.object(widget, 'query_one', return_value=MagicMock(spec=Static)):
            table = widget._create_metrics_table()
            assert table is not None
            # Should include price, participation, and shortfall metrics

    def test_create_slices_table_empty(self, widget):
        """Test slice table creation with no slices."""
        with patch.object(widget, 'query_one', return_value=MagicMock(spec=Static)):
            table = widget._create_slices_table()
            assert table is not None
            # Should show empty table

    def test_create_slices_table_with_history(self, widget, sample_slice_data):
        """Test slice table creation with execution history."""
        # Add multiple slices
        for i in range(15):
            slice_data = sample_slice_data.copy()
            slice_data["slice_number"] = i + 1
            widget.slice_history.append(slice_data)
        
        with patch.object(widget, 'query_one', return_value=MagicMock(spec=Static)):
            table = widget._create_slices_table()
            assert table is not None
            # Should show last 10 slices plus summary

    def test_status_colors(self, widget):
        """Test status color mapping."""
        statuses = ["ACTIVE", "PAUSED", "COMPLETED", "FAILED", "CANCELLED"]
        
        for status in statuses:
            widget.status = status
            widget.execution_id = "test"
            with patch.object(widget, 'query_one', return_value=MagicMock(spec=Static)):
                header = widget._create_header()
                assert header is not None

    def test_progress_calculation(self, widget, sample_execution_data):
        """Test progress percentage calculation."""
        widget.update_execution(sample_execution_data)
        
        # 3 out of 10 slices = 30%
        progress_percent = (widget.completed_slices / widget.slice_count) * 100
        assert progress_percent == 30.0
        
        # 3 out of 10 quantity = 30%
        quantity_percent = (widget.executed_quantity / widget.total_quantity) * 100
        assert quantity_percent == 30.0

    def test_multiple_slice_additions(self, widget):
        """Test adding multiple slices updates metrics correctly."""
        widget.total_quantity = Decimal("10.0")
        
        slices = [
            {"executed_quantity": "1.0", "execution_price": "50000", "status": "EXECUTED"},
            {"executed_quantity": "1.5", "execution_price": "50100", "status": "EXECUTED"},
            {"executed_quantity": "0.5", "execution_price": "50200", "status": "EXECUTED"},
        ]
        
        for slice_data in slices:
            widget.add_slice_execution(slice_data)
        
        assert widget.executed_quantity == Decimal("3.0")
        assert widget.remaining_quantity == Decimal("7.0")
        assert widget.completed_slices == 3
        assert len(widget.slice_history) == 3