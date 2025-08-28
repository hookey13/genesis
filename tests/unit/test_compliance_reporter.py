"""Unit tests for compliance reporting module."""

import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pandas as pd

from genesis.analytics.compliance_reporter import ComplianceReporter
from genesis.core.constants import TradingTier
from genesis.core.models import Order, OrderSide, OrderStatus, OrderType, Trade


@pytest.fixture
def mock_repository():
    """Create mock repository."""
    repo = AsyncMock()
    repo.get_events_by_aggregate = AsyncMock(return_value=[])
    repo.get_trades_by_account = AsyncMock(return_value=[])
    repo.get_orders_by_account = AsyncMock(return_value=[])
    repo.get_positions_by_account = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def compliance_reporter(mock_repository):
    """Create ComplianceReporter instance."""
    return ComplianceReporter(mock_repository)


@pytest.fixture
def sample_events():
    """Create sample events for audit log."""
    return [
        MagicMock(
            event_type="order.executed",
            timestamp=datetime.now(timezone.utc),
            sequence_number=1,
            event_data=json.dumps({
                "order_id": str(uuid4()),
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "0.1",
                "price": "50000",
                "order_type": "LIMIT",
            })
        ),
        MagicMock(
            event_type="trade.completed",
            timestamp=datetime.now(timezone.utc),
            sequence_number=2,
            event_data=json.dumps({
                "trade_id": str(uuid4()),
                "symbol": "BTC/USDT",
                "side": "SELL",
                "entry_price": "50000",
                "exit_price": "51000",
                "quantity": "0.1",
                "pnl_dollars": "100",
            })
        ),
        MagicMock(
            event_type="position.opened",
            timestamp=datetime.now(timezone.utc),
            sequence_number=3,
            event_data=json.dumps({
                "position_id": str(uuid4()),
                "symbol": "ETH/USDT",
                "side": "LONG",
                "quantity": "1",
                "price": "3000",
            })
        ),
    ]


@pytest.fixture
def sample_trades():
    """Create sample trades."""
    return [
        Trade(
            trade_id=str(uuid4()),
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            quantity=Decimal("0.1"),
            pnl_dollars=Decimal("100"),
            pnl_percent=Decimal("2"),
        ),
        Trade(
            trade_id=str(uuid4()),
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            strategy_id="test_strategy",
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            entry_price=Decimal("3000"),
            exit_price=Decimal("2950"),
            quantity=Decimal("1"),
            pnl_dollars=Decimal("-50"),
            pnl_percent=Decimal("-1.67"),
        ),
    ]


@pytest.fixture
def sample_orders():
    """Create sample orders."""
    return [
        Order(
            order_id=str(uuid4()),
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            filled_quantity=Decimal("0.1"),
            status=OrderStatus.FILLED,
            latency_ms=50,
            maker_fee_paid=Decimal("5"),
            taker_fee_paid=Decimal("0"),
        ),
        Order(
            order_id=str(uuid4()),
            symbol="ETH/USDT",
            type=OrderType.MARKET,
            side=OrderSide.SELL,
            price=Decimal("3000"),
            quantity=Decimal("1"),
            filled_quantity=Decimal("1"),
            status=OrderStatus.FILLED,
            latency_ms=30,
            maker_fee_paid=Decimal("0"),
            taker_fee_paid=Decimal("3"),
        ),
    ]


@pytest.mark.asyncio
async def test_extract_trade_audit_log(compliance_reporter, mock_repository, sample_events):
    """Test extracting trade audit log."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=7)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock
    mock_repository.get_events_by_aggregate.return_value = sample_events
    
    # Extract audit log
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        audit_log = await compliance_reporter.extract_trade_audit_log(
            account_id, start_date, end_date, include_metadata=True
        )
    
    # Verify results
    assert len(audit_log) == 3
    assert audit_log[0]["event_type"] == "order.executed"
    assert audit_log[0]["symbol"] == "BTC/USDT"
    assert audit_log[1]["event_type"] == "trade.completed"
    assert audit_log[1]["pnl_dollars"] == "100"
    assert audit_log[2]["event_type"] == "position.opened"
    assert audit_log[2]["symbol"] == "ETH/USDT"
    
    # Verify repository called
    mock_repository.get_events_by_aggregate.assert_called_once_with(
        aggregate_id=account_id,
        start_time=start_date,
        end_time=end_date,
    )


@pytest.mark.asyncio
async def test_generate_mifid_ii_report(compliance_reporter, mock_repository, sample_trades):
    """Test generating MiFID II compliance report."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock
    mock_repository.get_trades_by_account.return_value = sample_trades
    
    # Generate report
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        report = await compliance_reporter.generate_regulatory_report(
            account_id, "MiFID_II", start_date, end_date
        )
    
    # Verify report structure
    assert report["report_type"] == "MiFID_II"
    assert report["account_id"] == account_id
    assert "transaction_reporting" in report
    assert report["transaction_reporting"]["total_trades"] == 2
    assert "best_execution" in report
    assert "transparency_requirements" in report


@pytest.mark.asyncio
async def test_generate_emir_report(compliance_reporter, mock_repository):
    """Test generating EMIR report."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock positions
    positions = [
        MagicMock(dollar_value=Decimal("5000")),
        MagicMock(dollar_value=Decimal("3000")),
    ]
    mock_repository.get_positions_by_account.return_value = positions
    
    # Generate report
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        report = await compliance_reporter.generate_regulatory_report(
            account_id, "EMIR", start_date, end_date
        )
    
    # Verify report
    assert report["report_type"] == "EMIR"
    assert report["derivatives_exposure"]["total_notional"] == "8000"
    assert report["derivatives_exposure"]["number_of_positions"] == 2
    assert report["derivatives_exposure"]["largest_position"] == "5000"


@pytest.mark.asyncio
async def test_generate_cftc_report(compliance_reporter, mock_repository, sample_trades):
    """Test generating CFTC report."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock with large trades
    large_trade = Trade(
        trade_id=str(uuid4()),
        order_id=str(uuid4()),
        position_id=str(uuid4()),
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        entry_price=Decimal("50000"),
        exit_price=Decimal("51000"),
        quantity=Decimal("2"),  # Large position > $50k
        pnl_dollars=Decimal("2000"),
        pnl_percent=Decimal("2"),
    )
    mock_repository.get_trades_by_account.return_value = [large_trade] + sample_trades
    
    # Generate report
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        report = await compliance_reporter.generate_regulatory_report(
            account_id, "CFTC", start_date, end_date
        )
    
    # Verify report
    assert report["report_type"] == "CFTC"
    assert report["large_trader_reporting"]["reportable_positions"] == 1
    assert Decimal(report["large_trader_reporting"]["total_reportable_volume"]) == Decimal("100000")


@pytest.mark.asyncio
async def test_generate_best_execution_report(compliance_reporter, mock_repository, sample_orders):
    """Test generating best execution report."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock
    mock_repository.get_orders_by_account.return_value = sample_orders
    
    # Generate report
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        report = await compliance_reporter.generate_regulatory_report(
            account_id, "BEST_EXECUTION", start_date, end_date
        )
    
    # Verify report
    assert report["report_type"] == "BEST_EXECUTION"
    assert report["execution_metrics"]["total_orders"] == 2
    assert report["execution_metrics"]["fill_rate"] == 1.0  # All filled
    assert report["execution_metrics"]["average_latency_ms"] == 40  # (50 + 30) / 2


@pytest.mark.asyncio
async def test_generate_transaction_cost_report(compliance_reporter, mock_repository, sample_trades, sample_orders):
    """Test generating transaction cost analysis report."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock
    mock_repository.get_trades_by_account.return_value = sample_trades
    mock_repository.get_orders_by_account.return_value = sample_orders
    
    # Generate report
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        report = await compliance_reporter.generate_regulatory_report(
            account_id, "TRANSACTION_COST", start_date, end_date
        )
    
    # Verify report
    assert report["report_type"] == "TRANSACTION_COST"
    assert report["explicit_costs"]["maker_fees"] == "5"
    assert report["explicit_costs"]["taker_fees"] == "3"
    assert report["explicit_costs"]["total_fees"] == "8"


@pytest.mark.asyncio
async def test_export_compliance_data_json(compliance_reporter, mock_repository, sample_events):
    """Test exporting compliance data as JSON."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=7)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock
    mock_repository.get_events_by_aggregate.return_value = sample_events
    
    # Export as JSON string
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        json_output = await compliance_reporter.export_compliance_data(
            account_id, "JSON", start_date, end_date
        )
    
    # Verify JSON structure
    data = json.loads(json_output)
    assert data["account_id"] == account_id
    assert len(data["audit_log"]) == 3
    assert data["record_count"] == 3


@pytest.mark.asyncio
async def test_export_compliance_data_csv(compliance_reporter, mock_repository, sample_events):
    """Test exporting compliance data as CSV."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=7)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock
    mock_repository.get_events_by_aggregate.return_value = sample_events
    
    # Export as CSV string
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        csv_output = await compliance_reporter.export_compliance_data(
            account_id, "CSV", start_date, end_date
        )
    
    # Verify CSV output
    assert "audit_id" in csv_output
    assert "event_type" in csv_output
    assert "BTC/USDT" in csv_output


@pytest.mark.asyncio
async def test_export_compliance_data_to_file(compliance_reporter, mock_repository, sample_events, tmp_path):
    """Test exporting compliance data to file."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=7)
    end_date = datetime.now(timezone.utc)
    output_path = tmp_path / "compliance_export.json"
    
    # Setup mock
    mock_repository.get_events_by_aggregate.return_value = sample_events
    
    # Export to file
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        result_path = await compliance_reporter.export_compliance_data(
            account_id, "JSON", start_date, end_date, str(output_path)
        )
    
    # Verify file created
    assert result_path == str(output_path)
    assert output_path.exists()
    
    # Verify file content
    with open(output_path) as f:
        data = json.load(f)
        assert data["account_id"] == account_id
        assert len(data["audit_log"]) == 3


@pytest.mark.asyncio
async def test_validate_compliance_requirements(compliance_reporter, mock_repository):
    """Test validating compliance requirements."""
    account_id = str(uuid4())
    
    # Setup compliance settings
    compliance_settings = {
        "max_position_size": "10000",
        "max_daily_trades": 50,
        "reporting_frequency": "daily",
    }
    
    # Setup mock data
    positions = [
        MagicMock(dollar_value=Decimal("5000")),
        MagicMock(dollar_value=Decimal("3000")),
    ]
    trades = [MagicMock() for _ in range(10)]  # 10 trades today
    
    mock_repository.get_positions_by_account.return_value = positions
    mock_repository.get_trades_by_account.return_value = trades
    
    # Validate compliance
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        results = await compliance_reporter.validate_compliance_requirements(
            account_id, compliance_settings
        )
    
    # Verify results
    assert results["position_limits"] is True  # Max position 5000 < 10000 limit
    assert results["daily_trade_limits"] is True  # 10 trades < 50 limit
    assert results["reporting_compliance"] is True  # Placeholder always True


@pytest.mark.asyncio
async def test_validate_compliance_position_limit_exceeded(compliance_reporter, mock_repository):
    """Test compliance validation when position limit exceeded."""
    account_id = str(uuid4())
    
    # Setup compliance settings with low limit
    compliance_settings = {
        "max_position_size": "4000",
    }
    
    # Setup positions exceeding limit
    positions = [
        MagicMock(dollar_value=Decimal("5000")),  # Exceeds limit
        MagicMock(dollar_value=Decimal("3000")),
    ]
    
    mock_repository.get_positions_by_account.return_value = positions
    
    # Validate compliance
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        results = await compliance_reporter.validate_compliance_requirements(
            account_id, compliance_settings
        )
    
    # Verify position limit exceeded
    assert results["position_limits"] is False


@pytest.mark.asyncio
async def test_validate_compliance_trade_limit_exceeded(compliance_reporter, mock_repository):
    """Test compliance validation when daily trade limit exceeded."""
    account_id = str(uuid4())
    
    # Setup compliance settings with low limit
    compliance_settings = {
        "max_daily_trades": 5,
    }
    
    # Setup trades exceeding limit
    trades = [MagicMock() for _ in range(10)]  # 10 trades exceeds limit of 5
    
    mock_repository.get_trades_by_account.return_value = trades
    
    # Validate compliance
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        results = await compliance_reporter.validate_compliance_requirements(
            account_id, compliance_settings
        )
    
    # Verify trade limit exceeded
    assert results["daily_trade_limits"] is False


@pytest.mark.asyncio
async def test_unsupported_report_type(compliance_reporter):
    """Test generating report with unsupported type."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=30)
    end_date = datetime.now(timezone.utc)
    
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        with pytest.raises(ValueError, match="Unsupported report type"):
            await compliance_reporter.generate_regulatory_report(
                account_id, "INVALID_TYPE", start_date, end_date
            )


@pytest.mark.asyncio
async def test_unsupported_export_format(compliance_reporter, mock_repository, sample_events):
    """Test exporting with unsupported format."""
    account_id = str(uuid4())
    start_date = datetime.now(timezone.utc) - timedelta(days=7)
    end_date = datetime.now(timezone.utc)
    
    # Setup mock
    mock_repository.get_events_by_aggregate.return_value = sample_events
    
    with patch("genesis.analytics.compliance_reporter.requires_tier", lambda x: lambda f: f):
        with pytest.raises(ValueError, match="Unsupported export format"):
            await compliance_reporter.export_compliance_data(
                account_id, "PDF", start_date, end_date  # PDF not supported yet
            )