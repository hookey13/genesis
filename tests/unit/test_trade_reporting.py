"""Unit tests for trade reporting system."""
import json
import csv
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import pytest

from genesis.analytics.trade_reporting import (
    TradeReport,
    TradeReportingSystem,
    ReportFormat
)


@pytest.fixture
def sample_trades():
    """Create sample trade reports for testing."""
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    trades = []
    
    for i in range(5):
        trade = TradeReport(
            trade_id=f"TRADE_{i:03d}",
            symbol="BTC/USDT" if i % 2 == 0 else "ETH/USDT",
            side="BUY" if i % 2 == 0 else "SELL",
            quantity=Decimal("0.1") * (i + 1),
            price=Decimal("45000") if i % 2 == 0 else Decimal("2500"),
            executed_at=base_time + timedelta(hours=i),
            account_id=f"ACC_{i % 2}",
            order_type="MARKET",
            fee=Decimal("0.001") * (i + 1),
            fee_currency="USDT"
        )
        trades.append(trade)
    
    return trades


@pytest.fixture
def reporting_system():
    """Create TradeReportingSystem instance."""
    return TradeReportingSystem()


class TestTradeReport:
    """Test TradeReport data structure."""
    
    def test_trade_report_creation(self):
        """Test creating a trade report."""
        trade = TradeReport(
            trade_id="TEST_001",
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("45000"),
            executed_at=datetime.now(),
            account_id="ACC_001",
            order_type="LIMIT",
            fee=Decimal("0.001"),
            fee_currency="BTC"
        )
        
        assert trade.trade_id == "TEST_001"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "BUY"
        assert trade.quantity == Decimal("0.5")
        assert trade.venue == "BINANCE"
    
    def test_to_dict_conversion(self):
        """Test converting trade report to dictionary."""
        exec_time = datetime(2024, 1, 15, 10, 30, 0)
        trade = TradeReport(
            trade_id="TEST_001",
            symbol="BTC/USDT",
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("45000"),
            executed_at=exec_time,
            account_id="ACC_001",
            order_type="MARKET",
            fee=Decimal("0.001"),
            fee_currency="BTC",
            settlement_date=date(2024, 1, 17)
        )
        
        trade_dict = trade.to_dict()
        
        assert trade_dict['quantity'] == "0.5"
        assert trade_dict['price'] == "45000"
        assert trade_dict['fee'] == "0.001"
        assert trade_dict['executed_at'] == exec_time.isoformat()
        assert trade_dict['settlement_date'] == "2024-01-17"
    
    def test_to_fix_format(self):
        """Test converting trade report to FIX format."""
        exec_time = datetime(2024, 1, 15, 10, 30, 0)
        trade = TradeReport(
            trade_id="TEST_001",
            symbol="BTC/USDT",
            side="SELL",
            quantity=Decimal("0.5"),
            price=Decimal("45000"),
            executed_at=exec_time,
            account_id="ACC_001",
            order_type="LIMIT",
            fee=Decimal("0.001"),
            fee_currency="BTC"
        )
        
        fix_message = trade.to_fix_format()
        
        assert "35=8" in fix_message  # ExecutionReport
        assert "55=BTC/USDT" in fix_message  # Symbol
        assert "54=2" in fix_message  # Side (2 for SELL)
        assert "38=0.5" in fix_message  # OrderQty
        assert "44=45000" in fix_message  # Price
        assert "17=TEST_001" in fix_message  # ExecID
        assert chr(1) in fix_message  # FIX delimiter


class TestTradeReportingSystem:
    """Test TradeReportingSystem functionality."""
    
    def test_aggregate_trades_by_date(self, reporting_system, sample_trades):
        """Test aggregating trades by date."""
        start_date = date(2024, 1, 15)
        end_date = date(2024, 1, 15)
        
        aggregated = reporting_system.aggregate_trades_by_date(
            sample_trades, start_date, end_date
        )
        
        assert len(aggregated) == 1
        assert start_date in aggregated
        assert len(aggregated[start_date]) == 5
    
    def test_aggregate_trades_by_symbol(self, reporting_system, sample_trades):
        """Test aggregating trades by symbol."""
        aggregated = reporting_system.aggregate_trades_by_symbol(sample_trades)
        
        assert len(aggregated) == 2
        assert "BTC/USDT" in aggregated
        assert "ETH/USDT" in aggregated
        assert len(aggregated["BTC/USDT"]) == 3
        assert len(aggregated["ETH/USDT"]) == 2
    
    def test_aggregate_trades_by_account(self, reporting_system, sample_trades):
        """Test aggregating trades by account."""
        aggregated = reporting_system.aggregate_trades_by_account(sample_trades)
        
        assert len(aggregated) == 2
        assert "ACC_0" in aggregated
        assert "ACC_1" in aggregated
        assert len(aggregated["ACC_0"]) == 3
        assert len(aggregated["ACC_1"]) == 2
    
    def test_generate_csv_report(self, reporting_system, sample_trades):
        """Test generating CSV format report."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            reporting_system.generate_csv_report(sample_trades, output_path)
            
            # Verify CSV content
            with open(output_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                
                assert len(rows) == 5
                assert rows[0]['trade_id'] == 'TRADE_000'
                assert rows[0]['symbol'] == 'BTC/USDT'
                assert rows[0]['side'] == 'BUY'
                assert rows[0]['quantity'] == '0.1'
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_generate_json_report(self, reporting_system, sample_trades):
        """Test generating JSON format report."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            reporting_system.generate_json_report(sample_trades, output_path)
            
            # Verify JSON content
            with open(output_path, 'r') as jsonfile:
                data = json.load(jsonfile)
                
                assert data['total_trades'] == 5
                assert len(data['trades']) == 5
                assert data['trades'][0]['trade_id'] == 'TRADE_000'
                assert data['trades'][0]['symbol'] == 'BTC/USDT'
                assert 'report_timestamp' in data
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_generate_fix_report(self, reporting_system, sample_trades):
        """Test generating FIX format report."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fix', delete=False) as f:
            output_path = f.name
        
        try:
            reporting_system.generate_fix_report(sample_trades, output_path)
            
            # Verify FIX content
            with open(output_path, 'r') as fixfile:
                lines = fixfile.readlines()
                
                assert len(lines) == 5
                assert "35=8" in lines[0]  # ExecutionReport
                assert "55=BTC/USDT" in lines[0]  # Symbol
                assert chr(1) in lines[0]  # FIX delimiter
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_generate_report_with_format(self, reporting_system, sample_trades):
        """Test generating report with specified format."""
        formats_to_test = ["CSV", "JSON", "FIX"]
        
        for format in formats_to_test:
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'.{format.lower()}', 
                delete=False
            ) as f:
                output_path = f.name
            
            try:
                reporting_system.generate_report(
                    sample_trades, format, output_path
                )
                assert Path(output_path).exists()
                assert Path(output_path).stat().st_size > 0
            finally:
                Path(output_path).unlink(missing_ok=True)
    
    def test_generate_report_invalid_format(self, reporting_system, sample_trades):
        """Test generating report with invalid format."""
        with pytest.raises(ValueError, match="Unsupported report format"):
            reporting_system.generate_report(
                sample_trades, "INVALID", "output.txt"
            )
    
    def test_calculate_summary_statistics(self, reporting_system, sample_trades):
        """Test calculating summary statistics."""
        stats = reporting_system.calculate_summary_statistics(sample_trades)
        
        assert stats['total_trades'] == 5
        assert stats['unique_symbols'] == 2
        assert stats['buy_trades'] == 3
        assert stats['sell_trades'] == 2
        assert Decimal(stats['total_fees']) == Decimal("15") / 1000
        
        # Check total volume calculation
        expected_volume = (
            Decimal("0.1") * Decimal("45000") +  # Trade 0
            Decimal("0.2") * Decimal("2500") +   # Trade 1
            Decimal("0.3") * Decimal("45000") +  # Trade 2
            Decimal("0.4") * Decimal("2500") +   # Trade 3
            Decimal("0.5") * Decimal("45000")    # Trade 4
        )
        assert Decimal(stats['total_volume']) == expected_volume
    
    def test_calculate_summary_statistics_empty(self, reporting_system):
        """Test calculating summary statistics with no trades."""
        stats = reporting_system.calculate_summary_statistics([])
        
        assert stats['total_trades'] == 0
        assert stats['total_volume'] == "0"
        assert stats['total_fees'] == "0"
        assert stats['unique_symbols'] == 0
        assert stats['buy_trades'] == 0
        assert stats['sell_trades'] == 0
    
    def test_generate_report_empty_trades(self, reporting_system):
        """Test generating reports with empty trade list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            # Should not raise, just log warning
            reporting_system.generate_csv_report([], output_path)
            # File should not be created or be empty
            if Path(output_path).exists():
                assert Path(output_path).stat().st_size == 0
        finally:
            Path(output_path).unlink(missing_ok=True)