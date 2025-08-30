"""Integration tests for trade reporting system."""
import asyncio
import json
import csv
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import pytest

from genesis.analytics.trade_reporting import (
    TradeReport,
    TradeReportingSystem
)


class TestTradeReportingIntegration:
    """Integration tests for trade reporting functionality."""
    
    @pytest.fixture
    def large_trade_dataset(self):
        """Create large dataset of trades for integration testing."""
        trades = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
        accounts = ["ACC_001", "ACC_002", "ACC_003"]
        
        for day in range(30):  # 30 days of trades
            for hour in range(24):  # 24 hours per day
                for symbol_idx, symbol in enumerate(symbols):
                    trade = TradeReport(
                        trade_id=f"TRADE_{day:03d}_{hour:02d}_{symbol_idx:02d}",
                        symbol=symbol,
                        side="BUY" if (day + hour + symbol_idx) % 2 == 0 else "SELL",
                        quantity=Decimal("0.1") * (symbol_idx + 1),
                        price=Decimal(str(45000 - day * 100 + hour * 10)),
                        executed_at=base_time + timedelta(days=day, hours=hour),
                        account_id=accounts[(day + hour) % len(accounts)],
                        order_type="MARKET" if hour % 2 == 0 else "LIMIT",
                        fee=Decimal("0.001"),
                        fee_currency="USDT"
                    )
                    trades.append(trade)
        
        return trades
    
    def test_large_dataset_aggregation(self, large_trade_dataset):
        """Test aggregation with large dataset."""
        system = TradeReportingSystem()
        
        # Test date aggregation
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 30)
        date_aggregated = system.aggregate_trades_by_date(
            large_trade_dataset, start_date, end_date
        )
        
        assert len(date_aggregated) == 30  # 30 days
        for date_key, trades in date_aggregated.items():
            assert len(trades) == 120  # 24 hours * 5 symbols per day
        
        # Test symbol aggregation
        symbol_aggregated = system.aggregate_trades_by_symbol(large_trade_dataset)
        assert len(symbol_aggregated) == 5  # 5 unique symbols
        for symbol, trades in symbol_aggregated.items():
            assert len(trades) == 720  # 30 days * 24 hours per symbol
        
        # Test account aggregation
        account_aggregated = system.aggregate_trades_by_account(large_trade_dataset)
        assert len(account_aggregated) == 3  # 3 unique accounts
    
    def test_multi_format_report_generation(self, large_trade_dataset):
        """Test generating reports in multiple formats."""
        system = TradeReportingSystem()
        subset = large_trade_dataset[:100]  # Use subset for faster testing
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Generate all formats
            csv_path = tmpdir_path / "trades.csv"
            json_path = tmpdir_path / "trades.json"
            fix_path = tmpdir_path / "trades.fix"
            
            system.generate_csv_report(subset, str(csv_path))
            system.generate_json_report(subset, str(json_path))
            system.generate_fix_report(subset, str(fix_path))
            
            # Verify all files created
            assert csv_path.exists()
            assert json_path.exists()
            assert fix_path.exists()
            
            # Verify CSV content
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                assert len(rows) == 100
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                data = json.load(f)
                assert data['total_trades'] == 100
                assert len(data['trades']) == 100
            
            # Verify FIX content
            with open(fix_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 100
                for line in lines:
                    assert chr(1) in line  # FIX delimiter present
    
    def test_date_range_filtering(self, large_trade_dataset):
        """Test filtering trades by date range."""
        system = TradeReportingSystem()
        
        # Test week range
        week_start = date(2024, 1, 8)
        week_end = date(2024, 1, 14)
        week_trades = system.aggregate_trades_by_date(
            large_trade_dataset, week_start, week_end
        )
        
        assert len(week_trades) == 7  # 7 days
        total_week_trades = sum(len(trades) for trades in week_trades.values())
        assert total_week_trades == 840  # 7 days * 24 hours * 5 symbols
        
        # Test single day
        single_day = date(2024, 1, 15)
        day_trades = system.aggregate_trades_by_date(
            large_trade_dataset, single_day, single_day
        )
        
        assert len(day_trades) == 1
        assert len(day_trades[single_day]) == 120  # 24 hours * 5 symbols
    
    def test_summary_statistics_large_dataset(self, large_trade_dataset):
        """Test summary statistics calculation with large dataset."""
        system = TradeReportingSystem()
        
        stats = system.calculate_summary_statistics(large_trade_dataset)
        
        assert stats['total_trades'] == 3600  # 30 days * 24 hours * 5 symbols
        assert stats['unique_symbols'] == 5
        
        # Verify buy/sell distribution (should be roughly 50/50)
        assert 1700 < stats['buy_trades'] < 1900
        assert 1700 < stats['sell_trades'] < 1900
        assert stats['buy_trades'] + stats['sell_trades'] == 3600
        
        # Verify total fees
        total_fees = Decimal(stats['total_fees'])
        expected_fees = Decimal("0.001") * 3600
        assert total_fees == expected_fees
    
    def test_concurrent_report_generation(self, large_trade_dataset):
        """Test generating multiple reports concurrently."""
        system = TradeReportingSystem()
        subset = large_trade_dataset[:500]
        
        async def generate_reports_async():
            """Generate reports concurrently."""
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                tasks = []
                loop = asyncio.get_event_loop()
                
                # Create tasks for concurrent generation
                for i in range(3):
                    csv_path = tmpdir_path / f"trades_{i}.csv"
                    json_path = tmpdir_path / f"trades_{i}.json"
                    fix_path = tmpdir_path / f"trades_{i}.fix"
                    
                    tasks.append(loop.run_in_executor(
                        None, system.generate_csv_report, subset, str(csv_path)
                    ))
                    tasks.append(loop.run_in_executor(
                        None, system.generate_json_report, subset, str(json_path)
                    ))
                    tasks.append(loop.run_in_executor(
                        None, system.generate_fix_report, subset, str(fix_path)
                    ))
                
                # Wait for all tasks to complete
                await asyncio.gather(*tasks)
                
                # Verify all files created
                for i in range(3):
                    assert (tmpdir_path / f"trades_{i}.csv").exists()
                    assert (tmpdir_path / f"trades_{i}.json").exists()
                    assert (tmpdir_path / f"trades_{i}.fix").exists()
        
        # Run async test
        asyncio.run(generate_reports_async())
    
    def test_monthly_aggregation(self, large_trade_dataset):
        """Test monthly aggregation and reporting."""
        system = TradeReportingSystem()
        
        # Aggregate entire January
        jan_start = date(2024, 1, 1)
        jan_end = date(2024, 1, 30)
        
        jan_trades = system.aggregate_trades_by_date(
            large_trade_dataset, jan_start, jan_end
        )
        
        # Generate monthly report
        all_jan_trades = []
        for day_trades in jan_trades.values():
            all_jan_trades.extend(day_trades)
        
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            output_path = f.name
        
        try:
            system.generate_json_report(all_jan_trades, output_path)
            
            with open(output_path, 'r') as jsonfile:
                data = json.load(jsonfile)
                assert data['total_trades'] == 3600
                
                # Verify trade dates are within January
                for trade in data['trades']:
                    trade_date = datetime.fromisoformat(
                        trade['executed_at']
                    ).date()
                    assert jan_start <= trade_date <= jan_end
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_account_specific_reporting(self, large_trade_dataset):
        """Test generating reports for specific accounts."""
        system = TradeReportingSystem()
        
        # Aggregate by account
        account_trades = system.aggregate_trades_by_account(large_trade_dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Generate report for each account
            for account_id, trades in account_trades.items():
                report_path = tmpdir_path / f"{account_id}_trades.csv"
                system.generate_csv_report(trades, str(report_path))
                
                # Verify report
                assert report_path.exists()
                
                with open(report_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
                    
                    # Verify all trades belong to the account
                    for row in rows:
                        assert row['account_id'] == account_id