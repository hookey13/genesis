"""Unit tests for P&L reporting system."""
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
import tempfile
import pytest

from genesis.analytics.reports import (
    PnLEntry,
    MonthlyPnLSummary,
    PnLReportGenerator
)


@pytest.fixture
def sample_pnl_entries():
    """Create sample P&L entries for testing."""
    entries = []
    base_date = date(2024, 1, 1)
    
    # Create mix of winning and losing trades
    for i in range(20):
        is_winner = i % 3 != 0  # 2/3 winners
        
        entry = PnLEntry(
            date=base_date + timedelta(days=i),
            symbol="BTC/USDT" if i % 2 == 0 else "ETH/USDT",
            quantity=Decimal("0.1") * (i % 5 + 1),
            entry_price=Decimal("40000") if i % 2 == 0 else Decimal("2500"),
            exit_price=Decimal("41000") if is_winner else Decimal("39000"),
            gross_pnl=Decimal("100") if is_winner else Decimal("-100"),
            fees=Decimal("5"),
            net_pnl=Decimal("95") if is_winner else Decimal("-105"),
            position_type="LONG",
            holding_period_days=i % 10 + 1
        )
        entries.append(entry)
    
    return entries


@pytest.fixture
def report_generator():
    """Create PnLReportGenerator instance."""
    return PnLReportGenerator()


class TestPnLEntry:
    """Test PnLEntry data structure."""
    
    def test_pnl_entry_creation(self):
        """Test creating a P&L entry."""
        entry = PnLEntry(
            date=date(2024, 1, 15),
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            entry_price=Decimal("40000"),
            exit_price=Decimal("42000"),
            gross_pnl=Decimal("1000"),
            fees=Decimal("10"),
            net_pnl=Decimal("990"),
            position_type="LONG",
            holding_period_days=5
        )
        
        assert entry.date == date(2024, 1, 15)
        assert entry.symbol == "BTC/USDT"
        assert entry.quantity == Decimal("0.5")
        assert entry.net_pnl == Decimal("990")
    
    def test_pnl_entry_to_dict(self):
        """Test converting P&L entry to dictionary."""
        entry = PnLEntry(
            date=date(2024, 1, 15),
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            entry_price=Decimal("40000"),
            exit_price=Decimal("42000"),
            gross_pnl=Decimal("1000"),
            fees=Decimal("10"),
            net_pnl=Decimal("990"),
            position_type="LONG",
            holding_period_days=5
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["date"] == "2024-01-15"
        assert entry_dict["symbol"] == "BTC/USDT"
        assert entry_dict["quantity"] == "0.5"
        assert entry_dict["net_pnl"] == "990"
        assert entry_dict["position_type"] == "LONG"


class TestMonthlyPnLSummary:
    """Test MonthlyPnLSummary data structure."""
    
    def test_monthly_summary_creation(self):
        """Test creating monthly P&L summary."""
        summary = MonthlyPnLSummary(
            year=2024,
            month=1,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            gross_profit=Decimal("10000"),
            gross_loss=Decimal("5000"),
            total_fees=Decimal("200"),
            net_pnl=Decimal("4800"),
            win_rate=Decimal("60"),
            average_win=Decimal("166.67"),
            average_loss=Decimal("125"),
            profit_factor=Decimal("2.0"),
            max_drawdown=Decimal("1000")
        )
        
        assert summary.year == 2024
        assert summary.month == 1
        assert summary.total_trades == 100
        assert summary.net_pnl == Decimal("4800")
        assert summary.win_rate == Decimal("60")
    
    def test_monthly_summary_to_dict(self):
        """Test converting monthly summary to dictionary."""
        summary = MonthlyPnLSummary(
            year=2024,
            month=1,
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            gross_profit=Decimal("10000"),
            gross_loss=Decimal("5000"),
            total_fees=Decimal("200"),
            net_pnl=Decimal("4800"),
            win_rate=Decimal("60"),
            average_win=Decimal("166.67"),
            average_loss=Decimal("125"),
            profit_factor=Decimal("2.0"),
            max_drawdown=Decimal("1000")
        )
        
        summary_dict = summary.to_dict()
        
        assert summary_dict["year"] == 2024
        assert summary_dict["month"] == 1
        assert summary_dict["net_pnl"] == "4800"
        assert summary_dict["win_rate"] == "60"


class TestPnLReportGenerator:
    """Test PnLReportGenerator functionality."""
    
    def test_calculate_monthly_summary(self, report_generator, sample_pnl_entries):
        """Test calculating monthly summary from trades."""
        summary = report_generator.calculate_monthly_summary(
            sample_pnl_entries, 2024, 1
        )
        
        assert summary.year == 2024
        assert summary.month == 1
        assert summary.total_trades == 20  # All January trades
        
        # Check win rate calculation
        assert summary.winning_trades > 0
        assert summary.losing_trades > 0
        assert summary.winning_trades + summary.losing_trades == summary.total_trades
        
        # Check P&L calculations
        assert summary.net_pnl == sum(t.net_pnl for t in sample_pnl_entries)
        assert summary.total_fees == sum(t.fees for t in sample_pnl_entries)
    
    def test_calculate_monthly_summary_empty(self, report_generator):
        """Test calculating monthly summary with no trades."""
        summary = report_generator.calculate_monthly_summary([], 2024, 1)
        
        assert summary.year == 2024
        assert summary.month == 1
        assert summary.total_trades == 0
        assert summary.net_pnl == Decimal("0")
        assert summary.win_rate == Decimal("0")
    
    def test_best_worst_trade_identification(self, report_generator, sample_pnl_entries):
        """Test identifying best and worst trades."""
        summary = report_generator.calculate_monthly_summary(
            sample_pnl_entries, 2024, 1
        )
        
        assert summary.best_trade is not None
        assert summary.worst_trade is not None
        
        # Best trade should have highest P&L
        max_pnl = max(t.net_pnl for t in sample_pnl_entries)
        assert summary.best_trade.net_pnl == max_pnl
        
        # Worst trade should have lowest P&L
        min_pnl = min(t.net_pnl for t in sample_pnl_entries)
        assert summary.worst_trade.net_pnl == min_pnl
    
    def test_max_drawdown_calculation(self, report_generator):
        """Test max drawdown calculation."""
        # Create trades with specific P&L pattern
        trades = []
        base_date = date(2024, 1, 1)
        
        # Pattern: +100, +100, -150, -150, +100
        pnl_values = [100, 100, -150, -150, 100]
        
        for i, pnl in enumerate(pnl_values):
            entry = PnLEntry(
                date=base_date + timedelta(days=i),
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                entry_price=Decimal("40000"),
                exit_price=Decimal("40000") + Decimal(str(pnl)),
                gross_pnl=Decimal(str(pnl)),
                fees=Decimal("0"),
                net_pnl=Decimal(str(pnl)),
                position_type="LONG",
                holding_period_days=1
            )
            trades.append(entry)
        
        summary = report_generator.calculate_monthly_summary(trades, 2024, 1)
        
        # Peak is 200 (after first two trades)
        # Trough is -100 (after fourth trade)
        # Max drawdown should be 300
        assert summary.max_drawdown == Decimal("300")
    
    def test_profit_factor_calculation(self, report_generator):
        """Test profit factor calculation."""
        trades = []
        base_date = date(2024, 1, 1)
        
        # Create specific winning and losing trades
        # 3 wins of 100 each = 300 gross profit
        # 2 losses of 50 each = 100 gross loss
        # Profit factor = 300/100 = 3.0
        
        for i in range(5):
            if i < 3:  # Winners
                pnl = Decimal("100")
            else:  # Losers
                pnl = Decimal("-50")
            
            entry = PnLEntry(
                date=base_date + timedelta(days=i),
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                entry_price=Decimal("40000"),
                exit_price=Decimal("40000") + pnl,
                gross_pnl=pnl,
                fees=Decimal("0"),
                net_pnl=pnl,
                position_type="LONG",
                holding_period_days=1
            )
            trades.append(entry)
        
        summary = report_generator.calculate_monthly_summary(trades, 2024, 1)
        
        assert summary.profit_factor == Decimal("3")
    
    def test_generate_json_report(self, report_generator, sample_pnl_entries):
        """Test generating JSON format P&L report."""
        summary = report_generator.calculate_monthly_summary(
            sample_pnl_entries, 2024, 1
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            report_generator.generate_json_report(summary, output_path)
            
            # Verify JSON content
            with open(output_path, 'r') as jsonfile:
                data = json.load(jsonfile)
                
                assert data["report_type"] == "monthly_pnl"
                assert "generated_at" in data
                assert data["summary"]["year"] == 2024
                assert data["summary"]["month"] == 1
                assert data["summary"]["total_trades"] == 20
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_generate_html_report(self, report_generator, sample_pnl_entries):
        """Test generating HTML format P&L report."""
        summary = report_generator.calculate_monthly_summary(
            sample_pnl_entries, 2024, 1
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            output_path = f.name
        
        try:
            report_generator.generate_html_report(summary, output_path)
            
            # Check if file was created (may be empty if Jinja2 not available)
            assert Path(output_path).exists()
            
            # If Jinja2 is available, verify content
            try:
                import jinja2
                with open(output_path, 'r') as htmlfile:
                    content = htmlfile.read()
                    assert "Monthly P&L Report" in content
                    assert "2024" in content
            except ImportError:
                pass
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_generate_pdf_report(self, report_generator, sample_pnl_entries):
        """Test generating PDF format P&L report."""
        summary = report_generator.calculate_monthly_summary(
            sample_pnl_entries, 2024, 1
        )
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            output_path = f.name
        
        try:
            report_generator.generate_pdf_report(summary, output_path)
            
            # Check if file was created (may not exist if reportlab not available)
            try:
                import reportlab
                assert Path(output_path).exists()
                assert Path(output_path).stat().st_size > 0
            except ImportError:
                pass
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_generate_quarterly_report(self, report_generator):
        """Test generating quarterly P&L report."""
        # Create trades for Q1
        trades = []
        
        for month in [1, 2, 3]:
            for day in range(1, 10):
                entry = PnLEntry(
                    date=date(2024, month, day),
                    symbol="BTC/USDT",
                    quantity=Decimal("1"),
                    entry_price=Decimal("40000"),
                    exit_price=Decimal("41000"),
                    gross_pnl=Decimal("1000"),
                    fees=Decimal("10"),
                    net_pnl=Decimal("990"),
                    position_type="LONG",
                    holding_period_days=1
                )
                trades.append(entry)
        
        quarterly_report = report_generator.generate_quarterly_report(
            trades, 2024, 1
        )
        
        assert quarterly_report["year"] == 2024
        assert quarterly_report["quarter"] == 1
        assert len(quarterly_report["months"]) == 3
        assert quarterly_report["quarterly_summary"]["total_trades"] == 27  # 9 * 3
        
        # Check quarterly P&L
        expected_pnl = Decimal("990") * 27
        assert Decimal(quarterly_report["quarterly_summary"]["net_pnl"]) == expected_pnl
    
    def test_generate_quarterly_report_invalid_quarter(self, report_generator):
        """Test generating quarterly report with invalid quarter."""
        with pytest.raises(ValueError, match="Invalid quarter"):
            report_generator.generate_quarterly_report([], 2024, 5)
    
    def test_generate_annual_report(self, report_generator):
        """Test generating annual P&L report."""
        # Create trades for full year
        trades = []
        
        for month in range(1, 13):
            for day in range(1, 5):
                entry = PnLEntry(
                    date=date(2024, month, day),
                    symbol="BTC/USDT",
                    quantity=Decimal("1"),
                    entry_price=Decimal("40000"),
                    exit_price=Decimal("41000") if month % 2 == 0 else Decimal("39000"),
                    gross_pnl=Decimal("1000") if month % 2 == 0 else Decimal("-1000"),
                    fees=Decimal("10"),
                    net_pnl=Decimal("990") if month % 2 == 0 else Decimal("-1010"),
                    position_type="LONG",
                    holding_period_days=1
                )
                trades.append(entry)
        
        annual_report = report_generator.generate_annual_report(trades, 2024)
        
        assert annual_report["year"] == 2024
        assert annual_report["months_traded"] == 12
        assert len(annual_report["monthly_summaries"]) == 12
        assert annual_report["annual_summary"]["total_trades"] == 48  # 4 * 12
        
        # Check best and worst months
        assert annual_report["annual_summary"]["best_month"] in [2, 4, 6, 8, 10, 12]
        assert annual_report["annual_summary"]["worst_month"] in [1, 3, 5, 7, 9, 11]
    
    def test_win_rate_calculation(self, report_generator):
        """Test win rate calculation accuracy."""
        trades = []
        base_date = date(2024, 1, 1)
        
        # Create exactly 60% winners (6 out of 10)
        for i in range(10):
            is_winner = i < 6
            
            entry = PnLEntry(
                date=base_date + timedelta(days=i),
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                entry_price=Decimal("40000"),
                exit_price=Decimal("41000") if is_winner else Decimal("39000"),
                gross_pnl=Decimal("1000") if is_winner else Decimal("-1000"),
                fees=Decimal("0"),
                net_pnl=Decimal("1000") if is_winner else Decimal("-1000"),
                position_type="LONG",
                holding_period_days=1
            )
            trades.append(entry)
        
        summary = report_generator.calculate_monthly_summary(trades, 2024, 1)
        
        assert summary.winning_trades == 6
        assert summary.losing_trades == 4
        assert summary.win_rate == Decimal("60")