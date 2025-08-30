"""Unit tests for tax lot tracking system."""
from datetime import datetime, timedelta
from decimal import Decimal
import pytest

from genesis.analytics.tax_lots import (
    TaxLot,
    TaxLotTracker,
    LotAssignment,
    LotStatus
)


@pytest.fixture
def tax_lot_tracker():
    """Create TaxLotTracker instance."""
    return TaxLotTracker()


@pytest.fixture
def sample_lots():
    """Create sample tax lots for testing."""
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    lots = []
    
    # Create 5 lots with different acquisition times and costs
    for i in range(5):
        lot = TaxLot(
            lot_id=f"LOT_{i:03d}",
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            remaining_quantity=Decimal("1.0"),
            cost_per_unit=Decimal(str(40000 + i * 1000)),  # Varying costs
            acquired_at=base_time + timedelta(days=i),
            account_id="ACC_001",
            order_id=f"ORDER_{i:03d}"
        )
        lots.append(lot)
    
    return lots


class TestTaxLot:
    """Test TaxLot data structure."""
    
    def test_tax_lot_creation(self):
        """Test creating a tax lot."""
        lot = TaxLot(
            lot_id="LOT_001",
            symbol="BTC/USDT",
            quantity=Decimal("2.5"),
            remaining_quantity=Decimal("2.5"),
            cost_per_unit=Decimal("45000"),
            acquired_at=datetime.now(),
            account_id="ACC_001",
            order_id="ORDER_001"
        )
        
        assert lot.lot_id == "LOT_001"
        assert lot.symbol == "BTC/USDT"
        assert lot.quantity == Decimal("2.5")
        assert lot.remaining_quantity == Decimal("2.5")
        assert lot.cost_per_unit == Decimal("45000")
        assert lot.status == LotStatus.OPEN
        assert lot.closed_quantity == Decimal("0")
        assert lot.realized_pnl == Decimal("0")
    
    def test_total_cost_calculation(self):
        """Test total cost calculation."""
        lot = TaxLot(
            lot_id="LOT_001",
            symbol="BTC/USDT",
            quantity=Decimal("2.0"),
            remaining_quantity=Decimal("2.0"),
            cost_per_unit=Decimal("45000"),
            acquired_at=datetime.now(),
            account_id="ACC_001",
            order_id="ORDER_001"
        )
        
        assert lot.total_cost == Decimal("90000")
        assert lot.remaining_cost == Decimal("90000")
    
    def test_close_partial(self):
        """Test closing partial quantity from lot."""
        lot = TaxLot(
            lot_id="LOT_001",
            symbol="BTC/USDT",
            quantity=Decimal("2.0"),
            remaining_quantity=Decimal("2.0"),
            cost_per_unit=Decimal("40000"),
            acquired_at=datetime.now(),
            account_id="ACC_001",
            order_id="ORDER_001"
        )
        
        # Close 0.5 BTC at 50000
        pnl = lot.close_partial(Decimal("0.5"), Decimal("50000"))
        
        assert lot.remaining_quantity == Decimal("1.5")
        assert lot.closed_quantity == Decimal("0.5")
        assert lot.status == LotStatus.PARTIAL
        
        # P&L = (50000 - 40000) * 0.5 = 5000
        assert pnl == Decimal("5000")
        assert lot.realized_pnl == Decimal("5000")
    
    def test_close_full_lot(self):
        """Test closing entire lot."""
        lot = TaxLot(
            lot_id="LOT_001",
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            remaining_quantity=Decimal("1.0"),
            cost_per_unit=Decimal("40000"),
            acquired_at=datetime.now(),
            account_id="ACC_001",
            order_id="ORDER_001"
        )
        
        # Close entire lot at 45000
        pnl = lot.close_partial(Decimal("1.0"), Decimal("45000"))
        
        assert lot.remaining_quantity == Decimal("0")
        assert lot.closed_quantity == Decimal("1.0")
        assert lot.status == LotStatus.CLOSED
        assert pnl == Decimal("5000")
    
    def test_close_partial_invalid_quantity(self):
        """Test closing more than available quantity."""
        lot = TaxLot(
            lot_id="LOT_001",
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            remaining_quantity=Decimal("1.0"),
            cost_per_unit=Decimal("40000"),
            acquired_at=datetime.now(),
            account_id="ACC_001",
            order_id="ORDER_001"
        )
        
        with pytest.raises(ValueError, match="Cannot close"):
            lot.close_partial(Decimal("2.0"), Decimal("45000"))


class TestTaxLotTracker:
    """Test TaxLotTracker functionality."""
    
    def test_add_lot(self, tax_lot_tracker):
        """Test adding a new lot."""
        lot = TaxLot(
            lot_id="LOT_001",
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            remaining_quantity=Decimal("1.0"),
            cost_per_unit=Decimal("40000"),
            acquired_at=datetime.now(),
            account_id="ACC_001",
            order_id="ORDER_001"
        )
        
        tax_lot_tracker.add_lot(lot)
        
        assert "BTC/USDT" in tax_lot_tracker.lots
        assert len(tax_lot_tracker.lots["BTC/USDT"]) == 1
        assert "LOT_001" in tax_lot_tracker.lot_by_id
    
    def test_get_open_lots_fifo(self, tax_lot_tracker, sample_lots):
        """Test getting open lots in FIFO order."""
        # Add all sample lots
        for lot in sample_lots:
            tax_lot_tracker.add_lot(lot)
        
        open_lots = tax_lot_tracker.get_open_lots("BTC/USDT", "FIFO")
        
        # Should be ordered by acquisition date (oldest first)
        assert len(open_lots) == 5
        assert open_lots[0].lot_id == "LOT_000"  # Oldest
        assert open_lots[-1].lot_id == "LOT_004"  # Newest
    
    def test_get_open_lots_lifo(self, tax_lot_tracker, sample_lots):
        """Test getting open lots in LIFO order."""
        for lot in sample_lots:
            tax_lot_tracker.add_lot(lot)
        
        open_lots = tax_lot_tracker.get_open_lots("BTC/USDT", "LIFO")
        
        # Should be ordered by acquisition date (newest first)
        assert len(open_lots) == 5
        assert open_lots[0].lot_id == "LOT_004"  # Newest
        assert open_lots[-1].lot_id == "LOT_000"  # Oldest
    
    def test_get_open_lots_hifo(self, tax_lot_tracker, sample_lots):
        """Test getting open lots in HIFO order."""
        for lot in sample_lots:
            tax_lot_tracker.add_lot(lot)
        
        open_lots = tax_lot_tracker.get_open_lots("BTC/USDT", "HIFO")
        
        # Should be ordered by cost (highest first)
        assert len(open_lots) == 5
        assert open_lots[0].cost_per_unit == Decimal("44000")  # Highest cost
        assert open_lots[-1].cost_per_unit == Decimal("40000")  # Lowest cost
    
    def test_process_sale_fifo(self, tax_lot_tracker, sample_lots):
        """Test processing a sale with FIFO method."""
        for lot in sample_lots:
            tax_lot_tracker.add_lot(lot)
        
        # Sell 2.5 BTC at 50000
        assignment = tax_lot_tracker.process_sale(
            sale_id="SALE_001",
            symbol="BTC/USDT",
            quantity=Decimal("2.5"),
            price=Decimal("50000"),
            sale_date=datetime.now(),
            method="FIFO"
        )
        
        assert assignment.sale_quantity == Decimal("2.5")
        assert assignment.sale_price == Decimal("50000")
        assert assignment.method_used == "FIFO"
        assert len(assignment.lot_assignments) == 3  # First 3 lots
        
        # Check that first 2 lots are fully closed
        assert tax_lot_tracker.lot_by_id["LOT_000"].status == LotStatus.CLOSED
        assert tax_lot_tracker.lot_by_id["LOT_001"].status == LotStatus.CLOSED
        # Third lot should be partial
        assert tax_lot_tracker.lot_by_id["LOT_002"].status == LotStatus.PARTIAL
        assert tax_lot_tracker.lot_by_id["LOT_002"].remaining_quantity == Decimal("0.5")
    
    def test_process_sale_lifo(self, tax_lot_tracker, sample_lots):
        """Test processing a sale with LIFO method."""
        for lot in sample_lots:
            tax_lot_tracker.add_lot(lot)
        
        # Sell 1.5 BTC at 50000
        assignment = tax_lot_tracker.process_sale(
            sale_id="SALE_001",
            symbol="BTC/USDT",
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
            sale_date=datetime.now(),
            method="LIFO"
        )
        
        assert assignment.method_used == "LIFO"
        assert len(assignment.lot_assignments) == 2
        
        # Check that newest lots are used first
        assert tax_lot_tracker.lot_by_id["LOT_004"].status == LotStatus.CLOSED
        assert tax_lot_tracker.lot_by_id["LOT_003"].remaining_quantity == Decimal("0.5")
    
    def test_process_sale_insufficient_lots(self, tax_lot_tracker):
        """Test processing sale with insufficient lots."""
        # Add only 1 BTC
        lot = TaxLot(
            lot_id="LOT_001",
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            remaining_quantity=Decimal("1.0"),
            cost_per_unit=Decimal("40000"),
            acquired_at=datetime.now(),
            account_id="ACC_001",
            order_id="ORDER_001"
        )
        tax_lot_tracker.add_lot(lot)
        
        # Try to sell 2 BTC
        assignment = tax_lot_tracker.process_sale(
            sale_id="SALE_001",
            symbol="BTC/USDT",
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
            sale_date=datetime.now(),
            method="FIFO"
        )
        
        # Should handle gracefully with warning
        assert len(assignment.lot_assignments) == 1
        assert assignment.lot_assignments[0][1] == Decimal("1.0")  # Only 1 BTC assigned
    
    def test_get_position_summary(self, tax_lot_tracker, sample_lots):
        """Test getting position summary."""
        for lot in sample_lots:
            tax_lot_tracker.add_lot(lot)
        
        # Process a sale to create some realized P&L
        tax_lot_tracker.process_sale(
            sale_id="SALE_001",
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            sale_date=datetime.now(),
            method="FIFO"
        )
        
        summary = tax_lot_tracker.get_position_summary("BTC/USDT")
        
        assert summary["symbol"] == "BTC/USDT"
        assert Decimal(summary["total_quantity"]) == Decimal("4.0")  # 5 - 1 sold
        assert summary["open_lots"] == 4
        assert Decimal(summary["realized_pnl"]) == Decimal("10000")  # (50000-40000)*1
    
    def test_get_position_summary_no_lots(self, tax_lot_tracker):
        """Test getting position summary with no lots."""
        summary = tax_lot_tracker.get_position_summary("ETH/USDT")
        
        assert summary["symbol"] == "ETH/USDT"
        assert summary["total_quantity"] == "0"
        assert summary["average_cost"] == "0"
        assert summary["total_cost_basis"] == "0"
        assert summary["realized_pnl"] == "0"
        assert summary["open_lots"] == 0
    
    def test_calculate_unrealized_pnl(self, tax_lot_tracker, sample_lots):
        """Test calculating unrealized P&L."""
        for lot in sample_lots:
            tax_lot_tracker.add_lot(lot)
        
        # Calculate unrealized P&L at current price of 50000
        unrealized = tax_lot_tracker.calculate_unrealized_pnl(
            "BTC/USDT", Decimal("50000")
        )
        
        # Each lot has 1 BTC, costs vary from 40000 to 44000
        # Average cost = 42000, current = 50000
        # Unrealized = (50000 - 40000) + (50000 - 41000) + ... = 40000
        expected = Decimal("40000")  # (10000 + 9000 + 8000 + 7000 + 6000)
        assert unrealized == expected
    
    def test_get_tax_report(self, tax_lot_tracker, sample_lots):
        """Test generating tax report."""
        for lot in sample_lots:
            tax_lot_tracker.add_lot(lot)
        
        sale_date = datetime(2024, 1, 15, 10, 0, 0)
        
        # Process some sales
        tax_lot_tracker.process_sale(
            sale_id="SALE_001",
            symbol="BTC/USDT",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            sale_date=sale_date,
            method="FIFO"
        )
        
        tax_lot_tracker.process_sale(
            sale_id="SALE_002",
            symbol="BTC/USDT",
            quantity=Decimal("0.5"),
            price=Decimal("51000"),
            sale_date=sale_date + timedelta(days=1),
            method="FIFO"
        )
        
        # Generate report for January
        report = tax_lot_tracker.get_tax_report(
            datetime(2024, 1, 1),
            datetime(2024, 1, 31)
        )
        
        assert report["num_sales"] == 2
        assert Decimal(report["total_proceeds"]) == Decimal("75500")  # 50000 + 25500
        assert Decimal(report["total_realized_pnl"]) == Decimal("15500")  # 10000 + 5500
        assert "BTC/USDT" in report["by_symbol"]
        assert report["by_symbol"]["BTC/USDT"]["num_sales"] == 2
    
    def test_get_lot_details(self, tax_lot_tracker):
        """Test getting lot details."""
        lot = TaxLot(
            lot_id="LOT_001",
            symbol="BTC/USDT",
            quantity=Decimal("2.0"),
            remaining_quantity=Decimal("1.5"),
            cost_per_unit=Decimal("40000"),
            acquired_at=datetime(2024, 1, 1, 10, 0, 0),
            account_id="ACC_001",
            order_id="ORDER_001",
            closed_quantity=Decimal("0.5"),
            status=LotStatus.PARTIAL,
            realized_pnl=Decimal("5000")
        )
        
        tax_lot_tracker.add_lot(lot)
        
        details = tax_lot_tracker.get_lot_details("LOT_001")
        
        assert details["lot_id"] == "LOT_001"
        assert details["symbol"] == "BTC/USDT"
        assert details["quantity"] == "2.0"
        assert details["remaining_quantity"] == "1.5"
        assert details["closed_quantity"] == "0.5"
        assert details["cost_per_unit"] == "40000"
        assert details["total_cost"] == "80000"
        assert details["remaining_cost"] == "60000"
        assert details["status"] == "PARTIAL"
        assert details["realized_pnl"] == "5000"
    
    def test_get_lot_details_not_found(self, tax_lot_tracker):
        """Test getting details for non-existent lot."""
        details = tax_lot_tracker.get_lot_details("INVALID_LOT")
        assert details is None
    
    def test_lot_assignment_pnl_calculation(self):
        """Test P&L calculation in lot assignment."""
        assignment = LotAssignment(
            sale_id="SALE_001",
            symbol="BTC/USDT",
            sale_quantity=Decimal("2.0"),
            sale_price=Decimal("50000"),
            sale_date=datetime.now()
        )
        
        # Add lot assignments
        assignment.add_lot_assignment("LOT_001", Decimal("1.0"), Decimal("40000"))
        assignment.add_lot_assignment("LOT_002", Decimal("1.0"), Decimal("45000"))
        
        # Calculate P&L
        assignment.calculate_pnl()
        
        assert assignment.total_cost_basis == Decimal("85000")  # 40000 + 45000
        assert assignment.total_proceeds == Decimal("100000")  # 2 * 50000
        assert assignment.realized_pnl == Decimal("15000")  # 100000 - 85000