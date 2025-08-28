"""Unit tests for TaxOptimizer - Essential for compliance and tax calculations."""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from genesis.analytics.tax_optimizer import (
    TaxOptimizer,
    TaxLot,
    TaxMethod,
    TaxReport,
    CapitalGain,
    TaxCalculation,
)
from genesis.core.models import Position, Order, Account, Tier


class TestTaxOptimizer:
    """Test suite for tax lot optimization and calculations."""

    @pytest.fixture
    def sample_tax_lots(self):
        """Create sample tax lots for testing."""
        base_date = datetime.now(timezone.utc)
        return [
            TaxLot(
                lot_id="lot_1",
                position_id="pos_1",
                symbol="BTC/USDT",
                acquisition_date=base_date - timedelta(days=365),  # Long-term
                quantity=Decimal("0.5"),
                cost_basis=Decimal("20000.00"),  # $40k per BTC
                current_price=Decimal("45000.00"),
            ),
            TaxLot(
                lot_id="lot_2",
                position_id="pos_1",
                symbol="BTC/USDT",
                acquisition_date=base_date - timedelta(days=100),  # Short-term
                quantity=Decimal("0.3"),
                cost_basis=Decimal("12000.00"),  # $40k per BTC
                current_price=Decimal("45000.00"),
            ),
            TaxLot(
                lot_id="lot_3",
                position_id="pos_1",
                symbol="BTC/USDT",
                acquisition_date=base_date - timedelta(days=30),  # Short-term
                quantity=Decimal("0.2"),
                cost_basis=Decimal("9000.00"),  # $45k per BTC
                current_price=Decimal("45000.00"),
            ),
        ]

    @pytest.fixture
    def tax_optimizer(self):
        """Create TaxOptimizer instance."""
        return TaxOptimizer()

    def test_fifo_lot_selection(self, tax_optimizer, sample_tax_lots):
        """Test FIFO (First In First Out) lot selection method."""
        # Execute
        selected_lots = tax_optimizer.select_lots_fifo(
            lots=sample_tax_lots, quantity_to_sell=Decimal("0.6")
        )

        # Verify
        assert len(selected_lots) == 2
        assert selected_lots[0].lot_id == "lot_1"  # Oldest first
        assert selected_lots[0].quantity_to_sell == Decimal("0.5")
        assert selected_lots[1].lot_id == "lot_2"
        assert selected_lots[1].quantity_to_sell == Decimal("0.1")

    def test_lifo_lot_selection(self, tax_optimizer, sample_tax_lots):
        """Test LIFO (Last In First Out) lot selection method."""
        # Execute
        selected_lots = tax_optimizer.select_lots_lifo(
            lots=sample_tax_lots, quantity_to_sell=Decimal("0.4")
        )

        # Verify
        assert len(selected_lots) == 2
        assert selected_lots[0].lot_id == "lot_3"  # Newest first
        assert selected_lots[0].quantity_to_sell == Decimal("0.2")
        assert selected_lots[1].lot_id == "lot_2"
        assert selected_lots[1].quantity_to_sell == Decimal("0.2")

    def test_hifo_lot_selection(self, tax_optimizer, sample_tax_lots):
        """Test HIFO (Highest In First Out) lot selection method."""
        # Execute
        selected_lots = tax_optimizer.select_lots_hifo(
            lots=sample_tax_lots, quantity_to_sell=Decimal("0.3")
        )

        # Verify
        assert len(selected_lots) == 1
        assert selected_lots[0].lot_id == "lot_3"  # Highest cost basis first
        assert selected_lots[0].quantity_to_sell == Decimal("0.2")
        # Would continue with lot_1 or lot_2 (same cost basis) for remaining 0.1

    def test_capital_gains_calculation_long_term(self, tax_optimizer):
        """Test long-term capital gains calculation."""
        # Setup
        lot = TaxLot(
            lot_id="lot_1",
            position_id="pos_1",
            symbol="BTC/USDT",
            acquisition_date=datetime.now(timezone.utc) - timedelta(days=400),
            quantity=Decimal("1.0"),
            cost_basis=Decimal("30000.00"),
            current_price=Decimal("45000.00"),
        )

        # Execute
        gain = tax_optimizer.calculate_capital_gain(
            lot=lot, sale_price=Decimal("45000.00"), quantity_sold=Decimal("1.0")
        )

        # Verify
        assert gain.gain_type == "LONG_TERM"
        assert gain.cost_basis == Decimal("30000.00")
        assert gain.sale_proceeds == Decimal("45000.00")
        assert gain.capital_gain == Decimal("15000.00")
        assert gain.holding_period_days >= 365

    def test_capital_gains_calculation_short_term(self, tax_optimizer):
        """Test short-term capital gains calculation."""
        # Setup
        lot = TaxLot(
            lot_id="lot_1",
            position_id="pos_1",
            symbol="BTC/USDT",
            acquisition_date=datetime.now(timezone.utc) - timedelta(days=100),
            quantity=Decimal("1.0"),
            cost_basis=Decimal("40000.00"),
            current_price=Decimal("45000.00"),
        )

        # Execute
        gain = tax_optimizer.calculate_capital_gain(
            lot=lot, sale_price=Decimal("45000.00"), quantity_sold=Decimal("1.0")
        )

        # Verify
        assert gain.gain_type == "SHORT_TERM"
        assert gain.cost_basis == Decimal("40000.00")
        assert gain.sale_proceeds == Decimal("45000.00")
        assert gain.capital_gain == Decimal("5000.00")
        assert gain.holding_period_days < 365

    def test_capital_loss_calculation(self, tax_optimizer):
        """Test capital loss calculation."""
        # Setup
        lot = TaxLot(
            lot_id="lot_1",
            position_id="pos_1",
            symbol="BTC/USDT",
            acquisition_date=datetime.now(timezone.utc) - timedelta(days=200),
            quantity=Decimal("1.0"),
            cost_basis=Decimal("50000.00"),
            current_price=Decimal("45000.00"),
        )

        # Execute
        gain = tax_optimizer.calculate_capital_gain(
            lot=lot, sale_price=Decimal("45000.00"), quantity_sold=Decimal("1.0")
        )

        # Verify
        assert gain.capital_gain == Decimal("-5000.00")
        assert gain.is_loss is True

    def test_tax_optimization_minimize_taxes(self, tax_optimizer, sample_tax_lots):
        """Test tax optimization to minimize current year taxes."""
        # Execute
        optimal_lots = tax_optimizer.optimize_for_tax_minimization(
            lots=sample_tax_lots,
            quantity_to_sell=Decimal("0.5"),
            current_year_gains=Decimal("10000.00"),  # Already have gains
        )

        # Verify
        # Should prefer lots with losses or smallest gains
        assert len(optimal_lots) > 0
        total_quantity = sum(lot.quantity_to_sell for lot in optimal_lots)
        assert total_quantity == Decimal("0.5")

    def test_wash_sale_detection(self, tax_optimizer):
        """Test wash sale rule detection (30-day rule)."""
        # Setup
        base_date = datetime.now(timezone.utc)

        sale = Order(
            order_id="sell_1",
            symbol="BTC/USDT",
            side="sell",
            quantity=Decimal("1.0"),
            price=Decimal("40000.00"),
            executed_at=base_date - timedelta(days=20),
        )

        purchase = Order(
            order_id="buy_1",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("1.0"),
            price=Decimal("41000.00"),
            executed_at=base_date,
        )

        # Execute
        is_wash_sale = tax_optimizer.is_wash_sale(sale, purchase)

        # Verify
        assert is_wash_sale is True  # Within 30 days

    def test_year_end_tax_report_generation(self, tax_optimizer, sample_tax_lots):
        """Test year-end tax report generation."""
        # Setup
        trades = [
            CapitalGain(
                lot_id="lot_1",
                symbol="BTC/USDT",
                acquisition_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
                sale_date=datetime(2024, 11, 20, tzinfo=timezone.utc),
                cost_basis=Decimal("30000.00"),
                sale_proceeds=Decimal("45000.00"),
                capital_gain=Decimal("15000.00"),
                gain_type="LONG_TERM",
                holding_period_days=309,
            ),
            CapitalGain(
                lot_id="lot_2",
                symbol="ETH/USDT",
                acquisition_date=datetime(2024, 8, 1, tzinfo=timezone.utc),
                sale_date=datetime(2024, 10, 15, tzinfo=timezone.utc),
                cost_basis=Decimal("2000.00"),
                sale_proceeds=Decimal("2500.00"),
                capital_gain=Decimal("500.00"),
                gain_type="SHORT_TERM",
                holding_period_days=75,
            ),
        ]

        # Execute
        report = tax_optimizer.generate_year_end_report(year=2024, trades=trades)

        # Verify
        assert report.tax_year == 2024
        assert report.total_long_term_gains == Decimal("15000.00")
        assert report.total_short_term_gains == Decimal("500.00")
        assert report.total_gains == Decimal("15500.00")
        assert len(report.trades) == 2

    def test_tax_loss_harvesting(self, tax_optimizer):
        """Test tax loss harvesting strategy."""
        # Setup - Mix of gains and losses
        positions = [
            TaxLot(
                lot_id="lot_1",
                position_id="pos_1",
                symbol="BTC/USDT",
                acquisition_date=datetime.now(timezone.utc) - timedelta(days=200),
                quantity=Decimal("1.0"),
                cost_basis=Decimal("50000.00"),
                current_price=Decimal("45000.00"),  # $5k loss
            ),
            TaxLot(
                lot_id="lot_2",
                position_id="pos_2",
                symbol="ETH/USDT",
                acquisition_date=datetime.now(timezone.utc) - timedelta(days=180),
                quantity=Decimal("10.0"),
                cost_basis=Decimal("20000.00"),
                current_price=Decimal("25000.00"),  # $5k gain
            ),
        ]

        # Execute
        harvest_recommendations = (
            tax_optimizer.identify_tax_loss_harvesting_opportunities(
                positions=positions,
                current_year_gains=Decimal("10000.00"),
                max_loss_to_harvest=Decimal("3000.00"),
            )
        )

        # Verify
        assert len(harvest_recommendations) > 0
        assert harvest_recommendations[0].lot_id == "lot_1"  # Loss position
        assert harvest_recommendations[0].potential_tax_savings > Decimal("0")

    def test_decimal_precision_in_tax_calculations(self, tax_optimizer):
        """Test that Decimal precision is maintained in all tax calculations."""
        # Setup with precise decimal values
        lot = TaxLot(
            lot_id="lot_1",
            position_id="pos_1",
            symbol="BTC/USDT",
            acquisition_date=datetime.now(timezone.utc) - timedelta(days=200),
            quantity=Decimal("0.123456789"),
            cost_basis=Decimal("5432.109876543"),
            current_price=Decimal("44000.00"),
        )

        # Execute
        gain = tax_optimizer.calculate_capital_gain(
            lot=lot, sale_price=Decimal("44000.00"), quantity_sold=lot.quantity
        )

        # Verify all values are Decimal
        assert isinstance(gain.cost_basis, Decimal)
        assert isinstance(gain.sale_proceeds, Decimal)
        assert isinstance(gain.capital_gain, Decimal)

        # Verify calculation precision
        expected_proceeds = lot.quantity * Decimal("44000.00")
        expected_gain = expected_proceeds - lot.cost_basis
        assert gain.sale_proceeds == expected_proceeds
        assert gain.capital_gain == expected_gain

    def test_multi_currency_tax_calculations(self, tax_optimizer):
        """Test tax calculations for multi-currency positions."""
        # Setup
        lots = [
            TaxLot(
                lot_id="lot_1",
                position_id="pos_1",
                symbol="BTC/USDT",
                acquisition_date=datetime.now(timezone.utc) - timedelta(days=200),
                quantity=Decimal("1.0"),
                cost_basis=Decimal("40000.00"),
                current_price=Decimal("45000.00"),
            ),
            TaxLot(
                lot_id="lot_2",
                position_id="pos_2",
                symbol="ETH/USDT",
                acquisition_date=datetime.now(timezone.utc) - timedelta(days=150),
                quantity=Decimal("10.0"),
                cost_basis=Decimal("20000.00"),
                current_price=Decimal("25000.00"),
            ),
            TaxLot(
                lot_id="lot_3",
                position_id="pos_3",
                symbol="EUR/USD",
                acquisition_date=datetime.now(timezone.utc) - timedelta(days=100),
                quantity=Decimal("10000.0"),
                cost_basis=Decimal("11000.00"),
                current_price=Decimal("1.08"),
            ),
        ]

        # Execute
        portfolio_tax_summary = tax_optimizer.calculate_portfolio_tax_impact(lots)

        # Verify
        assert portfolio_tax_summary.total_unrealized_gains > Decimal("0")
        assert len(portfolio_tax_summary.by_currency) >= 2
        assert all(
            isinstance(v, Decimal) for v in portfolio_tax_summary.by_currency.values()
        )

    def test_specific_lot_identification(self, tax_optimizer, sample_tax_lots):
        """Test specific lot identification method for tax optimization."""
        # Execute
        selected_lots = tax_optimizer.select_specific_lots(
            lots=sample_tax_lots,
            lot_ids=["lot_2", "lot_3"],
            quantity_to_sell=Decimal("0.4"),
        )

        # Verify
        assert len(selected_lots) == 2
        assert selected_lots[0].lot_id == "lot_2"
        assert selected_lots[1].lot_id == "lot_3"
        assert sum(lot.quantity_to_sell for lot in selected_lots) == Decimal("0.4")
