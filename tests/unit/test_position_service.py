"""
Unit tests for PositionService with PnL calculations.

Tests FIFO position management, PnL realization, and Decimal precision.
"""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from genesis.data.models import Position, Trade
from genesis.data.services import PositionService, RiskService


class TestPositionService:
    """Tests for position management and PnL calculation."""

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return MagicMock()

    @pytest.fixture
    def position_service(self, mock_db_session):
        """Create position service with mocked session."""
        return PositionService(mock_db_session)

    def create_trade(
        self, symbol="BTC/USDT", side="buy", qty="1.0", price="50000", fee="0.1"
    ):
        """Helper to create trade objects."""
        trade = MagicMock(spec=Trade)
        trade.symbol = symbol
        trade.side = side
        trade.qty = Decimal(qty)
        trade.price = Decimal(price)
        trade.fee_amount = Decimal(fee)
        trade.trade_time = datetime.now(UTC)
        trade.id = "test-trade-id"
        return trade

    def test_opening_long_position(self, position_service):
        """Test opening a new long position."""
        # Mock repository
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("0")
        mock_position.avg_entry_price = Decimal("0")
        mock_position.realised_pnl = Decimal("0")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )
        position_service.position_repo.update = MagicMock()

        # Buy 1 BTC at 50000
        trade = self.create_trade(side="buy", qty="1.0", price="50000", fee="10")

        position, realized_pnl = position_service.apply_trade(trade)

        # Verify position updated correctly
        position_service.position_repo.update.assert_called_once_with(
            symbol="BTC/USDT",
            qty=Decimal("1.0"),  # Long position
            avg_entry_price=Decimal("50000"),
            realised_pnl_delta=Decimal("-10"),  # Only fee
        )

        assert realized_pnl == Decimal("-10")  # Fee only

    def test_increasing_long_position(self, position_service):
        """Test adding to an existing long position."""
        # Mock existing long position
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("1.0")
        mock_position.avg_entry_price = Decimal("50000")
        mock_position.realised_pnl = Decimal("0")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )
        position_service.position_repo.update = MagicMock()

        # Buy 1 more BTC at 52000
        trade = self.create_trade(side="buy", qty="1.0", price="52000", fee="10")

        position, realized_pnl = position_service.apply_trade(trade)

        # Weighted average: (1 * 50000 + 1 * 52000) / 2 = 51000
        position_service.position_repo.update.assert_called_once_with(
            symbol="BTC/USDT",
            qty=Decimal("2.0"),
            avg_entry_price=Decimal("51000"),
            realised_pnl_delta=Decimal("-10"),  # Only fee
        )

    def test_reducing_long_position_with_profit(self, position_service):
        """Test partially closing a long position with profit."""
        # Mock existing long position
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("2.0")
        mock_position.avg_entry_price = Decimal("50000")
        mock_position.realised_pnl = Decimal("0")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )
        position_service.position_repo.update = MagicMock()

        # Sell 1 BTC at 55000 (profit)
        trade = self.create_trade(side="sell", qty="1.0", price="55000", fee="10")

        position, realized_pnl = position_service.apply_trade(trade)

        # Realized PnL: 1 * (55000 - 50000) - 10 = 4990
        position_service.position_repo.update.assert_called_once_with(
            symbol="BTC/USDT",
            qty=Decimal("1.0"),  # Reduced to 1 BTC
            avg_entry_price=Decimal("50000"),  # Unchanged (FIFO)
            realised_pnl_delta=Decimal("4990"),
        )

        assert realized_pnl == Decimal("4990")

    def test_closing_long_position_with_loss(self, position_service):
        """Test fully closing a long position with loss."""
        # Mock existing long position
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("1.0")
        mock_position.avg_entry_price = Decimal("50000")
        mock_position.realised_pnl = Decimal("0")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )
        position_service.position_repo.update = MagicMock()

        # Sell 1 BTC at 48000 (loss)
        trade = self.create_trade(side="sell", qty="1.0", price="48000", fee="10")

        position, realized_pnl = position_service.apply_trade(trade)

        # Realized PnL: 1 * (48000 - 50000) - 10 = -2010
        position_service.position_repo.update.assert_called_once_with(
            symbol="BTC/USDT",
            qty=Decimal("0"),  # Position closed
            avg_entry_price=Decimal("0"),  # Reset
            realised_pnl_delta=Decimal("-2010"),
        )

        assert realized_pnl == Decimal("-2010")

    def test_reversing_position_long_to_short(self, position_service):
        """Test reversing from long to short position."""
        # Mock existing long position
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("1.0")
        mock_position.avg_entry_price = Decimal("50000")
        mock_position.realised_pnl = Decimal("0")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )
        position_service.position_repo.update = MagicMock()

        # Sell 2 BTC at 52000 (close 1 long, open 1 short)
        trade = self.create_trade(side="sell", qty="2.0", price="52000", fee="20")

        position, realized_pnl = position_service.apply_trade(trade)

        # Realized PnL on closed long: 1 * (52000 - 50000) - 20 = 1980
        position_service.position_repo.update.assert_called_once_with(
            symbol="BTC/USDT",
            qty=Decimal("-1.0"),  # Now short 1 BTC
            avg_entry_price=Decimal("52000"),  # New short entry
            realised_pnl_delta=Decimal("1980"),
        )

        assert realized_pnl == Decimal("1980")

    def test_opening_short_position(self, position_service):
        """Test opening a new short position."""
        # Mock repository
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("0")
        mock_position.avg_entry_price = Decimal("0")
        mock_position.realised_pnl = Decimal("0")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )
        position_service.position_repo.update = MagicMock()

        # Sell 1 BTC at 50000 (short)
        trade = self.create_trade(side="sell", qty="1.0", price="50000", fee="10")

        position, realized_pnl = position_service.apply_trade(trade)

        # Verify short position created
        position_service.position_repo.update.assert_called_once_with(
            symbol="BTC/USDT",
            qty=Decimal("-1.0"),  # Short position
            avg_entry_price=Decimal("50000"),
            realised_pnl_delta=Decimal("-10"),  # Only fee
        )

    def test_covering_short_position_with_profit(self, position_service):
        """Test covering a short position with profit."""
        # Mock existing short position
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("-1.0")
        mock_position.avg_entry_price = Decimal("50000")
        mock_position.realised_pnl = Decimal("0")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )
        position_service.position_repo.update = MagicMock()

        # Buy 1 BTC at 48000 (profit on short)
        trade = self.create_trade(side="buy", qty="1.0", price="48000", fee="10")

        position, realized_pnl = position_service.apply_trade(trade)

        # Realized PnL: 1 * (50000 - 48000) - 10 = 1990
        position_service.position_repo.update.assert_called_once_with(
            symbol="BTC/USDT",
            qty=Decimal("0"),  # Position closed
            avg_entry_price=Decimal("0"),
            realised_pnl_delta=Decimal("1990"),
        )

        assert realized_pnl == Decimal("1990")

    def test_calculate_unrealized_pnl_long(self, position_service):
        """Test unrealized PnL calculation for long position."""
        # Mock long position
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("2.0")
        mock_position.avg_entry_price = Decimal("50000")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )

        # Current price 55000
        unrealized_pnl = position_service.calculate_unrealized_pnl(
            "BTC/USDT", Decimal("55000")
        )

        # Unrealized PnL: 2 * (55000 - 50000) = 10000
        assert unrealized_pnl == Decimal("10000")

    def test_calculate_unrealized_pnl_short(self, position_service):
        """Test unrealized PnL calculation for short position."""
        # Mock short position
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("-1.5")
        mock_position.avg_entry_price = Decimal("50000")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )

        # Current price 48000
        unrealized_pnl = position_service.calculate_unrealized_pnl(
            "BTC/USDT", Decimal("48000")
        )

        # Unrealized PnL: 1.5 * (50000 - 48000) = 3000
        assert unrealized_pnl == Decimal("3000")

    def test_calculate_unrealized_pnl_flat(self, position_service):
        """Test unrealized PnL when position is flat."""
        # Mock flat position
        mock_position = MagicMock(spec=Position)
        mock_position.qty = Decimal("0")
        mock_position.avg_entry_price = Decimal("0")

        position_service.position_repo.get_or_create = MagicMock(
            return_value=mock_position
        )

        unrealized_pnl = position_service.calculate_unrealized_pnl(
            "BTC/USDT", Decimal("55000")
        )

        assert unrealized_pnl == Decimal("0")


class TestRiskService:
    """Tests for risk calculation services."""

    def test_calculate_position_size_basic(self):
        """Test basic position size calculation."""
        position_size = RiskService.calculate_position_size(
            account_balance=Decimal("10000"),
            risk_percent=Decimal("2"),  # 2% risk
            stop_distance=Decimal("100"),  # $100 stop distance
            price=Decimal("50000"),
            lot_step=Decimal("0.001"),
        )

        # Risk amount: 10000 * 0.02 = 200
        # Position size: 200 / 100 = 2.0
        assert position_size == Decimal("2.000")

    def test_calculate_position_size_with_rounding(self):
        """Test position size rounding to lot step."""
        position_size = RiskService.calculate_position_size(
            account_balance=Decimal("10000"),
            risk_percent=Decimal("1.5"),
            stop_distance=Decimal("73"),
            price=Decimal("50000"),
            lot_step=Decimal("0.01"),
        )

        # Risk amount: 10000 * 0.015 = 150
        # Position size: 150 / 73 = 2.0547...
        # Rounded to 0.01: 2.05
        assert position_size == Decimal("2.05")

    def test_calculate_kelly_fraction_positive_edge(self):
        """Test Kelly fraction with positive edge."""
        kelly_fraction = RiskService.calculate_kelly_fraction(
            win_probability=Decimal("0.6"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("80"),
            kelly_multiplier=Decimal("0.25"),  # Quarter Kelly
        )

        # b = 100/80 = 1.25
        # f = (0.6 * 1.25 - 0.4) / 1.25 = 0.35 / 1.25 = 0.28
        # With 0.25 multiplier: 0.28 * 0.25 = 0.07
        assert kelly_fraction == Decimal("0.07")

    def test_calculate_kelly_fraction_negative_edge(self):
        """Test Kelly fraction with negative edge."""
        kelly_fraction = RiskService.calculate_kelly_fraction(
            win_probability=Decimal("0.4"),
            avg_win=Decimal("100"),
            avg_loss=Decimal("100"),
            kelly_multiplier=Decimal("0.25"),
        )

        # Negative edge should return 0
        assert kelly_fraction == Decimal("0")

    def test_calculate_kelly_fraction_max_cap(self):
        """Test Kelly fraction is capped at 25%."""
        kelly_fraction = RiskService.calculate_kelly_fraction(
            win_probability=Decimal("0.9"),
            avg_win=Decimal("200"),
            avg_loss=Decimal("50"),
            kelly_multiplier=Decimal("1.0"),  # Full Kelly
        )

        # Even with high edge, should be capped at 0.25
        assert kelly_fraction == Decimal("0.25")
