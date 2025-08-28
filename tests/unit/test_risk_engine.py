"""
Unit tests for the risk engine module.

Tests position sizing, stop-loss calculations, P&L tracking,
and risk limit enforcement with 100% coverage for money paths.
"""

from datetime import datetime
from decimal import Decimal

import pytest

from genesis.core.exceptions import (
    DailyLossLimitReached,
    InsufficientBalance,
    MinimumPositionSize,
    RiskLimitExceeded,
    TierViolation,
)
from genesis.core.models import (
    Account,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.engine.risk_engine import RiskEngine


@pytest.fixture
def sniper_account():
    """Create a test account with Sniper tier."""
    return Account(
        account_id="test-account-1",
        balance_usdt=Decimal("500"),
        tier=TradingTier.SNIPER,
    )


@pytest.fixture
def hunter_account():
    """Create a test account with Hunter tier."""
    return Account(
        account_id="test-account-2",
        balance_usdt=Decimal("5000"),
        tier=TradingTier.HUNTER,
    )


@pytest.fixture
def trading_session(sniper_account):
    """Create a test trading session."""
    return TradingSession(
        session_id="test-session-1",
        account_id=sniper_account.account_id,
        session_date=datetime.now(),
        starting_balance=sniper_account.balance_usdt,
        current_balance=sniper_account.balance_usdt,
        daily_loss_limit=Decimal("25"),
    )


@pytest.fixture
def risk_engine(sniper_account, trading_session):
    """Create a risk engine instance."""
    return RiskEngine(sniper_account, trading_session)


class TestPositionSizing:
    """Test position sizing calculations."""

    def test_calculate_position_size_basic(self, risk_engine):
        """Test basic position size calculation."""
        # 5% of $500 = $25 risk
        # With 2% stop loss from $100, stop at $98
        # Risk per unit = $2
        # Position size = $25 / $2 = 12.5 units
        # Position value = 12.5 * $100 = $1250 (but limited by balance)

        size = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=Decimal("100")
        )

        # Should be limited by the 5% rule
        assert size > 0
        assert size * Decimal("100") <= risk_engine.account.balance_usdt

    def test_position_size_with_custom_risk(self, risk_engine):
        """Test position sizing with custom risk percentage."""
        size = risk_engine.calculate_position_size(
            symbol="BTC/USDT",
            entry_price=Decimal("100"),
            custom_risk_percent=Decimal("2"),  # 2% instead of 5%
        )

        # Should use 2% risk
        risk_amount = risk_engine.account.balance_usdt * Decimal("0.02")
        assert size > 0
        assert size * Decimal("100") <= risk_engine.account.balance_usdt

    def test_position_size_with_custom_stop_loss(self, risk_engine):
        """Test position sizing with custom stop loss."""
        entry_price = Decimal("100")
        stop_loss = Decimal("95")  # 5% stop loss

        size = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=entry_price, stop_loss_price=stop_loss
        )

        # Risk per unit = $5
        # Risk amount = $500 * 5% = $25
        # Position size = $25 / $5 = 5 units
        assert size == Decimal("5")

    def test_minimum_position_size_validation(self, risk_engine):
        """Test minimum position size enforcement."""
        # Set balance just below minimum position size
        risk_engine.account.balance_usdt = Decimal("9")

        # Try to create a position with insufficient balance
        with pytest.raises(MinimumPositionSize) as exc_info:
            risk_engine.calculate_position_size(
                symbol="BTC/USDT", entry_price=Decimal("100")
            )

        assert exc_info.value.minimum_size == Decimal("10")

    def test_insufficient_balance(self, risk_engine):
        """Test insufficient balance handling."""
        risk_engine.account.balance_usdt = Decimal("5")  # Below minimum position size

        with pytest.raises(MinimumPositionSize) as exc_info:
            risk_engine.calculate_position_size(
                symbol="BTC/USDT", entry_price=Decimal("100")
            )

        # Balance too low to meet minimum position size
        assert exc_info.value.minimum_size == Decimal("10")

    def test_position_size_decimal_precision(self, risk_engine):
        """Test Decimal precision in position sizing."""
        size = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=Decimal("12345.6789")
        )

        # Should have 8 decimal places (crypto standard)
        assert isinstance(size, Decimal)
        assert size.as_tuple().exponent >= -8


class TestStopLossCalculation:
    """Test stop-loss calculations."""

    def test_calculate_stop_loss_long(self, risk_engine):
        """Test stop loss calculation for long position."""
        entry_price = Decimal("100")
        stop_loss = risk_engine.calculate_stop_loss(entry_price, PositionSide.LONG)

        # Default 2% below entry for long
        expected = Decimal("98")
        assert stop_loss == expected

    def test_calculate_stop_loss_short(self, risk_engine):
        """Test stop loss calculation for short position."""
        entry_price = Decimal("100")
        stop_loss = risk_engine.calculate_stop_loss(entry_price, PositionSide.SHORT)

        # Default 2% above entry for short
        expected = Decimal("102")
        assert stop_loss == expected

    def test_stop_loss_custom_percentage(self, risk_engine):
        """Test stop loss with custom percentage."""
        entry_price = Decimal("100")
        stop_loss = risk_engine.calculate_stop_loss(
            entry_price, PositionSide.LONG, stop_loss_percent=Decimal("5")
        )

        expected = Decimal("95")
        assert stop_loss == expected

    def test_stop_loss_decimal_precision(self, risk_engine):
        """Test stop loss decimal precision."""
        entry_price = Decimal("12345.6789")
        stop_loss = risk_engine.calculate_stop_loss(entry_price, PositionSide.LONG)

        # Should have 8 decimal places
        assert isinstance(stop_loss, Decimal)
        assert stop_loss.as_tuple().exponent == -8


class TestPnLCalculations:
    """Test P&L calculation methods."""

    def test_calculate_pnl_long_profit(self, risk_engine):
        """Test P&L calculation for profitable long position."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )

        pnl = risk_engine.calculate_pnl(position, Decimal("110"))

        assert pnl["pnl_dollars"] == Decimal("10.00")
        assert pnl["pnl_percent"] == Decimal("10.0000")

    def test_calculate_pnl_long_loss(self, risk_engine):
        """Test P&L calculation for losing long position."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )

        pnl = risk_engine.calculate_pnl(position, Decimal("90"))

        assert pnl["pnl_dollars"] == Decimal("-10.00")
        assert pnl["pnl_percent"] == Decimal("-10.0000")

    def test_calculate_pnl_short_profit(self, risk_engine):
        """Test P&L calculation for profitable short position."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )

        # Price goes down = profit for short
        pnl = risk_engine.calculate_pnl(position, Decimal("90"))

        assert pnl["pnl_dollars"] == Decimal("10.00")
        assert pnl["pnl_percent"] == Decimal("10.0000")

    def test_calculate_pnl_short_loss(self, risk_engine):
        """Test P&L calculation for losing short position."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )

        # Price goes up = loss for short
        pnl = risk_engine.calculate_pnl(position, Decimal("110"))

        assert pnl["pnl_dollars"] == Decimal("-10.00")
        assert pnl["pnl_percent"] == Decimal("-10.0000")

    def test_pnl_decimal_precision(self, risk_engine):
        """Test P&L decimal precision."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("12345.6789"),
            quantity=Decimal("0.12345678"),
            dollar_value=Decimal("1524.5061"),
        )

        pnl = risk_engine.calculate_pnl(position, Decimal("12400"))

        # pnl_dollars should have 2 decimal places (USD)
        assert pnl["pnl_dollars"].as_tuple().exponent == -2
        # pnl_percent should have 4 decimal places
        assert pnl["pnl_percent"].as_tuple().exponent == -4


class TestRiskValidation:
    """Test risk validation and limit enforcement."""

    def test_validate_order_risk_pass(self, risk_engine):
        """Test successful order risk validation."""
        # Should pass all checks - position value must be >= $10
        risk_engine.validate_order_risk(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.2"),  # $20 position value
            entry_price=Decimal("100"),
        )

    def test_validate_minimum_position_size(self, risk_engine):
        """Test minimum position size validation."""
        with pytest.raises(MinimumPositionSize):
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.00001"),  # Too small
                entry_price=Decimal("100"),
            )

    def test_validate_insufficient_balance(self, risk_engine):
        """Test insufficient balance validation."""
        with pytest.raises(InsufficientBalance):
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("10"),  # $1000 position
                entry_price=Decimal("100"),
            )

    def test_validate_position_risk_exceeded(self, risk_engine):
        """Test position risk percentage limit."""
        risk_engine.account.balance_usdt = Decimal("1000")

        with pytest.raises(RiskLimitExceeded) as exc_info:
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("1"),  # $100 = 10% of balance
                entry_price=Decimal("100"),
            )

        assert exc_info.value.limit_type == "position_risk"
        assert exc_info.value.limit_value == Decimal("5")  # 5% limit

    def test_validate_daily_loss_limit(self, risk_engine):
        """Test daily loss limit validation."""
        # Set session to have reached daily limit
        risk_engine.session.realized_pnl = Decimal("-25")

        with pytest.raises(DailyLossLimitReached) as exc_info:
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.2"),  # $20 position value >= minimum
                entry_price=Decimal("100"),
            )

        assert exc_info.value.daily_limit == Decimal("25")

    def test_validate_max_positions(self, risk_engine):
        """Test maximum positions limit."""
        # Add max positions for Sniper tier (1)
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("0.01"),
            dollar_value=Decimal("1"),
        )
        risk_engine.add_position(position)

        with pytest.raises(RiskLimitExceeded) as exc_info:
            risk_engine.validate_order_risk(
                symbol="ETH/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.2"),  # $20 position value >= minimum
                entry_price=Decimal("100"),
            )

        assert exc_info.value.limit_type == "max_positions"


class TestDailyLossLimit:
    """Test daily loss limit enforcement."""

    def test_session_daily_limit_not_reached(self, trading_session):
        """Test when daily limit is not reached."""
        trading_session.realized_pnl = Decimal("-10")
        assert not trading_session.is_daily_limit_reached()

    def test_session_daily_limit_reached(self, trading_session):
        """Test when daily limit is reached."""
        trading_session.realized_pnl = Decimal("-25")
        assert trading_session.is_daily_limit_reached()

    def test_session_daily_limit_exceeded(self, trading_session):
        """Test when daily limit is exceeded."""
        trading_session.realized_pnl = Decimal("-30")
        assert trading_session.is_daily_limit_reached()

    def test_prevent_exceeding_limits_with_loss(self, risk_engine):
        """Test prevent_exceeding_limits with daily loss."""
        risk_engine.session.realized_pnl = Decimal("-25")
        assert not risk_engine.prevent_exceeding_limits()

    def test_prevent_exceeding_limits_max_positions(self, risk_engine):
        """Test prevent_exceeding_limits with max positions."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("0.01"),
            dollar_value=Decimal("1"),
        )
        risk_engine.add_position(position)

        assert not risk_engine.prevent_exceeding_limits()


class TestPositionManagement:
    """Test position tracking and management."""

    def test_add_position(self, risk_engine):
        """Test adding a position."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )

        risk_engine.add_position(position)
        assert position.position_id in risk_engine.positions
        assert len(risk_engine.positions) == 1

    def test_remove_position(self, risk_engine):
        """Test removing a position."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )

        risk_engine.add_position(position)
        risk_engine.remove_position("pos-1")

        assert "pos-1" not in risk_engine.positions
        assert len(risk_engine.positions) == 0

    def test_update_all_pnl(self, risk_engine):
        """Test updating P&L for all positions."""
        position1 = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )
        position2 = Position(
            position_id="pos-2",
            account_id="test-account",
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50"),
            quantity=Decimal("2"),
            dollar_value=Decimal("100"),
        )

        risk_engine.add_position(position1)
        risk_engine.add_position(position2)

        price_updates = {"BTC/USDT": Decimal("110"), "ETH/USDT": Decimal("45")}

        risk_engine.update_all_pnl(price_updates)

        # Check position 1 (long, price up = profit)
        assert risk_engine.positions["pos-1"].pnl_dollars == Decimal("10")
        # Check position 2 (short, price down = profit)
        assert risk_engine.positions["pos-2"].pnl_dollars == Decimal("10")

    def test_get_total_exposure(self, risk_engine):
        """Test total exposure calculation."""
        position1 = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )
        position2 = Position(
            position_id="pos-2",
            account_id="test-account",
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50"),
            quantity=Decimal("2"),
            dollar_value=Decimal("100"),
        )

        risk_engine.add_position(position1)
        risk_engine.add_position(position2)

        total = risk_engine.get_total_exposure()
        assert total == Decimal("200.00")

    def test_get_total_pnl(self, risk_engine):
        """Test total P&L calculation."""
        position1 = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
            pnl_dollars=Decimal("10"),
            pnl_percent=Decimal("10"),
        )
        position2 = Position(
            position_id="pos-2",
            account_id="test-account",
            symbol="ETH/USDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("50"),
            quantity=Decimal("2"),
            dollar_value=Decimal("100"),
            pnl_dollars=Decimal("-5"),
            pnl_percent=Decimal("-5"),
        )

        risk_engine.add_position(position1)
        risk_engine.add_position(position2)

        total_pnl = risk_engine.get_total_pnl()
        assert total_pnl["total_pnl_dollars"] == Decimal("5.00")
        assert total_pnl["total_pnl_percent"] == Decimal("2.5000")  # 5/200 * 100


class TestTierEnforcement:
    """Test tier-based feature enforcement."""

    @pytest.mark.asyncio
    async def test_correlation_requires_hunter_tier(self, risk_engine):
        """Test that correlation calculation requires Hunter tier."""
        # Sniper tier should not have access
        with pytest.raises(TierViolation) as exc_info:
            await risk_engine.calculate_position_correlations()

        assert exc_info.value.required_tier == "HUNTER"
        assert exc_info.value.current_tier == "SNIPER"

    @pytest.mark.asyncio
    async def test_correlation_allowed_hunter_tier(self, hunter_account):
        """Test that Hunter tier can access correlation."""
        risk_engine = RiskEngine(hunter_account)

        position1 = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("100"),
            quantity=Decimal("1"),
            dollar_value=Decimal("100"),
        )
        position2 = Position(
            position_id="pos-2",
            account_id="test-account",
            symbol="ETH/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50"),
            quantity=Decimal("2"),
            dollar_value=Decimal("100"),
        )

        risk_engine.add_position(position1)
        risk_engine.add_position(position2)

        correlations = await risk_engine.calculate_position_correlations()
        assert len(correlations) == 1  # One pair of positions


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_balance(self):
        """Test with zero account balance."""
        account = Account(
            account_id="test-zero", balance_usdt=Decimal("0"), tier=TradingTier.SNIPER
        )
        risk_engine = RiskEngine(account)

        with pytest.raises(InsufficientBalance):
            risk_engine.calculate_position_size(
                symbol="BTC/USDT", entry_price=Decimal("100")
            )

    def test_minimum_balance(self):
        """Test with minimum viable balance."""
        account = Account(
            account_id="test-min",
            balance_usdt=Decimal("10"),  # Exactly minimum position size
            tier=TradingTier.SNIPER,
        )
        risk_engine = RiskEngine(account)

        # With $10 balance and 5% risk, should meet minimum position size
        size = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=Decimal("100")
        )
        # Should allow minimum position size
        assert size * Decimal("100") >= Decimal("10")

    def test_no_session(self, sniper_account):
        """Test risk engine without session."""
        risk_engine = RiskEngine(sniper_account, session=None)

        # Should still work for basic operations
        size = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=Decimal("100")
        )
        assert size > 0

        # Daily limit check should pass without session
        risk_engine.validate_order_risk(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.2"),  # $20 position value >= minimum
            entry_price=Decimal("100"),
        )

    def test_extreme_prices(self, risk_engine):
        """Test with extreme price values."""
        # Very high price
        size = risk_engine.calculate_position_size(
            symbol="BTC/USDT", entry_price=Decimal("999999.99")
        )
        assert size > 0
        assert size * Decimal("999999.99") <= risk_engine.account.balance_usdt

        # Very low price
        size = risk_engine.calculate_position_size(
            symbol="SHIB/USDT", entry_price=Decimal("0.00001")
        )
        assert size > 0

    def test_decimal_overflow_protection(self, risk_engine):
        """Test protection against decimal overflow."""
        position = Position(
            position_id="pos-1",
            account_id="test-account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("0.00000001"),
            quantity=Decimal("999999999999"),
            dollar_value=Decimal("10"),
        )

        # Should handle extreme values without overflow
        pnl = risk_engine.calculate_pnl(position, Decimal("0.00000002"))
        assert isinstance(pnl["pnl_dollars"], Decimal)
        assert isinstance(pnl["pnl_percent"], Decimal)
