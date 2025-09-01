"""
Property-based tests for position calculations using Hypothesis.

Tests mathematical invariants and properties that must hold
for all valid inputs in position sizing and risk calculations.
"""

from decimal import Decimal, ROUND_DOWN

import pytest
from hypothesis import assume, given, settings, strategies as st

from genesis.core.models import (
    Account,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.engine.risk_engine import RiskEngine


# Custom strategies for generating test data
@st.composite
def decimal_price(draw, min_value=Decimal("0.00000001"), max_value=Decimal("1000000")):
    """Generate valid price decimals."""
    # Generate a float and convert to Decimal with 8 decimal places
    value = draw(st.floats(min_value=float(min_value), max_value=float(max_value)))
    return Decimal(str(value)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


@st.composite
def decimal_quantity(draw, min_value=Decimal("0.00000001"), max_value=Decimal("10000")):
    """Generate valid quantity decimals."""
    value = draw(st.floats(min_value=float(min_value), max_value=float(max_value)))
    return Decimal(str(value)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


@st.composite
def account_balance(draw, min_value=Decimal("10"), max_value=Decimal("1000000")):
    """Generate valid account balances."""
    value = draw(st.floats(min_value=float(min_value), max_value=float(max_value)))
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)


@st.composite
def risk_percentage(draw):
    """Generate valid risk percentages (0.1% to 10%)."""
    value = draw(st.floats(min_value=0.1, max_value=10.0))
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)


@st.composite
def trading_account(draw):
    """Generate a valid trading account."""
    balance = draw(account_balance())
    tier = draw(st.sampled_from(list(TradingTier)))
    return Account(
        account_id="test-account",
        balance_usdt=balance,
        tier=tier,
    )


@st.composite
def trading_position(draw):
    """Generate a valid trading position."""
    symbol = draw(st.sampled_from(["BTC/USDT", "ETH/USDT", "BNB/USDT"]))
    side = draw(st.sampled_from(list(PositionSide)))
    entry_price = draw(decimal_price())
    quantity = draw(decimal_quantity())
    
    return Position(
        position_id="test-position",
        account_id="test-account",
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        dollar_value=entry_price * quantity,
    )


class TestPositionSizingProperties:
    """Property tests for position sizing calculations."""

    @given(
        account=trading_account(),
        entry_price=decimal_price(),
        risk_percent=risk_percentage(),
    )
    @settings(max_examples=100, deadline=1000)
    def test_position_size_never_exceeds_balance(self, account, entry_price, risk_percent):
        """Position size * price should never exceed account balance."""
        risk_engine = RiskEngine(account)
        
        try:
            size = risk_engine.calculate_position_size(
                symbol="BTC/USDT",
                entry_price=entry_price,
                custom_risk_percent=risk_percent,
            )
            
            # Property: position value <= account balance
            position_value = size * entry_price
            assert position_value <= account.balance_usdt
        except (MinimumPositionSize, InsufficientBalance):
            # These exceptions are valid for certain inputs
            pass

    @given(
        account=trading_account(),
        entry_price=decimal_price(),
        stop_loss_percent=st.floats(min_value=0.1, max_value=20.0),
    )
    @settings(max_examples=100, deadline=1000)
    def test_position_size_respects_risk_limits(self, account, entry_price, stop_loss_percent):
        """Position size should respect risk per trade limits."""
        risk_engine = RiskEngine(account)
        stop_loss_percent = Decimal(str(stop_loss_percent))
        
        # Calculate stop loss price
        stop_loss_price = entry_price * (Decimal("1") - stop_loss_percent / Decimal("100"))
        
        try:
            size = risk_engine.calculate_position_size(
                symbol="BTC/USDT",
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
            )
            
            # Property: maximum loss should not exceed risk limit
            max_loss = size * (entry_price - stop_loss_price)
            risk_limit = account.balance_usdt * Decimal("0.05")  # 5% default risk
            assert max_loss <= risk_limit * Decimal("1.01")  # Allow 1% tolerance for rounding
        except (MinimumPositionSize, InsufficientBalance, ValueError):
            pass

    @given(
        account=trading_account(),
        entry_price=decimal_price(),
    )
    @settings(max_examples=100, deadline=1000)
    def test_position_size_is_deterministic(self, account, entry_price):
        """Same inputs should always produce same position size."""
        risk_engine = RiskEngine(account)
        
        try:
            size1 = risk_engine.calculate_position_size(
                symbol="BTC/USDT",
                entry_price=entry_price,
            )
            size2 = risk_engine.calculate_position_size(
                symbol="BTC/USDT",
                entry_price=entry_price,
            )
            
            # Property: deterministic function
            assert size1 == size2
        except (MinimumPositionSize, InsufficientBalance):
            pass

    @given(
        account=trading_account(),
        prices=st.lists(
            decimal_price(min_value=Decimal("100"), max_value=Decimal("100000")),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=50, deadline=1000)
    def test_position_size_inversely_proportional_to_price(self, account, prices):
        """Higher prices should result in smaller position sizes for same risk."""
        risk_engine = RiskEngine(account)
        
        sizes = []
        for price in prices:
            try:
                size = risk_engine.calculate_position_size(
                    symbol="BTC/USDT",
                    entry_price=price,
                )
                sizes.append((price, size))
            except (MinimumPositionSize, InsufficientBalance):
                pass
        
        if len(sizes) >= 2:
            # Property: inverse relationship between price and size
            for i in range(len(sizes) - 1):
                price1, size1 = sizes[i]
                price2, size2 = sizes[i + 1]
                
                if price1 < price2:
                    # Higher price should have smaller or equal size
                    assert size1 >= size2 * Decimal("0.95")  # 5% tolerance


class TestPnLCalculationProperties:
    """Property tests for P&L calculations."""

    @given(
        position=trading_position(),
        price_multiplier=st.floats(min_value=0.5, max_value=2.0),
    )
    @settings(max_examples=100, deadline=1000)
    def test_pnl_zero_at_entry_price(self, position, price_multiplier):
        """P&L should be zero when current price equals entry price."""
        risk_engine = RiskEngine(Account(
            account_id="test",
            balance_usdt=Decimal("1000"),
            tier=TradingTier.SNIPER,
        ))
        
        # Test at entry price
        pnl = risk_engine.calculate_pnl(position, position.entry_price)
        
        # Property: P&L is zero at entry (ignoring fees)
        assert abs(pnl["pnl_dollars"]) < Decimal("0.01")
        assert abs(pnl["pnl_percent"]) < Decimal("0.01")

    @given(
        position=trading_position(),
        current_price=decimal_price(),
    )
    @settings(max_examples=100, deadline=1000)
    def test_pnl_sign_consistency(self, position, current_price):
        """P&L sign should be consistent with position direction and price movement."""
        risk_engine = RiskEngine(Account(
            account_id="test",
            balance_usdt=Decimal("1000"),
            tier=TradingTier.SNIPER,
        ))
        
        pnl = risk_engine.calculate_pnl(position, current_price)
        
        # Property: P&L sign consistency
        if position.side == PositionSide.LONG:
            if current_price > position.entry_price:
                assert pnl["pnl_dollars"] > 0  # Profit on long when price goes up
            elif current_price < position.entry_price:
                assert pnl["pnl_dollars"] < 0  # Loss on long when price goes down
        else:  # SHORT
            if current_price < position.entry_price:
                assert pnl["pnl_dollars"] > 0  # Profit on short when price goes down
            elif current_price > position.entry_price:
                assert pnl["pnl_dollars"] < 0  # Loss on short when price goes up

    @given(
        position=trading_position(),
        prices=st.lists(decimal_price(), min_size=3, max_size=10),
    )
    @settings(max_examples=50, deadline=1000)
    def test_pnl_linearity(self, position, prices):
        """P&L should be linear with price changes for same position."""
        risk_engine = RiskEngine(Account(
            account_id="test",
            balance_usdt=Decimal("1000"),
            tier=TradingTier.SNIPER,
        ))
        
        pnls = []
        for price in prices:
            pnl = risk_engine.calculate_pnl(position, price)
            pnls.append((price, pnl["pnl_dollars"]))
        
        # Property: Linear relationship between price change and P&L
        if len(pnls) >= 3:
            # Sort by price
            pnls.sort(key=lambda x: x[0])
            
            # Check linearity (constant rate of change)
            for i in range(len(pnls) - 2):
                price1, pnl1 = pnls[i]
                price2, pnl2 = pnls[i + 1]
                price3, pnl3 = pnls[i + 2]
                
                # Rate of change should be consistent
                if price2 != price1 and price3 != price2:
                    rate1 = (pnl2 - pnl1) / (price2 - price1)
                    rate2 = (pnl3 - pnl2) / (price3 - price2)
                    
                    # Allow 1% tolerance for rounding
                    if position.side == PositionSide.LONG:
                        assert abs(rate1 - rate2) < abs(rate1) * Decimal("0.01")


class TestRiskValidationProperties:
    """Property tests for risk validation logic."""

    @given(
        account=trading_account(),
        quantity=decimal_quantity(),
        entry_price=decimal_price(),
    )
    @settings(max_examples=100, deadline=1000)
    def test_risk_validation_consistency(self, account, quantity, entry_price):
        """Risk validation should be consistent for same inputs."""
        risk_engine = RiskEngine(account)
        
        # Property: Validation is deterministic
        try:
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=quantity,
                entry_price=entry_price,
            )
            # If it passes once, it should always pass
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=quantity,
                entry_price=entry_price,
            )
        except Exception as e1:
            # If it fails once, it should always fail with same error
            with pytest.raises(type(e1)):
                risk_engine.validate_order_risk(
                    symbol="BTC/USDT",
                    side=PositionSide.LONG,
                    quantity=quantity,
                    entry_price=entry_price,
                )

    @given(
        account=trading_account(),
        positions=st.lists(trading_position(), max_size=5),
    )
    @settings(max_examples=50, deadline=1000)
    def test_total_exposure_calculation(self, account, positions):
        """Total exposure should equal sum of individual position values."""
        risk_engine = RiskEngine(account)
        
        # Add positions
        for position in positions:
            risk_engine.add_position(position)
        
        # Property: Total exposure = sum of position values
        calculated_total = risk_engine.get_total_exposure()
        manual_total = sum(p.dollar_value for p in positions)
        
        assert abs(calculated_total - manual_total) < Decimal("0.01")

    @given(
        balance=account_balance(min_value=Decimal("100"), max_value=Decimal("10000")),
        daily_loss_percent=st.floats(min_value=1.0, max_value=10.0),
        current_loss=st.floats(min_value=0.0, max_value=15.0),
    )
    @settings(max_examples=100, deadline=1000)
    def test_daily_loss_limit_enforcement(self, balance, daily_loss_percent, current_loss):
        """Daily loss limit should correctly trigger at threshold."""
        account = Account(
            account_id="test",
            balance_usdt=balance,
            tier=TradingTier.SNIPER,
        )
        
        daily_limit = balance * Decimal(str(daily_loss_percent)) / Decimal("100")
        current_loss_amount = balance * Decimal(str(current_loss)) / Decimal("100")
        
        session = TradingSession(
            session_id="test-session",
            account_id="test",
            session_date=datetime.now(),
            starting_balance=balance,
            current_balance=balance - current_loss_amount,
            daily_loss_limit=daily_limit,
            realized_pnl=-current_loss_amount,
        )
        
        # Property: Limit should trigger when loss >= limit
        is_limit_reached = session.is_daily_limit_reached()
        
        if current_loss_amount >= daily_limit:
            assert is_limit_reached is True
        else:
            assert is_limit_reached is False


class TestStopLossProperties:
    """Property tests for stop-loss calculations."""

    @given(
        entry_price=decimal_price(min_value=Decimal("1"), max_value=Decimal("100000")),
        stop_percent=st.floats(min_value=0.1, max_value=10.0),
        side=st.sampled_from(list(PositionSide)),
    )
    @settings(max_examples=100, deadline=1000)
    def test_stop_loss_distance(self, entry_price, stop_percent, side):
        """Stop loss should be correct distance from entry price."""
        risk_engine = RiskEngine(Account(
            account_id="test",
            balance_usdt=Decimal("1000"),
            tier=TradingTier.SNIPER,
        ))
        
        stop_percent_decimal = Decimal(str(stop_percent))
        stop_loss = risk_engine.calculate_stop_loss(
            entry_price, side, stop_percent_decimal
        )
        
        # Property: Stop loss distance matches percentage
        if side == PositionSide.LONG:
            # Stop should be below entry for long
            assert stop_loss < entry_price
            distance = (entry_price - stop_loss) / entry_price * Decimal("100")
        else:
            # Stop should be above entry for short
            assert stop_loss > entry_price
            distance = (stop_loss - entry_price) / entry_price * Decimal("100")
        
        # Allow small tolerance for rounding
        assert abs(distance - stop_percent_decimal) < Decimal("0.01")

    @given(
        entry_price=decimal_price(),
        stop_percents=st.lists(
            st.floats(min_value=0.5, max_value=5.0),
            min_size=2,
            max_size=5,
        ),
    )
    @settings(max_examples=50, deadline=1000)
    def test_stop_loss_monotonicity(self, entry_price, stop_percents):
        """Larger stop percentages should result in stops further from entry."""
        risk_engine = RiskEngine(Account(
            account_id="test",
            balance_usdt=Decimal("1000"),
            tier=TradingTier.SNIPER,
        ))
        
        stops_long = []
        stops_short = []
        
        for percent in stop_percents:
            stop_long = risk_engine.calculate_stop_loss(
                entry_price, PositionSide.LONG, Decimal(str(percent))
            )
            stop_short = risk_engine.calculate_stop_loss(
                entry_price, PositionSide.SHORT, Decimal(str(percent))
            )
            stops_long.append((percent, stop_long))
            stops_short.append((percent, stop_short))
        
        # Property: Monotonic relationship
        stops_long.sort(key=lambda x: x[0])
        stops_short.sort(key=lambda x: x[0])
        
        for i in range(len(stops_long) - 1):
            # For longs, larger percentage = lower stop price
            assert stops_long[i+1][1] <= stops_long[i][1]
            # For shorts, larger percentage = higher stop price  
            assert stops_short[i+1][1] >= stops_short[i][1]


# Import required for tests
from datetime import datetime
from genesis.core.exceptions import InsufficientBalance, MinimumPositionSize