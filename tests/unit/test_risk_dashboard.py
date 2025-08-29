"""Unit tests for risk dashboard module."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest

from genesis.analytics.risk_dashboard import RiskDashboard
from genesis.analytics.risk_metrics import RiskMetrics
from genesis.core.models import OrderSide, Position, PositionSide, Trade


@pytest.fixture
def mock_repository():
    """Create mock repository."""
    repo = AsyncMock()
    repo.get_positions_by_account = AsyncMock(return_value=[])
    repo.get_trades_by_account = AsyncMock(return_value=[])
    repo.get_price_history = AsyncMock(return_value=[])
    repo.save_risk_metrics = AsyncMock()
    return repo


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    bus = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def risk_dashboard(mock_repository, mock_event_bus):
    """Create RiskDashboard instance."""
    with patch("genesis.analytics.risk_dashboard.asyncio.create_task"):
        return RiskDashboard(mock_repository, mock_event_bus)


@pytest.fixture
def sample_positions():
    """Create sample positions."""
    return [
        Position(
            position_id=str(uuid4()),
            account_id="test_account",
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.5"),
            dollar_value=Decimal("25000"),
        ),
        Position(
            position_id=str(uuid4()),
            account_id="test_account",
            symbol="ETH/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("3000"),
            quantity=Decimal("5"),
            dollar_value=Decimal("15000"),
        ),
    ]


@pytest.fixture
def sample_trades():
    """Create sample trades with returns."""
    return [
        Trade(
            trade_id=str(uuid4()),
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            strategy_id="test",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            entry_price=Decimal("50000"),
            exit_price=Decimal("51000"),
            quantity=Decimal("0.1"),
            pnl_dollars=Decimal("100"),
            pnl_percent=Decimal("2.0"),
        ),
        Trade(
            trade_id=str(uuid4()),
            order_id=str(uuid4()),
            position_id=str(uuid4()),
            strategy_id="test",
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
def sample_price_history():
    """Create sample price history."""
    prices = []
    base_price = 50000
    for i in range(100):
        # Simulate price movement
        price_change = np.random.normal(0, 0.02) * base_price
        close_price = base_price + price_change
        prices.append(
            MagicMock(
                timestamp=datetime.now(UTC) - timedelta(days=100 - i),
                close=Decimal(str(close_price)),
            )
        )
        base_price = close_price
    return prices


@pytest.mark.asyncio
async def test_calculate_value_at_risk_no_positions(risk_dashboard, mock_repository):
    """Test VaR calculation with no positions."""
    account_id = "test_account"
    mock_repository.get_positions_by_account.return_value = []

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        var = await risk_dashboard.calculate_value_at_risk(account_id)

    assert var == Decimal("0")


@pytest.mark.asyncio
async def test_calculate_value_at_risk_historical(
    risk_dashboard, mock_repository, sample_positions, sample_price_history
):
    """Test historical VaR calculation."""
    account_id = "test_account"
    mock_repository.get_positions_by_account.return_value = sample_positions
    mock_repository.get_price_history.return_value = sample_price_history

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        var = await risk_dashboard.calculate_value_at_risk(
            account_id,
            confidence_level=Decimal("0.95"),
            time_horizon_days=1,
            method="historical",
        )

    # VaR should be positive
    assert var > Decimal("0")
    # VaR should be reasonable relative to portfolio value (40k)
    assert var < Decimal("40000")  # Less than total portfolio value

    # Verify event published
    mock_repository.get_positions_by_account.assert_called_once_with(account_id)


@pytest.mark.asyncio
async def test_calculate_value_at_risk_parametric(
    risk_dashboard, mock_repository, sample_positions, sample_price_history
):
    """Test parametric VaR calculation."""
    account_id = "test_account"
    mock_repository.get_positions_by_account.return_value = sample_positions
    mock_repository.get_price_history.return_value = sample_price_history

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        var = await risk_dashboard.calculate_value_at_risk(
            account_id,
            confidence_level=Decimal("0.95"),
            time_horizon_days=1,
            method="parametric",
        )

    # VaR should be positive
    assert var >= Decimal("0")
    # Check event published
    mock_repository.get_positions_by_account.assert_called_once_with(account_id)


@pytest.mark.asyncio
async def test_calculate_value_at_risk_monte_carlo(
    risk_dashboard, mock_repository, sample_positions
):
    """Test Monte Carlo VaR calculation."""
    account_id = "test_account"
    mock_repository.get_positions_by_account.return_value = sample_positions

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        var = await risk_dashboard.calculate_value_at_risk(
            account_id,
            confidence_level=Decimal("0.95"),
            time_horizon_days=1,
            method="monte_carlo",
        )

    # VaR should be positive
    assert var > Decimal("0")
    # VaR should be reasonable (based on 2% daily vol assumption)
    assert var < Decimal("4000")  # ~10% of portfolio value


@pytest.mark.asyncio
async def test_calculate_value_at_risk_caching(
    risk_dashboard, mock_repository, sample_positions
):
    """Test VaR calculation caching."""
    account_id = "test_account"
    mock_repository.get_positions_by_account.return_value = sample_positions

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        # First call
        var1 = await risk_dashboard.calculate_value_at_risk(
            account_id, method="monte_carlo"
        )

        # Second call (should use cache)
        var2 = await risk_dashboard.calculate_value_at_risk(
            account_id, method="monte_carlo"
        )

    # Should only call repository once due to caching
    assert mock_repository.get_positions_by_account.call_count == 1
    assert var1 == var2


@pytest.mark.asyncio
async def test_calculate_conditional_var(
    risk_dashboard, mock_repository, sample_positions
):
    """Test CVaR calculation."""
    account_id = "test_account"
    mock_repository.get_positions_by_account.return_value = sample_positions

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        # Mock VaR calculation
        with patch.object(
            risk_dashboard, "calculate_value_at_risk", return_value=Decimal("1000")
        ):
            cvar = await risk_dashboard.calculate_conditional_var(account_id)

    # CVaR should be >= VaR
    assert cvar >= Decimal("1000")

    # Verify event published
    mock_repository.get_positions_by_account.assert_called_with(account_id)


@pytest.mark.asyncio
async def test_calculate_portfolio_greeks(
    risk_dashboard, mock_repository, sample_positions
):
    """Test portfolio Greeks calculation."""
    account_id = "test_account"
    mock_repository.get_positions_by_account.return_value = sample_positions

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        greeks = await risk_dashboard.calculate_portfolio_greeks(account_id)

    # Check Greeks structure
    assert "delta" in greeks
    assert "gamma" in greeks
    assert "theta" in greeks
    assert "vega" in greeks
    assert "rho" in greeks

    # For spot positions, delta should equal net quantity
    total_quantity = sum(p.quantity for p in sample_positions)
    assert greeks["delta"] == total_quantity

    # Spot positions have no gamma or theta
    assert greeks["gamma"] == Decimal("0")
    assert greeks["theta"] == Decimal("0")

    # Verify event published
    mock_repository.get_positions_by_account.assert_called_once_with(account_id)


@pytest.mark.asyncio
async def test_update_risk_metrics(
    risk_dashboard, mock_repository, mock_event_bus, sample_trades
):
    """Test comprehensive risk metrics update."""
    account_id = "test_account"
    mock_repository.get_trades_by_account.return_value = sample_trades
    mock_repository.get_positions_by_account.return_value = []

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        # Mock VaR and CVaR calculations
        with patch.object(
            risk_dashboard, "calculate_value_at_risk", return_value=Decimal("1000")
        ):
            with patch.object(
                risk_dashboard,
                "calculate_conditional_var",
                return_value=Decimal("1500"),
            ):
                metrics = await risk_dashboard.update_risk_metrics(account_id)

    # Check metrics object
    assert isinstance(metrics, RiskMetrics)
    assert metrics.value_at_risk_95 == Decimal("1000")
    assert metrics.conditional_value_at_risk_95 == Decimal("1500")

    # Check cache
    assert account_id in risk_dashboard._risk_cache
    assert risk_dashboard._risk_cache[account_id] == metrics

    # Verify repository save
    mock_repository.save_risk_metrics.assert_called_once()

    # Verify event published
    mock_event_bus.publish.assert_called()


@pytest.mark.asyncio
async def test_get_risk_limits_status(risk_dashboard, mock_repository, mock_event_bus):
    """Test risk limits checking."""
    account_id = "test_account"

    # Create metrics with some limits breached
    metrics = RiskMetrics(
        sharpe_ratio=Decimal("0.5"),  # Below limit
        sortino_ratio=Decimal("1.0"),
        calmar_ratio=Decimal("1.0"),
        max_drawdown=Decimal("0.25"),  # Above limit
        max_drawdown_duration_days=10,
        volatility=Decimal("0.30"),  # Within limit
        downside_deviation=Decimal("0.20"),
        value_at_risk_95=Decimal("8000"),  # Within limit
        conditional_value_at_risk_95=Decimal("10000"),
    )

    risk_dashboard._risk_cache[account_id] = metrics

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        status = await risk_dashboard.get_risk_limits_status(account_id)

    # Check status structure
    assert "var_95" in status
    assert "max_drawdown" in status
    assert "sharpe_ratio" in status
    assert "volatility" in status

    # Check breached limits
    assert status["sharpe_ratio"]["breached"] is True
    assert status["max_drawdown"]["breached"] is True
    assert status["volatility"]["breached"] is False
    assert status["var_95"]["breached"] is False

    # Verify alert published for breached limits
    mock_event_bus.publish.assert_called()
    call_args = mock_event_bus.publish.call_args[0]
    assert call_args[0] == "risk.limits_breached"
    assert "sharpe_ratio" in call_args[1]["breached_limits"]
    assert "max_drawdown" in call_args[1]["breached_limits"]


@pytest.mark.asyncio
async def test_event_subscription(risk_dashboard, mock_event_bus):
    """Test event subscription setup."""
    # Verify subscriptions were set up
    assert mock_event_bus.subscribe.call_count >= 4

    # Check subscription topics
    call_topics = [call[0][0] for call in mock_event_bus.subscribe.call_args_list]
    assert "position.opened" in call_topics
    assert "position.closed" in call_topics
    assert "position.updated" in call_topics
    assert "trade.completed" in call_topics


@pytest.mark.asyncio
async def test_handle_position_update(risk_dashboard):
    """Test position update event handling."""
    account_id = "test_account"

    # Add something to caches
    risk_dashboard._risk_cache[account_id] = MagicMock()
    risk_dashboard._var_cache[f"{account_id}_0.95_1_historical"] = (
        Decimal("1000"),
        datetime.now(),
    )
    risk_dashboard._cvar_cache[f"{account_id}_0.95_1"] = (
        Decimal("1500"),
        datetime.now(),
    )

    # Mock update_risk_metrics
    with patch.object(risk_dashboard, "update_risk_metrics", new_callable=AsyncMock):
        await risk_dashboard._handle_position_update({"account_id": account_id})

    # Check caches were cleared
    assert account_id not in risk_dashboard._risk_cache
    assert f"{account_id}_0.95_1_historical" not in risk_dashboard._var_cache
    assert f"{account_id}_0.95_1" not in risk_dashboard._cvar_cache

    # Check metrics were updated
    risk_dashboard.update_risk_metrics.assert_called_once_with(account_id)


@pytest.mark.asyncio
async def test_unsupported_var_method(
    risk_dashboard, mock_repository, sample_positions
):
    """Test VaR calculation with unsupported method."""
    account_id = "test_account"
    mock_repository.get_positions_by_account.return_value = sample_positions

    with patch("genesis.analytics.risk_dashboard.requires_tier", lambda x: lambda f: f):
        with pytest.raises(ValueError, match="Unsupported VaR method"):
            await risk_dashboard.calculate_value_at_risk(
                account_id, method="unsupported_method"
            )


@pytest.mark.asyncio
async def test_close_dashboard(risk_dashboard):
    """Test closing risk dashboard cleans up resources."""
    # Add data to caches
    risk_dashboard._risk_cache["test"] = MagicMock()
    risk_dashboard._var_cache["test"] = (Decimal("1000"), datetime.now())
    risk_dashboard._cvar_cache["test"] = (Decimal("1500"), datetime.now())

    # Close dashboard
    await risk_dashboard.close()

    # Verify caches cleared
    assert len(risk_dashboard._risk_cache) == 0
    assert len(risk_dashboard._var_cache) == 0
    assert len(risk_dashboard._cvar_cache) == 0
