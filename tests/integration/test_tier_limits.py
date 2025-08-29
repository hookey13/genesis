"""
Integration tests for tier-based position limits.

Tests that tier limits are properly enforced across the system,
including position sizing, daily loss limits, and strategy restrictions.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from genesis.config.settings import Settings
from genesis.core.events import Event, EventBus, EventType
from genesis.core.exceptions import (
    DailyLossLimitReached,
    RiskLimitExceeded,
)
from genesis.core.models import (
    Account,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.engine.event_bus import EventBus
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.trading_loop import TradingLoop
from genesis.exchange.gateway import ExchangeGateway


class TestTierLimits:
    """Integration tests for tier-based limits."""

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create temporary config file for testing."""
        config_path = tmp_path / "trading_rules.yaml"
        config_data = {
            "tiers": {
                "SNIPER": {
                    "max_position_size": 100.0,
                    "daily_loss_limit": 10.0,
                    "position_risk_percent": 5.0,
                    "max_positions": 3,
                    "stop_loss_percent": 2.0,
                    "allowed_strategies": ["momentum", "breakout"],
                    "max_leverage": 1.0,
                    "min_win_rate": 0.45,
                    "max_drawdown": 25.0,
                    "recovery_lockout_hours": 4,
                    "tilt_thresholds": {
                        "level_1": 3,
                        "level_2": 5,
                        "level_3": 7
                    }
                },
                "HUNTER": {
                    "max_position_size": 500.0,
                    "daily_loss_limit": 10.0,
                    "position_risk_percent": 5.0,
                    "max_positions": 5,
                    "stop_loss_percent": 2.0,
                    "allowed_strategies": ["momentum", "breakout", "multi_pair"],
                    "max_leverage": 2.0,
                    "min_win_rate": 0.50,
                    "max_drawdown": 20.0,
                    "recovery_lockout_hours": 2,
                    "tilt_thresholds": {
                        "level_1": 4,
                        "level_2": 6,
                        "level_3": 8
                    },
                    "kelly_sizing": {
                        "enabled": True,
                        "max_kelly_fraction": 0.25,
                        "lookback_days": 30,
                        "min_trades": 20
                    }
                }
            },
            "global_rules": {
                "min_position_size": 10.0,
                "max_portfolio_exposure": 90.0,
                "max_correlation_exposure": 60.0,
                "force_close_loss_percent": 50.0,
                "max_daily_trades": 100,
                "max_slippage_percent": 0.5,
                "order_timeout_seconds": 30
            },
            "exchange_limits": {
                "binance": {
                    "rate_limit_per_second": 10,
                    "max_order_size_usdt": 100000.0,
                    "min_order_size_usdt": 10.0,
                    "max_websocket_connections": 5
                }
            },
            "market_conditions": {
                "high_volatility": {
                    "position_size_multiplier": 0.5,
                    "stop_loss_multiplier": 1.5,
                    "max_positions_multiplier": 0.7
                }
            },
            "tier_gates": {
                "SNIPER_TO_HUNTER": {
                    "capital_required": 2000.0,
                    "trades_required": 100,
                    "win_rate_required": 0.45,
                    "days_required": 30,
                    "max_drawdown_allowed": 25.0
                }
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        return config_path

    @pytest.fixture
    def settings(self, config_file, monkeypatch):
        """Create settings with test config."""
        # Use monkeypatch to set the environment variable for the config path
        monkeypatch.setenv("GENESIS_TRADING_RULES_PATH", str(config_file))
        settings = Settings()
        return settings

    @pytest.fixture
    def sniper_account(self):
        """Create SNIPER tier account."""
        return Account(
            account_id="sniper_test",
            tier=TradingTier.SNIPER,
            balance_usdt=Decimal("500"),
            locked_features=["multi_pair", "kelly_sizing"]
        )

    @pytest.fixture
    def hunter_account(self):
        """Create HUNTER tier account."""
        return Account(
            account_id="hunter_test",
            tier=TradingTier.HUNTER,
            balance_usdt=Decimal("5000"),
            locked_features=["statistical_arb", "market_making"]
        )

    @pytest.fixture
    def exchange_gateway(self):
        """Create mock exchange gateway."""
        gateway = MagicMock(spec=ExchangeGateway)
        gateway.validate_connection = AsyncMock(return_value=True)
        gateway.execute_order = AsyncMock(return_value={
            "success": True,
            "exchange_order_id": "test_order_123",
            "fill_price": Decimal("50000"),
            "latency_ms": 10
        })
        return gateway

    @pytest.mark.asyncio
    async def test_sniper_tier_position_limit(self, sniper_account, settings):
        """Test SNIPER tier respects $100 max position size."""
        # Load tier configuration
        tier_limits = settings.get_tier_limits(TradingTier.SNIPER)

        # Create risk engine with loaded config
        risk_engine = RiskEngine(
            account=sniper_account,
            session=TradingSession(
                account_id=sniper_account.account_id,
                daily_loss_limit=Decimal(str(tier_limits["daily_loss_limit"])),
                starting_balance=sniper_account.balance_usdt,
                current_balance=sniper_account.balance_usdt,
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0")
            )
        )

        # Override tier limits with loaded config
        risk_engine.tier_limits = tier_limits

        # Try to create position larger than $100
        with pytest.raises(RiskLimitExceeded):
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.003"),  # $150 at $50k
                entry_price=Decimal("50000")
            )

    @pytest.mark.asyncio
    async def test_hunter_tier_increased_limits(self, hunter_account, settings):
        """Test HUNTER tier has increased position limits."""
        # Load tier configuration
        tier_limits = settings.get_tier_limits(TradingTier.HUNTER)

        # Create risk engine
        risk_engine = RiskEngine(
            account=hunter_account,
            session=TradingSession(
                account_id=hunter_account.account_id,
                daily_loss_limit=Decimal(str(tier_limits["daily_loss_limit"])),
                starting_balance=hunter_account.balance_usdt,
                current_balance=hunter_account.balance_usdt,
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0")
            )
        )

        # Override tier limits
        risk_engine.tier_limits = tier_limits

        # HUNTER can have $500 positions - this should pass
        risk_engine.validate_order_risk(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.008"),  # $400 at $50k - within HUNTER limit
            entry_price=Decimal("50000")
        )

        # But not more than $500
        with pytest.raises(RiskLimitExceeded):
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.012"),  # $600 at $50k - exceeds HUNTER limit
                entry_price=Decimal("50000")
            )

    @pytest.mark.asyncio
    async def test_tier_specific_max_positions(self, sniper_account, hunter_account, settings):
        """Test tier-specific maximum position counts."""
        # SNIPER tier - max 3 positions
        sniper_limits = settings.get_tier_limits(TradingTier.SNIPER)
        sniper_risk_engine = RiskEngine(
            account=sniper_account,
            session=TradingSession(
                account_id=sniper_account.account_id,
                daily_loss_limit=Decimal(str(sniper_limits["daily_loss_limit"])),
                starting_balance=sniper_account.balance_usdt,
                current_balance=sniper_account.balance_usdt,
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0")
            )
        )
        sniper_risk_engine.tier_limits = sniper_limits

        # Add 3 positions (at limit)
        for i in range(3):
            pos = Position(
                position_id=f"pos_{i}",
                account_id=sniper_account.account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("0.1"),
                dollar_value=Decimal("10")
            )
            sniper_risk_engine.add_position(pos)

        # 4th position should fail
        with pytest.raises(RiskLimitExceeded) as exc_info:
            sniper_risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.001"),
                entry_price=Decimal("50000")
            )
        assert "Maximum positions" in str(exc_info.value)

        # HUNTER tier - max 5 positions
        hunter_limits = settings.get_tier_limits(TradingTier.HUNTER)
        hunter_risk_engine = RiskEngine(
            account=hunter_account,
            session=TradingSession(
                account_id=hunter_account.account_id,
                daily_loss_limit=Decimal(str(hunter_limits["daily_loss_limit"])),
                starting_balance=hunter_account.balance_usdt,
                current_balance=hunter_account.balance_usdt,
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0")
            )
        )
        hunter_risk_engine.tier_limits = hunter_limits

        # Add 5 positions (at limit for HUNTER)
        for i in range(5):
            pos = Position(
                position_id=f"pos_{i}",
                account_id=hunter_account.account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("0.1"),
                dollar_value=Decimal("10")
            )
            hunter_risk_engine.add_position(pos)

        # 6th position should fail
        with pytest.raises(RiskLimitExceeded):
            hunter_risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.001"),
                entry_price=Decimal("50000")
            )

    @pytest.mark.asyncio
    async def test_daily_loss_limit_enforcement(self, sniper_account, settings):
        """Test daily loss limit is enforced based on tier."""
        tier_limits = settings.get_tier_limits(TradingTier.SNIPER)

        # Create session with losses approaching limit
        session = TradingSession(
            account_id=sniper_account.account_id,
            daily_loss_limit=Decimal("50"),  # 10% of $500
            realized_pnl=Decimal("-45"),  # $45 loss
            unrealized_pnl=Decimal("0"),
            starting_balance=Decimal("500"),
            current_balance=Decimal("455")
        )

        risk_engine = RiskEngine(
            account=sniper_account,
            session=session
        )
        risk_engine.tier_limits = tier_limits

        # Should still allow trades at -$45 (9% loss)
        risk_engine.validate_order_risk(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.001"),
            entry_price=Decimal("50000")
        )

        # But not after reaching limit
        session.realized_pnl = Decimal("-51")  # Over 10% limit
        session.is_daily_limit_reached = MagicMock(return_value=True)

        with pytest.raises(DailyLossLimitReached):
            risk_engine.validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.001"),
                entry_price=Decimal("50000")
            )

    @pytest.mark.asyncio
    async def test_stop_loss_percentage_by_tier(self, sniper_account, hunter_account, settings):
        """Test stop loss percentage is correctly set by tier."""
        # SNIPER tier - 2% stop loss
        sniper_limits = settings.get_tier_limits(TradingTier.SNIPER)
        sniper_risk_engine = RiskEngine(
            account=sniper_account,
            session=None
        )
        sniper_risk_engine.tier_limits = sniper_limits

        entry_price = Decimal("50000")
        sniper_stop_loss = sniper_risk_engine.calculate_stop_loss(
            entry_price=entry_price,
            side=PositionSide.LONG
        )

        expected_sl = entry_price * Decimal("0.98")  # 2% below
        assert sniper_stop_loss == expected_sl

        # HUNTER tier - also 2% in our config
        hunter_limits = settings.get_tier_limits(TradingTier.HUNTER)
        hunter_risk_engine = RiskEngine(
            account=hunter_account,
            session=None
        )
        hunter_risk_engine.tier_limits = hunter_limits

        hunter_stop_loss = hunter_risk_engine.calculate_stop_loss(
            entry_price=entry_price,
            side=PositionSide.LONG
        )

        assert hunter_stop_loss == expected_sl  # Same 2% for HUNTER

    @pytest.mark.asyncio
    async def test_complete_trading_flow_with_tier_limits(
        self, sniper_account, settings, exchange_gateway
    ):
        """Test complete trading flow respecting tier limits."""
        # Set up components
        event_bus = EventBus()
        tier_limits = settings.get_tier_limits(TradingTier.SNIPER)

        risk_engine = RiskEngine(
            account=sniper_account,
            session=TradingSession(
                account_id=sniper_account.account_id,
                daily_loss_limit=Decimal(str(tier_limits["daily_loss_limit"])),
                starting_balance=sniper_account.balance_usdt,
                current_balance=sniper_account.balance_usdt,
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0")
            )
        )
        risk_engine.tier_limits = tier_limits

        trading_loop = TradingLoop(
            event_bus=event_bus,
            risk_engine=risk_engine,
            exchange_gateway=exchange_gateway
        )

        await trading_loop.startup()

        # Create trading signal
        signal_event = Event(
            event_type=EventType.ARBITRAGE_SIGNAL,
            event_data={
                "strategy_id": "momentum",  # Allowed for SNIPER
                "pair1_symbol": "BTC/USDT",
                "signal_type": "ENTRY",
                "confidence_score": 0.8,
                "entry_price": "50000"
            }
        )

        # Process signal
        await trading_loop._handle_trading_signal(signal_event)

        # Verify order was created with proper position size
        assert trading_loop.signals_generated == 1
        assert exchange_gateway.execute_order.called

        # Check the order had correct position size
        order_call = exchange_gateway.execute_order.call_args[0][0]
        position_value = order_call.quantity * Decimal("50000")

        # Should be within SNIPER limits
        assert position_value <= Decimal("100")  # Max position size for SNIPER
        assert position_value >= Decimal("10")   # Min position size

    @pytest.mark.asyncio
    async def test_portfolio_exposure_limits(self, sniper_account, settings):
        """Test portfolio-wide exposure limits."""
        tier_limits = settings.get_tier_limits(TradingTier.SNIPER)
        config = settings.load_trading_rules()

        risk_engine = RiskEngine(
            account=sniper_account,
            session=None
        )
        risk_engine.tier_limits = tier_limits

        # Create positions totaling near 90% exposure limit
        positions = [
            Position(
                position_id="pos1",
                account_id=sniper_account.account_id,
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("50000"),
                quantity=Decimal("0.004"),
                dollar_value=Decimal("200")  # 40% of $500
            ),
            Position(
                position_id="pos2",
                account_id=sniper_account.account_id,
                symbol="ETH/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("3000"),
                quantity=Decimal("0.08"),
                dollar_value=Decimal("240")  # 48% of $500
            )
        ]

        # Total exposure: 88% - should pass
        portfolio_risk = risk_engine.validate_portfolio_risk(positions)
        assert portfolio_risk["approved"] is True
        assert portfolio_risk["portfolio_exposure"] == Decimal("440")

        # Add position that would exceed 90% limit
        positions.append(
            Position(
                position_id="pos3",
                account_id=sniper_account.account_id,
                symbol="SOL/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("0.2"),
                dollar_value=Decimal("20")  # Would make total 92%
            )
        )

        # Should fail now
        portfolio_risk = risk_engine.validate_portfolio_risk(positions)
        assert portfolio_risk["approved"] is False
        assert "exposure" in portfolio_risk["rejections"][0].lower()
