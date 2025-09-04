"""Risk engine integration tests."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog
from datetime import datetime

from genesis.engine.risk_engine import RiskEngine
from genesis.engine.state_machine import TierState
from genesis.core.models import Position, Order, OrderType, OrderSide

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
class TestRiskEngineIntegration:
    """Test risk engine integration with the trading system."""

    async def test_position_size_validation(self, trading_system):
        """Test position size validation against tier limits."""
        risk_engine = trading_system.risk_engine
        
        risk_engine.set_tier(TierState.SNIPER)
        risk_engine.set_max_position_size(Decimal("1.0"))
        
        valid_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5")
        )
        
        invalid_order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0")
        )
        
        assert await risk_engine.validate_order(valid_order)
        assert not await risk_engine.validate_order(invalid_order)

    async def test_stop_loss_enforcement(self, trading_system):
        """Test automatic stop-loss enforcement."""
        risk_engine = trading_system.risk_engine
        
        risk_engine.set_stop_loss_percentage(Decimal("0.05"))  # 5% stop loss
        
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000.00"),
            current_price=Decimal("50000.00")
        )
        
        await trading_system.engine.open_position(position)
        
        position.current_price = Decimal("47000.00")  # 6% loss
        
        should_stop = await risk_engine.check_stop_loss(position)
        assert should_stop
        
        if should_stop:
            await trading_system.engine.emergency_close_position(position)
        
        assert not position.is_open

    async def test_maximum_exposure_limits(self, trading_system):
        """Test enforcement of maximum exposure limits."""
        risk_engine = trading_system.risk_engine
        
        risk_engine.set_max_total_exposure(Decimal("10000.00"))
        
        positions = []
        total_exposure = Decimal("0")
        
        for i in range(5):
            position = Position(
                symbol=f"COIN{i}/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                entry_price=Decimal("2500.00"),
                current_price=Decimal("2500.00")
            )
            
            exposure = position.quantity * position.entry_price
            
            if total_exposure + exposure <= risk_engine.max_total_exposure:
                await trading_system.engine.open_position(position)
                positions.append(position)
                total_exposure += exposure
            else:
                rejected = await risk_engine.validate_position(position)
                assert not rejected
        
        assert len(positions) == 4  # Should only allow 4 positions (4 * 2500 = 10000)

    async def test_tier_based_restrictions(self, trading_system):
        """Test tier-based trading restrictions."""
        risk_engine = trading_system.risk_engine
        
        tier_limits = {
            TierState.SNIPER: {
                "max_position_size": Decimal("1.0"),
                "max_positions": 3,
                "allowed_strategies": ["simple_arb", "spread_capture"]
            },
            TierState.HUNTER: {
                "max_position_size": Decimal("5.0"),
                "max_positions": 10,
                "allowed_strategies": ["simple_arb", "spread_capture", "mean_reversion", "multi_pair"]
            },
            TierState.STRATEGIST: {
                "max_position_size": Decimal("20.0"),
                "max_positions": 50,
                "allowed_strategies": "all"
            }
        }
        
        for tier, limits in tier_limits.items():
            risk_engine.set_tier(tier)
            risk_engine.apply_tier_limits(limits)
            
            order = Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=limits["max_position_size"] + Decimal("0.1")
            )
            
            is_valid = await risk_engine.validate_order(order)
            assert not is_valid  # Should reject order exceeding tier limit

    async def test_leverage_limits(self, trading_system):
        """Test leverage limit enforcement."""
        risk_engine = trading_system.risk_engine
        
        risk_engine.set_max_leverage(Decimal("3.0"))
        
        account_balance = Decimal("10000.00")
        risk_engine.update_account_balance(account_balance)
        
        order_with_low_leverage = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            price=Decimal("50000.00")
        )
        
        order_with_high_leverage = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            price=Decimal("50000.00")
        )
        
        low_leverage_valid = await risk_engine.validate_leverage(order_with_low_leverage)
        high_leverage_valid = await risk_engine.validate_leverage(order_with_high_leverage)
        
        assert low_leverage_valid
        assert not high_leverage_valid

    async def test_drawdown_protection(self, trading_system):
        """Test maximum drawdown protection."""
        risk_engine = trading_system.risk_engine
        
        risk_engine.set_max_drawdown(Decimal("0.20"))  # 20% max drawdown
        
        initial_balance = Decimal("10000.00")
        risk_engine.update_peak_balance(initial_balance)
        
        current_balance = Decimal("7500.00")  # 25% drawdown
        risk_engine.update_account_balance(current_balance)
        
        is_trading_allowed = await risk_engine.check_drawdown_limit()
        assert not is_trading_allowed
        
        await trading_system.engine.pause_trading("Maximum drawdown exceeded")

    async def test_correlation_risk_management(self, trading_system):
        """Test correlation-based risk management."""
        risk_engine = trading_system.risk_engine
        
        risk_engine.set_max_correlation(Decimal("0.7"))
        
        existing_positions = [
            Position(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                entry_price=Decimal("50000.00")
            ),
            Position(
                symbol="ETH/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("10.0"),
                entry_price=Decimal("3000.00")
            )
        ]
        
        for position in existing_positions:
            await trading_system.engine.open_position(position)
        
        new_order = Order(
            symbol="BCH/USDT",  # Highly correlated with BTC
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("5.0")
        )
        
        correlation = await risk_engine.calculate_correlation("BTC/USDT", "BCH/USDT")
        
        if correlation > risk_engine.max_correlation:
            is_valid = await risk_engine.validate_correlation_risk(new_order)
            assert not is_valid

    async def test_risk_metrics_calculation(self, trading_system):
        """Test real-time risk metrics calculation."""
        risk_engine = trading_system.risk_engine
        
        positions = [
            Position(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                entry_price=Decimal("50000.00"),
                current_price=Decimal("51000.00")
            ),
            Position(
                symbol="ETH/USDT",
                side=OrderSide.SELL,
                quantity=Decimal("1.0"),
                entry_price=Decimal("3000.00"),
                current_price=Decimal("2900.00")
            )
        ]
        
        for position in positions:
            await trading_system.engine.open_position(position)
        
        metrics = await risk_engine.calculate_risk_metrics()
        
        assert "total_exposure" in metrics
        assert "unrealized_pnl" in metrics
        assert "var_95" in metrics  # Value at Risk
        assert "sharpe_ratio" in metrics
        assert "max_position_size" in metrics
        
        assert metrics["total_exposure"] == Decimal("5000.00") + Decimal("3000.00")
        assert metrics["unrealized_pnl"] == Decimal("100.00") + Decimal("100.00")

    async def test_emergency_risk_actions(self, trading_system):
        """Test emergency risk management actions."""
        risk_engine = trading_system.risk_engine
        
        risk_engine.set_emergency_thresholds({
            "max_loss_per_minute": Decimal("1000.00"),
            "max_orders_per_minute": 100,
            "min_balance": Decimal("1000.00")
        })
        
        loss_events = []
        for i in range(10):
            loss_events.append({
                "amount": Decimal("150.00"),
                "timestamp": datetime.now()
            })
        
        for event in loss_events:
            await risk_engine.record_loss(event)
        
        emergency_triggered = await risk_engine.check_emergency_conditions()
        
        if emergency_triggered:
            await trading_system.engine.emergency_shutdown()
            assert trading_system.engine.state == "EMERGENCY_STOP"

    async def test_position_sizing_algorithm(self, trading_system):
        """Test Kelly Criterion position sizing."""
        risk_engine = trading_system.risk_engine
        
        risk_engine.set_position_sizing_method("kelly")
        
        signal = {
            "symbol": "BTC/USDT",
            "win_probability": Decimal("0.6"),
            "avg_win": Decimal("100.00"),
            "avg_loss": Decimal("50.00")
        }
        
        account_balance = Decimal("10000.00")
        risk_engine.update_account_balance(account_balance)
        
        position_size = await risk_engine.calculate_position_size(signal)
        
        expected_kelly_fraction = (
            signal["win_probability"] - 
            (Decimal("1") - signal["win_probability"]) / 
            (signal["avg_win"] / signal["avg_loss"])
        )
        
        expected_position_value = account_balance * expected_kelly_fraction * Decimal("0.25")  # 25% of Kelly
        
        assert abs(position_size - expected_position_value / Decimal("50000")) < Decimal("0.001")