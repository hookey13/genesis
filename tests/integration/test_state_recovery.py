"""State recovery and persistence tests."""

import asyncio
import pytest
from decimal import Decimal
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import structlog
import json
import pickle
from datetime import datetime

from genesis.engine.engine import TradingEngine
from genesis.engine.state_machine import StateMachine
from genesis.core.models import Position, Order, OrderType, OrderSide
from genesis.data.repository import StateRepository

logger = structlog.get_logger(__name__)


@pytest.mark.asyncio
class TestStateRecovery:
    """Test system state recovery and persistence."""

    async def test_graceful_shutdown_and_restart(self, trading_system):
        """Test graceful shutdown and restart with state preservation."""
        positions_before = [
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
                current_price=Decimal("2950.00")
            )
        ]
        
        for position in positions_before:
            await trading_system.engine.open_position(position)
        
        pending_orders = [
            Order(
                order_id="pending_1",
                symbol="BTC/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("52000.00")
            )
        ]
        
        for order in pending_orders:
            await trading_system.engine.submit_order(order)
        
        state_before_shutdown = await trading_system.engine.get_state()
        
        await trading_system.engine.graceful_shutdown()
        
        new_engine = TradingEngine(
            exchange_gateway=trading_system.exchange_gateway,
            risk_engine=trading_system.risk_engine,
            state_machine=trading_system.state_machine
        )
        
        await new_engine.restore_from_state(state_before_shutdown)
        
        positions_after = await new_engine.get_positions()
        orders_after = await new_engine.get_pending_orders()
        
        assert len(positions_after) == len(positions_before)
        assert len(orders_after) == len(pending_orders)
        
        for pos_before, pos_after in zip(positions_before, positions_after):
            assert pos_before.symbol == pos_after.symbol
            assert pos_before.quantity == pos_after.quantity
            assert pos_before.entry_price == pos_after.entry_price

    async def test_position_restoration(self, trading_system):
        """Test position restoration from database."""
        repo = StateRepository()
        
        positions_to_save = [
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": Decimal("0.5"),
                "entry_price": Decimal("49000.00"),
                "current_price": Decimal("50000.00"),
                "unrealized_pnl": Decimal("500.00")
            },
            {
                "symbol": "ETH/USDT",
                "side": "sell",
                "quantity": Decimal("2.0"),
                "entry_price": Decimal("3100.00"),
                "current_price": Decimal("3000.00"),
                "unrealized_pnl": Decimal("200.00")
            }
        ]
        
        for position_data in positions_to_save:
            await repo.save_position(position_data)
        
        await trading_system.engine.stop()
        
        new_engine = TradingEngine(
            exchange_gateway=trading_system.exchange_gateway,
            risk_engine=trading_system.risk_engine,
            state_machine=trading_system.state_machine
        )
        
        await new_engine.restore_positions_from_db()
        
        restored_positions = await new_engine.get_positions()
        
        assert len(restored_positions) == len(positions_to_save)
        
        for saved, restored in zip(positions_to_save, restored_positions):
            assert saved["symbol"] == restored.symbol
            assert saved["quantity"] == restored.quantity

    async def test_order_state_recovery(self, trading_system):
        """Test order state recovery after system restart."""
        repo = StateRepository()
        
        orders_to_save = [
            {
                "order_id": "order_123",
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "limit",
                "quantity": Decimal("0.1"),
                "price": Decimal("48000.00"),
                "status": "pending",
                "timestamp": datetime.now().isoformat()
            },
            {
                "order_id": "order_124",
                "symbol": "ETH/USDT",
                "side": "sell",
                "type": "limit",
                "quantity": Decimal("1.0"),
                "price": Decimal("3200.00"),
                "status": "partially_filled",
                "filled_quantity": Decimal("0.5"),
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        for order_data in orders_to_save:
            await repo.save_order(order_data)
        
        await trading_system.engine.stop()
        
        new_engine = TradingEngine(
            exchange_gateway=trading_system.exchange_gateway,
            risk_engine=trading_system.risk_engine,
            state_machine=trading_system.state_machine
        )
        
        await new_engine.restore_orders_from_db()
        
        restored_orders = await new_engine.get_all_orders()
        
        assert len(restored_orders) == len(orders_to_save)
        
        partially_filled = [o for o in restored_orders if o.status == "partially_filled"]
        assert len(partially_filled) == 1
        assert partially_filled[0].filled_quantity == Decimal("0.5")

    async def test_strategy_state_persistence(self, trading_system):
        """Test strategy state persistence and recovery."""
        from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
        
        strategy = SimpleArbitrageStrategy(
            symbol="BTC/USDT",
            min_spread=Decimal("0.001")
        )
        
        await trading_system.engine.register_strategy(strategy)
        
        strategy_state = {
            "last_signal_time": datetime.now().isoformat(),
            "signals_generated": 42,
            "positions_opened": 15,
            "total_pnl": Decimal("1234.56"),
            "custom_data": {"key": "value"}
        }
        
        await strategy.save_state(strategy_state)
        
        await trading_system.engine.stop()
        
        new_strategy = SimpleArbitrageStrategy(
            symbol="BTC/USDT",
            min_spread=Decimal("0.001")
        )
        
        restored_state = await new_strategy.load_state()
        
        assert restored_state["signals_generated"] == 42
        assert restored_state["positions_opened"] == 15
        assert restored_state["total_pnl"] == Decimal("1234.56")
        assert restored_state["custom_data"]["key"] == "value"

    async def test_tier_state_recovery(self, trading_system):
        """Test tier state and progression recovery."""
        state_machine = trading_system.state_machine
        
        state_machine.set_tier(TierState.HUNTER)
        state_machine.set_balance(Decimal("5000.00"))
        state_machine.record_trade_count(150)
        state_machine.record_win_rate(Decimal("0.65"))
        
        tier_state = await state_machine.save_state()
        
        new_state_machine = StateMachine()
        await new_state_machine.restore_state(tier_state)
        
        assert new_state_machine.current_tier == TierState.HUNTER
        assert new_state_machine.current_balance == Decimal("5000.00")
        assert new_state_machine.trade_count == 150
        assert new_state_machine.win_rate == Decimal("0.65")

    async def test_configuration_state_recovery(self, trading_system):
        """Test configuration state recovery."""
        config_state = {
            "risk_settings": {
                "max_position_size": Decimal("1.0"),
                "stop_loss_pct": Decimal("0.05"),
                "max_drawdown": Decimal("0.20")
            },
            "trading_settings": {
                "enabled_strategies": ["simple_arb", "mean_reversion"],
                "trading_pairs": ["BTC/USDT", "ETH/USDT"],
                "order_timeout": 30
            },
            "system_settings": {
                "log_level": "INFO",
                "backup_interval": 3600,
                "health_check_interval": 60
            }
        }
        
        await trading_system.engine.save_configuration(config_state)
        
        await trading_system.engine.stop()
        
        new_engine = TradingEngine(
            exchange_gateway=trading_system.exchange_gateway,
            risk_engine=trading_system.risk_engine,
            state_machine=trading_system.state_machine
        )
        
        restored_config = await new_engine.load_configuration()
        
        assert restored_config["risk_settings"]["max_position_size"] == Decimal("1.0")
        assert "simple_arb" in restored_config["trading_settings"]["enabled_strategies"]
        assert restored_config["system_settings"]["backup_interval"] == 3600

    async def test_partial_state_recovery(self, trading_system):
        """Test recovery with partially corrupted state."""
        repo = StateRepository()
        
        valid_position = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": Decimal("0.1"),
            "entry_price": Decimal("50000.00")
        }
        
        corrupted_position = {
            "symbol": "ETH/USDT",
            "side": "invalid_side",  # Corrupted data
            "quantity": "not_a_number",  # Corrupted data
            "entry_price": Decimal("3000.00")
        }
        
        await repo.save_position(valid_position)
        await repo.save_position(corrupted_position)
        
        new_engine = TradingEngine(
            exchange_gateway=trading_system.exchange_gateway,
            risk_engine=trading_system.risk_engine,
            state_machine=trading_system.state_machine
        )
        
        recovered_positions = await new_engine.restore_positions_from_db()
        
        assert len(recovered_positions) == 1  # Only valid position recovered
        assert recovered_positions[0].symbol == "BTC/USDT"

    async def test_state_versioning(self, trading_system):
        """Test state versioning for backward compatibility."""
        repo = StateRepository()
        
        old_state_v1 = {
            "version": "1.0",
            "positions": [{"symbol": "BTC/USDT", "quantity": 0.1}],
            "orders": []
        }
        
        new_state_v2 = {
            "version": "2.0",
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "quantity": Decimal("0.1"),
                    "entry_price": Decimal("50000.00"),
                    "risk_score": Decimal("0.3")  # New field in v2
                }
            ],
            "orders": [],
            "strategies": []  # New in v2
        }
        
        await repo.save_state(old_state_v1)
        
        migrated_state = await repo.migrate_state(old_state_v1, "2.0")
        
        assert migrated_state["version"] == "2.0"
        assert "risk_score" in migrated_state["positions"][0]
        assert "strategies" in migrated_state

    async def test_atomic_state_save(self, trading_system):
        """Test atomic state saving to prevent partial writes."""
        repo = StateRepository()
        
        state_to_save = {
            "positions": [
                {"symbol": "BTC/USDT", "quantity": Decimal("0.1")},
                {"symbol": "ETH/USDT", "quantity": Decimal("1.0")}
            ],
            "orders": [
                {"order_id": "123", "symbol": "BTC/USDT"},
                {"order_id": "124", "symbol": "ETH/USDT"}
            ],
            "balances": {
                "USDT": Decimal("10000.00"),
                "BTC": Decimal("0.5"),
                "ETH": Decimal("10.0")
            }
        }
        
        async with repo.atomic_transaction() as tx:
            for component, data in state_to_save.items():
                await tx.save_component(component, data)
            
            await tx.simulate_failure()  # Simulate failure during save
        
        recovered_state = await repo.load_state()
        
        assert recovered_state is None or recovered_state == state_to_save  # Either all or nothing

    async def test_state_compression(self, trading_system):
        """Test state compression for efficient storage."""
        import gzip
        
        large_state = {
            "positions": [
                {
                    "symbol": f"COIN{i}/USDT",
                    "quantity": Decimal(f"{i * 0.1}"),
                    "entry_price": Decimal(f"{1000 + i}")
                }
                for i in range(1000)
            ]
        }
        
        uncompressed_size = len(json.dumps(large_state, default=str).encode())
        
        compressed_state = gzip.compress(
            json.dumps(large_state, default=str).encode()
        )
        compressed_size = len(compressed_state)
        
        compression_ratio = uncompressed_size / compressed_size
        
        assert compression_ratio > 2  # Should achieve at least 2x compression
        
        decompressed = json.loads(
            gzip.decompress(compressed_state).decode()
        )
        
        assert len(decompressed["positions"]) == 1000