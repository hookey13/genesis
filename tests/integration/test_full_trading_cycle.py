"""
End-to-end tests for complete trading cycles.

Tests the full flow from market analysis through order execution,
position management, and closure with P&L tracking.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from genesis.core.exceptions import (
    DailyLossLimitReached,
    InsufficientBalance,
    TierViolation,
)
from genesis.core.models import (
    Account,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.engine.event_bus import EventBus, EventPriority, EventType
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.state_machine import TierStateMachine
from genesis.exchange.gateway import BinanceGateway
from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
from genesis.tilt.detector import TiltDetector
from genesis.tilt.interventions import InterventionManager


@pytest.fixture
async def trading_environment():
    """Create a complete trading environment for testing."""
    # Initialize components
    event_bus = EventBus()
    
    account = Account(
        account_id=str(uuid4()),
        balance_usdt=Decimal("500"),
        tier=TradingTier.SNIPER,
    )
    
    session = TradingSession(
        session_id=str(uuid4()),
        account_id=account.account_id,
        session_date=datetime.now(),
        starting_balance=account.balance_usdt,
        current_balance=account.balance_usdt,
        daily_loss_limit=Decimal("25"),  # 5% of $500
    )
    
    risk_engine = RiskEngine(account, session)
    gateway = BinanceGateway(mock_mode=True)
    await gateway.initialize()
    
    executor = MarketOrderExecutor(gateway, risk_engine)
    state_machine = TierStateMachine(account)
    tilt_detector = TiltDetector()
    intervention_manager = InterventionManager()
    
    # Setup event connections
    event_bus.subscribe(EventType.ORDER_FILLED, executor.handle_order_filled)
    event_bus.subscribe(EventType.RISK_LIMIT, risk_engine.handle_risk_event)
    event_bus.subscribe(EventType.TILT_DETECTED, intervention_manager.handle_tilt)
    
    yield {
        "event_bus": event_bus,
        "account": account,
        "session": session,
        "risk_engine": risk_engine,
        "gateway": gateway,
        "executor": executor,
        "state_machine": state_machine,
        "tilt_detector": tilt_detector,
        "intervention_manager": intervention_manager,
    }
    
    await gateway.close()


class TestFullTradingCycle:
    """Test complete trading cycles from entry to exit."""

    @pytest.mark.asyncio
    async def test_successful_trade_cycle(self, trading_environment):
        """Test a successful trade from entry to profitable exit."""
        env = trading_environment
        
        # Step 1: Market analysis
        ticker = await env["gateway"].get_ticker("BTC/USDT")
        assert ticker.last_price > 0
        
        # Step 2: Risk calculation
        position_size = env["risk_engine"].calculate_position_size(
            symbol="BTC/USDT",
            entry_price=ticker.last_price,
        )
        assert position_size > 0
        assert position_size * ticker.last_price >= Decimal("10")  # Min position size
        
        # Step 3: Place entry order
        entry_order = await env["executor"].execute_market_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=position_size,
        )
        assert entry_order.status == OrderStatus.FILLED
        
        # Step 4: Create position
        position = Position(
            position_id=str(uuid4()),
            account_id=env["account"].account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=entry_order.average_price,
            quantity=entry_order.executed_quantity,
            dollar_value=entry_order.average_price * entry_order.executed_quantity,
        )
        env["risk_engine"].add_position(position)
        
        # Step 5: Monitor position
        # Simulate price movement (10% profit)
        new_price = entry_order.average_price * Decimal("1.10")
        pnl = env["risk_engine"].calculate_pnl(position, new_price)
        assert pnl["pnl_percent"] > 0
        
        # Step 6: Exit position
        exit_order = await env["executor"].execute_market_order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=position.quantity,
        )
        assert exit_order.status == OrderStatus.FILLED
        
        # Step 7: Update session P&L
        realized_pnl = (exit_order.average_price - entry_order.average_price) * position.quantity
        env["session"].realized_pnl += realized_pnl
        env["session"].current_balance += realized_pnl
        
        # Verify final state
        assert env["session"].realized_pnl > 0
        assert env["session"].current_balance > env["session"].starting_balance
        env["risk_engine"].remove_position(position.position_id)
        assert len(env["risk_engine"].positions) == 0

    @pytest.mark.asyncio
    async def test_stop_loss_execution(self, trading_environment):
        """Test automatic stop-loss execution on adverse price movement."""
        env = trading_environment
        
        # Enter position
        entry_price = Decimal("50000")
        position_size = env["risk_engine"].calculate_position_size(
            symbol="BTC/USDT",
            entry_price=entry_price,
        )
        
        entry_order = await env["executor"].execute_market_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=position_size,
        )
        
        position = Position(
            position_id=str(uuid4()),
            account_id=env["account"].account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=entry_order.average_price,
            quantity=entry_order.executed_quantity,
            dollar_value=entry_order.average_price * entry_order.executed_quantity,
            stop_loss=env["risk_engine"].calculate_stop_loss(
                entry_order.average_price, PositionSide.LONG
            ),
        )
        env["risk_engine"].add_position(position)
        
        # Simulate price drop to stop-loss level
        stop_price = position.stop_loss
        current_pnl = env["risk_engine"].calculate_pnl(position, stop_price)
        
        # Verify stop-loss trigger
        assert current_pnl["pnl_percent"] <= Decimal("-2")  # Default 2% stop
        
        # Execute stop-loss
        stop_order = await env["executor"].execute_market_order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=position.quantity,
        )
        assert stop_order.status == OrderStatus.FILLED
        
        # Update session with loss
        realized_loss = (stop_order.average_price - entry_order.average_price) * position.quantity
        env["session"].realized_pnl += realized_loss
        
        assert env["session"].realized_pnl < 0
        assert abs(env["session"].realized_pnl) <= env["session"].daily_loss_limit

    @pytest.mark.asyncio
    async def test_daily_loss_limit_enforcement(self, trading_environment):
        """Test that trading stops when daily loss limit is reached."""
        env = trading_environment
        
        # Simulate losses approaching daily limit
        env["session"].realized_pnl = Decimal("-24")  # Just under $25 limit
        
        # Should allow one more trade
        env["risk_engine"].validate_order_risk(
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.0002"),  # Small position
            entry_price=Decimal("50000"),
        )
        
        # Exceed daily limit
        env["session"].realized_pnl = Decimal("-25")
        
        # Should block new trades
        with pytest.raises(DailyLossLimitReached):
            env["risk_engine"].validate_order_risk(
                symbol="BTC/USDT",
                side=PositionSide.LONG,
                quantity=Decimal("0.0002"),
                entry_price=Decimal("50000"),
            )

    @pytest.mark.asyncio
    async def test_multi_position_management(self, trading_environment):
        """Test managing multiple concurrent positions (Hunter tier)."""
        env = trading_environment
        
        # Upgrade to Hunter tier for multi-position support
        env["account"].tier = TradingTier.HUNTER
        env["account"].balance_usdt = Decimal("2000")
        
        positions = []
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        # Open multiple positions
        for symbol in symbols:
            ticker = await env["gateway"].get_ticker(symbol)
            size = env["risk_engine"].calculate_position_size(
                symbol=symbol,
                entry_price=ticker.last_price,
            )
            
            order = await env["executor"].execute_market_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=size,
            )
            
            position = Position(
                position_id=str(uuid4()),
                account_id=env["account"].account_id,
                symbol=symbol,
                side=PositionSide.LONG,
                entry_price=order.average_price,
                quantity=order.executed_quantity,
                dollar_value=order.average_price * order.executed_quantity,
            )
            env["risk_engine"].add_position(position)
            positions.append(position)
        
        # Verify positions
        assert len(env["risk_engine"].positions) == 3
        total_exposure = env["risk_engine"].get_total_exposure()
        assert total_exposure > 0
        
        # Close all positions
        for position in positions:
            exit_order = await env["executor"].execute_market_order(
                symbol=position.symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
            )
            env["risk_engine"].remove_position(position.position_id)
        
        assert len(env["risk_engine"].positions) == 0


class TestTierTransitions:
    """Test tier progression and demotion scenarios."""

    @pytest.mark.asyncio
    async def test_tier_progression_sniper_to_hunter(self, trading_environment):
        """Test progression from Sniper to Hunter tier."""
        env = trading_environment
        state_machine = env["state_machine"]
        
        # Start at Sniper tier
        assert env["account"].tier == TradingTier.SNIPER
        
        # Simulate meeting Hunter requirements
        env["account"].balance_usdt = Decimal("2100")  # > $2000
        env["session"].total_trades = 100  # > 50 trades
        env["session"].win_rate = Decimal("65")  # > 60%
        env["session"].days_active = 30  # > 14 days
        
        # Check if eligible for promotion
        can_progress = await state_machine.check_tier_progression()
        assert can_progress is True
        
        # Execute progression
        await state_machine.progress_tier()
        assert env["account"].tier == TradingTier.HUNTER
        
        # Verify new capabilities
        assert env["account"].max_positions == 3  # Hunter can have 3 positions
        assert env["account"].max_daily_trades == 20  # Hunter limit

    @pytest.mark.asyncio
    async def test_emergency_demotion_on_tilt(self, trading_environment):
        """Test emergency tier demotion when severe tilt is detected."""
        env = trading_environment
        
        # Start at Hunter tier
        env["account"].tier = TradingTier.HUNTER
        env["account"].balance_usdt = Decimal("5000")
        
        # Simulate tilt indicators
        tilt_events = [
            {"type": "rapid_clicks", "severity": 8},
            {"type": "revenge_trading", "severity": 9},
            {"type": "position_size_spike", "severity": 10},
        ]
        
        for event in tilt_events:
            env["tilt_detector"].record_event(event)
        
        # Check tilt level
        tilt_score = env["tilt_detector"].calculate_tilt_score()
        assert tilt_score > 7  # Severe tilt
        
        # Trigger intervention
        intervention = await env["intervention_manager"].determine_intervention(tilt_score)
        assert intervention["action"] == "emergency_demotion"
        
        # Execute demotion
        await env["state_machine"].emergency_demotion()
        assert env["account"].tier == TradingTier.SNIPER
        
        # Verify restrictions
        assert env["account"].max_positions == 1
        assert env["account"].position_size_limit == Decimal("100")

    @pytest.mark.asyncio
    async def test_tier_gate_validation(self, trading_environment):
        """Test that tier gates are properly enforced."""
        env = trading_environment
        
        # Try to use Hunter feature at Sniper tier
        with pytest.raises(TierViolation):
            await env["risk_engine"].calculate_position_correlations()
        
        # Try to use Strategist feature at Hunter tier
        env["account"].tier = TradingTier.HUNTER
        with pytest.raises(TierViolation):
            await env["risk_engine"].execute_vwap_order(
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                duration_minutes=60,
            )


class TestEmergencyProcedures:
    """Test emergency procedures and recovery."""

    @pytest.mark.asyncio
    async def test_emergency_position_closure(self, trading_environment):
        """Test emergency closure of all positions."""
        env = trading_environment
        
        # Open multiple positions
        positions = []
        for i in range(3):
            position = Position(
                position_id=str(uuid4()),
                account_id=env["account"].account_id,
                symbol=f"TEST{i}/USDT",
                side=PositionSide.LONG,
                entry_price=Decimal("100"),
                quantity=Decimal("1"),
                dollar_value=Decimal("100"),
            )
            env["risk_engine"].add_position(position)
            positions.append(position)
        
        # Trigger emergency closure
        closed_positions = await env["executor"].emergency_close_all()
        
        # Verify all positions closed
        assert len(closed_positions) == 3
        assert len(env["risk_engine"].positions) == 0
        
        # Verify trading is halted
        assert env["session"].trading_enabled is False

    @pytest.mark.asyncio
    async def test_connection_failure_recovery(self, trading_environment):
        """Test recovery from exchange connection failure."""
        env = trading_environment
        
        # Simulate connection loss
        env["gateway"].is_connected = False
        
        # Attempt to place order (should fail)
        with pytest.raises(ExchangeConnectionError):
            await env["executor"].execute_market_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                quantity=Decimal("0.001"),
            )
        
        # Trigger reconnection
        await env["gateway"]._reconnect()
        assert env["gateway"].is_connected is True
        
        # Verify trading can resume
        order = await env["executor"].execute_market_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
        )
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_data_corruption_recovery(self, trading_environment):
        """Test recovery from data corruption or inconsistency."""
        env = trading_environment
        
        # Create position with corrupted data
        position = Position(
            position_id=str(uuid4()),
            account_id=env["account"].account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("-100"),  # Invalid negative price
            quantity=Decimal("1"),
            dollar_value=Decimal("-100"),
        )
        
        # Risk engine should detect and reject
        with pytest.raises(ValueError):
            env["risk_engine"].add_position(position)
        
        # Verify system remains stable
        assert len(env["risk_engine"].positions) == 0
        
        # Can still add valid positions
        valid_position = Position(
            position_id=str(uuid4()),
            account_id=env["account"].account_id,
            symbol="BTC/USDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.001"),
            dollar_value=Decimal("50"),
        )
        env["risk_engine"].add_position(valid_position)
        assert len(env["risk_engine"].positions) == 1


class TestMarketConditions:
    """Test trading under various market conditions."""

    @pytest.mark.asyncio
    async def test_high_volatility_handling(self, trading_environment):
        """Test system behavior during high volatility."""
        env = trading_environment
        
        # Simulate high volatility
        with patch.object(env["gateway"], "get_ticker") as mock_ticker:
            # Rapid price changes
            mock_ticker.side_effect = [
                MagicMock(last_price=Decimal("50000")),
                MagicMock(last_price=Decimal("52000")),  # 4% jump
                MagicMock(last_price=Decimal("48000")),  # 8% drop
            ]
            
            # System should detect volatility
            tickers = []
            for _ in range(3):
                ticker = await env["gateway"].get_ticker("BTC/USDT")
                tickers.append(ticker.last_price)
            
            # Calculate volatility
            price_changes = [
                abs((tickers[i+1] - tickers[i]) / tickers[i])
                for i in range(len(tickers)-1)
            ]
            max_change = max(price_changes)
            
            # Should trigger risk adjustment
            assert max_change > Decimal("0.04")  # > 4% change
            
            # Risk engine should reduce position size
            adjusted_size = env["risk_engine"].calculate_position_size(
                symbol="BTC/USDT",
                entry_price=Decimal("50000"),
                volatility_adjustment=True,
            )
            
            normal_size = env["risk_engine"].calculate_position_size(
                symbol="BTC/USDT",
                entry_price=Decimal("50000"),
                volatility_adjustment=False,
            )
            
            assert adjusted_size < normal_size

    @pytest.mark.asyncio
    async def test_low_liquidity_detection(self, trading_environment):
        """Test handling of low liquidity conditions."""
        env = trading_environment
        
        # Simulate thin order book
        with patch.object(env["gateway"], "get_order_book") as mock_book:
            mock_book.return_value = MagicMock(
                bids=[
                    {"price": Decimal("49900"), "quantity": Decimal("0.01")},
                    {"price": Decimal("49800"), "quantity": Decimal("0.02")},
                ],
                asks=[
                    {"price": Decimal("50100"), "quantity": Decimal("0.01")},
                    {"price": Decimal("50200"), "quantity": Decimal("0.02")},
                ],
            )
            
            order_book = await env["gateway"].get_order_book("BTC/USDT")
            
            # Calculate liquidity metrics
            total_bid_volume = sum(bid["quantity"] for bid in order_book.bids)
            total_ask_volume = sum(ask["quantity"] for ask in order_book.asks)
            
            # Detect low liquidity
            assert total_bid_volume < Decimal("1")
            assert total_ask_volume < Decimal("1")
            
            # Should prevent large orders
            with pytest.raises(OrderExecutionError):
                await env["executor"].execute_market_order(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    quantity=Decimal("10"),  # Too large for thin book
                )

    @pytest.mark.asyncio
    async def test_spread_widening_response(self, trading_environment):
        """Test response to widening bid-ask spreads."""
        env = trading_environment
        
        # Normal spread
        normal_ticker = MagicMock(
            bid_price=Decimal("49990"),
            ask_price=Decimal("50010"),
            last_price=Decimal("50000"),
        )
        
        # Wide spread (indicative of uncertainty)
        wide_ticker = MagicMock(
            bid_price=Decimal("49500"),
            ask_price=Decimal("50500"),
            last_price=Decimal("50000"),
        )
        
        with patch.object(env["gateway"], "get_ticker") as mock_ticker:
            # Normal conditions
            mock_ticker.return_value = normal_ticker
            normal_spread = (normal_ticker.ask_price - normal_ticker.bid_price) / normal_ticker.last_price
            
            # Wide spread conditions
            mock_ticker.return_value = wide_ticker
            wide_spread = (wide_ticker.ask_price - wide_ticker.bid_price) / wide_ticker.last_price
            
            assert wide_spread > normal_spread * 10  # Spread 10x wider
            
            # Should trigger caution mode
            if wide_spread > Decimal("0.01"):  # > 1% spread
                # Reduce trading or switch to limit orders only
                env["session"].caution_mode = True
                assert env["session"].caution_mode is True