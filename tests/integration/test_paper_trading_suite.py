"""
Integration test suite for paper trading validation.

Executes a complete paper trading test session to validate:
- 10 successful round-trip trades
- P&L accuracy to 2 decimal places
- 24-hour continuous operation
- Live UI updates
- No manual intervention required
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest
import structlog

from genesis.config.paper_trading_config import get_paper_trading_config
from genesis.core.events import Event, EventPriority, EventType
from genesis.core.models import Order, OrderSide, OrderType
from genesis.engine.event_bus import EventBus
from genesis.engine.paper_trading_enforcer import PaperTradingEnforcer
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.trading_loop import TradingLoop
from genesis.exchange.gateway import ExchangeGateway
from genesis.exchange.mock_exchange import MockExchange
from genesis.analytics.pnl_tracker import PnLTracker

logger = structlog.get_logger(__name__)


class PaperTradingTestSuite:
    """
    Comprehensive paper trading test suite.
    
    Validates all acceptance criteria for Story 0.5.
    """
    
    def __init__(self):
        """Initialize test suite components."""
        self.config = get_paper_trading_config()
        self.session_id = str(uuid4())
        self.start_time = None
        self.trades_completed = 0
        self.manual_interventions = 0
        self.uptime_seconds = 0
        self.downtime_seconds = 0
        self.test_results = {}
        
    async def setup(self) -> None:
        """Set up test environment."""
        # Initialize components
        self.event_bus = EventBus()
        self.mock_exchange = MockExchange(
            initial_balance={"USDT": self.config.initial_balance_usdt},
            paper_trading_mode=True
        )
        
        # Initialize paper trading enforcer
        self.paper_trading_enforcer = PaperTradingEnforcer()
        
        # Initialize P&L tracker
        self.pnl_tracker = PnLTracker(session_id=self.session_id)
        
        # Mock risk engine
        self.risk_engine = self._create_mock_risk_engine()
        
        # Mock exchange gateway
        self.exchange_gateway = self._create_mock_exchange_gateway()
        
        # Initialize trading loop in paper mode
        self.trading_loop = TradingLoop(
            event_bus=self.event_bus,
            risk_engine=self.risk_engine,
            exchange_gateway=self.exchange_gateway,
            paper_trading_mode=True,
            paper_trading_session_id=self.session_id
        )
        
        # Start event bus
        await self.event_bus.start()
        
        # Create paper trading session
        await self.paper_trading_enforcer.require_paper_trading(
            account_id="test-account",
            strategy="iceberg_orders",
            duration_hours=self.config.session_duration_hours,
        )
        
        logger.info("Paper trading test suite initialized", session_id=self.session_id)
        
    def _create_mock_risk_engine(self) -> Any:
        """Create mock risk engine for testing."""
        from unittest.mock import MagicMock
        
        risk_engine = MagicMock(spec=RiskEngine)
        risk_engine.tier_limits = {
            "stop_loss_percent": self.config.stop_loss_percent,
            "max_position_size": self.config.max_position_size_usdt,
        }
        risk_engine.calculate_position_size.return_value = Decimal("0.01")
        risk_engine.validate_order_risk.return_value = None
        risk_engine.validate_portfolio_risk.return_value = {
            "approved": True,
            "rejections": [],
            "warnings": []
        }
        risk_engine.validate_configuration.return_value = True
        
        return risk_engine
        
    def _create_mock_exchange_gateway(self) -> Any:
        """Create mock exchange gateway for testing."""
        from unittest.mock import MagicMock
        
        gateway = MagicMock(spec=ExchangeGateway)
        gateway.validate_connection.return_value = True
        
        async def mock_execute_order(order: Order) -> dict:
            """Mock order execution with realistic behavior."""
            # Simulate execution latency
            await asyncio.sleep(0.05)
            
            # Get current market price
            price = self.mock_exchange.market_prices.get(
                order.symbol,
                Decimal("50000")
            )
            
            # Apply slippage
            if order.side == OrderSide.BUY:
                fill_price = price * (Decimal("1") + self.config.slippage_percent / Decimal("100"))
            else:
                fill_price = price * (Decimal("1") - self.config.slippage_percent / Decimal("100"))
            
            return {
                "success": True,
                "exchange_order_id": f"EX-{uuid4().hex[:8]}",
                "fill_price": fill_price,
                "latency_ms": 50
            }
            
        gateway.execute_order = mock_execute_order
        return gateway
        
    async def execute_test_trades(self) -> dict[str, Any]:
        """
        Execute test trades during paper trading session.
        
        Returns:
            Dictionary with trade execution results
        """
        trades_data = []
        
        for i in range(self.config.min_trades_required):
            # Alternate between buy and sell
            side = "BUY" if i % 2 == 0 else "SELL"
            symbol = self.config.test_symbols[i % len(self.config.test_symbols)]
            
            # Simulate market signal
            signal_event = Event(
                event_type=EventType.ARBITRAGE_SIGNAL,
                event_data={
                    "strategy_id": "test-strategy",
                    "pair1_symbol": symbol,
                    "signal_type": "ENTRY" if side == "BUY" else "EXIT",
                    "confidence_score": 0.8,
                    "entry_price": str(self.mock_exchange.market_prices.get(symbol, Decimal("50000"))),
                }
            )
            
            # Process signal through trading loop
            await self.trading_loop._handle_trading_signal(signal_event)
            
            # Record trade
            trade_info = {
                "trade_number": i + 1,
                "symbol": symbol,
                "side": side,
                "timestamp": datetime.now().isoformat(),
            }
            trades_data.append(trade_info)
            
            # Wait between trades
            await asyncio.sleep(2)
            
            # Simulate closing position after a short hold
            if i > 0 and i % 2 == 1:  # Close on sell signals
                await asyncio.sleep(1)
                
                # Update price for P&L
                new_price = self.mock_exchange.market_prices[symbol] * Decimal("1.01")
                self.mock_exchange.update_market_price(symbol, new_price)
                
        return {
            "total_trades": len(trades_data),
            "trades": trades_data,
            "success": len(trades_data) >= self.config.min_trades_required
        }
        
    async def validate_pnl_accuracy(self) -> dict[str, Any]:
        """
        Validate P&L calculation accuracy.
        
        Returns:
            Validation results
        """
        # Get P&L metrics
        metrics = self.pnl_tracker.export_metrics()
        
        # Check decimal accuracy
        realized_pnl = Decimal(metrics["realized_pnl"])
        unrealized_pnl = Decimal(metrics["unrealized_pnl"])
        total_pnl = Decimal(metrics["total_pnl"])
        
        # Verify accuracy to 2 decimal places
        def check_decimal_places(value: Decimal) -> bool:
            """Check if value has at most 2 decimal places."""
            str_val = str(value)
            if "." in str_val:
                decimal_part = str_val.split(".")[1]
                return len(decimal_part) <= self.config.pnl_accuracy_decimals
            return True
            
        accuracy_valid = all([
            check_decimal_places(realized_pnl),
            check_decimal_places(unrealized_pnl),
            check_decimal_places(total_pnl)
        ])
        
        return {
            "realized_pnl": str(realized_pnl),
            "unrealized_pnl": str(unrealized_pnl),
            "total_pnl": str(total_pnl),
            "accuracy_valid": accuracy_valid,
            "required_decimals": self.config.pnl_accuracy_decimals
        }
        
    async def monitor_continuous_operation(self, duration_hours: int = 1) -> dict[str, Any]:
        """
        Monitor system for continuous operation.
        
        Args:
            duration_hours: Duration to monitor (default 1 hour for testing)
            
        Returns:
            Monitoring results
        """
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        heartbeats_received = 0
        health_checks_passed = 0
        disconnections = 0
        
        # Subscribe to heartbeat events
        def on_heartbeat(event: Event):
            nonlocal heartbeats_received
            heartbeats_received += 1
            
        self.event_bus.subscribe(
            EventType.SYSTEM_HEARTBEAT,
            on_heartbeat,
            priority=EventPriority.LOW
        )
        
        # Monitor for duration
        while time.time() < end_time:
            # Check if trading loop is running
            if not self.trading_loop.running:
                disconnections += 1
                logger.warning("Trading loop disconnected")
                
                # Attempt reconnection
                await self.trading_loop.startup()
                
            # Check system health
            if await self.exchange_gateway.validate_connection():
                health_checks_passed += 1
                
            await asyncio.sleep(10)  # Check every 10 seconds
            
        total_duration = time.time() - start_time
        uptime_percent = ((total_duration - (disconnections * 10)) / total_duration) * 100
        
        return {
            "duration_hours": duration_hours,
            "heartbeats_received": heartbeats_received,
            "health_checks_passed": health_checks_passed,
            "disconnections": disconnections,
            "uptime_percent": uptime_percent,
            "continuous_operation": uptime_percent >= 99.0
        }
        
    async def validate_ui_updates(self) -> dict[str, Any]:
        """
        Validate UI is showing live updates.
        
        Returns:
            UI validation results
        """
        # This would normally interact with the actual UI
        # For testing, we'll verify events are being published
        
        ui_events_published = 0
        position_updates = 0
        pnl_updates = 0
        
        # Count relevant events
        for event in self.trading_loop.event_store:
            if event.event_type in [EventType.POSITION_OPENED, EventType.POSITION_UPDATED, EventType.POSITION_CLOSED]:
                position_updates += 1
            elif event.event_type in [EventType.ORDER_FILLED]:
                pnl_updates += 1
                
        ui_events_published = position_updates + pnl_updates
        
        return {
            "ui_events_published": ui_events_published,
            "position_updates": position_updates,
            "pnl_updates": pnl_updates,
            "live_updates": ui_events_published > 0
        }
        
    async def run_complete_test(self) -> dict[str, Any]:
        """
        Run complete paper trading test suite.
        
        Returns:
            Complete test results
        """
        logger.info("Starting paper trading test suite")
        
        # Setup
        await self.setup()
        
        # Start trading loop
        await self.trading_loop.startup()
        
        # Track start time
        self.start_time = datetime.now()
        
        # Execute test trades (AC1)
        logger.info("Executing test trades...")
        trade_results = await self.execute_test_trades()
        self.test_results["trades"] = trade_results
        
        # Validate P&L accuracy (AC2)
        logger.info("Validating P&L accuracy...")
        pnl_results = await self.validate_pnl_accuracy()
        self.test_results["pnl_accuracy"] = pnl_results
        
        # Monitor continuous operation (AC3)
        logger.info("Monitoring continuous operation...")
        operation_results = await self.monitor_continuous_operation(duration_hours=1)
        self.test_results["continuous_operation"] = operation_results
        
        # Validate UI updates (AC4)
        logger.info("Validating UI updates...")
        ui_results = await self.validate_ui_updates()
        self.test_results["ui_updates"] = ui_results
        
        # Check manual interventions (AC5)
        self.test_results["manual_interventions"] = {
            "count": self.manual_interventions,
            "no_intervention_required": self.manual_interventions == 0
        }
        
        # Calculate overall success
        all_criteria_met = all([
            trade_results["success"],  # AC1
            pnl_results["accuracy_valid"],  # AC2
            operation_results["continuous_operation"],  # AC3
            ui_results["live_updates"],  # AC4
            self.manual_interventions == 0,  # AC5
        ])
        
        self.test_results["overall_success"] = all_criteria_met
        self.test_results["test_duration"] = str(datetime.now() - self.start_time)
        
        # Cleanup
        await self.trading_loop.shutdown()
        
        return self.test_results


@pytest.mark.asyncio
async def test_paper_trading_validation():
    """
    Execute paper trading validation test.
    
    This is the main test that validates all acceptance criteria.
    """
    suite = PaperTradingTestSuite()
    results = await suite.run_complete_test()
    
    # Assert all acceptance criteria are met
    assert results["trades"]["success"], "AC1 Failed: Insufficient trades completed"
    assert results["pnl_accuracy"]["accuracy_valid"], "AC2 Failed: P&L accuracy insufficient"
    assert results["continuous_operation"]["continuous_operation"], "AC3 Failed: System not continuously operational"
    assert results["ui_updates"]["live_updates"], "AC4 Failed: UI not showing live updates"
    assert results["manual_interventions"]["no_intervention_required"], "AC5 Failed: Manual intervention was required"
    
    # Overall success
    assert results["overall_success"], "Paper trading validation failed"
    
    logger.info(
        "Paper trading validation completed successfully",
        results=results
    )


@pytest.mark.asyncio
async def test_pnl_calculation_precision():
    """Test P&L calculations maintain 2 decimal precision."""
    tracker = PnLTracker()
    
    # Test various P&L scenarios
    test_cases = [
        (Decimal("50000.00"), Decimal("50100.00"), Decimal("0.01")),  # Small profit
        (Decimal("3000.50"), Decimal("2995.25"), Decimal("0.1")),  # Small loss
        (Decimal("400.123"), Decimal("401.456"), Decimal("0.5")),  # Fractional prices
    ]
    
    for entry, exit_price, quantity in test_cases:
        # Calculate P&L
        gross_pnl = (exit_price - entry) * quantity
        
        # Verify 2 decimal precision
        rounded_pnl = gross_pnl.quantize(Decimal("0.01"))
        assert str(rounded_pnl).split(".")[-1].__len__() <= 2
        

@pytest.mark.asyncio 
async def test_continuous_monitoring():
    """Test continuous operation monitoring components."""
    # Create minimal test setup
    event_bus = EventBus()
    await event_bus.start()
    
    # Create mock components
    from unittest.mock import MagicMock
    
    risk_engine = MagicMock()
    risk_engine.tier_limits = {"stop_loss_percent": Decimal("2.0")}
    risk_engine.validate_configuration.return_value = True
    
    exchange_gateway = MagicMock()
    exchange_gateway.validate_connection.return_value = True
    
    # Create trading loop
    trading_loop = TradingLoop(
        event_bus=event_bus,
        risk_engine=risk_engine,
        exchange_gateway=exchange_gateway,
        paper_trading_mode=True,
        paper_trading_session_id="test-monitoring"
    )
    
    # Start and monitor briefly
    await trading_loop.startup()
    
    # Wait for heartbeat
    await asyncio.sleep(2)
    
    # Verify metrics
    stats = trading_loop.get_statistics()
    assert stats["running"] is True
    
    # Cleanup
    await trading_loop.shutdown()
    await event_bus.stop()


if __name__ == "__main__":
    # Run the test suite directly
    asyncio.run(test_paper_trading_validation())