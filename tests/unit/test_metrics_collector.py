"""Unit tests for metrics collector."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4

from genesis.monitoring.metrics_collector import (
    TradingMetrics,
    MetricsCollector
)
from genesis.monitoring.prometheus_exporter import MetricsRegistry
from genesis.core.models import Order, OrderStatus, Position
from genesis.core.events import Event, EventType


class TestTradingMetrics:
    """Test TradingMetrics dataclass."""
    
    def test_trading_metrics_creation(self):
        """Test creating TradingMetrics."""
        metrics = TradingMetrics()
        
        assert metrics.orders_placed == 0
        assert metrics.orders_filled == 0
        assert metrics.trades_executed == 0
        assert metrics.realized_pnl == Decimal("0")
        assert metrics.current_positions == 0
        assert metrics.win_rate == 0.0
        assert metrics.websocket_connected == False
    
    def test_trading_metrics_defaults(self):
        """Test TradingMetrics default values."""
        metrics = TradingMetrics()
        
        assert isinstance(metrics.total_volume, Decimal)
        assert metrics.total_volume == Decimal("0")
        assert isinstance(metrics.tilt_indicators, dict)
        assert len(metrics.tilt_indicators) == 0
        assert isinstance(metrics.last_update, datetime)


@pytest.mark.asyncio
class TestMetricsCollector:
    """Test MetricsCollector class."""
    
    async def test_collector_creation(self):
        """Test creating metrics collector."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        assert collector.registry == registry
        assert isinstance(collector.metrics, TradingMetrics)
        assert collector._collection_interval == 10
    
    async def test_start_stop_collector(self):
        """Test starting and stopping collector."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Start collector
        await collector.start()
        assert collector._collection_task is not None
        assert not collector._collection_task.done()
        
        # Stop collector
        await collector.stop()
        assert collector._collection_task is None
    
    async def test_record_order_placed(self):
        """Test recording placed order."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Register counter metrics
        await registry.register(MagicMock(name="genesis_orders_total", type=MagicMock()))
        
        order = MagicMock(spec=Order)
        order.client_order_id = "test-123"
        order.status = OrderStatus.NEW
        
        await collector.record_order(order)
        
        assert collector.metrics.orders_placed == 1
    
    async def test_record_order_filled(self):
        """Test recording filled order."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Register counter metrics
        await registry.register(MagicMock(name="genesis_orders_total", type=MagicMock()))
        await registry.register(MagicMock(name="genesis_trades_total", type=MagicMock()))
        
        order = MagicMock(spec=Order)
        order.client_order_id = "test-456"
        order.status = OrderStatus.FILLED
        
        await collector.record_order(order)
        
        assert collector.metrics.orders_placed == 1
        assert collector.metrics.orders_filled == 1
        assert collector.metrics.trades_executed == 1
    
    async def test_record_order_cancelled(self):
        """Test recording cancelled order."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Register counter metric
        await registry.register(MagicMock(name="genesis_orders_total", type=MagicMock()))
        
        order = MagicMock(spec=Order)
        order.client_order_id = "test-789"
        order.status = OrderStatus.CANCELLED
        
        await collector.record_order(order)
        
        assert collector.metrics.orders_placed == 1
        assert collector.metrics.orders_cancelled == 1
    
    async def test_record_order_failed(self):
        """Test recording failed order."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Register counter metrics
        await registry.register(MagicMock(name="genesis_orders_total", type=MagicMock()))
        await registry.register(MagicMock(name="genesis_orders_failed_total", type=MagicMock()))
        
        order = MagicMock(spec=Order)
        order.client_order_id = "test-fail"
        order.status = OrderStatus.REJECTED
        
        await collector.record_order(order)
        
        assert collector.metrics.orders_placed == 1
        assert collector.metrics.orders_failed == 1
    
    async def test_record_execution_time(self):
        """Test recording execution time."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Mock histogram metric
        registry.observe_histogram = AsyncMock()
        
        await collector.record_execution_time(0.125)
        
        registry.observe_histogram.assert_called_once_with(
            "genesis_order_execution_time_seconds",
            0.125
        )
    
    async def test_record_websocket_latency(self):
        """Test recording WebSocket latency."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Mock histogram metric
        registry.observe_histogram = AsyncMock()
        
        await collector.record_websocket_latency(15.5)
        
        assert collector.metrics.websocket_latency_ms == 15.5
        assert 15.5 in collector._latency_samples
        registry.observe_histogram.assert_called_once_with(
            "genesis_websocket_latency_ms",
            15.5
        )
    
    async def test_update_position_metrics(self):
        """Test updating position metrics."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Create mock positions
        positions = [
            MagicMock(spec=Position, unrealized_pnl=Decimal("100.50")),
            MagicMock(spec=Position, unrealized_pnl=Decimal("50.25")),
            MagicMock(spec=Position, unrealized_pnl=Decimal("-25.00"))
        ]
        
        await collector.update_position_metrics(positions)
        
        assert collector.metrics.current_positions == 3
        assert collector.metrics.unrealized_pnl == Decimal("125.75")
    
    async def test_update_pnl(self):
        """Test updating P&L metrics."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        await collector.update_pnl(Decimal("1000.00"), Decimal("250.50"))
        
        assert collector.metrics.realized_pnl == Decimal("1000.00")
        assert collector.metrics.unrealized_pnl == Decimal("250.50")
    
    async def test_update_connection_status(self):
        """Test updating connection status."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        await collector.update_connection_status(True, "binance")
        assert collector.metrics.websocket_connected == True
        
        await collector.update_connection_status(False, "binance")
        assert collector.metrics.websocket_connected == False
    
    async def test_update_rate_limits(self):
        """Test updating rate limit metrics."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        await collector.update_rate_limits(800, 1000)
        
        assert collector.metrics.rate_limit_usage == 0.8
        assert collector.metrics.rate_limit_remaining == 200
    
    async def test_update_rate_limits_warning(self):
        """Test rate limit warning when usage is high."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        with patch('genesis.monitoring.metrics_collector.logger') as mock_logger:
            await collector.update_rate_limits(850, 1000)
            
            # Should log warning for high usage
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "High rate limit usage" in call_args[0][0]
    
    async def test_update_tilt_score(self):
        """Test updating tilt score."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        indicators = {
            "click_speed": 0.8,
            "cancel_rate": 0.6,
            "revenge_trading": 0.9
        }
        
        await collector.update_tilt_score(65.0, indicators)
        
        assert collector.metrics.tilt_score == 65.0
        assert collector.metrics.tilt_indicators == indicators
    
    async def test_update_tilt_score_warning(self):
        """Test tilt score warning when high."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        with patch('genesis.monitoring.metrics_collector.logger') as mock_logger:
            await collector.update_tilt_score(75.0, {})
            
            # Should log warning for high tilt
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            assert "High tilt score detected" in call_args[0][0]
    
    async def test_record_trade(self):
        """Test recording a trade."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        trade = {
            "symbol": "BTC/USDT",
            "side": "buy",
            "quantity": 0.1,
            "price": 50000,
            "pnl": 100,
            "return": 0.02
        }
        
        await collector.record_trade(trade)
        
        assert len(collector._trades_history) == 1
        assert collector._trades_history[0]["symbol"] == "BTC/USDT"
        assert collector._trades_history[0]["pnl"] == 100
    
    async def test_calculate_win_rate(self):
        """Test win rate calculation."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Add winning and losing trades
        await collector.record_trade({"pnl": 100, "return": 0.02})
        await collector.record_trade({"pnl": -50, "return": -0.01})
        await collector.record_trade({"pnl": 75, "return": 0.015})
        
        collector.metrics.trades_executed = 3
        await collector._calculate_derived_metrics()
        
        # 2 wins out of 3 trades
        assert collector.metrics.win_rate == 2/3
    
    async def test_calculate_drawdown(self):
        """Test drawdown calculation."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Set initial peak
        collector._peak_balance = Decimal("10000")
        collector.metrics.realized_pnl = Decimal("8500")
        collector.metrics.unrealized_pnl = Decimal("0")
        
        await collector._calculate_derived_metrics()
        
        # (10000 - 8500) / 10000 * 100 = 15%
        assert collector.metrics.current_drawdown == 15.0
        assert collector.metrics.max_drawdown >= 15.0
    
    async def test_handle_event_order_placed(self):
        """Test handling order placed event."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Mock record_order
        collector.record_order = AsyncMock()
        
        order = MagicMock(spec=Order)
        event = Event(
            type=EventType.ORDER_PLACED,
            data={"order": order}
        )
        
        await collector.handle_event(event)
        collector.record_order.assert_called_once_with(order)
    
    async def test_handle_event_position_opened(self):
        """Test handling position opened event."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        event = Event(
            type=EventType.POSITION_OPENED,
            data={}
        )
        
        await collector.handle_event(event)
        assert collector.metrics.positions_opened == 1
    
    async def test_handle_event_connection_lost(self):
        """Test handling connection lost event."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Mock update_connection_status
        collector.update_connection_status = AsyncMock()
        
        event = Event(
            type=EventType.CONNECTION_LOST,
            data={}
        )
        
        await collector.handle_event(event)
        collector.update_connection_status.assert_called_once_with(False)
    
    async def test_handle_event_tilt_warning(self):
        """Test handling tilt warning event."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Mock update_tilt_score
        collector.update_tilt_score = AsyncMock()
        
        event = Event(
            type=EventType.TILT_WARNING,
            data={"score": 75.0, "indicators": {"test": 0.5}}
        )
        
        await collector.handle_event(event)
        collector.update_tilt_score.assert_called_once_with(75.0, {"test": 0.5})
    
    async def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Set some metrics
        collector.metrics.orders_placed = 10
        collector.metrics.orders_filled = 8
        collector.metrics.current_positions = 2
        collector.metrics.realized_pnl = Decimal("500.00")
        collector.metrics.win_rate = 0.75
        collector.metrics.websocket_connected = True
        
        summary = collector.get_metrics_summary()
        
        assert summary["orders"]["placed"] == 10
        assert summary["orders"]["filled"] == 8
        assert summary["positions"]["current"] == 2
        assert summary["pnl"]["realized"] == 500.0
        assert summary["performance"]["win_rate"] == 0.75
        assert summary["connection"]["websocket"] == True
        assert "last_update" in summary
    
    @patch('genesis.monitoring.metrics_collector.psutil')
    async def test_collect_system_metrics(self, mock_psutil):
        """Test collecting system metrics."""
        registry = MetricsRegistry()
        collector = MetricsCollector(registry)
        
        # Mock psutil
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 25.5
        mock_process.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_process.connections.return_value = [1, 2, 3]  # 3 connections
        mock_psutil.Process.return_value = mock_process
        
        await collector._collect_system_metrics()
        
        assert collector.metrics.cpu_usage == 25.5
        assert collector.metrics.memory_usage == 1024 * 1024 * 100
        assert collector.metrics.connection_count == 3