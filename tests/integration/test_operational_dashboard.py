"""Integration tests for the complete operational dashboard (Story 8.9)."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from genesis.monitoring.error_budget import SLO, ErrorBudgetStatus
from genesis.monitoring.metrics_collector import MetricsCollector, TradingMetrics
from genesis.monitoring.prometheus_exporter import MetricsRegistry
from genesis.monitoring.rate_limit_metrics import RateLimitMetricsCollector
from genesis.tilt.detector import TiltLevel
from genesis.ui.widgets.pnl import PnLWidget
from genesis.ui.widgets.positions import PositionWidget
from genesis.ui.widgets.rate_limit_viz import (
    RateLimitEndpoint,
    RateLimitWidget,
    CircuitBreakerStatus,
    TokenBucketVisualization,
)
from genesis.ui.widgets.system_health import SystemHealthWidget
from genesis.ui.widgets.tilt_indicator import TiltIndicator


class TestOperationalDashboard:
    """Test the complete operational dashboard functionality."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock metrics registry."""
        registry = MagicMock(spec=MetricsRegistry)
        registry.set_gauge = AsyncMock()
        registry.set_counter = AsyncMock()
        registry.set_histogram = AsyncMock()
        registry.register_collector = MagicMock()
        return registry

    @pytest.fixture
    def metrics_collector(self, mock_registry):
        """Create metrics collector instance."""
        return MetricsCollector(mock_registry)

    @pytest.fixture
    def rate_limit_collector(self):
        """Create rate limit metrics collector."""
        return RateLimitMetricsCollector()

    def test_dashboard_widgets_initialization(self):
        """Test that all dashboard widgets initialize correctly."""
        # P&L Widget
        pnl_widget = PnLWidget()
        assert pnl_widget.current_pnl == Decimal("0.00")
        assert pnl_widget.max_drawdown == Decimal("0.00")
        assert pnl_widget.sharpe_ratio == Decimal("0.00")
        
        # Position Widget
        position_widget = PositionWidget()
        assert not position_widget.has_position
        assert position_widget.risk_reward_ratio == Decimal("0.00")
        
        # System Health Widget
        health_widget = SystemHealthWidget()
        assert health_widget.cpu_usage == 0.0
        assert health_widget.health_score == 100.0
        
        # Tilt Indicator
        tilt_widget = TiltIndicator()
        assert tilt_widget.tilt_level == TiltLevel.NORMAL
        assert tilt_widget.tilt_score == 0
        assert len(tilt_widget.tilt_history) == 0
        
        # Rate Limit Widget
        rate_widget = RateLimitWidget()
        assert rate_widget.total_utilization == 0.0
        assert len(rate_widget.endpoints) == 0

    def test_pnl_widget_with_historical_data(self):
        """Test P&L widget with historical tracking."""
        widget = PnLWidget()
        
        # Add historical data points
        for i in range(10):
            value = Decimal(str(100 * (i + 1)))
            widget.add_pnl_data_point(value)
        
        # Calculate metrics
        widget.calculate_risk_metrics()
        
        # Verify history tracking
        assert len(widget.pnl_data_points) == 10
        assert widget.pnl_data_points[-1] == Decimal("1000")
        
        # Test chart rendering
        chart = widget._render_pnl_chart()
        assert "P&L Trend:" in chart
        assert chart != "[dim]No historical data available[/dim]"

    def test_position_widget_risk_metrics(self):
        """Test position widget with comprehensive risk metrics."""
        widget = PositionWidget()
        
        # Set a position with stop loss
        widget.set_mock_position(
            symbol="ETH/USDT",
            side="LONG",
            qty=Decimal("10"),
            entry=Decimal("2000"),
            current=Decimal("2100"),
            stop_loss=Decimal("1900"),
        )
        
        # Calculate risk metrics
        widget.calculate_risk_metrics()
        
        # Verify risk calculations
        assert widget.has_position
        assert widget.unrealized_pnl == Decimal("1000")  # (2100-2000) * 10
        assert widget.risk_reward_ratio == Decimal("2")  # Default 2:1
        
        # Update time in position
        widget.entry_time = datetime.now(UTC) - timedelta(hours=1)
        widget.calculate_risk_metrics()
        assert widget.time_in_position > 3500  # More than 1 hour in seconds
        
        # Test risk metrics rendering
        metrics = widget._render_risk_metrics()
        assert len(metrics) > 0
        metrics_str = "\n".join(metrics)
        assert "R:R Ratio" in metrics_str
        assert "2.00:1" in metrics_str

    def test_system_health_monitoring(self, metrics_collector):
        """Test system health monitoring and scoring."""
        # Set various system metrics
        metrics_collector.metrics.cpu_usage = 45.0
        metrics_collector.metrics.memory_percent = 35.0
        metrics_collector.metrics.disk_usage_percent = 55.0
        metrics_collector.metrics.connection_count = 50
        metrics_collector.metrics.thread_count = 15
        metrics_collector.metrics.open_files = 100
        metrics_collector.metrics.rate_limit_usage = 40.0
        metrics_collector.metrics.tilt_score = 25.0
        
        # Calculate health score
        metrics_collector._calculate_health_score()
        
        # Should have good health score with these metrics
        assert metrics_collector.metrics.health_score == 100.0
        
        # Test with degraded metrics
        metrics_collector.metrics.cpu_usage = 95.0
        metrics_collector.metrics.memory_percent = 92.0
        metrics_collector._calculate_health_score()
        
        # Health score should be reduced
        assert metrics_collector.metrics.health_score < 50.0

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, metrics_collector, mock_registry):
        """Test comprehensive system metrics collection."""
        with patch('psutil.Process') as mock_process_class:
            # Setup mock process
            mock_process = MagicMock()
            mock_process.cpu_percent.return_value = 55.5
            mock_process.memory_info.return_value = MagicMock(rss=2 * 1024 * 1024 * 1024)  # 2GB
            mock_process.num_threads.return_value = 20
            mock_process.open_files.return_value = [1, 2, 3, 4, 5]  # 5 files
            mock_process.connections.return_value = list(range(15))  # 15 connections
            mock_process.create_time.return_value = time.time() - 7200  # 2 hours ago
            mock_process_class.return_value = mock_process
            
            with patch('psutil.virtual_memory') as mock_vm:
                mock_vm.return_value = MagicMock(total=8 * 1024 * 1024 * 1024)  # 8GB
                
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk.return_value = MagicMock(percent=45.0)
                    
                    with patch('psutil.boot_time') as mock_boot:
                        mock_boot.return_value = time.time() - 86400 * 5  # 5 days ago
                        
                        # Collect metrics
                        await metrics_collector._collect_system_metrics()
            
            # Verify metrics were collected
            assert metrics_collector.metrics.cpu_usage == 55.5
            assert metrics_collector.metrics.memory_usage == 2 * 1024 * 1024 * 1024
            assert metrics_collector.metrics.memory_percent == 25.0
            assert metrics_collector.metrics.disk_usage_percent == 45.0
            assert metrics_collector.metrics.connection_count == 15
            assert metrics_collector.metrics.thread_count == 20
            assert metrics_collector.metrics.open_files == 5
            
            # Verify Prometheus export
            await metrics_collector._update_prometheus_metrics()
            
            # Check key metrics were exported
            mock_registry.set_gauge.assert_any_call(
                "genesis_health_score",
                100.0,  # Should be healthy with these metrics
                {"description": "Overall system health score (0-100)"}
            )

    def test_tilt_indicator_with_history(self):
        """Test tilt indicator with history tracking."""
        widget = TiltIndicator()
        
        # Add tilt events to history
        now = datetime.now(UTC)
        widget.tilt_history.append((now - timedelta(minutes=30), TiltLevel.NORMAL, 10))
        widget.tilt_history.append((now - timedelta(minutes=20), TiltLevel.LEVEL1, 35))
        widget.tilt_history.append((now - timedelta(minutes=10), TiltLevel.LEVEL2, 65))
        widget.tilt_history.append((now, TiltLevel.LEVEL1, 40))
        
        widget.tilt_events_today = 4
        widget.max_tilt_score_today = 65
        widget.tilt_level = TiltLevel.LEVEL1
        widget.tilt_score = 40
        
        # Add intervention history
        widget.intervention_history.append((now - timedelta(minutes=20), "Take a breath"))
        widget.intervention_history.append((now - timedelta(minutes=10), "Position sizes reduced"))
        
        # Test history rendering
        history_panel = widget._render_tilt_history()
        assert history_panel is not None
        
        # Test status summary
        summary = widget.get_status_summary()
        assert summary["level"] == "LEVEL1"
        assert summary["score"] == 40
        assert summary["trading_allowed"] is True
        
        # Test position size multiplier
        assert widget.position_size_multiplier == Decimal("1.0")
        widget.tilt_level = TiltLevel.LEVEL2
        assert widget.position_size_multiplier == Decimal("0.5")

    def test_rate_limit_visualization(self):
        """Test rate limit visualization widget."""
        widget = RateLimitWidget()
        
        # Add endpoints
        endpoint1 = RateLimitEndpoint(
            name="POST /api/v3/order",
            tokens_used=800,
            tokens_available=200,
            tokens_capacity=1000,
            utilization_percent=80.0,
            window="1m",
            requests_allowed=800,
            requests_rejected=5,
        )
        widget.add_endpoint(endpoint1)
        
        endpoint2 = RateLimitEndpoint(
            name="GET /api/v3/ticker",
            tokens_used=300,
            tokens_available=700,
            tokens_capacity=1000,
            utilization_percent=30.0,
            window="1m",
            requests_allowed=300,
        )
        widget.add_endpoint(endpoint2)
        
        # Add circuit breakers
        cb1 = CircuitBreakerStatus(
            name="order_api",
            state="closed",
            failure_count=1,
            success_count=99,
        )
        widget.add_circuit_breaker(cb1)
        
        cb2 = CircuitBreakerStatus(
            name="websocket",
            state="open",
            failure_count=10,
            success_count=5,
        )
        widget.add_circuit_breaker(cb2)
        
        # Update sliding windows
        widget.update_sliding_windows(
            weight_1m=900,
            limit_1m=1200,
            orders_10s=40,
            limit_10s=50,
        )
        
        # Verify calculations
        assert len(widget.endpoints) == 2
        assert len(widget.circuit_breakers) == 2
        assert widget.total_utilization == 55.0  # Weighted average
        
        # Test rendering
        output = widget.render()
        assert "Rate Limit Status" in output
        assert "POST /api/v3/order" in output
        assert "80.0%" in output
        assert "Circuit Breakers" in output
        assert "OPEN" in output  # Websocket is open

    def test_token_bucket_visualization(self):
        """Test token bucket visualization."""
        bucket = TokenBucketVisualization(
            bucket_name="orders",
            capacity=100,
        )
        
        # Test with full bucket
        bucket.tokens = 100
        content = bucket._render_bucket()
        assert "100/100" in str(content)
        assert "100.0%" in str(content)
        
        # Test with partial bucket
        bucket.update_tokens(50)
        content = bucket._render_bucket()
        assert "50/100" in str(content)
        assert "50.0%" in str(content)
        
        # Test with empty bucket
        bucket.update_tokens(0)
        content = bucket._render_bucket()
        assert "0/100" in str(content)
        assert "0.0%" in str(content)

    def test_rate_limit_metrics_update(self, rate_limit_collector):
        """Test rate limit metrics collector."""
        # Update rate limiter metrics
        metrics = {
            'requests_allowed': 100,
            'requests_rejected': 5,
            'requests_queued': 10,
            'token_bucket_tokens': 700,
            'token_bucket_capacity': 1000,
            'critical_overrides': 2,
        }
        
        with patch('genesis.monitoring.rate_limit_metrics.rate_limit_requests_total') as mock_counter:
            with patch('genesis.monitoring.rate_limit_metrics.rate_limit_tokens_available') as mock_gauge:
                rate_limit_collector.update_rate_limiter_metrics("test_limiter", metrics)
                
                # Verify metrics were updated
                assert mock_counter.labels.called
                assert mock_gauge.labels.called
        
        # Update circuit breaker metrics
        cb_status = {
            'state': 'open',
            'failure_count': 10,
            'success_count': 90,
            'circuit_open_duration': 30.5,
        }
        
        with patch('genesis.monitoring.rate_limit_metrics.circuit_breaker_state') as mock_state:
            rate_limit_collector.update_circuit_breaker_metrics("test_circuit", cb_status)
            assert mock_state.labels.called

    @pytest.mark.asyncio
    async def test_full_dashboard_integration(self, metrics_collector, mock_registry):
        """Test full dashboard integration with all components."""
        # Initialize all widgets
        pnl_widget = PnLWidget()
        position_widget = PositionWidget()
        health_widget = SystemHealthWidget()
        tilt_widget = TiltIndicator()
        rate_widget = RateLimitWidget()
        
        # Simulate trading session data
        pnl_widget.set_mock_data(
            current=Decimal("2500"),
            daily=Decimal("500"),
            balance=Decimal("50000"),
        )
        
        position_widget.set_mock_position(
            symbol="BTC/USDT",
            side="LONG",
            qty=Decimal("0.5"),
            entry=Decimal("40000"),
            current=Decimal("42000"),
            stop_loss=Decimal("38000"),
        )
        
        health_widget.set_mock_data()
        rate_widget.set_mock_data()
        
        # Update tilt status
        tilt_widget.tilt_level = TiltLevel.NORMAL
        tilt_widget.tilt_score = 15
        
        # Verify all widgets can render
        assert pnl_widget.render()
        assert position_widget.render()
        assert health_widget.render()
        assert rate_widget.render()
        
        # Simulate metrics collection
        metrics_collector.metrics.cpu_usage = 40.0
        metrics_collector.metrics.memory_percent = 30.0
        metrics_collector.metrics.health_score = 85.0
        
        # Update Prometheus metrics
        await metrics_collector._update_prometheus_metrics()
        
        # Verify comprehensive metrics export
        assert mock_registry.set_gauge.call_count > 15  # Many metrics exported
        
        # Verify health score calculation
        metrics_collector._calculate_health_score()
        assert metrics_collector.metrics.health_score > 50.0

    def test_error_scenarios(self):
        """Test error handling in dashboard components."""
        # Test with invalid data
        pnl_widget = PnLWidget()
        pnl_widget.add_pnl_data_point(Decimal("0"))
        pnl_widget.calculate_risk_metrics()  # Should not crash with zero values
        
        # Test position widget with no position
        position_widget = PositionWidget()
        position_widget.calculate_risk_metrics()  # Should handle no position gracefully
        assert position_widget.risk_reward_ratio == Decimal("0")
        
        # Test rate limit widget with empty data
        rate_widget = RateLimitWidget()
        rate_widget._recalculate_total_utilization()  # Should handle empty endpoints
        assert rate_widget.total_utilization == 0.0
        
        # Test system health with extreme values
        health_widget = SystemHealthWidget()
        health_widget.update_metrics(
            cpu=150.0,  # Invalid percentage
            memory_pct=200.0,  # Invalid percentage
        )
        # Widget should handle invalid values gracefully
        output = health_widget.render()
        assert output is not None


@pytest.mark.asyncio
async def test_dashboard_real_time_updates():
    """Test real-time update capabilities of dashboard."""
    # Create widgets
    pnl_widget = PnLWidget()
    health_widget = SystemHealthWidget()
    
    # Simulate real-time updates
    for i in range(5):
        # Update P&L
        pnl_widget.add_pnl_data_point(Decimal(str(100 * (i + 1))))
        pnl_widget.current_pnl = Decimal(str(100 * (i + 1)))
        
        # Update health metrics
        health_widget.update_metrics(
            cpu=30.0 + i * 5,
            memory_pct=20.0 + i * 3,
            health=100.0 - i * 5,
        )
        
        # Small delay to simulate real-time
        await asyncio.sleep(0.01)
    
    # Verify final state
    assert len(pnl_widget.pnl_data_points) == 5
    assert pnl_widget.current_pnl == Decimal("500")
    assert health_widget.cpu_usage == 50.0
    assert health_widget.memory_percent == 32.0
    assert health_widget.health_score == 80.0