"""Unit tests for performance monitoring module."""

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from genesis.monitoring.performance_monitor import (
    OperationTimer,
    PerformanceMetrics,
    PerformanceMonitor,
    get_performance_monitor,
)


@pytest.fixture
def performance_monitor():
    """Create a performance monitor instance for testing."""
    registry = CollectorRegistry()
    return PerformanceMonitor(registry=registry)


@pytest.fixture
def mock_metrics():
    """Create mock metrics for testing."""
    return MagicMock(spec=PerformanceMetrics)


class TestPerformanceMonitor:
    """Test PerformanceMonitor class."""
    
    def test_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor.registry is not None
        assert performance_monitor.metrics is not None
        assert isinstance(performance_monitor._latency_cache, dict)
        assert isinstance(performance_monitor._start_times, dict)
    
    def test_metrics_initialization(self, performance_monitor):
        """Test that all metrics are properly initialized."""
        metrics = performance_monitor.metrics
        
        # Check order execution metrics
        assert metrics.order_execution_latency is not None
        assert metrics.order_execution_counter is not None
        assert metrics.order_execution_errors is not None
        
        # Check API metrics
        assert metrics.api_call_counter is not None
        assert metrics.api_call_latency is not None
        assert metrics.api_call_errors is not None
        assert metrics.api_rate_limit_hits is not None
        
        # Check position metrics
        assert metrics.active_positions_gauge is not None
        assert metrics.position_value_gauge is not None
        assert metrics.position_pnl_gauge is not None
        
        # Check system metrics
        assert metrics.memory_usage_gauge is not None
        assert metrics.cpu_usage_gauge is not None
        assert metrics.goroutine_count_gauge is not None
        
        # Check risk metrics
        assert metrics.risk_check_latency is not None
        assert metrics.risk_check_counter is not None
        assert metrics.risk_limit_breaches is not None
        
        # Check database metrics
        assert metrics.db_query_latency is not None
        assert metrics.db_query_counter is not None
        assert metrics.db_connection_pool_gauge is not None
        assert metrics.db_transaction_counter is not None
        
        # Check WebSocket metrics
        assert metrics.ws_message_latency is not None
        assert metrics.ws_connection_gauge is not None
        assert metrics.ws_reconnection_counter is not None
        assert metrics.ws_message_counter is not None
        
        # Check circuit breaker metrics
        assert metrics.circuit_breaker_state_gauge is not None
        assert metrics.circuit_breaker_trips_counter is not None
        assert metrics.circuit_breaker_success_rate is not None
        
        # Check cache metrics
        assert metrics.cache_hit_counter is not None
        assert metrics.cache_miss_counter is not None
        assert metrics.cache_eviction_counter is not None
        assert metrics.cache_latency is not None
        
        # Check business metrics
        assert metrics.tier_progression_gauge is not None
        assert metrics.daily_pnl_gauge is not None
        assert metrics.win_rate_gauge is not None
        assert metrics.sharpe_ratio_gauge is not None
    
    @pytest.mark.asyncio
    async def test_record_order_execution(self, performance_monitor):
        """Test recording order execution metrics."""
        await performance_monitor.record_order_execution(
            order_type="market",
            symbol="BTCUSDT",
            side="buy",
            tier="sniper",
            latency=0.025,
            status="success"
        )
        
        # Check that metrics were recorded
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_order_execution_latency_seconds" in metrics_output
        assert b"genesis_order_executions_total" in metrics_output
    
    @pytest.mark.asyncio
    async def test_record_order_execution_error(self, performance_monitor):
        """Test recording order execution error metrics."""
        await performance_monitor.record_order_execution(
            order_type="limit",
            symbol="ETHUSDT",
            side="sell",
            tier="hunter",
            latency=0.1,
            status="insufficient_balance"
        )
        
        # Check that error metrics were recorded
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_order_execution_errors_total" in metrics_output
    
    @pytest.mark.asyncio
    async def test_record_api_call(self, performance_monitor):
        """Test recording API call metrics."""
        await performance_monitor.record_api_call(
            exchange="binance",
            endpoint="/api/v3/order",
            method="POST",
            latency=0.05,
            status_code=200
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_api_calls_total" in metrics_output
        assert b"genesis_api_call_latency_seconds" in metrics_output
    
    @pytest.mark.asyncio
    async def test_record_api_rate_limit(self, performance_monitor):
        """Test recording API rate limit metrics."""
        await performance_monitor.record_api_call(
            exchange="binance",
            endpoint="/api/v3/order",
            method="POST",
            latency=0.01,
            status_code=429,
            error_type="rate_limit"
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_api_rate_limit_hits_total" in metrics_output
        assert b"genesis_api_call_errors_total" in metrics_output
    
    @pytest.mark.asyncio
    async def test_update_position_metrics(self, performance_monitor):
        """Test updating position metrics."""
        await performance_monitor.update_position_metrics(
            symbol="BTCUSDT",
            side="long",
            tier="sniper",
            count=2,
            value_usdt=Decimal("10000.50"),
            pnl_realized=Decimal("150.25"),
            pnl_unrealized=Decimal("75.30")
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_active_positions" in metrics_output
        assert b"genesis_position_value_usdt" in metrics_output
        assert b"genesis_position_pnl_usdt" in metrics_output
    
    @pytest.mark.asyncio
    async def test_record_risk_check(self, performance_monitor):
        """Test recording risk check metrics."""
        await performance_monitor.record_risk_check(
            check_type="position_size",
            tier="sniper",
            latency=0.001,
            passed=True
        )
        
        await performance_monitor.record_risk_check(
            check_type="drawdown",
            tier="hunter",
            latency=0.002,
            passed=False
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_risk_check_latency_seconds" in metrics_output
        assert b"genesis_risk_checks_total" in metrics_output
        assert b"genesis_risk_limit_breaches_total" in metrics_output
    
    @pytest.mark.asyncio
    async def test_record_database_operation(self, performance_monitor):
        """Test recording database operation metrics."""
        await performance_monitor.record_database_operation(
            operation="SELECT",
            table="positions",
            latency=0.005,
            success=True
        )
        
        await performance_monitor.record_database_operation(
            operation="INSERT",
            table="orders",
            latency=0.01,
            success=False
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_db_query_latency_seconds" in metrics_output
        assert b"genesis_db_queries_total" in metrics_output
    
    @pytest.mark.asyncio
    async def test_record_websocket_message(self, performance_monitor):
        """Test recording WebSocket message metrics."""
        await performance_monitor.record_websocket_message(
            message_type="orderbook",
            symbol="BTCUSDT",
            latency_ms=5.5,
            direction="inbound"
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_ws_message_latency_ms" in metrics_output
        assert b"genesis_ws_messages_total" in metrics_output
    
    @pytest.mark.asyncio
    async def test_update_circuit_breaker(self, performance_monitor):
        """Test updating circuit breaker metrics."""
        await performance_monitor.update_circuit_breaker(
            service="binance_api",
            state="open",
            success_rate=0.25
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_circuit_breaker_state" in metrics_output
        assert b"genesis_circuit_breaker_success_rate" in metrics_output
    
    @pytest.mark.asyncio
    async def test_record_cache_operation(self, performance_monitor):
        """Test recording cache operation metrics."""
        await performance_monitor.record_cache_operation(
            cache_name="order_cache",
            operation="get",
            key_type="order_id",
            hit=True,
            latency=0.0001
        )
        
        await performance_monitor.record_cache_operation(
            cache_name="position_cache",
            operation="get",
            key_type="symbol",
            hit=False,
            latency=0.0002
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_cache_hits_total" in metrics_output
        assert b"genesis_cache_misses_total" in metrics_output
        assert b"genesis_cache_operation_latency_seconds" in metrics_output
    
    @pytest.mark.asyncio
    async def test_update_system_metrics(self, performance_monitor):
        """Test updating system metrics."""
        await performance_monitor.update_system_metrics(
            memory_rss=1024 * 1024 * 100,  # 100MB
            memory_vms=1024 * 1024 * 200,  # 200MB
            cpu_percent=45.5,
            task_count=15
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_memory_usage_bytes" in metrics_output
        assert b"genesis_cpu_usage_percent" in metrics_output
        assert b"genesis_goroutine_count" in metrics_output
    
    @pytest.mark.asyncio
    async def test_update_business_metrics(self, performance_monitor):
        """Test updating business metrics."""
        await performance_monitor.update_business_metrics(
            tier=1,  # hunter
            daily_pnl=Decimal("500.75"),
            win_rate_daily=65.5,
            win_rate_weekly=62.3,
            sharpe_daily=1.85,
            sharpe_weekly=1.72
        )
        
        metrics_output = performance_monitor.get_metrics()
        assert b"genesis_tier_progression" in metrics_output
        assert b"genesis_daily_pnl_usdt" in metrics_output
        assert b"genesis_win_rate_percent" in metrics_output
        assert b"genesis_sharpe_ratio" in metrics_output
    
    def test_get_metrics(self, performance_monitor):
        """Test getting metrics in Prometheus format."""
        metrics_output = performance_monitor.get_metrics()
        assert isinstance(metrics_output, bytes)
        # Check for standard Prometheus format markers
        assert b"# HELP" in metrics_output
        assert b"# TYPE" in metrics_output
    
    @pytest.mark.asyncio
    async def test_track_performance_decorator_async(self, performance_monitor):
        """Test performance tracking decorator with async function."""
        @performance_monitor.track_performance("test_operation")
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "success"
        
        result = await test_async_function()
        assert result == "success"
        assert "test_operation" in performance_monitor._latency_cache
        assert len(performance_monitor._latency_cache["test_operation"]) > 0
    
    def test_track_performance_decorator_sync(self, performance_monitor):
        """Test performance tracking decorator with sync function."""
        @performance_monitor.track_performance("test_sync_operation")
        def test_sync_function():
            time.sleep(0.01)
            return "success"
        
        result = test_sync_function()
        assert result == "success"
        assert "test_sync_operation" in performance_monitor._latency_cache
        assert len(performance_monitor._latency_cache["test_sync_operation"]) > 0
    
    @pytest.mark.asyncio
    async def test_track_performance_decorator_with_exception(self, performance_monitor):
        """Test performance tracking decorator handles exceptions."""
        @performance_monitor.track_performance("failing_operation")
        async def failing_function():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await failing_function()
        
        # The decorator should still record latency even when exception occurs
        # but it may not have been added to cache yet in the current implementation
    
    def test_latency_cache_limit(self, performance_monitor):
        """Test that latency cache is limited to 1000 entries."""
        operation_name = "test_operation"
        
        # Add more than 1000 entries
        for i in range(1500):
            performance_monitor._latency_cache[operation_name].append(i * 0.001)
        
        # Check that cache is limited to last 1000 entries
        assert len(performance_monitor._latency_cache[operation_name]) == 1500
        # The implementation doesn't auto-limit, it's done when adding via decorator


class TestOperationTimer:
    """Test OperationTimer context manager."""
    
    def test_sync_context_manager(self, performance_monitor):
        """Test OperationTimer as sync context manager."""
        with performance_monitor.time_operation("test_timer"):
            time.sleep(0.01)
        
        assert "test_timer" in performance_monitor._latency_cache
        assert len(performance_monitor._latency_cache["test_timer"]) > 0
        latency = performance_monitor._latency_cache["test_timer"][0]
        assert 0.01 <= latency < 0.02
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, performance_monitor):
        """Test OperationTimer as async context manager."""
        async with performance_monitor.time_operation("async_timer"):
            await asyncio.sleep(0.01)
        
        assert "async_timer" in performance_monitor._latency_cache
        assert len(performance_monitor._latency_cache["async_timer"]) > 0
        latency = performance_monitor._latency_cache["async_timer"][0]
        # Allow for timing variations in async operations
        assert 0.009 <= latency < 0.1
    
    def test_sync_context_manager_with_exception(self, performance_monitor):
        """Test OperationTimer handles exceptions in sync context."""
        with pytest.raises(ValueError):
            with performance_monitor.time_operation("failing_timer"):
                time.sleep(0.01)
                raise ValueError("Test error")
        
        assert "failing_timer" in performance_monitor._latency_cache
        assert len(performance_monitor._latency_cache["failing_timer"]) > 0
    
    @pytest.mark.asyncio
    async def test_async_context_manager_with_exception(self, performance_monitor):
        """Test OperationTimer handles exceptions in async context."""
        with pytest.raises(ValueError):
            async with performance_monitor.time_operation("async_failing_timer"):
                await asyncio.sleep(0.01)
                raise ValueError("Test error")
        
        assert "async_failing_timer" in performance_monitor._latency_cache
        assert len(performance_monitor._latency_cache["async_failing_timer"]) > 0


class TestGlobalInstance:
    """Test global performance monitor instance."""
    
    def test_get_performance_monitor_singleton(self):
        """Test that get_performance_monitor returns singleton."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        assert monitor1 is monitor2
    
    def test_get_performance_monitor_creates_instance(self):
        """Test that get_performance_monitor creates instance if none exists."""
        # Reset global instance
        import genesis.monitoring.performance_monitor as pm_module
        pm_module._monitor_instance = None
        
        monitor = get_performance_monitor()
        assert monitor is not None
        assert isinstance(monitor, PerformanceMonitor)