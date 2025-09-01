"""Unit tests for system health monitoring functionality."""

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

from genesis.monitoring.metrics_collector import MetricsCollector, TradingMetrics
from genesis.monitoring.prometheus_exporter import MetricsRegistry
from genesis.ui.widgets.system_health import SystemHealthWidget


class TestTradingMetrics:
    """Test TradingMetrics dataclass."""

    def test_system_metrics_initialization(self):
        """Test that system metrics are properly initialized."""
        metrics = TradingMetrics()
        
        # Check system health metrics
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_usage_percent == 0.0
        assert metrics.disk_io_read_bytes == 0.0
        assert metrics.disk_io_write_bytes == 0.0
        assert metrics.network_bytes_sent == 0.0
        assert metrics.network_bytes_recv == 0.0
        assert metrics.connection_count == 0
        assert metrics.thread_count == 0
        assert metrics.open_files == 0
        assert metrics.system_uptime == 0.0
        assert metrics.process_uptime == 0.0
        assert metrics.health_score == 100.0


class TestMetricsCollector:
    """Test MetricsCollector system health functionality."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock metrics registry."""
        registry = MagicMock(spec=MetricsRegistry)
        registry.set_gauge = AsyncMock()
        registry.register_collector = MagicMock()
        return registry

    @pytest.fixture
    def collector(self, mock_registry):
        """Create metrics collector instance."""
        return MetricsCollector(mock_registry)

    def test_health_score_calculation(self, collector):
        """Test health score calculation logic."""
        # Test with good metrics
        collector.metrics.cpu_usage = 30
        collector.metrics.memory_percent = 40
        collector.metrics.disk_usage_percent = 50
        collector.metrics.connection_count = 100
        collector.metrics.thread_count = 20
        collector.metrics.open_files = 100
        collector.metrics.rate_limit_usage = 30
        collector.metrics.tilt_score = 20
        
        collector._calculate_health_score()
        assert collector.metrics.health_score == 100.0
        
        # Test with high CPU usage
        collector.metrics.cpu_usage = 95
        collector._calculate_health_score()
        assert collector.metrics.health_score < 100.0
        assert collector.metrics.health_score >= 70.0
        
        # Test with multiple high metrics
        collector.metrics.memory_percent = 95
        collector.metrics.disk_usage_percent = 95
        collector._calculate_health_score()
        assert collector.metrics.health_score < 50.0

    @pytest.mark.asyncio
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.disk_io_counters')
    @patch('psutil.net_io_counters')
    @patch('psutil.boot_time')
    async def test_collect_system_metrics(
        self,
        mock_boot_time,
        mock_net_io,
        mock_disk_io,
        mock_disk_usage,
        mock_virtual_memory,
        mock_process_class,
        collector,
    ):
        """Test system metrics collection."""
        # Setup mocks
        mock_process = MagicMock()
        mock_process.cpu_percent.return_value = 45.5
        mock_process.memory_info.return_value = MagicMock(rss=1024 * 1024 * 512)  # 512 MB
        mock_process.num_threads.return_value = 10
        mock_process.open_files.return_value = []
        mock_process.connections.return_value = [1, 2, 3]  # 3 connections
        mock_process.create_time.return_value = time.time() - 3600  # 1 hour ago
        mock_process_class.return_value = mock_process
        
        mock_virtual_memory.return_value = MagicMock(total=1024 * 1024 * 1024 * 8)  # 8 GB
        mock_disk_usage.return_value = MagicMock(percent=65.0)
        
        mock_disk_io.return_value = MagicMock(
            read_bytes=1000000,
            write_bytes=500000,
        )
        
        mock_net_io.return_value = MagicMock(
            bytes_sent=2000000,
            bytes_recv=1500000,
            packets_sent=1000,
            packets_recv=800,
        )
        
        mock_boot_time.return_value = time.time() - 86400  # 1 day ago
        
        # Collect metrics
        await collector._collect_system_metrics()
        
        # Verify metrics were collected
        assert collector.metrics.cpu_usage == 45.5
        assert collector.metrics.memory_usage == 1024 * 1024 * 512
        assert collector.metrics.memory_percent == pytest.approx(6.25, rel=0.1)
        assert collector.metrics.disk_usage_percent == 65.0
        assert collector.metrics.connection_count == 3
        assert collector.metrics.thread_count == 10
        assert collector.metrics.process_uptime >= 3599  # Allow small variance
        assert collector.metrics.system_uptime >= 86399

    @pytest.mark.asyncio
    async def test_prometheus_metrics_export(self, collector, mock_registry):
        """Test that system health metrics are exported to Prometheus."""
        # Set test metrics
        collector.metrics.cpu_usage = 50.0
        collector.metrics.memory_usage = 1024 * 1024 * 1024  # 1 GB
        collector.metrics.memory_percent = 25.0
        collector.metrics.disk_usage_percent = 70.0
        collector.metrics.health_score = 85.0
        collector.metrics.connection_count = 10
        collector.metrics.thread_count = 5
        
        # Update Prometheus metrics
        await collector._update_prometheus_metrics()
        
        # Verify gauges were set
        mock_registry.set_gauge.assert_any_call(
            "genesis_cpu_usage_percent",
            50.0,
            {"type": "process"}
        )
        
        mock_registry.set_gauge.assert_any_call(
            "genesis_memory_usage_bytes",
            float(1024 * 1024 * 1024),
            {"type": "rss"}
        )
        
        mock_registry.set_gauge.assert_any_call(
            "genesis_memory_usage_percent",
            25.0
        )
        
        mock_registry.set_gauge.assert_any_call(
            "genesis_disk_usage_percent",
            70.0
        )
        
        mock_registry.set_gauge.assert_any_call(
            "genesis_health_score",
            85.0,
            {"description": "Overall system health score (0-100)"}
        )
        
        mock_registry.set_gauge.assert_any_call(
            "genesis_connection_count",
            10.0
        )
        
        mock_registry.set_gauge.assert_any_call(
            "genesis_thread_count",
            5.0
        )

    @pytest.mark.asyncio
    @patch('psutil.Process')
    async def test_metrics_collection_error_handling(self, mock_process_class, collector):
        """Test error handling in metrics collection."""
        # Simulate access denied
        mock_process = MagicMock()
        mock_process.cpu_percent.side_effect = psutil.AccessDenied("test")
        mock_process_class.return_value = mock_process
        
        # Should not raise, just log warning
        await collector._collect_system_metrics()
        
        # Simulate process not found
        mock_process_class.side_effect = psutil.NoSuchProcess(1234)
        
        # Should not raise, just log warning
        await collector._collect_system_metrics()


class TestSystemHealthWidget:
    """Test SystemHealthWidget functionality."""

    @pytest.fixture
    def widget(self):
        """Create system health widget instance."""
        return SystemHealthWidget()

    def test_init(self, widget):
        """Test widget initialization."""
        assert widget.cpu_usage == 0.0
        assert widget.memory_usage == 0.0
        assert widget.memory_percent == 0.0
        assert widget.disk_usage == 0.0
        assert widget.health_score == 100.0

    def test_update_metrics(self, widget):
        """Test updating widget metrics."""
        widget.update_metrics(
            cpu=75.0,
            memory_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
            memory_pct=50.0,
            disk_pct=60.0,
            net_in=1024 * 100,  # 100 KB/s
            net_out=1024 * 50,  # 50 KB/s
            connections=25,
            threads=12,
            files=100,
            health=70.0,
        )
        
        assert widget.cpu_usage == 75.0
        assert widget.memory_usage == 2 * 1024 * 1024 * 1024
        assert widget.memory_percent == 50.0
        assert widget.disk_usage == 60.0
        assert widget.network_in == 1024 * 100
        assert widget.network_out == 1024 * 50
        assert widget.connection_count == 25
        assert widget.thread_count == 12
        assert widget.open_files == 100
        assert widget.health_score == 70.0

    def test_render_with_good_health(self, widget):
        """Test rendering with good health metrics."""
        widget.update_metrics(
            cpu=30.0,
            memory_pct=40.0,
            disk_pct=50.0,
            health=85.0,
        )
        
        output = widget.render()
        assert "System Health" in output
        assert "Healthy" in output
        assert "85/100" in output

    def test_render_with_poor_health(self, widget):
        """Test rendering with poor health metrics."""
        widget.update_metrics(
            cpu=95.0,
            memory_pct=90.0,
            disk_pct=95.0,
            health=25.0,
        )
        
        output = widget.render()
        assert "Critical" in output
        assert "25/100" in output

    def test_format_bytes_per_sec(self, widget):
        """Test bytes per second formatting."""
        assert widget._format_bytes_per_sec(512) == "512 B/s"
        assert widget._format_bytes_per_sec(1024 * 5.5) == "5.5 KB/s"
        assert widget._format_bytes_per_sec(1024 * 1024 * 2.5) == "2.5 MB/s"
        assert widget._format_bytes_per_sec(1024 * 1024 * 1024 * 1.5) == "1.5 GB/s"

    def test_format_duration(self, widget):
        """Test duration formatting."""
        assert widget._format_duration(30) == "30s"
        assert widget._format_duration(90) == "2m"  # Rounds to minutes
        assert widget._format_duration(3600 * 2.5) == "2.5h"
        assert widget._format_duration(86400 * 3.5) == "3.5d"

    def test_get_usage_color(self, widget):
        """Test usage color determination."""
        assert widget._get_usage_color(25) == "green"
        assert widget._get_usage_color(55) == "yellow"
        assert widget._get_usage_color(75) == "orange"
        assert widget._get_usage_color(95) == "red"

    def test_create_progress_bar(self, widget):
        """Test progress bar creation."""
        bar = widget._create_progress_bar(50, 100, "green")
        assert "█" in bar
        assert "░" in bar
        
        # Test full bar
        bar = widget._create_progress_bar(100, 100, "red")
        assert "░" not in bar or bar.count("░") == 0
        
        # Test empty bar
        bar = widget._create_progress_bar(0, 100, "green")
        assert "█" not in bar or bar.count("█") == 0

    def test_mock_data(self, widget):
        """Test setting mock data."""
        widget.set_mock_data()
        
        assert widget.cpu_usage > 0
        assert widget.memory_usage > 0
        assert widget.health_score > 0
        
        output = widget.render()
        assert "System Health" in output


@pytest.mark.asyncio
async def test_integration_metrics_collector_lifecycle():
    """Test the full lifecycle of metrics collector."""
    # Create mock registry
    registry = MagicMock(spec=MetricsRegistry)
    registry.set_gauge = AsyncMock()
    registry.register_collector = MagicMock()
    
    # Create collector
    collector = MetricsCollector(registry)
    
    # Start collection
    await collector.start()
    assert collector._collection_task is not None
    
    # Let it run briefly
    await asyncio.sleep(0.1)
    
    # Stop collection
    await collector.stop()
    assert collector._collection_task is None