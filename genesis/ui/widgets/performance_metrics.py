"""Performance metrics dashboard widget for Genesis trading terminal."""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Optional

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


@dataclass
class LatencyPercentiles:
    """Latency percentile data."""
    p50: float  # Median
    p95: float  # 95th percentile
    p99: float  # 99th percentile
    p999: float  # 99.9th percentile
    max: float  # Maximum
    min: float  # Minimum
    avg: float  # Average


@dataclass 
class TradingVolumeMetrics:
    """Trading volume and frequency metrics."""
    total_volume_24h: float
    total_trades_24h: int
    avg_trade_size: float
    peak_volume_hour: float
    peak_trades_minute: int
    current_volume_hour: float
    current_trades_minute: int


class PerformanceMetricsWidget(Static):
    """Widget for displaying performance metrics and analytics."""
    
    # Latency metrics
    order_latency = reactive(None)  # LatencyPercentiles
    api_latency = reactive(None)  # LatencyPercentiles
    websocket_latency = reactive(None)  # LatencyPercentiles
    db_latency = reactive(None)  # LatencyPercentiles
    
    # Trading volume metrics
    volume_metrics = reactive(None)  # TradingVolumeMetrics
    
    # Throughput metrics
    orders_per_second = reactive(0.0)
    events_per_second = reactive(0.0)
    messages_per_second = reactive(0.0)
    
    # Resource efficiency
    cpu_efficiency = reactive(0.0)  # Orders processed per CPU %
    memory_efficiency = reactive(0.0)  # Orders per MB
    
    # Historical data
    latency_history = reactive(list)  # List of (timestamp, latency_ms) tuples
    throughput_history = reactive(list)  # List of (timestamp, ops/sec) tuples
    
    DEFAULT_CSS = """
    PerformanceMetricsWidget {
        content-align: center middle;
        padding: 1;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the performance metrics widget."""
        super().__init__("Loading performance metrics...", **kwargs)
        self.last_update = datetime.now(UTC)
        self.latency_samples = deque(maxlen=1000)  # Keep last 1000 samples
    
    def render(self) -> str:
        """Render the performance metrics display."""
        lines = [
            "[bold]═══ Performance Metrics ═══[/bold]",
            "",
            self._render_latency_summary(),
            "",
            self._render_latency_percentiles(),
            "",
            self._render_throughput_metrics(),
            "",
            self._render_volume_analytics(),
            "",
            self._render_efficiency_metrics(),
            "",
            self._render_performance_trend(),
            "",
            f"[dim]Last update: {self.last_update.strftime('%H:%M:%S')}[/dim]",
        ]
        
        return "\n".join(lines)
    
    def _render_latency_summary(self) -> str:
        """Render latency summary."""
        if not self.order_latency:
            return "[dim]No latency data available[/dim]"
        
        lines = ["[bold]Latency Overview:[/bold]"]
        
        # Determine health based on p95 latency
        if self.order_latency.p95 < 50:
            color = "green"
            status = "Excellent"
        elif self.order_latency.p95 < 100:
            color = "yellow"
            status = "Good"
        elif self.order_latency.p95 < 250:
            color = "orange"
            status = "Acceptable"
        else:
            color = "red"
            status = "Poor"
        
        lines.append(f"  [{color}]● {status}[/{color}] - Order P95: {self.order_latency.p95:.1f}ms")
        
        return "\n".join(lines)
    
    def _render_latency_percentiles(self) -> str:
        """Render detailed latency percentiles table."""
        lines = ["[bold]Latency Percentiles (ms):[/bold]"]
        
        # Create table
        table_lines = []
        table_lines.append("  Component    P50    P95    P99   P99.9    Max")
        table_lines.append("  " + "-" * 48)
        
        # Order execution latency
        if self.order_latency:
            table_lines.append(self._format_latency_row("Orders", self.order_latency))
        
        # API latency
        if self.api_latency:
            table_lines.append(self._format_latency_row("API", self.api_latency))
        
        # WebSocket latency
        if self.websocket_latency:
            table_lines.append(self._format_latency_row("WebSocket", self.websocket_latency))
        
        # Database latency
        if self.db_latency:
            table_lines.append(self._format_latency_row("Database", self.db_latency))
        
        lines.extend(table_lines)
        return "\n".join(lines)
    
    def _format_latency_row(self, name: str, latency: LatencyPercentiles) -> str:
        """Format a latency row for the table."""
        # Color code based on p95
        if latency.p95 < 50:
            p95_color = "green"
        elif latency.p95 < 100:
            p95_color = "yellow"
        elif latency.p95 < 250:
            p95_color = "orange"
        else:
            p95_color = "red"
        
        # Color code p99
        if latency.p99 < 100:
            p99_color = "green"
        elif latency.p99 < 500:
            p99_color = "yellow"
        else:
            p99_color = "red"
        
        return (f"  {name:12} {latency.p50:6.1f} "
                f"[{p95_color}]{latency.p95:6.1f}[/{p95_color}] "
                f"[{p99_color}]{latency.p99:6.1f}[/{p99_color}] "
                f"{latency.p999:6.1f} {latency.max:6.1f}")
    
    def _render_throughput_metrics(self) -> str:
        """Render throughput metrics."""
        lines = ["[bold]Throughput:[/bold]"]
        
        # Orders per second
        if self.orders_per_second > 100:
            ops_color = "green"
        elif self.orders_per_second > 50:
            ops_color = "yellow"
        else:
            ops_color = "dim"
        
        lines.append(f"  Orders/sec:   [{ops_color}]{self.orders_per_second:7.1f}[/{ops_color}] ops/s")
        lines.append(f"  Events/sec:   {self.events_per_second:7.1f} evt/s")
        lines.append(f"  Messages/sec: {self.messages_per_second:7.1f} msg/s")
        
        return "\n".join(lines)
    
    def _render_volume_analytics(self) -> str:
        """Render trading volume analytics."""
        if not self.volume_metrics:
            return ""
        
        lines = ["[bold]Trading Volume (24h):[/bold]"]
        
        # Format volume with appropriate units
        volume_str = self._format_volume(self.volume_metrics.total_volume_24h)
        
        lines.append(f"  Total Volume: {volume_str}")
        lines.append(f"  Total Trades: {self.volume_metrics.total_trades_24h:,}")
        lines.append(f"  Avg Size:     ${self.volume_metrics.avg_trade_size:,.2f}")
        
        # Current activity
        if self.volume_metrics.current_trades_minute > 10:
            activity_color = "green"
            activity = "High"
        elif self.volume_metrics.current_trades_minute > 5:
            activity_color = "yellow"
            activity = "Moderate"
        else:
            activity_color = "dim"
            activity = "Low"
        
        lines.append(f"  Activity:     [{activity_color}]{activity}[/{activity_color}] ({self.volume_metrics.current_trades_minute} trades/min)")
        
        return "\n".join(lines)
    
    def _render_efficiency_metrics(self) -> str:
        """Render resource efficiency metrics."""
        lines = ["[bold]Resource Efficiency:[/bold]"]
        
        # CPU efficiency
        if self.cpu_efficiency > 10:
            cpu_color = "green"
        elif self.cpu_efficiency > 5:
            cpu_color = "yellow"
        else:
            cpu_color = "orange"
        
        lines.append(f"  CPU Efficiency:    [{cpu_color}]{self.cpu_efficiency:.1f}[/{cpu_color}] orders/CPU%")
        lines.append(f"  Memory Efficiency: {self.memory_efficiency:.2f} orders/MB")
        
        return "\n".join(lines)
    
    def _render_performance_trend(self) -> str:
        """Render performance trend visualization."""
        if len(self.latency_history) < 2:
            return ""
        
        lines = ["[bold]Performance Trend:[/bold]"]
        
        # Create latency trend sparkline
        latencies = [lat for _, lat in self.latency_history[-20:]]
        if latencies:
            sparkline = self._create_sparkline(latencies)
            
            # Determine trend
            if len(latencies) >= 2:
                if latencies[-1] > latencies[-2] * 1.2:
                    trend_color = "red"
                    trend = "↑ Degrading"
                elif latencies[-1] < latencies[-2] * 0.8:
                    trend_color = "green"
                    trend = "↓ Improving"
                else:
                    trend_color = "dim"
                    trend = "→ Stable"
            else:
                trend_color = "dim"
                trend = "→ Stable"
            
            lines.append(f"  Latency: {sparkline} [{trend_color}]{trend}[/{trend_color}]")
        
        # Create throughput trend sparkline
        if self.throughput_history:
            throughputs = [tps for _, tps in self.throughput_history[-20:]]
            if throughputs:
                sparkline = self._create_sparkline(throughputs)
                lines.append(f"  Throughput: {sparkline}")
        
        return "\n".join(lines)
    
    def _create_sparkline(self, values: List[float]) -> str:
        """Create a sparkline from values."""
        if not values:
            return ""
        
        blocks = "▁▂▃▄▅▆▇█"
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return blocks[0] * min(20, len(values))
        
        sparkline = []
        for val in values:
            normalized = (val - min_val) / (max_val - min_val)
            index = min(7, int(normalized * 8))
            sparkline.append(blocks[index])
        
        return "".join(sparkline)
    
    def _format_volume(self, volume: float) -> str:
        """Format volume with appropriate units."""
        if volume >= 1_000_000_000:
            return f"${volume / 1_000_000_000:.2f}B"
        elif volume >= 1_000_000:
            return f"${volume / 1_000_000:.2f}M"
        elif volume >= 1_000:
            return f"${volume / 1_000:.2f}K"
        else:
            return f"${volume:.2f}"
    
    def update_latency(
        self,
        component: str,
        latency_ms: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update latency for a component."""
        if timestamp is None:
            timestamp = datetime.now(UTC)
        
        # Add to samples
        self.latency_samples.append((component, latency_ms, timestamp))
        
        # Recalculate percentiles
        self._calculate_percentiles()
        
        # Add to history
        self.latency_history.append((timestamp, latency_ms))
        
        # Keep only last hour
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        self.latency_history = [(ts, lat) for ts, lat in self.latency_history if ts > cutoff]
        
        self.last_update = datetime.now(UTC)
    
    def _calculate_percentiles(self) -> None:
        """Calculate latency percentiles from samples."""
        # Group samples by component
        component_samples = {}
        for component, latency, _ in self.latency_samples:
            if component not in component_samples:
                component_samples[component] = []
            component_samples[component].append(latency)
        
        # Calculate percentiles for each component
        for component, samples in component_samples.items():
            if len(samples) >= 10:  # Need enough samples
                sorted_samples = sorted(samples)
                n = len(sorted_samples)
                
                percentiles = LatencyPercentiles(
                    p50=sorted_samples[int(n * 0.50)],
                    p95=sorted_samples[int(n * 0.95)],
                    p99=sorted_samples[int(n * 0.99)],
                    p999=sorted_samples[min(int(n * 0.999), n-1)],
                    max=sorted_samples[-1],
                    min=sorted_samples[0],
                    avg=sum(samples) / len(samples),
                )
                
                # Update the appropriate metric
                if component == "order":
                    self.order_latency = percentiles
                elif component == "api":
                    self.api_latency = percentiles
                elif component == "websocket":
                    self.websocket_latency = percentiles
                elif component == "database":
                    self.db_latency = percentiles
    
    def update_throughput(
        self,
        orders_per_sec: Optional[float] = None,
        events_per_sec: Optional[float] = None,
        messages_per_sec: Optional[float] = None,
    ) -> None:
        """Update throughput metrics."""
        if orders_per_sec is not None:
            self.orders_per_second = orders_per_sec
            self.throughput_history.append((datetime.now(UTC), orders_per_sec))
        
        if events_per_sec is not None:
            self.events_per_second = events_per_sec
        
        if messages_per_sec is not None:
            self.messages_per_second = messages_per_sec
        
        # Keep only last hour
        cutoff = datetime.now(UTC) - timedelta(hours=1)
        self.throughput_history = [(ts, tps) for ts, tps in self.throughput_history if ts > cutoff]
    
    def update_volume_metrics(self, metrics: TradingVolumeMetrics) -> None:
        """Update trading volume metrics."""
        self.volume_metrics = metrics
    
    def update_efficiency(
        self,
        cpu_efficiency: Optional[float] = None,
        memory_efficiency: Optional[float] = None,
    ) -> None:
        """Update efficiency metrics."""
        if cpu_efficiency is not None:
            self.cpu_efficiency = cpu_efficiency
        if memory_efficiency is not None:
            self.memory_efficiency = memory_efficiency
    
    def set_mock_data(self) -> None:
        """Set mock data for testing."""
        # Set latency percentiles
        self.order_latency = LatencyPercentiles(
            p50=25.5, p95=45.2, p99=89.3, p999=125.6, max=250.0, min=5.2, avg=28.7
        )
        self.api_latency = LatencyPercentiles(
            p50=15.3, p95=28.9, p99=55.2, p999=95.4, max=150.0, min=3.1, avg=18.2
        )
        self.websocket_latency = LatencyPercentiles(
            p50=2.1, p95=4.5, p99=8.9, p999=15.2, max=25.0, min=0.5, avg=2.8
        )
        self.db_latency = LatencyPercentiles(
            p50=5.2, p95=12.3, p99=25.6, p999=45.2, max=75.0, min=1.2, avg=7.5
        )
        
        # Set throughput
        self.orders_per_second = 125.5
        self.events_per_second = 850.3
        self.messages_per_second = 2150.7
        
        # Set volume metrics
        self.volume_metrics = TradingVolumeMetrics(
            total_volume_24h=5_250_000,
            total_trades_24h=12_500,
            avg_trade_size=420.0,
            peak_volume_hour=450_000,
            peak_trades_minute=25,
            current_volume_hour=180_000,
            current_trades_minute=8,
        )
        
        # Set efficiency
        self.cpu_efficiency = 8.5
        self.memory_efficiency = 0.45
        
        # Add history
        now = datetime.now(UTC)
        for i in range(20):
            timestamp = now - timedelta(minutes=i * 3)
            latency = 25 + (i % 5) * 5
            throughput = 100 + (i % 3) * 25
            self.latency_history.append((timestamp, latency))
            self.throughput_history.append((timestamp, throughput))
    
    def watch_orders_per_second(self, value: float) -> None:
        """React to throughput changes."""
        self.update(self.render())