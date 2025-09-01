"""System health monitoring widget for Genesis trading terminal."""

from datetime import UTC, datetime
from decimal import Decimal

from rich.console import RenderableType
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class SystemHealthWidget(Static):
    """Widget for displaying system health metrics and status."""

    # System metrics (reactive for auto-updates)
    cpu_usage = reactive(0.0)
    memory_usage = reactive(0.0)
    memory_percent = reactive(0.0)
    disk_usage = reactive(0.0)
    network_in = reactive(0.0)  # bytes/sec
    network_out = reactive(0.0)  # bytes/sec
    network_recv_rate = reactive(0.0)  # Alias for network_in
    network_send_rate = reactive(0.0)  # Alias for network_out
    connection_count = reactive(0)
    thread_count = reactive(0)
    open_files = reactive(0)
    
    # Health score
    health_score = reactive(100.0)
    
    # Process info
    process_uptime = reactive(0.0)  # seconds
    system_uptime = reactive(0.0)  # seconds
    
    DEFAULT_CSS = """
    SystemHealthWidget {
        content-align: center middle;
        padding: 1;
    }
    """

    def __init__(self, **kwargs):
        """Initialize the system health widget."""
        super().__init__("Loading system health...", **kwargs)
        self.last_update = datetime.now(UTC)

    def render(self) -> str:
        """Render the system health display."""
        # Determine health status color
        if self.health_score >= 80:
            health_color = "green"
            health_status = "Healthy"
        elif self.health_score >= 60:
            health_color = "yellow"
            health_status = "Warning"
        elif self.health_score >= 40:
            health_color = "orange"
            health_status = "Degraded"
        else:
            health_color = "red"
            health_status = "Critical"

        # Build display
        lines = [
            "[bold]═══ System Health ═══[/bold]",
            "",
            f"[{health_color}]● {health_status}[/{health_color}] - Score: {self.health_score:.0f}/100",
            "",
            self._render_resource_bars(),
            "",
            self._render_metrics_table(),
            "",
            self._render_uptime_info(),
        ]

        return "\n".join(lines)

    def _render_resource_bars(self) -> str:
        """Render resource usage progress bars."""
        lines = []
        
        # CPU bar
        cpu_color = self._get_usage_color(self.cpu_usage)
        cpu_bar = self._create_progress_bar(self.cpu_usage, 100, cpu_color)
        lines.append(f"CPU:    {cpu_bar} {self.cpu_usage:5.1f}%")
        
        # Memory bar
        mem_color = self._get_usage_color(self.memory_percent)
        mem_bar = self._create_progress_bar(self.memory_percent, 100, mem_color)
        mem_mb = self.memory_usage / (1024 * 1024)
        lines.append(f"Memory: {mem_bar} {self.memory_percent:5.1f}% ({mem_mb:.0f} MB)")
        
        # Disk bar
        disk_color = self._get_usage_color(self.disk_usage)
        disk_bar = self._create_progress_bar(self.disk_usage, 100, disk_color)
        lines.append(f"Disk:   {disk_bar} {self.disk_usage:5.1f}%")
        
        return "\n".join(lines)

    def _render_metrics_table(self) -> str:
        """Render detailed metrics in a table format."""
        lines = [
            "[bold]Network & Process:[/bold]",
            f"  Network In:  {self._format_bytes_per_sec(self.network_in)}",
            f"  Network Out: {self._format_bytes_per_sec(self.network_out)}",
            f"  Connections: {self.connection_count}",
            f"  Threads:     {self.thread_count}",
            f"  Open Files:  {self.open_files}",
        ]
        return "\n".join(lines)

    def _render_uptime_info(self) -> str:
        """Render uptime information."""
        process_uptime_str = self._format_duration(self.process_uptime)
        system_uptime_str = self._format_duration(self.system_uptime)
        
        return f"[dim]Process: {process_uptime_str} | System: {system_uptime_str}[/dim]"

    def _create_progress_bar(self, value: float, max_value: float, color: str) -> str:
        """Create a simple text-based progress bar."""
        bar_width = 20
        filled = int((value / max_value) * bar_width) if max_value > 0 else 0
        filled = min(bar_width, max(0, filled))
        empty = bar_width - filled
        
        return f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"

    def _get_usage_color(self, percentage: float) -> str:
        """Get color based on usage percentage."""
        if percentage >= 90:
            return "red"
        elif percentage >= 70:
            return "orange"
        elif percentage >= 50:
            return "yellow"
        else:
            return "green"

    def _format_bytes_per_sec(self, bytes_per_sec: float) -> str:
        """Format bytes per second for display."""
        if bytes_per_sec < 1024:
            return f"{bytes_per_sec:.0f} B/s"
        elif bytes_per_sec < 1024 * 1024:
            return f"{bytes_per_sec / 1024:.1f} KB/s"
        elif bytes_per_sec < 1024 * 1024 * 1024:
            return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
        else:
            return f"{bytes_per_sec / (1024 * 1024 * 1024):.1f} GB/s"

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"

    def update_metrics(
        self,
        cpu: float = None,
        memory_bytes: float = None,
        memory_pct: float = None,
        disk_pct: float = None,
        net_in: float = None,
        net_out: float = None,
        connections: int = None,
        threads: int = None,
        files: int = None,
        proc_uptime: float = None,
        sys_uptime: float = None,
        health: float = None,
    ) -> None:
        """Update system metrics.
        
        Args:
            cpu: CPU usage percentage
            memory_bytes: Memory usage in bytes
            memory_pct: Memory usage percentage
            disk_pct: Disk usage percentage
            net_in: Network input bytes/sec
            net_out: Network output bytes/sec
            connections: Number of network connections
            threads: Number of threads
            files: Number of open files
            proc_uptime: Process uptime in seconds
            sys_uptime: System uptime in seconds
            health: Overall health score (0-100)
        """
        if cpu is not None:
            self.cpu_usage = cpu
        if memory_bytes is not None:
            self.memory_usage = memory_bytes
        if memory_pct is not None:
            self.memory_percent = memory_pct
        if disk_pct is not None:
            self.disk_usage = disk_pct
        if net_in is not None:
            self.network_in = net_in
        if net_out is not None:
            self.network_out = net_out
        if connections is not None:
            self.connection_count = connections
        if threads is not None:
            self.thread_count = threads
        if files is not None:
            self.open_files = files
        if proc_uptime is not None:
            self.process_uptime = proc_uptime
        if sys_uptime is not None:
            self.system_uptime = sys_uptime
        if health is not None:
            self.health_score = health
            
        self.last_update = datetime.now(UTC)

    def watch_health_score(self, value: float) -> None:
        """React to health score changes."""
        self.update(self.render())

    def watch_cpu_usage(self, value: float) -> None:
        """React to CPU usage changes."""
        self.update(self.render())

    def watch_memory_percent(self, value: float) -> None:
        """React to memory changes."""
        self.update(self.render())

    def set_mock_data(self) -> None:
        """Set mock data for testing."""
        self.update_metrics(
            cpu=35.5,
            memory_bytes=512 * 1024 * 1024,  # 512 MB
            memory_pct=25.0,
            disk_pct=45.0,
            net_in=1024 * 50,  # 50 KB/s
            net_out=1024 * 20,  # 20 KB/s
            connections=15,
            threads=8,
            files=42,
            proc_uptime=3600 * 2.5,  # 2.5 hours
            sys_uptime=3600 * 24 * 5,  # 5 days
            health=85.0,
        )


class SystemHealthPanel(Panel):
    """Rich panel wrapper for system health display."""
    
    def __init__(self, metrics_collector=None, **kwargs):
        """Initialize system health panel.
        
        Args:
            metrics_collector: Optional MetricsCollector instance
        """
        self.metrics_collector = metrics_collector
        super().__init__(self._render_content(), title="System Health", **kwargs)
        
    def _render_content(self) -> RenderableType:
        """Render the panel content."""
        if not self.metrics_collector:
            return Text("No metrics collector configured", style="dim")
            
        metrics = self.metrics_collector.metrics
        
        # Create health indicator
        health_score = metrics.health_score
        if health_score >= 80:
            health_icon = "✓"
            health_color = "green"
            health_text = "Healthy"
        elif health_score >= 60:
            health_icon = "⚠"
            health_color = "yellow"
            health_text = "Warning"
        else:
            health_icon = "✗"
            health_color = "red"
            health_text = "Critical"
            
        # Create metrics table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row("Health", f"[{health_color}]{health_icon} {health_text} ({health_score:.0f}%)[/{health_color}]")
        table.add_row("CPU", f"{metrics.cpu_usage:.1f}%")
        table.add_row("Memory", f"{metrics.memory_percent:.1f}%")
        table.add_row("Disk", f"{metrics.disk_usage_percent:.1f}%")
        table.add_row("Connections", str(metrics.connection_count))
        table.add_row("Threads", str(metrics.thread_count))
        
        return table