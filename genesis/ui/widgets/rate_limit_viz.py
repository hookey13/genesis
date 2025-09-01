"""Rate limit visualization widget for Genesis trading terminal."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, List, Optional

from rich.console import RenderableType
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


@dataclass
class RateLimitEndpoint:
    """Rate limit information for an endpoint."""
    name: str
    tokens_used: int
    tokens_available: int
    tokens_capacity: int
    utilization_percent: float
    window: str  # e.g., "1m", "10s"
    requests_allowed: int = 0
    requests_rejected: int = 0
    requests_queued: int = 0
    last_update: datetime = None


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status information."""
    name: str
    state: str  # "closed", "open", "half_open"
    failure_count: int = 0
    success_count: int = 0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    last_failure: Optional[datetime] = None
    last_transition: Optional[datetime] = None


class RateLimitWidget(Static):
    """Widget for displaying rate limit usage and status."""
    
    # Rate limit data
    endpoints = reactive(list)  # List[RateLimitEndpoint]
    circuit_breakers = reactive(list)  # List[CircuitBreakerStatus]
    rate_limits = reactive(dict)  # Dict[str, dict] for compatibility
    
    # Overall metrics
    total_utilization = reactive(0.0)
    critical_overrides = reactive(0)
    coalesced_requests = reactive(0)
    
    # Sliding window metrics
    window_1m_weight = reactive(0)
    window_1m_limit = reactive(1200)
    window_10s_orders = reactive(0)
    window_10s_limit = reactive(50)
    
    DEFAULT_CSS = """
    RateLimitWidget {
        content-align: center middle;
        padding: 1;
    }
    """
    
    def __init__(self, **kwargs):
        """Initialize the rate limit widget."""
        super().__init__("Loading rate limits...", **kwargs)
        self.last_update = datetime.now(UTC)
        self.endpoints = []
        self.circuit_breakers = []
    
    def render(self) -> str:
        """Render the rate limit display."""
        lines = [
            "[bold]═══ Rate Limit Status ═══[/bold]",
            "",
            self._render_overall_status(),
            "",
            self._render_sliding_windows(),
            "",
            self._render_endpoint_table(),
            "",
            self._render_circuit_breakers(),
            "",
            f"[dim]Last update: {self.last_update.strftime('%H:%M:%S')}[/dim]",
        ]
        
        return "\n".join(lines)
    
    def _render_overall_status(self) -> str:
        """Render overall rate limit status."""
        # Determine health based on utilization
        if self.total_utilization >= 90:
            status_color = "red"
            status_text = "CRITICAL"
        elif self.total_utilization >= 70:
            status_color = "orange"
            status_text = "WARNING"
        elif self.total_utilization >= 50:
            status_color = "yellow"
            status_text = "MODERATE"
        else:
            status_color = "green"
            status_text = "HEALTHY"
        
        lines = [
            f"[{status_color}]● {status_text}[/{status_color}] - Overall: {self.total_utilization:.1f}%",
        ]
        
        if self.critical_overrides > 0:
            lines.append(f"[yellow]⚠ Critical overrides: {self.critical_overrides}[/yellow]")
        
        if self.coalesced_requests > 0:
            lines.append(f"[dim]Coalesced requests: {self.coalesced_requests}[/dim]")
        
        return "\n".join(lines)
    
    def _render_sliding_windows(self) -> str:
        """Render sliding window rate limits."""
        lines = ["[bold]Sliding Windows:[/bold]"]
        
        # 1-minute weight window
        weight_percent = (self.window_1m_weight / self.window_1m_limit * 100) if self.window_1m_limit > 0 else 0
        weight_color = self._get_usage_color(weight_percent)
        weight_bar = self._create_progress_bar(weight_percent, 100, weight_color)
        lines.append(f"1m Weight:  {weight_bar} {self.window_1m_weight}/{self.window_1m_limit}")
        
        # 10-second order window
        order_percent = (self.window_10s_orders / self.window_10s_limit * 100) if self.window_10s_limit > 0 else 0
        order_color = self._get_usage_color(order_percent)
        order_bar = self._create_progress_bar(order_percent, 100, order_color)
        lines.append(f"10s Orders: {order_bar} {self.window_10s_orders}/{self.window_10s_limit}")
        
        return "\n".join(lines)
    
    def _render_endpoint_table(self) -> str:
        """Render endpoint rate limit table."""
        if not self.endpoints:
            return "[dim]No endpoint data available[/dim]"
        
        lines = ["[bold]Endpoint Limits:[/bold]"]
        
        # Sort endpoints by utilization (highest first)
        sorted_endpoints = sorted(self.endpoints, key=lambda e: e.utilization_percent, reverse=True)
        
        for endpoint in sorted_endpoints[:5]:  # Show top 5
            color = self._get_usage_color(endpoint.utilization_percent)
            bar = self._create_progress_bar(endpoint.utilization_percent, 100, color, width=15)
            
            status = f"{endpoint.tokens_used}/{endpoint.tokens_capacity}"
            if endpoint.requests_rejected > 0:
                status += f" [red]({endpoint.requests_rejected} rejected)[/red]"
            
            lines.append(
                f"  {endpoint.name[:20]:20} {bar} {endpoint.utilization_percent:5.1f}% {status}"
            )
        
        return "\n".join(lines)
    
    def _render_circuit_breakers(self) -> str:
        """Render circuit breaker status."""
        if not self.circuit_breakers:
            return "[dim]No circuit breakers configured[/dim]"
        
        lines = ["[bold]Circuit Breakers:[/bold]"]
        
        for cb in self.circuit_breakers:
            # Status icon and color based on state
            if cb.state == "open":
                icon = "⛔"
                color = "red"
                status = "OPEN"
            elif cb.state == "half_open":
                icon = "⚠"
                color = "yellow"
                status = "HALF"
            else:  # closed
                icon = "✓"
                color = "green"
                status = "OK"
            
            failure_rate = (cb.failure_count / (cb.failure_count + cb.success_count) * 100) \
                if (cb.failure_count + cb.success_count) > 0 else 0
            
            lines.append(
                f"  [{color}]{icon}[/{color}] {cb.name[:15]:15} {status:5} "
                f"Failures: {cb.failure_count:3} ({failure_rate:.1f}%)"
            )
        
        return "\n".join(lines)
    
    def _create_progress_bar(self, value: float, max_value: float, color: str, width: int = 20) -> str:
        """Create a simple text-based progress bar."""
        filled = int((value / max_value) * width) if max_value > 0 else 0
        filled = min(width, max(0, filled))
        empty = width - filled
        
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
    
    def add_endpoint(self, endpoint: RateLimitEndpoint) -> None:
        """Add or update an endpoint."""
        # Find existing endpoint
        for i, e in enumerate(self.endpoints):
            if e.name == endpoint.name:
                self.endpoints[i] = endpoint
                self._recalculate_total_utilization()
                return
        
        # Add new endpoint
        self.endpoints.append(endpoint)
        self._recalculate_total_utilization()
    
    def add_circuit_breaker(self, cb: CircuitBreakerStatus) -> None:
        """Add or update a circuit breaker."""
        # Find existing circuit breaker
        for i, existing in enumerate(self.circuit_breakers):
            if existing.name == cb.name:
                self.circuit_breakers[i] = cb
                return
        
        # Add new circuit breaker
        self.circuit_breakers.append(cb)
    
    def _recalculate_total_utilization(self) -> None:
        """Recalculate total utilization across all endpoints."""
        if not self.endpoints:
            self.total_utilization = 0.0
            return
        
        # Calculate weighted average
        total_weight = sum(e.tokens_capacity for e in self.endpoints)
        if total_weight > 0:
            weighted_sum = sum(e.utilization_percent * e.tokens_capacity for e in self.endpoints)
            self.total_utilization = weighted_sum / total_weight
        else:
            self.total_utilization = 0.0
    
    def update_sliding_windows(
        self,
        weight_1m: Optional[int] = None,
        limit_1m: Optional[int] = None,
        orders_10s: Optional[int] = None,
        limit_10s: Optional[int] = None,
    ) -> None:
        """Update sliding window metrics."""
        if weight_1m is not None:
            self.window_1m_weight = weight_1m
        if limit_1m is not None:
            self.window_1m_limit = limit_1m
        if orders_10s is not None:
            self.window_10s_orders = orders_10s
        if limit_10s is not None:
            self.window_10s_limit = limit_10s
        
        self.last_update = datetime.now(UTC)
    
    def update_counters(
        self,
        critical_overrides: Optional[int] = None,
        coalesced: Optional[int] = None,
    ) -> None:
        """Update counter metrics."""
        if critical_overrides is not None:
            self.critical_overrides = critical_overrides
        if coalesced is not None:
            self.coalesced_requests = coalesced
    
    def set_mock_data(self) -> None:
        """Set mock data for testing."""
        # Add mock endpoints
        self.endpoints = [
            RateLimitEndpoint(
                name="GET /api/v3/order",
                tokens_used=450,
                tokens_available=550,
                tokens_capacity=1000,
                utilization_percent=45.0,
                window="1m",
                requests_allowed=450,
                requests_rejected=2,
            ),
            RateLimitEndpoint(
                name="POST /api/v3/order",
                tokens_used=750,
                tokens_available=250,
                tokens_capacity=1000,
                utilization_percent=75.0,
                window="1m",
                requests_allowed=750,
                requests_rejected=10,
                requests_queued=5,
            ),
            RateLimitEndpoint(
                name="GET /api/v3/ticker",
                tokens_used=200,
                tokens_available=800,
                tokens_capacity=1000,
                utilization_percent=20.0,
                window="1m",
                requests_allowed=200,
            ),
        ]
        
        # Add mock circuit breakers
        self.circuit_breakers = [
            CircuitBreakerStatus(
                name="order_api",
                state="closed",
                failure_count=2,
                success_count=98,
            ),
            CircuitBreakerStatus(
                name="market_data",
                state="closed",
                failure_count=0,
                success_count=1000,
            ),
            CircuitBreakerStatus(
                name="websocket",
                state="half_open",
                failure_count=5,
                success_count=45,
            ),
        ]
        
        # Set sliding windows
        self.window_1m_weight = 600
        self.window_1m_limit = 1200
        self.window_10s_orders = 15
        self.window_10s_limit = 50
        
        # Set counters
        self.critical_overrides = 0
        self.coalesced_requests = 25
        
        self._recalculate_total_utilization()
    
    def watch_total_utilization(self, value: float) -> None:
        """React to total utilization changes."""
        self.update(self.render())
    
    def watch_window_1m_weight(self, value: int) -> None:
        """React to weight changes."""
        self.update(self.render())


class TokenBucketVisualization(Panel):
    """Visual representation of token bucket state."""
    
    def __init__(self, bucket_name: str = "default", capacity: int = 100, **kwargs):
        """Initialize token bucket visualization.
        
        Args:
            bucket_name: Name of the token bucket
            capacity: Maximum capacity of the bucket
        """
        self.bucket_name = bucket_name
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = 10  # tokens per second
        super().__init__(self._render_bucket(), title=f"Token Bucket: {bucket_name}", **kwargs)
    
    def _render_bucket(self) -> RenderableType:
        """Render the token bucket visualization."""
        # Calculate fill percentage
        fill_percent = (self.tokens / self.capacity * 100) if self.capacity > 0 else 0
        
        # Create visual representation
        bucket_height = 10
        filled_rows = int((fill_percent / 100) * bucket_height)
        
        lines = []
        for i in range(bucket_height, 0, -1):
            if i <= filled_rows:
                # Filled portion
                if fill_percent >= 80:
                    lines.append("[green]████████████[/green]")
                elif fill_percent >= 50:
                    lines.append("[yellow]████████████[/yellow]")
                else:
                    lines.append("[red]████████████[/red]")
            else:
                # Empty portion
                lines.append("[dim]░░░░░░░░░░░░[/dim]")
        
        # Add bottom
        lines.append("╚════════════╝")
        
        # Add stats
        lines.extend([
            "",
            f"Tokens: {self.tokens}/{self.capacity}",
            f"Fill: {fill_percent:.1f}%",
            f"Refill: {self.refill_rate}/sec",
        ])
        
        return Text.from_markup("\n".join(lines))
    
    def update_tokens(self, tokens: int) -> None:
        """Update the number of tokens in the bucket."""
        self.tokens = min(self.capacity, max(0, tokens))
        self.renderable = self._render_bucket()