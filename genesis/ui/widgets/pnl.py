"""P&L display widget for Genesis trading terminal with historical charts."""

from collections import deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class PnLWidget(Static):
    """Widget for displaying P&L information with color coding and historical charts."""

    # Reactive values that trigger updates
    current_pnl = reactive(Decimal("0.00"))
    daily_pnl = reactive(Decimal("0.00"))
    daily_pnl_pct = reactive(Decimal("0.00"))

    # Additional metrics for paper trading
    realized_pnl = reactive(Decimal("0.00"))
    unrealized_pnl = reactive(Decimal("0.00"))
    win_rate = reactive(Decimal("0.00"))
    total_trades = reactive(0)
    paper_trading_mode = reactive(False)
    
    # Historical data for charts
    pnl_history = reactive(list)
    max_drawdown = reactive(Decimal("0.00"))
    sharpe_ratio = reactive(Decimal("0.00"))

    DEFAULT_CSS = """
    PnLWidget {
        content-align: center middle;
        padding: 1;
    }
    """

    def __init__(self, **kwargs):
        """Initialize the P&L widget."""
        super().__init__("Loading P&L...", **kwargs)
        self.account_balance = Decimal("0.00")
        # Initialize historical data storage (last 24 hours of 5-min intervals)
        self.pnl_data_points = deque(maxlen=288)  # 24h * 12 intervals per hour
        self.hourly_pnl = deque(maxlen=24)  # Last 24 hours
        self.daily_pnl_history = deque(maxlen=30)  # Last 30 days

    def render(self) -> str:
        """Render the P&L display."""
        # Color coding: green for profit, gray for loss (no red)
        current_color = "green" if self.current_pnl >= 0 else "grey50"
        daily_color = "green" if self.daily_pnl >= 0 else "grey50"
        realized_color = "green" if self.realized_pnl >= 0 else "grey50"
        unrealized_color = "green" if self.unrealized_pnl >= 0 else "grey50"

        # Format values to 2 decimal places
        current_str = f"${self.current_pnl:,.2f}"
        daily_str = f"${self.daily_pnl:,.2f}"
        daily_pct_str = f"{self.daily_pnl_pct:+.2f}%"
        realized_str = f"${self.realized_pnl:,.2f}"
        unrealized_str = f"${self.unrealized_pnl:,.2f}"

        # Build display header
        if self.paper_trading_mode:
            header = "[bold yellow]═══ Paper Trading P&L ═══[/bold yellow]"
        else:
            header = "[bold]═══ P&L Dashboard ═══[/bold]"

        # Build display
        lines = [
            header,
            "",
            f"[bold]Total P&L:[/bold] [{current_color}]{current_str}[/{current_color}]",
            "",
            f"[bold]Realized:[/bold] [{realized_color}]{realized_str}[/{realized_color}]",
            f"[bold]Unrealized:[/bold] [{unrealized_color}]{unrealized_str}[/{unrealized_color}]",
            "",
            f"[bold]Daily P&L:[/bold] [{daily_color}]{daily_str} ({daily_pct_str})[/{daily_color}]",
            "",
        ]

        # Add trading metrics if available
        if self.total_trades > 0:
            win_rate_color = "green" if self.win_rate >= 50 else "grey50"
            lines.extend([
                f"[bold]Win Rate:[/bold] [{win_rate_color}]{self.win_rate:.2f}%[/{win_rate_color}]",
                f"[dim]Total Trades: {self.total_trades}[/dim]",
                "",
            ])

        lines.append(f"[dim]Account Balance: ${self.account_balance:,.2f}[/dim]")
        
        # Add historical chart if data available
        if self.pnl_data_points:
            lines.extend(["", self._render_pnl_chart()])
            
        # Add max drawdown and sharpe ratio
        if self.max_drawdown != 0 or self.sharpe_ratio != 0:
            lines.extend([
                "",
                f"[bold]Risk Metrics:[/bold]",
                f"Max Drawdown: {self.max_drawdown:.2f}%",
                f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            ])

        return "\n".join(lines)

    async def update_data(self) -> None:
        """Update P&L data from connected components."""
        # TODO: Connect to AccountManager and RiskEngine
        # For now, using mock data to demonstrate
        pass

    def watch_current_pnl(self, value: Decimal) -> None:
        """React to current P&L changes."""
        self.update(self.render())

    def watch_daily_pnl(self, value: Decimal) -> None:
        """React to daily P&L changes."""
        self.update(self.render())

    def watch_daily_pnl_pct(self, value: Decimal) -> None:
        """React to daily P&L percentage changes."""
        self.update(self.render())

    def _render_pnl_chart(self) -> str:
        """Render a simple ASCII chart of P&L history."""
        if not self.pnl_data_points:
            return "[dim]No historical data available[/dim]"
            
        # Create a simple sparkline chart
        values = list(self.pnl_data_points)
        if not values:
            return ""
            
        # Find min and max for scaling
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            # All values are the same
            return f"[dim]P&L Chart: ─{'─' * 20}─ ${min_val:,.2f}[/dim]"
            
        # Create sparkline using block characters
        blocks = "▁▂▃▄▅▆▇█"
        chart_width = min(40, len(values))
        
        # Sample values if too many
        if len(values) > chart_width:
            step = len(values) // chart_width
            sampled_values = values[::step][:chart_width]
        else:
            sampled_values = values
            
        # Scale values to 0-7 range for block characters
        scaled = []
        range_val = max_val - min_val
        for val in sampled_values:
            if range_val > 0:
                normalized = (val - min_val) / range_val
                index = min(7, int(normalized * 8))
                scaled.append(blocks[index])
            else:
                scaled.append(blocks[0])
                
        chart = "".join(scaled)
        
        # Color based on trend
        if values[-1] > values[0]:
            color = "green"
        else:
            color = "grey50"
            
        return f"[{color}]P&L Trend: {chart}[/{color}] [dim](24h)[/dim]"
    
    def add_pnl_data_point(self, value: Decimal, timestamp=None) -> None:  # Optional[datetime]
        """Add a P&L data point to history.
        
        Args:
            value: P&L value to add
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)
            
        self.pnl_data_points.append(value)
        
        # Update hourly aggregation
        current_hour = timestamp.replace(minute=0, second=0, microsecond=0)
        if not self.hourly_pnl or self.hourly_pnl[-1][0] != current_hour:
            self.hourly_pnl.append((current_hour, value))
        else:
            # Update the last hour's value
            self.hourly_pnl[-1] = (current_hour, value)
    
    def calculate_risk_metrics(self) -> None:
        """Calculate max drawdown and Sharpe ratio from historical data."""
        if not self.pnl_data_points or len(self.pnl_data_points) < 2:
            return
            
        values = list(self.pnl_data_points)
        
        # Calculate max drawdown
        peak = values[0]
        max_dd = Decimal("0")
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100 if peak != 0 else Decimal("0")
            max_dd = max(max_dd, drawdown)
        self.max_drawdown = max_dd
        
        # Calculate simple Sharpe ratio (assuming risk-free rate = 0)
        if len(values) > 1:
            returns = [(values[i] - values[i-1]) / values[i-1] if values[i-1] != 0 else Decimal("0") 
                      for i in range(1, len(values))]
            if returns:
                avg_return = sum(returns) / len(returns)
                if len(returns) > 1:
                    variance = sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1)
                    std_dev = variance ** Decimal("0.5")
                    self.sharpe_ratio = (avg_return / std_dev * Decimal("15.87")) if std_dev != 0 else Decimal("0")  # Annualized (√252)

    def set_mock_data(self, current: Decimal, daily: Decimal, balance: Decimal) -> None:
        """Set mock data for testing."""
        self.current_pnl = current
        self.daily_pnl = daily
        self.account_balance = balance

        # Calculate percentage
        if balance > 0:
            self.daily_pnl_pct = (daily / balance) * 100
        else:
            self.daily_pnl_pct = Decimal("0.00")
            
        # Add to history
        self.add_pnl_data_point(current)
        self.calculate_risk_metrics()
