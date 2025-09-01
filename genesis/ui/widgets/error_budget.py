"""
Error budget tracking widget for operational dashboard.

Displays SLO compliance, error rates, and budget consumption
with visual indicators and historical trends.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Deque

from rich.align import Align
from rich.console import RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

from genesis.monitoring.error_budget import ErrorBudgetStatus, SLO


@dataclass
class ErrorBudgetData:
    """Error budget display data."""
    
    slo_name: str
    slo_description: str
    target: float
    current_success_rate: float
    budget_consumed_pct: float  # Percentage of budget consumed
    budget_remaining_pct: float  # Percentage of budget remaining
    burn_rate: float
    time_to_exhaustion_hours: Optional[float]
    is_critical: bool
    is_exhausted: bool
    error_trend: str  # "improving", "degrading", "stable"
    trend_confidence: float
    top_error_categories: List[tuple[str, float]]  # [(category, percentage), ...]
    error_rate_per_minute: List[int]  # Last 60 minutes
    hourly_trend: List[float]  # Last 24 hours


class ErrorBudgetWidget(Widget):
    """
    Error budget tracking widget for operational dashboard.
    
    Features:
    - Real-time SLO compliance tracking
    - Error budget consumption visualization
    - Burn rate and time to exhaustion
    - Error categorization and trends
    - Historical error rate charts
    """
    
    # Reactive data
    budgets = reactive({})  # Dict[str, ErrorBudgetData]
    selected_slo = reactive(None)  # Currently selected SLO for details
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Historical data for mini charts (last 60 minutes)
        self.error_history: Dict[str, Deque[int]] = {}
        
        # Alert thresholds
        self.warning_threshold = 0.7  # 70% consumed
        self.critical_threshold = 0.9  # 90% consumed
    
    def update_budgets(self, budget_statuses: Dict[str, ErrorBudgetStatus], slos: Dict[str, SLO]):
        """
        Update error budget data from monitoring system.
        
        Args:
            budget_statuses: Current budget statuses from ErrorBudget
            slos: SLO definitions
        """
        new_budgets = {}
        
        for slo_name, status in budget_statuses.items():
            if status and slo_name in slos:
                slo = slos[slo_name]
                
                # Calculate percentages
                if status.error_budget_total > 0:
                    consumed_pct = (status.error_budget_consumed / status.error_budget_total) * 100
                    remaining_pct = (status.error_budget_remaining / status.error_budget_total) * 100
                else:
                    consumed_pct = 0.0
                    remaining_pct = 100.0
                
                # Calculate time to exhaustion in hours
                tte_hours = None
                if status.time_to_exhaustion:
                    tte_hours = status.time_to_exhaustion.total_seconds() / 3600
                
                # Get top error categories
                total_errors = sum(status.error_categories.values())
                if total_errors > 0:
                    sorted_categories = sorted(
                        status.error_categories.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]  # Top 3
                    top_categories = [
                        (cat, (count / total_errors) * 100)
                        for cat, count in sorted_categories
                    ]
                else:
                    top_categories = []
                
                # Initialize history if needed
                if slo_name not in self.error_history:
                    self.error_history[slo_name] = deque([0] * 60, maxlen=60)
                
                # Create budget data
                new_budgets[slo_name] = ErrorBudgetData(
                    slo_name=slo_name,
                    slo_description=slo.description,
                    target=status.target,
                    current_success_rate=status.current_success_rate,
                    budget_consumed_pct=consumed_pct,
                    budget_remaining_pct=remaining_pct,
                    burn_rate=status.burn_rate,
                    time_to_exhaustion_hours=tte_hours,
                    is_critical=slo.critical,
                    is_exhausted=status.is_exhausted,
                    error_trend=status.error_trend,
                    trend_confidence=status.confidence_level,
                    top_error_categories=top_categories,
                    error_rate_per_minute=list(self.error_history[slo_name]),
                    hourly_trend=[],  # Would be populated from historical data
                )
        
        self.budgets = new_budgets
    
    def render(self) -> RenderableType:
        """Render the error budget widget."""
        layout = Layout()
        
        # Create main sections
        layout.split_column(
            Layout(name="summary", size=8),
            Layout(name="details", size=12),
            Layout(name="trends", size=10),
        )
        
        # Render each section
        layout["summary"].update(self._render_summary())
        layout["details"].update(self._render_details())
        layout["trends"].update(self._render_trends())
        
        return Panel(
            layout,
            title="[bold cyan]Error Budget Tracking[/bold cyan]",
            border_style="cyan",
        )
    
    def _render_summary(self) -> RenderableType:
        """Render budget summary table."""
        table = Table(
            title="SLO Compliance Overview",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
        )
        
        table.add_column("SLO", style="white", width=20)
        table.add_column("Target", justify="center", width=8)
        table.add_column("Current", justify="center", width=8)
        table.add_column("Budget Used", justify="center", width=12)
        table.add_column("Burn Rate", justify="center", width=10)
        table.add_column("Status", justify="center", width=12)
        
        for slo_name, budget in self.budgets.items():
            # Format values
            target_str = f"{budget.target:.2%}"
            current_str = f"{budget.current_success_rate:.2%}"
            
            # Budget usage with color coding
            if budget.budget_consumed_pct >= self.critical_threshold * 100:
                budget_color = "red"
                budget_icon = "🔴"
            elif budget.budget_consumed_pct >= self.warning_threshold * 100:
                budget_color = "yellow"
                budget_icon = "🟡"
            else:
                budget_color = "green"
                budget_icon = "🟢"
            
            budget_str = f"[{budget_color}]{budget.budget_consumed_pct:.1f}%[/{budget_color}] {budget_icon}"
            
            # Burn rate with indicator
            if budget.burn_rate > 2.0:
                burn_color = "red"
                burn_icon = "🔥"
            elif budget.burn_rate > 1.5:
                burn_color = "yellow"
                burn_icon = "⚠️"
            else:
                burn_color = "green"
                burn_icon = "✓"
            
            burn_str = f"[{burn_color}]{budget.burn_rate:.2f}x[/{burn_color}] {burn_icon}"
            
            # Overall status
            if budget.is_exhausted:
                status = "[red bold]EXHAUSTED[/red bold]"
            elif budget.time_to_exhaustion_hours and budget.time_to_exhaustion_hours < 1:
                status = f"[red]< 1h left[/red]"
            elif budget.time_to_exhaustion_hours and budget.time_to_exhaustion_hours < 24:
                status = f"[yellow]{budget.time_to_exhaustion_hours:.1f}h left[/yellow]"
            else:
                status = "[green]Healthy[/green]"
            
            # Add critical indicator
            if budget.is_critical:
                slo_display = f"⚡ {budget.slo_name}"
            else:
                slo_display = budget.slo_name
            
            table.add_row(
                slo_display,
                target_str,
                current_str,
                budget_str,
                burn_str,
                status,
            )
        
        return table
    
    def _render_details(self) -> RenderableType:
        """Render detailed budget information."""
        if not self.budgets:
            return Panel("No error budget data available", border_style="dim")
        
        # Show details for most critical budget
        critical_budget = None
        max_consumption = 0
        
        for budget in self.budgets.values():
            if budget.is_critical and budget.budget_consumed_pct > max_consumption:
                critical_budget = budget
                max_consumption = budget.budget_consumed_pct
        
        if not critical_budget:
            # Fall back to highest consumption
            critical_budget = max(
                self.budgets.values(),
                key=lambda b: b.budget_consumed_pct,
                default=None
            )
        
        if not critical_budget:
            return Panel("No budget details available", border_style="dim")
        
        # Create detail table
        table = Table(
            title=f"Critical SLO: {critical_budget.slo_description}",
            show_header=False,
            show_lines=False,
            padding=(0, 1),
        )
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Add metrics
        table.add_row(
            "Success Rate",
            f"{critical_budget.current_success_rate:.3%} (target: {critical_budget.target:.3%})"
        )
        
        table.add_row(
            "Budget Status",
            f"{critical_budget.budget_remaining_pct:.1f}% remaining"
        )
        
        if critical_budget.time_to_exhaustion_hours:
            if critical_budget.time_to_exhaustion_hours < 1:
                tte_str = f"[red bold]{critical_budget.time_to_exhaustion_hours * 60:.0f} minutes[/red bold]"
            elif critical_budget.time_to_exhaustion_hours < 24:
                tte_str = f"[yellow]{critical_budget.time_to_exhaustion_hours:.1f} hours[/yellow]"
            else:
                tte_str = f"{critical_budget.time_to_exhaustion_hours / 24:.1f} days"
            table.add_row("Time to Exhaustion", tte_str)
        
        # Error trend
        trend_icon = {
            "improving": "📈 ",
            "degrading": "📉 ",
            "stable": "➡️ "
        }.get(critical_budget.error_trend, "")
        
        trend_color = {
            "improving": "green",
            "degrading": "red",
            "stable": "yellow"
        }.get(critical_budget.error_trend, "white")
        
        table.add_row(
            "Error Trend",
            f"[{trend_color}]{trend_icon}{critical_budget.error_trend.title()}[/{trend_color}] "
            f"(confidence: {critical_budget.trend_confidence:.0%})"
        )
        
        # Top error categories
        if critical_budget.top_error_categories:
            categories_str = "\n".join(
                f"  • {cat}: {pct:.1f}%"
                for cat, pct in critical_budget.top_error_categories
            )
            table.add_row("Top Error Categories", categories_str)
        
        return Panel(table, border_style="cyan" if not critical_budget.is_exhausted else "red")
    
    def _render_trends(self) -> RenderableType:
        """Render error trend charts."""
        if not self.budgets:
            return Panel("No trend data available", border_style="dim")
        
        # Create mini sparkline charts for each SLO
        charts = []
        
        for slo_name, budget in list(self.budgets.items())[:3]:  # Show top 3
            if budget.error_rate_per_minute:
                # Create sparkline
                sparkline = self._create_sparkline(
                    budget.error_rate_per_minute[-30:],  # Last 30 minutes
                    width=40,
                    height=3
                )
                
                # Format chart
                chart_text = Text()
                chart_text.append(f"{slo_name[:20]:20} ", style="cyan")
                chart_text.append(sparkline)
                
                # Add current rate
                current_rate = budget.error_rate_per_minute[-1] if budget.error_rate_per_minute else 0
                chart_text.append(f" {current_rate:3d}/min", style="white")
                
                charts.append(chart_text)
        
        if charts:
            # Combine charts
            combined = Text("\n").join(charts)
            return Panel(
                combined,
                title="Error Rate Trends (30 min)",
                border_style="cyan",
            )
        
        return Panel("No trend data available", border_style="dim")
    
    def _create_sparkline(self, data: List[int], width: int = 40, height: int = 3) -> str:
        """
        Create a sparkline chart from data.
        
        Args:
            data: List of values to chart
            width: Width of the chart in characters
            height: Height of the chart in lines
            
        Returns:
            Sparkline chart as string
        """
        if not data:
            return "─" * width
        
        # Normalize data to 0-7 range (8 block characters)
        blocks = " ▁▂▃▄▅▆▇█"
        max_val = max(data) if max(data) > 0 else 1
        
        # Sample or interpolate to fit width
        if len(data) > width:
            # Sample evenly
            step = len(data) / width
            sampled = [data[int(i * step)] for i in range(width)]
        else:
            sampled = data
        
        # Convert to blocks
        sparkline = ""
        for val in sampled:
            normalized = int((val / max_val) * 8)
            sparkline += blocks[normalized]
        
        return sparkline
    
    def _create_progress_bar(self, value: float, max_value: float, width: int = 20) -> str:
        """
        Create a text-based progress bar.
        
        Args:
            value: Current value
            max_value: Maximum value
            width: Width of the bar in characters
            
        Returns:
            Progress bar as string
        """
        if max_value <= 0:
            return "─" * width
        
        percentage = min(1.0, value / max_value)
        filled = int(width * percentage)
        
        # Choose color based on percentage
        if percentage >= 0.9:
            color = "red"
        elif percentage >= 0.7:
            color = "yellow"
        else:
            color = "green"
        
        bar = f"[{color}]{'█' * filled}[/{color}]"
        bar += "░" * (width - filled)
        
        return bar