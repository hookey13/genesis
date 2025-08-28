"""
Risk metrics widget for Project GENESIS UI.

Displays real-time risk metrics including VaR, CVaR, Sharpe ratio,
and other institutional risk indicators.
"""

from datetime import datetime
from decimal import Decimal

from rich.align import Align
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from genesis.analytics.risk_metrics import RiskMetrics


class RiskMetricsWidget(Widget):
    """Widget displaying comprehensive risk metrics."""

    # Reactive properties for real-time updates
    var_95 = reactive(Decimal("0"))
    cvar_95 = reactive(Decimal("0"))
    sharpe_ratio = reactive(Decimal("0"))
    sortino_ratio = reactive(Decimal("0"))
    max_drawdown = reactive(Decimal("0"))
    volatility = reactive(Decimal("0"))
    last_update = reactive(datetime.now())

    # Risk limit status
    limits_breached = reactive(False)
    breached_metrics = reactive([])

    def __init__(self, metrics: RiskMetrics = None, **kwargs):
        """Initialize risk metrics widget."""
        super().__init__(**kwargs)
        if metrics:
            self.update_metrics(metrics)

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Static(id="risk-metrics-display")

    def update_metrics(self, metrics: RiskMetrics) -> None:
        """Update displayed metrics."""
        self.var_95 = metrics.value_at_risk_95
        self.cvar_95 = metrics.conditional_value_at_risk_95
        self.sharpe_ratio = metrics.sharpe_ratio
        self.sortino_ratio = metrics.sortino_ratio
        self.max_drawdown = metrics.max_drawdown
        self.volatility = metrics.volatility
        self.last_update = datetime.now()

        # Check limits (example thresholds)
        breached = []
        if self.var_95 > Decimal("10000"):
            breached.append("VaR")
        if self.max_drawdown > Decimal("0.20"):
            breached.append("Drawdown")
        if self.sharpe_ratio < Decimal("1.0"):
            breached.append("Sharpe")
        if self.volatility > Decimal("0.50"):
            breached.append("Volatility")

        self.limits_breached = len(breached) > 0
        self.breached_metrics = breached

        # Update display
        self.refresh()

    def update_var(self, var_value: Decimal, cvar_value: Decimal) -> None:
        """Update VaR and CVaR values."""
        self.var_95 = var_value
        self.cvar_95 = cvar_value
        self.last_update = datetime.now()
        self.refresh()

    def update_greeks(self, greeks: dict) -> None:
        """Update portfolio Greeks display."""
        # Store Greeks for display
        self.greeks = greeks
        self.refresh()

    def render(self) -> RenderableType:
        """Render the risk metrics display."""
        # Create main table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="white")
        table.add_column("Value", justify="right")
        table.add_column("Status", justify="center")

        # VaR and CVaR
        var_style = "red" if "VaR" in self.breached_metrics else "green"
        table.add_row(
            "VaR (95%)",
            f"${self.var_95:,.2f}",
            self._get_status_icon(self.var_95 <= Decimal("10000")),
        )
        table.add_row(
            "CVaR (95%)",
            f"${self.cvar_95:,.2f}",
            self._get_status_icon(self.cvar_95 <= Decimal("15000")),
        )

        # Ratios
        sharpe_style = "red" if "Sharpe" in self.breached_metrics else "green"
        table.add_row(
            "Sharpe Ratio",
            f"{self.sharpe_ratio:.2f}",
            self._get_status_icon(self.sharpe_ratio >= Decimal("1.0")),
        )
        table.add_row(
            "Sortino Ratio",
            f"{self.sortino_ratio:.2f}",
            self._get_status_icon(self.sortino_ratio >= Decimal("1.5")),
        )

        # Drawdown and Volatility
        dd_style = "red" if "Drawdown" in self.breached_metrics else "green"
        table.add_row(
            "Max Drawdown",
            f"{self.max_drawdown * 100:.1f}%",
            self._get_status_icon(self.max_drawdown <= Decimal("0.20")),
        )

        vol_style = "red" if "Volatility" in self.breached_metrics else "green"
        table.add_row(
            "Volatility",
            f"{self.volatility * 100:.1f}%",
            self._get_status_icon(self.volatility <= Decimal("0.50")),
        )

        # Add Greeks if available
        if hasattr(self, "greeks"):
            table.add_row("", "", "")  # Separator
            table.add_row("Delta", f"{self.greeks.get('delta', 0):.4f}", "")
            table.add_row("Vega", f"{self.greeks.get('vega', 0):.4f}", "")

        # Create panel with title showing status
        title = "🎯 Risk Metrics"
        if self.limits_breached:
            title = (
                f"⚠️ Risk Metrics - LIMITS BREACHED: {', '.join(self.breached_metrics)}"
            )
            title_style = "bold red"
        else:
            title_style = "bold green"

        # Add last update time
        footer = f"Last Update: {self.last_update.strftime('%H:%M:%S')}"

        panel = Panel(
            Align.center(table),
            title=title,
            title_align="center",
            border_style=title_style,
            subtitle=footer,
            subtitle_align="right",
        )

        return panel

    def _get_status_icon(self, is_ok: bool) -> str:
        """Get status icon based on condition."""
        return "✅" if is_ok else "❌"

    def _format_decimal(self, value: Decimal, decimals: int = 2) -> str:
        """Format decimal value for display."""
        return f"{value:,.{decimals}f}"
