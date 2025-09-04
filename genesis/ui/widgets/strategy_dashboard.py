"""Strategy performance dashboard widget for terminal UI."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, DataTable, Label, Sparkline, Static

from genesis.monitoring.strategy_monitor import StrategyMetrics
from genesis.monitoring.risk_metrics import RiskMetrics


class StrategyPerformanceWidget(Widget):
    """Widget displaying performance metrics for a single strategy."""
    
    def __init__(self, strategy_id: str, metrics: Optional[StrategyMetrics] = None):
        """Initialize strategy performance widget.
        
        Args:
            strategy_id: Strategy identifier
            metrics: Initial metrics to display
        """
        super().__init__()
        self.strategy_id = strategy_id
        self.metrics = metrics
        
    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Container(
            Label(f"Strategy: {self.strategy_id}", classes="strategy-title"),
            Horizontal(
                Static(self._render_pnl_panel(), classes="pnl-panel"),
                Static(self._render_stats_panel(), classes="stats-panel"),
                Static(self._render_risk_panel(), classes="risk-panel"),
                classes="metrics-row"
            ),
            classes="strategy-container"
        )
        
    def _render_pnl_panel(self) -> Panel:
        """Render P&L metrics panel."""
        if not self.metrics:
            return Panel("No data", title="P&L")
            
        content = Table.grid(padding=1)
        content.add_column(justify="left")
        content.add_column(justify="right")
        
        # Color code P&L values
        total_color = "green" if self.metrics.total_pnl >= 0 else "red"
        realized_color = "green" if self.metrics.realized_pnl >= 0 else "red"
        unrealized_color = "green" if self.metrics.unrealized_pnl >= 0 else "red"
        
        content.add_row("Total P&L:", f"[{total_color}]${self.metrics.total_pnl:,.2f}[/]")
        content.add_row("Realized:", f"[{realized_color}]${self.metrics.realized_pnl:,.2f}[/]")
        content.add_row("Unrealized:", f"[{unrealized_color}]${self.metrics.unrealized_pnl:,.2f}[/]")
        content.add_row("Expectancy:", f"${self.metrics.expectancy:,.2f}")
        
        return Panel(content, title="💰 P&L Metrics", border_style="cyan")
        
    def _render_stats_panel(self) -> Panel:
        """Render trading statistics panel."""
        if not self.metrics:
            return Panel("No data", title="Statistics")
            
        content = Table.grid(padding=1)
        content.add_column(justify="left")
        content.add_column(justify="right")
        
        win_rate_color = "green" if self.metrics.win_rate >= 50 else "yellow" if self.metrics.win_rate >= 30 else "red"
        profit_factor_color = "green" if self.metrics.profit_factor >= 1.5 else "yellow" if self.metrics.profit_factor >= 1 else "red"
        
        content.add_row("Total Trades:", str(self.metrics.total_trades))
        content.add_row("Win Rate:", f"[{win_rate_color}]{self.metrics.win_rate:.1f}%[/]")
        content.add_row("Profit Factor:", f"[{profit_factor_color}]{self.metrics.profit_factor:.2f}[/]")
        content.add_row("Avg Win:", f"${self.metrics.average_win:,.2f}")
        content.add_row("Avg Loss:", f"${self.metrics.average_loss:,.2f}")
        content.add_row("Slippage:", f"${self.metrics.slippage_total:,.2f}")
        
        return Panel(content, title="📊 Statistics", border_style="green")
        
    def _render_risk_panel(self) -> Panel:
        """Render risk metrics panel."""
        if not self.metrics:
            return Panel("No data", title="Risk")
            
        content = Table.grid(padding=1)
        content.add_column(justify="left")
        content.add_column(justify="right")
        
        dd_color = "green" if self.metrics.current_drawdown == 0 else "yellow" if self.metrics.current_drawdown < self.metrics.max_drawdown * Decimal("0.5") else "red"
        max_dd_color = "green" if self.metrics.max_drawdown < Decimal("10") else "yellow" if self.metrics.max_drawdown < Decimal("20") else "red"
        
        content.add_row("Current DD:", f"[{dd_color}]{self.metrics.current_drawdown:.2f}%[/]")
        content.add_row("Max DD:", f"[{max_dd_color}]{self.metrics.max_drawdown:.2f}%[/]")
        
        if self.metrics.drawdown_start:
            duration = datetime.utcnow() - self.metrics.drawdown_start
            content.add_row("DD Duration:", f"{duration.days}d {duration.seconds//3600}h")
        else:
            content.add_row("DD Duration:", "N/A")
            
        content.add_row("Peak P&L:", f"${self.metrics.peak_pnl:,.2f}")
        
        return Panel(content, title="⚠️ Risk Metrics", border_style="yellow")
        
    def update_metrics(self, metrics: StrategyMetrics) -> None:
        """Update displayed metrics.
        
        Args:
            metrics: Updated strategy metrics
        """
        self.metrics = metrics
        self.refresh()


class RiskMetricsWidget(Widget):
    """Widget displaying aggregate risk metrics."""
    
    def __init__(self, risk_metrics: Optional[Dict[str, RiskMetrics]] = None):
        """Initialize risk metrics widget.
        
        Args:
            risk_metrics: Dictionary of strategy risk metrics
        """
        super().__init__()
        self.risk_metrics = risk_metrics or {}
        
    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Container(
            Label("Risk Dashboard", classes="risk-title"),
            Static(self._render_risk_table(), classes="risk-table"),
            classes="risk-container"
        )
        
    def _render_risk_table(self) -> Panel:
        """Render risk metrics table."""
        table = Table(title="Risk Metrics by Strategy", show_header=True, header_style="bold cyan")
        
        table.add_column("Strategy", style="cyan", no_wrap=True)
        table.add_column("VaR 95%", justify="right")
        table.add_column("CVaR 95%", justify="right")
        table.add_column("Beta", justify="right")
        table.add_column("Sharpe", justify="right")
        table.add_column("Sortino", justify="right")
        table.add_column("Volatility", justify="right")
        
        for strategy_id, metrics in self.risk_metrics.items():
            var_style = "red" if metrics.var_95 > Decimal("5") else "yellow" if metrics.var_95 > Decimal("3") else "green"
            sharpe_style = "green" if metrics.sharpe_ratio > Decimal("2") else "yellow" if metrics.sharpe_ratio > Decimal("1") else "red"
            
            table.add_row(
                strategy_id[:15],
                f"[{var_style}]{metrics.var_95:.2f}%[/]",
                f"{metrics.cvar_95:.2f}%",
                f"{metrics.beta:.2f}",
                f"[{sharpe_style}]{metrics.sharpe_ratio:.2f}[/]",
                f"{metrics.sortino_ratio:.2f}",
                f"{metrics.volatility:.1f}%"
            )
            
        return Panel(table, border_style="red")
        
    def update_metrics(self, risk_metrics: Dict[str, RiskMetrics]) -> None:
        """Update displayed risk metrics.
        
        Args:
            risk_metrics: Updated risk metrics by strategy
        """
        self.risk_metrics = risk_metrics
        self.refresh()


class CorrelationMatrixWidget(Widget):
    """Widget displaying strategy correlation matrix."""
    
    def __init__(self, correlation_data: Optional[Dict[str, Dict[str, Decimal]]] = None):
        """Initialize correlation matrix widget.
        
        Args:
            correlation_data: Correlation matrix data
        """
        super().__init__()
        self.correlation_data = correlation_data or {}
        
    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Container(
            Label("Strategy Correlations", classes="correlation-title"),
            Static(self._render_correlation_matrix(), classes="correlation-matrix"),
            classes="correlation-container"
        )
        
    def _render_correlation_matrix(self) -> Panel:
        """Render correlation matrix."""
        if not self.correlation_data:
            return Panel("No correlation data available", title="Correlation Matrix")
            
        table = Table(show_header=True, header_style="bold magenta")
        
        strategies = list(self.correlation_data.keys())
        
        # Add header columns
        table.add_column("", style="magenta")
        for strategy in strategies[:5]:  # Limit to 5 for display
            table.add_column(strategy[:8], justify="center")
            
        # Add data rows
        for i, strategy_i in enumerate(strategies[:5]):
            row = [strategy_i[:8]]
            for j, strategy_j in enumerate(strategies[:5]):
                if strategy_j in self.correlation_data.get(strategy_i, {}):
                    corr = self.correlation_data[strategy_i][strategy_j]
                    
                    # Color code correlation
                    if corr >= Decimal("0.7"):
                        style = "red"
                    elif corr >= Decimal("0.3"):
                        style = "yellow"
                    elif corr >= Decimal("-0.3"):
                        style = "white"
                    elif corr >= Decimal("-0.7"):
                        style = "cyan"
                    else:
                        style = "blue"
                        
                    row.append(f"[{style}]{corr:.2f}[/]")
                else:
                    row.append("--")
                    
            table.add_row(*row)
            
        return Panel(table, title="📈 Correlation Matrix", border_style="magenta")
        
    def update_correlations(self, correlation_data: Dict[str, Dict[str, Decimal]]) -> None:
        """Update correlation matrix.
        
        Args:
            correlation_data: Updated correlation data
        """
        self.correlation_data = correlation_data
        self.refresh()


class PerformanceAttributionWidget(Widget):
    """Widget displaying performance attribution analysis."""
    
    def __init__(self, attribution_data: Optional[Dict[str, any]] = None):
        """Initialize performance attribution widget.
        
        Args:
            attribution_data: Attribution analysis data
        """
        super().__init__()
        self.attribution_data = attribution_data or {}
        
    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Container(
            Label("Performance Attribution", classes="attribution-title"),
            Horizontal(
                Static(self._render_by_symbol(), classes="attr-symbol"),
                Static(self._render_by_time(), classes="attr-time"),
                classes="attribution-row"
            ),
            classes="attribution-container"
        )
        
    def _render_by_symbol(self) -> Panel:
        """Render attribution by symbol."""
        if not self.attribution_data or "by_symbol" not in self.attribution_data:
            return Panel("No symbol data", title="By Symbol")
            
        table = Table.grid(padding=1)
        table.add_column(justify="left")
        table.add_column(justify="right")
        
        symbol_data = self.attribution_data.get("by_symbol", {})
        
        # Sort by contribution
        sorted_symbols = sorted(
            symbol_data.items(),
            key=lambda x: x[1].pnl_contribution if hasattr(x[1], 'pnl_contribution') else 0,
            reverse=True
        )
        
        for symbol, component in sorted_symbols[:5]:  # Top 5 symbols
            if hasattr(component, 'pnl_contribution'):
                color = "green" if component.pnl_contribution >= 0 else "red"
                table.add_row(
                    symbol,
                    f"[{color}]${component.pnl_contribution:,.2f} ({component.percentage_contribution:.1f}%)[/]"
                )
                
        return Panel(table, title="🪙 By Symbol", border_style="blue")
        
    def _render_by_time(self) -> Panel:
        """Render attribution by time period."""
        if not self.attribution_data or "by_hour" not in self.attribution_data:
            return Panel("No time data", title="By Time")
            
        table = Table.grid(padding=1)
        table.add_column(justify="left")
        table.add_column(justify="right")
        
        hour_data = self.attribution_data.get("by_hour", {})
        
        # Find best and worst hours
        if hour_data:
            sorted_hours = sorted(
                hour_data.items(),
                key=lambda x: x[1].pnl_contribution if hasattr(x[1], 'pnl_contribution') else 0,
                reverse=True
            )
            
            if sorted_hours:
                best_hour = sorted_hours[0]
                worst_hour = sorted_hours[-1]
                
                if hasattr(best_hour[1], 'pnl_contribution'):
                    table.add_row(
                        f"Best Hour: {best_hour[0]:02d}:00",
                        f"[green]${best_hour[1].pnl_contribution:,.2f}[/]"
                    )
                    
                if hasattr(worst_hour[1], 'pnl_contribution'):
                    table.add_row(
                        f"Worst Hour: {worst_hour[0]:02d}:00",
                        f"[red]${worst_hour[1].pnl_contribution:,.2f}[/]"
                    )
                    
        return Panel(table, title="🕐 By Time", border_style="cyan")
        
    def update_attribution(self, attribution_data: Dict[str, any]) -> None:
        """Update attribution data.
        
        Args:
            attribution_data: Updated attribution analysis
        """
        self.attribution_data = attribution_data
        self.refresh()


class StrategyDashboard(Widget):
    """Main strategy performance dashboard."""
    
    DEFAULT_CSS = """
    .strategy-title {
        text-style: bold;
        color: cyan;
        margin: 1;
    }
    
    .risk-title {
        text-style: bold;
        color: red;
        margin: 1;
    }
    
    .correlation-title {
        text-style: bold;
        color: magenta;
        margin: 1;
    }
    
    .attribution-title {
        text-style: bold;
        color: blue;
        margin: 1;
    }
    
    .strategy-container {
        border: solid cyan;
        margin: 1;
        padding: 1;
    }
    
    .risk-container {
        border: solid red;
        margin: 1;
        padding: 1;
    }
    
    .correlation-container {
        border: solid magenta;
        margin: 1;
        padding: 1;
    }
    
    .attribution-container {
        border: solid blue;
        margin: 1;
        padding: 1;
    }
    """
    
    def __init__(self):
        """Initialize the strategy dashboard."""
        super().__init__()
        self.strategy_widgets: Dict[str, StrategyPerformanceWidget] = {}
        self.risk_widget: Optional[RiskMetricsWidget] = None
        self.correlation_widget: Optional[CorrelationMatrixWidget] = None
        self.attribution_widget: Optional[PerformanceAttributionWidget] = None
        
    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        with ScrollableContainer():
            with Vertical():
                yield Label("📊 Strategy Performance Dashboard", classes="dashboard-title")
                
                # Strategy performance widgets will be added dynamically
                yield Container(id="strategy-widgets")
                
                # Risk metrics widget
                self.risk_widget = RiskMetricsWidget()
                yield self.risk_widget
                
                # Correlation matrix widget
                self.correlation_widget = CorrelationMatrixWidget()
                yield self.correlation_widget
                
                # Performance attribution widget
                self.attribution_widget = PerformanceAttributionWidget()
                yield self.attribution_widget
                
    def add_strategy(self, strategy_id: str, metrics: StrategyMetrics) -> None:
        """Add a strategy to the dashboard.
        
        Args:
            strategy_id: Strategy identifier
            metrics: Strategy metrics
        """
        if strategy_id not in self.strategy_widgets:
            widget = StrategyPerformanceWidget(strategy_id, metrics)
            self.strategy_widgets[strategy_id] = widget
            container = self.query_one("#strategy-widgets")
            container.mount(widget)
        else:
            self.strategy_widgets[strategy_id].update_metrics(metrics)
            
    def update_risk_metrics(self, risk_metrics: Dict[str, RiskMetrics]) -> None:
        """Update risk metrics display.
        
        Args:
            risk_metrics: Risk metrics by strategy
        """
        if self.risk_widget:
            self.risk_widget.update_metrics(risk_metrics)
            
    def update_correlations(self, correlation_data: Dict[str, Dict[str, Decimal]]) -> None:
        """Update correlation matrix.
        
        Args:
            correlation_data: Correlation matrix data
        """
        if self.correlation_widget:
            self.correlation_widget.update_correlations(correlation_data)
            
    def update_attribution(self, attribution_data: Dict[str, any]) -> None:
        """Update performance attribution.
        
        Args:
            attribution_data: Attribution analysis data
        """
        if self.attribution_widget:
            self.attribution_widget.update_attribution(attribution_data)