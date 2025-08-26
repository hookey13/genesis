"""Multi-pair trading dashboard widgets for Textual UI."""

from datetime import datetime
from decimal import Decimal

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable

from genesis.analytics.pair_performance import PairMetrics
from genesis.core.models import Position, Signal
from genesis.engine.executor.multi_pair import PortfolioRisk


class PositionGrid(Widget):
    """Display grid of all active positions across pairs."""

    positions = reactive([], recompose=True)

    def __init__(self, positions: list[Position] = None, **kwargs):
        super().__init__(**kwargs)
        self.positions = positions or []

    def compose(self) -> ComposeResult:
        """Compose the position grid."""
        with Container(id="position-grid-container"):
            yield DataTable(id="position-grid")

    def on_mount(self) -> None:
        """Initialize the data table when mounted."""
        table = self.query_one("#position-grid", DataTable)
        table.add_columns(
            "Symbol",
            "Side",
            "Size",
            "Entry",
            "Current",
            "P&L ($)",
            "P&L (%)",
            "Value",
            "Time"
        )
        self.update_positions(self.positions)

    def update_positions(self, positions: list[Position]) -> None:
        """Update the positions display."""
        self.positions = positions
        table = self.query_one("#position-grid", DataTable)
        table.clear()

        for position in positions:
            # Format P&L with color
            pnl_dollars = position.pnl_dollars
            pnl_text = f"${pnl_dollars:,.2f}"
            pnl_percent = (pnl_dollars / position.dollar_value * 100) if position.dollar_value else 0
            pnl_percent_text = f"{pnl_percent:+.2f}%"

            # Calculate hold time
            hold_time = datetime.utcnow() - position.opened_at
            hold_time_str = f"{hold_time.total_seconds() / 60:.0f}m"

            table.add_row(
                position.symbol,
                position.side.value,
                f"{position.quantity:.4f}",
                f"${position.entry_price:,.2f}",
                f"${position.current_price:,.2f}",
                pnl_text,
                pnl_percent_text,
                f"${position.dollar_value:,.2f}",
                hold_time_str
            )


class CorrelationHeatmap(Widget):
    """Display correlation heatmap between trading pairs."""

    correlations = reactive({}, recompose=True)

    def __init__(self, correlations: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.correlations = correlations or {}

    def render(self) -> RenderableType:
        """Render the correlation heatmap."""
        if not self.correlations:
            return Panel("No correlation data available", title="Correlation Matrix")

        # Extract unique symbols
        symbols = set()
        for (sym1, sym2) in self.correlations.keys():
            symbols.add(sym1)
            symbols.add(sym2)
        symbols = sorted(list(symbols))

        if not symbols:
            return Panel("No positions to correlate", title="Correlation Matrix")

        # Create table
        table = Table(title="Correlation Heatmap", show_header=True, header_style="bold")
        table.add_column("Pair", style="cyan")

        for symbol in symbols:
            table.add_column(symbol[:6], justify="center", width=8)

        # Add rows
        for sym1 in symbols:
            row_data = [sym1[:6]]
            for sym2 in symbols:
                if sym1 == sym2:
                    row_data.append("1.00")
                else:
                    corr = self.correlations.get((sym1, sym2)) or self.correlations.get((sym2, sym1), 0)

                    # Color code based on correlation strength
                    if isinstance(corr, Decimal):
                        corr_float = float(corr)
                    else:
                        corr_float = corr

                    if abs(corr_float) > 0.8:
                        style = "red bold"
                    elif abs(corr_float) > 0.6:
                        style = "yellow"
                    else:
                        style = "green"

                    corr_text = Text(f"{corr_float:.2f}", style=style)
                    row_data.append(corr_text)

            table.add_row(*row_data)

        return Panel(table, title="Correlation Matrix", border_style="blue")

    def update_correlations(self, correlations: dict) -> None:
        """Update correlation data."""
        self.correlations = correlations


class PerformanceAttribution(Widget):
    """Display performance attribution across pairs."""

    metrics = reactive({}, recompose=True)

    def __init__(self, metrics: dict[str, PairMetrics] = None, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics or {}

    def render(self) -> RenderableType:
        """Render performance attribution."""
        if not self.metrics:
            return Panel("No performance data available", title="Performance Attribution")

        table = Table(title="Pair Performance", show_header=True, header_style="bold")
        table.add_column("Symbol", style="cyan")
        table.add_column("P&L", justify="right")
        table.add_column("Trades", justify="center")
        table.add_column("Win Rate", justify="center")
        table.add_column("Sharpe", justify="center")
        table.add_column("Max DD", justify="right")
        table.add_column("Weight", justify="center")

        # Calculate total P&L for weight calculation
        total_pnl = sum(m.total_pnl_dollars for m in self.metrics.values())

        # Sort by P&L
        sorted_pairs = sorted(
            self.metrics.items(),
            key=lambda x: x[1].total_pnl_dollars,
            reverse=True
        )

        for symbol, metrics in sorted_pairs:
            # P&L coloring
            pnl = metrics.total_pnl_dollars
            pnl_style = "green" if pnl >= 0 else "red"
            pnl_text = Text(f"${pnl:,.2f}", style=pnl_style)

            # Win rate coloring
            win_rate = metrics.win_rate * 100 if metrics.win_rate else 0
            wr_style = "green" if win_rate >= 50 else "red"
            wr_text = Text(f"{win_rate:.1f}%", style=wr_style)

            # Sharpe coloring
            sharpe = metrics.sharpe_ratio
            sharpe_style = "green" if sharpe > 1 else "yellow" if sharpe > 0 else "red"
            sharpe_text = Text(f"{sharpe:.2f}", style=sharpe_style)

            # Weight calculation
            weight = (abs(pnl) / abs(total_pnl) * 100) if total_pnl else 0

            table.add_row(
                symbol,
                pnl_text,
                str(metrics.total_trades),
                wr_text,
                sharpe_text,
                f"${metrics.max_drawdown_dollars:.2f}",
                f"{weight:.1f}%"
            )

        return Panel(table, title="Performance Attribution", border_style="green")

    def update_metrics(self, metrics: dict[str, PairMetrics]) -> None:
        """Update performance metrics."""
        self.metrics = metrics


class CapitalAllocation(Widget):
    """Display capital allocation across pairs."""

    allocations = reactive({}, recompose=True)
    total_capital = reactive(Decimal("0"), recompose=True)

    def __init__(self, allocations: dict[str, Decimal] = None, total_capital: Decimal = None, **kwargs):
        super().__init__(**kwargs)
        self.allocations = allocations or {}
        self.total_capital = total_capital or Decimal("0")

    def render(self) -> RenderableType:
        """Render capital allocation display."""
        if not self.allocations:
            return Panel("No allocation data", title="Capital Allocation")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Symbol", style="cyan")
        table.add_column("Allocated", justify="right")
        table.add_column("Percentage", justify="center")
        table.add_column("Bar", justify="left", width=20)

        # Sort by allocation
        sorted_allocs = sorted(
            self.allocations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for symbol, amount in sorted_allocs:
            percentage = (amount / self.total_capital * 100) if self.total_capital else 0

            # Create visual bar
            bar_length = int(percentage / 5)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)

            # Color based on concentration
            if percentage > 40:
                bar_style = "red"
            elif percentage > 25:
                bar_style = "yellow"
            else:
                bar_style = "green"

            table.add_row(
                symbol,
                f"${amount:,.2f}",
                f"{percentage:.1f}%",
                Text(bar, style=bar_style)
            )

        # Add total row
        table.add_row(
            "TOTAL",
            f"${sum(self.allocations.values()):,.2f}",
            "100%",
            "",
            style="bold"
        )

        return Panel(table, title="Capital Allocation", border_style="yellow")

    def update_allocations(self, allocations: dict[str, Decimal], total_capital: Decimal) -> None:
        """Update allocation data."""
        self.allocations = allocations
        self.total_capital = total_capital


class PortfolioRiskIndicator(Widget):
    """Display portfolio-level risk indicators."""

    portfolio_risk = reactive(None, recompose=True)

    def __init__(self, portfolio_risk: PortfolioRisk = None, **kwargs):
        super().__init__(**kwargs)
        self.portfolio_risk = portfolio_risk

    def render(self) -> RenderableType:
        """Render portfolio risk indicators."""
        if not self.portfolio_risk:
            return Panel("No risk data available", title="Portfolio Risk")

        risk = self.portfolio_risk

        # Create risk summary
        lines = []

        # Risk score with color coding
        risk_score = risk.risk_score
        if risk_score > 75:
            risk_style = "red bold"
            risk_level = "HIGH RISK ⚠️"
        elif risk_score > 50:
            risk_style = "yellow"
            risk_level = "MODERATE"
        else:
            risk_style = "green"
            risk_level = "LOW"

        lines.append(f"Risk Level: [{risk_style}]{risk_level}[/{risk_style}]")
        lines.append(f"Risk Score: {risk_score:.1f}/100")
        lines.append("")

        # Key metrics
        lines.append(f"Total Exposure: ${risk.total_exposure_dollars:,.2f}")
        lines.append(f"Position Count: {risk.position_count}")
        lines.append(f"Available Capital: ${risk.available_capital:,.2f}")
        lines.append(f"Max Drawdown: ${risk.max_drawdown_dollars:,.2f}")
        lines.append("")

        # Risk components
        lines.append("Risk Components:")
        lines.append(f"  Correlation Risk: {risk.correlation_risk:.1%}")
        lines.append(f"  Concentration Risk: {risk.concentration_risk:.1%}")

        # Warnings
        if risk.warnings:
            lines.append("")
            lines.append("⚠️ Warnings:")
            for warning in risk.warnings[:3]:  # Show top 3 warnings
                lines.append(f"  • {warning}")

        content = "\n".join(lines)

        # Panel style based on risk level
        if risk.is_high_risk:
            border_style = "red"
        elif risk.risk_score > 50:
            border_style = "yellow"
        else:
            border_style = "green"

        return Panel(content, title="Portfolio Risk", border_style=border_style)

    def update_risk(self, portfolio_risk: PortfolioRisk) -> None:
        """Update portfolio risk data."""
        self.portfolio_risk = portfolio_risk


class SignalQueueWidget(Widget):
    """Display pending signals in queue."""

    signals = reactive([], recompose=True)

    def __init__(self, signals: list[Signal] = None, **kwargs):
        super().__init__(**kwargs)
        self.signals = signals or []

    def render(self) -> RenderableType:
        """Render signal queue."""
        if not self.signals:
            return Panel("No pending signals", title="Signal Queue")

        table = Table(show_header=True, header_style="bold")
        table.add_column("#", justify="center", width=3)
        table.add_column("Symbol", style="cyan")
        table.add_column("Type", justify="center")
        table.add_column("Priority", justify="center")
        table.add_column("Confidence", justify="center")
        table.add_column("Strategy", justify="left")
        table.add_column("Expires", justify="right")

        for i, signal in enumerate(self.signals[:10], 1):  # Show top 10
            # Signal type coloring
            sig_type = signal.signal_type.value
            type_style = "green" if sig_type == "BUY" else "red" if sig_type == "SELL" else "yellow"
            type_text = Text(sig_type, style=type_style)

            # Priority coloring
            priority = signal.priority
            pri_style = "red" if priority >= 80 else "yellow" if priority >= 50 else "white"
            pri_text = Text(str(priority), style=pri_style)

            # Confidence coloring
            confidence = float(signal.confidence_score * 100)
            conf_style = "green" if confidence >= 80 else "yellow" if confidence >= 60 else "red"
            conf_text = Text(f"{confidence:.0f}%", style=conf_style)

            # Time to expiry
            if signal.expiry_time:
                ttl = (signal.expiry_time - datetime.utcnow()).total_seconds()
                if ttl > 0:
                    ttl_str = f"{ttl/60:.0f}m"
                else:
                    ttl_str = "EXPIRED"
            else:
                ttl_str = "∞"

            table.add_row(
                str(i),
                signal.symbol,
                type_text,
                pri_text,
                conf_text,
                signal.strategy_name[:15],
                ttl_str
            )

        return Panel(table, title=f"Signal Queue ({len(self.signals)} pending)", border_style="blue")

    def update_signals(self, signals: list[Signal]) -> None:
        """Update signal queue."""
        self.signals = signals


class MultiPairDashboard(Container):
    """Main multi-pair trading dashboard container."""

    def compose(self) -> ComposeResult:
        """Compose the multi-pair dashboard."""
        with Vertical():
            # Top row: Positions and Risk
            with Horizontal(classes="dashboard-row"):
                yield PositionGrid(id="position-grid")
                yield PortfolioRiskIndicator(id="risk-indicator")

            # Middle row: Correlations and Performance
            with Horizontal(classes="dashboard-row"):
                yield CorrelationHeatmap(id="correlation-heatmap")
                yield PerformanceAttribution(id="performance-attribution")

            # Bottom row: Allocations and Signal Queue
            with Horizontal(classes="dashboard-row"):
                yield CapitalAllocation(id="capital-allocation")
                yield SignalQueueWidget(id="signal-queue")

    def update_all(
        self,
        positions: list[Position] = None,
        portfolio_risk: PortfolioRisk = None,
        correlations: dict = None,
        performance_metrics: dict[str, PairMetrics] = None,
        allocations: dict[str, Decimal] = None,
        total_capital: Decimal = None,
        signals: list[Signal] = None
    ) -> None:
        """Update all dashboard components."""
        if positions is not None:
            self.query_one("#position-grid", PositionGrid).update_positions(positions)

        if portfolio_risk is not None:
            self.query_one("#risk-indicator", PortfolioRiskIndicator).update_risk(portfolio_risk)

        if correlations is not None:
            self.query_one("#correlation-heatmap", CorrelationHeatmap).update_correlations(correlations)

        if performance_metrics is not None:
            self.query_one("#performance-attribution", PerformanceAttribution).update_metrics(performance_metrics)

        if allocations is not None and total_capital is not None:
            self.query_one("#capital-allocation", CapitalAllocation).update_allocations(allocations, total_capital)

        if signals is not None:
            self.query_one("#signal-queue", SignalQueueWidget).update_signals(signals)
