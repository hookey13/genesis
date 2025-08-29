"""
Spread Alerts Widget

Terminal UI widget for displaying spread compression alerts and notifications
with real-time updates and filtering capabilities.
"""

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import structlog
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Static

from genesis.analytics.spread_analyzer import SpreadCompressionEvent

logger = structlog.get_logger(__name__)

AlertLevel = Literal["info", "warning", "critical"]


@dataclass
class SpreadAlert:
    """Spread alert notification"""

    symbol: str
    alert_type: str
    message: str
    level: AlertLevel
    timestamp: datetime
    data: dict | None = None


class SpreadAlertsWidget(Widget):
    """
    Widget for displaying and managing spread alerts
    with filtering and notification capabilities
    """

    # Reactive attributes
    filter_symbol: reactive[str | None] = reactive(None)
    filter_level: reactive[AlertLevel | None] = reactive(None)
    max_alerts: reactive[int] = reactive(50)

    # CSS for styling
    DEFAULT_CSS = """
    SpreadAlertsWidget {
        height: 100%;
        width: 100%;
        background: $surface;
        border: solid $primary;
    }

    .alerts-container {
        padding: 1;
    }

    .filter-bar {
        height: 3;
        margin-bottom: 1;
    }

    .alerts-display {
        height: 1fr;
        overflow-y: auto;
    }

    .alert-info {
        color: $text;
    }

    .alert-warning {
        color: yellow;
    }

    .alert-critical {
        color: red;
    }
    """

    def __init__(self, max_alerts: int = 50, **kwargs):
        """
        Initialize spread alerts widget

        Args:
            max_alerts: Maximum number of alerts to keep
        """
        super().__init__(**kwargs)
        self.max_alerts = max_alerts
        self.alerts: deque[SpreadAlert] = deque(maxlen=max_alerts)
        self._logger = logger.bind(component="SpreadAlertsWidget")

    def compose(self) -> ComposeResult:
        """Compose widget layout"""
        with Vertical(classes="alerts-container"):
            # Filter bar
            with Horizontal(classes="filter-bar"):
                yield Label("Filter: ")
                yield Input(placeholder="Symbol", id="filter-symbol")
                yield Button("All", id="level-all", variant="primary")
                yield Button("Info", id="level-info")
                yield Button("Warning", id="level-warning")
                yield Button("Critical", id="level-critical")
                yield Button("Clear", id="clear-alerts", variant="error")

            # Alerts display
            yield Static(
                self._render_alerts(), classes="alerts-display", id="alerts-list"
            )

    def _render_alerts(self) -> RenderableType:
        """
        Render the alerts list

        Returns:
            Rich renderable for alerts
        """
        # Filter alerts
        filtered_alerts = self._filter_alerts()

        if not filtered_alerts:
            return Panel(
                "[dim]No alerts to display[/dim]",
                title=f"Spread Alerts ({len(self.alerts)} total)",
            )

        # Create table
        table = Table(
            title=f"Spread Alerts ({len(filtered_alerts)} of {len(self.alerts)})",
            show_header=True,
            header_style="bold magenta",
            show_lines=False,
            expand=True,
        )

        # Add columns
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Symbol", style="white", width=10)
        table.add_column("Type", style="yellow", width=15)
        table.add_column("Message", style="white", no_wrap=False)
        table.add_column("Level", justify="center", width=10)

        # Add rows (most recent first)
        for alert in reversed(filtered_alerts):
            time_str = alert.timestamp.strftime("%H:%M:%S")
            level_style = self._get_level_style(alert.level)
            level_icon = self._get_level_icon(alert.level)

            table.add_row(
                time_str,
                alert.symbol,
                alert.alert_type,
                alert.message,
                Text(level_icon, style=level_style),
            )

        return table

    def _filter_alerts(self) -> list[SpreadAlert]:
        """
        Filter alerts based on current filters

        Returns:
            Filtered list of alerts
        """
        filtered = list(self.alerts)

        # Filter by symbol
        if self.filter_symbol:
            filtered = [
                a for a in filtered if self.filter_symbol.lower() in a.symbol.lower()
            ]

        # Filter by level
        if self.filter_level:
            filtered = [a for a in filtered if a.level == self.filter_level]

        return filtered

    def _get_level_style(self, level: AlertLevel) -> str:
        """
        Get style for alert level

        Args:
            level: Alert level

        Returns:
            Rich style string
        """
        return {"info": "blue", "warning": "yellow", "critical": "red"}.get(
            level, "white"
        )

    def _get_level_icon(self, level: AlertLevel) -> str:
        """
        Get icon for alert level

        Args:
            level: Alert level

        Returns:
            Icon string
        """
        return {"info": "ℹ", "warning": "⚠", "critical": "🔴"}.get(level, "•")

    def add_alert(
        self,
        symbol: str,
        alert_type: str,
        message: str,
        level: AlertLevel = "info",
        data: dict | None = None,
    ) -> None:
        """
        Add a new alert

        Args:
            symbol: Trading pair symbol
            alert_type: Type of alert
            message: Alert message
            level: Alert severity level
            data: Optional additional data
        """
        alert = SpreadAlert(
            symbol=symbol,
            alert_type=alert_type,
            message=message,
            level=level,
            timestamp=datetime.now(UTC),
            data=data,
        )

        self.alerts.append(alert)
        self._refresh_display()

        self._logger.info(
            "Alert added", symbol=symbol, alert_type=alert_type, level=level
        )

    def add_compression_alert(self, event: SpreadCompressionEvent) -> None:
        """
        Add alert for spread compression event

        Args:
            event: SpreadCompressionEvent
        """
        message = (
            f"Spread compressed to {event.current_spread:.2f} bps "
            f"({event.compression_ratio:.1%} of average)"
        )

        level = "warning" if event.compression_ratio < Decimal("0.5") else "info"

        self.add_alert(
            symbol=event.symbol,
            alert_type="SPREAD_COMPRESSION",
            message=message,
            level=level,
            data={
                "current_spread": float(event.current_spread),
                "average_spread": float(event.average_spread),
                "compression_ratio": float(event.compression_ratio),
            },
        )

    def add_imbalance_alert(
        self, symbol: str, ratio: Decimal, bid_weight: Decimal, ask_weight: Decimal
    ) -> None:
        """
        Add alert for order imbalance

        Args:
            symbol: Trading pair symbol
            ratio: Imbalance ratio
            bid_weight: Bid weight
            ask_weight: Ask weight
        """
        if ratio > Decimal("2.0"):
            message = f"Strong bid pressure detected (ratio: {ratio:.2f})"
            level = "warning"
        elif ratio < Decimal("0.5"):
            message = f"Strong ask pressure detected (ratio: {ratio:.2f})"
            level = "warning"
        else:
            message = f"Order imbalance: {ratio:.2f}"
            level = "info"

        self.add_alert(
            symbol=symbol,
            alert_type="ORDER_IMBALANCE",
            message=message,
            level=level,
            data={
                "ratio": float(ratio),
                "bid_weight": float(bid_weight),
                "ask_weight": float(ask_weight),
            },
        )

    def add_anomaly_alert(
        self,
        symbol: str,
        anomaly_type: str,
        severity: Decimal,
        current_value: Decimal,
        expected_value: Decimal,
    ) -> None:
        """
        Add alert for spread anomaly

        Args:
            symbol: Trading pair symbol
            anomaly_type: Type of anomaly
            severity: Severity score
            current_value: Current spread value
            expected_value: Expected spread value
        """
        deviation = abs(current_value - expected_value)
        message = (
            f"Anomaly detected: {anomaly_type} "
            f"(current: {current_value:.2f}, expected: {expected_value:.2f})"
        )

        if severity > Decimal("7"):
            level = "critical"
        elif severity > Decimal("4"):
            level = "warning"
        else:
            level = "info"

        self.add_alert(
            symbol=symbol,
            alert_type=f"ANOMALY_{anomaly_type.upper()}",
            message=message,
            level=level,
            data={
                "anomaly_type": anomaly_type,
                "severity": float(severity),
                "current_value": float(current_value),
                "expected_value": float(expected_value),
                "deviation": float(deviation),
            },
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events"""
        button_id = event.button.id

        if button_id == "clear-alerts":
            self.clear_alerts()
        elif button_id == "level-all":
            self.filter_level = None
        elif button_id == "level-info":
            self.filter_level = "info"
        elif button_id == "level-warning":
            self.filter_level = "warning"
        elif button_id == "level-critical":
            self.filter_level = "critical"

        # Update button styles
        for btn_id in ["level-all", "level-info", "level-warning", "level-critical"]:
            btn = self.query_one(f"#{btn_id}", Button)
            if btn_id == "level-all":
                btn.variant = "primary" if self.filter_level is None else "default"
            else:
                level = btn_id.replace("level-", "")
                btn.variant = "primary" if self.filter_level == level else "default"

        self._refresh_display()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        if event.input.id == "filter-symbol":
            self.filter_symbol = event.value if event.value else None
            self._refresh_display()

    def _refresh_display(self) -> None:
        """Refresh the alerts display"""
        display = self.query_one("#alerts-list", Static)
        display.update(self._render_alerts())

    def clear_alerts(self) -> None:
        """Clear all alerts"""
        self.alerts.clear()
        self._refresh_display()
        self._logger.info("Alerts cleared")

    def get_alert_summary(self) -> dict:
        """
        Get summary of current alerts

        Returns:
            Dictionary with alert statistics
        """
        total = len(self.alerts)
        by_level = {
            "info": sum(1 for a in self.alerts if a.level == "info"),
            "warning": sum(1 for a in self.alerts if a.level == "warning"),
            "critical": sum(1 for a in self.alerts if a.level == "critical"),
        }

        by_type = {}
        for alert in self.alerts:
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1

        return {
            "total": total,
            "by_level": by_level,
            "by_type": by_type,
            "oldest": self.alerts[0].timestamp if self.alerts else None,
            "newest": self.alerts[-1].timestamp if self.alerts else None,
        }
