from __future__ import annotations

from typing import Optional

"""Tilt indicator widget for multi-level tilt visualization."""

from datetime import UTC, datetime
from decimal import Decimal

import structlog
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from genesis.core.events import EventType
from genesis.engine.event_bus import EventBus
from genesis.tilt.detector import TiltLevel

logger = structlog.get_logger(__name__)


class TiltIndicator(Widget):
    """Multi-level tilt indicator with visual warnings."""

    # Reactive attributes for real-time updates
    tilt_level = reactive(TiltLevel.NORMAL)
    tilt_score = reactive(0)
    anomaly_count = reactive(0)
    intervention_message = reactive("")

    # Border styles by level
    BORDER_STYLES = {
        TiltLevel.NORMAL: "",
        TiltLevel.LEVEL1: "border: solid yellow;",
        TiltLevel.LEVEL2: "border: solid orange;",
        TiltLevel.LEVEL3: "border: solid red;"
    }

    # Colors for different elements
    LEVEL_COLORS = {
        TiltLevel.NORMAL: "green",
        TiltLevel.LEVEL1: "yellow",
        TiltLevel.LEVEL2: "dark_orange",
        TiltLevel.LEVEL3: "red"
    }

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        **kwargs
    ):
        """Initialize tilt indicator.

        Args:
            event_bus: Event bus for subscribing to tilt events
        """
        super().__init__(**kwargs)
        self.event_bus = event_bus
        self.anomalies: list[dict] = []
        self.last_update = datetime.now(UTC)

        # Subscribe to tilt events if event bus provided
        if self.event_bus:
            self._subscribe_to_events()

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Container(id="tilt-container"):
            # Main tilt status
            with Vertical(id="tilt-status"):
                yield Label("Tilt Detection System", id="tilt-title")
                yield Static(self._render_level(), id="tilt-level-display")
                yield Static(self._render_score_bar(), id="tilt-score-bar")
                yield Static(self._render_anomaly_count(), id="anomaly-counter")

            # Intervention message (if any)
            with Container(id="intervention-container"):
                yield Static("", id="intervention-message")

            # Anomaly details
            with Container(id="anomaly-details"):
                yield Static(self._render_anomaly_list(), id="anomaly-list")

    def _render_level(self) -> RenderableType:
        """Render the current tilt level."""
        color = self.LEVEL_COLORS[self.tilt_level]
        level_text = {
            TiltLevel.NORMAL: "✓ Normal",
            TiltLevel.LEVEL1: "⚠ Level 1 - Caution",
            TiltLevel.LEVEL2: "⚠ Level 2 - Warning",
            TiltLevel.LEVEL3: "⛔ Level 3 - Lockout"
        }[self.tilt_level]

        return Text(level_text, style=f"bold {color}")

    def _render_score_bar(self) -> RenderableType:
        """Render the tilt score progress bar."""
        # Create a text-based progress bar
        bar_width = 30
        filled = int((self.tilt_score / 100) * bar_width)
        empty = bar_width - filled

        # Color based on score
        if self.tilt_score < 30:
            color = "green"
        elif self.tilt_score < 60:
            color = "yellow"
        elif self.tilt_score < 80:
            color = "dark_orange"
        else:
            color = "red"

        bar = f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"
        score_text = f"Tilt Score: {self.tilt_score}/100 {bar}"

        return Text.from_markup(score_text)

    def _render_anomaly_count(self) -> RenderableType:
        """Render the anomaly counter."""
        if self.anomaly_count == 0:
            return Text("No anomalies detected", style="green")
        elif self.anomaly_count < 3:
            return Text(f"Anomalies: {self.anomaly_count}", style="yellow")
        elif self.anomaly_count < 6:
            return Text(f"Anomalies: {self.anomaly_count}", style="dark_orange")
        else:
            return Text(f"Anomalies: {self.anomaly_count}", style="bold red")

    def _render_anomaly_list(self) -> RenderableType:
        """Render the list of current anomalies."""
        if not self.anomalies:
            return Text("No active anomalies", style="dim")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Indicator", style="cyan")
        table.add_column("Severity", justify="center")
        table.add_column("Description")

        for anomaly in self.anomalies[:5]:  # Show top 5
            severity = anomaly.get("severity", 0)
            if severity <= 3:
                severity_style = "green"
            elif severity <= 6:
                severity_style = "yellow"
            else:
                severity_style = "red"

            table.add_row(
                anomaly.get("indicator", "Unknown"),
                Text(str(severity), style=severity_style),
                anomaly.get("description", "")[:40]
            )

        return Panel(table, title="Active Anomalies", border_style="dim")

    def update_tilt_status(
        self,
        level: TiltLevel,
        score: int,
        anomalies: list[dict],
        message: Optional[str] = None
    ) -> None:
        """Update the tilt status display.

        Args:
            level: Current tilt level
            score: Current tilt score (0-100)
            anomalies: List of active anomalies
            message: Optional intervention message
        """
        self.tilt_level = level
        self.tilt_score = score
        self.anomaly_count = len(anomalies)
        self.anomalies = anomalies
        self.intervention_message = message or ""
        self.last_update = datetime.now(UTC)

        # Update display elements
        self._update_display()

        # Apply border style based on level
        self._apply_border_style()

        # Show intervention message if present
        if message:
            self._show_intervention_message(message)

    def _update_display(self) -> None:
        """Update all display elements."""
        # Update level display
        if level_display := self.query_one("#tilt-level-display", Static):
            level_display.update(self._render_level())

        # Update score bar
        if score_bar := self.query_one("#tilt-score-bar", Static):
            score_bar.update(self._render_score_bar())

        # Update anomaly counter
        if counter := self.query_one("#anomaly-counter", Static):
            counter.update(self._render_anomaly_count())

        # Update anomaly list
        if anomaly_list := self.query_one("#anomaly-list", Static):
            anomaly_list.update(self._render_anomaly_list())

    def _apply_border_style(self) -> None:
        """Apply border style based on tilt level."""
        container = self.query_one("#tilt-container", Container)
        if container:
            # Apply the appropriate border style
            style = self.BORDER_STYLES[self.tilt_level]
            if style:
                container.styles.border = ("solid", self.LEVEL_COLORS[self.tilt_level])
            else:
                container.styles.border = ("none", "")

    def _show_intervention_message(self, message: str) -> None:
        """Show intervention message with auto-dismiss.

        Args:
            message: Intervention message to display
        """
        msg_widget = self.query_one("#intervention-message", Static)
        if msg_widget:
            # Style based on level
            color = self.LEVEL_COLORS[self.tilt_level]
            styled_message = Text(message, style=f"{color}")
            msg_widget.update(Panel(styled_message, border_style=color))

            # Auto-dismiss after 10 seconds (except for Level 3)
            if self.tilt_level != TiltLevel.LEVEL3:
                self.set_timer(10, self._clear_intervention_message)

    def _clear_intervention_message(self) -> None:
        """Clear the intervention message."""
        msg_widget = self.query_one("#intervention-message", Static)
        if msg_widget:
            msg_widget.update("")

    def _subscribe_to_events(self) -> None:
        """Subscribe to tilt detection events."""
        if not self.event_bus:
            return

        # Subscribe to all tilt level events
        async def handle_tilt_event(event_type: EventType, data: dict) -> None:
            """Handle tilt detection events."""
            if event_type == EventType.TILT_LEVEL1_DETECTED:
                self.update_tilt_status(
                    TiltLevel.LEVEL1,
                    data.get("tilt_score", 0),
                    data.get("anomalies", []),
                    "Taking a moment to breathe can improve your trading decisions."
                )
            elif event_type == EventType.TILT_LEVEL2_DETECTED:
                self.update_tilt_status(
                    TiltLevel.LEVEL2,
                    data.get("tilt_score", 0),
                    data.get("anomalies", []),
                    "Your trading patterns suggest heightened stress. Position sizes reduced for safety."
                )
            elif event_type == EventType.TILT_LEVEL3_DETECTED:
                self.update_tilt_status(
                    TiltLevel.LEVEL3,
                    data.get("tilt_score", 0),
                    data.get("anomalies", []),
                    "Let's take a break. Trading paused to protect your capital."
                )
            elif event_type == EventType.TILT_RECOVERED:
                self.update_tilt_status(
                    TiltLevel.NORMAL,
                    0,
                    [],
                    "Well done! You've recovered your composure."
                )

        # Register event handlers
        self.event_bus.subscribe(EventType.TILT_LEVEL1_DETECTED, handle_tilt_event)
        self.event_bus.subscribe(EventType.TILT_LEVEL2_DETECTED, handle_tilt_event)
        self.event_bus.subscribe(EventType.TILT_LEVEL3_DETECTED, handle_tilt_event)
        self.event_bus.subscribe(EventType.TILT_RECOVERED, handle_tilt_event)

    @property
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed based on tilt level.

        Returns:
            True if trading is allowed
        """
        return self.tilt_level != TiltLevel.LEVEL3

    @property
    def position_size_multiplier(self) -> Decimal:
        """Get position size multiplier based on tilt level.

        Returns:
            Multiplier for position sizing
        """
        if self.tilt_level == TiltLevel.LEVEL2:
            return Decimal("0.5")  # 50% reduction
        elif self.tilt_level == TiltLevel.LEVEL3:
            return Decimal("0")  # No trading
        else:
            return Decimal("1.0")  # Normal sizing

    def get_status_summary(self) -> dict:
        """Get current tilt status summary.

        Returns:
            Dictionary with status information
        """
        return {
            "level": self.tilt_level.value,
            "score": self.tilt_score,
            "anomaly_count": self.anomaly_count,
            "trading_allowed": self.is_trading_allowed,
            "position_multiplier": float(self.position_size_multiplier),
            "has_intervention": bool(self.intervention_message),
            "last_update": self.last_update.isoformat()
        }
