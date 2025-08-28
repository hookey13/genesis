"""Tier gate progress dashboard widget.

Displays real-time progress towards tier requirements and
celebrates transitions with appropriate restraint.
"""

from typing import Any, Optional

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Label, ProgressBar, Static

from genesis.engine.state_machine import TierStateMachine


class GateProgressBar(Widget):
    """Individual gate requirement progress bar."""

    def __init__(
        self,
        gate_name: str,
        current_value: float,
        required_value: float,
        unit: str = "",
        **kwargs,
    ):
        """Initialize gate progress bar.

        Args:
            gate_name: Name of the gate requirement
            current_value: Current progress value
            required_value: Required value to pass gate
            unit: Unit of measurement
        """
        super().__init__(**kwargs)
        self.gate_name = gate_name
        self.current_value = current_value
        self.required_value = required_value
        self.unit = unit
        self.progress = (
            min(current_value / required_value * 100, 100) if required_value > 0 else 0
        )

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with Horizontal(classes="gate-progress-container"):
            yield Label(f"{self.gate_name}:", classes="gate-label")
            yield ProgressBar(total=100, progress=self.progress, classes="gate-bar")
            yield Label(
                f"{self.current_value:.1f}/{self.required_value:.1f} {self.unit}",
                classes="gate-value",
            )

    def update_progress(self, current_value: float) -> None:
        """Update progress value.

        Args:
            current_value: New current value
        """
        self.current_value = current_value
        self.progress = (
            min(current_value / self.required_value * 100, 100)
            if self.required_value > 0
            else 0
        )
        self.refresh()


class TierCeremonyAnimation(Widget):
    """Tier transition ceremony animation."""

    def __init__(self, from_tier: str, to_tier: str, **kwargs):
        """Initialize ceremony animation.

        Args:
            from_tier: Previous tier
            to_tier: New tier achieved
        """
        super().__init__(**kwargs)
        self.from_tier = from_tier
        self.to_tier = to_tier
        self.animation_frame = 0
        self.timer: Optional[Timer] = None

    def on_mount(self) -> None:
        """Start animation on mount."""
        self.timer = self.set_interval(0.1, self.animate)

    def animate(self) -> None:
        """Animate the ceremony."""
        self.animation_frame += 1
        if self.animation_frame > 30:  # 3 seconds
            if self.timer:
                self.timer.stop()
            self.remove()
        self.refresh()

    def render(self) -> RenderableType:
        """Render the animation."""
        # Simple fade-in effect
        opacity = min(self.animation_frame / 10, 1.0)

        text = Text()
        text.append("\n" * 2)
        text.append(f"  Tier Achieved: {self.to_tier}  ", style="bold green on black")
        text.append("\n")
        text.append(f"  Previous: {self.from_tier}  ", style="dim white")
        text.append("\n" * 2)

        if self.animation_frame > 20:
            # Fade out
            opacity = max(0, 1.0 - (self.animation_frame - 20) / 10)

        return Panel(
            text,
            title="Tier Transition",
            border_style="green" if opacity > 0.5 else "dim green",
            padding=(1, 2),
        )


class FeatureTutorial(Widget):
    """Tutorial display for newly unlocked features."""

    def __init__(self, feature_name: str, description: str, **kwargs):
        """Initialize feature tutorial.

        Args:
            feature_name: Name of the unlocked feature
            description: Tutorial description
        """
        super().__init__(**kwargs)
        self.feature_name = feature_name
        self.description = description

    def compose(self) -> ComposeResult:
        """Compose the tutorial widget."""
        with Container(classes="tutorial-container"):
            yield Label(f"New Feature: {self.feature_name}", classes="tutorial-title")
            yield Static(self.description, classes="tutorial-text")
            yield Button("Dismiss", id="dismiss-tutorial", classes="tutorial-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "dismiss-tutorial":
            self.remove()


class TierGateProgressWidget(Widget):
    """Main tier gate progress dashboard widget."""

    # Reactive properties
    current_tier = reactive("SNIPER")
    next_tier = reactive("HUNTER")
    gates_completed = reactive(0)
    total_gates = reactive(4)

    def __init__(self, state_machine: TierStateMachine, **kwargs):
        """Initialize tier gate progress widget.

        Args:
            state_machine: Tier state machine instance
        """
        super().__init__(**kwargs)
        self.state_machine = state_machine
        self.gate_bars: dict[str, GateProgressBar] = {}
        self.update_timer: Optional[Timer] = None

    def on_mount(self) -> None:
        """Start update timer on mount."""
        self.update_timer = self.set_interval(1.0, self.update_progress)
        self.update_progress()

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with Container(classes="tier-progress-container"):
            # Header
            yield Label(
                f"Current Tier: {self.current_tier} → Next: {self.next_tier}",
                classes="tier-header",
            )

            # Overall progress
            with Horizontal(classes="overall-progress"):
                yield Label("Overall Progress:", classes="progress-label")
                yield ProgressBar(
                    total=self.total_gates,
                    progress=self.gates_completed,
                    id="overall-bar",
                )
                yield Label(
                    f"{self.gates_completed}/{self.total_gates} gates", id="gates-count"
                )

            # Individual gate progress bars
            with Vertical(id="gate-bars-container"):
                # These will be dynamically created
                pass

    async def update_progress(self) -> None:
        """Update gate progress from state machine."""
        try:
            # Get current requirements
            requirements = self.state_machine.get_tier_requirements(self.next_tier)

            if not requirements:
                return

            # Update or create progress bars for each requirement
            gates_container = self.query_one("#gate-bars-container")

            # Balance requirement
            if "min_balance" in requirements:
                if "balance" not in self.gate_bars:
                    bar = GateProgressBar(
                        "Minimum Balance", 0, requirements["min_balance"], "USDT"
                    )
                    self.gate_bars["balance"] = bar
                    gates_container.mount(bar)

            # Trades requirement
            if "min_trades" in requirements:
                if "trades" not in self.gate_bars:
                    bar = GateProgressBar(
                        "Minimum Trades", 0, requirements["min_trades"], "trades"
                    )
                    self.gate_bars["trades"] = bar
                    gates_container.mount(bar)

            # Tilt events requirement
            if "max_tilt_events" in requirements:
                if "tilt" not in self.gate_bars:
                    bar = GateProgressBar(
                        "Max Tilt Events", 0, requirements["max_tilt_events"], "events"
                    )
                    self.gate_bars["tilt"] = bar
                    gates_container.mount(bar)

            # Paper trading requirement
            if requirements.get("paper_trading_required"):
                if "paper_trading" not in self.gate_bars:
                    bar = GateProgressBar("Paper Trading", 0, 1, "complete")
                    self.gate_bars["paper_trading"] = bar
                    gates_container.mount(bar)

            # Update progress values (would fetch from database in production)
            # Placeholder values for demonstration
            if "balance" in self.gate_bars:
                self.gate_bars["balance"].update_progress(1800)
            if "trades" in self.gate_bars:
                self.gate_bars["trades"].update_progress(35)
            if "tilt" in self.gate_bars:
                self.gate_bars["tilt"].update_progress(1)
            if "paper_trading" in self.gate_bars:
                self.gate_bars["paper_trading"].update_progress(0)

            # Update overall progress
            completed = sum(1 for bar in self.gate_bars.values() if bar.progress >= 100)
            self.gates_completed = completed

        except Exception as e:
            self.log.error(f"Failed to update gate progress: {e}")

    async def trigger_ceremony(self, from_tier: str, to_tier: str) -> None:
        """Trigger tier transition ceremony.

        Args:
            from_tier: Previous tier
            to_tier: New tier achieved
        """
        # Mount ceremony animation
        ceremony = TierCeremonyAnimation(from_tier, to_tier)
        self.mount(ceremony)

        # Update current tier
        self.current_tier = to_tier
        self.next_tier = self.state_machine.get_next_tier(to_tier) or "MAX"

        # Clear old progress bars
        for bar in self.gate_bars.values():
            bar.remove()
        self.gate_bars.clear()

        # Show tutorials for new features
        features = self.state_machine.get_available_features(to_tier)
        await self.show_feature_tutorials(features)

    async def show_feature_tutorials(self, features: list[str]) -> None:
        """Show tutorials for newly unlocked features.

        Args:
            features: List of feature names
        """
        # Define tutorials (would be loaded from config in production)
        tutorials = {
            "iceberg_orders": "Iceberg orders allow you to split large orders into smaller chunks.",
            "multi_pair_trading": "Trade multiple pairs simultaneously with coordinated risk management.",
            "twap_execution": "Time-Weighted Average Price execution for optimal trade timing.",
            "statistical_arbitrage": "Exploit price inefficiencies using statistical models.",
        }

        # Show relevant tutorials
        for feature in features:
            if feature in tutorials:
                tutorial = FeatureTutorial(feature, tutorials[feature])
                self.mount(tutorial)
                # Only show first tutorial to avoid overwhelming
                break


class TierTransitionHistory(Widget):
    """Widget showing historical tier transitions."""

    def __init__(self, transitions: list[dict[str, Any]], **kwargs):
        """Initialize transition history.

        Args:
            transitions: List of transition records
        """
        super().__init__(**kwargs)
        self.transitions = transitions

    def render(self) -> RenderableType:
        """Render the transition history."""
        table = Table(title="Tier Transition History")
        table.add_column("Date", style="cyan")
        table.add_column("From", style="yellow")
        table.add_column("To", style="green")
        table.add_column("Reason", style="white")
        table.add_column("Duration", style="magenta")

        for transition in self.transitions[-10:]:  # Show last 10
            table.add_row(
                transition.get("date", ""),
                transition.get("from_tier", ""),
                transition.get("to_tier", ""),
                transition.get("reason", ""),
                transition.get("duration", ""),
            )

        return Panel(table, title="Transition History", border_style="blue")
