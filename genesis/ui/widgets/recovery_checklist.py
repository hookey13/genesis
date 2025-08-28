from __future__ import annotations

from typing import Optional

"""Recovery checklist UI widget for tilt recovery."""

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Checkbox, Label, ProgressBar, Static

from genesis.tilt.recovery_checklist import (
    ChecklistItemType,
    RecoveryChecklistManager,
)


class ChecklistItemCompleted(Message):
    """Message sent when a checklist item is completed."""

    def __init__(self, item_name: str, is_complete: bool) -> None:
        """Initialize message.

        Args:
            item_name: Name of completed item
            is_complete: Whether entire checklist is complete
        """
        super().__init__()
        self.item_name = item_name
        self.is_complete = is_complete


class RecoveryChecklistWidget(Container):
    """Widget for recovery checklist during tilt recovery."""

    DEFAULT_CSS = """
    RecoveryChecklistWidget {
        height: auto;
        border: solid $accent;
        padding: 1;
        margin: 1;
    }

    RecoveryChecklistWidget .title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    RecoveryChecklistWidget .subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }

    RecoveryChecklistWidget .progress-container {
        height: 3;
        margin: 1 0;
    }

    RecoveryChecklistWidget .progress-label {
        text-align: center;
        color: $text;
        margin-bottom: 1;
    }

    RecoveryChecklistWidget .checklist-items {
        margin: 1 0;
        padding: 0 1;
    }

    RecoveryChecklistWidget .checklist-item {
        margin: 0;
        padding: 0;
        height: 2;
    }

    RecoveryChecklistWidget .item-required {
        color: $error;
    }

    RecoveryChecklistWidget .item-optional {
        color: $primary;
    }

    RecoveryChecklistWidget .item-completed {
        color: $success;
        text-decoration: line-through;
    }

    RecoveryChecklistWidget Checkbox {
        margin-right: 1;
    }

    RecoveryChecklistWidget .status-message {
        margin-top: 1;
        text-align: center;
        padding: 1;
        border: solid $success;
    }

    RecoveryChecklistWidget .status-incomplete {
        border: solid $warning;
        color: $warning;
    }

    RecoveryChecklistWidget .status-complete {
        border: solid $success;
        color: $success;
    }

    RecoveryChecklistWidget Button {
        width: 16;
        margin: 1;
    }
    """

    def __init__(
        self,
        checklist_manager: Optional[RecoveryChecklistManager] = None,
        profile_id: str = "default",
        **kwargs,
    ) -> None:
        """Initialize recovery checklist widget.

        Args:
            checklist_manager: Checklist manager instance
            profile_id: Profile identifier
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.checklist_manager = checklist_manager or RecoveryChecklistManager()
        self.profile_id = profile_id
        self.checklist = None
        self.checkbox_map = {}  # Map checkbox IDs to item names

    def on_mount(self) -> None:
        """Initialize checklist when widget is mounted."""
        # Create or get existing checklist
        self.checklist = self.checklist_manager.get_checklist(self.profile_id)
        if not self.checklist:
            self.checklist = self.checklist_manager.create_checklist(self.profile_id)

        # Refresh the display
        self.refresh_checklist()

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Vertical():
            # Title
            yield Static(
                "✅ RECOVERY CHECKLIST",
                classes="title",
            )

            # Subtitle
            yield Static(
                "Complete required items before resuming trading",
                classes="subtitle",
            )

            # Progress bar
            with Container(classes="progress-container"):
                yield Label(
                    "Progress: 0%",
                    id="progress-label",
                    classes="progress-label",
                )
                yield ProgressBar(
                    total=100,
                    show_eta=False,
                    id="progress-bar",
                )

            # Checklist items container
            yield Container(
                id="checklist-items",
                classes="checklist-items",
            )

            # Status message
            yield Static(
                "",
                id="status-message",
                classes="status-message status-incomplete",
            )

            # Buttons
            with Horizontal():
                yield Button(
                    "Refresh",
                    id="refresh-button",
                    variant="default",
                )
                yield Button(
                    "Reset",
                    id="reset-button",
                    variant="warning",
                )

    def refresh_checklist(self) -> None:
        """Refresh the checklist display."""
        if not self.checklist:
            return

        # Get progress
        progress = self.checklist_manager.get_progress(self.profile_id)

        # Update progress label and bar
        progress_label = self.query_one("#progress-label", Label)
        progress_label.update(f"Progress: {progress['progress_percentage']}%")

        progress_bar = self.query_one("#progress-bar", ProgressBar)
        progress_bar.update(progress=progress["progress_percentage"])

        # Clear and rebuild checklist items
        items_container = self.query_one("#checklist-items", Container)
        items_container.remove_children()
        self.checkbox_map.clear()

        # Add items
        for item in self.checklist.items:
            checkbox_id = f"checkbox_{item.item_id}"
            self.checkbox_map[checkbox_id] = item.name

            # Create checkbox with styling
            checkbox = Checkbox(
                f"{item.name}",
                value=item.is_completed,
                id=checkbox_id,
            )

            # Add type indicator
            if item.item_type == ChecklistItemType.REQUIRED:
                checkbox.add_class("item-required")
            else:
                checkbox.add_class("item-optional")

            if item.is_completed:
                checkbox.add_class("item-completed")

            # Add description as tooltip (simplified)
            checkbox.tooltip = item.description

            items_container.mount(checkbox)

        # Update status message
        status_message = self.query_one("#status-message", Static)
        if progress["is_complete"]:
            status_message.update(
                "✅ All required items complete! Ready to resume trading."
            )
            status_message.remove_class("status-incomplete")
            status_message.add_class("status-complete")
        else:
            remaining = progress["required_total"] - progress["required_complete"]
            if remaining > 0:
                status_message.update(f"⚠️ {remaining} required item(s) remaining")
                status_message.remove_class("status-complete")
                status_message.add_class("status-incomplete")
            else:
                status_message.update("✅ All required items complete!")
                status_message.remove_class("status-incomplete")
                status_message.add_class("status-complete")

    @on(Checkbox.Changed)
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox state changes.

        Args:
            event: Checkbox change event
        """
        checkbox_id = event.checkbox.id
        if checkbox_id not in self.checkbox_map:
            return

        item_name = self.checkbox_map[checkbox_id]

        if event.value:
            # Mark item as complete
            success = self.checklist_manager.complete_item(
                self.profile_id,
                item_name,
            )

            if success:
                # Add completed styling
                event.checkbox.add_class("item-completed")

                # Check if checklist is complete
                is_complete = self.checklist_manager.validate_checklist_completion(
                    self.profile_id
                )

                # Post message
                self.post_message(ChecklistItemCompleted(item_name, is_complete))

                # Refresh display
                self.refresh_checklist()
        else:
            # Don't allow unchecking completed items in this implementation
            # Reset the checkbox to checked state
            event.checkbox.value = True

    @on(Button.Pressed, "#refresh-button")
    def on_refresh_pressed(self, event: Button.Pressed) -> None:
        """Handle refresh button press.

        Args:
            event: Button press event
        """
        self.refresh_checklist()

    @on(Button.Pressed, "#reset-button")
    def on_reset_pressed(self, event: Button.Pressed) -> None:
        """Handle reset button press.

        Args:
            event: Button press event
        """
        # Reset the checklist
        if self.checklist_manager.reset_checklist(self.profile_id):
            # Refresh display
            self.refresh_checklist()

    def set_profile(self, profile_id: str) -> None:
        """Set the profile for the checklist.

        Args:
            profile_id: Profile identifier
        """
        self.profile_id = profile_id

        # Get or create checklist for new profile
        self.checklist = self.checklist_manager.get_checklist(profile_id)
        if not self.checklist:
            self.checklist = self.checklist_manager.create_checklist(profile_id)

        # Refresh display
        self.refresh_checklist()

    def get_incomplete_items(self) -> list[str]:
        """Get list of incomplete required items.

        Returns:
            List of incomplete required item names
        """
        return self.checklist_manager.get_incomplete_required_items(self.profile_id)

    def can_resume_trading(self) -> bool:
        """Check if trading can resume based on checklist.

        Returns:
            True if all required items are complete
        """
        return self.checklist_manager.can_resume_trading(self.profile_id)
