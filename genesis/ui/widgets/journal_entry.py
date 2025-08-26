"""Journal entry widget for tilt recovery UI."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Label, ProgressBar, Static, TextArea

from genesis.tilt.journal_system import JournalSystem


class JournalSubmitted(Message):
    """Message sent when journal entry is submitted."""

    def __init__(self, entry_id: str, word_count: int) -> None:
        """Initialize message.

        Args:
            entry_id: ID of submitted entry
            word_count: Word count of entry
        """
        super().__init__()
        self.entry_id = entry_id
        self.word_count = word_count


class JournalEntryWidget(Container):
    """Widget for journal entry during tilt recovery."""

    DEFAULT_CSS = """
    JournalEntryWidget {
        height: auto;
        border: solid $accent;
        padding: 1;
        margin: 1;
    }

    JournalEntryWidget .title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    JournalEntryWidget .prompts {
        color: $text-muted;
        margin-bottom: 1;
        padding: 0 1;
    }

    JournalEntryWidget .prompt-item {
        margin: 0;
        padding-left: 2;
    }

    JournalEntryWidget TextArea {
        height: 15;
        margin: 1 0;
    }

    JournalEntryWidget .word-count {
        color: $text;
        margin-right: 2;
    }

    JournalEntryWidget .word-count.insufficient {
        color: $error;
    }

    JournalEntryWidget .word-count.sufficient {
        color: $success;
    }

    JournalEntryWidget Button {
        width: 12;
        margin: 0 1;
    }

    JournalEntryWidget Button:disabled {
        opacity: 0.5;
    }

    JournalEntryWidget .progress-container {
        height: 1;
        margin: 1 0;
    }

    JournalEntryWidget .validation-message {
        color: $error;
        margin-top: 1;
        text-align: center;
    }

    JournalEntryWidget .success-message {
        color: $success;
        margin-top: 1;
        text-align: center;
    }
    """

    def __init__(
        self,
        journal_system: JournalSystem | None = None,
        profile_id: str = "default",
        min_word_count: int = 100,
        **kwargs,
    ) -> None:
        """Initialize journal entry widget.

        Args:
            journal_system: Journal system instance
            profile_id: Profile identifier
            min_word_count: Minimum required word count
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.journal_system = journal_system or JournalSystem()
        self.profile_id = profile_id
        self.min_word_count = min_word_count
        self.current_word_count = 0
        self.validation_message = ""
        self.is_submitted = False

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Vertical():
            # Title
            yield Static(
                "📝 TILT RECOVERY JOURNAL",
                classes="title",
            )

            # Introspection prompts
            with Container(classes="prompts"):
                yield Static("Reflect on these questions:")
                for i, prompt in enumerate(self.journal_system.get_introspection_prompts(), 1):
                    yield Static(
                        f"{i}. {prompt}",
                        classes="prompt-item",
                    )

            # Main text area
            yield TextArea(
                placeholder=(
                    f"Write your reflection here (minimum {self.min_word_count} words)...\n\n"
                    "Be honest about what triggered this episode and how you plan to prevent it."
                ),
                id="journal-text",
            )

            # Trigger analysis (optional)
            yield Static("What triggered this? (optional):", classes="prompt-label")
            yield TextArea(
                placeholder="Describe the specific trigger...",
                id="trigger-text",
                show_line_numbers=False,
            )

            # Prevention plan (optional)
            yield Static("How will you prevent this? (optional):", classes="prompt-label")
            yield TextArea(
                placeholder="Your prevention strategy...",
                id="prevention-text",
                show_line_numbers=False,
            )

            # Word count progress
            with Container(classes="progress-container"):
                yield ProgressBar(
                    total=self.min_word_count,
                    show_eta=False,
                    id="word-progress",
                )

            # Controls
            with Horizontal():
                yield Label(
                    f"Words: 0 / {self.min_word_count}",
                    id="word-count",
                    classes="word-count insufficient",
                )
                yield Button(
                    "Submit",
                    id="submit-button",
                    disabled=True,
                    variant="primary",
                )
                yield Button(
                    "Clear",
                    id="clear-button",
                    variant="default",
                )

            # Validation/success message
            yield Static(
                "",
                id="message",
                classes="validation-message",
            )

    @on(TextArea.Changed, "#journal-text")
    def on_text_changed(self, event: TextArea.Changed) -> None:
        """Handle text changes in journal area.

        Args:
            event: Text change event
        """
        if self.is_submitted:
            return

        # Count words
        text = event.text_area.text
        self.current_word_count = self.journal_system.count_words(text)

        # Update word count display
        word_count_label = self.query_one("#word-count", Label)
        word_count_label.update(
            f"Words: {self.current_word_count} / {self.min_word_count}"
        )

        # Update word count style
        if self.current_word_count >= self.min_word_count:
            word_count_label.remove_class("insufficient")
            word_count_label.add_class("sufficient")
        else:
            word_count_label.remove_class("sufficient")
            word_count_label.add_class("insufficient")

        # Update progress bar
        progress = self.query_one("#word-progress", ProgressBar)
        progress.update(progress=min(self.current_word_count, self.min_word_count))

        # Enable/disable submit button
        submit_button = self.query_one("#submit-button", Button)
        submit_button.disabled = self.current_word_count < self.min_word_count

        # Clear any validation message
        if self.validation_message:
            self.query_one("#message", Static).update("")
            self.validation_message = ""

    @on(Button.Pressed, "#submit-button")
    async def on_submit_pressed(self, event: Button.Pressed) -> None:
        """Handle submit button press.

        Args:
            event: Button press event
        """
        if self.is_submitted:
            return

        # Get text from all fields
        journal_text = self.query_one("#journal-text", TextArea).text
        trigger_text = self.query_one("#trigger-text", TextArea).text
        prevention_text = self.query_one("#prevention-text", TextArea).text

        # Validate content
        is_valid, message = self.journal_system.validate_entry_content(
            journal_text,
            trigger_text if trigger_text.strip() else None,
            prevention_text if prevention_text.strip() else None,
        )

        if not is_valid:
            # Show validation error
            message_widget = self.query_one("#message", Static)
            message_widget.update(f"❌ {message}")
            message_widget.remove_class("success-message")
            message_widget.add_class("validation-message")
            self.validation_message = message
            return

        # Submit journal entry
        entry = await self.journal_system.submit_journal_entry(
            profile_id=self.profile_id,
            content=journal_text,
            trigger_analysis=trigger_text if trigger_text.strip() else None,
            prevention_plan=prevention_text if prevention_text.strip() else None,
        )

        if entry:
            # Mark as submitted
            self.is_submitted = True

            # Disable text areas and buttons
            self.query_one("#journal-text", TextArea).disabled = True
            self.query_one("#trigger-text", TextArea).disabled = True
            self.query_one("#prevention-text", TextArea).disabled = True
            self.query_one("#submit-button", Button).disabled = True
            self.query_one("#clear-button", Button).disabled = True

            # Show success message
            message_widget = self.query_one("#message", Static)
            remaining = self.journal_system.get_pending_requirements(self.profile_id)
            if remaining > 0:
                message_widget.update(
                    f"✅ Journal submitted! {remaining} more required."
                )
            else:
                message_widget.update(
                    "✅ Journal submitted! All requirements complete."
                )
            message_widget.remove_class("validation-message")
            message_widget.add_class("success-message")

            # Post message to parent
            self.post_message(JournalSubmitted(entry.entry_id, entry.word_count))

    @on(Button.Pressed, "#clear-button")
    def on_clear_pressed(self, event: Button.Pressed) -> None:
        """Handle clear button press.

        Args:
            event: Button press event
        """
        if self.is_submitted:
            return

        # Clear all text areas
        self.query_one("#journal-text", TextArea).clear()
        self.query_one("#trigger-text", TextArea).clear()
        self.query_one("#prevention-text", TextArea).clear()

        # Reset word count
        self.current_word_count = 0
        word_count_label = self.query_one("#word-count", Label)
        word_count_label.update(f"Words: 0 / {self.min_word_count}")
        word_count_label.remove_class("sufficient")
        word_count_label.add_class("insufficient")

        # Reset progress bar
        self.query_one("#word-progress", ProgressBar).update(progress=0)

        # Disable submit button
        self.query_one("#submit-button", Button).disabled = True

        # Clear any messages
        self.query_one("#message", Static).update("")
        self.validation_message = ""

    def reset_for_new_entry(self) -> None:
        """Reset widget for a new journal entry."""
        self.is_submitted = False

        # Re-enable controls
        self.query_one("#journal-text", TextArea).disabled = False
        self.query_one("#trigger-text", TextArea).disabled = False
        self.query_one("#prevention-text", TextArea).disabled = False
        self.query_one("#clear-button", Button).disabled = False

        # Clear content
        self.on_clear_pressed(None)

    def set_profile(self, profile_id: str) -> None:
        """Set the profile for journal submission.

        Args:
            profile_id: Profile identifier
        """
        self.profile_id = profile_id
        if not self.is_submitted:
            self.reset_for_new_entry()
