from typing import Optional

"""Psychological support messaging system for recovery."""

import random
from decimal import Decimal
from pathlib import Path

import structlog
import yaml

from genesis.tilt.recovery_protocols import RecoveryStage

logger = structlog.get_logger(__name__)


class RecoverySupportMessenger:
    """Provides psychological support messages during recovery."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize recovery support messenger.

        Args:
            config_path: Path to recovery messages config file
        """
        if config_path is None:
            config_path = Path("config/recovery_messages.yaml")

        self.config_path = config_path
        self.messages = self._load_messages()

    def _load_messages(self) -> dict:
        """Load messages from configuration file.

        Returns:
            Dictionary of recovery messages
        """
        try:
            with open(self.config_path) as f:
                messages = yaml.safe_load(f)
                logger.info("Recovery messages loaded", path=str(self.config_path))
                return messages
        except Exception as e:
            logger.error(
                "Failed to load recovery messages",
                path=str(self.config_path),
                error=str(e),
            )
            return self._get_default_messages()

    def _get_default_messages(self) -> dict:
        """Get default messages if config file fails to load.

        Returns:
            Default recovery messages
        """
        return {
            "recovery_stage_messages": {
                "stage_0": {
                    "entry": [
                        "Recovery mode activated. Starting with 25% position size."
                    ],
                    "milestone": ["Good progress. Keep going."],
                    "tips": ["Focus on process, not P&L."],
                },
                "stage_1": {
                    "entry": ["Position size increased to 50%. Stay disciplined."],
                    "milestone": ["Halfway recovered!"],
                    "tips": ["Consistency is key."],
                },
                "stage_2": {
                    "entry": ["75% position size restored. Almost there!"],
                    "milestone": ["75% recovered!"],
                    "tips": ["Maintain your discipline."],
                },
                "stage_3": {
                    "completion": ["Full recovery complete! Well done."],
                    "tips": ["Remember these lessons."],
                },
            },
            "drawdown_messages": {
                "detection": ["Drawdown detected. Recovery protocol initiated."],
                "encouragement": {
                    "light": ["Small setback. You've got this."],
                    "moderate": ["Stay focused. One trade at a time."],
                    "severe": ["Focus on quality over quantity."],
                },
            },
        }

    def get_recovery_message(self, recovery_stage: RecoveryStage, context: dict) -> str:
        """Get appropriate recovery message based on stage and context.

        Args:
            recovery_stage: Current recovery stage
            context: Additional context (e.g., milestone reached)

        Returns:
            Supportive message string
        """
        stage_key = f"stage_{recovery_stage.value}"
        stage_messages = self.messages.get("recovery_stage_messages", {}).get(
            stage_key, {}
        )

        # Check context for message type
        if context.get("is_milestone"):
            message_list = stage_messages.get("milestone", [])
        elif context.get("is_entry"):
            message_list = stage_messages.get("entry", [])
        elif context.get("is_completion"):
            message_list = stage_messages.get("completion", [])
        else:
            message_list = stage_messages.get("tips", [])

        if message_list:
            return random.choice(message_list)

        return "Keep following your recovery plan."

    def get_drawdown_message(self, drawdown_pct: Decimal) -> str:
        """Get drawdown detection message.

        Args:
            drawdown_pct: Drawdown percentage

        Returns:
            Drawdown message
        """
        messages = self.messages.get("drawdown_messages", {}).get("detection", [])

        if messages:
            message = random.choice(messages)
            return message.format(drawdown_pct=float(drawdown_pct * 100))

        return f"Drawdown of {float(drawdown_pct * 100):.1f}% detected."

    def get_encouragement_message(self, drawdown_pct: Decimal) -> str:
        """Get encouragement message based on drawdown severity.

        Args:
            drawdown_pct: Drawdown percentage

        Returns:
            Encouragement message
        """
        encouragement = self.messages.get("drawdown_messages", {}).get(
            "encouragement", {}
        )

        # Determine severity
        if drawdown_pct < Decimal("0.15"):
            severity = "light"
        elif drawdown_pct < Decimal("0.20"):
            severity = "moderate"
        else:
            severity = "severe"

        messages = encouragement.get(severity, [])

        if messages:
            return random.choice(messages)

        return "Stay focused and trust your process."

    def get_consecutive_loss_message(
        self, loss_count: int, is_break: bool = False, duration_minutes: int = 30
    ) -> str:
        """Get message for consecutive losses.

        Args:
            loss_count: Number of consecutive losses
            is_break: Whether a break is being enforced
            duration_minutes: Break duration if applicable

        Returns:
            Consecutive loss message
        """
        if is_break:
            messages = self.messages.get("consecutive_loss_messages", {}).get(
                "break_required", []
            )
            if messages:
                message = random.choice(messages)
                return message.format(duration=duration_minutes)
            return f"Trading break required for {duration_minutes} minutes."

        if loss_count == 2:
            messages = (
                self.messages.get("consecutive_loss_messages", {})
                .get("warning", {})
                .get("2_losses", [])
            )
        elif loss_count >= 3:
            messages = (
                self.messages.get("consecutive_loss_messages", {})
                .get("warning", {})
                .get("3_losses", [])
            )
        else:
            messages = []

        if messages:
            return random.choice(messages)

        return f"{loss_count} consecutive losses detected."

    def get_milestone_message(self, milestone_pct: int) -> str:
        """Get celebration message for recovery milestone.

        Args:
            milestone_pct: Milestone percentage (25, 50, 75, 100)

        Returns:
            Milestone celebration message
        """
        milestones = self.messages.get("milestone_celebrations", {})

        milestone_key = f"{milestone_pct}_percent"
        messages = milestones.get(milestone_key, [])

        if messages:
            return random.choice(messages)

        return f"{milestone_pct}% recovery milestone reached!"

    def get_educational_tip(self, category: str = None) -> str:
        """Get educational tip for avoiding revenge trading.

        Args:
            category: Specific category of tips

        Returns:
            Educational tip
        """
        tips = self.messages.get("educational_tips", {})

        if category and category in tips:
            tip_list = tips[category]
        else:
            # Get random category
            all_tips = []
            for tip_list in tips.values():
                all_tips.extend(tip_list)
            tip_list = all_tips

        if tip_list:
            return random.choice(tip_list)

        return "Focus on process, not outcomes."

    def get_journal_prompt(self, prompt_type: str = "after_losses") -> str:
        """Get journal prompt for recovery reflection.

        Args:
            prompt_type: Type of journal prompt

        Returns:
            Journal prompt question
        """
        prompts = self.messages.get("journal_prompts", {})
        prompt_list = prompts.get(prompt_type, [])

        if prompt_list:
            return random.choice(prompt_list)

        return "What did you learn from recent trades?"

    def format_recovery_status_message(
        self,
        recovery_stage: RecoveryStage,
        drawdown_pct: Decimal,
        recovery_pct: Decimal,
        consecutive_losses: int,
    ) -> list[str]:
        """Format comprehensive recovery status message.

        Args:
            recovery_stage: Current recovery stage
            drawdown_pct: Original drawdown percentage
            recovery_pct: Percentage recovered so far
            consecutive_losses: Current consecutive loss count

        Returns:
            List of status message lines
        """
        messages = []

        # Status line
        stage_pct = {
            RecoveryStage.STAGE_0: 25,
            RecoveryStage.STAGE_1: 50,
            RecoveryStage.STAGE_2: 75,
            RecoveryStage.STAGE_3: 100,
        }

        messages.append(
            f"Recovery Status: Stage {recovery_stage.value} "
            f"({stage_pct[recovery_stage]}% position size)"
        )

        # Progress line
        messages.append(
            f"Progress: {float(recovery_pct * 100):.1f}% recovered "
            f"from {float(drawdown_pct * 100):.1f}% drawdown"
        )

        # Consecutive losses warning
        if consecutive_losses > 0:
            loss_msg = self.get_consecutive_loss_message(consecutive_losses)
            messages.append(f"Warning: {loss_msg}")

        # Educational tip
        if recovery_pct < Decimal("0.5"):
            tip = self.get_educational_tip("position_sizing")
        elif recovery_pct < Decimal("0.75"):
            tip = self.get_educational_tip("process_focus")
        else:
            tip = self.get_educational_tip("mental_state")

        messages.append(f"Tip: {tip}")

        return messages
