"""Psychological preparation checklist for tier transitions.

Forces traders to consciously acknowledge and prepare for the behavioral
changes required at the next tier level through structured reflection.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from genesis.core.exceptions import ValidationError
from genesis.data.models_db import (
    Session,
    TierTransition,
    TransitionChecklist,
    get_session,
)

logger = structlog.get_logger(__name__)


class ChecklistCategory(Enum):
    """Categories for checklist items."""

    RISK_MANAGEMENT = "risk_management"
    STRATEGY_UNDERSTANDING = "strategy_understanding"
    EMOTIONAL_PREPARATION = "emotional_preparation"
    GOAL_SETTING = "goal_setting"
    REFLECTION = "reflection"
    CAPITAL_MANAGEMENT = "capital_management"


class ChecklistStatus(Enum):
    """Status of a checklist item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


# Checklist items by target tier
TIER_CHECKLIST_ITEMS = {
    "HUNTER": [
        {
            "name": "risk_acknowledgment",
            "description": "I understand that Hunter tier strategies involve more complex execution methods and higher capital at risk.",
            "prompt": "Describe your understanding of the increased risks at Hunter tier and how you will manage them:",
            "min_response_length": 100,
            "is_required": True,
        },
        {
            "name": "strategy_understanding",
            "description": "I have studied and understand iceberg orders and order slicing techniques.",
            "prompt": "Explain in your own words how iceberg orders work and when to use them:",
            "min_response_length": 150,
            "is_required": True,
        },
        {
            "name": "emotional_preparation",
            "description": "I am emotionally prepared for larger position sizes and potential drawdowns.",
            "prompt": "Describe your emotional preparation plan for handling larger trades:",
            "min_response_length": 100,
            "is_required": True,
        },
        {
            "name": "journal_reflection",
            "description": "I have reviewed my trading journal and identified key lessons from Sniper tier.",
            "prompt": "What are the three most important lessons you learned at Sniper tier?",
            "min_response_length": 150,
            "is_required": True,
        },
        {
            "name": "goal_setting",
            "description": "I have set clear, measurable goals for Hunter tier.",
            "prompt": "List your specific goals for the first 30 days at Hunter tier:",
            "min_response_length": 100,
            "is_required": True,
        },
    ],
    "STRATEGIST": [
        {
            "name": "risk_acknowledgment",
            "description": "I understand that Strategist tier involves market making and statistical arbitrage with significant capital exposure.",
            "prompt": "Describe your risk management framework for Strategist tier strategies:",
            "min_response_length": 200,
            "is_required": True,
        },
        {
            "name": "strategy_mastery",
            "description": "I have mastered Hunter tier strategies and am ready for advanced techniques.",
            "prompt": "Provide examples of successful Hunter tier trades and what you learned:",
            "min_response_length": 200,
            "is_required": True,
        },
        {
            "name": "psychological_stability",
            "description": "I have demonstrated consistent emotional control and tilt resistance.",
            "prompt": "Describe your tilt management system and recent successes:",
            "min_response_length": 150,
            "is_required": True,
        },
        {
            "name": "capital_management",
            "description": "I understand position sizing and portfolio management at scale.",
            "prompt": "Explain your approach to managing multiple concurrent positions:",
            "min_response_length": 150,
            "is_required": True,
        },
        {
            "name": "continuous_improvement",
            "description": "I commit to continuous learning and strategy refinement.",
            "prompt": "Describe your learning plan for Strategist tier:",
            "min_response_length": 100,
            "is_required": True,
        },
        {
            "name": "emergency_protocols",
            "description": "I have defined emergency protocols for system failures or black swan events.",
            "prompt": "Detail your emergency response procedures:",
            "min_response_length": 150,
            "is_required": True,
        },
    ],
}


@dataclass
class ChecklistItem:
    """Represents a single checklist item."""

    item_id: str
    name: str
    description: str
    prompt: str
    min_response_length: int
    is_required: bool
    response: str | None = None
    completed_at: datetime | None = None
    category: ChecklistCategory | None = None
    status: ChecklistStatus = ChecklistStatus.PENDING

    @property
    def is_complete(self) -> bool:
        """Check if item is complete."""
        if not self.is_required:
            return True
        return (
            self.response is not None
            and len(self.response) >= self.min_response_length
            and self.completed_at is not None
            and self.status == ChecklistStatus.COMPLETED
        )

    def validate_response(self, response: str) -> tuple[bool, str | None]:
        """Validate a response for this item.

        Args:
            response: Response text to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not response or not response.strip():
            return False, "Response cannot be empty"

        if len(response) < self.min_response_length:
            return (
                False,
                f"Response must be at least {self.min_response_length} characters (current: {len(response)})",
            )

        # Check for low-effort responses
        words = response.split()
        unique_words = set(words)
        if len(unique_words) < 10:
            return (
                False,
                "Response appears to be low-effort. Please provide a thoughtful answer.",
            )

        return True, None


@dataclass
class ChecklistProgress:
    """Tracks checklist completion progress."""

    transition_id: str
    total_items: int
    completed_items: int
    required_items: int
    required_completed: int
    completion_percentage: float
    is_complete: bool
    pending_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transition_id": self.transition_id,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "required_items": self.required_items,
            "required_completed": self.required_completed,
            "completion_percentage": self.completion_percentage,
            "is_complete": self.is_complete,
            "pending_items": self.pending_items,
        }


class TransitionChecklist:
    """Manages psychological preparation checklists for tier transitions."""

    def __init__(self, session: Session | None = None):
        """Initialize transition checklist.

        Args:
            session: Optional database session
        """
        self.session = session or get_session()
        self._active_checklists: dict[str, list[ChecklistItem]] = {}

    async def create_checklist(
        self, transition_id: str, target_tier: str
    ) -> list[ChecklistItem]:
        """Create a checklist for tier transition.

        Args:
            transition_id: Tier transition ID
            target_tier: Target tier name

        Returns:
            List of checklist items

        Raises:
            ValidationError: If target tier unknown
        """
        if target_tier not in TIER_CHECKLIST_ITEMS:
            raise ValidationError(f"No checklist defined for tier: {target_tier}")

        # Get checklist template
        template_items = TIER_CHECKLIST_ITEMS[target_tier]

        # Create checklist items
        checklist_items = []

        for item_template in template_items:
            item_id = str(uuid.uuid4())

            # Create database record
            db_item = TransitionChecklist(
                checklist_id=item_id,
                transition_id=transition_id,
                item_name=item_template["name"],
                item_description=item_template["description"],
                is_required=item_template["is_required"],
                created_at=datetime.utcnow(),
            )

            try:
                self.session.add(db_item)

                # Create in-memory item
                item = ChecklistItem(
                    item_id=item_id,
                    name=item_template["name"],
                    description=item_template["description"],
                    prompt=item_template["prompt"],
                    min_response_length=item_template["min_response_length"],
                    is_required=item_template["is_required"],
                )
                checklist_items.append(item)

            except Exception as e:
                logger.error(
                    "Failed to create checklist item",
                    transition_id=transition_id,
                    item_name=item_template["name"],
                    error=str(e),
                )
                self.session.rollback()
                raise

        self.session.commit()

        # Store in memory
        self._active_checklists[transition_id] = checklist_items

        logger.info(
            "Transition checklist created",
            transition_id=transition_id,
            target_tier=target_tier,
            item_count=len(checklist_items),
        )

        return checklist_items

    async def complete_item(
        self, transition_id: str, item_name: str, response: str
    ) -> ChecklistItem:
        """Complete a checklist item with response.

        Args:
            transition_id: Transition ID
            item_name: Item name to complete
            response: Response text

        Returns:
            Updated checklist item

        Raises:
            ValidationError: If item not found or response invalid
        """
        # Get checklist
        if transition_id not in self._active_checklists:
            # Try to load from database
            await self._load_checklist(transition_id)

        checklist = self._active_checklists.get(transition_id, [])
        if not checklist:
            raise ValidationError(f"No checklist found for transition: {transition_id}")

        # Find item
        item = None
        for check_item in checklist:
            if check_item.name == item_name:
                item = check_item
                break

        if not item:
            raise ValidationError(f"Checklist item not found: {item_name}")

        # Validate response
        is_valid, error_msg = item.validate_response(response)
        if not is_valid:
            raise ValidationError(f"Invalid response: {error_msg}")

        # Update item
        item.response = response
        item.completed_at = datetime.utcnow()
        item.status = ChecklistStatus.COMPLETED

        # Update database
        db_item = (
            self.session.query(TransitionChecklist)
            .filter_by(transition_id=transition_id, item_name=item_name)
            .first()
        )

        if db_item:
            db_item.item_response = response
            db_item.completed_at = item.completed_at
            self.session.commit()

            logger.info(
                "Checklist item completed",
                transition_id=transition_id,
                item_name=item_name,
                response_length=len(response),
            )

        # Check if entire checklist is complete
        await self._check_checklist_completion(transition_id)

        return item

    async def get_progress(self, transition_id: str) -> ChecklistProgress:
        """Get checklist completion progress.

        Args:
            transition_id: Transition to check

        Returns:
            ChecklistProgress with current status
        """
        # Get or load checklist
        if transition_id not in self._active_checklists:
            await self._load_checklist(transition_id)

        checklist = self._active_checklists.get(transition_id, [])

        if not checklist:
            # No checklist exists
            return ChecklistProgress(
                transition_id=transition_id,
                total_items=0,
                completed_items=0,
                required_items=0,
                required_completed=0,
                completion_percentage=0.0,
                is_complete=True,  # No checklist = nothing to complete
                pending_items=[],
            )

        # Calculate progress
        total_items = len(checklist)
        completed_items = sum(1 for item in checklist if item.is_complete)
        required_items = sum(1 for item in checklist if item.is_required)
        required_completed = sum(
            1 for item in checklist if item.is_required and item.is_complete
        )

        # Pending items
        pending_items = [
            item.name for item in checklist if item.is_required and not item.is_complete
        ]

        # Completion percentage (based on required items)
        completion_percentage = (
            (required_completed / required_items * 100) if required_items > 0 else 100
        )

        # Is complete when all required items done
        is_complete = required_completed == required_items

        return ChecklistProgress(
            transition_id=transition_id,
            total_items=total_items,
            completed_items=completed_items,
            required_items=required_items,
            required_completed=required_completed,
            completion_percentage=completion_percentage,
            is_complete=is_complete,
            pending_items=pending_items,
        )

    async def get_checklist_items(self, transition_id: str) -> list[ChecklistItem]:
        """Get all checklist items for a transition.

        Args:
            transition_id: Transition ID

        Returns:
            List of checklist items
        """
        if transition_id not in self._active_checklists:
            await self._load_checklist(transition_id)

        return self._active_checklists.get(transition_id, [])

    async def validate_completion(self, transition_id: str) -> tuple[bool, list[str]]:
        """Validate that checklist is complete for transition.

        Args:
            transition_id: Transition to validate

        Returns:
            Tuple of (is_complete, missing_items)
        """
        progress = await self.get_progress(transition_id)

        if progress.is_complete:
            return True, []

        return False, progress.pending_items

    async def _load_checklist(self, transition_id: str) -> None:
        """Load checklist from database.

        Args:
            transition_id: Transition ID to load
        """
        # Get checklist items from database
        db_items = (
            self.session.query(TransitionChecklist)
            .filter_by(transition_id=transition_id)
            .all()
        )

        if not db_items:
            return

        # Get target tier from transition
        transition = (
            self.session.query(TierTransition)
            .filter_by(transition_id=transition_id)
            .first()
        )

        if not transition:
            return

        # Get template for prompts
        template_items = TIER_CHECKLIST_ITEMS.get(transition.to_tier, [])
        template_map = {item["name"]: item for item in template_items}

        # Convert to ChecklistItems
        checklist_items = []
        for db_item in db_items:
            template = template_map.get(db_item.item_name, {})

            item = ChecklistItem(
                item_id=db_item.checklist_id,
                name=db_item.item_name,
                description=db_item.item_description or template.get("description", ""),
                prompt=template.get("prompt", ""),
                min_response_length=template.get("min_response_length", 100),
                is_required=db_item.is_required,
                response=db_item.item_response,
                completed_at=db_item.completed_at,
            )
            checklist_items.append(item)

        self._active_checklists[transition_id] = checklist_items

    async def _check_checklist_completion(self, transition_id: str) -> None:
        """Check if checklist is complete and update transition.

        Args:
            transition_id: Transition to check
        """
        progress = await self.get_progress(transition_id)

        if progress.is_complete:
            # Update transition record
            transition = (
                self.session.query(TierTransition)
                .filter_by(transition_id=transition_id)
                .first()
            )

            if transition and not transition.checklist_completed:
                transition.checklist_completed = True
                transition.updated_at = datetime.utcnow()
                self.session.commit()

                logger.info(
                    "Transition checklist completed", transition_id=transition_id
                )

    def get_item_prompt(self, target_tier: str, item_name: str) -> str | None:
        """Get the prompt for a specific checklist item.

        Args:
            target_tier: Target tier
            item_name: Item name

        Returns:
            Prompt text or None if not found
        """
        items = TIER_CHECKLIST_ITEMS.get(target_tier, [])
        for item in items:
            if item["name"] == item_name:
                return item["prompt"]
        return None

    async def clear_checklist(self, transition_id: str) -> None:
        """Clear a checklist from memory.

        Args:
            transition_id: Transition ID to clear
        """
        if transition_id in self._active_checklists:
            del self._active_checklists[transition_id]
