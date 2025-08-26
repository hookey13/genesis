"""Old habits funeral ceremony for tier transitions.

Provides a symbolic ritual for consciously letting go of limiting behaviors
and patterns from the previous tier, creating psychological closure and
commitment to new trading behaviors.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from genesis.core.exceptions import ValidationError
from genesis.data.models_db import (
    HabitFuneralRecord,
    Session,
    TierTransition,
    get_session,
)

logger = structlog.get_logger(__name__)


# Bad habits commonly associated with each tier
TIER_BAD_HABITS = {
    'SNIPER': [
        'Overtrading on small wins',
        'Revenge trading after losses',
        'Ignoring stop losses',
        'Trading without a plan',
        'Chasing price movements',
        'Holding losing positions too long',
        'Taking profits too early',
        'Trading on emotions',
        'Ignoring risk management',
        'FOMO trading'
    ],
    'HUNTER': [
        'Over-leveraging positions',
        'Ignoring correlation risks',
        'Trading too many pairs simultaneously',
        'Neglecting execution costs',
        'Overconfidence after winning streaks',
        'Inadequate position sizing',
        'Failing to adapt to market conditions',
        'Ignoring liquidity constraints',
        'Poor order management',
        'Insufficient backtesting'
    ],
    'STRATEGIST': [
        'Overcomplicated strategies',
        'Ignoring black swan events',
        'Excessive automation without monitoring',
        'Neglecting market microstructure',
        'Poor portfolio diversification',
        'Inadequate stress testing',
        'Ignoring regime changes',
        'Overreliance on historical data',
        'Insufficient capital reserves',
        'Neglecting operational risks'
    ]
}


@dataclass
class OldHabit:
    """Represents an old habit to be buried."""
    habit_id: str
    description: str
    impact_description: str
    frequency: str  # DAILY, WEEKLY, OCCASIONAL
    severity: str  # LOW, MEDIUM, HIGH
    commitment_to_change: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'habit_id': self.habit_id,
            'description': self.description,
            'impact_description': self.impact_description,
            'frequency': self.frequency,
            'severity': self.severity,
            'commitment_to_change': self.commitment_to_change
        }


@dataclass
class CeremonyCommitment:
    """Represents a commitment made during the ceremony."""
    commitment_id: str
    commitment_text: str
    accountability_measure: str
    review_date: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'commitment_id': self.commitment_id,
            'commitment_text': self.commitment_text,
            'accountability_measure': self.accountability_measure,
            'review_date': self.review_date.isoformat()
        }


@dataclass
class CeremonyRecord:
    """Record of the funeral ceremony."""
    funeral_id: str
    transition_id: str
    ceremony_timestamp: datetime
    old_habits: list[OldHabit]
    commitments: list[CeremonyCommitment]
    eulogy_text: str
    certificate_hash: str
    certificate_generated: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'funeral_id': self.funeral_id,
            'transition_id': self.transition_id,
            'ceremony_timestamp': self.ceremony_timestamp.isoformat(),
            'old_habits': [h.to_dict() for h in self.old_habits],
            'commitments': [c.to_dict() for c in self.commitments],
            'eulogy_text': self.eulogy_text,
            'certificate_hash': self.certificate_hash,
            'certificate_generated': self.certificate_generated
        }


class HabitFuneralCeremony:
    """Manages the old habits funeral ceremony."""

    def __init__(self, session: Session | None = None):
        """Initialize funeral ceremony manager.
        
        Args:
            session: Optional database session
        """
        self.session = session or get_session()
        self._active_ceremonies: dict[str, CeremonyRecord] = {}

    async def conduct_funeral(
        self,
        transition_id: str,
        old_habits: list[str],
        impact_descriptions: list[str],
        commitments: list[str]
    ) -> CeremonyRecord:
        """Conduct a funeral ceremony for old habits.
        
        Args:
            transition_id: Tier transition ID
            old_habits: List of bad habit descriptions
            impact_descriptions: How each habit impacted trading
            commitments: Commitments to new behaviors
            
        Returns:
            CeremonyRecord with ceremony details
            
        Raises:
            ValidationError: If inputs invalid
        """
        # Validate inputs
        if len(old_habits) < 3:
            raise ValidationError("Must identify at least 3 bad habits to bury")

        if len(old_habits) > 10:
            raise ValidationError("Focus on maximum 10 most impactful habits")

        if len(impact_descriptions) != len(old_habits):
            raise ValidationError("Must provide impact description for each habit")

        if len(commitments) < len(old_habits):
            raise ValidationError("Must commit to change for each old habit")

        # Create OldHabit objects
        habit_objects = []
        for i, habit_desc in enumerate(old_habits):
            habit = OldHabit(
                habit_id=str(uuid.uuid4()),
                description=habit_desc,
                impact_description=impact_descriptions[i],
                frequency=self._assess_frequency(habit_desc),
                severity=self._assess_severity(impact_descriptions[i]),
                commitment_to_change=commitments[i] if i < len(commitments) else ""
            )
            habit_objects.append(habit)

        # Create commitments
        commitment_objects = []
        for i, commitment_text in enumerate(commitments):
            commitment = CeremonyCommitment(
                commitment_id=str(uuid.uuid4()),
                commitment_text=commitment_text,
                accountability_measure="Review in weekly journal entry",
                review_date=datetime.utcnow().replace(day=datetime.utcnow().day + 30)
            )
            commitment_objects.append(commitment)

        # Generate eulogy
        eulogy = await self._generate_eulogy(habit_objects)

        # Create ceremony record
        funeral_id = str(uuid.uuid4())
        ceremony_timestamp = datetime.utcnow()

        # Generate certificate hash
        certificate_data = {
            'funeral_id': funeral_id,
            'transition_id': transition_id,
            'timestamp': ceremony_timestamp.isoformat(),
            'habits': [h.description for h in habit_objects],
            'commitments': [c.commitment_text for c in commitment_objects]
        }
        certificate_hash = hashlib.sha256(
            json.dumps(certificate_data, sort_keys=True).encode()
        ).hexdigest()

        ceremony = CeremonyRecord(
            funeral_id=funeral_id,
            transition_id=transition_id,
            ceremony_timestamp=ceremony_timestamp,
            old_habits=habit_objects,
            commitments=commitment_objects,
            eulogy_text=eulogy,
            certificate_hash=certificate_hash,
            certificate_generated=False
        )

        # Store in database
        await self._store_ceremony(ceremony)

        # Store in memory
        self._active_ceremonies[transition_id] = ceremony

        # Generate certificate
        await self._generate_certificate(ceremony)

        logger.info(
            "Habit funeral ceremony conducted",
            funeral_id=funeral_id,
            transition_id=transition_id,
            habits_buried=len(old_habits),
            commitments_made=len(commitments)
        )

        return ceremony

    async def get_suggested_habits(
        self,
        current_tier: str
    ) -> list[str]:
        """Get suggested bad habits for a tier.
        
        Args:
            current_tier: Current tier name
            
        Returns:
            List of common bad habits for that tier
        """
        return TIER_BAD_HABITS.get(current_tier, [])

    async def validate_ceremony_completion(
        self,
        transition_id: str
    ) -> tuple[bool, str | None]:
        """Validate that funeral ceremony is complete.
        
        Args:
            transition_id: Transition to check
            
        Returns:
            Tuple of (is_complete, certificate_hash)
        """
        # Check in memory first
        if transition_id in self._active_ceremonies:
            ceremony = self._active_ceremonies[transition_id]
            return ceremony.certificate_generated, ceremony.certificate_hash

        # Check database
        record = self.session.query(HabitFuneralRecord).filter_by(
            transition_id=transition_id
        ).first()

        if record and record.certificate_generated:
            return True, record.certificate_hash

        return False, None

    async def get_ceremony_record(
        self,
        transition_id: str
    ) -> CeremonyRecord | None:
        """Get ceremony record for a transition.
        
        Args:
            transition_id: Transition ID
            
        Returns:
            CeremonyRecord or None if not found
        """
        # Check memory first
        if transition_id in self._active_ceremonies:
            return self._active_ceremonies[transition_id]

        # Load from database
        record = self.session.query(HabitFuneralRecord).filter_by(
            transition_id=transition_id
        ).first()

        if record:
            # Convert from database format
            old_habits = [
                OldHabit(**habit) for habit in json.loads(record.old_habits)
            ]
            commitments = [
                CeremonyCommitment(
                    commitment_id=c['commitment_id'],
                    commitment_text=c['commitment_text'],
                    accountability_measure=c['accountability_measure'],
                    review_date=datetime.fromisoformat(c['review_date'])
                )
                for c in json.loads(record.commitments)
            ]

            ceremony = CeremonyRecord(
                funeral_id=record.funeral_id,
                transition_id=record.transition_id,
                ceremony_timestamp=record.ceremony_timestamp,
                old_habits=old_habits,
                commitments=commitments,
                eulogy_text="",  # Not stored in simplified version
                certificate_hash=record.certificate_hash or "",
                certificate_generated=record.certificate_generated
            )

            return ceremony

        return None

    def _assess_frequency(self, habit_description: str) -> str:
        """Assess frequency of a habit based on description.
        
        Args:
            habit_description: Habit text
            
        Returns:
            Frequency classification
        """
        daily_keywords = ['always', 'every', 'constantly', 'daily']
        weekly_keywords = ['often', 'frequently', 'regularly', 'sometimes']

        habit_lower = habit_description.lower()

        if any(keyword in habit_lower for keyword in daily_keywords):
            return 'DAILY'
        elif any(keyword in habit_lower for keyword in weekly_keywords):
            return 'WEEKLY'
        else:
            return 'OCCASIONAL'

    def _assess_severity(self, impact_description: str) -> str:
        """Assess severity based on impact description.
        
        Args:
            impact_description: Impact text
            
        Returns:
            Severity classification
        """
        high_keywords = ['destroyed', 'massive', 'catastrophic', 'severe', 'major']
        medium_keywords = ['significant', 'notable', 'moderate', 'impacted']

        impact_lower = impact_description.lower()

        if any(keyword in impact_lower for keyword in high_keywords):
            return 'HIGH'
        elif any(keyword in impact_lower for keyword in medium_keywords):
            return 'MEDIUM'
        else:
            return 'LOW'

    async def _generate_eulogy(self, habits: list[OldHabit]) -> str:
        """Generate eulogy text for the habits.
        
        Args:
            habits: List of habits being buried
            
        Returns:
            Eulogy text
        """
        high_severity = [h for h in habits if h.severity == 'HIGH']

        eulogy = f"""
Today we lay to rest {len(habits)} trading behaviors that no longer serve our journey.

These habits, born from inexperience and emotion, taught us valuable lessons through their consequences.
We acknowledge their role in our growth, but recognize they have no place in our future.

{"Most notably, we release " + high_severity[0].description.lower() if high_severity else "We release these patterns"} 
and all associated emotional attachments.

We commit to conscious, disciplined trading practices moving forward.
May these old ways rest in peace, never to return.

The past is buried. The future is earned through discipline.
        """.strip()

        return eulogy

    async def _generate_certificate(self, ceremony: CeremonyRecord) -> None:
        """Generate ceremony certificate.
        
        Args:
            ceremony: Ceremony record
        """
        # In a real implementation, this would generate a PDF or image
        # For now, we'll just mark it as generated

        certificate_text = f"""
================================================================================
                        CERTIFICATE OF BEHAVIORAL TRANSFORMATION
================================================================================

This certifies that on {ceremony.ceremony_timestamp.strftime('%B %d, %Y')}

A funeral ceremony was conducted for {len(ceremony.old_habits)} limiting trading behaviors.

HABITS LAID TO REST:
{chr(10).join(f'  - {h.description}' for h in ceremony.old_habits[:5])}

COMMITMENTS MADE:
{chr(10).join(f'  - {c.commitment_text}' for c in ceremony.commitments[:3])}

This transformation is permanent and irrevocable.

Certificate Hash: {ceremony.certificate_hash[:16]}...
Ceremony ID: {ceremony.funeral_id}

================================================================================
                     "The past is buried. The future is earned."
================================================================================
        """

        ceremony.certificate_generated = True

        # Update database
        record = self.session.query(HabitFuneralRecord).filter_by(
            transition_id=ceremony.transition_id
        ).first()

        if record:
            record.certificate_generated = True
            self.session.commit()

        logger.info(
            "Ceremony certificate generated",
            funeral_id=ceremony.funeral_id,
            certificate_hash=ceremony.certificate_hash[:8]
        )

    async def _store_ceremony(self, ceremony: CeremonyRecord) -> None:
        """Store ceremony record in database.
        
        Args:
            ceremony: Ceremony to store
        """
        try:
            # Convert to database format
            db_record = HabitFuneralRecord(
                funeral_id=ceremony.funeral_id,
                transition_id=ceremony.transition_id,
                old_habits=json.dumps([h.to_dict() for h in ceremony.old_habits]),
                commitments=json.dumps([c.to_dict() for c in ceremony.commitments]),
                ceremony_timestamp=ceremony.ceremony_timestamp,
                certificate_hash=ceremony.certificate_hash,
                certificate_generated=ceremony.certificate_generated,
                created_at=datetime.utcnow()
            )

            self.session.add(db_record)

            # Update transition record
            transition = self.session.query(TierTransition).filter_by(
                transition_id=ceremony.transition_id
            ).first()

            if transition:
                transition.funeral_completed = True
                transition.updated_at = datetime.utcnow()

            self.session.commit()

        except Exception as e:
            logger.error(
                "Failed to store ceremony",
                funeral_id=ceremony.funeral_id,
                error=str(e)
            )
            self.session.rollback()
            raise

    async def review_commitments(
        self,
        transition_id: str,
        days_since: int = 30
    ) -> dict[str, Any]:
        """Review commitments made during ceremony.
        
        Args:
            transition_id: Transition to review
            days_since: Days since ceremony
            
        Returns:
            Review summary
        """
        ceremony = await self.get_ceremony_record(transition_id)

        if not ceremony:
            return {'error': 'No ceremony found'}

        days_elapsed = (datetime.utcnow() - ceremony.ceremony_timestamp).days

        # Check which commitments are due for review
        due_for_review = [
            c for c in ceremony.commitments
            if (datetime.utcnow() - c.review_date).days >= 0
        ]

        return {
            'funeral_id': ceremony.funeral_id,
            'days_since_ceremony': days_elapsed,
            'total_commitments': len(ceremony.commitments),
            'commitments_due_for_review': len(due_for_review),
            'habits_buried': len(ceremony.old_habits),
            'certificate_hash': ceremony.certificate_hash[:16]
        }
