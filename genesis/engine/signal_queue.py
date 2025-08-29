
"""Signal queue management system for multi-pair trading."""

import asyncio
import heapq
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import structlog

from genesis.core.exceptions import ValidationError
from genesis.core.models import Signal, SignalType
from genesis.data.repository import Repository
from genesis.engine.event_bus import Event, EventBus
from genesis.engine.event_bus import Priority as EventPriority

logger = structlog.get_logger(__name__)


class SignalStatus(Enum):
    """Signal processing status."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CONFLICTED = "CONFLICTED"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""

    HIGHEST_PRIORITY = "HIGHEST_PRIORITY"
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
    FIRST_IN = "FIRST_IN"
    MERGE = "MERGE"
    CANCEL_ALL = "CANCEL_ALL"


@dataclass(order=True)
class QueuedSignal:
    """Signal with queue metadata."""

    priority_score: float = field(compare=True)  # For heap ordering
    signal: Signal = field(compare=False)
    status: SignalStatus = field(default=SignalStatus.PENDING, compare=False)
    queued_at: datetime = field(default_factory=datetime.utcnow, compare=False)
    processed_at: datetime | None = field(default=None, compare=False)
    conflict_group: str | None = field(default=None, compare=False)
    rejection_reason: str | None = field(default=None, compare=False)

    def __post_init__(self):
        """Calculate negative priority for min-heap (higher priority = lower score)."""
        # Combine priority (0-100) and confidence (0-1) for scoring
        # Higher values should be processed first, so negate for min-heap
        self.priority_score = -(
            self.signal.priority + self.signal.confidence_score * 20
        )

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.signal.expiry_time:
            return datetime.utcnow() > self.signal.expiry_time
        # Default expiry of 5 minutes if not specified
        return (datetime.utcnow() - self.queued_at) > timedelta(minutes=5)


class SignalQueue:
    """Manages and prioritizes trading signals across multiple pairs."""

    DEFAULT_EXPIRY_MINUTES = 5
    MAX_QUEUE_SIZE = 1000
    CONFLICT_WINDOW_SECONDS = 10

    def __init__(
        self,
        repository: Repository,
        event_bus: EventBus | None = None,
        conflict_resolution: ConflictResolution = ConflictResolution.HIGHEST_PRIORITY,
    ):
        """Initialize signal queue.

        Args:
            repository: Data repository for persistence
            event_bus: Event bus for signal events
            conflict_resolution: Default conflict resolution strategy
        """
        self.repository = repository
        self.event_bus = event_bus
        self.conflict_resolution = conflict_resolution

        # Priority queue (min-heap)
        self._queue: list[QueuedSignal] = []

        # Track signals by symbol for conflict detection
        self._signals_by_symbol: dict[str, list[QueuedSignal]] = {}

        # Track processed signals
        self._processed_signals: set[str] = set()

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_queued": 0,
            "total_executed": 0,
            "total_rejected": 0,
            "total_expired": 0,
            "total_conflicts": 0,
        }

    async def add_signal(self, signal: Signal, priority: int | None = None) -> None:
        """Add a signal to the queue.

        Args:
            signal: Trading signal to queue
            priority: Override signal priority (optional)
        """
        async with self._lock:
            # Check if already processed
            if signal.signal_id in self._processed_signals:
                logger.warning(
                    "signal_already_processed",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                )
                return

            # Check queue size limit
            if len(self._queue) >= self.MAX_QUEUE_SIZE:
                # Remove expired signals to make room
                await self._clean_expired_signals()

                if len(self._queue) >= self.MAX_QUEUE_SIZE:
                    logger.error(
                        "signal_queue_full",
                        queue_size=len(self._queue),
                        max_size=self.MAX_QUEUE_SIZE,
                    )
                    raise ValidationError(
                        f"Signal queue full: {len(self._queue)} signals"
                    )

            # Override priority if provided
            if priority is not None:
                signal.priority = max(0, min(100, priority))

            # Set default expiry if not specified
            if signal.expiry_time is None:
                signal.expiry_time = datetime.utcnow() + timedelta(
                    minutes=self.DEFAULT_EXPIRY_MINUTES
                )

            # Create queued signal
            queued_signal = QueuedSignal(
                signal=signal, priority_score=0  # Will be calculated in __post_init__
            )

            # Check for conflicts
            conflict_group = await self._check_conflicts(queued_signal)
            if conflict_group:
                queued_signal.conflict_group = conflict_group
                await self._resolve_conflict(queued_signal)
            else:
                # Add to queue
                heapq.heappush(self._queue, queued_signal)

                # Track by symbol
                if signal.symbol not in self._signals_by_symbol:
                    self._signals_by_symbol[signal.symbol] = []
                self._signals_by_symbol[signal.symbol].append(queued_signal)

                self._stats["total_queued"] += 1

                logger.info(
                    "signal_queued",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    priority=signal.priority,
                    confidence=signal.confidence_score,
                    queue_size=len(self._queue),
                )

                # Publish event
                if self.event_bus:
                    await self.event_bus.publish(
                        Event(
                            type="signal.queued",
                            data={
                                "signal_id": signal.signal_id,
                                "symbol": signal.symbol,
                                "priority": signal.priority,
                            },
                        ),
                        priority=EventPriority.HIGH,
                    )

            # Persist to database
            await self._persist_signal(queued_signal)

    async def get_next_signal(self) -> Signal | None:
        """Get the next signal to process.

        Returns:
            Next signal to process, or None if queue is empty
        """
        async with self._lock:
            # Clean expired signals first
            await self._clean_expired_signals()

            while self._queue:
                # Get highest priority signal
                queued_signal = heapq.heappop(self._queue)

                # Check if expired
                if queued_signal.is_expired:
                    await self._mark_expired(queued_signal)
                    continue

                # Mark as processing
                queued_signal.status = SignalStatus.PROCESSING
                queued_signal.processed_at = datetime.utcnow()

                # Remove from symbol tracking
                if queued_signal.signal.symbol in self._signals_by_symbol:
                    self._signals_by_symbol[queued_signal.signal.symbol].remove(
                        queued_signal
                    )
                    if not self._signals_by_symbol[queued_signal.signal.symbol]:
                        del self._signals_by_symbol[queued_signal.signal.symbol]

                # Mark as processed
                self._processed_signals.add(queued_signal.signal.signal_id)

                logger.info(
                    "signal_dequeued",
                    signal_id=queued_signal.signal.signal_id,
                    symbol=queued_signal.signal.symbol,
                    priority=queued_signal.signal.priority,
                    queue_time_seconds=(
                        queued_signal.processed_at - queued_signal.queued_at
                    ).total_seconds(),
                )

                # Update database
                await self._update_signal_status(queued_signal)

                return queued_signal.signal

            return None

    async def get_pending_signals(self, symbol: str | None = None) -> list[Signal]:
        """Get all pending signals.

        Args:
            symbol: Filter by symbol (optional)

        Returns:
            List of pending signals
        """
        async with self._lock:
            if symbol:
                symbol_signals = self._signals_by_symbol.get(symbol, [])
                return [
                    qs.signal
                    for qs in symbol_signals
                    if qs.status == SignalStatus.PENDING
                ]
            else:
                return [
                    qs.signal for qs in self._queue if qs.status == SignalStatus.PENDING
                ]

    async def cancel_signal(self, signal_id: str) -> bool:
        """Cancel a pending signal.

        Args:
            signal_id: ID of signal to cancel

        Returns:
            True if cancelled, False if not found
        """
        async with self._lock:
            # Find signal in queue
            for i, queued_signal in enumerate(self._queue):
                if queued_signal.signal.signal_id == signal_id:
                    # Remove from queue
                    self._queue.pop(i)
                    heapq.heapify(self._queue)  # Restore heap property

                    # Remove from symbol tracking
                    symbol = queued_signal.signal.symbol
                    if symbol in self._signals_by_symbol:
                        self._signals_by_symbol[symbol].remove(queued_signal)
                        if not self._signals_by_symbol[symbol]:
                            del self._signals_by_symbol[symbol]

                    # Mark as rejected
                    queued_signal.status = SignalStatus.REJECTED
                    queued_signal.rejection_reason = "User cancelled"
                    self._stats["total_rejected"] += 1

                    # Update database
                    await self._update_signal_status(queued_signal)

                    logger.info("signal_cancelled", signal_id=signal_id, symbol=symbol)

                    return True

            return False

    async def clear_expired(self) -> int:
        """Clear all expired signals.

        Returns:
            Number of expired signals cleared
        """
        async with self._lock:
            return await self._clean_expired_signals()

    async def get_queue_stats(self) -> dict[str, any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue statistics
        """
        async with self._lock:
            symbol_counts = {}
            for symbol, signals in self._signals_by_symbol.items():
                symbol_counts[symbol] = len(signals)

            return {
                "queue_size": len(self._queue),
                "by_symbol": symbol_counts,
                "total_queued": self._stats["total_queued"],
                "total_executed": self._stats["total_executed"],
                "total_rejected": self._stats["total_rejected"],
                "total_expired": self._stats["total_expired"],
                "total_conflicts": self._stats["total_conflicts"],
                "oldest_signal_age": self._get_oldest_signal_age(),
            }

    # Private methods

    async def _check_conflicts(self, new_signal: QueuedSignal) -> str | None:
        """Check for conflicting signals.

        Args:
            new_signal: Signal to check

        Returns:
            Conflict group ID if conflicts exist, None otherwise
        """
        symbol = new_signal.signal.symbol
        if symbol not in self._signals_by_symbol:
            return None

        existing_signals = self._signals_by_symbol[symbol]
        if not existing_signals:
            return None

        # Check for recent signals on same symbol
        conflict_window = timedelta(seconds=self.CONFLICT_WINDOW_SECONDS)
        now = datetime.utcnow()

        for existing in existing_signals:
            if existing.status != SignalStatus.PENDING:
                continue

            time_diff = abs((new_signal.queued_at - existing.queued_at).total_seconds())
            if time_diff <= self.CONFLICT_WINDOW_SECONDS:
                # Check if signals are conflicting
                if self._are_signals_conflicting(new_signal.signal, existing.signal):
                    # Create or reuse conflict group
                    if existing.conflict_group:
                        return existing.conflict_group
                    else:
                        conflict_group = str(uuid.uuid4())
                        existing.conflict_group = conflict_group
                        self._stats["total_conflicts"] += 1
                        return conflict_group

        return None

    def _are_signals_conflicting(self, signal1: Signal, signal2: Signal) -> bool:
        """Check if two signals are conflicting.

        Args:
            signal1: First signal
            signal2: Second signal

        Returns:
            True if signals conflict
        """
        # Same symbol, opposite directions
        if signal1.symbol == signal2.symbol:
            if (
                signal1.signal_type == SignalType.BUY
                and signal2.signal_type == SignalType.SELL
            ) or (
                signal1.signal_type == SignalType.SELL
                and signal2.signal_type == SignalType.BUY
            ):
                return True

            # Both trying to close
            if (
                signal1.signal_type == SignalType.CLOSE
                and signal2.signal_type == SignalType.CLOSE
            ):
                return True

        return False

    async def _resolve_conflict(self, new_signal: QueuedSignal) -> None:
        """Resolve conflicts between signals.

        Args:
            new_signal: New conflicting signal
        """
        symbol = new_signal.signal.symbol
        conflict_group = new_signal.conflict_group

        # Get all signals in conflict group
        conflicting_signals = [
            qs
            for qs in self._signals_by_symbol.get(symbol, [])
            if qs.conflict_group == conflict_group and qs.status == SignalStatus.PENDING
        ]
        conflicting_signals.append(new_signal)

        logger.warning(
            "signal_conflict_detected",
            symbol=symbol,
            conflict_group=conflict_group,
            num_signals=len(conflicting_signals),
            resolution_strategy=self.conflict_resolution.value,
        )

        if self.conflict_resolution == ConflictResolution.HIGHEST_PRIORITY:
            # Keep highest priority signal
            conflicting_signals.sort(key=lambda qs: qs.priority_score)
            winner = conflicting_signals[0]
            losers = conflicting_signals[1:]

        elif self.conflict_resolution == ConflictResolution.HIGHEST_CONFIDENCE:
            # Keep highest confidence signal
            conflicting_signals.sort(key=lambda qs: -qs.signal.confidence_score)
            winner = conflicting_signals[0]
            losers = conflicting_signals[1:]

        elif self.conflict_resolution == ConflictResolution.FIRST_IN:
            # Keep first signal
            conflicting_signals.sort(key=lambda qs: qs.queued_at)
            winner = conflicting_signals[0]
            losers = conflicting_signals[1:]

        elif self.conflict_resolution == ConflictResolution.CANCEL_ALL:
            # Cancel all conflicting signals
            winner = None
            losers = conflicting_signals

        else:  # MERGE
            # Merge signals (custom logic needed)
            # For now, default to highest priority
            conflicting_signals.sort(key=lambda qs: qs.priority_score)
            winner = conflicting_signals[0]
            losers = conflicting_signals[1:]

        # Process winner
        if winner:
            if winner != new_signal:
                # New signal lost, don't add to queue
                new_signal.status = SignalStatus.CONFLICTED
                new_signal.rejection_reason = (
                    f"Lost conflict to {winner.signal.signal_id}"
                )
                await self._update_signal_status(new_signal)
            else:
                # New signal won, add to queue
                heapq.heappush(self._queue, new_signal)
                if symbol not in self._signals_by_symbol:
                    self._signals_by_symbol[symbol] = []
                self._signals_by_symbol[symbol].append(new_signal)
                self._stats["total_queued"] += 1

        # Reject losers
        for loser in losers:
            if loser in self._queue:
                self._queue.remove(loser)
                heapq.heapify(self._queue)

            if (
                symbol in self._signals_by_symbol
                and loser in self._signals_by_symbol[symbol]
            ):
                self._signals_by_symbol[symbol].remove(loser)

            loser.status = SignalStatus.CONFLICTED
            loser.rejection_reason = (
                f"Conflict resolution: {self.conflict_resolution.value}"
            )
            await self._update_signal_status(loser)

    async def _clean_expired_signals(self) -> int:
        """Remove expired signals from queue.

        Returns:
            Number of expired signals removed
        """
        expired_count = 0
        cleaned_queue = []

        for queued_signal in self._queue:
            if queued_signal.is_expired:
                await self._mark_expired(queued_signal)
                expired_count += 1
            else:
                cleaned_queue.append(queued_signal)

        self._queue = cleaned_queue
        heapq.heapify(self._queue)

        if expired_count > 0:
            logger.info(
                "expired_signals_cleaned",
                count=expired_count,
                remaining=len(self._queue),
            )

        return expired_count

    async def _mark_expired(self, queued_signal: QueuedSignal) -> None:
        """Mark a signal as expired.

        Args:
            queued_signal: Signal to mark as expired
        """
        queued_signal.status = SignalStatus.EXPIRED
        self._stats["total_expired"] += 1

        # Remove from symbol tracking
        symbol = queued_signal.signal.symbol
        if symbol in self._signals_by_symbol:
            if queued_signal in self._signals_by_symbol[symbol]:
                self._signals_by_symbol[symbol].remove(queued_signal)
                if not self._signals_by_symbol[symbol]:
                    del self._signals_by_symbol[symbol]

        # Update database
        await self._update_signal_status(queued_signal)

        logger.debug(
            "signal_expired", signal_id=queued_signal.signal.signal_id, symbol=symbol
        )

    def _get_oldest_signal_age(self) -> float | None:
        """Get age of oldest signal in queue.

        Returns:
            Age in seconds, or None if queue is empty
        """
        if not self._queue:
            return None

        oldest = min(self._queue, key=lambda qs: qs.queued_at)
        return (datetime.utcnow() - oldest.queued_at).total_seconds()

    async def _persist_signal(self, queued_signal: QueuedSignal) -> None:
        """Persist signal to database.

        Args:
            queued_signal: Signal to persist
        """
        try:
            await self.repository.save_queued_signal(
                {
                    "signal_id": queued_signal.signal.signal_id,
                    "symbol": queued_signal.signal.symbol,
                    "signal_type": queued_signal.signal.signal_type.value,
                    "confidence_score": queued_signal.signal.confidence_score,
                    "priority": queued_signal.signal.priority,
                    "status": queued_signal.status.value,
                    "queued_at": queued_signal.queued_at,
                    "conflict_group": queued_signal.conflict_group,
                }
            )
        except Exception as e:
            logger.error(
                "failed_to_persist_signal",
                signal_id=queued_signal.signal.signal_id,
                error=str(e),
            )

    async def _update_signal_status(self, queued_signal: QueuedSignal) -> None:
        """Update signal status in database.

        Args:
            queued_signal: Signal to update
        """
        try:
            await self.repository.update_signal_status(
                signal_id=queued_signal.signal.signal_id,
                status=queued_signal.status.value,
                processed_at=queued_signal.processed_at,
                rejection_reason=queued_signal.rejection_reason,
            )
        except Exception as e:
            logger.error(
                "failed_to_update_signal_status",
                signal_id=queued_signal.signal.signal_id,
                error=str(e),
            )
