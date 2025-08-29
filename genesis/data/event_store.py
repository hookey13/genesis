"""
Persistent event store with replay capability and compaction.

Provides durable event storage with archival and replay features.
"""

import asyncio
import gzip
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from genesis.core.events import Event, EventType

logger = structlog.get_logger(__name__)

Base = declarative_base()


class StoredEvent(Base):
    """SQLAlchemy model for persistent event storage."""

    __tablename__ = "event_store"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(36), unique=True, nullable=False)
    event_type = Column(String(50), nullable=False, index=True)
    aggregate_id = Column(String(36), index=True)
    correlation_id = Column(String(36), index=True)
    sequence_number = Column(Integer, nullable=False)
    event_data = Column(Text, nullable=False)  # JSON serialized
    metadata = Column(Text)  # JSON serialized
    created_at = Column(DateTime, nullable=False, index=True)
    archived = Column(Boolean, default=False)
    compressed = Column(Boolean, default=False)

    __table_args__ = (
        Index("idx_event_lookup", "event_type", "created_at"),
        Index("idx_aggregate_events", "aggregate_id", "sequence_number"),
        Index("idx_correlation", "correlation_id", "created_at"),
    )


class EventArchive(Base):
    """SQLAlchemy model for archived events."""

    __tablename__ = "event_archive"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(36), nullable=False, index=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    event_count = Column(Integer, nullable=False)
    compressed_data = Column(Text, nullable=False)  # Gzipped JSON
    checksum = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_archive_dates", "start_date", "end_date"),
    )


class PersistentEventStore:
    """
    Persistent event store with replay and archival capabilities.
    
    Features:
    - Durable event storage in SQLite/PostgreSQL
    - Event replay from any point in time
    - Automatic archival of old events
    - Event compaction to save space
    - Query capabilities for forensic analysis
    """

    def __init__(
        self,
        database_url: str = "sqlite+aiosqlite:///.genesis/event_store.db",
        archive_after_days: int = 30,
        compact_after_days: int = 7
    ):
        self.database_url = database_url
        self.archive_after_days = archive_after_days
        self.compact_after_days = compact_after_days
        self.engine = None
        self.async_session = None
        self.sequence_counter = 0
        self.event_buffer: list[Event] = []
        self.buffer_size = 100  # Batch writes for performance
        self.flush_interval = 1.0  # Flush every second
        self.running = False
        self.flush_task = None

    async def initialize(self) -> None:
        """Initialize database and create tables."""
        logger.info("Initializing persistent event store", database_url=self.database_url)

        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True
        )

        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create async session factory
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Load sequence counter
        await self.load_sequence_counter()

        # Start background tasks
        self.running = True
        self.flush_task = asyncio.create_task(self.flush_worker())

        logger.info("Event store initialized", sequence_start=self.sequence_counter)

    async def shutdown(self) -> None:
        """Shutdown event store and flush pending events."""
        logger.info("Shutting down event store")

        self.running = False

        # Flush remaining events
        await self.flush_buffer()

        # Cancel background tasks
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass

        # Close database connection
        if self.engine:
            await self.engine.dispose()

        logger.info("Event store shutdown complete")

    async def store_event(self, event: Event) -> int:
        """
        Store an event persistently.
        
        Args:
            event: Event to store
            
        Returns:
            Sequence number assigned to event
        """
        # Assign sequence number
        self.sequence_counter += 1
        sequence_number = self.sequence_counter

        # Add to buffer
        self.event_buffer.append((event, sequence_number))

        # Flush if buffer is full
        if len(self.event_buffer) >= self.buffer_size:
            await self.flush_buffer()

        return sequence_number

    async def flush_buffer(self) -> None:
        """Flush event buffer to database."""
        if not self.event_buffer:
            return

        events_to_store = self.event_buffer.copy()
        self.event_buffer.clear()

        try:
            async with self.async_session() as session:
                for event, sequence_number in events_to_store:
                    stored_event = StoredEvent(
                        event_id=str(event.event_id),
                        event_type=event.event_type.value,
                        aggregate_id=event.aggregate_id,
                        correlation_id=event.correlation_id,
                        sequence_number=sequence_number,
                        event_data=json.dumps(
                            event.event_data,
                            default=self._json_serializer
                        ),
                        metadata=json.dumps(
                            event.metadata,
                            default=self._json_serializer
                        ) if event.metadata else None,
                        created_at=event.timestamp,
                        archived=False,
                        compressed=False
                    )
                    session.add(stored_event)

                await session.commit()

                logger.debug(
                    "Events flushed to storage",
                    count=len(events_to_store),
                    sequence_range=(
                        events_to_store[0][1],
                        events_to_store[-1][1]
                    )
                )

        except Exception as e:
            logger.error("Failed to flush events", error=str(e))
            # Re-add events to buffer for retry
            self.event_buffer.extend(events_to_store)

    async def flush_worker(self) -> None:
        """Background worker to periodically flush events."""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_buffer()
            except Exception as e:
                logger.error("Flush worker error", error=str(e))

    async def replay_events(
        self,
        start_sequence: int | None = None,
        end_sequence: int | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        event_types: list[EventType] | None = None,
        aggregate_id: str | None = None,
        correlation_id: str | None = None,
        callback: callable | None = None
    ) -> list[Event]:
        """
        Replay events based on criteria.
        
        Args:
            start_sequence: Starting sequence number
            end_sequence: Ending sequence number
            start_time: Start timestamp
            end_time: End timestamp
            event_types: Filter by event types
            aggregate_id: Filter by aggregate ID
            correlation_id: Filter by correlation ID
            callback: Async callback for each event
            
        Returns:
            List of replayed events
        """
        logger.info(
            "Starting event replay",
            start_sequence=start_sequence,
            end_sequence=end_sequence,
            start_time=start_time,
            end_time=end_time
        )

        replayed_events = []

        async with self.async_session() as session:
            # Build query
            query = select(StoredEvent)

            if start_sequence:
                query = query.where(StoredEvent.sequence_number >= start_sequence)
            if end_sequence:
                query = query.where(StoredEvent.sequence_number <= end_sequence)
            if start_time:
                query = query.where(StoredEvent.created_at >= start_time)
            if end_time:
                query = query.where(StoredEvent.created_at <= end_time)
            if event_types:
                type_values = [et.value for et in event_types]
                query = query.where(StoredEvent.event_type.in_(type_values))
            if aggregate_id:
                query = query.where(StoredEvent.aggregate_id == aggregate_id)
            if correlation_id:
                query = query.where(StoredEvent.correlation_id == correlation_id)

            query = query.order_by(StoredEvent.sequence_number)

            # Execute query
            result = await session.execute(query)
            stored_events = result.scalars().all()

            # Convert to Event objects
            for stored_event in stored_events:
                event = self._deserialize_event(stored_event)
                replayed_events.append(event)

                # Call callback if provided
                if callback:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                    except Exception as e:
                        logger.error(
                            "Replay callback error",
                            event_id=event.event_id,
                            error=str(e)
                        )

        logger.info(
            "Event replay complete",
            events_replayed=len(replayed_events)
        )

        return replayed_events

    async def archive_old_events(self) -> int:
        """
        Archive events older than threshold.
        
        Returns:
            Number of events archived
        """
        cutoff_date = datetime.now() - timedelta(days=self.archive_after_days)

        logger.info(
            "Starting event archival",
            cutoff_date=cutoff_date,
            archive_after_days=self.archive_after_days
        )

        archived_count = 0

        async with self.async_session() as session:
            # Get events to archive
            query = select(StoredEvent).where(
                StoredEvent.created_at < cutoff_date,
                StoredEvent.archived == False
            ).order_by(StoredEvent.created_at)

            result = await session.execute(query)
            events_to_archive = result.scalars().all()

            if not events_to_archive:
                logger.info("No events to archive")
                return 0

            # Group events by day for batch archival
            events_by_day: dict[str, list[StoredEvent]] = {}
            for event in events_to_archive:
                day_key = event.created_at.strftime("%Y-%m-%d")
                if day_key not in events_by_day:
                    events_by_day[day_key] = []
                events_by_day[day_key].append(event)

            # Archive each day's events
            for day_key, day_events in events_by_day.items():
                # Serialize events
                events_data = [
                    {
                        "event_id": e.event_id,
                        "event_type": e.event_type,
                        "aggregate_id": e.aggregate_id,
                        "correlation_id": e.correlation_id,
                        "sequence_number": e.sequence_number,
                        "event_data": e.event_data,
                        "metadata": e.metadata,
                        "created_at": e.created_at.isoformat()
                    }
                    for e in day_events
                ]

                # Compress data
                json_data = json.dumps(events_data)
                compressed_data = gzip.compress(json_data.encode())

                # Calculate checksum
                import hashlib
                checksum = hashlib.sha256(compressed_data).hexdigest()

                # Create archive entry
                archive = EventArchive(
                    batch_id=f"archive_{day_key}",
                    start_date=day_events[0].created_at,
                    end_date=day_events[-1].created_at,
                    event_count=len(day_events),
                    compressed_data=compressed_data.hex(),
                    checksum=checksum,
                    created_at=datetime.now()
                )
                session.add(archive)

                # Mark events as archived
                for event in day_events:
                    event.archived = True
                    archived_count += 1

            await session.commit()

        logger.info(
            "Event archival complete",
            events_archived=archived_count,
            days_processed=len(events_by_day)
        )

        return archived_count

    async def compact_events(self) -> int:
        """
        Compact events to save space.
        
        Returns:
            Number of events compacted
        """
        cutoff_date = datetime.now() - timedelta(days=self.compact_after_days)

        logger.info(
            "Starting event compaction",
            cutoff_date=cutoff_date,
            compact_after_days=self.compact_after_days
        )

        compacted_count = 0

        async with self.async_session() as session:
            # Get events to compact
            query = select(StoredEvent).where(
                StoredEvent.created_at < cutoff_date,
                StoredEvent.compressed == False,
                StoredEvent.archived == False
            )

            result = await session.execute(query)
            events_to_compact = result.scalars().all()

            for event in events_to_compact:
                # Compress event data
                compressed_data = gzip.compress(event.event_data.encode())
                event.event_data = compressed_data.hex()

                if event.metadata:
                    compressed_metadata = gzip.compress(event.metadata.encode())
                    event.metadata = compressed_metadata.hex()

                event.compressed = True
                compacted_count += 1

            await session.commit()

        logger.info(
            "Event compaction complete",
            events_compacted=compacted_count
        )

        return compacted_count

    async def get_event_statistics(self) -> dict[str, Any]:
        """Get statistics about stored events."""
        async with self.async_session() as session:
            # Total events
            total_query = select(StoredEvent)
            total_result = await session.execute(total_query)
            total_events = len(total_result.scalars().all())

            # Events by type
            from sqlalchemy import func
            type_query = select(
                StoredEvent.event_type,
                func.count(StoredEvent.id)
            ).group_by(StoredEvent.event_type)

            type_result = await session.execute(type_query)
            events_by_type = dict(type_result.all())

            # Archive statistics
            archive_query = select(EventArchive)
            archive_result = await session.execute(archive_query)
            archives = archive_result.scalars().all()

            total_archived = sum(a.event_count for a in archives)

            return {
                "total_events": total_events,
                "events_by_type": events_by_type,
                "total_archived": total_archived,
                "archive_count": len(archives),
                "current_sequence": self.sequence_counter,
                "buffer_size": len(self.event_buffer)
            }

    async def load_sequence_counter(self) -> None:
        """Load the current sequence counter from database."""
        async with self.async_session() as session:
            from sqlalchemy import func
            query = select(func.max(StoredEvent.sequence_number))
            result = await session.execute(query)
            max_sequence = result.scalar()
            self.sequence_counter = max_sequence or 0

    def _deserialize_event(self, stored_event: StoredEvent) -> Event:
        """Deserialize stored event to Event object."""
        # Decompress if needed
        if stored_event.compressed:
            event_data = json.loads(
                gzip.decompress(bytes.fromhex(stored_event.event_data))
            )
            metadata = json.loads(
                gzip.decompress(bytes.fromhex(stored_event.metadata))
            ) if stored_event.metadata else None
        else:
            event_data = json.loads(stored_event.event_data)
            metadata = json.loads(stored_event.metadata) if stored_event.metadata else None

        return Event(
            event_id=UUID(stored_event.event_id),
            event_type=EventType(stored_event.event_type),
            aggregate_id=stored_event.aggregate_id,
            correlation_id=stored_event.correlation_id,
            event_data=event_data,
            metadata=metadata,
            timestamp=stored_event.created_at
        )

    def _json_serializer(self, obj):
        """JSON serializer for special types."""
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        return str(obj)
