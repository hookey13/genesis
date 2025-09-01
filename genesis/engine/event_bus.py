"""
Event Bus implementation with priority lanes.

Manages event publishing, subscription, and delivery with support
for different priority levels and event filtering.
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import structlog

from genesis.core.events import Event, EventPriority, EventType

logger = structlog.get_logger(__name__)


@dataclass
class Subscription:
    """Event subscription."""

    subscription_id: str
    event_types: set[EventType]
    callback: Callable[[Event], None]
    filter_func: Callable[[Event], bool] | None = None
    priority: EventPriority = EventPriority.NORMAL


class PriorityQueue:
    """Priority queue for events."""

    def __init__(self):
        """Initialize priority queues."""
        self.queues = {
            EventPriority.CRITICAL: asyncio.Queue(),
            EventPriority.HIGH: asyncio.Queue(),
            EventPriority.NORMAL: asyncio.Queue(),
            EventPriority.LOW: asyncio.Queue(),
        }

    async def put(self, event: Event, priority: EventPriority) -> None:
        """Add event to appropriate priority queue."""
        await self.queues[priority].put(event)

    async def get(self) -> Event:
        """Get next event respecting priority."""
        # Check queues in priority order
        for priority in [
            EventPriority.CRITICAL,
            EventPriority.HIGH,
            EventPriority.NORMAL,
            EventPriority.LOW,
        ]:
            queue = self.queues[priority]
            if not queue.empty():
                return await queue.get()

        # If all queues empty, wait on critical queue
        return await self.queues[EventPriority.CRITICAL].get()

    def qsize(self) -> dict[EventPriority, int]:
        """Get size of each queue."""
        return {priority: queue.qsize() for priority, queue in self.queues.items()}


class EventBus:
    """
    Central event bus for the trading system.

    Manages event publishing and subscription with priority-based
    delivery and support for both sync and async callbacks.
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize the event bus.

        Args:
            max_queue_size: Maximum events per priority queue
        """
        self.subscriptions: dict[EventType, list[Subscription]] = defaultdict(list)
        self.global_subscriptions: list[Subscription] = []  # Subscribe to all events
        self.priority_queue = PriorityQueue()
        self.max_queue_size = max_queue_size
        self.running = False
        self.processor_task: asyncio.Task | None = None

        # Statistics
        self.events_published = 0
        self.events_delivered = 0
        self.events_dropped = 0
        self.delivery_errors = 0

        # Event batching
        self.batch_size = 10
        self.batch_timeout = 0.1  # seconds
        
        # Backpressure handling
        self.backpressure_threshold = 0.8  # 80% of max queue size
        self.backpressure_active = False
        self.shed_low_priority = False  # Shed low priority events when under pressure
        self.backpressure_callbacks: list[Callable[[bool], None]] = []

        logger.info("EventBus initialized", max_queue_size=max_queue_size)

    async def start(self) -> None:
        """Start the event bus."""
        if self.running:
            return

        self.running = True
        self.processor_task = asyncio.create_task(self._process_events())
        logger.info("EventBus started")

    async def stop(self) -> None:
        """Stop the event bus."""
        if not self.running:
            return

        self.running = False

        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "EventBus stopped",
            events_published=self.events_published,
            events_delivered=self.events_delivered,
            events_dropped=self.events_dropped,
        )

    def subscribe(
        self,
        event_type_or_callback: Any,
        callback_or_event_types: Any = None,
        filter_func: Callable[[Event], bool] | None = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> str:
        """
        Subscribe to events.

        Args:
            event_type_or_callback: EventType or callback function
            callback_or_event_types: Callback or Set of event types
            filter_func: Optional filter function
            priority: Subscription priority

        Returns:
            Subscription ID
        """
        # Handle both calling patterns for backwards compatibility
        if isinstance(event_type_or_callback, EventType):
            # Old pattern: subscribe(EventType, callback)
            event_types = {event_type_or_callback}
            callback = callback_or_event_types
        elif callable(event_type_or_callback):
            # New pattern: subscribe(callback, event_types)
            callback = event_type_or_callback
            if callback_or_event_types:
                if isinstance(callback_or_event_types, EventType):
                    event_types = {callback_or_event_types}
                elif isinstance(callback_or_event_types, set):
                    event_types = callback_or_event_types
                else:
                    event_types = set()
            else:
                event_types = set()
        else:
            raise ValueError("First argument must be EventType or callable")

        subscription = Subscription(
            subscription_id=str(uuid4()),
            event_types=event_types or set(),
            callback=callback,
            filter_func=filter_func,
            priority=priority,
        )

        if event_types:
            # Subscribe to specific event types
            for event_type in event_types:
                self.subscriptions[event_type].append(subscription)
        else:
            # Subscribe to all events
            self.global_subscriptions.append(subscription)

        logger.info(
            "Subscription added",
            subscription_id=subscription.subscription_id,
            event_types=[e.value for e in (event_types or [])],
        )

        return subscription.subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID of subscription to remove

        Returns:
            True if subscription was found and removed
        """
        removed = False

        # Check event-specific subscriptions
        for event_type, subs in self.subscriptions.items():
            for sub in subs[:]:
                if sub.subscription_id == subscription_id:
                    subs.remove(sub)
                    removed = True

        # Check global subscriptions
        for sub in self.global_subscriptions[:]:
            if sub.subscription_id == subscription_id:
                self.global_subscriptions.remove(sub)
                removed = True

        if removed:
            logger.info("Subscription removed", subscription_id=subscription_id)

        return removed

    def _check_backpressure(self) -> bool:
        """
        Check if backpressure conditions are met.
        
        Returns:
            True if backpressure should be applied
        """
        queue_sizes = self.priority_queue.qsize()
        total_queued = sum(queue_sizes.values())
        max_total = self.max_queue_size * len(queue_sizes)
        
        return total_queued > (max_total * self.backpressure_threshold)
    
    def _update_backpressure_state(self) -> None:
        """Update backpressure state and notify callbacks."""
        was_active = self.backpressure_active
        self.backpressure_active = self._check_backpressure()
        
        if self.backpressure_active != was_active:
            logger.warning(
                "Backpressure state changed",
                active=self.backpressure_active,
                queue_sizes=self.priority_queue.qsize()
            )
            
            # Notify callbacks
            for callback in self.backpressure_callbacks:
                try:
                    callback(self.backpressure_active)
                except Exception as e:
                    logger.error("Backpressure callback failed", error=str(e))
    
    def register_backpressure_callback(self, callback: Callable[[bool], None]) -> None:
        """Register a callback for backpressure state changes."""
        self.backpressure_callbacks.append(callback)
    
    async def publish(
        self, event: Event, priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """
        Publish an event.

        Args:
            event: Event to publish
            priority: Event priority
        """
        # Check backpressure
        self._update_backpressure_state()
        
        # Shed low priority events if under pressure
        if self.backpressure_active and self.shed_low_priority:
            if priority == EventPriority.LOW:
                self.events_dropped += 1
                logger.warning(
                    "Low priority event shed due to backpressure",
                    event_type=event.event_type.value,
                )
                return
        
        # Check queue size
        queue_sizes = self.priority_queue.qsize()
        if queue_sizes[priority] >= self.max_queue_size:
            self.events_dropped += 1
            logger.warning(
                "Event dropped - queue full",
                event_type=event.event_type.value,
                priority=priority.value,
                queue_size=queue_sizes[priority],
            )
            return

        # Add to priority queue
        await self.priority_queue.put(event, priority)
        self.events_published += 1

        logger.debug(
            "Event published",
            event_type=event.event_type.value,
            priority=priority.value,
            aggregate_id=event.aggregate_id,
        )

    async def _process_events(self) -> None:
        """Process events from priority queues."""
        batch = []
        last_batch_time = asyncio.get_event_loop().time()

        while self.running:
            try:
                # Get next event with timeout
                try:
                    event = await asyncio.wait_for(
                        self.priority_queue.get(), timeout=self.batch_timeout
                    )
                    batch.append(event)
                except TimeoutError:
                    pass

                # Check if we should process batch
                current_time = asyncio.get_event_loop().time()
                should_process = (
                    len(batch) >= self.batch_size
                    or (current_time - last_batch_time) >= self.batch_timeout
                    or (batch and not self.running)
                )

                if should_process and batch:
                    # Process batch
                    await self._deliver_batch(batch)
                    batch = []
                    last_batch_time = current_time

            except Exception as e:
                logger.error("Error processing events", error=str(e))
                await asyncio.sleep(1)

    async def _deliver_batch(self, events: list[Event]) -> None:
        """Deliver a batch of events to subscribers."""
        for event in events:
            await self._deliver_event(event)

    async def _deliver_event(self, event: Event) -> None:
        """Deliver an event to all relevant subscribers."""
        # Get subscribers for this event type
        subscribers = list(self.subscriptions.get(event.event_type, []))
        subscribers.extend(self.global_subscriptions)

        # Sort by priority
        subscribers.sort(key=lambda s: s.priority.value)

        # Deliver to each subscriber
        for subscription in subscribers:
            try:
                # Apply filter if present
                if subscription.filter_func and not subscription.filter_func(event):
                    continue

                # Call callback
                if asyncio.iscoroutinefunction(subscription.callback):
                    await subscription.callback(event)
                else:
                    subscription.callback(event)

                self.events_delivered += 1

            except Exception as e:
                self.delivery_errors += 1
                logger.error(
                    "Error delivering event",
                    subscription_id=subscription.subscription_id,
                    event_type=event.event_type.value,
                    error=str(e),
                )

    def get_backpressure_metrics(self) -> dict[str, Any]:
        """Get backpressure metrics."""
        queue_sizes = self.priority_queue.qsize()
        total_queued = sum(queue_sizes.values())
        max_total = self.max_queue_size * len(queue_sizes)
        
        return {
            "backpressure_active": self.backpressure_active,
            "queue_utilization_percent": (total_queued / max_total * 100) if max_total > 0 else 0,
            "total_queued": total_queued,
            "max_capacity": max_total,
            "shed_low_priority": self.shed_low_priority,
            "queue_sizes": queue_sizes,
            "events_dropped": self.events_dropped,
        }
    
    def get_statistics(self) -> dict[str, Any]:
        """Get event bus statistics."""
        queue_sizes = self.priority_queue.qsize()

        return {
            "running": self.running,
            "events_published": self.events_published,
            "events_delivered": self.events_delivered,
            "events_dropped": self.events_dropped,
            "delivery_errors": self.delivery_errors,
            "queue_sizes": {
                priority.value: size for priority, size in queue_sizes.items()
            },
            "subscriptions": {
                event_type.value: len(subs)
                for event_type, subs in self.subscriptions.items()
            },
            "global_subscriptions": len(self.global_subscriptions),
        }

    async def wait_for_event(
        self,
        event_type: EventType,
        timeout: float | None = None,
        filter_func: Callable[[Event], bool] | None = None,
    ) -> Event | None:
        """
        Wait for a specific event type.

        Args:
            event_type: Event type to wait for
            timeout: Maximum time to wait
            filter_func: Optional filter function

        Returns:
            Event if received, None if timeout
        """
        received_event = None
        event_received = asyncio.Event()

        def callback(event: Event):
            nonlocal received_event
            if not filter_func or filter_func(event):
                received_event = event
                event_received.set()

        # Subscribe
        sub_id = self.subscribe(callback=callback, event_types={event_type})

        try:
            # Wait for event
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return received_event
        except TimeoutError:
            return None
        finally:
            # Unsubscribe
            self.unsubscribe(sub_id)
