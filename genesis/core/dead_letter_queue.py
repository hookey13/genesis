"""
Dead Letter Queue (DLQ) system for handling failed operations.

Provides a mechanism to store, retry, and manage failed operations that
couldn't be processed successfully after retries.
"""

import asyncio
import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import structlog

from genesis.core.exceptions import BaseError


class DLQItemStatus(Enum):
    """Status of items in the dead letter queue."""
    
    PENDING = "pending"  # Waiting for retry
    PROCESSING = "processing"  # Currently being processed
    FAILED = "failed"  # Permanently failed
    SUCCEEDED = "succeeded"  # Successfully processed on retry
    EXPIRED = "expired"  # Exceeded max retention time


@dataclass
class DLQItem:
    """Represents an item in the dead letter queue."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    error_type: str = ""
    retry_count: int = 0
    max_retries: int = 3
    status: DLQItemStatus = DLQItemStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    next_retry_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.next_retry_at:
            data["next_retry_at"] = self.next_retry_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DLQItem":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = DLQItemStatus(data["status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("next_retry_at"):
            data["next_retry_at"] = datetime.fromisoformat(data["next_retry_at"])
        return cls(**data)


class DeadLetterQueue:
    """
    Dead Letter Queue implementation for failed operations.
    
    Provides:
    - In-memory queue for MVP (asyncio.Queue based)
    - SQLite persistence for durability
    - Retry mechanism with exponential backoff
    - Manual inspection and intervention interface
    """
    
    def __init__(
        self,
        name: str = "default",
        max_size: int = 10000,
        db_path: Optional[Path] = None,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.name = name
        self.max_size = max_size
        self.logger = logger or structlog.get_logger(__name__)
        
        # In-memory queue for fast access
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._items: Dict[str, DLQItem] = {}
        
        # SQLite for persistence
        self.db_path = db_path or Path(".genesis/dlq.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # Retry handlers
        self._retry_handlers: Dict[str, Callable] = {}
        
        # Background retry task
        self._retry_task: Optional[asyncio.Task] = None
        self._shutdown = False
    
    def _init_database(self):
        """Initialize SQLite database for persistence."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dlq_items (
                    id TEXT PRIMARY KEY,
                    queue_name TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    error_message TEXT,
                    error_type TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    next_retry_at TEXT,
                    correlation_id TEXT,
                    metadata TEXT,
                    INDEX idx_queue_status (queue_name, status),
                    INDEX idx_next_retry (next_retry_at)
                )
            """)
            conn.commit()
    
    async def add(
        self,
        operation_type: str,
        payload: Dict[str, Any],
        error: Exception,
        max_retries: int = 3,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a failed operation to the dead letter queue.
        
        Args:
            operation_type: Type of operation that failed
            payload: Operation payload to retry
            error: The exception that caused the failure
            max_retries: Maximum retry attempts
            correlation_id: Correlation ID for tracking
            metadata: Additional metadata
            
        Returns:
            ID of the DLQ item
        """
        # Calculate next retry time with exponential backoff
        next_retry = datetime.utcnow() + timedelta(seconds=60)  # Start with 1 minute
        
        item = DLQItem(
            operation_type=operation_type,
            payload=payload,
            error_message=str(error),
            error_type=type(error).__name__,
            max_retries=max_retries,
            next_retry_at=next_retry,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )
        
        # Add to in-memory storage
        self._items[item.id] = item
        
        # Persist to database
        self._persist_item(item)
        
        # Add to queue if space available
        if not self._queue.full():
            await self._queue.put(item.id)
        
        self.logger.warning(
            "Added item to dead letter queue",
            dlq_item_id=item.id,
            operation_type=operation_type,
            error_type=type(error).__name__,
            correlation_id=correlation_id,
        )
        
        return item.id
    
    def _persist_item(self, item: DLQItem):
        """Persist item to SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO dlq_items (
                    id, queue_name, operation_type, payload, error_message,
                    error_type, retry_count, max_retries, status,
                    created_at, updated_at, next_retry_at, correlation_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.id,
                self.name,
                item.operation_type,
                json.dumps(item.payload),
                item.error_message,
                item.error_type,
                item.retry_count,
                item.max_retries,
                item.status.value,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
                item.next_retry_at.isoformat() if item.next_retry_at else None,
                item.correlation_id,
                json.dumps(item.metadata),
            ))
            conn.commit()
    
    def _load_items_from_db(self):
        """Load pending items from database on startup."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM dlq_items
                WHERE queue_name = ? AND status IN (?, ?)
                ORDER BY next_retry_at
            """, (self.name, DLQItemStatus.PENDING.value, DLQItemStatus.PROCESSING.value))
            
            for row in cursor:
                item_data = {
                    "id": row["id"],
                    "operation_type": row["operation_type"],
                    "payload": json.loads(row["payload"]),
                    "error_message": row["error_message"],
                    "error_type": row["error_type"],
                    "retry_count": row["retry_count"],
                    "max_retries": row["max_retries"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "next_retry_at": row["next_retry_at"],
                    "correlation_id": row["correlation_id"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
                item = DLQItem.from_dict(item_data)
                self._items[item.id] = item
    
    def register_retry_handler(
        self,
        operation_type: str,
        handler: Callable,
    ):
        """
        Register a handler for retrying specific operation types.
        
        Args:
            operation_type: Type of operation to handle
            handler: Async function to retry the operation
        """
        self._retry_handlers[operation_type] = handler
        self.logger.info(
            "Registered retry handler",
            operation_type=operation_type,
        )
    
    async def retry_item(self, item_id: str) -> bool:
        """
        Manually retry a specific DLQ item.
        
        Args:
            item_id: ID of the item to retry
            
        Returns:
            True if retry succeeded, False otherwise
        """
        item = self._items.get(item_id)
        if not item:
            self.logger.error("DLQ item not found", item_id=item_id)
            return False
        
        if item.status != DLQItemStatus.PENDING:
            self.logger.warning(
                "Cannot retry item with status",
                item_id=item_id,
                status=item.status.value,
            )
            return False
        
        handler = self._retry_handlers.get(item.operation_type)
        if not handler:
            self.logger.error(
                "No retry handler registered",
                operation_type=item.operation_type,
            )
            return False
        
        # Update status to processing
        item.status = DLQItemStatus.PROCESSING
        item.updated_at = datetime.utcnow()
        self._persist_item(item)
        
        try:
            # Execute retry handler
            if asyncio.iscoroutinefunction(handler):
                await handler(item.payload)
            else:
                handler(item.payload)
            
            # Mark as succeeded
            item.status = DLQItemStatus.SUCCEEDED
            item.updated_at = datetime.utcnow()
            self._persist_item(item)
            
            self.logger.info(
                "DLQ item retry succeeded",
                item_id=item_id,
                operation_type=item.operation_type,
            )
            return True
            
        except Exception as e:
            # Increment retry count
            item.retry_count += 1
            item.updated_at = datetime.utcnow()
            
            if item.retry_count >= item.max_retries:
                # Max retries exceeded, mark as failed
                item.status = DLQItemStatus.FAILED
                self.logger.error(
                    "DLQ item permanently failed",
                    item_id=item_id,
                    retry_count=item.retry_count,
                    error=str(e),
                )
            else:
                # Schedule next retry with exponential backoff
                backoff_seconds = min(300, 60 * (2 ** item.retry_count))
                item.status = DLQItemStatus.PENDING
                item.next_retry_at = datetime.utcnow() + timedelta(seconds=backoff_seconds)
                
                self.logger.warning(
                    "DLQ item retry failed, rescheduling",
                    item_id=item_id,
                    retry_count=item.retry_count,
                    next_retry_seconds=backoff_seconds,
                    error=str(e),
                )
            
            self._persist_item(item)
            return False
    
    async def start_retry_worker(self, interval: int = 60):
        """
        Start background worker for automatic retries.
        
        Args:
            interval: Check interval in seconds
        """
        self._shutdown = False
        self._retry_task = asyncio.create_task(
            self._retry_worker(interval)
        )
        self.logger.info(
            "Started DLQ retry worker",
            queue=self.name,
            interval=interval,
        )
    
    async def stop_retry_worker(self):
        """Stop the background retry worker."""
        self._shutdown = True
        if self._retry_task:
            await self._retry_task
            self._retry_task = None
        self.logger.info("Stopped DLQ retry worker", queue=self.name)
    
    async def _retry_worker(self, interval: int):
        """Background worker for automatic retries."""
        while not self._shutdown:
            try:
                # Find items ready for retry
                now = datetime.utcnow()
                ready_items = [
                    item for item in self._items.values()
                    if (
                        item.status == DLQItemStatus.PENDING and
                        item.next_retry_at and
                        item.next_retry_at <= now
                    )
                ]
                
                # Retry each ready item
                for item in ready_items:
                    await self.retry_item(item.id)
                
                # Clean up old succeeded/failed items
                self._cleanup_old_items()
                
            except Exception as e:
                self.logger.error(
                    "Error in DLQ retry worker",
                    error=str(e),
                )
            
            # Wait for next interval
            await asyncio.sleep(interval)
    
    def _cleanup_old_items(self, retention_days: int = 7):
        """Remove old succeeded/failed items."""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        
        # Find items to remove
        to_remove = [
            item_id for item_id, item in self._items.items()
            if (
                item.status in (DLQItemStatus.SUCCEEDED, DLQItemStatus.FAILED) and
                item.updated_at < cutoff
            )
        ]
        
        if to_remove:
            # Remove from memory
            for item_id in to_remove:
                del self._items[item_id]
            
            # Remove from database
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" * len(to_remove))
                conn.execute(
                    f"DELETE FROM dlq_items WHERE id IN ({placeholders})",
                    to_remove,
                )
                conn.commit()
            
            self.logger.info(
                "Cleaned up old DLQ items",
                count=len(to_remove),
            )
    
    def get_item(self, item_id: str) -> Optional[DLQItem]:
        """Get a specific DLQ item."""
        return self._items.get(item_id)
    
    def get_items(
        self,
        status: Optional[DLQItemStatus] = None,
        operation_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[DLQItem]:
        """
        Get DLQ items with optional filtering.
        
        Args:
            status: Filter by status
            operation_type: Filter by operation type
            limit: Maximum items to return
            
        Returns:
            List of matching DLQ items
        """
        items = list(self._items.values())
        
        if status:
            items = [i for i in items if i.status == status]
        
        if operation_type:
            items = [i for i in items if i.operation_type == operation_type]
        
        # Sort by next retry time
        items.sort(key=lambda x: x.next_retry_at or datetime.max)
        
        return items[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        status_counts = {}
        for item in self._items.values():
            status_counts[item.status.value] = status_counts.get(item.status.value, 0) + 1
        
        return {
            "queue_name": self.name,
            "total_items": len(self._items),
            "status_counts": status_counts,
            "queue_size": self._queue.qsize(),
            "max_size": self.max_size,
        }
    
    async def clear(self, status: Optional[DLQItemStatus] = None):
        """
        Clear items from the queue.
        
        Args:
            status: Only clear items with this status (None = all)
        """
        if status:
            to_remove = [
                item_id for item_id, item in self._items.items()
                if item.status == status
            ]
        else:
            to_remove = list(self._items.keys())
        
        # Remove from memory
        for item_id in to_remove:
            self._items.pop(item_id, None)
        
        # Clear from database
        with sqlite3.connect(self.db_path) as conn:
            if status:
                conn.execute(
                    "DELETE FROM dlq_items WHERE queue_name = ? AND status = ?",
                    (self.name, status.value),
                )
            else:
                conn.execute(
                    "DELETE FROM dlq_items WHERE queue_name = ?",
                    (self.name,),
                )
            conn.commit()
        
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self.logger.info(
            "Cleared DLQ items",
            count=len(to_remove),
            status=status.value if status else "all",
        )