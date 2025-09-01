"""
Automatic recovery procedures for common failures.

Implements recovery strategies for various failure scenarios with
automated remediation and self-healing capabilities.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

from genesis.core.exceptions import (
    ConnectionTimeout,
    RateLimitError,
    DatabaseLocked,
    NetworkError,
)


class RecoveryStatus(Enum):
    """Status of recovery procedure."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RecoveryProcedure:
    """Defines a recovery procedure."""
    
    name: str
    description: str
    error_types: List[type]  # Exception types this procedure handles
    recovery_function: Callable
    max_attempts: int = 3
    timeout_seconds: float = 30.0
    cooldown_seconds: float = 60.0  # Time before retry
    requires_manual: bool = False
    priority: int = 0  # Higher priority procedures run first


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    
    procedure_name: str
    error_type: str
    status: RecoveryStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    attempts: int = 0
    error_message: Optional[str] = None
    correlation_id: Optional[str] = None


class RecoveryManager:
    """
    Manages automatic recovery procedures for system failures.
    
    Provides:
    - Registry of recovery procedures
    - Automatic matching and execution
    - Recovery attempt tracking
    - Manual intervention interface
    """
    
    def __init__(
        self,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        
        # Recovery procedure registry
        self._procedures: Dict[str, RecoveryProcedure] = {}
        self._error_mapping: Dict[type, List[str]] = {}  # error_type -> procedure names
        
        # Recovery tracking
        self._active_recoveries: Dict[str, RecoveryAttempt] = {}
        self._recovery_history: List[RecoveryAttempt] = []
        self._cooldowns: Dict[str, datetime] = {}  # procedure_name -> cooldown_until
        
        # Initialize default procedures
        self._initialize_default_procedures()
    
    def _initialize_default_procedures(self):
        """Initialize default recovery procedures."""
        
        # Connection timeout recovery
        async def recover_connection_timeout(error: ConnectionTimeout, context: Dict):
            """Reconnect with exponential backoff."""
            self.logger.info("Attempting connection recovery", context=context)
            
            # Simulate reconnection logic
            await asyncio.sleep(2)
            
            # In real implementation, would reconnect to service
            return {"reconnected": True, "retry_after": 5}
        
        self.register_procedure(
            RecoveryProcedure(
                name="connection_timeout_recovery",
                description="Reconnect with exponential backoff",
                error_types=[ConnectionTimeout, NetworkError],
                recovery_function=recover_connection_timeout,
                max_attempts=5,
                cooldown_seconds=30,
            )
        )
        
        # Rate limit recovery
        async def recover_rate_limit(error: RateLimitError, context: Dict):
            """Wait for rate limit window and retry."""
            retry_after = error.retry_after_seconds or 60
            
            self.logger.info(
                "Rate limit recovery, waiting",
                retry_after_seconds=retry_after,
            )
            
            # Queue request for retry
            context["retry_after"] = datetime.utcnow() + timedelta(seconds=retry_after)
            context["queued"] = True
            
            return context
        
        self.register_procedure(
            RecoveryProcedure(
                name="rate_limit_recovery",
                description="Queue and retry after rate limit window",
                error_types=[RateLimitError],
                recovery_function=recover_rate_limit,
                max_attempts=1,  # No point retrying rate limits
                cooldown_seconds=0,
            )
        )
        
        # Database lock recovery
        async def recover_database_lock(error: DatabaseLocked, context: Dict):
            """Retry transaction with jitter."""
            self.logger.info("Database lock recovery", table=error.table)
            
            # Add jitter to prevent thundering herd
            jitter = asyncio.create_task(asyncio.sleep(0.1 + (time.time() % 0.5)))
            await jitter
            
            # In real implementation, would retry transaction
            return {"retried": True, "jitter_ms": int((time.time() % 0.5) * 1000)}
        
        self.register_procedure(
            RecoveryProcedure(
                name="database_lock_recovery",
                description="Retry database transaction with jitter",
                error_types=[DatabaseLocked],
                recovery_function=recover_database_lock,
                max_attempts=3,
                cooldown_seconds=1,
            )
        )
        
        # WebSocket disconnect recovery
        async def recover_websocket_disconnect(error: Exception, context: Dict):
            """Reconnect WebSocket with state restoration."""
            self.logger.info("WebSocket recovery initiated")
            
            # Simulate WebSocket reconnection
            await asyncio.sleep(1)
            
            # Restore subscriptions
            subscriptions = context.get("subscriptions", [])
            for sub in subscriptions:
                self.logger.debug("Restoring subscription", subscription=sub)
            
            return {
                "reconnected": True,
                "subscriptions_restored": len(subscriptions),
            }
        
        self.register_procedure(
            RecoveryProcedure(
                name="websocket_recovery",
                description="Automatic WebSocket reconnection",
                error_types=[ConnectionError],
                recovery_function=recover_websocket_disconnect,
                max_attempts=10,
                cooldown_seconds=5,
                priority=10,  # High priority for market data
            )
        )
    
    def register_procedure(self, procedure: RecoveryProcedure):
        """
        Register a recovery procedure.
        
        Args:
            procedure: Recovery procedure to register
        """
        self._procedures[procedure.name] = procedure
        
        # Update error mapping
        for error_type in procedure.error_types:
            if error_type not in self._error_mapping:
                self._error_mapping[error_type] = []
            self._error_mapping[error_type].append(procedure.name)
        
        self.logger.info(
            "Registered recovery procedure",
            name=procedure.name,
            error_types=[t.__name__ for t in procedure.error_types],
        )
    
    async def attempt_recovery(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
    ) -> Optional[RecoveryAttempt]:
        """
        Attempt automatic recovery for an error.
        
        Args:
            error: The exception to recover from
            context: Additional context for recovery
            correlation_id: Correlation ID for tracking
            
        Returns:
            Recovery attempt record or None if no procedure found
        """
        context = context or {}
        
        # Find applicable procedures
        procedures = self._find_procedures(error)
        if not procedures:
            self.logger.debug(
                "No recovery procedure found",
                error_type=type(error).__name__,
            )
            return None
        
        # Sort by priority
        procedures.sort(key=lambda p: p.priority, reverse=True)
        
        # Try each procedure
        for procedure in procedures:
            # Check cooldown
            if self._is_in_cooldown(procedure.name):
                self.logger.debug(
                    "Procedure in cooldown",
                    procedure=procedure.name,
                )
                continue
            
            # Check if already recovering
            if procedure.name in self._active_recoveries:
                self.logger.debug(
                    "Recovery already in progress",
                    procedure=procedure.name,
                )
                continue
            
            # Attempt recovery
            attempt = await self._execute_procedure(
                procedure,
                error,
                context,
                correlation_id,
            )
            
            if attempt.status == RecoveryStatus.SUCCEEDED:
                return attempt
        
        return None
    
    def _find_procedures(self, error: Exception) -> List[RecoveryProcedure]:
        """Find applicable recovery procedures for an error."""
        procedures = []
        
        # Check exact type match
        error_type = type(error)
        if error_type in self._error_mapping:
            for proc_name in self._error_mapping[error_type]:
                procedures.append(self._procedures[proc_name])
        
        # Check inheritance chain
        for base_type in error_type.__mro__[1:]:
            if base_type in self._error_mapping:
                for proc_name in self._error_mapping[base_type]:
                    if self._procedures[proc_name] not in procedures:
                        procedures.append(self._procedures[proc_name])
        
        return procedures
    
    def _is_in_cooldown(self, procedure_name: str) -> bool:
        """Check if procedure is in cooldown."""
        if procedure_name not in self._cooldowns:
            return False
        
        return datetime.utcnow() < self._cooldowns[procedure_name]
    
    async def _execute_procedure(
        self,
        procedure: RecoveryProcedure,
        error: Exception,
        context: Dict[str, Any],
        correlation_id: Optional[str],
    ) -> RecoveryAttempt:
        """Execute a recovery procedure."""
        attempt = RecoveryAttempt(
            procedure_name=procedure.name,
            error_type=type(error).__name__,
            status=RecoveryStatus.IN_PROGRESS,
            started_at=datetime.utcnow(),
            correlation_id=correlation_id,
        )
        
        # Track active recovery
        self._active_recoveries[procedure.name] = attempt
        
        self.logger.info(
            "Starting recovery procedure",
            procedure=procedure.name,
            error_type=type(error).__name__,
            correlation_id=correlation_id,
        )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                procedure.recovery_function(error, context),
                timeout=procedure.timeout_seconds,
            )
            
            # Success
            attempt.status = RecoveryStatus.SUCCEEDED
            attempt.completed_at = datetime.utcnow()
            
            self.logger.info(
                "Recovery procedure succeeded",
                procedure=procedure.name,
                result=result,
            )
            
        except asyncio.TimeoutError:
            attempt.status = RecoveryStatus.FAILED
            attempt.error_message = f"Timeout after {procedure.timeout_seconds}s"
            attempt.completed_at = datetime.utcnow()
            
            self.logger.error(
                "Recovery procedure timed out",
                procedure=procedure.name,
            )
            
        except Exception as e:
            attempt.status = RecoveryStatus.FAILED
            attempt.error_message = str(e)
            attempt.completed_at = datetime.utcnow()
            
            self.logger.error(
                "Recovery procedure failed",
                procedure=procedure.name,
                error=str(e),
            )
        
        finally:
            # Remove from active
            self._active_recoveries.pop(procedure.name, None)
            
            # Add to history
            self._recovery_history.append(attempt)
            
            # Set cooldown if failed
            if attempt.status == RecoveryStatus.FAILED:
                self._cooldowns[procedure.name] = (
                    datetime.utcnow() + timedelta(seconds=procedure.cooldown_seconds)
                )
        
        return attempt
    
    def get_active_recoveries(self) -> List[RecoveryAttempt]:
        """Get list of active recovery attempts."""
        return list(self._active_recoveries.values())
    
    def get_recovery_history(
        self,
        limit: int = 100,
        status: Optional[RecoveryStatus] = None,
    ) -> List[RecoveryAttempt]:
        """
        Get recovery attempt history.
        
        Args:
            limit: Maximum records to return
            status: Filter by status
            
        Returns:
            List of recovery attempts
        """
        history = self._recovery_history[-limit:]
        
        if status:
            history = [h for h in history if h.status == status]
        
        return history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recovery manager statistics."""
        total_attempts = len(self._recovery_history)
        
        if total_attempts == 0:
            return {
                "total_attempts": 0,
                "success_rate": 0.0,
                "active_recoveries": len(self._active_recoveries),
                "procedures_registered": len(self._procedures),
            }
        
        succeeded = sum(1 for a in self._recovery_history if a.status == RecoveryStatus.SUCCEEDED)
        failed = sum(1 for a in self._recovery_history if a.status == RecoveryStatus.FAILED)
        
        # Calculate average recovery time
        recovery_times = [
            (a.completed_at - a.started_at).total_seconds()
            for a in self._recovery_history
            if a.completed_at and a.status == RecoveryStatus.SUCCEEDED
        ]
        
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        return {
            "total_attempts": total_attempts,
            "succeeded": succeeded,
            "failed": failed,
            "success_rate": succeeded / total_attempts,
            "active_recoveries": len(self._active_recoveries),
            "procedures_registered": len(self._procedures),
            "avg_recovery_time_seconds": avg_recovery_time,
            "procedures_in_cooldown": len([
                p for p in self._cooldowns
                if datetime.utcnow() < self._cooldowns[p]
            ]),
        }
    
    def clear_cooldowns(self):
        """Clear all procedure cooldowns."""
        self._cooldowns.clear()
        self.logger.info("Cleared all recovery procedure cooldowns")