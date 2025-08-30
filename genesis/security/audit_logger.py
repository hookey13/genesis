"""Comprehensive audit logging system for security events.

Implements tamper-proof audit trail with structured logging,
rotation, and reporting capabilities.
"""

import os
import json
import hashlib
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import asyncio
from collections import deque

logger = structlog.get_logger(__name__)


class AuditEventType(Enum):
    """Types of audit events."""
    
    # Authentication events
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_LOGOUT = "auth_logout"
    
    # Authorization events
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    
    # API key events
    KEY_CREATED = "key_created"
    KEY_ROTATED = "key_rotated"
    KEY_DELETED = "key_deleted"
    KEY_USED = "key_used"
    
    # Configuration changes
    CONFIG_CHANGED = "config_changed"
    PERMISSION_CHANGED = "permission_changed"
    
    # Trading events
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    TRADE_EXECUTED = "trade_executed"
    
    # Security events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    IP_BLOCKED = "ip_blocked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    ip_address: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    result: str  # success, failure
    metadata: Dict[str, Any]
    checksum: Optional[str] = None
    
    def calculate_checksum(self, previous_checksum: Optional[str] = None) -> str:
        """Calculate checksum for tamper detection.
        
        Args:
            previous_checksum: Checksum of previous event for chaining
            
        Returns:
            SHA256 checksum
        """
        # Create a deterministic string representation
        event_str = json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "metadata": self.metadata,
            "previous": previous_checksum or ""
        }, sort_keys=True)
        
        return hashlib.sha256(event_str.encode()).hexdigest()
    
    def to_json(self) -> str:
        """Convert event to JSON string.
        
        Returns:
            JSON representation
        """
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "metadata": self.metadata,
            "checksum": self.checksum
        })


class AuditLogger:
    """Manages audit logging with tamper protection and rotation."""
    
    def __init__(
        self,
        log_dir: str = ".genesis/logs/audit",
        retention_days: int = 90,
        max_file_size_mb: int = 100,
        enable_compression: bool = True,
        enable_tamper_protection: bool = True
    ):
        """Initialize audit logger.
        
        Args:
            log_dir: Directory for audit logs
            retention_days: How long to keep logs
            max_file_size_mb: Maximum log file size before rotation
            enable_compression: Compress rotated logs
            enable_tamper_protection: Enable checksum chaining
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.enable_compression = enable_compression
        self.enable_tamper_protection = enable_tamper_protection
        
        self.current_log_file = None
        self.current_file_size = 0
        self.last_checksum = None
        
        # In-memory buffer for recent events
        self.recent_events = deque(maxlen=1000)
        
        # Initialize log file
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize or open current log file."""
        today = datetime.now().strftime("%Y%m%d")
        log_filename = f"audit_{today}.log"
        self.current_log_file = self.log_dir / log_filename
        
        # Get file size if it exists
        if self.current_log_file.exists():
            self.current_file_size = self.current_log_file.stat().st_size
            
            # Load last checksum if tamper protection enabled
            if self.enable_tamper_protection:
                self._load_last_checksum()
        else:
            self.current_file_size = 0
            self.last_checksum = None
    
    def _load_last_checksum(self):
        """Load the last checksum from current log file."""
        try:
            with open(self.current_log_file, 'r') as f:
                # Read last line
                for line in f:
                    pass
                if line:
                    event_data = json.loads(line)
                    self.last_checksum = event_data.get("checksum")
        except (IOError, json.JSONDecodeError) as e:
            logger.error("Failed to load last checksum", error=str(e))
            self.last_checksum = None
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log an audit event.
        
        Args:
            event_type: Type of event
            user_id: User identifier
            ip_address: Source IP address
            resource: Resource accessed
            action: Action performed
            result: Result of the action
            metadata: Additional event data
        """
        # Create audit event
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            resource=resource,
            action=action,
            result=result,
            metadata=metadata or {}
        )
        
        # Calculate checksum if enabled
        if self.enable_tamper_protection:
            event.checksum = event.calculate_checksum(self.last_checksum)
            self.last_checksum = event.checksum
        
        # Add to recent events buffer
        self.recent_events.append(event)
        
        # Write to file
        await self._write_event(event)
        
        # Check if rotation needed
        if self.current_file_size >= self.max_file_size_bytes:
            await self._rotate_log()
        
        # Log to standard logger as well
        logger.info("Audit event",
                   event_type=event_type.value,
                   user_id=user_id,
                   ip_address=ip_address,
                   resource=resource,
                   action=action,
                   result=result)
    
    async def _write_event(self, event: AuditEvent):
        """Write event to log file.
        
        Args:
            event: Audit event to write
        """
        try:
            # Append to log file
            with open(self.current_log_file, 'a') as f:
                f.write(event.to_json() + '\n')
                self.current_file_size = f.tell()
        except IOError as e:
            logger.error("Failed to write audit event", error=str(e))
            raise
    
    async def _rotate_log(self):
        """Rotate current log file."""
        if not self.current_log_file or not self.current_log_file.exists():
            return
        
        # Generate rotation filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_file = self.log_dir / f"audit_{timestamp}_rotated.log"
        
        # Move current file
        shutil.move(str(self.current_log_file), str(rotated_file))
        
        # Compress if enabled
        if self.enable_compression:
            await self._compress_log(rotated_file)
        
        # Initialize new log file
        self._initialize_log_file()
        
        logger.info("Rotated audit log", 
                   old_file=rotated_file.name,
                   new_file=self.current_log_file.name)
    
    async def _compress_log(self, log_file: Path):
        """Compress a log file.
        
        Args:
            log_file: Log file to compress
        """
        compressed_file = log_file.with_suffix('.log.gz')
        
        try:
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            log_file.unlink()
            
            logger.info("Compressed audit log", 
                       original=log_file.name,
                       compressed=compressed_file.name)
        except IOError as e:
            logger.error("Failed to compress log", 
                        file=log_file.name,
                        error=str(e))
    
    async def cleanup_old_logs(self):
        """Remove logs older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.log_dir.glob("audit_*.log*"):
            # Parse date from filename
            try:
                file_date_str = log_file.stem.split('_')[1]
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    logger.info("Deleted old audit log", file=log_file.name)
            except (IndexError, ValueError):
                # Skip files that don't match expected pattern
                continue
    
    def verify_integrity(self, log_file: Optional[Path] = None) -> bool:
        """Verify integrity of audit log using checksums.
        
        Args:
            log_file: Log file to verify, or current if None
            
        Returns:
            True if log is intact, False if tampered
        """
        if not self.enable_tamper_protection:
            logger.warning("Tamper protection not enabled")
            return True
        
        file_to_check = log_file or self.current_log_file
        if not file_to_check or not file_to_check.exists():
            return True
        
        try:
            previous_checksum = None
            
            with open(file_to_check, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        event_data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON in audit log",
                                   file=file_to_check.name,
                                   line=line_num)
                        return False
                    
                    # Recreate event
                    event = AuditEvent(
                        timestamp=datetime.fromisoformat(event_data["timestamp"]),
                        event_type=AuditEventType(event_data["event_type"]),
                        user_id=event_data.get("user_id"),
                        ip_address=event_data.get("ip_address"),
                        resource=event_data.get("resource"),
                        action=event_data.get("action"),
                        result=event_data["result"],
                        metadata=event_data.get("metadata", {})
                    )
                    
                    # Calculate expected checksum
                    expected_checksum = event.calculate_checksum(previous_checksum)
                    actual_checksum = event_data.get("checksum")
                    
                    if expected_checksum != actual_checksum:
                        logger.error("Checksum mismatch - log may be tampered",
                                   file=file_to_check.name,
                                   line=line_num,
                                   expected=expected_checksum,
                                   actual=actual_checksum)
                        return False
                    
                    previous_checksum = actual_checksum
            
            logger.info("Audit log integrity verified", 
                       file=file_to_check.name)
            return True
            
        except IOError as e:
            logger.error("Failed to verify log integrity",
                        file=file_to_check.name,
                        error=str(e))
            return False
    
    def search_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        result: Optional[str] = None
    ) -> List[AuditEvent]:
        """Search audit events.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user
            ip_address: Filter by IP
            start_time: Start of time range
            end_time: End of time range
            result: Filter by result (success/failure)
            
        Returns:
            List of matching events
        """
        matching_events = []
        
        # Search in recent events first
        for event in self.recent_events:
            if self._event_matches_criteria(
                event, event_type, user_id, ip_address, 
                start_time, end_time, result
            ):
                matching_events.append(event)
        
        # If more events needed, search log files
        # (Implementation would search through rotated logs)
        
        return matching_events
    
    def _event_matches_criteria(
        self,
        event: AuditEvent,
        event_type: Optional[AuditEventType],
        user_id: Optional[str],
        ip_address: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        result: Optional[str]
    ) -> bool:
        """Check if event matches search criteria.
        
        Returns:
            True if event matches all criteria
        """
        if event_type and event.event_type != event_type:
            return False
        if user_id and event.user_id != user_id:
            return False
        if ip_address and event.ip_address != ip_address:
            return False
        if start_time and event.timestamp < start_time:
            return False
        if end_time and event.timestamp > end_time:
            return False
        if result and event.result != result:
            return False
        return True
    
    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit statistics.
        
        Args:
            start_time: Start of period
            end_time: End of period
            
        Returns:
            Statistics dictionary
        """
        events = self.search_events(start_time=start_time, end_time=end_time)
        
        stats = {
            "total_events": len(events),
            "by_type": {},
            "by_result": {"success": 0, "failure": 0},
            "unique_users": set(),
            "unique_ips": set()
        }
        
        for event in events:
            # Count by type
            event_type = event.event_type.value
            stats["by_type"][event_type] = stats["by_type"].get(event_type, 0) + 1
            
            # Count by result
            if event.result in stats["by_result"]:
                stats["by_result"][event.result] += 1
            
            # Collect unique users and IPs
            if event.user_id:
                stats["unique_users"].add(event.user_id)
            if event.ip_address:
                stats["unique_ips"].add(event.ip_address)
        
        # Convert sets to counts
        stats["unique_users"] = len(stats["unique_users"])
        stats["unique_ips"] = len(stats["unique_ips"])
        
        return stats


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger instance.
    
    Returns:
        AuditLogger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


async def log_authentication(
    user_id: str,
    ip_address: str,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None
):
    """Log authentication attempt.
    
    Args:
        user_id: User attempting authentication
        ip_address: Source IP
        success: Whether authentication succeeded
        metadata: Additional data
    """
    logger = get_audit_logger()
    await logger.log_event(
        event_type=AuditEventType.AUTH_SUCCESS if success else AuditEventType.AUTH_FAILURE,
        user_id=user_id,
        ip_address=ip_address,
        result="success" if success else "failure",
        metadata=metadata
    )


async def log_permission_check(
    user_id: str,
    resource: str,
    action: str,
    granted: bool,
    ip_address: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Log permission check.
    
    Args:
        user_id: User requesting permission
        resource: Resource being accessed
        action: Action being performed
        granted: Whether permission was granted
        ip_address: Source IP
        metadata: Additional data
    """
    logger = get_audit_logger()
    await logger.log_event(
        event_type=AuditEventType.PERMISSION_GRANTED if granted else AuditEventType.PERMISSION_DENIED,
        user_id=user_id,
        ip_address=ip_address,
        resource=resource,
        action=action,
        result="success" if granted else "failure",
        metadata=metadata
    )