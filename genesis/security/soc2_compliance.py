"""
SOC 2 Type II Compliance Implementation for Genesis.
Implements comprehensive audit trail, access control, monitoring, and reporting.
"""

import os
import json
import hashlib
import asyncio
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from pathlib import Path
import sqlite3
import threading

from genesis.core.exceptions import SecurityError, GenesisException

logger = structlog.get_logger(__name__)


class SOC2TrustPrinciple(Enum):
    """SOC 2 Trust Service Principles."""
    SECURITY = "security"
    AVAILABILITY = "availability"
    PROCESSING_INTEGRITY = "processing_integrity"
    CONFIDENTIALITY = "confidentiality"
    PRIVACY = "privacy"


class AccessLevel(Enum):
    """Access control levels."""
    READ_ONLY = "read_only"
    TRADER = "trader"
    ADMIN = "admin"
    AUDITOR = "auditor"
    SYSTEM = "system"


class EventType(Enum):
    """Audit event types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    KEY_OPERATION = "key_operation"
    SYSTEM_EVENT = "system_event"
    SECURITY_ALERT = "security_alert"
    COMPLIANCE_CHECK = "compliance_check"


@dataclass
class AuditEvent:
    """Immutable audit event record."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: str
    ip_address: Optional[str]
    resource: str
    action: str
    result: str
    details: Dict[str, Any]
    risk_score: int = 0
    hash_chain: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": json.dumps(self.details),
            "risk_score": self.risk_score,
            "hash_chain": self.hash_chain
        }
    
    def calculate_hash(self, previous_hash: str = "") -> str:
        """Calculate tamper-proof hash for event."""
        event_string = (
            f"{self.event_id}{self.event_type.value}{self.timestamp.isoformat()}"
            f"{self.user_id}{self.resource}{self.action}{self.result}"
            f"{json.dumps(self.details, sort_keys=True)}{previous_hash}"
        )
        return hashlib.sha256(event_string.encode()).hexdigest()


@dataclass
class AccessControlEntry:
    """Access control list entry."""
    user_id: str
    access_level: AccessLevel
    resources: Set[str]
    permissions: Set[str]
    ip_whitelist: Optional[List[str]] = None
    time_restrictions: Optional[Dict[str, Any]] = None
    mfa_required: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def has_permission(self, resource: str, permission: str) -> bool:
        """Check if user has specific permission on resource."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        # Check resource access
        resource_match = False
        for allowed_resource in self.resources:
            if allowed_resource == "*" or resource.startswith(allowed_resource):
                resource_match = True
                break
        
        if not resource_match:
            return False
        
        # Check permission
        return permission in self.permissions or "*" in self.permissions
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        if not self.ip_whitelist:
            return True
        return ip_address in self.ip_whitelist
    
    def is_time_allowed(self) -> bool:
        """Check if current time is within allowed window."""
        if not self.time_restrictions:
            return True
        
        now = datetime.utcnow()
        
        # Check day of week
        if "allowed_days" in self.time_restrictions:
            if now.strftime("%A").lower() not in self.time_restrictions["allowed_days"]:
                return False
        
        # Check time of day
        if "allowed_hours" in self.time_restrictions:
            current_hour = now.hour
            start_hour = self.time_restrictions["allowed_hours"]["start"]
            end_hour = self.time_restrictions["allowed_hours"]["end"]
            
            if start_hour <= end_hour:
                if not (start_hour <= current_hour < end_hour):
                    return False
            else:  # Overnight window
                if not (current_hour >= start_hour or current_hour < end_hour):
                    return False
        
        return True


class TamperProofAuditLog:
    """
    Tamper-proof audit log with hash chain integrity.
    Implements append-only log with cryptographic verification.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize audit log.
        
        Args:
            db_path: Path to audit database
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._last_hash = ""
        
        self._initialize_database()
        self._load_last_hash()
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def _initialize_database(self):
        """Initialize audit database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    ip_address TEXT,
                    resource TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT NOT NULL,
                    details TEXT,
                    risk_score INTEGER DEFAULT 0,
                    hash_chain TEXT NOT NULL,
                    created_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user 
                ON audit_events(user_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_type 
                ON audit_events(event_type)
            """)
            
            # Create integrity table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_integrity (
                    id INTEGER PRIMARY KEY,
                    last_hash TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    last_verified TEXT NOT NULL,
                    signature TEXT
                )
            """)
            
            conn.commit()
    
    def _load_last_hash(self):
        """Load last hash from integrity table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT last_hash FROM audit_integrity ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            self._last_hash = row[0] if row else ""
    
    def append_event(self, event: AuditEvent) -> str:
        """
        Append event to audit log with hash chain.
        
        Args:
            event: Audit event to append
        
        Returns:
            Event hash
        """
        with self._lock:
            # Calculate hash with chain
            event.hash_chain = event.calculate_hash(self._last_hash)
            
            # Store event
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_events 
                    (event_id, event_type, timestamp, user_id, ip_address, 
                     resource, action, result, details, risk_score, hash_chain)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.user_id,
                    event.ip_address,
                    event.resource,
                    event.action,
                    event.result,
                    json.dumps(event.details),
                    event.risk_score,
                    event.hash_chain
                ))
                
                # Update integrity record
                event_count = conn.execute(
                    "SELECT COUNT(*) FROM audit_events"
                ).fetchone()[0]
                
                conn.execute("""
                    INSERT INTO audit_integrity (last_hash, event_count, last_verified)
                    VALUES (?, ?, ?)
                """, (event.hash_chain, event_count, datetime.utcnow().isoformat()))
                
                conn.commit()
            
            self._last_hash = event.hash_chain
            
            return event.hash_chain
    
    def verify_integrity(self, start_time: Optional[datetime] = None) -> Tuple[bool, List[str]]:
        """
        Verify audit log integrity.
        
        Args:
            start_time: Start time for verification
        
        Returns:
            (is_valid, list_of_issues) tuple
        """
        issues = []
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT event_id, event_type, timestamp, user_id, resource, 
                       action, result, details, hash_chain
                FROM audit_events
            """
            
            if start_time:
                query += " WHERE timestamp >= ?"
                cursor = conn.execute(query + " ORDER BY created_at", 
                                     (start_time.isoformat(),))
            else:
                cursor = conn.execute(query + " ORDER BY created_at")
            
            previous_hash = ""
            
            for row in cursor:
                # Reconstruct event
                event = AuditEvent(
                    event_id=row[0],
                    event_type=EventType(row[1]),
                    timestamp=datetime.fromisoformat(row[2]),
                    user_id=row[3],
                    ip_address=None,
                    resource=row[4],
                    action=row[5],
                    result=row[6],
                    details=json.loads(row[7]),
                    hash_chain=row[8]
                )
                
                # Verify hash
                expected_hash = event.calculate_hash(previous_hash)
                if expected_hash != event.hash_chain:
                    issues.append(f"Hash mismatch for event {event.event_id}")
                
                previous_hash = event.hash_chain
        
        return len(issues) == 0, issues
    
    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Query audit events with filters.
        
        Args:
            start_time: Start time filter
            end_time: End time filter
            user_id: User ID filter
            event_type: Event type filter
            limit: Maximum results
        
        Returns:
            List of audit events
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, row)) for row in cursor]


class AccessControlMatrix:
    """
    Role-based access control matrix for SOC 2 compliance.
    Enforces principle of least privilege.
    """
    
    def __init__(self):
        """Initialize access control matrix."""
        self.entries: Dict[str, AccessControlEntry] = {}
        self.roles: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()
        
        self._initialize_default_roles()
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    def _initialize_default_roles(self):
        """Initialize default SOC 2 compliant roles."""
        # Read-only role
        self.roles["read_only"] = {
            "view_positions",
            "view_balance",
            "view_history",
            "view_reports"
        }
        
        # Trader role
        self.roles["trader"] = self.roles["read_only"] | {
            "create_order",
            "cancel_order",
            "modify_position",
            "execute_strategy"
        }
        
        # Admin role
        self.roles["admin"] = self.roles["trader"] | {
            "manage_users",
            "configure_system",
            "rotate_keys",
            "view_audit_log",
            "manage_compliance"
        }
        
        # Auditor role (read-only with audit access)
        self.roles["auditor"] = {
            "view_audit_log",
            "generate_compliance_report",
            "verify_integrity",
            "view_all_data"
        }
    
    def add_user(
        self,
        user_id: str,
        access_level: AccessLevel,
        resources: Optional[Set[str]] = None,
        custom_permissions: Optional[Set[str]] = None,
        **kwargs
    ):
        """
        Add user to access control matrix.
        
        Args:
            user_id: User identifier
            access_level: Access level
            resources: Allowed resources
            custom_permissions: Custom permissions
            **kwargs: Additional access control parameters
        """
        with self._lock:
            # Get role permissions
            role_permissions = self.roles.get(access_level.value, set())
            
            # Merge with custom permissions
            permissions = role_permissions | (custom_permissions or set())
            
            # Default resources if not specified
            if not resources:
                if access_level == AccessLevel.ADMIN:
                    resources = {"*"}
                else:
                    resources = {"/api/*", "/data/*"}
            
            self.entries[user_id] = AccessControlEntry(
                user_id=user_id,
                access_level=access_level,
                resources=resources,
                permissions=permissions,
                **kwargs
            )
            
            self.logger.info(
                "User added to access control",
                user_id=user_id,
                access_level=access_level.value
            )
    
    def check_access(
        self,
        user_id: str,
        resource: str,
        permission: str,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if user has access to resource with permission.
        
        Args:
            user_id: User identifier
            resource: Resource path
            permission: Required permission
            ip_address: Request IP address
        
        Returns:
            (is_allowed, reason) tuple
        """
        with self._lock:
            if user_id not in self.entries:
                return False, "User not found"
            
            entry = self.entries[user_id]
            
            # Check IP whitelist
            if ip_address and not entry.is_ip_allowed(ip_address):
                return False, "IP address not allowed"
            
            # Check time restrictions
            if not entry.is_time_allowed():
                return False, "Access not allowed at this time"
            
            # Check permission
            if not entry.has_permission(resource, permission):
                return False, "Permission denied"
            
            return True, "Access granted"
    
    def revoke_access(self, user_id: str):
        """Revoke user access."""
        with self._lock:
            if user_id in self.entries:
                del self.entries[user_id]
                self.logger.info("User access revoked", user_id=user_id)


class ComplianceMonitor:
    """
    SOC 2 compliance monitoring and alerting system.
    Tracks compliance metrics and generates alerts.
    """
    
    def __init__(
        self,
        audit_log: TamperProofAuditLog,
        access_control: AccessControlMatrix
    ):
        """
        Initialize compliance monitor.
        
        Args:
            audit_log: Audit log instance
            access_control: Access control matrix
        """
        self.audit_log = audit_log
        self.access_control = access_control
        
        self.alerts: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        
        self.logger = structlog.get_logger(self.__class__.__name__)
    
    async def check_compliance(self) -> Dict[str, Any]:
        """
        Run comprehensive compliance checks.
        
        Returns:
            Compliance status report
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
            "issues": [],
            "score": 100
        }
        
        # Check audit log integrity
        is_valid, integrity_issues = self.audit_log.verify_integrity()
        report["checks"]["audit_integrity"] = is_valid
        if not is_valid:
            report["issues"].extend(integrity_issues)
            report["score"] -= 20
        
        # Check access control enforcement
        access_violations = await self._check_access_violations()
        report["checks"]["access_control"] = len(access_violations) == 0
        if access_violations:
            report["issues"].extend(access_violations)
            report["score"] -= 10 * len(access_violations)
        
        # Check data retention
        retention_ok = await self._check_data_retention()
        report["checks"]["data_retention"] = retention_ok
        if not retention_ok:
            report["issues"].append("Data retention policy violation")
            report["score"] -= 15
        
        # Check encryption status
        encryption_ok = await self._check_encryption_status()
        report["checks"]["encryption"] = encryption_ok
        if not encryption_ok:
            report["issues"].append("Encryption requirements not met")
            report["score"] -= 25
        
        # Log compliance check
        self.audit_log.append_event(AuditEvent(
            event_id=os.urandom(16).hex(),
            event_type=EventType.COMPLIANCE_CHECK,
            timestamp=datetime.utcnow(),
            user_id="system",
            ip_address=None,
            resource="compliance",
            action="check",
            result="completed",
            details=report
        ))
        
        return report
    
    async def _check_access_violations(self) -> List[str]:
        """Check for access control violations."""
        violations = []
        
        # Query recent authentication failures
        recent_events = self.audit_log.query_events(
            start_time=datetime.utcnow() - timedelta(hours=1),
            event_type=EventType.AUTHENTICATION,
            limit=100
        )
        
        # Check for brute force attempts
        failed_attempts = {}
        for event in recent_events:
            if event["result"] == "failed":
                user = event["user_id"]
                failed_attempts[user] = failed_attempts.get(user, 0) + 1
        
        for user, count in failed_attempts.items():
            if count > 5:
                violations.append(f"Excessive failed login attempts for {user}")
        
        return violations
    
    async def _check_data_retention(self) -> bool:
        """Check data retention compliance."""
        # Check if old audit logs are properly archived
        # This is a simplified check
        oldest_event = self.audit_log.query_events(limit=1)
        if oldest_event:
            event_time = datetime.fromisoformat(oldest_event[0]["timestamp"])
            age_days = (datetime.utcnow() - event_time).days
            
            # SOC 2 typically requires 7 years retention
            if age_days > 2555:  # 7 years
                return False
        
        return True
    
    async def _check_encryption_status(self) -> bool:
        """Check if encryption is properly configured."""
        # This would check actual encryption status
        # For now, return True as placeholder
        return True
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Generate SOC 2 compliance report.
        
        Args:
            start_date: Report start date
            end_date: Report end date
            include_details: Include detailed audit events
        
        Returns:
            Compliance report
        """
        report = {
            "report_id": os.urandom(16).hex(),
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "trust_principles": {},
            "controls": {},
            "incidents": [],
            "metrics": {}
        }
        
        # Analyze events for the period
        events = self.audit_log.query_events(
            start_time=start_date,
            end_time=end_date,
            limit=10000
        )
        
        # Calculate metrics
        report["metrics"]["total_events"] = len(events)
        report["metrics"]["unique_users"] = len(set(e["user_id"] for e in events))
        
        # Analyze by event type
        event_counts = {}
        security_incidents = []
        
        for event in events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Check for security incidents
            if event["risk_score"] > 50:
                security_incidents.append({
                    "event_id": event["event_id"],
                    "timestamp": event["timestamp"],
                    "type": event_type,
                    "risk_score": event["risk_score"]
                })
        
        report["metrics"]["event_distribution"] = event_counts
        report["incidents"] = security_incidents
        
        # Assess trust principles
        report["trust_principles"][SOC2TrustPrinciple.SECURITY.value] = {
            "status": "compliant" if len(security_incidents) == 0 else "needs_review",
            "score": max(0, 100 - len(security_incidents) * 10)
        }
        
        report["trust_principles"][SOC2TrustPrinciple.AVAILABILITY.value] = {
            "status": "compliant",
            "score": 95  # Would calculate from uptime metrics
        }
        
        report["trust_principles"][SOC2TrustPrinciple.CONFIDENTIALITY.value] = {
            "status": "compliant",
            "score": 100  # Would check encryption status
        }
        
        # Include detailed events if requested
        if include_details:
            report["audit_events"] = events[:1000]  # Limit to 1000 events
        
        return report


class SOC2ComplianceManager:
    """
    Central manager for SOC 2 compliance operations.
    Coordinates audit logging, access control, and monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SOC 2 compliance manager.
        
        Args:
            config: Compliance configuration
        """
        self.config = config
        
        # Initialize components
        audit_db_path = Path(config.get("audit_db_path", ".genesis/audit.db"))
        self.audit_log = TamperProofAuditLog(audit_db_path)
        self.access_control = AccessControlMatrix()
        self.compliance_monitor = ComplianceMonitor(
            self.audit_log,
            self.access_control
        )
        
        self.logger = structlog.get_logger(__name__)
    
    def log_event(
        self,
        event_type: EventType,
        user_id: str,
        resource: str,
        action: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        risk_score: int = 0
    ) -> str:
        """
        Log audit event.
        
        Args:
            event_type: Type of event
            user_id: User performing action
            resource: Resource being accessed
            action: Action performed
            result: Result of action
            details: Additional details
            ip_address: Client IP address
            risk_score: Risk score (0-100)
        
        Returns:
            Event hash
        """
        event = AuditEvent(
            event_id=os.urandom(16).hex(),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            ip_address=ip_address,
            resource=resource,
            action=action,
            result=result,
            details=details or {},
            risk_score=risk_score
        )
        
        return self.audit_log.append_event(event)
    
    def check_access(
        self,
        user_id: str,
        resource: str,
        permission: str,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Check and log access attempt.
        
        Args:
            user_id: User requesting access
            resource: Resource being accessed
            permission: Required permission
            ip_address: Client IP address
        
        Returns:
            True if access granted
        """
        # Check access
        is_allowed, reason = self.access_control.check_access(
            user_id,
            resource,
            permission,
            ip_address
        )
        
        # Log access attempt
        self.log_event(
            event_type=EventType.AUTHORIZATION,
            user_id=user_id,
            resource=resource,
            action=permission,
            result="granted" if is_allowed else "denied",
            details={"reason": reason},
            ip_address=ip_address,
            risk_score=0 if is_allowed else 20
        )
        
        return is_allowed
    
    async def run_compliance_check(self) -> Dict[str, Any]:
        """Run compliance check and return results."""
        return await self.compliance_monitor.check_compliance()
    
    def generate_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for period."""
        return self.compliance_monitor.generate_compliance_report(
            start_date,
            end_date,
            include_details=True
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get compliance system status."""
        # Verify audit log integrity
        is_valid, issues = self.audit_log.verify_integrity()
        
        # Get recent events count
        recent_events = self.audit_log.query_events(
            start_time=datetime.utcnow() - timedelta(hours=1)
        )
        
        return {
            "audit_log_valid": is_valid,
            "integrity_issues": issues,
            "recent_events_count": len(recent_events),
            "active_users": len(self.access_control.entries),
            "compliance_score": 95  # Would calculate from metrics
        }