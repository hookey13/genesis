#!/usr/bin/env python3
"""Generate audit reports from audit logs.

This script generates comprehensive audit reports for security analysis
and compliance requirements.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import structlog

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.security.audit_logger import (
    AuditLogger,
    AuditEventType,
    AuditEvent
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class AuditReportGenerator:
    """Generates audit reports from audit logs."""
    
    def __init__(self, audit_logger: AuditLogger):
        """Initialize report generator.
        
        Args:
            audit_logger: Audit logger instance
        """
        self.audit_logger = audit_logger
    
    def generate_summary_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate summary audit report.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Report dictionary
        """
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=1)
        
        # Get statistics
        stats = self.audit_logger.get_statistics(start_time, end_time)
        
        # Get all events for detailed analysis
        events = self.audit_logger.search_events(
            start_time=start_time,
            end_time=end_time
        )
        
        # Analyze security events
        security_analysis = self._analyze_security_events(events)
        
        # Generate report
        report = {
            "report_type": "audit_summary",
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "statistics": stats,
            "security_analysis": security_analysis,
            "top_users": self._get_top_users(events),
            "top_ips": self._get_top_ips(events),
            "failed_attempts": self._get_failed_attempts(events),
            "suspicious_activity": self._detect_suspicious_activity(events)
        }
        
        return report
    
    def _analyze_security_events(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze security-related events.
        
        Args:
            events: List of audit events
            
        Returns:
            Security analysis
        """
        analysis = {
            "auth_failures": 0,
            "permission_denials": 0,
            "rate_limit_violations": 0,
            "ip_blocks": 0,
            "key_rotations": 0,
            "config_changes": 0
        }
        
        for event in events:
            if event.event_type == AuditEventType.AUTH_FAILURE:
                analysis["auth_failures"] += 1
            elif event.event_type == AuditEventType.PERMISSION_DENIED:
                analysis["permission_denials"] += 1
            elif event.event_type == AuditEventType.RATE_LIMIT_EXCEEDED:
                analysis["rate_limit_violations"] += 1
            elif event.event_type == AuditEventType.IP_BLOCKED:
                analysis["ip_blocks"] += 1
            elif event.event_type == AuditEventType.KEY_ROTATED:
                analysis["key_rotations"] += 1
            elif event.event_type == AuditEventType.CONFIG_CHANGED:
                analysis["config_changes"] += 1
        
        return analysis
    
    def _get_top_users(self, events: List[AuditEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most active users.
        
        Args:
            events: List of audit events
            limit: Number of users to return
            
        Returns:
            List of top users with activity counts
        """
        user_counts = {}
        
        for event in events:
            if event.user_id:
                if event.user_id not in user_counts:
                    user_counts[event.user_id] = {
                        "user_id": event.user_id,
                        "total_events": 0,
                        "success_count": 0,
                        "failure_count": 0
                    }
                
                user_counts[event.user_id]["total_events"] += 1
                if event.result == "success":
                    user_counts[event.user_id]["success_count"] += 1
                else:
                    user_counts[event.user_id]["failure_count"] += 1
        
        # Sort by total events
        sorted_users = sorted(
            user_counts.values(),
            key=lambda x: x["total_events"],
            reverse=True
        )
        
        return sorted_users[:limit]
    
    def _get_top_ips(self, events: List[AuditEvent], limit: int = 10) -> List[Dict[str, Any]]:
        """Get most active IP addresses.
        
        Args:
            events: List of audit events
            limit: Number of IPs to return
            
        Returns:
            List of top IPs with activity counts
        """
        ip_counts = {}
        
        for event in events:
            if event.ip_address:
                if event.ip_address not in ip_counts:
                    ip_counts[event.ip_address] = {
                        "ip_address": event.ip_address,
                        "total_events": 0,
                        "unique_users": set(),
                        "failure_count": 0
                    }
                
                ip_counts[event.ip_address]["total_events"] += 1
                if event.user_id:
                    ip_counts[event.ip_address]["unique_users"].add(event.user_id)
                if event.result == "failure":
                    ip_counts[event.ip_address]["failure_count"] += 1
        
        # Convert sets to counts
        for ip_data in ip_counts.values():
            ip_data["unique_users"] = len(ip_data["unique_users"])
        
        # Sort by total events
        sorted_ips = sorted(
            ip_counts.values(),
            key=lambda x: x["total_events"],
            reverse=True
        )
        
        return sorted_ips[:limit]
    
    def _get_failed_attempts(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Get failed authentication and authorization attempts.
        
        Args:
            events: List of audit events
            
        Returns:
            List of failed attempts
        """
        failed_attempts = []
        
        for event in events:
            if event.result == "failure" and event.event_type in [
                AuditEventType.AUTH_FAILURE,
                AuditEventType.PERMISSION_DENIED
            ]:
                failed_attempts.append({
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "user_id": event.user_id,
                    "ip_address": event.ip_address,
                    "resource": event.resource,
                    "action": event.action
                })
        
        return failed_attempts
    
    def _detect_suspicious_activity(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Detect potentially suspicious activity patterns.
        
        Args:
            events: List of audit events
            
        Returns:
            List of suspicious patterns detected
        """
        suspicious_patterns = []
        
        # Pattern 1: Multiple auth failures from same IP
        ip_auth_failures = {}
        for event in events:
            if event.event_type == AuditEventType.AUTH_FAILURE and event.ip_address:
                if event.ip_address not in ip_auth_failures:
                    ip_auth_failures[event.ip_address] = []
                ip_auth_failures[event.ip_address].append(event.timestamp)
        
        for ip, timestamps in ip_auth_failures.items():
            if len(timestamps) >= 5:
                suspicious_patterns.append({
                    "pattern": "multiple_auth_failures",
                    "ip_address": ip,
                    "count": len(timestamps),
                    "first_attempt": min(timestamps).isoformat(),
                    "last_attempt": max(timestamps).isoformat()
                })
        
        # Pattern 2: Rapid permission denials (possible enumeration)
        user_permission_denials = {}
        for event in events:
            if event.event_type == AuditEventType.PERMISSION_DENIED and event.user_id:
                if event.user_id not in user_permission_denials:
                    user_permission_denials[event.user_id] = []
                user_permission_denials[event.user_id].append(event.timestamp)
        
        for user_id, timestamps in user_permission_denials.items():
            if len(timestamps) >= 10:
                # Check if they happened within a short time window
                timestamps_sorted = sorted(timestamps)
                time_span = timestamps_sorted[-1] - timestamps_sorted[0]
                if time_span.total_seconds() < 300:  # 5 minutes
                    suspicious_patterns.append({
                        "pattern": "rapid_permission_denials",
                        "user_id": user_id,
                        "count": len(timestamps),
                        "time_span_seconds": time_span.total_seconds()
                    })
        
        # Pattern 3: IP hopping (same user from multiple IPs)
        user_ips = {}
        for event in events:
            if event.user_id and event.ip_address:
                if event.user_id not in user_ips:
                    user_ips[event.user_id] = set()
                user_ips[event.user_id].add(event.ip_address)
        
        for user_id, ips in user_ips.items():
            if len(ips) > 3:
                suspicious_patterns.append({
                    "pattern": "ip_hopping",
                    "user_id": user_id,
                    "ip_count": len(ips),
                    "ips": list(ips)
                })
        
        return suspicious_patterns
    
    def generate_compliance_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate compliance-focused audit report.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Compliance report
        """
        events = self.audit_logger.search_events(
            start_time=start_time,
            end_time=end_time
        )
        
        report = {
            "report_type": "compliance_audit",
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "user_activity": self._get_user_activity_summary(events),
            "configuration_changes": self._get_config_changes(events),
            "key_management": self._get_key_management_events(events),
            "access_control": self._get_access_control_summary(events),
            "integrity_check": self.audit_logger.verify_integrity()
        }
        
        return report
    
    def _get_user_activity_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Get user activity summary for compliance.
        
        Args:
            events: List of audit events
            
        Returns:
            User activity summary
        """
        summary = {
            "total_users": len(set(e.user_id for e in events if e.user_id)),
            "authentication_events": sum(1 for e in events if e.event_type in [
                AuditEventType.AUTH_SUCCESS,
                AuditEventType.AUTH_FAILURE,
                AuditEventType.AUTH_LOGOUT
            ]),
            "trading_events": sum(1 for e in events if e.event_type in [
                AuditEventType.ORDER_PLACED,
                AuditEventType.ORDER_CANCELLED,
                AuditEventType.TRADE_EXECUTED
            ])
        }
        return summary
    
    def _get_config_changes(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Get configuration changes for compliance.
        
        Args:
            events: List of audit events
            
        Returns:
            List of configuration changes
        """
        config_changes = []
        
        for event in events:
            if event.event_type in [
                AuditEventType.CONFIG_CHANGED,
                AuditEventType.PERMISSION_CHANGED
            ]:
                config_changes.append({
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "change_type": event.event_type.value,
                    "resource": event.resource,
                    "metadata": event.metadata
                })
        
        return config_changes
    
    def _get_key_management_events(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Get key management events for compliance.
        
        Args:
            events: List of audit events
            
        Returns:
            List of key management events
        """
        key_events = []
        
        for event in events:
            if event.event_type in [
                AuditEventType.KEY_CREATED,
                AuditEventType.KEY_ROTATED,
                AuditEventType.KEY_DELETED
            ]:
                key_events.append({
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type.value,
                    "user_id": event.user_id,
                    "metadata": event.metadata
                })
        
        return key_events
    
    def _get_access_control_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Get access control summary for compliance.
        
        Args:
            events: List of audit events
            
        Returns:
            Access control summary
        """
        summary = {
            "permission_checks": sum(1 for e in events if e.event_type in [
                AuditEventType.PERMISSION_GRANTED,
                AuditEventType.PERMISSION_DENIED
            ]),
            "denials": sum(1 for e in events if e.event_type == AuditEventType.PERMISSION_DENIED),
            "rate_limit_violations": sum(1 for e in events if e.event_type == AuditEventType.RATE_LIMIT_EXCEEDED),
            "ip_blocks": sum(1 for e in events if e.event_type == AuditEventType.IP_BLOCKED)
        }
        return summary


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate audit reports for Project GENESIS"
    )
    
    parser.add_argument(
        "--report-type",
        choices=["summary", "compliance", "security"],
        default="summary",
        help="Type of report to generate"
    )
    
    parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="Number of days to include (if dates not specified)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path (default: stdout)"
    )
    
    parser.add_argument(
        "--verify-integrity",
        action="store_true",
        help="Verify audit log integrity"
    )
    
    args = parser.parse_args()
    
    # Determine time range
    if args.end_date:
        end_time = args.end_date
    else:
        end_time = datetime.now()
    
    if args.start_date:
        start_time = args.start_date
    else:
        start_time = end_time - timedelta(days=args.days)
    
    # Initialize audit logger
    audit_logger = AuditLogger()
    
    # Verify integrity if requested
    if args.verify_integrity:
        if audit_logger.verify_integrity():
            logger.info("Audit log integrity verified")
        else:
            logger.error("Audit log integrity check failed")
            sys.exit(1)
    
    # Generate report
    generator = AuditReportGenerator(audit_logger)
    
    if args.report_type == "summary":
        report = generator.generate_summary_report(start_time, end_time)
    elif args.report_type == "compliance":
        report = generator.generate_compliance_report(start_time, end_time)
    else:  # security
        # For now, use summary report with focus on security
        report = generator.generate_summary_report(start_time, end_time)
    
    # Output report
    report_json = json.dumps(report, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report_json)
        logger.info(f"Report written to {args.output}")
    else:
        print(report_json)


if __name__ == "__main__":
    main()