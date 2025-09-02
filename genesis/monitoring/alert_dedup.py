"""
Intelligent alert deduplication and routing system.

This module implements smart alert deduplication using fingerprinting,
pattern matching, and time-based grouping to reduce alert fatigue.
"""

import asyncio
import hashlib
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import structlog

from genesis.monitoring.alert_channels import Alert, AlertPriority, AlertChannel

logger = structlog.get_logger(__name__)


class GroupingStrategy(Enum):
    """Alert grouping strategies."""
    SERVICE = "service"
    ALERTNAME = "alertname"
    SEVERITY = "severity"
    CUSTOM = "custom"
    COMPOSITE = "composite"


@dataclass
class AlertFingerprint:
    """Unique fingerprint for alert deduplication."""
    hash: str
    components: Dict[str, str]
    timestamp: datetime
    
    @classmethod
    def generate(cls, alert: Alert, fields: List[str]) -> 'AlertFingerprint':
        """Generate fingerprint from alert fields."""
        components = {}
        
        for field in fields:
            if field == "name":
                components[field] = alert.name
            elif field == "service":
                components[field] = alert.service
            elif field == "severity":
                components[field] = alert.severity
            elif field == "source":
                components[field] = alert.source
            elif field.startswith("label:"):
                label_name = field.replace("label:", "")
                components[field] = str(alert.labels.get(label_name, ""))
        
        # Create hash from components
        hash_input = "|".join(f"{k}:{v}" for k, v in sorted(components.items()))
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        return cls(
            hash=hash_value,
            components=components,
            timestamp=datetime.utcnow()
        )


@dataclass
class AlertGroup:
    """Group of related alerts."""
    id: str
    strategy: GroupingStrategy
    fingerprint: str
    alerts: List[Alert] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    count: int = 0
    suppressed_count: int = 0
    
    def add_alert(self, alert: Alert, suppress: bool = False) -> None:
        """Add alert to group."""
        self.last_seen = datetime.utcnow()
        self.count += 1
        
        if suppress:
            self.suppressed_count += 1
        else:
            self.alerts.append(alert)
    
    def should_suppress(self, max_alerts: int = 10) -> bool:
        """Check if new alerts should be suppressed."""
        return len(self.alerts) >= max_alerts


@dataclass
class RoutingRule:
    """Alert routing rule."""
    name: str
    priority: int
    match_conditions: Dict[str, Any]
    channels: List[AlertChannel]
    transformations: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[int] = None  # Max alerts per minute
    
    def matches(self, alert: Alert) -> bool:
        """Check if alert matches this rule."""
        for field, pattern in self.match_conditions.items():
            if field == "service":
                if not self._match_pattern(alert.service, pattern):
                    return False
            elif field == "severity":
                if not self._match_pattern(alert.severity, pattern):
                    return False
            elif field == "priority":
                if not self._match_pattern(alert.priority.value, pattern):
                    return False
            elif field.startswith("label:"):
                label_name = field.replace("label:", "")
                label_value = str(alert.labels.get(label_name, ""))
                if not self._match_pattern(label_value, pattern):
                    return False
        return True
    
    def _match_pattern(self, value: str, pattern: Any) -> bool:
        """Match value against pattern (string, regex, or list)."""
        if isinstance(pattern, str):
            if pattern.startswith("~"):
                # Regex pattern
                return bool(re.match(pattern[1:], value))
            else:
                # Exact match
                return value == pattern
        elif isinstance(pattern, list):
            # Match any in list
            return value in pattern
        return False


class AlertDeduplicator:
    """
    Intelligent alert deduplication and routing engine.
    
    Features:
    - Fingerprint-based deduplication
    - Time-window grouping
    - Pattern-based suppression
    - Smart routing with priorities
    """
    
    def __init__(
        self,
        dedup_window_minutes: int = 5,
        group_wait_seconds: int = 30,
        group_interval_seconds: int = 300
    ):
        self.dedup_window = timedelta(minutes=dedup_window_minutes)
        self.group_wait = timedelta(seconds=group_wait_seconds)
        self.group_interval = timedelta(seconds=group_interval_seconds)
        
        # Alert groups by fingerprint
        self.groups: Dict[str, AlertGroup] = {}
        
        # Recently seen alerts for deduplication
        self.recent_alerts: Dict[str, datetime] = {}
        
        # Routing rules
        self.routing_rules: List[RoutingRule] = []
        
        # Suppression patterns
        self.suppression_patterns: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            "total_received": 0,
            "deduplicated": 0,
            "suppressed": 0,
            "routed": 0,
            "grouped": 0
        }
        
        # Background cleanup task
        self._cleanup_task = None
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize the deduplicator."""
        self._load_default_rules()
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Alert deduplicator initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the deduplicator."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    def _load_default_rules(self) -> None:
        """Load default routing rules."""
        # Critical alerts go to PagerDuty
        self.routing_rules.append(RoutingRule(
            name="critical_to_pagerduty",
            priority=100,
            match_conditions={"severity": "critical"},
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK],
            rate_limit=10
        ))
        
        # High priority trading alerts
        self.routing_rules.append(RoutingRule(
            name="trading_high_priority",
            priority=90,
            match_conditions={
                "service": ["trading_api", "order_executor"],
                "priority": ["critical", "high"]
            },
            channels=[AlertChannel.PAGERDUTY, AlertChannel.SLACK],
            rate_limit=20
        ))
        
        # Database alerts
        self.routing_rules.append(RoutingRule(
            name="database_alerts",
            priority=80,
            match_conditions={"service": "~database.*"},
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            rate_limit=30
        ))
        
        # Warning alerts to Slack
        self.routing_rules.append(RoutingRule(
            name="warnings_to_slack",
            priority=50,
            match_conditions={"severity": "warning"},
            channels=[AlertChannel.SLACK],
            rate_limit=60
        ))
        
        # Info alerts to email
        self.routing_rules.append(RoutingRule(
            name="info_to_email",
            priority=10,
            match_conditions={"severity": "info"},
            channels=[AlertChannel.EMAIL],
            rate_limit=30
        ))
        
        # Sort by priority
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
    
    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add a custom routing rule."""
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info("Added routing rule", rule_name=rule.name)
    
    def add_suppression_pattern(self, pattern: Dict[str, Any]) -> None:
        """Add a suppression pattern."""
        self.suppression_patterns.append(pattern)
        logger.info("Added suppression pattern", pattern=pattern)
    
    async def process_alert(self, alert: Alert) -> Tuple[bool, Optional[AlertGroup], List[AlertChannel]]:
        """
        Process an alert through deduplication and routing.
        
        Returns:
            Tuple of (should_send, alert_group, channels)
        """
        self.stats["total_received"] += 1
        
        # Check if alert should be suppressed
        if self._should_suppress(alert):
            self.stats["suppressed"] += 1
            logger.info(
                "Alert suppressed",
                alert_id=alert.id,
                alert_name=alert.name,
                service=alert.service
            )
            return False, None, []
        
        # Generate fingerprint for deduplication
        fingerprint = self._generate_fingerprint(alert)
        
        # Check for duplicate
        if self._is_duplicate(fingerprint, alert):
            self.stats["deduplicated"] += 1
            logger.debug(
                "Alert deduplicated",
                alert_id=alert.id,
                fingerprint=fingerprint
            )
            return False, None, []
        
        # Find or create group
        group = self._get_or_create_group(alert, fingerprint)
        
        # Check if group should suppress
        if group.should_suppress():
            group.add_alert(alert, suppress=True)
            self.stats["suppressed"] += 1
            logger.info(
                "Alert suppressed by group limit",
                alert_id=alert.id,
                group_id=group.id,
                group_size=len(group.alerts)
            )
            return False, group, []
        
        # Add to group
        group.add_alert(alert)
        self.stats["grouped"] += 1
        
        # Determine routing
        channels = self._route_alert(alert)
        if channels:
            self.stats["routed"] += 1
        
        # Record for deduplication
        self.recent_alerts[fingerprint] = datetime.utcnow()
        
        return True, group, channels
    
    def _generate_fingerprint(self, alert: Alert) -> str:
        """Generate fingerprint for alert."""
        # Default fingerprinting strategy
        fields = ["name", "service", "severity"]
        
        # Add critical labels to fingerprint
        if "instance" in alert.labels:
            fields.append("label:instance")
        if "job" in alert.labels:
            fields.append("label:job")
        
        fp = AlertFingerprint.generate(alert, fields)
        return fp.hash
    
    def _is_duplicate(self, fingerprint: str, alert: Alert) -> bool:
        """Check if alert is a duplicate within dedup window."""
        if fingerprint in self.recent_alerts:
            last_seen = self.recent_alerts[fingerprint]
            if datetime.utcnow() - last_seen < self.dedup_window:
                return True
        return False
    
    def _should_suppress(self, alert: Alert) -> bool:
        """Check if alert matches suppression patterns."""
        for pattern in self.suppression_patterns:
            matches = True
            
            for field, value in pattern.items():
                if field == "service" and alert.service != value:
                    matches = False
                    break
                elif field == "severity" and alert.severity != value:
                    matches = False
                    break
                elif field == "name_regex":
                    if not re.match(value, alert.name):
                        matches = False
                        break
                elif field.startswith("label:"):
                    label_name = field.replace("label:", "")
                    if alert.labels.get(label_name) != value:
                        matches = False
                        break
            
            if matches:
                return True
        
        return False
    
    def _get_or_create_group(self, alert: Alert, fingerprint: str) -> AlertGroup:
        """Get existing group or create new one."""
        if fingerprint in self.groups:
            group = self.groups[fingerprint]
            # Check if group is still active
            if datetime.utcnow() - group.last_seen < self.group_interval:
                return group
        
        # Create new group
        group = AlertGroup(
            id=f"group_{fingerprint}_{int(datetime.utcnow().timestamp())}",
            strategy=GroupingStrategy.COMPOSITE,
            fingerprint=fingerprint
        )
        self.groups[fingerprint] = group
        
        return group
    
    def _route_alert(self, alert: Alert) -> List[AlertChannel]:
        """Determine which channels to route alert to."""
        channels = []
        
        for rule in self.routing_rules:
            if rule.matches(alert):
                channels.extend(rule.channels)
                # Apply transformations if any
                if rule.transformations:
                    self._apply_transformations(alert, rule.transformations)
                break  # Use first matching rule
        
        # Deduplicate channels
        return list(set(channels))
    
    def _apply_transformations(self, alert: Alert, transformations: Dict[str, Any]) -> None:
        """Apply transformations to alert."""
        for field, value in transformations.items():
            if field == "priority":
                alert.priority = AlertPriority(value)
            elif field == "add_label":
                alert.labels.update(value)
            elif field == "add_annotation":
                alert.annotations.update(value)
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up old data."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                now = datetime.utcnow()
                
                # Clean up old groups
                expired_groups = []
                for fingerprint, group in self.groups.items():
                    if now - group.last_seen > self.group_interval:
                        expired_groups.append(fingerprint)
                
                for fingerprint in expired_groups:
                    del self.groups[fingerprint]
                
                if expired_groups:
                    logger.debug(
                        "Cleaned up expired alert groups",
                        count=len(expired_groups)
                    )
                
                # Clean up old dedup records
                expired_alerts = []
                for fingerprint, last_seen in self.recent_alerts.items():
                    if now - last_seen > self.dedup_window:
                        expired_alerts.append(fingerprint)
                
                for fingerprint in expired_alerts:
                    del self.recent_alerts[fingerprint]
                
                if expired_alerts:
                    logger.debug(
                        "Cleaned up expired dedup records",
                        count=len(expired_alerts)
                    )
                
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        total = self.stats["total_received"]
        
        return {
            **self.stats,
            "dedup_ratio": self.stats["deduplicated"] / total if total > 0 else 0,
            "suppression_ratio": self.stats["suppressed"] / total if total > 0 else 0,
            "routing_ratio": self.stats["routed"] / total if total > 0 else 0,
            "active_groups": len(self.groups),
            "dedup_cache_size": len(self.recent_alerts)
        }
    
    def get_active_groups(self) -> List[Dict[str, Any]]:
        """Get information about active alert groups."""
        groups_info = []
        
        for group in self.groups.values():
            groups_info.append({
                "id": group.id,
                "fingerprint": group.fingerprint,
                "strategy": group.strategy.value,
                "alert_count": len(group.alerts),
                "total_count": group.count,
                "suppressed_count": group.suppressed_count,
                "first_seen": group.first_seen.isoformat(),
                "last_seen": group.last_seen.isoformat(),
                "age_minutes": (datetime.utcnow() - group.first_seen).total_seconds() / 60
            })
        
        return groups_info