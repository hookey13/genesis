"""
PagerDuty API integration for incident escalation and management.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class PagerDutyPriority(Enum):
    """PagerDuty incident priorities."""
    P1 = "P1"  # Critical
    P2 = "P2"  # High
    P3 = "P3"  # Medium
    P4 = "P4"  # Low
    P5 = "P5"  # Informational


class PagerDutyClient:
    """Client for PagerDuty API integration."""
    
    API_BASE = "https://api.pagerduty.com"
    EVENTS_BASE = "https://events.pagerduty.com/v2"
    
    # Map our severity to PagerDuty severity
    SEVERITY_MAP = {
        "critical": "critical",
        "high": "error",
        "medium": "warning",
        "low": "info"
    }
    
    # Map our severity to PagerDuty priority
    PRIORITY_MAP = {
        "critical": PagerDutyPriority.P1,
        "high": PagerDutyPriority.P2,
        "medium": PagerDutyPriority.P3,
        "low": PagerDutyPriority.P4
    }
    
    def __init__(self, api_key: str, integration_key: str, 
                 email: str, escalation_policy_id: Optional[str] = None):
        """
        Initialize PagerDuty client.
        
        Args:
            api_key: PagerDuty API key for management operations
            integration_key: Integration key for sending events
            email: Email address for API requests
            escalation_policy_id: Default escalation policy ID
        """
        self.api_key = api_key
        self.integration_key = integration_key
        self.email = email
        self.escalation_policy_id = escalation_policy_id
        
        self.headers = {
            "Authorization": f"Token token={api_key}",
            "Accept": "application/vnd.pagerduty+json;version=2",
            "Content-Type": "application/json",
            "From": email
        }
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._incident_cache: Dict[str, str] = {}  # Our ID -> PD ID
        
        logger.info("PagerDuty client initialized", 
                   escalation_policy=escalation_policy_id)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def trigger_incident(self, incident_key: str, summary: str,
                              severity: str, details: Dict[str, Any],
                              dedup_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Trigger a PagerDuty incident.
        
        Args:
            incident_key: Unique incident identifier
            summary: Brief incident summary
            severity: Incident severity (critical/high/medium/low)
            details: Additional incident details
            dedup_key: Deduplication key (uses incident_key if not provided)
        
        Returns:
            PagerDuty response
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            pd_severity = self.SEVERITY_MAP.get(severity, "error")
            
            payload = {
                "routing_key": self.integration_key,
                "event_action": "trigger",
                "dedup_key": dedup_key or incident_key,
                "payload": {
                    "summary": summary,
                    "severity": pd_severity,
                    "source": "genesis-trading-system",
                    "component": "incident-manager",
                    "group": "trading",
                    "class": "incident",
                    "custom_details": details,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Add links if runbook URL provided
            if details.get("runbook_url"):
                payload["links"] = [{
                    "href": details["runbook_url"],
                    "text": "Runbook"
                }]
            
            async with self.session.post(
                f"{self.EVENTS_BASE}/enqueue",
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 202:
                    logger.info("PagerDuty incident triggered",
                               incident_key=incident_key,
                               dedup_key=result.get("dedup_key"))
                    
                    # Cache the mapping
                    self._incident_cache[incident_key] = result.get("dedup_key", incident_key)
                else:
                    logger.error("Failed to trigger PagerDuty incident",
                                status=response.status,
                                response=result)
                
                return result
                
        except Exception as e:
            logger.error("PagerDuty trigger failed",
                        incident_key=incident_key,
                        error=str(e))
            raise
    
    async def acknowledge_incident(self, incident_key: str) -> Dict[str, Any]:
        """Acknowledge a PagerDuty incident."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            dedup_key = self._incident_cache.get(incident_key, incident_key)
            
            payload = {
                "routing_key": self.integration_key,
                "event_action": "acknowledge",
                "dedup_key": dedup_key
            }
            
            async with self.session.post(
                f"{self.EVENTS_BASE}/enqueue",
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 202:
                    logger.info("PagerDuty incident acknowledged",
                               incident_key=incident_key)
                else:
                    logger.error("Failed to acknowledge PagerDuty incident",
                                status=response.status,
                                response=result)
                
                return result
                
        except Exception as e:
            logger.error("PagerDuty acknowledge failed",
                        incident_key=incident_key,
                        error=str(e))
            raise
    
    async def resolve_incident(self, incident_key: str) -> Dict[str, Any]:
        """Resolve a PagerDuty incident."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            dedup_key = self._incident_cache.get(incident_key, incident_key)
            
            payload = {
                "routing_key": self.integration_key,
                "event_action": "resolve",
                "dedup_key": dedup_key
            }
            
            async with self.session.post(
                f"{self.EVENTS_BASE}/enqueue",
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 202:
                    logger.info("PagerDuty incident resolved",
                               incident_key=incident_key)
                    # Remove from cache
                    self._incident_cache.pop(incident_key, None)
                else:
                    logger.error("Failed to resolve PagerDuty incident",
                                status=response.status,
                                response=result)
                
                return result
                
        except Exception as e:
            logger.error("PagerDuty resolve failed",
                        incident_key=incident_key,
                        error=str(e))
            raise
    
    async def create_incident_note(self, incident_id: str, content: str) -> Dict[str, Any]:
        """Add a note to a PagerDuty incident."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            payload = {
                "note": {
                    "content": content
                }
            }
            
            async with self.session.post(
                f"{self.API_BASE}/incidents/{incident_id}/notes",
                headers=self.headers,
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 201:
                    logger.info("Note added to PagerDuty incident",
                               incident_id=incident_id)
                else:
                    logger.error("Failed to add note to PagerDuty incident",
                                status=response.status,
                                response=result)
                
                return result
                
        except Exception as e:
            logger.error("Failed to add incident note",
                        incident_id=incident_id,
                        error=str(e))
            raise
    
    async def get_on_call_users(self, escalation_policy_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently on-call users for an escalation policy."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            policy_id = escalation_policy_id or self.escalation_policy_id
            if not policy_id:
                logger.warning("No escalation policy ID provided")
                return []
            
            params = {
                "escalation_policy_ids[]": policy_id,
                "include[]": "users"
            }
            
            async with self.session.get(
                f"{self.API_BASE}/oncalls",
                headers=self.headers,
                params=params
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    oncalls = result.get("oncalls", [])
                    users = []
                    for oncall in oncalls:
                        user = oncall.get("user", {})
                        users.append({
                            "id": user.get("id"),
                            "name": user.get("name"),
                            "email": user.get("email"),
                            "escalation_level": oncall.get("escalation_level")
                        })
                    
                    logger.info("Retrieved on-call users",
                               count=len(users))
                    return users
                else:
                    logger.error("Failed to get on-call users",
                                status=response.status,
                                response=result)
                    return []
                
        except Exception as e:
            logger.error("Failed to get on-call users",
                        error=str(e))
            return []
    
    async def get_escalation_policies(self) -> List[Dict[str, Any]]:
        """Get all escalation policies."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(
                f"{self.API_BASE}/escalation_policies",
                headers=self.headers
            ) as response:
                result = await response.json()
                
                if response.status == 200:
                    policies = result.get("escalation_policies", [])
                    logger.info("Retrieved escalation policies",
                               count=len(policies))
                    return policies
                else:
                    logger.error("Failed to get escalation policies",
                                status=response.status,
                                response=result)
                    return []
                
        except Exception as e:
            logger.error("Failed to get escalation policies",
                        error=str(e))
            return []
    
    async def test_connection(self) -> bool:
        """Test PagerDuty connection and credentials."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Test API key by getting abilities
            async with self.session.get(
                f"{self.API_BASE}/abilities",
                headers=self.headers
            ) as response:
                if response.status != 200:
                    logger.error("PagerDuty API key test failed",
                                status=response.status)
                    return False
            
            # Test integration key with a test event
            test_payload = {
                "routing_key": self.integration_key,
                "event_action": "trigger",
                "dedup_key": f"test-{datetime.utcnow().timestamp()}",
                "payload": {
                    "summary": "Test event - please ignore",
                    "severity": "info",
                    "source": "genesis-test",
                    "component": "test"
                }
            }
            
            async with self.session.post(
                f"{self.EVENTS_BASE}/enqueue",
                json=test_payload
            ) as response:
                if response.status != 202:
                    logger.error("PagerDuty integration key test failed",
                                status=response.status)
                    return False
                
                # Immediately resolve the test incident
                result = await response.json()
                test_dedup = result.get("dedup_key")
                
                resolve_payload = {
                    "routing_key": self.integration_key,
                    "event_action": "resolve",
                    "dedup_key": test_dedup
                }
                
                await self.session.post(
                    f"{self.EVENTS_BASE}/enqueue",
                    json=resolve_payload
                )
            
            logger.info("PagerDuty connection test successful")
            return True
            
        except Exception as e:
            logger.error("PagerDuty connection test failed",
                        error=str(e))
            return False
    
    async def create_maintenance_window(self, start_time: datetime, 
                                       end_time: datetime,
                                       description: str) -> Dict[str, Any]:
        """Create a maintenance window to suppress alerts."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            payload = {
                "maintenance_window": {
                    "type": "maintenance_window",
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "description": description,
                    "services": []  # Would need service IDs
                }
            }
            
            async with self.session.post(
                f"{self.API_BASE}/maintenance_windows",
                headers=self.headers,
                json=payload
            ) as response:
                result = await response.json()
                
                if response.status == 201:
                    logger.info("Maintenance window created",
                               start=start_time.isoformat(),
                               end=end_time.isoformat())
                else:
                    logger.error("Failed to create maintenance window",
                                status=response.status,
                                response=result)
                
                return result
                
        except Exception as e:
            logger.error("Failed to create maintenance window",
                        error=str(e))
            raise