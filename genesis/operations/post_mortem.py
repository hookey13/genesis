"""
Automated post-mortem report generation and analysis system.
"""

import json
import re
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import structlog
from pathlib import Path

from genesis.core.events import Event, EventType
from genesis.operations.incident_manager import Incident, IncidentSeverity

logger = structlog.get_logger(__name__)


class ImpactLevel(Enum):
    """Impact level classification."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class TimelineEvent:
    """Event in incident timeline."""
    timestamp: datetime
    event: str
    actor: str
    notes: str
    event_type: str = "info"  # info, detection, action, resolution


@dataclass
class RootCauseAnalysis:
    """Root cause analysis results."""
    what_happened: str
    five_whys: List[Tuple[str, str]]  # [(question, answer), ...]
    contributing_factors: Dict[str, str]
    root_cause: str


@dataclass
class ActionItem:
    """Post-mortem action item."""
    action: str
    owner: str
    due_date: datetime
    priority: str  # immediate, short-term, long-term
    status: str = "not_started"
    ticket_id: Optional[str] = None


@dataclass
class PostMortemReport:
    """Complete post-mortem report."""
    incident_id: str
    incident_date: datetime
    duration: timedelta
    severity: IncidentSeverity
    executive_summary: str
    quantitative_impact: Dict[str, Any]
    qualitative_impact: Dict[str, ImpactLevel]
    timeline: List[TimelineEvent]
    root_cause_analysis: RootCauseAnalysis
    detection_analysis: Dict[str, Any]
    response_analysis: Dict[str, Any]
    resolution: Dict[str, Any]
    what_went_well: List[str]
    what_could_be_improved: List[str]
    lessons_learned: Dict[str, List[str]]
    action_items: List[ActionItem]
    related_incidents: List[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)


class PostMortemGenerator:
    """Generates post-mortem reports from incident data."""
    
    def __init__(self, template_path: str = "docs/templates/post-mortem-template.md"):
        """Initialize post-mortem generator."""
        self.template_path = template_path
        self.reports: Dict[str, PostMortemReport] = {}
        
        logger.info("Post-mortem generator initialized")
    
    async def generate_report(self, incident: Incident, 
                             logs: Optional[List[Dict[str, Any]]] = None,
                             metrics: Optional[Dict[str, Any]] = None) -> PostMortemReport:
        """
        Generate post-mortem report from incident data.
        
        Args:
            incident: The incident to analyze
            logs: Relevant log entries
            metrics: Performance metrics during incident
        
        Returns:
            Generated post-mortem report
        """
        try:
            # Reconstruct timeline from logs
            timeline = await self._reconstruct_timeline(incident, logs)
            
            # Analyze root cause
            root_cause_analysis = await self._analyze_root_cause(incident, logs, timeline)
            
            # Calculate impact
            quantitative_impact = await self._calculate_quantitative_impact(
                incident, metrics
            )
            qualitative_impact = self._assess_qualitative_impact(incident)
            
            # Analyze detection and response
            detection_analysis = self._analyze_detection(incident, timeline)
            response_analysis = self._analyze_response(incident, timeline)
            
            # Generate resolution details
            resolution = self._extract_resolution(incident, timeline)
            
            # Identify improvements
            what_went_well = self._identify_positives(incident, timeline, metrics)
            what_could_be_improved = self._identify_improvements(
                incident, timeline, detection_analysis, response_analysis
            )
            
            # Extract lessons learned
            lessons_learned = self._extract_lessons(
                incident, root_cause_analysis, timeline
            )
            
            # Generate action items
            action_items = self._generate_action_items(
                incident, root_cause_analysis, what_could_be_improved
            )
            
            # Find related incidents
            related_incidents = await self._find_related_incidents(incident)
            
            # Create report
            report = PostMortemReport(
                incident_id=incident.incident_id,
                incident_date=incident.created_at,
                duration=incident.duration() or timedelta(),
                severity=incident.severity,
                executive_summary=self._generate_executive_summary(
                    incident, quantitative_impact
                ),
                quantitative_impact=quantitative_impact,
                qualitative_impact=qualitative_impact,
                timeline=timeline,
                root_cause_analysis=root_cause_analysis,
                detection_analysis=detection_analysis,
                response_analysis=response_analysis,
                resolution=resolution,
                what_went_well=what_went_well,
                what_could_be_improved=what_could_be_improved,
                lessons_learned=lessons_learned,
                action_items=action_items,
                related_incidents=related_incidents
            )
            
            self.reports[incident.incident_id] = report
            
            logger.info("Post-mortem report generated",
                       incident_id=incident.incident_id,
                       timeline_events=len(timeline),
                       action_items=len(action_items))
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate post-mortem",
                        incident_id=incident.incident_id,
                        error=str(e))
            raise
    
    async def _reconstruct_timeline(self, incident: Incident,
                                   logs: Optional[List[Dict[str, Any]]]) -> List[TimelineEvent]:
        """Reconstruct timeline from incident and logs."""
        timeline = []
        
        # Add incident creation
        timeline.append(TimelineEvent(
            timestamp=incident.created_at,
            event="Incident detected",
            actor="System",
            notes=incident.description,
            event_type="detection"
        ))
        
        # Add events from incident timeline
        for entry in incident.timeline:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            timeline.append(TimelineEvent(
                timestamp=timestamp,
                event=entry["action"],
                actor=entry["actor"],
                notes=entry.get("details", ""),
                event_type=self._classify_event_type(entry["action"])
            ))
        
        # Add resolution if present
        if incident.resolved_at:
            timeline.append(TimelineEvent(
                timestamp=incident.resolved_at,
                event="Incident resolved",
                actor=incident.timeline[-1]["actor"] if incident.timeline else "Unknown",
                notes=incident.root_cause or "",
                event_type="resolution"
            ))
        
        # Parse logs for additional events
        if logs:
            log_events = self._parse_logs_for_events(logs, 
                                                    incident.created_at,
                                                    incident.resolved_at)
            timeline.extend(log_events)
        
        # Sort by timestamp
        timeline.sort(key=lambda x: x.timestamp)
        
        return timeline
    
    def _classify_event_type(self, action: str) -> str:
        """Classify event type from action string."""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ["detected", "alert", "triggered"]):
            return "detection"
        elif any(word in action_lower for word in ["resolved", "fixed", "completed"]):
            return "resolution"
        elif any(word in action_lower for word in ["started", "began", "initiated",
                                                   "acknowledged", "investigating"]):
            return "action"
        else:
            return "info"
    
    def _parse_logs_for_events(self, logs: List[Dict[str, Any]],
                               start: datetime, 
                               end: Optional[datetime]) -> List[TimelineEvent]:
        """Parse logs for timeline events."""
        events = []
        
        for log in logs:
            timestamp = log.get("timestamp")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Only include logs within incident timeframe
            if timestamp < start:
                continue
            if end and timestamp > end:
                continue
            
            # Look for significant events
            message = log.get("message", "")
            level = log.get("level", "info")
            
            if level in ["error", "critical"]:
                events.append(TimelineEvent(
                    timestamp=timestamp,
                    event=f"Error: {message[:100]}",
                    actor=log.get("service", "System"),
                    notes=message,
                    event_type="detection"
                ))
            elif "restart" in message.lower():
                events.append(TimelineEvent(
                    timestamp=timestamp,
                    event="Service restarted",
                    actor=log.get("service", "System"),
                    notes=message,
                    event_type="action"
                ))
        
        return events
    
    async def _analyze_root_cause(self, incident: Incident,
                                 logs: Optional[List[Dict[str, Any]]],
                                 timeline: List[TimelineEvent]) -> RootCauseAnalysis:
        """Analyze root cause using 5 whys methodology."""
        # Extract initial problem
        what_happened = incident.description
        
        # If root cause was provided
        if incident.root_cause:
            what_happened += f" Root cause identified: {incident.root_cause}"
        
        # Perform 5 whys analysis
        five_whys = []
        
        # Analyze based on alert type
        if incident.alerts:
            alert_type = incident.alerts[0].alert_type.value
            five_whys = self._perform_five_whys(alert_type, logs)
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(incident, logs, timeline)
        
        # Determine root cause
        root_cause = incident.root_cause or self._infer_root_cause(five_whys, contributing_factors)
        
        return RootCauseAnalysis(
            what_happened=what_happened,
            five_whys=five_whys,
            contributing_factors=contributing_factors,
            root_cause=root_cause
        )
    
    def _perform_five_whys(self, alert_type: str, 
                          logs: Optional[List[Dict[str, Any]]]) -> List[Tuple[str, str]]:
        """Perform 5 whys analysis based on alert type."""
        # This is a simplified example - in practice, this would be more sophisticated
        whys_map = {
            "system_down": [
                ("Why did the system go down?", "Process crashed unexpectedly"),
                ("Why did the process crash?", "Out of memory error occurred"),
                ("Why did we run out of memory?", "Memory leak in order processing"),
                ("Why was there a memory leak?", "Objects not being garbage collected"),
                ("Why weren't objects garbage collected?", "Circular references in event handlers")
            ],
            "order_failure": [
                ("Why did orders fail?", "API requests were rejected"),
                ("Why were API requests rejected?", "Rate limit exceeded"),
                ("Why was rate limit exceeded?", "Burst of retry attempts"),
                ("Why were there retry attempts?", "Network timeout errors"),
                ("Why were there network timeouts?", "Insufficient connection pool size")
            ]
        }
        
        return whys_map.get(alert_type, [
            ("Why did the incident occur?", "Unknown - requires investigation"),
            ("Why?", "Insufficient data for analysis"),
            ("Why?", "Monitoring gap prevented root cause identification"),
            ("Why?", "Logging was not detailed enough"),
            ("Why?", "Observability practices need improvement")
        ])
    
    def _identify_contributing_factors(self, incident: Incident,
                                      logs: Optional[List[Dict[str, Any]]],
                                      timeline: List[TimelineEvent]) -> Dict[str, str]:
        """Identify contributing factors to the incident."""
        factors = {}
        
        # Check for technical debt indicators
        if any("deprecated" in str(log.get("message", "")).lower() for log in (logs or [])):
            factors["technical_debt"] = "Deprecated components still in use"
        
        # Check for process gaps
        detection_time = None
        for event in timeline:
            if event.event_type == "detection":
                detection_time = event.timestamp
                break
        
        if detection_time and (detection_time - incident.created_at).total_seconds() > 300:
            factors["process_gap"] = "Detection took more than 5 minutes"
        
        # Check for monitoring gaps
        if not incident.alerts or len(incident.alerts) == 0:
            factors["monitoring_gap"] = "No automated alerts triggered"
        
        # Check for documentation issues
        if not incident.runbook_url:
            factors["documentation"] = "No runbook available for this scenario"
        
        return factors
    
    def _infer_root_cause(self, five_whys: List[Tuple[str, str]],
                         contributing_factors: Dict[str, str]) -> str:
        """Infer root cause from analysis."""
        if five_whys and len(five_whys) >= 5:
            return five_whys[-1][1]  # Last answer in 5 whys
        elif contributing_factors:
            # Pick most significant contributing factor
            if "technical_debt" in contributing_factors:
                return contributing_factors["technical_debt"]
            elif "process_gap" in contributing_factors:
                return contributing_factors["process_gap"]
            else:
                return list(contributing_factors.values())[0]
        else:
            return "Root cause requires further investigation"
    
    async def _calculate_quantitative_impact(self, incident: Incident,
                                            metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quantitative impact of incident."""
        impact = {
            "duration_minutes": int(incident.duration().total_seconds() / 60) if incident.duration() else 0,
            "alerts_triggered": len(incident.alerts),
            "responders_involved": len(incident.responders)
        }
        
        # Add metrics if available
        if metrics:
            impact.update({
                "orders_affected": metrics.get("failed_orders", 0),
                "positions_affected": metrics.get("affected_positions", 0),
                "estimated_loss": metrics.get("estimated_loss", 0),
                "api_errors": metrics.get("api_errors", 0),
                "response_time_degradation": metrics.get("response_time_increase", 0)
            })
        
        return impact
    
    def _assess_qualitative_impact(self, incident: Incident) -> Dict[str, ImpactLevel]:
        """Assess qualitative impact of incident."""
        impact = {}
        
        # Reputation impact based on severity and duration
        if incident.severity == IncidentSeverity.CRITICAL:
            impact["reputation"] = ImpactLevel.SEVERE
        elif incident.severity == IncidentSeverity.HIGH:
            impact["reputation"] = ImpactLevel.MODERATE
        else:
            impact["reputation"] = ImpactLevel.MINOR
        
        # Regulatory impact
        if "compliance" in incident.description.lower():
            impact["regulatory"] = ImpactLevel.MODERATE
        else:
            impact["regulatory"] = ImpactLevel.NONE
        
        # Customer trust
        duration = incident.duration()
        if duration and duration.total_seconds() > 3600:  # More than 1 hour
            impact["customer_trust"] = ImpactLevel.MODERATE
        elif duration and duration.total_seconds() > 300:  # More than 5 minutes
            impact["customer_trust"] = ImpactLevel.MINOR
        else:
            impact["customer_trust"] = ImpactLevel.NONE
        
        return impact
    
    def _analyze_detection(self, incident: Incident,
                          timeline: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze incident detection."""
        detection_event = None
        for event in timeline:
            if event.event_type == "detection":
                detection_event = event
                break
        
        if not detection_event:
            return {
                "time_to_detection": "Unknown",
                "detection_method": "Unknown",
                "could_detect_sooner": "Unknown"
            }
        
        time_to_detection = (detection_event.timestamp - incident.created_at).total_seconds()
        
        return {
            "time_to_detection_seconds": time_to_detection,
            "detection_method": detection_event.actor,
            "could_detect_sooner": time_to_detection > 60,  # More than 1 minute
            "detection_notes": detection_event.notes
        }
    
    def _analyze_response(self, incident: Incident,
                         timeline: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze incident response."""
        acknowledgment_time = None
        first_action_time = None
        
        for event in timeline:
            if "acknowledged" in event.event.lower() and not acknowledgment_time:
                acknowledgment_time = event.timestamp
            if event.event_type == "action" and not first_action_time:
                first_action_time = event.timestamp
        
        response = {
            "responders": incident.responders,
            "response_efficiency": "Unknown"
        }
        
        if acknowledgment_time:
            time_to_ack = (acknowledgment_time - incident.created_at).total_seconds()
            response["time_to_acknowledgment_seconds"] = time_to_ack
            
            if time_to_ack < 300:  # Less than 5 minutes
                response["response_efficiency"] = "Good"
            elif time_to_ack < 900:  # Less than 15 minutes
                response["response_efficiency"] = "Adequate"
            else:
                response["response_efficiency"] = "Poor"
        
        if first_action_time:
            response["time_to_first_action_seconds"] = (
                first_action_time - incident.created_at
            ).total_seconds()
        
        if incident.resolved_at:
            response["time_to_resolution_seconds"] = (
                incident.resolved_at - incident.created_at
            ).total_seconds()
        
        return response
    
    def _extract_resolution(self, incident: Incident,
                           timeline: List[TimelineEvent]) -> Dict[str, Any]:
        """Extract resolution details."""
        resolution_events = [e for e in timeline if e.event_type == "resolution"]
        action_events = [e for e in timeline if e.event_type == "action"]
        
        return {
            "resolved": incident.status.value == "resolved",
            "resolution_time": incident.resolved_at.isoformat() if incident.resolved_at else None,
            "root_cause": incident.root_cause,
            "immediate_actions": [e.event for e in action_events[:3]],  # First 3 actions
            "resolution_steps": [e.event for e in resolution_events],
            "mitigation_steps": incident.mitigation_steps
        }
    
    def _identify_positives(self, incident: Incident,
                           timeline: List[TimelineEvent],
                           metrics: Optional[Dict[str, Any]]) -> List[str]:
        """Identify what went well during incident."""
        positives = []
        
        # Fast detection
        detection_time = None
        for event in timeline:
            if event.event_type == "detection":
                detection_time = (event.timestamp - incident.created_at).total_seconds()
                break
        
        if detection_time and detection_time < 60:
            positives.append(f"Issue detected within {int(detection_time)} seconds")
        
        # Fast acknowledgment
        for event in timeline:
            if "acknowledged" in event.event.lower():
                ack_time = (event.timestamp - incident.created_at).total_seconds()
                if ack_time < 300:
                    positives.append(f"Incident acknowledged within {int(ack_time/60)} minutes")
                break
        
        # Multiple responders
        if len(incident.responders) > 1:
            positives.append(f"{len(incident.responders)} responders collaborated on resolution")
        
        # Runbook available
        if incident.runbook_url:
            positives.append("Runbook was available and used")
        
        # Fast resolution
        if incident.duration() and incident.duration().total_seconds() < 1800:
            positives.append(f"Incident resolved in {int(incident.duration().total_seconds()/60)} minutes")
        
        return positives
    
    def _identify_improvements(self, incident: Incident,
                              timeline: List[TimelineEvent],
                              detection_analysis: Dict[str, Any],
                              response_analysis: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement."""
        improvements = []
        
        # Slow detection
        if detection_analysis.get("could_detect_sooner"):
            improvements.append("Detection time could be improved with better monitoring")
        
        # Slow response
        if response_analysis.get("response_efficiency") == "Poor":
            improvements.append("Response time exceeded acceptable thresholds")
        
        # No runbook
        if not incident.runbook_url:
            improvements.append("No runbook available for this incident type")
        
        # Long resolution time
        if incident.duration() and incident.duration().total_seconds() > 3600:
            improvements.append("Resolution took more than 1 hour")
        
        # Single responder
        if len(incident.responders) <= 1:
            improvements.append("Only one responder involved - consider escalation")
        
        # No root cause
        if not incident.root_cause:
            improvements.append("Root cause not definitively identified")
        
        return improvements
    
    def _extract_lessons(self, incident: Incident,
                        root_cause_analysis: RootCauseAnalysis,
                        timeline: List[TimelineEvent]) -> Dict[str, List[str]]:
        """Extract lessons learned."""
        lessons = {
            "technical": [],
            "process": [],
            "organizational": []
        }
        
        # Technical lessons
        if "memory" in root_cause_analysis.root_cause.lower():
            lessons["technical"].append("Memory management needs improvement")
        if "rate limit" in root_cause_analysis.root_cause.lower():
            lessons["technical"].append("Rate limiting strategy needs adjustment")
        if "timeout" in root_cause_analysis.root_cause.lower():
            lessons["technical"].append("Timeout configurations need review")
        
        # Process lessons
        if "monitoring_gap" in root_cause_analysis.contributing_factors:
            lessons["process"].append("Monitoring coverage needs expansion")
        if "documentation" in root_cause_analysis.contributing_factors:
            lessons["process"].append("Documentation needs to be updated")
        if len(timeline) > 20:
            lessons["process"].append("Incident had many steps - consider automation")
        
        # Organizational lessons
        if len(incident.responders) == 0:
            lessons["organizational"].append("No responders tracked - improve incident tracking")
        if incident.severity == IncidentSeverity.CRITICAL and len(incident.responders) < 3:
            lessons["organizational"].append("Critical incidents need more responders")
        
        return lessons
    
    def _generate_action_items(self, incident: Incident,
                              root_cause_analysis: RootCauseAnalysis,
                              improvements: List[str]) -> List[ActionItem]:
        """Generate action items from analysis."""
        action_items = []
        
        # Immediate actions (within 24 hours)
        if not incident.root_cause:
            action_items.append(ActionItem(
                action="Complete root cause analysis",
                owner="Engineering Lead",
                due_date=datetime.utcnow() + timedelta(days=1),
                priority="immediate"
            ))
        
        if "monitoring_gap" in root_cause_analysis.contributing_factors:
            action_items.append(ActionItem(
                action="Add monitoring for identified gap",
                owner="SRE Team",
                due_date=datetime.utcnow() + timedelta(days=1),
                priority="immediate"
            ))
        
        # Short-term actions (within 1 week)
        if "documentation" in root_cause_analysis.contributing_factors:
            action_items.append(ActionItem(
                action="Update runbook for this scenario",
                owner="On-call Team",
                due_date=datetime.utcnow() + timedelta(days=7),
                priority="short-term"
            ))
        
        for improvement in improvements[:2]:  # Take first 2 improvements
            action_items.append(ActionItem(
                action=f"Address: {improvement}",
                owner="Engineering Team",
                due_date=datetime.utcnow() + timedelta(days=7),
                priority="short-term"
            ))
        
        # Long-term actions (within 1 month)
        if "technical_debt" in root_cause_analysis.contributing_factors:
            action_items.append(ActionItem(
                action="Refactor components with technical debt",
                owner="Engineering Team",
                due_date=datetime.utcnow() + timedelta(days=30),
                priority="long-term"
            ))
        
        return action_items
    
    async def _find_related_incidents(self, incident: Incident) -> List[str]:
        """Find related historical incidents."""
        related = []
        
        # In a real implementation, this would query historical incidents
        # For now, return empty list
        
        return related
    
    def _generate_executive_summary(self, incident: Incident,
                                   impact: Dict[str, Any]) -> str:
        """Generate executive summary."""
        duration_str = f"{impact.get('duration_minutes', 0)} minutes"
        
        summary = (
            f"On {incident.created_at.strftime('%Y-%m-%d')}, a {incident.severity.value} "
            f"severity incident occurred lasting {duration_str}. "
        )
        
        if impact.get("orders_affected"):
            summary += f"{impact['orders_affected']} orders were affected. "
        
        if incident.resolved_at:
            summary += "The incident has been resolved and systems are stable."
        else:
            summary += "The incident is ongoing with mitigation in place."
        
        return summary
    
    def export_to_markdown(self, report: PostMortemReport) -> str:
        """Export report to markdown format."""
        try:
            # Load template
            with open(self.template_path, 'r') as f:
                template = f.read()
            
            # Replace placeholders
            markdown = template
            
            # Basic information
            markdown = markdown.replace("[INC-YYYYMMDD-XXXX]", report.incident_id)
            markdown = markdown.replace("[YYYY-MM-DD]", report.incident_date.strftime("%Y-%m-%d"))
            markdown = markdown.replace("[Start Time]", report.incident_date.strftime("%H:%M"))
            
            if report.incident_date + report.duration:
                end_time = (report.incident_date + report.duration).strftime("%H:%M")
                markdown = markdown.replace("[End Time]", end_time)
            
            markdown = markdown.replace("[X hours Y minutes]", 
                                      f"{int(report.duration.total_seconds()//3600)} hours "
                                      f"{int((report.duration.total_seconds()%3600)//60)} minutes")
            
            markdown = markdown.replace("[Critical/High/Medium/Low]", report.severity.value.capitalize())
            
            # Executive summary
            summary_section = report.executive_summary
            markdown = re.sub(r'\[2-3 sentence summary.*?\]', summary_section, markdown)
            
            # Generate timeline table
            timeline_rows = []
            for event in report.timeline:
                timeline_rows.append(
                    f"| {event.timestamp.strftime('%H:%M')} | {event.event} | "
                    f"{event.actor} | {event.notes[:50]} |"
                )
            timeline_table = "\n".join(timeline_rows)
            
            # Root cause analysis
            whys_text = "\n".join([
                f"{i+1}. **{q}**\n   - {a}"
                for i, (q, a) in enumerate(report.root_cause_analysis.five_whys)
            ])
            
            # Action items table
            action_rows = []
            for item in report.action_items:
                action_rows.append(
                    f"| {item.action} | {item.owner} | "
                    f"{item.due_date.strftime('%Y-%m-%d')} | {item.status} | "
                    f"{item.ticket_id or 'TBD'} |"
                )
            
            # What went well
            well_text = "\n".join([f"- ✅ {item}" for item in report.what_went_well])
            
            # What could be improved
            improve_text = "\n".join([f"- ⚠️ {item}" for item in report.what_could_be_improved])
            
            return markdown
            
        except Exception as e:
            logger.error("Failed to export to markdown",
                        incident_id=report.incident_id,
                        error=str(e))
            return f"# Post-Mortem Report\n\nError generating report: {str(e)}"