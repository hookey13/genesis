"""Forensic analysis tools for event replay and audit trail verification."""
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AuditEvent:
    """Audit event for forensic analysis."""

    event_id: str
    event_type: str
    aggregate_id: str
    aggregate_type: str
    event_data: dict[str, Any]
    event_metadata: dict[str, Any]
    created_at: datetime
    sequence_number: int
    checksum: str | None = None

    def calculate_checksum(self) -> str:
        """Calculate SHA256 checksum for event integrity."""
        data_str = json.dumps({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "event_data": self.event_data,
            "sequence_number": self.sequence_number,
            "created_at": self.created_at.isoformat()
        }, sort_keys=True)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify event integrity using checksum."""
        if not self.checksum:
            return False
        return self.calculate_checksum() == self.checksum


@dataclass
class ForensicAnalysisResult:
    """Result of forensic analysis."""

    total_events: int
    verified_events: int
    corrupted_events: int
    missing_sequences: list[int]
    duplicate_sequences: list[int]
    time_gaps: list[tuple[datetime, datetime, float]]  # start, end, gap_seconds
    suspicious_patterns: list[dict[str, Any]]
    integrity_verified: bool
    analysis_timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_events": self.total_events,
            "verified_events": self.verified_events,
            "corrupted_events": self.corrupted_events,
            "missing_sequences": self.missing_sequences,
            "duplicate_sequences": self.duplicate_sequences,
            "time_gaps": [
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "gap_seconds": gap
                }
                for start, end, gap in self.time_gaps
            ],
            "suspicious_patterns": self.suspicious_patterns,
            "integrity_verified": self.integrity_verified,
            "analysis_timestamp": self.analysis_timestamp.isoformat()
        }


class ForensicAnalyzer:
    """Forensic analysis system for audit trail verification."""

    def __init__(self):
        self.logger = structlog.get_logger(self.__class__.__name__)

    def verify_event_chain(
        self,
        events: list[AuditEvent],
        aggregate_id: str | None = None
    ) -> ForensicAnalysisResult:
        """Verify integrity of event chain."""
        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]

        if not events:
            return ForensicAnalysisResult(
                total_events=0,
                verified_events=0,
                corrupted_events=0,
                missing_sequences=[],
                duplicate_sequences=[],
                time_gaps=[],
                suspicious_patterns=[],
                integrity_verified=True,
                analysis_timestamp=datetime.now()
            )

        # Sort events by sequence number
        sorted_events = sorted(events, key=lambda x: x.sequence_number)

        verified_count = 0
        corrupted_count = 0
        missing_sequences = []
        duplicate_sequences = []
        time_gaps = []
        suspicious_patterns = []

        # Check sequence continuity
        seen_sequences = set()
        expected_sequence = sorted_events[0].sequence_number

        for i, event in enumerate(sorted_events):
            # Verify checksum
            if event.verify_integrity():
                verified_count += 1
            else:
                corrupted_count += 1
                suspicious_patterns.append({
                    "type": "corrupted_event",
                    "event_id": event.event_id,
                    "sequence": event.sequence_number
                })

            # Check sequence
            if event.sequence_number in seen_sequences:
                duplicate_sequences.append(event.sequence_number)
            seen_sequences.add(event.sequence_number)

            while expected_sequence < event.sequence_number:
                missing_sequences.append(expected_sequence)
                expected_sequence += 1

            expected_sequence = event.sequence_number + 1

            # Check time gaps
            if i > 0:
                time_diff = (event.created_at - sorted_events[i-1].created_at).total_seconds()

                # Flag gaps > 1 hour
                if time_diff > 3600:
                    time_gaps.append((
                        sorted_events[i-1].created_at,
                        event.created_at,
                        time_diff
                    ))

                # Flag negative time (events out of order)
                if time_diff < 0:
                    suspicious_patterns.append({
                        "type": "time_reversal",
                        "event_id": event.event_id,
                        "previous_event_id": sorted_events[i-1].event_id,
                        "time_diff": time_diff
                    })

        integrity_verified = (
            corrupted_count == 0 and
            len(missing_sequences) == 0 and
            len(duplicate_sequences) == 0 and
            len(suspicious_patterns) == 0
        )

        result = ForensicAnalysisResult(
            total_events=len(events),
            verified_events=verified_count,
            corrupted_events=corrupted_count,
            missing_sequences=missing_sequences,
            duplicate_sequences=duplicate_sequences,
            time_gaps=time_gaps,
            suspicious_patterns=suspicious_patterns,
            integrity_verified=integrity_verified,
            analysis_timestamp=datetime.now()
        )

        self.logger.info(
            "event_chain_verified",
            aggregate_id=aggregate_id,
            total_events=result.total_events,
            integrity_verified=result.integrity_verified,
            issues_found=len(suspicious_patterns)
        )

        return result

    def replay_events(
        self,
        events: list[AuditEvent],
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> dict[str, Any]:
        """Replay events to reconstruct state at a point in time."""
        # Filter events by time range
        if start_time:
            events = [e for e in events if e.created_at >= start_time]
        if end_time:
            events = [e for e in events if e.created_at <= end_time]

        # Sort by sequence number
        sorted_events = sorted(events, key=lambda x: (x.aggregate_id, x.sequence_number))

        # Reconstruct state by aggregate
        state_by_aggregate = {}

        for event in sorted_events:
            if event.aggregate_id not in state_by_aggregate:
                state_by_aggregate[event.aggregate_id] = {
                    "aggregate_type": event.aggregate_type,
                    "events": [],
                    "final_state": {}
                }

            aggregate_state = state_by_aggregate[event.aggregate_id]
            aggregate_state["events"].append(event.event_id)

            # Apply event to state
            self._apply_event_to_state(aggregate_state["final_state"], event)

        self.logger.info(
            "events_replayed",
            total_events=len(sorted_events),
            aggregates_reconstructed=len(state_by_aggregate),
            time_range={
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            }
        )

        return state_by_aggregate

    def _apply_event_to_state(
        self,
        state: dict[str, Any],
        event: AuditEvent
    ) -> None:
        """Apply an event to reconstruct state."""
        event_type = event.event_type
        event_data = event.event_data

        # Handle different event types
        if event_type == "POSITION_OPENED":
            state["position_id"] = event_data.get("position_id")
            state["symbol"] = event_data.get("symbol")
            state["side"] = event_data.get("side")
            state["entry_price"] = event_data.get("entry_price")
            state["quantity"] = event_data.get("quantity")
            state["status"] = "OPEN"

        elif event_type == "POSITION_CLOSED":
            state["status"] = "CLOSED"
            state["exit_price"] = event_data.get("exit_price")
            state["pnl"] = event_data.get("pnl")
            state["close_reason"] = event_data.get("close_reason")

        elif event_type == "ORDER_FILLED":
            state["filled_quantity"] = event_data.get("filled_quantity")
            state["fill_price"] = event_data.get("fill_price")
            state["status"] = "FILLED"

        # Update last modified
        state["last_modified"] = event.created_at.isoformat()
        state["last_event_id"] = event.event_id

    def detect_anomalies(
        self,
        events: list[AuditEvent],
        rules: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Detect anomalous patterns in event stream."""
        anomalies = []

        if not rules:
            # Default anomaly detection rules
            rules = {
                "max_events_per_minute": 100,
                "max_position_size": Decimal("10000"),
                "max_daily_trades": 500,
                "suspicious_patterns": [
                    "rapid_order_cancellation",
                    "position_size_spike",
                    "unusual_trading_hours"
                ]
            }

        # Group events by time windows
        events_by_minute = {}
        events_by_day = {}

        for event in events:
            minute_key = event.created_at.replace(second=0, microsecond=0)
            day_key = event.created_at.date()

            if minute_key not in events_by_minute:
                events_by_minute[minute_key] = []
            events_by_minute[minute_key].append(event)

            if day_key not in events_by_day:
                events_by_day[day_key] = []
            events_by_day[day_key].append(event)

        # Check rate limits
        for minute, minute_events in events_by_minute.items():
            if len(minute_events) > rules["max_events_per_minute"]:
                anomalies.append({
                    "type": "excessive_event_rate",
                    "timestamp": minute.isoformat(),
                    "event_count": len(minute_events),
                    "limit": rules["max_events_per_minute"]
                })

        # Check daily trade limits
        for day, day_events in events_by_day.items():
            trade_events = [
                e for e in day_events
                if e.event_type in ["ORDER_PLACED", "ORDER_FILLED"]
            ]

            if len(trade_events) > rules["max_daily_trades"]:
                anomalies.append({
                    "type": "excessive_daily_trades",
                    "date": day.isoformat(),
                    "trade_count": len(trade_events),
                    "limit": rules["max_daily_trades"]
                })

        # Check for rapid cancellations
        self._detect_rapid_cancellations(events, anomalies)

        # Check for position size spikes
        self._detect_position_size_spikes(events, rules["max_position_size"], anomalies)

        self.logger.info(
            "anomalies_detected",
            total_events=len(events),
            anomalies_found=len(anomalies)
        )

        return anomalies

    def _detect_rapid_cancellations(
        self,
        events: list[AuditEvent],
        anomalies: list[dict[str, Any]]
    ) -> None:
        """Detect rapid order cancellation patterns."""
        order_events = {}

        for event in events:
            if event.event_type in ["ORDER_PLACED", "ORDER_CANCELLED"]:
                order_id = event.event_data.get("order_id")
                if order_id:
                    if order_id not in order_events:
                        order_events[order_id] = []
                    order_events[order_id].append(event)

        for order_id, order_event_list in order_events.items():
            if len(order_event_list) >= 2:
                placed = next((e for e in order_event_list if e.event_type == "ORDER_PLACED"), None)
                cancelled = next((e for e in order_event_list if e.event_type == "ORDER_CANCELLED"), None)

                if placed and cancelled:
                    time_diff = (cancelled.created_at - placed.created_at).total_seconds()

                    # Flag if cancelled within 1 second
                    if time_diff < 1:
                        anomalies.append({
                            "type": "rapid_cancellation",
                            "order_id": order_id,
                            "placed_at": placed.created_at.isoformat(),
                            "cancelled_at": cancelled.created_at.isoformat(),
                            "duration_seconds": time_diff
                        })

    def _detect_position_size_spikes(
        self,
        events: list[AuditEvent],
        max_size: Decimal,
        anomalies: list[dict[str, Any]]
    ) -> None:
        """Detect unusual position size spikes."""
        for event in events:
            if event.event_type == "POSITION_OPENED":
                position_value = Decimal(str(
                    event.event_data.get("dollar_value", 0)
                ))

                if position_value > max_size:
                    anomalies.append({
                        "type": "position_size_spike",
                        "position_id": event.event_data.get("position_id"),
                        "size": str(position_value),
                        "limit": str(max_size),
                        "timestamp": event.created_at.isoformat()
                    })

    def generate_audit_report(
        self,
        events: list[AuditEvent],
        output_path: str
    ) -> None:
        """Generate comprehensive audit report."""
        # Verify event chain
        verification_result = self.verify_event_chain(events)

        # Detect anomalies
        anomalies = self.detect_anomalies(events)

        # Group events by type
        events_by_type = {}
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = 0
            events_by_type[event.event_type] += 1

        # Create report
        report = {
            "audit_report": {
                "generated_at": datetime.now().isoformat(),
                "period": {
                    "start": min(e.created_at for e in events).isoformat() if events else None,
                    "end": max(e.created_at for e in events).isoformat() if events else None
                },
                "summary": {
                    "total_events": len(events),
                    "event_types": events_by_type,
                    "integrity_verified": verification_result.integrity_verified
                },
                "verification": verification_result.to_dict(),
                "anomalies": anomalies,
                "recommendations": self._generate_recommendations(
                    verification_result, anomalies
                )
            }
        }

        # Write report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(
            "audit_report_generated",
            path=output_path,
            total_events=len(events),
            integrity_verified=verification_result.integrity_verified
        )

    def _generate_recommendations(
        self,
        verification_result: ForensicAnalysisResult,
        anomalies: list[dict[str, Any]]
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if verification_result.corrupted_events > 0:
            recommendations.append(
                "CRITICAL: Corrupted events detected. Investigate data integrity issues."
            )

        if verification_result.missing_sequences:
            recommendations.append(
                f"WARNING: Missing sequence numbers detected: {verification_result.missing_sequences[:5]}..."
            )

        if verification_result.duplicate_sequences:
            recommendations.append(
                "WARNING: Duplicate sequence numbers found. Check for race conditions."
            )

        rapid_cancellations = [
            a for a in anomalies if a["type"] == "rapid_cancellation"
        ]
        if len(rapid_cancellations) > 10:
            recommendations.append(
                "INFO: High frequency of rapid order cancellations detected. Review trading strategy."
            )

        position_spikes = [
            a for a in anomalies if a["type"] == "position_size_spike"
        ]
        if position_spikes:
            recommendations.append(
                "WARNING: Position size limit breaches detected. Review risk management."
            )

        if not recommendations:
            recommendations.append("No significant issues detected. System operating normally.")

        return recommendations
