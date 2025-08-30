"""Integration tests for audit trail system."""
import json
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
import pytest
from uuid import uuid4

from genesis.analytics.forensics import (
    AuditEvent,
    ForensicAnalyzer,
    ForensicAnalysisResult
)


class TestAuditTrailIntegration:
    """Integration tests for audit trail functionality."""
    
    @pytest.fixture
    def forensic_analyzer(self):
        """Create ForensicAnalyzer instance."""
        return ForensicAnalyzer()
    
    @pytest.fixture
    def sample_event_chain(self):
        """Create a sample chain of audit events."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        aggregate_id = str(uuid4())
        
        # Create a sequence of trading events
        event_sequence = [
            ("POSITION_OPENED", {
                "position_id": str(uuid4()),
                "symbol": "BTC/USDT",
                "side": "LONG",
                "entry_price": "45000",
                "quantity": "0.5",
                "dollar_value": "22500"
            }),
            ("ORDER_PLACED", {
                "order_id": str(uuid4()),
                "symbol": "BTC/USDT",
                "side": "BUY",
                "quantity": "0.5",
                "price": "45000"
            }),
            ("ORDER_FILLED", {
                "order_id": str(uuid4()),
                "filled_quantity": "0.5",
                "fill_price": "45000",
                "fees": "22.50"
            }),
            ("POSITION_UPDATED", {
                "position_id": str(uuid4()),
                "current_price": "46000",
                "pnl": "500",
                "pnl_percent": "2.22"
            }),
            ("POSITION_CLOSED", {
                "position_id": str(uuid4()),
                "exit_price": "46000",
                "pnl": "500",
                "close_reason": "take_profit"
            })
        ]
        
        for i, (event_type, event_data) in enumerate(event_sequence):
            event = AuditEvent(
                event_id=str(uuid4()),
                event_type=event_type,
                aggregate_id=aggregate_id,
                aggregate_type="Position",
                event_data=event_data,
                event_metadata={"user": "system", "source": "trading_engine"},
                created_at=base_time + timedelta(minutes=i),
                sequence_number=i + 1
            )
            event.checksum = event.calculate_checksum()
            events.append(event)
        
        return events
    
    def test_event_chain_verification_valid(self, forensic_analyzer, sample_event_chain):
        """Test verification of a valid event chain."""
        result = forensic_analyzer.verify_event_chain(sample_event_chain)
        
        assert result.integrity_verified is True
        assert result.total_events == 5
        assert result.verified_events == 5
        assert result.corrupted_events == 0
        assert len(result.missing_sequences) == 0
        assert len(result.duplicate_sequences) == 0
    
    def test_event_chain_verification_corrupted(self, forensic_analyzer, sample_event_chain):
        """Test detection of corrupted events."""
        # Corrupt one event's checksum
        sample_event_chain[2].checksum = "invalid_checksum"
        
        result = forensic_analyzer.verify_event_chain(sample_event_chain)
        
        assert result.integrity_verified is False
        assert result.corrupted_events == 1
        assert result.verified_events == 4
        assert len(result.suspicious_patterns) == 1
        assert result.suspicious_patterns[0]["type"] == "corrupted_event"
    
    def test_event_chain_missing_sequences(self, forensic_analyzer, sample_event_chain):
        """Test detection of missing sequence numbers."""
        # Remove middle event (sequence 3)
        del sample_event_chain[2]
        
        result = forensic_analyzer.verify_event_chain(sample_event_chain)
        
        assert result.integrity_verified is False
        assert len(result.missing_sequences) == 1
        assert 3 in result.missing_sequences
    
    def test_event_chain_duplicate_sequences(self, forensic_analyzer, sample_event_chain):
        """Test detection of duplicate sequence numbers."""
        # Duplicate a sequence number
        sample_event_chain[3].sequence_number = sample_event_chain[2].sequence_number
        
        result = forensic_analyzer.verify_event_chain(sample_event_chain)
        
        assert result.integrity_verified is False
        assert len(result.duplicate_sequences) == 1
    
    def test_event_replay(self, forensic_analyzer, sample_event_chain):
        """Test event replay to reconstruct state."""
        state = forensic_analyzer.replay_events(sample_event_chain)
        
        aggregate_id = sample_event_chain[0].aggregate_id
        assert aggregate_id in state
        
        aggregate_state = state[aggregate_id]
        assert aggregate_state["aggregate_type"] == "Position"
        assert len(aggregate_state["events"]) == 5
        assert aggregate_state["final_state"]["status"] == "CLOSED"
        assert aggregate_state["final_state"]["pnl"] == "500"
    
    def test_event_replay_with_time_filter(self, forensic_analyzer, sample_event_chain):
        """Test event replay with time range filtering."""
        start_time = sample_event_chain[1].created_at
        end_time = sample_event_chain[3].created_at
        
        state = forensic_analyzer.replay_events(
            sample_event_chain,
            start_time=start_time,
            end_time=end_time
        )
        
        aggregate_id = sample_event_chain[0].aggregate_id
        aggregate_state = state[aggregate_id]
        
        # Should only have events 2, 3, 4 (indices 1, 2, 3)
        assert len(aggregate_state["events"]) == 3
        assert aggregate_state["final_state"]["status"] != "CLOSED"  # Not yet closed
    
    def test_anomaly_detection_excessive_events(self, forensic_analyzer):
        """Test detection of excessive event rate."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create 150 events in one minute (exceeds default limit of 100)
        for i in range(150):
            event = AuditEvent(
                event_id=str(uuid4()),
                event_type="ORDER_PLACED",
                aggregate_id=str(uuid4()),
                aggregate_type="Order",
                event_data={"order_id": str(uuid4())},
                event_metadata={},
                created_at=base_time + timedelta(seconds=i * 0.4),  # All within 60 seconds
                sequence_number=i + 1
            )
            event.checksum = event.calculate_checksum()
            events.append(event)
        
        anomalies = forensic_analyzer.detect_anomalies(events)
        
        assert len(anomalies) > 0
        excessive_rate = [a for a in anomalies if a["type"] == "excessive_event_rate"]
        assert len(excessive_rate) > 0
        assert excessive_rate[0]["event_count"] > 100
    
    def test_anomaly_detection_rapid_cancellation(self, forensic_analyzer):
        """Test detection of rapid order cancellations."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        order_id = str(uuid4())
        
        # Place and cancel order within 0.5 seconds
        place_event = AuditEvent(
            event_id=str(uuid4()),
            event_type="ORDER_PLACED",
            aggregate_id=order_id,
            aggregate_type="Order",
            event_data={"order_id": order_id},
            event_metadata={},
            created_at=base_time,
            sequence_number=1
        )
        place_event.checksum = place_event.calculate_checksum()
        
        cancel_event = AuditEvent(
            event_id=str(uuid4()),
            event_type="ORDER_CANCELLED",
            aggregate_id=order_id,
            aggregate_type="Order",
            event_data={"order_id": order_id},
            event_metadata={},
            created_at=base_time + timedelta(milliseconds=500),
            sequence_number=2
        )
        cancel_event.checksum = cancel_event.calculate_checksum()
        
        events = [place_event, cancel_event]
        
        anomalies = forensic_analyzer.detect_anomalies(events)
        
        rapid_cancellations = [a for a in anomalies if a["type"] == "rapid_cancellation"]
        assert len(rapid_cancellations) == 1
        assert rapid_cancellations[0]["order_id"] == order_id
        assert rapid_cancellations[0]["duration_seconds"] < 1
    
    def test_anomaly_detection_position_size_spike(self, forensic_analyzer):
        """Test detection of position size spikes."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create position with size exceeding limit
        event = AuditEvent(
            event_id=str(uuid4()),
            event_type="POSITION_OPENED",
            aggregate_id=str(uuid4()),
            aggregate_type="Position",
            event_data={
                "position_id": str(uuid4()),
                "symbol": "BTC/USDT",
                "dollar_value": "15000"  # Exceeds default limit of 10000
            },
            event_metadata={},
            created_at=base_time,
            sequence_number=1
        )
        event.checksum = event.calculate_checksum()
        events.append(event)
        
        anomalies = forensic_analyzer.detect_anomalies(events)
        
        position_spikes = [a for a in anomalies if a["type"] == "position_size_spike"]
        assert len(position_spikes) == 1
        assert Decimal(position_spikes[0]["size"]) > Decimal("10000")
    
    def test_time_gap_detection(self, forensic_analyzer):
        """Test detection of time gaps in event stream."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        aggregate_id = str(uuid4())
        
        # Create events with a 2-hour gap
        for i in range(3):
            time_offset = timedelta(minutes=i) if i < 2 else timedelta(hours=2, minutes=i)
            
            event = AuditEvent(
                event_id=str(uuid4()),
                event_type="POSITION_UPDATED",
                aggregate_id=aggregate_id,
                aggregate_type="Position",
                event_data={"update": i},
                event_metadata={},
                created_at=base_time + time_offset,
                sequence_number=i + 1
            )
            event.checksum = event.calculate_checksum()
            events.append(event)
        
        result = forensic_analyzer.verify_event_chain(events)
        
        assert len(result.time_gaps) == 1
        gap_start, gap_end, gap_seconds = result.time_gaps[0]
        assert gap_seconds > 3600  # More than 1 hour
    
    def test_audit_report_generation(self, forensic_analyzer, sample_event_chain):
        """Test comprehensive audit report generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            forensic_analyzer.generate_audit_report(sample_event_chain, output_path)
            
            # Verify report was created
            assert Path(output_path).exists()
            
            # Load and verify report content
            with open(output_path, 'r') as f:
                report = json.load(f)
            
            assert "audit_report" in report
            audit_report = report["audit_report"]
            
            assert "generated_at" in audit_report
            assert audit_report["summary"]["total_events"] == 5
            assert audit_report["verification"]["integrity_verified"] is True
            assert "recommendations" in audit_report
            assert len(audit_report["recommendations"]) > 0
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_multiple_aggregate_replay(self, forensic_analyzer):
        """Test event replay with multiple aggregates."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create events for two different positions
        for position_num in range(2):
            aggregate_id = str(uuid4())
            
            for i in range(3):
                event = AuditEvent(
                    event_id=str(uuid4()),
                    event_type="POSITION_OPENED" if i == 0 else "POSITION_UPDATED",
                    aggregate_id=aggregate_id,
                    aggregate_type="Position",
                    event_data={
                        "position_id": aggregate_id,
                        "symbol": f"BTC/USDT",
                        "update": i
                    },
                    event_metadata={},
                    created_at=base_time + timedelta(minutes=position_num * 10 + i),
                    sequence_number=i + 1
                )
                event.checksum = event.calculate_checksum()
                events.append(event)
        
        state = forensic_analyzer.replay_events(events)
        
        assert len(state) == 2  # Two aggregates
        for aggregate_id, aggregate_state in state.items():
            assert aggregate_state["aggregate_type"] == "Position"
            assert len(aggregate_state["events"]) == 3
    
    def test_time_reversal_detection(self, forensic_analyzer):
        """Test detection of events with reversed timestamps."""
        events = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        aggregate_id = str(uuid4())
        
        # Create events with time going backwards
        for i in range(3):
            # Second event has earlier timestamp than first
            if i == 1:
                time_offset = timedelta(minutes=-1)
            else:
                time_offset = timedelta(minutes=i)
            
            event = AuditEvent(
                event_id=str(uuid4()),
                event_type="POSITION_UPDATED",
                aggregate_id=aggregate_id,
                aggregate_type="Position",
                event_data={"update": i},
                event_metadata={},
                created_at=base_time + time_offset,
                sequence_number=i + 1
            )
            event.checksum = event.calculate_checksum()
            events.append(event)
        
        result = forensic_analyzer.verify_event_chain(events)
        
        assert result.integrity_verified is False
        time_reversals = [
            p for p in result.suspicious_patterns
            if p["type"] == "time_reversal"
        ]
        assert len(time_reversals) > 0
        assert time_reversals[0]["time_diff"] < 0