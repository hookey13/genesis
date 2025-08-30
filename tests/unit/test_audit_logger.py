"""Unit tests for audit logging system."""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from genesis.security.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    get_audit_logger,
    log_authentication,
    log_permission_check
)


class TestAuditEvent:
    """Test AuditEvent class."""
    
    def test_audit_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id="test_user",
            ip_address="192.168.1.1",
            resource="api",
            action="login",
            result="success",
            metadata={"method": "password"}
        )
        
        assert event.event_type == AuditEventType.AUTH_SUCCESS
        assert event.user_id == "test_user"
        assert event.ip_address == "192.168.1.1"
        assert event.result == "success"
    
    def test_checksum_calculation(self):
        """Test checksum calculation for tamper detection."""
        event = AuditEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id="test_user",
            ip_address="192.168.1.1",
            resource="api",
            action="login",
            result="success",
            metadata={}
        )
        
        # Calculate checksum
        checksum1 = event.calculate_checksum()
        assert checksum1 is not None
        assert len(checksum1) == 64  # SHA256 hex string
        
        # Same event should produce same checksum
        checksum2 = event.calculate_checksum()
        assert checksum1 == checksum2
        
        # Checksum with previous should be different
        checksum3 = event.calculate_checksum(previous_checksum="previous123")
        assert checksum3 != checksum1
    
    def test_event_to_json(self):
        """Test converting event to JSON."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        event = AuditEvent(
            timestamp=timestamp,
            event_type=AuditEventType.PERMISSION_DENIED,
            user_id="test_user",
            ip_address="192.168.1.1",
            resource="orders",
            action="delete",
            result="failure",
            metadata={"reason": "insufficient_permissions"},
            checksum="test_checksum"
        )
        
        json_str = event.to_json()
        data = json.loads(json_str)
        
        assert data["timestamp"] == timestamp.isoformat()
        assert data["event_type"] == "permission_denied"
        assert data["user_id"] == "test_user"
        assert data["resource"] == "orders"
        assert data["checksum"] == "test_checksum"


@pytest.mark.asyncio
class TestAuditLogger:
    """Test AuditLogger class."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    async def audit_logger(self, temp_log_dir):
        """Create an audit logger instance."""
        logger = AuditLogger(
            log_dir=temp_log_dir,
            retention_days=7,
            max_file_size_mb=1,
            enable_compression=True,
            enable_tamper_protection=True
        )
        yield logger
    
    async def test_log_event(self, audit_logger):
        """Test logging an audit event."""
        await audit_logger.log_event(
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id="test_user",
            ip_address="192.168.1.1",
            resource="api",
            action="login",
            result="success",
            metadata={"method": "api_key"}
        )
        
        # Check that event was logged
        assert len(audit_logger.recent_events) == 1
        event = audit_logger.recent_events[0]
        assert event.event_type == AuditEventType.AUTH_SUCCESS
        assert event.user_id == "test_user"
        
        # Check that file was created
        assert audit_logger.current_log_file.exists()
    
    async def test_checksum_chain(self, audit_logger):
        """Test that checksums are chained properly."""
        # Log multiple events
        await audit_logger.log_event(
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id="user1",
            ip_address="192.168.1.1",
            result="success"
        )
        
        await audit_logger.log_event(
            event_type=AuditEventType.ORDER_PLACED,
            user_id="user1",
            resource="orders",
            result="success"
        )
        
        # Verify checksums are different and chained
        events = list(audit_logger.recent_events)
        assert len(events) == 2
        assert events[0].checksum != events[1].checksum
        assert events[0].checksum is not None
        assert events[1].checksum is not None
    
    async def test_log_rotation(self, audit_logger):
        """Test log file rotation."""
        # Set small file size to trigger rotation
        audit_logger.max_file_size_bytes = 100  # Very small
        
        # Log events until rotation occurs
        for i in range(10):
            await audit_logger.log_event(
                event_type=AuditEventType.AUTH_SUCCESS,
                user_id=f"user{i}",
                result="success",
                metadata={"index": i}
            )
        
        # Check that rotation occurred
        log_files = list(Path(audit_logger.log_dir).glob("audit_*.log*"))
        assert len(log_files) >= 2  # Current and at least one rotated
    
    def test_verify_integrity_valid(self, audit_logger, temp_log_dir):
        """Test integrity verification with valid log."""
        # Create a valid log file
        log_file = Path(temp_log_dir) / "test_audit.log"
        
        # Create events with proper checksums
        previous_checksum = None
        events_data = []
        
        for i in range(3):
            event = AuditEvent(
                timestamp=datetime.now(),
                event_type=AuditEventType.AUTH_SUCCESS,
                user_id=f"user{i}",
                ip_address="192.168.1.1",
                resource="api",
                action="login",
                result="success",
                metadata={}
            )
            
            checksum = event.calculate_checksum(previous_checksum)
            event.checksum = checksum
            events_data.append(json.loads(event.to_json()))
            previous_checksum = checksum
        
        # Write to file
        with open(log_file, 'w') as f:
            for event_data in events_data:
                f.write(json.dumps(event_data) + '\n')
        
        # Verify integrity
        assert audit_logger.verify_integrity(log_file) is True
    
    def test_verify_integrity_tampered(self, audit_logger, temp_log_dir):
        """Test integrity verification with tampered log."""
        # Create a log file with invalid checksums
        log_file = Path(temp_log_dir) / "tampered_audit.log"
        
        events_data = []
        for i in range(3):
            event = AuditEvent(
                timestamp=datetime.now(),
                event_type=AuditEventType.AUTH_SUCCESS,
                user_id=f"user{i}",
                ip_address="192.168.1.1",
                resource="api",
                action="login",
                result="success",
                metadata={}
            )
            
            # Use wrong checksum
            event.checksum = "invalid_checksum"
            events_data.append(json.loads(event.to_json()))
        
        # Write to file
        with open(log_file, 'w') as f:
            for event_data in events_data:
                f.write(json.dumps(event_data) + '\n')
        
        # Verify integrity should fail
        assert audit_logger.verify_integrity(log_file) is False
    
    async def test_search_events(self, audit_logger):
        """Test searching audit events."""
        # Log various events
        await audit_logger.log_event(
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id="user1",
            ip_address="192.168.1.1",
            result="success"
        )
        
        await audit_logger.log_event(
            event_type=AuditEventType.AUTH_FAILURE,
            user_id="user2",
            ip_address="192.168.1.2",
            result="failure"
        )
        
        await audit_logger.log_event(
            event_type=AuditEventType.PERMISSION_DENIED,
            user_id="user1",
            ip_address="192.168.1.1",
            resource="orders",
            result="failure"
        )
        
        # Search by event type
        auth_events = audit_logger.search_events(
            event_type=AuditEventType.AUTH_SUCCESS
        )
        assert len(auth_events) == 1
        assert auth_events[0].user_id == "user1"
        
        # Search by user
        user1_events = audit_logger.search_events(user_id="user1")
        assert len(user1_events) == 2
        
        # Search by result
        failure_events = audit_logger.search_events(result="failure")
        assert len(failure_events) == 2
        
        # Search by IP
        ip_events = audit_logger.search_events(ip_address="192.168.1.1")
        assert len(ip_events) == 2
    
    async def test_get_statistics(self, audit_logger):
        """Test getting audit statistics."""
        # Log various events
        await audit_logger.log_event(
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id="user1",
            ip_address="192.168.1.1",
            result="success"
        )
        
        await audit_logger.log_event(
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id="user2",
            ip_address="192.168.1.2",
            result="success"
        )
        
        await audit_logger.log_event(
            event_type=AuditEventType.AUTH_FAILURE,
            user_id="user3",
            ip_address="192.168.1.3",
            result="failure"
        )
        
        # Get statistics
        stats = audit_logger.get_statistics()
        
        assert stats["total_events"] == 3
        assert stats["by_type"]["auth_success"] == 2
        assert stats["by_type"]["auth_failure"] == 1
        assert stats["by_result"]["success"] == 2
        assert stats["by_result"]["failure"] == 1
        assert stats["unique_users"] == 3
        assert stats["unique_ips"] == 3
    
    async def test_cleanup_old_logs(self, audit_logger, temp_log_dir):
        """Test cleanup of old log files."""
        # Create old log files
        old_date = datetime.now() - timedelta(days=100)
        old_file = Path(temp_log_dir) / f"audit_{old_date.strftime('%Y%m%d')}.log"
        old_file.write_text("old log content")
        
        recent_date = datetime.now() - timedelta(days=1)
        recent_file = Path(temp_log_dir) / f"audit_{recent_date.strftime('%Y%m%d')}.log"
        recent_file.write_text("recent log content")
        
        # Run cleanup
        await audit_logger.cleanup_old_logs()
        
        # Old file should be deleted
        assert not old_file.exists()
        # Recent file should remain
        assert recent_file.exists()


@pytest.mark.asyncio
class TestAuditLoggerHelpers:
    """Test helper functions for audit logging."""
    
    async def test_log_authentication(self):
        """Test authentication logging helper."""
        with patch('genesis.security.audit_logger.get_audit_logger') as mock_get_logger:
            mock_logger = AsyncMock()
            mock_get_logger.return_value = mock_logger
            
            await log_authentication(
                user_id="test_user",
                ip_address="192.168.1.1",
                success=True,
                metadata={"method": "password"}
            )
            
            mock_logger.log_event.assert_called_once()
            call_args = mock_logger.log_event.call_args[1]
            assert call_args["event_type"] == AuditEventType.AUTH_SUCCESS
            assert call_args["user_id"] == "test_user"
            assert call_args["ip_address"] == "192.168.1.1"
            assert call_args["result"] == "success"
    
    async def test_log_permission_check(self):
        """Test permission check logging helper."""
        with patch('genesis.security.audit_logger.get_audit_logger') as mock_get_logger:
            mock_logger = AsyncMock()
            mock_get_logger.return_value = mock_logger
            
            await log_permission_check(
                user_id="test_user",
                resource="orders",
                action="delete",
                granted=False,
                ip_address="192.168.1.1",
                metadata={"reason": "insufficient_role"}
            )
            
            mock_logger.log_event.assert_called_once()
            call_args = mock_logger.log_event.call_args[1]
            assert call_args["event_type"] == AuditEventType.PERMISSION_DENIED
            assert call_args["user_id"] == "test_user"
            assert call_args["resource"] == "orders"
            assert call_args["action"] == "delete"
            assert call_args["result"] == "failure"