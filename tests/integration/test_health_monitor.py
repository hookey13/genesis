"""Integration tests for health monitoring and self-healing system."""

import pytest
import asyncio
import psutil
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch, call
import json

from genesis.operations.health_monitor import (
    HealthMonitor,
    HealthCheck,
    HealthStatus,
    RemediationAction,
    IssueType,
    HealthDashboard,
    EscalationManager
)


class TestHealthCheck:
    """Test health check definitions and execution."""
    
    def test_health_check_levels(self):
        """Test different health check levels."""
        shallow = HealthCheck(
            name="api_connectivity",
            level="shallow",
            timeout=5,
            critical=True
        )
        
        deep = HealthCheck(
            name="database_integrity",
            level="deep",
            timeout=30,
            critical=False
        )
        
        diagnostic = HealthCheck(
            name="full_system_audit",
            level="diagnostic",
            timeout=300,
            critical=False
        )
        
        assert shallow.timeout < deep.timeout < diagnostic.timeout
        assert shallow.critical
        assert not diagnostic.critical
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self):
        """Test executing a health check."""
        check = HealthCheck(
            name="test_check",
            level="shallow",
            function=AsyncMock(return_value={"status": "healthy", "latency": 10})
        )
        
        result = await check.execute()
        
        assert result["status"] == "healthy"
        assert result["latency"] == 10
        check.function.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test health check timeout handling."""
        async def slow_check():
            await asyncio.sleep(10)
            return {"status": "healthy"}
        
        check = HealthCheck(
            name="slow_check",
            level="shallow",
            function=slow_check,
            timeout=0.1
        )
        
        with pytest.raises(asyncio.TimeoutError):
            await check.execute()


class TestHealthStatus:
    """Test health status tracking and reporting."""
    
    def test_status_initialization(self):
        """Test health status initialization."""
        status = HealthStatus()
        
        assert status.overall == "healthy"
        assert status.checks == {}
        assert status.last_check is None
        assert status.issues == []
    
    def test_status_aggregation(self):
        """Test aggregating multiple check results."""
        status = HealthStatus()
        
        status.update_check("api", {"status": "healthy", "latency": 50})
        status.update_check("database", {"status": "healthy", "latency": 10})
        status.update_check("memory", {"status": "warning", "usage": 85})
        
        assert status.overall == "warning"  # Worst status wins
        assert len(status.checks) == 3
        assert status.checks["memory"]["status"] == "warning"
    
    def test_status_history(self):
        """Test maintaining status history."""
        status = HealthStatus()
        
        # Add multiple status updates
        for i in range(10):
            status.update_check("test", {"status": "healthy", "value": i})
        
        history = status.get_history("test", limit=5)
        assert len(history) == 5
        assert history[-1]["value"] == 9  # Most recent
    
    def test_issue_tracking(self):
        """Test tracking health issues."""
        status = HealthStatus()
        
        issue1 = {"type": "memory_high", "severity": "warning"}
        issue2 = {"type": "api_degraded", "severity": "critical"}
        
        status.add_issue(issue1)
        status.add_issue(issue2)
        
        assert len(status.issues) == 2
        
        critical_issues = status.get_issues_by_severity("critical")
        assert len(critical_issues) == 1
        assert critical_issues[0]["type"] == "api_degraded"


class TestRemediationAction:
    """Test automated remediation actions."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_reset(self):
        """Test resetting database connection pool."""
        action = RemediationAction(
            issue_type=IssueType.CONNECTION_POOL_EXHAUSTION,
            action="reset_connections"
        )
        
        with patch('genesis.operations.health_monitor.get_db_pool') as mock_pool:
            mock_pool.return_value = MagicMock(
                reset=AsyncMock(),
                size=10,
                available=0
            )
            
            result = await action.execute()
            
            assert result.success
            assert "reset" in result.message.lower()
            mock_pool.return_value.reset.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """Test memory cleanup remediation."""
        action = RemediationAction(
            issue_type=IssueType.MEMORY_LEAK,
            action="cleanup_memory"
        )
        
        with patch('gc.collect') as mock_gc:
            with patch('genesis.operations.health_monitor.clear_caches') as mock_clear:
                mock_gc.return_value = 1000  # Objects collected
                
                result = await action.execute()
                
                assert result.success
                assert "memory" in result.message.lower()
                mock_gc.assert_called()
                mock_clear.assert_called()
    
    @pytest.mark.asyncio
    async def test_disk_space_cleanup(self):
        """Test disk space cleanup."""
        action = RemediationAction(
            issue_type=IssueType.DISK_SPACE_LOW,
            action="cleanup_logs"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test log files
            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()
            
            for i in range(10):
                log_file = log_dir / f"old_log_{i}.log"
                log_file.write_text("x" * 1000000)  # 1MB each
            
            with patch('genesis.operations.health_monitor.LOG_DIR', log_dir):
                result = await action.execute(
                    max_age_days=0,  # Delete all for test
                    max_size_mb=5    # Keep only 5MB
                )
                
                assert result.success
                remaining_files = list(log_dir.glob("*.log"))
                assert len(remaining_files) < 10  # Some deleted
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self):
        """Test activating circuit breaker for API degradation."""
        action = RemediationAction(
            issue_type=IssueType.API_DEGRADATION,
            action="activate_circuit_breaker"
        )
        
        with patch('genesis.operations.health_monitor.circuit_breaker') as mock_cb:
            mock_cb.activate = AsyncMock()
            mock_cb.is_active = False
            
            result = await action.execute(
                service="binance_api",
                duration_seconds=60
            )
            
            assert result.success
            mock_cb.activate.assert_called_once_with("binance_api", 60)
    
    @pytest.mark.asyncio
    async def test_controlled_restart(self):
        """Test controlled service restart."""
        action = RemediationAction(
            issue_type=IssueType.MEMORY_LEAK,
            action="controlled_restart"
        )
        
        with patch('genesis.operations.health_monitor.save_state') as mock_save:
            with patch('genesis.operations.health_monitor.graceful_shutdown') as mock_shutdown:
                with patch('os.execv') as mock_exec:
                    mock_save.return_value = True
                    mock_shutdown.return_value = True
                    
                    # Should save state and prepare for restart
                    result = await action.prepare_restart()
                    
                    assert result.ready
                    mock_save.assert_called_once()
                    mock_shutdown.assert_called_once()


class TestHealthMonitor:
    """Test main health monitoring system."""
    
    @pytest.mark.asyncio
    async def test_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor()
        
        assert len(monitor.checks) > 0
        assert monitor.status.overall == "healthy"
        assert monitor.remediation_enabled is True
        assert monitor.escalation_threshold == 3
    
    @pytest.mark.asyncio
    async def test_shallow_health_checks(self):
        """Test running shallow health checks."""
        monitor = HealthMonitor()
        
        # Mock check functions
        monitor.checks = [
            HealthCheck("api", "shallow", AsyncMock(return_value={"status": "healthy"})),
            HealthCheck("db", "shallow", AsyncMock(return_value={"status": "healthy"})),
            HealthCheck("redis", "shallow", AsyncMock(return_value={"status": "healthy"}))
        ]
        
        await monitor.run_shallow_checks()
        
        assert monitor.status.overall == "healthy"
        for check in monitor.checks:
            if check.level == "shallow":
                check.function.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deep_health_checks(self):
        """Test running deep health checks."""
        monitor = HealthMonitor()
        
        # Mock checks with different results
        monitor.checks = [
            HealthCheck("db_integrity", "deep", 
                       AsyncMock(return_value={"status": "healthy", "tables": 10})),
            HealthCheck("data_consistency", "deep",
                       AsyncMock(return_value={"status": "warning", "issues": 2}))
        ]
        
        await monitor.run_deep_checks()
        
        assert monitor.status.overall == "warning"  # Due to data_consistency
        assert len(monitor.status.issues) > 0
    
    @pytest.mark.asyncio
    async def test_diagnostic_checks(self):
        """Test running diagnostic checks."""
        monitor = HealthMonitor()
        
        monitor.checks = [
            HealthCheck("full_audit", "diagnostic",
                       AsyncMock(return_value={
                           "status": "healthy",
                           "audit_results": {
                               "trades": 1000,
                               "positions": 5,
                               "errors": 0
                           }
                       }))
        ]
        
        results = await monitor.run_diagnostic_checks()
        
        assert "full_audit" in results
        assert results["full_audit"]["audit_results"]["trades"] == 1000
    
    @pytest.mark.asyncio
    async def test_automatic_remediation(self):
        """Test automatic issue remediation."""
        monitor = HealthMonitor()
        monitor.remediation_enabled = True
        
        # Simulate connection pool exhaustion
        with patch.object(monitor, 'check_connection_pool') as mock_check:
            mock_check.return_value = {
                "status": "critical",
                "available": 0,
                "total": 10
            }
            
            with patch.object(monitor, 'reset_connection_pool') as mock_reset:
                mock_reset.return_value = True
                
                await monitor.detect_and_remediate()
                
                mock_reset.assert_called_once()
                assert len(monitor.remediation_history) > 0
    
    @pytest.mark.asyncio
    async def test_escalation_to_human(self):
        """Test escalation to human operator."""
        monitor = HealthMonitor()
        escalation = EscalationManager()
        monitor.escalation_manager = escalation
        
        with patch('genesis.operations.health_monitor.send_alert') as mock_alert:
            # Simulate repeated failures
            for i in range(4):
                monitor.status.add_issue({
                    "type": "critical_error",
                    "severity": "critical",
                    "timestamp": datetime.now()
                })
            
            await monitor.check_escalation_needed()
            
            mock_alert.assert_called()
            alert = mock_alert.call_args[0][0]
            assert "escalation" in alert.lower()
            assert "human intervention" in alert.lower()
    
    @pytest.mark.asyncio
    async def test_health_dashboard(self):
        """Test health status dashboard generation."""
        monitor = HealthMonitor()
        dashboard = HealthDashboard(monitor)
        
        # Add some test data
        monitor.status.update_check("api", {"status": "healthy", "latency": 50})
        monitor.status.update_check("database", {"status": "warning", "connections": 8})
        monitor.status.add_issue({"type": "high_latency", "severity": "warning"})
        
        # Generate dashboard
        data = await dashboard.get_dashboard_data()
        
        assert data["overall_status"] == "warning"
        assert len(data["checks"]) == 2
        assert len(data["active_issues"]) == 1
        assert "uptime" in data
        assert "last_check" in data
    
    @pytest.mark.asyncio
    async def test_remediation_rollback(self):
        """Test rolling back failed remediation."""
        monitor = HealthMonitor()
        
        with patch.object(monitor, 'apply_remediation') as mock_apply:
            # First attempt fails
            mock_apply.side_effect = [
                Exception("Remediation failed"),
                True  # Second attempt succeeds
            ]
            
            with patch.object(monitor, 'rollback_remediation') as mock_rollback:
                mock_rollback.return_value = True
                
                result = await monitor.safe_remediate(
                    issue_type=IssueType.MEMORY_LEAK
                )
                
                # Should rollback after failure
                mock_rollback.assert_called_once()
                assert result.rolled_back
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self):
        """Test continuous monitoring loop."""
        monitor = HealthMonitor()
        monitor.check_interval_seconds = 0.01  # Very short for testing
        
        with patch.object(monitor, 'run_shallow_checks') as mock_shallow:
            with patch('asyncio.sleep', side_effect=[None, asyncio.CancelledError()]):
                with pytest.raises(asyncio.CancelledError):
                    await monitor.start_monitoring()
                
                # Should have run checks at least once
                assert mock_shallow.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_health_metrics_export(self):
        """Test exporting health metrics to monitoring systems."""
        monitor = HealthMonitor()
        
        with patch('prometheus_client.Gauge') as mock_gauge:
            monitor.status.update_check("api", {"status": "healthy", "latency": 50})
            monitor.status.update_check("database", {"status": "healthy", "connections": 5})
            
            await monitor.export_metrics()
            
            # Should create gauges for health metrics
            assert mock_gauge.called
            gauge_calls = mock_gauge.call_args_list
            gauge_names = [call[0][0] for call in gauge_calls]
            assert any("health" in name for name in gauge_names)


class TestHealthIntegration:
    """Integration tests for complete health monitoring workflow."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_health_workflow(self):
        """Test complete health monitoring and remediation workflow."""
        monitor = HealthMonitor()
        
        # Phase 1: Detection
        with patch.object(monitor, 'check_memory_usage') as mock_memory:
            mock_memory.return_value = {
                "status": "critical",
                "usage_percent": 92,
                "available_mb": 500
            }
            
            # Phase 2: Diagnosis
            with patch.object(monitor, 'diagnose_memory_issue') as mock_diagnose:
                mock_diagnose.return_value = {
                    "cause": "memory_leak",
                    "process": "trading_engine",
                    "growth_rate_mb_per_hour": 100
                }
                
                # Phase 3: Remediation
                with patch.object(monitor, 'cleanup_memory') as mock_cleanup:
                    mock_cleanup.return_value = {
                        "freed_mb": 2000,
                        "new_usage_percent": 65
                    }
                    
                    # Run health check cycle
                    await monitor.run_health_cycle()
                    
                    # Verify workflow
                    mock_memory.assert_called()
                    mock_diagnose.assert_called()
                    mock_cleanup.assert_called()
                    
                    # Check status updated
                    assert monitor.status.checks["memory"]["status"] == "recovered"
                    
                    # Check remediation logged
                    assert len(monitor.remediation_history) > 0
                    last_action = monitor.remediation_history[-1]
                    assert last_action["issue_type"] == "memory_leak"
                    assert last_action["success"]
    
    @pytest.mark.asyncio
    async def test_cascading_failure_handling(self):
        """Test handling cascading failures."""
        monitor = HealthMonitor()
        
        # Simulate cascading failures
        failures = [
            ("database", "connection_timeout"),
            ("redis", "connection_refused"),
            ("api", "rate_limited")
        ]
        
        for service, error in failures:
            monitor.status.add_issue({
                "service": service,
                "error": error,
                "severity": "critical",
                "timestamp": datetime.now()
            })
        
        # Should detect pattern and take action
        with patch.object(monitor, 'emergency_mode') as mock_emergency:
            await monitor.detect_cascading_failure()
            
            mock_emergency.assert_called_once()
            
            # Should have activated circuit breakers
            assert monitor.circuit_breakers_active
            
            # Should have notified operators
            assert monitor.escalation_triggered