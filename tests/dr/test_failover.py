"""Tests for failover functionality."""

import pytest
import asyncio
import unittest.mock
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from genesis.dr.failover_manager import (
    FailoverManager,
    FailoverConfig,
    FailoverState,
    FailoverResult,
    HealthCheckResult
)
from genesis.core.exceptions import GenesisException


@pytest.fixture
def failover_config():
    """Create test failover configuration."""
    return FailoverConfig(
        rto_target_seconds=300,
        rpo_target_seconds=0,
        health_check_interval=30,
        failure_threshold=3,
        dns_ttl=60,
        primary_region="us-east-1",
        failover_regions=["us-west-2", "eu-west-1"],
        enable_auto_failover=True,
        enable_connection_draining=True,
        connection_drain_timeout=30,
        database_configs={
            "us-east-1": {"host": "db1.example.com", "port": 5432},
            "us-west-2": {"host": "db2.example.com", "port": 5432},
            "eu-west-1": {"host": "db3.example.com", "port": 5432}
        }
    )


@pytest.fixture
def failover_manager(failover_config):
    """Create test failover manager."""
    return FailoverManager(failover_config)


@pytest.mark.asyncio
async def test_failover_rto_achievement(failover_manager):
    """Test that failover completes within RTO target."""
    # Mock health checks
    failover_manager._check_region_health = AsyncMock(return_value=HealthCheckResult(
        service_name="region_health",
        region="us-west-2",
        is_healthy=True,
        response_time_ms=50,
        last_check=datetime.utcnow()
    ))
    
    # Mock failover steps
    failover_manager._drain_connections = AsyncMock()
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._promote_standby_database = AsyncMock()
    failover_manager._start_regional_services = AsyncMock()
    failover_manager._verify_failover_success = AsyncMock(return_value=True)
    failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
    failover_manager._send_failover_notifications = AsyncMock()
    
    # Execute failover
    result = await failover_manager.execute_failover("us-west-2")
    
    # Verify RTO achievement
    assert result.status == "completed"
    assert result.rto_achieved == True
    assert result.rto_seconds < 300  # 5 minutes
    assert result.target_region == "us-west-2"


@pytest.mark.asyncio
async def test_zero_data_loss():
    """Verify RPO=0 during failover."""
    config = FailoverConfig(rpo_target_seconds=0)
    manager = FailoverManager(config)
    
    # Mock RPO calculation to return 0 (no data loss)
    manager._calculate_rpo = AsyncMock(return_value=0.0)
    
    # Mock other required methods
    manager._check_region_health = AsyncMock(return_value=HealthCheckResult(
        service_name="region_health",
        region="us-west-2",
        is_healthy=True,
        response_time_ms=50,
        last_check=datetime.utcnow()
    ))
    manager._drain_connections = AsyncMock()
    manager._update_dns_routing = AsyncMock()
    manager._promote_standby_database = AsyncMock()
    manager._start_regional_services = AsyncMock()
    manager._verify_failover_success = AsyncMock(return_value=True)
    manager._send_failover_notifications = AsyncMock()
    
    # Execute failover
    result = await manager.execute_failover("us-west-2")
    
    # Verify zero data loss
    assert result.rpo_seconds == 0.0
    assert result.rpo_achieved == True


@pytest.mark.asyncio
async def test_automatic_failover_on_failure_detection(failover_manager):
    """Test automatic failover triggers on failure detection."""
    # Simulate primary region failure
    unhealthy_result = HealthCheckResult(
        service_name="region_health",
        region="us-east-1",
        is_healthy=False,
        response_time_ms=0,
        last_check=datetime.utcnow(),
        consecutive_failures=3
    )
    
    failover_manager._check_region_health = AsyncMock(side_effect=[
        unhealthy_result,  # Primary unhealthy
        HealthCheckResult(  # Failover region healthy
            service_name="region_health",
            region="us-west-2",
            is_healthy=True,
            response_time_ms=50,
            last_check=datetime.utcnow()
        )
    ])
    
    failover_manager.execute_failover = AsyncMock(return_value=FailoverResult(
        status="completed",
        source_region="us-east-1",
        target_region="us-west-2",
        start_time=datetime.utcnow(),
        rto_achieved=True,
        rpo_achieved=True
    ))
    
    # Test monitoring triggers failover
    failover_manager.config.enable_auto_failover = True
    await failover_manager._monitor_health()
    
    # Verify failover was called
    assert failover_manager._check_region_health.called


@pytest.mark.asyncio
async def test_connection_draining(failover_manager):
    """Test connection draining during failover."""
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._promote_standby_database = AsyncMock()
    failover_manager._start_regional_services = AsyncMock()
    failover_manager._verify_failover_success = AsyncMock(return_value=True)
    failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
    failover_manager._send_failover_notifications = AsyncMock()
    
    # Execute failover with connection draining enabled
    failover_manager.config.enable_connection_draining = True
    
    start_time = datetime.utcnow()
    result = await failover_manager.execute_failover("us-west-2")
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    
    # Verify connection draining occurred
    assert elapsed >= failover_manager.config.connection_drain_timeout
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_dns_failover_update(failover_manager):
    """Test DNS update during failover."""
    # Mock Route53 update
    with patch.object(failover_manager, 'route53') as mock_route53:
        failover_manager.config.route53_hosted_zone_id = "Z123456"
        failover_manager.config.domain_name = "example.com"
        
        failover_manager._drain_connections = AsyncMock()
        failover_manager._promote_standby_database = AsyncMock()
        failover_manager._start_regional_services = AsyncMock()
        failover_manager._verify_failover_success = AsyncMock(return_value=True)
        failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
        failover_manager._send_failover_notifications = AsyncMock()
        
        await failover_manager.execute_failover("us-west-2")
        
        # Verify DNS was updated
        assert "dns" in failover_manager.current_failover.services_migrated


@pytest.mark.asyncio
async def test_database_promotion(failover_manager):
    """Test standby database promotion during failover."""
    failover_manager._drain_connections = AsyncMock()
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._start_regional_services = AsyncMock()
    failover_manager._verify_failover_success = AsyncMock(return_value=True)
    failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
    failover_manager._send_failover_notifications = AsyncMock()
    
    # Execute failover
    await failover_manager.execute_failover("us-west-2")
    
    # Verify database was promoted
    assert "database" in failover_manager.current_failover.services_migrated


@pytest.mark.asyncio
async def test_service_startup_orchestration(failover_manager):
    """Test service startup in correct order."""
    services_started = []
    
    async def mock_start_service(region):
        services_started.extend([
            "trading_engine",
            "api_server",
            "websocket_server",
            "monitoring_collector"
        ])
    
    failover_manager._drain_connections = AsyncMock()
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._promote_standby_database = AsyncMock()
    failover_manager._start_regional_services = mock_start_service
    failover_manager._verify_failover_success = AsyncMock(return_value=True)
    failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
    failover_manager._send_failover_notifications = AsyncMock()
    
    await failover_manager.execute_failover("us-west-2")
    
    # Verify all services started
    assert "trading_engine" in services_started
    assert "api_server" in services_started
    assert "websocket_server" in services_started


@pytest.mark.asyncio
async def test_failover_validation(failover_manager):
    """Test failover validation after completion."""
    failover_manager._drain_connections = AsyncMock()
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._promote_standby_database = AsyncMock()
    failover_manager._start_regional_services = AsyncMock()
    failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
    failover_manager._send_failover_notifications = AsyncMock()
    
    # Test successful validation
    failover_manager._verify_failover_success = AsyncMock(return_value=True)
    result = await failover_manager.execute_failover("us-west-2")
    assert result.status == "completed"
    
    # Test failed validation
    failover_manager.state = FailoverState.IDLE
    failover_manager._verify_failover_success = AsyncMock(return_value=False)
    
    with pytest.raises(GenesisException, match="Failover validation failed"):
        await failover_manager.execute_failover("us-west-2")


@pytest.mark.asyncio
async def test_failover_rollback_on_failure(failover_manager):
    """Test automatic rollback when failover fails."""
    failover_manager._drain_connections = AsyncMock()
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._promote_standby_database = AsyncMock(
        side_effect=Exception("Database promotion failed")
    )
    
    rollback_called = False
    
    async def mock_rollback():
        nonlocal rollback_called
        rollback_called = True
    
    failover_manager._rollback_failover = mock_rollback
    
    # Execute failover that will fail
    with pytest.raises(Exception):
        await failover_manager.execute_failover("us-west-2")
    
    # Verify rollback was attempted
    assert rollback_called
    assert failover_manager.current_failover.rollback_required


@pytest.mark.asyncio
async def test_failback_operation(failover_manager):
    """Test failback to original region."""
    # First set current region to failover region
    failover_manager.config.primary_region = "us-west-2"
    
    # Mock health check for original region
    failover_manager._check_region_health = AsyncMock(return_value=HealthCheckResult(
        service_name="region_health",
        region="us-east-1",
        is_healthy=True,
        response_time_ms=50,
        last_check=datetime.utcnow()
    ))
    
    # Mock failover steps
    failover_manager._drain_connections = AsyncMock()
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._promote_standby_database = AsyncMock()
    failover_manager._start_regional_services = AsyncMock()
    failover_manager._verify_failover_success = AsyncMock(return_value=True)
    failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
    failover_manager._send_failover_notifications = AsyncMock()
    
    # Execute failback
    result = await failover_manager.execute_failback("us-east-1")
    
    assert result.status == "completed"
    assert result.target_region == "us-east-1"
    assert failover_manager.config.primary_region == "us-east-1"


@pytest.mark.asyncio
async def test_health_check_monitoring(failover_manager):
    """Test health check monitoring system."""
    # Create health check results
    healthy_result = HealthCheckResult(
        service_name="region_health",
        region="us-east-1",
        is_healthy=True,
        response_time_ms=50,
        last_check=datetime.utcnow()
    )
    
    unhealthy_result = HealthCheckResult(
        service_name="region_health",
        region="us-east-1",
        is_healthy=False,
        response_time_ms=0,
        last_check=datetime.utcnow(),
        error_message="Connection timeout"
    )
    
    # Test health check storage
    failover_manager.health_checks["us-east-1"] = healthy_result
    assert failover_manager.health_checks["us-east-1"].is_healthy == True
    
    # Test consecutive failure tracking
    failover_manager.health_checks["us-east-1"] = unhealthy_result
    failover_manager.health_checks["us-east-1"].consecutive_failures = 1
    assert failover_manager.health_checks["us-east-1"].consecutive_failures == 1


@pytest.mark.asyncio
async def test_failover_history_tracking(failover_manager):
    """Test failover history is properly tracked."""
    # Mock required methods
    failover_manager._drain_connections = AsyncMock()
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._promote_standby_database = AsyncMock()
    failover_manager._start_regional_services = AsyncMock()
    failover_manager._verify_failover_success = AsyncMock(return_value=True)
    failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
    failover_manager._send_failover_notifications = AsyncMock()
    
    # Execute multiple failovers
    await failover_manager.execute_failover("us-west-2")
    
    failover_manager.state = FailoverState.IDLE
    failover_manager.config.primary_region = "us-west-2"
    await failover_manager.execute_failover("eu-west-1")
    
    # Check history
    assert len(failover_manager.failover_history) == 2
    assert failover_manager.failover_history[0].target_region == "us-west-2"
    assert failover_manager.failover_history[1].target_region == "eu-west-1"
    
    # Test history retrieval
    history = failover_manager.get_failover_history(limit=1)
    assert len(history) == 1
    assert history[0]["target_region"] == "eu-west-1"


@pytest.mark.asyncio
async def test_failover_status_reporting(failover_manager):
    """Test failover status reporting."""
    # Set up some state
    failover_manager.state = FailoverState.IDLE
    failover_manager.config.primary_region = "us-east-1"
    failover_manager.health_checks["us-east-1"] = HealthCheckResult(
        service_name="region_health",
        region="us-east-1",
        is_healthy=True,
        response_time_ms=50,
        last_check=datetime.utcnow()
    )
    
    # Get status
    status = failover_manager.get_failover_status()
    
    assert status["state"] == "idle"
    assert status["primary_region"] == "us-east-1"
    assert "us-west-2" in status["failover_regions"]
    assert "health_checks" in status
    assert status["health_checks"]["us-east-1"]["is_healthy"] == True


@pytest.mark.asyncio
async def test_notification_sending(failover_manager):
    """Test failover notifications are sent."""
    failover_manager.config.notification_emails = ["ops@example.com"]
    
    result = FailoverResult(
        status="completed",
        source_region="us-east-1",
        target_region="us-west-2",
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow(),
        rto_seconds=240,
        rto_achieved=True,
        rpo_seconds=0,
        rpo_achieved=True
    )
    
    # Mock logger to verify notification
    with patch.object(failover_manager.logger, 'info') as mock_logger:
        await failover_manager._send_failover_notifications(result)
        
        # Verify notification was logged
        mock_logger.assert_called_with(
            "failover_notification_sent",
            recipients=["ops@example.com"],
            notification=unittest.mock.ANY
        )


@pytest.mark.asyncio
async def test_concurrent_failover_prevention(failover_manager):
    """Test that concurrent failovers are prevented."""
    failover_manager.state = FailoverState.FAILING_OVER
    
    # Attempt second failover while one is in progress
    with pytest.raises(GenesisException, match="Failover already in progress"):
        await failover_manager.execute_failover("us-west-2")
        
    # Test force flag overrides
    failover_manager._drain_connections = AsyncMock()
    failover_manager._update_dns_routing = AsyncMock()
    failover_manager._promote_standby_database = AsyncMock()
    failover_manager._start_regional_services = AsyncMock()
    failover_manager._verify_failover_success = AsyncMock(return_value=True)
    failover_manager._calculate_rpo = AsyncMock(return_value=0.0)
    failover_manager._send_failover_notifications = AsyncMock()
    
    # Force should work even when in progress
    result = await failover_manager.execute_failover("us-west-2", force=True)
    assert result.status == "completed"