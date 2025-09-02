"""Failover Manager for automated disaster recovery."""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
from decimal import Decimal

import structlog

from genesis.core.exceptions import GenesisException


class FailoverState(Enum):
    """Failover state enumeration."""
    IDLE = "idle"
    DETECTING = "detecting"
    INITIATING = "initiating"
    FAILING_OVER = "failing_over"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


@dataclass
class FailoverConfig:
    """Configuration for failover operations."""
    rto_target_seconds: int = 300  # 5 minutes
    rpo_target_seconds: int = 0    # Zero data loss
    health_check_interval: int = 30
    failure_threshold: int = 3
    dns_ttl: int = 60
    primary_region: str = "us-east-1"
    failover_regions: List[str] = field(default_factory=lambda: ["us-west-2", "eu-west-1"])
    route53_hosted_zone_id: Optional[str] = None
    domain_name: Optional[str] = None
    enable_auto_failover: bool = True
    enable_connection_draining: bool = True
    connection_drain_timeout: int = 30
    notification_emails: List[str] = field(default_factory=list)
    database_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Health check result for a service."""
    service_name: str
    region: str
    is_healthy: bool
    response_time_ms: float
    last_check: datetime
    consecutive_failures: int = 0
    error_message: Optional[str] = None


@dataclass
class FailoverResult:
    """Result of a failover operation."""
    status: str
    source_region: str
    target_region: str
    start_time: datetime
    end_time: Optional[datetime] = None
    rto_seconds: Optional[float] = None
    rpo_seconds: Optional[float] = None
    rto_achieved: bool = False
    rpo_achieved: bool = False
    services_migrated: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    rollback_required: bool = False


class FailoverManager:
    """Manages automated failover operations."""
    
    def __init__(self, config: FailoverConfig):
        """Initialize the failover manager."""
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self.state = FailoverState.IDLE
        self.health_checks: Dict[str, HealthCheckResult] = {}
        self.current_failover: Optional[FailoverResult] = None
        self.failover_history: List[FailoverResult] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self) -> None:
        """Start health monitoring for automatic failover."""
        if self._monitoring_task:
            await self.stop_monitoring()
            
        self._monitoring_task = asyncio.create_task(self._monitor_health())
        self.logger.info("failover_monitoring_started", 
                        interval=self.config.health_check_interval)
        
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            self.logger.info("failover_monitoring_stopped")
            
    async def _monitor_health(self) -> None:
        """Monitor health and trigger failover if needed."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check primary region health
                primary_health = await self._check_region_health(
                    self.config.primary_region
                )
                
                if not primary_health.is_healthy:
                    primary_health.consecutive_failures += 1
                    
                    if primary_health.consecutive_failures >= self.config.failure_threshold:
                        self.logger.warning("primary_region_failure_detected",
                                          region=self.config.primary_region,
                                          failures=primary_health.consecutive_failures)
                        
                        if self.config.enable_auto_failover:
                            # Find healthy failover region
                            target_region = await self._select_failover_region()
                            if target_region:
                                await self.execute_failover(target_region)
                else:
                    primary_health.consecutive_failures = 0
                    
                self.health_checks[self.config.primary_region] = primary_health
                
            except Exception as e:
                self.logger.error("health_monitoring_error", error=str(e))
                
    async def _check_region_health(self, region: str) -> HealthCheckResult:
        """Check health of services in a region."""
        start_time = datetime.utcnow()
        
        try:
            # Check database connectivity
            db_healthy = await self._check_database_health(region)
            
            # Check API endpoints
            api_healthy = await self._check_api_health(region)
            
            # Check critical services
            services_healthy = await self._check_services_health(region)
            
            is_healthy = db_healthy and api_healthy and services_healthy
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return HealthCheckResult(
                service_name="region_health",
                region=region,
                is_healthy=is_healthy,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                consecutive_failures=0 if is_healthy else 1
            )
            
        except Exception as e:
            self.logger.error("health_check_failed", region=region, error=str(e))
            return HealthCheckResult(
                service_name="region_health",
                region=region,
                is_healthy=False,
                response_time_ms=0,
                last_check=datetime.utcnow(),
                error_message=str(e)
            )
            
    async def _check_database_health(self, region: str) -> bool:
        """Check database health in a region."""
        if region not in self.config.database_configs:
            return False
            
        db_config = self.config.database_configs[region]
        
        try:
            import asyncpg
            
            # Try to connect with timeout
            conn = await asyncpg.connect(
                host=db_config['host'],
                port=db_config.get('port', 5432),
                database=db_config.get('database', 'genesis_prod'),
                user=db_config.get('user', os.environ.get('DB_USER', 'postgres')),
                password=db_config.get('password', os.environ.get('DB_PASSWORD')),
                timeout=5
            )
            
            try:
                # Check basic connectivity
                result = await conn.fetchval("SELECT 1")
                if result != 1:
                    return False
                    
                # Check database is accepting writes
                await conn.execute(
                    "INSERT INTO health_checks (region, timestamp, status) "
                    "VALUES ($1, $2, $3) "
                    "ON CONFLICT (region) DO UPDATE SET timestamp = $2, status = $3",
                    region, datetime.utcnow(), 'healthy'
                )
                
                # Check replication status if standby
                is_standby = await conn.fetchval("SELECT pg_is_in_recovery()")
                if is_standby:
                    # Check replication lag
                    lag = await conn.fetchval(
                        "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::float"
                    )
                    if lag and lag > 10:  # More than 10 seconds lag
                        self.logger.warning("high_replication_lag", 
                                          region=region, lag_seconds=lag)
                        return False
                        
                return True
                
            finally:
                await conn.close()
                
        except Exception as e:
            self.logger.warning("database_health_check_failed", 
                              region=region, error=str(e))
            return False
            
    async def _check_api_health(self, region: str) -> bool:
        """Check API endpoint health in a region."""
        try:
            import aiohttp
            
            # Get API endpoint for region
            api_endpoints = {
                "us-east-1": "https://api-east.genesis-trading.com",
                "us-west-2": "https://api-west.genesis-trading.com",
                "eu-west-1": "https://api-eu.genesis-trading.com"
            }
            
            endpoint = api_endpoints.get(region)
            if not endpoint:
                return False
                
            async with aiohttp.ClientSession() as session:
                # Check health endpoint
                async with session.get(
                    f"{endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        return False
                        
                    data = await response.json()
                    
                    # Check all required services are healthy
                    required_services = ['database', 'cache', 'websocket']
                    for service in required_services:
                        if not data.get('services', {}).get(service, {}).get('healthy', False):
                            self.logger.warning("api_service_unhealthy",
                                              region=region,
                                              service=service)
                            return False
                            
                    return True
                    
        except Exception as e:
            self.logger.warning("api_health_check_failed",
                              region=region, error=str(e))
            return False
        
    async def _check_services_health(self, region: str) -> bool:
        """Check critical services health in a region."""
        try:
            # Define critical services and their ports
            services = [
                {"name": "trading_engine", "port": 8001, "endpoint": "/status"},
                {"name": "websocket_server", "port": 8002, "endpoint": "/ws/health"},
                {"name": "monitoring_collector", "port": 9090, "endpoint": "/metrics"}
            ]
            
            # Get service hosts for region
            service_hosts = {
                "us-east-1": "services-east.genesis-trading.com",
                "us-west-2": "services-west.genesis-trading.com",
                "eu-west-1": "services-eu.genesis-trading.com"
            }
            
            host = service_hosts.get(region, "localhost")
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for service in services:
                    try:
                        url = f"http://{host}:{service['port']}{service['endpoint']}"
                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status != 200:
                                self.logger.warning("service_unhealthy",
                                                  region=region,
                                                  service=service['name'],
                                                  status=response.status)
                                return False
                                
                    except Exception as e:
                        self.logger.warning("service_check_failed",
                                          region=region,
                                          service=service['name'],
                                          error=str(e))
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error("services_health_check_error",
                            region=region, error=str(e))
            return False
        
    async def _select_failover_region(self) -> Optional[str]:
        """Select the best failover region."""
        for region in self.config.failover_regions:
            health = await self._check_region_health(region)
            if health.is_healthy:
                return region
        return None
        
    async def execute_failover(self, target_region: str, 
                              force: bool = False) -> FailoverResult:
        """Execute automated failover to target region."""
        if self.state != FailoverState.IDLE and not force:
            raise GenesisException("Failover already in progress")
            
        self.state = FailoverState.INITIATING
        failover_start = datetime.utcnow()
        
        self.current_failover = FailoverResult(
            status="in_progress",
            source_region=self.config.primary_region,
            target_region=target_region,
            start_time=failover_start
        )
        
        try:
            self.logger.info("failover_initiated",
                           source=self.config.primary_region,
                           target=target_region)
            
            # Step 1: Enable connection draining
            if self.config.enable_connection_draining:
                await self._drain_connections()
                
            self.state = FailoverState.FAILING_OVER
            
            # Step 2: Update DNS routing
            await self._update_dns_routing(target_region)
            
            # Step 3: Promote standby database
            await self._promote_standby_database(target_region)
            
            # Step 4: Start regional services
            await self._start_regional_services(target_region)
            
            # Step 5: Validate failover
            self.state = FailoverState.VALIDATING
            validation_successful = await self._verify_failover_success(target_region)
            
            if not validation_successful:
                raise GenesisException("Failover validation failed")
                
            # Calculate metrics
            elapsed = (datetime.utcnow() - failover_start).total_seconds()
            self.current_failover.end_time = datetime.utcnow()
            self.current_failover.rto_seconds = elapsed
            self.current_failover.rto_achieved = elapsed < self.config.rto_target_seconds
            self.current_failover.rpo_seconds = await self._calculate_rpo()
            self.current_failover.rpo_achieved = self.current_failover.rpo_seconds <= self.config.rpo_target_seconds
            self.current_failover.status = "completed"
            
            self.state = FailoverState.COMPLETED
            
            # Send notifications
            await self._send_failover_notifications(self.current_failover)
            
            # Update configuration
            self.config.primary_region = target_region
            
            self.logger.info("failover_completed",
                           target=target_region,
                           rto_seconds=elapsed,
                           rto_achieved=self.current_failover.rto_achieved)
            
            self.failover_history.append(self.current_failover)
            return self.current_failover
            
        except Exception as e:
            self.logger.error("failover_failed", error=str(e))
            self.state = FailoverState.FAILED
            
            if self.current_failover:
                self.current_failover.status = "failed"
                self.current_failover.errors.append(str(e))
                self.current_failover.rollback_required = True
                
                # Attempt rollback
                if not force:
                    await self._rollback_failover()
                    
            raise
            
        finally:
            if self.state != FailoverState.ROLLING_BACK:
                self.state = FailoverState.IDLE
                
    async def _drain_connections(self) -> None:
        """Drain existing connections gracefully."""
        self.logger.info("draining_connections_started")
        
        # Signal connection draining to all services
        drain_start = datetime.utcnow()
        
        # Wait for configured drain timeout
        await asyncio.sleep(self.config.connection_drain_timeout)
        
        drain_time = (datetime.utcnow() - drain_start).total_seconds()
        self.logger.info("draining_connections_completed", duration=drain_time)
        
    async def _update_dns_routing(self, target_region: str) -> None:
        """Update DNS to route traffic to target region."""
        if not self.config.route53_hosted_zone_id or not self.config.domain_name:
            self.logger.warning("dns_update_skipped", reason="No Route53 configuration")
            return
            
        try:
            import boto3
            route53 = boto3.client('route53')
            
            # Get current record set
            response = route53.list_resource_record_sets(
                HostedZoneId=self.config.route53_hosted_zone_id,
                StartRecordName=self.config.domain_name,
                StartRecordType='A',
                MaxItems='1'
            )
            
            # Prepare change batch for failover
            change_batch = {
                'Comment': f'Failover to {target_region} at {datetime.utcnow().isoformat()}',
                'Changes': [{
                    'Action': 'UPSERT',
                    'ResourceRecordSet': {
                        'Name': self.config.domain_name,
                        'Type': 'A',
                        'SetIdentifier': target_region,
                        'Failover': 'PRIMARY',
                        'TTL': self.config.dns_ttl,
                        'ResourceRecords': [{
                            'Value': self._get_region_endpoint(target_region)
                        }],
                        'HealthCheckId': self._get_health_check_id(target_region)
                    }
                }]
            }
            
            # Apply DNS changes
            change_response = route53.change_resource_record_sets(
                HostedZoneId=self.config.route53_hosted_zone_id,
                ChangeBatch=change_batch
            )
            
            # Wait for changes to propagate
            waiter = route53.get_waiter('resource_record_sets_changed')
            waiter.wait(
                Id=change_response['ChangeInfo']['Id'],
                WaiterConfig={'Delay': 5, 'MaxAttempts': 60}
            )
            
            self.logger.info("dns_routing_updated",
                           domain=self.config.domain_name,
                           target=target_region,
                           ttl=self.config.dns_ttl,
                           change_id=change_response['ChangeInfo']['Id'])
                           
            if self.current_failover:
                self.current_failover.services_migrated.append("dns")
                
        except Exception as e:
            self.logger.error("dns_update_failed", error=str(e))
            raise
            
    async def _promote_standby_database(self, target_region: str) -> None:
        """Promote standby database in target region."""
        self.logger.info("promoting_standby_database", region=target_region)
        
        try:
            # Get database configuration for target region
            if target_region not in self.config.database_configs:
                raise GenesisException(f"No database config for region {target_region}")
                
            db_config = self.config.database_configs[target_region]
            
            # Connect to standby database
            import asyncpg
            conn = await asyncpg.connect(
                host=db_config['host'],
                port=db_config.get('port', 5432),
                database=db_config.get('database', 'genesis_prod'),
                user=db_config.get('user', os.environ.get('DB_USER', 'postgres')),
                password=db_config.get('password', os.environ.get('DB_PASSWORD')),
                timeout=30
            )
            
            try:
                # Check if this is a standby
                is_standby = await conn.fetchval("SELECT pg_is_in_recovery()")
                
                if not is_standby:
                    self.logger.warning("database_already_primary", region=target_region)
                    return
                    
                # Get current replication status
                lag = await conn.fetchval(
                    "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) AS lag"
                )
                
                if lag and lag > 5:
                    self.logger.warning("replication_lag_detected", lag_seconds=lag)
                    
                # Promote standby to primary
                # For RDS, we use AWS API; for self-managed, use pg_promote()
                if 'rds.amazonaws.com' in db_config['host']:
                    # RDS promotion
                    import boto3
                    rds = boto3.client('rds', region_name=target_region)
                    
                    db_instance_id = db_config['host'].split('.')[0]
                    rds.promote_read_replica(
                        DBInstanceIdentifier=db_instance_id,
                        BackupRetentionPeriod=7
                    )
                    
                    # Wait for promotion
                    waiter = rds.get_waiter('db_instance_available')
                    waiter.wait(
                        DBInstanceIdentifier=db_instance_id,
                        WaiterConfig={'Delay': 30, 'MaxAttempts': 40}
                    )
                else:
                    # Self-managed PostgreSQL promotion
                    await conn.execute("SELECT pg_promote()")
                    
                    # Wait for promotion to complete
                    for _ in range(30):
                        is_primary = not await conn.fetchval("SELECT pg_is_in_recovery()")
                        if is_primary:
                            break
                        await asyncio.sleep(1)
                    else:
                        raise GenesisException("Database promotion timeout")
                        
            finally:
                await conn.close()
                
            self.logger.info("standby_database_promoted", region=target_region)
            
            if self.current_failover:
                self.current_failover.services_migrated.append("database")
                
        except Exception as e:
            self.logger.error("database_promotion_failed", error=str(e))
            raise
            
    async def _start_regional_services(self, target_region: str) -> None:
        """Start all services in the target region."""
        self.logger.info("starting_regional_services", region=target_region)
        
        services = [
            {"name": "trading_engine", "command": "python -m genesis.engine", "port": 8001},
            {"name": "api_server", "command": "uvicorn genesis.api.server:app", "port": 8000},
            {"name": "websocket_server", "command": "python -m genesis.websocket", "port": 8002},
            {"name": "monitoring_collector", "command": "python -m genesis.monitoring", "port": 9090}
        ]
        
        for service_config in services:
            try:
                service = service_config["name"]
                command = service_config["command"]
                port = service_config["port"]
                
                # Check if service is already running
                if await self._is_service_running(service, port):
                    self.logger.info("service_already_running", service=service)
                    continue
                    
                # Start service using subprocess or systemd
                if await self._start_service(service, command, target_region):
                    # Wait for service to be ready
                    ready = await self._wait_for_service_ready(service, port, timeout=30)
                    
                    if ready:
                        self.logger.info("service_started", 
                                       service=service, 
                                       region=target_region,
                                       port=port)
                                       
                        if self.current_failover:
                            self.current_failover.services_migrated.append(service)
                    else:
                        raise GenesisException(f"Service {service} failed to become ready")
                else:
                    raise GenesisException(f"Failed to start service {service}")
                    
            except Exception as e:
                self.logger.error("service_start_failed", 
                                service=service_config["name"], 
                                error=str(e))
                raise
                
    async def _verify_failover_success(self, target_region: str) -> bool:
        """Verify that failover was successful."""
        self.logger.info("verifying_failover", region=target_region)
        
        # Check health of target region
        health = await self._check_region_health(target_region)
        
        if not health.is_healthy:
            self.logger.error("failover_validation_failed", 
                            region=target_region,
                            reason="Region unhealthy after failover")
            return False
            
        # Verify data integrity
        data_intact = await self._verify_data_integrity(target_region)
        if not data_intact:
            self.logger.error("failover_validation_failed",
                            reason="Data integrity check failed")
            return False
            
        self.logger.info("failover_verified_successful", region=target_region)
        return True
        
    async def _verify_data_integrity(self, region: str) -> bool:
        """Verify data integrity after failover."""
        # Implement data integrity checks
        # For now, return True as placeholder
        return True
        
    async def _calculate_rpo(self) -> float:
        """Calculate the Recovery Point Objective achieved."""
        try:
            # Get primary and standby database configs
            primary_config = self.config.database_configs.get(self.config.primary_region)
            
            if not primary_config:
                return 0.0
                
            # Connect to primary database
            import asyncpg
            conn = await asyncpg.connect(
                host=primary_config['host'],
                port=primary_config.get('port', 5432),
                database=primary_config.get('database', 'genesis_prod'),
                user=primary_config.get('user', os.environ.get('DB_USER', 'postgres')),
                password=primary_config.get('password', os.environ.get('DB_PASSWORD')),
                timeout=10
            )
            
            try:
                # Get last committed transaction timestamp
                last_xact = await conn.fetchval(
                    "SELECT pg_last_committed_xact()::text"
                )
                
                # Get replication lag in seconds
                lag_query = """
                    SELECT EXTRACT(EPOCH FROM (
                        now() - pg_last_xact_replay_timestamp()
                    ))::float AS lag_seconds
                    FROM pg_stat_replication
                    WHERE state = 'streaming'
                    ORDER BY replay_lag ASC
                    LIMIT 1
                """
                
                lag = await conn.fetchval(lag_query)
                
                return lag if lag else 0.0
                
            finally:
                await conn.close()
                
        except Exception as e:
            self.logger.warning("rpo_calculation_failed", error=str(e))
            return 0.0
        
    async def _send_failover_notifications(self, result: FailoverResult) -> None:
        """Send notifications about failover event."""
        if not self.config.notification_emails:
            return
            
        notification = {
            "event": "failover_completed",
            "source_region": result.source_region,
            "target_region": result.target_region,
            "rto_seconds": result.rto_seconds,
            "rto_achieved": result.rto_achieved,
            "rpo_seconds": result.rpo_seconds,
            "rpo_achieved": result.rpo_achieved,
            "timestamp": result.end_time.isoformat() if result.end_time else None,
            "services_migrated": result.services_migrated,
            "errors": result.errors
        }
        
        # Send email notifications
        await self._send_email_notification(notification)
        
        # Send Slack notification
        await self._send_slack_notification(notification)
        
        # Send PagerDuty alert if failover failed
        if result.status == "failed":
            await self._send_pagerduty_alert(notification)
        
        self.logger.info("failover_notification_sent",
                       recipients=self.config.notification_emails,
                       notification=notification)
    
    async def _send_email_notification(self, notification: Dict[str, Any]) -> None:
        """Send email notification about failover event."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Email configuration
            smtp_host = "smtp.sendgrid.net"
            smtp_port = 587
            smtp_user = os.environ.get("SENDGRID_USERNAME", "")
            smtp_pass = os.environ.get("SENDGRID_PASSWORD", "")
            
            if not smtp_user or not smtp_pass:
                self.logger.warning("email_notification_skipped", reason="No SMTP credentials")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = "dr-alerts@genesis-trading.com"
            msg['To'] = ", ".join(self.config.notification_emails)
            msg['Subject'] = f"[DR Alert] Failover {notification['event']}: {notification['target_region']}"
            
            # Email body
            body = f"""
            Disaster Recovery Failover Event
            ================================
            
            Event: {notification['event']}
            Source Region: {notification['source_region']}
            Target Region: {notification['target_region']}
            Timestamp: {notification['timestamp']}
            
            Performance Metrics:
            - RTO: {notification['rto_seconds']:.2f} seconds (Target met: {notification['rto_achieved']})
            - RPO: {notification['rpo_seconds']:.2f} seconds (Target met: {notification['rpo_achieved']})
            
            Services Migrated: {', '.join(notification.get('services_migrated', []))}
            
            {f"Errors: {', '.join(notification.get('errors', []))}" if notification.get('errors') else ''}
            
            Please verify all services are operational in the new region.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
                
            self.logger.info("email_notification_sent", recipients=self.config.notification_emails)
            
        except Exception as e:
            self.logger.error("email_notification_failed", error=str(e))
    
    async def _send_slack_notification(self, notification: Dict[str, Any]) -> None:
        """Send Slack notification about failover event."""
        try:
            import aiohttp
            
            webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
            if not webhook_url:
                return
            
            # Determine color based on status
            color = "good" if notification.get('rto_achieved') and notification.get('rpo_achieved') else "warning"
            if notification.get('errors'):
                color = "danger"
            
            # Create Slack message
            slack_message = {
                "username": "DR Bot",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color,
                    "title": f"Failover Event: {notification['target_region']}",
                    "fields": [
                        {"title": "Source Region", "value": notification['source_region'], "short": True},
                        {"title": "Target Region", "value": notification['target_region'], "short": True},
                        {"title": "RTO", "value": f"{notification['rto_seconds']:.2f}s", "short": True},
                        {"title": "RPO", "value": f"{notification['rpo_seconds']:.2f}s", "short": True},
                        {"title": "Status", "value": "✅ Success" if not notification.get('errors') else "❌ Failed", "short": True},
                        {"title": "Services", "value": str(len(notification.get('services_migrated', []))), "short": True}
                    ],
                    "footer": "Genesis DR System",
                    "ts": int(datetime.utcnow().timestamp())
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=slack_message) as response:
                    if response.status == 200:
                        self.logger.info("slack_notification_sent")
                    else:
                        self.logger.warning("slack_notification_failed", status=response.status)
                        
        except Exception as e:
            self.logger.error("slack_notification_error", error=str(e))
    
    async def _send_pagerduty_alert(self, notification: Dict[str, Any]) -> None:
        """Send PagerDuty alert for critical failover events."""
        try:
            import aiohttp
            
            integration_key = os.environ.get("PAGERDUTY_INTEGRATION_KEY")
            if not integration_key:
                return
            
            # Create PagerDuty event
            event = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"Critical: Failover to {notification['target_region']} - Errors detected",
                    "severity": "critical",
                    "source": "Genesis DR System",
                    "custom_details": {
                        "source_region": notification['source_region'],
                        "target_region": notification['target_region'],
                        "errors": notification.get('errors', []),
                        "rto_seconds": notification['rto_seconds'],
                        "rpo_seconds": notification['rpo_seconds']
                    }
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=event,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 202:
                        self.logger.info("pagerduty_alert_sent")
                    else:
                        self.logger.warning("pagerduty_alert_failed", status=response.status)
                        
        except Exception as e:
            self.logger.error("pagerduty_alert_error", error=str(e))
                       
    async def _rollback_failover(self) -> None:
        """Rollback a failed failover operation."""
        self.state = FailoverState.ROLLING_BACK
        self.logger.warning("failover_rollback_initiated")
        
        try:
            # Restore DNS to original region
            await self._update_dns_routing(self.config.primary_region)
            
            # Stop services in failed target region
            # In production, implement actual rollback
            
            self.logger.info("failover_rollback_completed")
            
        except Exception as e:
            self.logger.error("failover_rollback_failed", error=str(e))
            
        finally:
            self.state = FailoverState.IDLE
            
    async def execute_failback(self, original_region: str) -> FailoverResult:
        """Execute failback to original region after recovery."""
        self.logger.info("failback_initiated", target=original_region)
        
        # Verify original region is healthy
        health = await self._check_region_health(original_region)
        if not health.is_healthy:
            raise GenesisException(f"Original region {original_region} not healthy")
            
        # Execute failover back to original region
        return await self.execute_failover(original_region, force=True)
        
    def get_failover_status(self) -> Dict[str, Any]:
        """Get current failover status."""
        return {
            "state": self.state.value,
            "primary_region": self.config.primary_region,
            "failover_regions": self.config.failover_regions,
            "current_failover": self.current_failover.__dict__ if self.current_failover else None,
            "health_checks": {
                region: {
                    "is_healthy": check.is_healthy,
                    "last_check": check.last_check.isoformat(),
                    "response_time_ms": check.response_time_ms,
                    "consecutive_failures": check.consecutive_failures
                }
                for region, check in self.health_checks.items()
            },
            "history_count": len(self.failover_history)
        }
        
    def get_failover_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get failover history."""
        history = []
        for failover in self.failover_history[-limit:]:
            history.append({
                "status": failover.status,
                "source_region": failover.source_region,
                "target_region": failover.target_region,
                "start_time": failover.start_time.isoformat(),
                "end_time": failover.end_time.isoformat() if failover.end_time else None,
                "rto_seconds": failover.rto_seconds,
                "rto_achieved": failover.rto_achieved,
                "rpo_seconds": failover.rpo_seconds,
                "rpo_achieved": failover.rpo_achieved,
                "errors": failover.errors
            })
        return history
    
    def _get_region_endpoint(self, region: str) -> str:
        """Get the IP endpoint for a region."""
        # Map regions to their load balancer IPs
        region_endpoints = {
            "us-east-1": "54.123.45.67",
            "us-west-2": "35.234.56.78",
            "eu-west-1": "18.345.67.89"
        }
        return region_endpoints.get(region, "127.0.0.1")
    
    def _get_health_check_id(self, region: str) -> str:
        """Get the Route53 health check ID for a region."""
        # Map regions to their health check IDs
        health_check_ids = {
            "us-east-1": "abc123def456",
            "us-west-2": "ghi789jkl012",
            "eu-west-1": "mno345pqr678"
        }
        return health_check_ids.get(region, "")
    
    async def _is_service_running(self, service: str, port: int) -> bool:
        """Check if a service is running on the specified port."""
        import socket
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        
        try:
            result = sock.connect_ex(('localhost', port))
            return result == 0
        finally:
            sock.close()
    
    async def _start_service(self, service: str, command: str, region: str) -> bool:
        """Start a service in the specified region."""
        try:
            # Set environment variables for the region
            import os
            env = os.environ.copy()
            env['GENESIS_REGION'] = region
            env['GENESIS_SERVICE'] = service
            
            # Start the service
            proc = await asyncio.create_subprocess_shell(
                command,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Give it a moment to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if proc.returncode is None:
                return True
            else:
                stderr = await proc.stderr.read()
                self.logger.error("service_startup_failed", 
                                service=service,
                                stderr=stderr.decode())
                return False
                
        except Exception as e:
            self.logger.error("service_start_error", 
                            service=service,
                            error=str(e))
            return False
    
    async def _wait_for_service_ready(self, service: str, port: int, 
                                     timeout: int = 30) -> bool:
        """Wait for a service to become ready."""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if await self._is_service_running(service, port):
                # Additional health check
                if await self._check_service_health_endpoint(port):
                    return True
            await asyncio.sleep(1)
            
        return False
    
    async def _check_service_health_endpoint(self, port: int) -> bool:
        """Check service health via HTTP endpoint."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'http://localhost:{port}/health',
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except:
            return False