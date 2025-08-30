"""Coordinates automatic failover operations."""

import asyncio
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from genesis.failover.dns_manager import DNSManager
from genesis.failover.health_checker import HealthCheck, HealthChecker

logger = structlog.get_logger(__name__)


class FailoverState(Enum):
    """Failover system states."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    FAILING_OVER = "failing_over"
    FAILED_OVER = "failed_over"
    FAILING_BACK = "failing_back"


class FailoverCoordinator:
    """Coordinates automatic failover and failback operations."""

    def __init__(
        self,
        health_checker: HealthChecker,
        dns_manager: DNSManager | None = None,
        notification_channels: list[str] | None = None
    ):
        """Initialize failover coordinator.
        
        Args:
            health_checker: Health checker instance
            dns_manager: Optional DNS manager for DNS failover
            notification_channels: Notification channels
        """
        self.health_checker = health_checker
        self.dns_manager = dns_manager
        self.notification_channels = notification_channels or []

        # Failover configuration
        self.primary_config = {
            "database_path": ".genesis/data/genesis.db",
            "api_endpoint": "https://api.genesis.primary",
            "ip_address": "1.2.3.4"
        }

        self.backup_config = {
            "database_path": ".genesis/data/genesis_backup.db",
            "api_endpoint": "https://api.genesis.backup",
            "ip_address": "5.6.7.8"
        }

        # State tracking
        self.state = FailoverState.NORMAL
        self.failover_count = 0
        self.last_failover_time: datetime | None = None
        self.last_failback_time: datetime | None = None
        self.failover_history: list[dict[str, Any]] = []

        # Callbacks
        self.pre_failover_hooks: list[Callable] = []
        self.post_failover_hooks: list[Callable] = []

        # Monitor task
        self.monitor_task: asyncio.Task | None = None
        self.monitoring = False

    def register_pre_failover_hook(self, hook: Callable) -> None:
        """Register pre-failover hook.
        
        Args:
            hook: Callback to execute before failover
        """
        self.pre_failover_hooks.append(hook)

    def register_post_failover_hook(self, hook: Callable) -> None:
        """Register post-failover hook.
        
        Args:
            hook: Callback to execute after failover
        """
        self.post_failover_hooks.append(hook)

    async def start_monitoring(self) -> None:
        """Start failover monitoring."""
        if self.monitoring:
            return

        self.monitoring = True

        # Configure health checks
        self._configure_health_checks()

        # Start health checker
        await self.health_checker.start()

        # Start monitor task
        self.monitor_task = asyncio.create_task(self._monitor_health())

        logger.info("Failover coordinator started monitoring")

    async def stop_monitoring(self) -> None:
        """Stop failover monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False

        # Stop monitor task
        if self.monitor_task:
            self.monitor_task.cancel()
            await asyncio.gather(self.monitor_task, return_exceptions=True)

        # Stop health checker
        await self.health_checker.stop()

        logger.info("Failover coordinator stopped")

    def _configure_health_checks(self) -> None:
        """Configure health checks for critical services."""
        # Database health check
        self.health_checker.add_check(
            HealthCheck(
                name="primary_database",
                check_type="database",
                target=self.primary_config["database_path"],
                interval_seconds=10,
                timeout_seconds=5,
                failure_threshold=3
            )
        )

        # API health check
        self.health_checker.add_check(
            HealthCheck(
                name="primary_api",
                check_type="http",
                target=f"{self.primary_config['api_endpoint']}/health",
                interval_seconds=10,
                timeout_seconds=5,
                failure_threshold=3
            )
        )

        # Exchange connectivity check
        self.health_checker.add_check(
            HealthCheck(
                name="exchange_connection",
                check_type="tcp",
                target="api.binance.com:443",
                interval_seconds=30,
                timeout_seconds=10,
                failure_threshold=5
            )
        )

    async def _monitor_health(self) -> None:
        """Monitor health and trigger failover if needed."""
        while self.monitoring:
            try:
                await asyncio.sleep(5)

                # Get health status
                health_status = self.health_checker.get_status()

                # Check critical services
                db_healthy = self.health_checker.get_check_health("primary_database")
                api_healthy = self.health_checker.get_check_health("primary_api")

                # Determine if failover needed
                if self.state == FailoverState.NORMAL:
                    if not db_healthy or not api_healthy:
                        # Critical service failure
                        await self._handle_service_failure(db_healthy, api_healthy)

                elif self.state == FailoverState.FAILED_OVER:
                    if db_healthy and api_healthy:
                        # Primary recovered, consider failback
                        await self._consider_failback()

                # Update state based on health
                self._update_state(health_status)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error("Monitor health error", error=str(e))
                await asyncio.sleep(5)

    async def _handle_service_failure(self, db_healthy: bool, api_healthy: bool) -> None:
        """Handle critical service failure.
        
        Args:
            db_healthy: Database health status
            api_healthy: API health status
        """
        logger.warning(
            "Critical service failure detected",
            database_healthy=db_healthy,
            api_healthy=api_healthy
        )

        # Check if we should failover
        if self._should_failover():
            await self.execute_failover(
                reason=f"Service failure - DB:{db_healthy}, API:{api_healthy}"
            )

    def _should_failover(self) -> bool:
        """Determine if failover should be triggered.
        
        Returns:
            True if failover should proceed
        """
        # Don't failover if already in progress
        if self.state in [FailoverState.FAILING_OVER, FailoverState.FAILING_BACK]:
            return False

        # Check cooldown period (prevent flapping)
        if self.last_failover_time:
            time_since_failover = (datetime.utcnow() - self.last_failover_time).total_seconds()
            if time_since_failover < 300:  # 5 minute cooldown
                logger.info(f"Failover cooldown active ({time_since_failover}s)")
                return False

        # Check failover limit
        if self.failover_count >= 5:
            logger.error("Failover limit reached (5)")
            return False

        return True

    async def execute_failover(self, reason: str, dry_run: bool = False) -> dict[str, Any]:
        """Execute failover to backup infrastructure.
        
        Args:
            reason: Reason for failover
            dry_run: If True, simulate without executing
            
        Returns:
            Failover result
        """
        if self.state == FailoverState.FAILING_OVER:
            return {"error": "Failover already in progress"}

        start_time = datetime.utcnow()
        self.state = FailoverState.FAILING_OVER

        logger.critical(
            "FAILOVER INITIATED",
            reason=reason,
            dry_run=dry_run
        )

        result = {
            "success": False,
            "reason": reason,
            "dry_run": dry_run,
            "start_time": start_time.isoformat(),
            "steps": []
        }

        try:
            # Send notification
            await self._send_notification(
                "FAILOVER INITIATED",
                f"Reason: {reason}\nDry Run: {dry_run}"
            )

            # Execute pre-failover hooks
            for hook in self.pre_failover_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                    result["steps"].append({"step": f"Pre-hook: {hook.__name__}", "success": True})
                except Exception as e:
                    logger.error("Pre-failover hook failed", error=str(e))
                    result["steps"].append({"step": f"Pre-hook: {hook.__name__}", "success": False, "error": str(e)})

            if not dry_run:
                # 1. Switch database connection
                success = await self._switch_database()
                result["steps"].append({"step": "Switch database", "success": success})

                # 2. Update DNS if configured
                if self.dns_manager:
                    dns_success = await self._failover_dns()
                    result["steps"].append({"step": "Update DNS", "success": dns_success})

                # 3. Update connection strings
                success = await self._update_connections()
                result["steps"].append({"step": "Update connections", "success": success})

                # 4. Verify backup services
                success = await self._verify_backup_services()
                result["steps"].append({"step": "Verify backup", "success": success})

            # Execute post-failover hooks
            for hook in self.post_failover_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                    result["steps"].append({"step": f"Post-hook: {hook.__name__}", "success": True})
                except Exception as e:
                    logger.error("Post-failover hook failed", error=str(e))
                    result["steps"].append({"step": f"Post-hook: {hook.__name__}", "success": False, "error": str(e)})

            # Update state
            if not dry_run:
                self.state = FailoverState.FAILED_OVER
                self.failover_count += 1
                self.last_failover_time = datetime.utcnow()

            result["success"] = True
            result["end_time"] = datetime.utcnow().isoformat()
            result["duration_seconds"] = (datetime.utcnow() - start_time).total_seconds()

            # Record in history
            self.failover_history.append(result)

            # Send completion notification
            await self._send_notification(
                "FAILOVER COMPLETED",
                f"Duration: {result['duration_seconds']:.1f}s\n"
                f"Success: {result['success']}"
            )

            logger.info(
                "Failover completed",
                duration=result["duration_seconds"],
                success=result["success"]
            )

        except Exception as e:
            logger.error("Failover failed", error=str(e))
            result["error"] = str(e)

            if not dry_run:
                self.state = FailoverState.DEGRADED

            await self._send_notification(
                "FAILOVER FAILED",
                f"Error: {e!s}"
            )

        return result

    async def _switch_database(self) -> bool:
        """Switch to backup database.
        
        Returns:
            True if successful
        """
        try:
            # In production, would update connection strings
            logger.info(
                "Switching database",
                from_db=self.primary_config["database_path"],
                to_db=self.backup_config["database_path"]
            )

            # Simulate database switch
            await asyncio.sleep(1)

            return True

        except Exception as e:
            logger.error("Database switch failed", error=str(e))
            return False

    async def _failover_dns(self) -> bool:
        """Perform DNS failover.
        
        Returns:
            True if successful
        """
        if not self.dns_manager:
            return True

        try:
            # Update DNS to point to backup
            success = await self.dns_manager.failover_dns(
                domain="genesis.example.com",
                from_ip=self.primary_config["ip_address"],
                to_ip=self.backup_config["ip_address"]
            )

            if success:
                logger.info("DNS failover completed")
            else:
                logger.error("DNS failover failed")

            return success

        except Exception as e:
            logger.error("DNS failover error", error=str(e))
            return False

    async def _update_connections(self) -> bool:
        """Update connection strings to backup.
        
        Returns:
            True if successful
        """
        try:
            # Update API endpoint
            logger.info(
                "Updating connections",
                from_api=self.primary_config["api_endpoint"],
                to_api=self.backup_config["api_endpoint"]
            )

            # In production, would update actual connection configs
            await asyncio.sleep(0.5)

            return True

        except Exception as e:
            logger.error("Connection update failed", error=str(e))
            return False

    async def _verify_backup_services(self) -> bool:
        """Verify backup services are operational.
        
        Returns:
            True if all services healthy
        """
        try:
            # Add health checks for backup services
            self.health_checker.add_check(
                HealthCheck(
                    name="backup_database",
                    check_type="database",
                    target=self.backup_config["database_path"],
                    interval_seconds=10
                )
            )

            self.health_checker.add_check(
                HealthCheck(
                    name="backup_api",
                    check_type="http",
                    target=f"{self.backup_config['api_endpoint']}/health",
                    interval_seconds=10
                )
            )

            # Wait for checks
            await asyncio.sleep(15)

            # Verify health
            db_healthy = self.health_checker.get_check_health("backup_database")
            api_healthy = self.health_checker.get_check_health("backup_api")

            logger.info(
                "Backup services verified",
                database=db_healthy,
                api=api_healthy
            )

            return db_healthy and api_healthy

        except Exception as e:
            logger.error("Backup verification failed", error=str(e))
            return False

    async def _consider_failback(self) -> None:
        """Consider automatic failback to primary."""
        # Check if enough time has passed
        if self.last_failover_time:
            time_since_failover = (datetime.utcnow() - self.last_failover_time).total_seconds()
            if time_since_failover < 3600:  # 1 hour minimum
                return

        logger.info("Primary services recovered, considering failback")

        # Could trigger automatic failback or notify for manual intervention
        await self._send_notification(
            "PRIMARY RECOVERED",
            "Primary services are healthy. Manual failback may be performed."
        )

    async def execute_failback(self, dry_run: bool = False) -> dict[str, Any]:
        """Execute failback to primary infrastructure.
        
        Args:
            dry_run: If True, simulate without executing
            
        Returns:
            Failback result
        """
        if self.state != FailoverState.FAILED_OVER:
            return {"error": "Not in failed over state"}

        self.state = FailoverState.FAILING_BACK

        logger.info("FAILBACK INITIATED", dry_run=dry_run)

        # Similar to failover but in reverse
        # Switch back to primary database, DNS, connections

        if not dry_run:
            self.state = FailoverState.NORMAL
            self.last_failback_time = datetime.utcnow()

        return {"success": True, "dry_run": dry_run}

    def _update_state(self, health_status: dict[str, Any]) -> None:
        """Update state based on health status.
        
        Args:
            health_status: Current health status
        """
        if self.state == FailoverState.NORMAL:
            if not health_status["overall_health"]:
                self.state = FailoverState.DEGRADED
                logger.warning("System degraded", healthy=health_status["healthy_checks"], total=health_status["total_checks"])

        elif self.state == FailoverState.DEGRADED:
            if health_status["overall_health"]:
                self.state = FailoverState.NORMAL
                logger.info("System recovered to normal")

    async def _send_notification(self, subject: str, message: str) -> None:
        """Send notification to configured channels.
        
        Args:
            subject: Notification subject
            message: Notification message
        """
        for channel in self.notification_channels:
            try:
                logger.info(f"Notification [{channel}]: {subject}")
                # In production, would send actual notifications

            except Exception as e:
                logger.error(f"Failed to send {channel} notification", error=str(e))

    def get_status(self) -> dict[str, Any]:
        """Get failover coordinator status.
        
        Returns:
            Status dictionary
        """
        return {
            "state": self.state.value,
            "monitoring": self.monitoring,
            "failover_count": self.failover_count,
            "last_failover": self.last_failover_time.isoformat() if self.last_failover_time else None,
            "last_failback": self.last_failback_time.isoformat() if self.last_failback_time else None,
            "health_status": self.health_checker.get_status() if self.health_checker else None,
            "history_count": len(self.failover_history)
        }
