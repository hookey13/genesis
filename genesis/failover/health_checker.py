"""Health checking for infrastructure components."""

import asyncio
import sqlite3
from datetime import datetime
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger(__name__)


class HealthCheck:
    """Individual health check configuration."""

    def __init__(
        self,
        name: str,
        check_type: str,
        target: str,
        interval_seconds: int = 30,
        timeout_seconds: int = 10,
        failure_threshold: int = 3
    ):
        """Initialize health check.
        
        Args:
            name: Check name
            check_type: Type (http, tcp, database, process)
            target: Target to check (URL, host:port, etc.)
            interval_seconds: Check interval
            timeout_seconds: Check timeout
            failure_threshold: Failures before unhealthy
        """
        self.name = name
        self.check_type = check_type
        self.target = target
        self.interval_seconds = interval_seconds
        self.timeout_seconds = timeout_seconds
        self.failure_threshold = failure_threshold

        # State tracking
        self.consecutive_failures = 0
        self.last_check_time: datetime | None = None
        self.last_success_time: datetime | None = None
        self.is_healthy = True
        self.last_error: str | None = None


class HealthChecker:
    """Monitors health of infrastructure components."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: dict[str, HealthCheck] = {}
        self.check_tasks: dict[str, asyncio.Task] = {}
        self.running = False
        self.health_history: list[dict[str, Any]] = []

    def add_check(self, check: HealthCheck) -> None:
        """Add health check.
        
        Args:
            check: Health check to add
        """
        self.checks[check.name] = check

        if self.running:
            # Start check task if already running
            task = asyncio.create_task(self._run_check_loop(check))
            self.check_tasks[check.name] = task

        logger.info(f"Health check added: {check.name}")

    async def start(self) -> None:
        """Start health checking."""
        if self.running:
            return

        self.running = True

        # Start all check tasks
        for check in self.checks.values():
            task = asyncio.create_task(self._run_check_loop(check))
            self.check_tasks[check.name] = task

        logger.info(f"Health checker started with {len(self.checks)} checks")

    async def stop(self) -> None:
        """Stop health checking."""
        if not self.running:
            return

        self.running = False

        # Cancel all tasks
        for task in self.check_tasks.values():
            task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)

        self.check_tasks.clear()

        logger.info("Health checker stopped")

    async def _run_check_loop(self, check: HealthCheck) -> None:
        """Run health check loop.
        
        Args:
            check: Health check to run
        """
        while self.running:
            try:
                # Perform check
                is_healthy = await self._perform_check(check)

                # Update state
                check.last_check_time = datetime.utcnow()

                if is_healthy:
                    check.consecutive_failures = 0
                    check.last_success_time = datetime.utcnow()
                    check.last_error = None

                    if not check.is_healthy:
                        check.is_healthy = True
                        logger.info(f"Health check recovered: {check.name}")
                        await self._on_health_change(check, True)
                else:
                    check.consecutive_failures += 1

                    if check.consecutive_failures >= check.failure_threshold:
                        if check.is_healthy:
                            check.is_healthy = False
                            logger.warning(
                                f"Health check failed: {check.name}",
                                consecutive_failures=check.consecutive_failures,
                                error=check.last_error
                            )
                            await self._on_health_change(check, False)

                # Record history
                self._record_history(check, is_healthy)

                # Wait for next check
                await asyncio.sleep(check.interval_seconds)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Health check loop error: {check.name}", error=str(e))
                await asyncio.sleep(check.interval_seconds)

    async def _perform_check(self, check: HealthCheck) -> bool:
        """Perform individual health check.
        
        Args:
            check: Health check to perform
            
        Returns:
            True if healthy
        """
        try:
            if check.check_type == "http":
                return await self._check_http(check)
            elif check.check_type == "tcp":
                return await self._check_tcp(check)
            elif check.check_type == "database":
                return await self._check_database(check)
            elif check.check_type == "process":
                return await self._check_process(check)
            else:
                logger.warning(f"Unknown check type: {check.check_type}")
                return False

        except Exception as e:
            check.last_error = str(e)
            return False

    async def _check_http(self, check: HealthCheck) -> bool:
        """Perform HTTP health check.
        
        Args:
            check: Health check configuration
            
        Returns:
            True if healthy
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    check.target,
                    timeout=aiohttp.ClientTimeout(total=check.timeout_seconds)
                ) as response:
                    # 2xx or 3xx status codes are healthy
                    is_healthy = response.status < 400

                    if not is_healthy:
                        check.last_error = f"HTTP {response.status}"

                    return is_healthy

        except TimeoutError:
            check.last_error = "Timeout"
            return False

        except Exception as e:
            check.last_error = str(e)
            return False

    async def _check_tcp(self, check: HealthCheck) -> bool:
        """Perform TCP health check.
        
        Args:
            check: Health check configuration
            
        Returns:
            True if healthy
        """
        try:
            # Parse host and port
            parts = check.target.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 80

            # Try to connect
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=check.timeout_seconds
            )

            writer.close()
            await writer.wait_closed()

            return True

        except TimeoutError:
            check.last_error = "Connection timeout"
            return False

        except Exception as e:
            check.last_error = str(e)
            return False

    async def _check_database(self, check: HealthCheck) -> bool:
        """Perform database health check.
        
        Args:
            check: Health check configuration
            
        Returns:
            True if healthy
        """
        loop = asyncio.get_event_loop()

        def check_db():
            try:
                conn = sqlite3.connect(check.target, timeout=check.timeout_seconds)

                # Run simple query
                cursor = conn.execute("SELECT 1")
                result = cursor.fetchone()

                conn.close()

                return result is not None

            except Exception as e:
                check.last_error = str(e)
                return False

        return await loop.run_in_executor(None, check_db)

    async def _check_process(self, check: HealthCheck) -> bool:
        """Perform process health check.
        
        Args:
            check: Health check configuration
            
        Returns:
            True if healthy
        """
        try:
            import psutil

            # Check if process is running
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == check.target:
                    return True

            check.last_error = "Process not found"
            return False

        except Exception as e:
            check.last_error = str(e)
            return False

    async def _on_health_change(self, check: HealthCheck, is_healthy: bool) -> None:
        """Handle health status change.
        
        Args:
            check: Health check that changed
            is_healthy: New health status
        """
        # Log status change
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "check_name": check.name,
            "check_type": check.check_type,
            "target": check.target,
            "is_healthy": is_healthy,
            "consecutive_failures": check.consecutive_failures,
            "error": check.last_error
        }

        logger.info(
            "Health status changed",
            check=check.name,
            healthy=is_healthy,
            failures=check.consecutive_failures
        )

        # Could trigger failover here if critical service fails

    def _record_history(self, check: HealthCheck, is_healthy: bool) -> None:
        """Record health check history.
        
        Args:
            check: Health check
            is_healthy: Check result
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "check_name": check.name,
            "is_healthy": is_healthy,
            "response_time": None,  # Could measure this
            "error": check.last_error
        }

        self.health_history.append(record)

        # Limit history size
        if len(self.health_history) > 10000:
            self.health_history = self.health_history[-5000:]

    def get_status(self) -> dict[str, Any]:
        """Get current health status.
        
        Returns:
            Status dictionary
        """
        checks_status = {}

        for name, check in self.checks.items():
            checks_status[name] = {
                "is_healthy": check.is_healthy,
                "consecutive_failures": check.consecutive_failures,
                "last_check": check.last_check_time.isoformat() if check.last_check_time else None,
                "last_success": check.last_success_time.isoformat() if check.last_success_time else None,
                "last_error": check.last_error
            }

        # Calculate overall health
        total_checks = len(self.checks)
        healthy_checks = sum(1 for c in self.checks.values() if c.is_healthy)

        return {
            "running": self.running,
            "overall_health": healthy_checks == total_checks,
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "checks": checks_status,
            "history_size": len(self.health_history)
        }

    def get_check_health(self, check_name: str) -> bool:
        """Get health status of specific check.
        
        Args:
            check_name: Name of check
            
        Returns:
            True if healthy
        """
        check = self.checks.get(check_name)
        return check.is_healthy if check else False
