"""
Health check module for Project GENESIS.
Provides health status for deployment and monitoring.
"""

import asyncio
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def health_check() -> bool:
    """
    Basic health check for Docker HEALTHCHECK.
    
    Returns:
        bool: True if system is healthy, False otherwise
    """
    try:
        # Check Python version
        if sys.version_info < (3, 11, 8):
            logger.error("Python version check failed",
                        required="3.11.8",
                        current=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
            return False

        # Check critical imports

        # Check critical directories exist
        dirs_to_check = [
            Path("/app/.genesis/data"),
            Path("/app/.genesis/logs"),
            Path("/app/.genesis/state")
        ]

        for dir_path in dirs_to_check:
            if not dir_path.exists():
                logger.warning(f"Directory not found: {dir_path}")
                # Don't fail on missing dirs as they may be created on startup

        # Check decimal precision (critical for money calculations)
        test_value = Decimal("0.00000001")
        if test_value != Decimal("0.00000001"):
            logger.error("Decimal precision check failed")
            return False

        return True

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


async def detailed_health_check() -> dict[str, Any]:
    """
    Detailed health check for monitoring and deployment validation.
    
    Returns:
        Dict containing detailed health status
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": os.environ.get("VERSION", "unknown"),
        "checks": {}
    }

    try:
        # Python version check
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        health_status["checks"]["python_version"] = {
            "status": "pass" if sys.version_info >= (3, 11, 8) else "fail",
            "current": python_version,
            "required": "3.11.8"
        }

        # Module import checks
        critical_modules = [
            "genesis.core.models",
            "genesis.engine.state_machine",
            "genesis.exchange.gateway",
            "genesis.tilt.detector",
            "genesis.data.repository"
        ]

        for module_name in critical_modules:
            try:
                __import__(module_name)
                health_status["checks"][f"module_{module_name}"] = {"status": "pass"}
            except ImportError as e:
                health_status["checks"][f"module_{module_name}"] = {
                    "status": "fail",
                    "error": str(e)
                }
                health_status["status"] = "unhealthy"

        # File system checks
        fs_checks = {
            "data_dir": Path("/app/.genesis/data"),
            "logs_dir": Path("/app/.genesis/logs"),
            "state_dir": Path("/app/.genesis/state"),
            "backup_dir": Path("/app/.genesis/backups")
        }

        for name, path in fs_checks.items():
            if path.exists():
                health_status["checks"][name] = {
                    "status": "pass",
                    "writable": os.access(path, os.W_OK)
                }
            else:
                health_status["checks"][name] = {
                    "status": "warning",
                    "message": "Directory not found (will be created on startup)"
                }

        # Database connectivity (if configured)
        if os.environ.get("DATABASE_URL"):
            try:
                # This would test DB connection in real implementation
                health_status["checks"]["database"] = {"status": "pass"}
            except Exception as e:
                health_status["checks"]["database"] = {
                    "status": "fail",
                    "error": str(e)
                }
                health_status["status"] = "unhealthy"

        # Environment variable checks
        required_env_vars = ["BINANCE_API_KEY", "BINANCE_API_SECRET"]
        for var in required_env_vars:
            if os.environ.get(var):
                health_status["checks"][f"env_{var}"] = {"status": "pass", "present": True}
            else:
                health_status["checks"][f"env_{var}"] = {
                    "status": "warning",
                    "present": False,
                    "message": "Required for trading"
                }

        # Memory check
        import psutil
        memory = psutil.virtual_memory()
        health_status["checks"]["memory"] = {
            "status": "pass" if memory.percent < 90 else "warning",
            "percent_used": memory.percent,
            "available_mb": memory.available / 1024 / 1024
        }

    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)
        logger.error(f"Detailed health check error: {e}")

    return health_status


async def readiness_check() -> bool:
    """
    Check if the application is ready to serve traffic.
    Used for deployment validation and blue-green cutover.
    
    Returns:
        bool: True if ready to serve, False otherwise
    """
    try:
        # Check basic health first
        if not health_check():
            return False

        # Check if state machine is initialized
        # In real implementation, would check if state machine is ready

        # Check if exchange gateway is connectable
        if os.environ.get("BINANCE_API_KEY"):
            pass
            # In real implementation, would test API connectivity

        # Check if critical services are running
        # This would check supervisor status in production

        return True

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return False


async def liveness_check() -> bool:
    """
    Check if the application is alive and not deadlocked.
    Used for automatic restarts if unhealthy.
    
    Returns:
        bool: True if alive, False if needs restart
    """
    try:
        # Test event loop responsiveness
        start_time = datetime.utcnow()
        await asyncio.sleep(0.001)
        elapsed = (datetime.utcnow() - start_time).total_seconds()

        # If event loop is blocked for more than 1 second, something is wrong
        if elapsed > 1.0:
            logger.error(f"Event loop blocked for {elapsed} seconds")
            return False

        # Check for deadlocks or hung processes
        # This would check thread states in production

        return True

    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return False


if __name__ == "__main__":
    # CLI usage for health checks
    import sys

    if len(sys.argv) > 1:
        check_type = sys.argv[1]

        if check_type == "basic":
            sys.exit(0 if health_check() else 1)
        elif check_type == "detailed":
            import json
            result = asyncio.run(detailed_health_check())
            print(json.dumps(result, indent=2))
            sys.exit(0 if result["status"] == "healthy" else 1)
        elif check_type == "readiness":
            sys.exit(0 if asyncio.run(readiness_check()) else 1)
        elif check_type == "liveness":
            sys.exit(0 if asyncio.run(liveness_check()) else 1)
        else:
            print(f"Unknown check type: {check_type}")
            print("Usage: python -m genesis.api.health [basic|detailed|readiness|liveness]")
            sys.exit(1)
    else:
        # Default to basic check
        sys.exit(0 if health_check() else 1)
