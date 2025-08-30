#!/usr/bin/env python3
"""
Main operational automation script for Project GENESIS.

Coordinates all maintenance operations including log archival, database optimization,
certificate renewal, and health monitoring.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genesis.operations.log_archiver import LogArchiver, LogArchivalConfig, setup_log_archival
from genesis.operations.db_optimizer import DBOptimizer, MaintenanceConfig, setup_db_maintenance
from genesis.operations.cert_manager import CertManager, CertificateConfig
from genesis.operations.performance_baseline import PerformanceBaseline
from genesis.operations.correlation_updater import CorrelationUpdater, CorrelationConfig
from genesis.operations.strategy_optimizer import StrategyOptimizer, OptimizationConfig
from genesis.operations.dependency_updater import DependencyUpdater, DependencyConfig
from genesis.operations.health_monitor import HealthMonitor, HealthConfig
from genesis.utils.logger import setup_logging, get_logger, LoggerType


async def run_log_maintenance(args):
    """Run log archival and maintenance."""
    logger = get_logger(__name__, LoggerType.SYSTEM)
    logger.info("Starting log maintenance")
    
    archiver = await setup_log_archival()
    
    if args.archive_now:
        archived = await archiver.archive_rotated_logs()
        logger.info(f"Archived {len(archived)} log files")
    
    if args.enforce_retention:
        local_deleted, remote_deleted = await archiver.enforce_retention_policy()
        logger.info(f"Deleted {local_deleted} local and {remote_deleted} remote files")
    
    if args.stats:
        stats = await archiver.get_storage_stats()
        logger.info("Log storage statistics", stats=stats)


async def run_db_maintenance(args):
    """Run database maintenance operations."""
    logger = get_logger(__name__, LoggerType.SYSTEM)
    logger.info("Starting database maintenance")
    
    optimizer = await setup_db_maintenance()
    
    if args.vacuum:
        success = await optimizer.perform_vacuum(force=args.force)
        if success:
            logger.info("Database vacuum completed")
    
    if args.analyze:
        success = await optimizer.perform_analyze()
        if success:
            logger.info("Database analyze completed")
    
    if args.check_indexes:
        index_stats = await optimizer.analyze_indexes()
        logger.info(f"Analyzed {len(index_stats)} indexes")
    
    if args.show_baseline:
        baseline = await optimizer.get_performance_baseline()
        logger.info("Performance baseline", baseline=baseline)


async def run_cert_renewal(args):
    """Run certificate renewal."""
    logger = get_logger(__name__, LoggerType.SYSTEM)
    logger.info("Starting certificate management")
    
    config = CertificateConfig(
        domain=args.domain or "genesis.trading",
        email=args.email or "admin@genesis.trading",
        use_staging=args.staging
    )
    
    manager = CertManager(config)
    
    if args.check_expiry:
        expiry = await manager.check_certificate_expiry()
        if expiry:
            logger.info(f"Certificate expires: {expiry}")
    
    if args.renew:
        success = await manager.renew_certificate()
        if success:
            logger.info("Certificate renewed successfully")


async def run_health_check(args):
    """Run health checks."""
    logger = get_logger(__name__, LoggerType.SYSTEM)
    logger.info("Starting health check")
    
    config = HealthConfig(
        enable_auto_remediation=not args.no_remediation
    )
    
    monitor = HealthMonitor(config)
    
    if args.full_check:
        checks = await monitor.perform_health_checks()
        for name, check in checks.items():
            logger.info(
                f"Health check: {name}",
                status=check.status,
                message=check.message
            )
    
    if args.summary:
        summary = await monitor.get_health_summary()
        logger.info("Health summary", summary=summary)
    
    if args.continuous:
        logger.info("Starting continuous health monitoring")
        await monitor.health_monitoring_loop()


async def run_all_maintenance(args):
    """Run all maintenance tasks."""
    logger = get_logger(__name__, LoggerType.SYSTEM)
    logger.info("Starting comprehensive maintenance")
    
    tasks = []
    
    # Setup all maintenance systems
    if not args.skip_logs:
        tasks.append(setup_log_archival())
    
    if not args.skip_db:
        tasks.append(setup_db_maintenance())
    
    if not args.skip_health:
        config = HealthConfig()
        monitor = HealthMonitor(config)
        tasks.append(monitor.health_monitoring_loop())
    
    # Run all tasks concurrently
    await asyncio.gather(*tasks)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Genesis Maintenance Operations")
    
    subparsers = parser.add_subparsers(dest="command", help="Maintenance commands")
    
    # Log maintenance
    log_parser = subparsers.add_parser("logs", help="Log maintenance")
    log_parser.add_argument("--archive-now", action="store_true", help="Archive rotated logs now")
    log_parser.add_argument("--enforce-retention", action="store_true", help="Enforce retention policies")
    log_parser.add_argument("--stats", action="store_true", help="Show storage statistics")
    
    # Database maintenance
    db_parser = subparsers.add_parser("db", help="Database maintenance")
    db_parser.add_argument("--vacuum", action="store_true", help="Run VACUUM operation")
    db_parser.add_argument("--analyze", action="store_true", help="Run ANALYZE operation")
    db_parser.add_argument("--check-indexes", action="store_true", help="Analyze index usage")
    db_parser.add_argument("--show-baseline", action="store_true", help="Show performance baseline")
    db_parser.add_argument("--force", action="store_true", help="Force operation")
    
    # Certificate management
    cert_parser = subparsers.add_parser("cert", help="Certificate management")
    cert_parser.add_argument("--domain", help="Domain name")
    cert_parser.add_argument("--email", help="Email for notifications")
    cert_parser.add_argument("--check-expiry", action="store_true", help="Check certificate expiry")
    cert_parser.add_argument("--renew", action="store_true", help="Renew certificate")
    cert_parser.add_argument("--staging", action="store_true", help="Use Let's Encrypt staging")
    
    # Health monitoring
    health_parser = subparsers.add_parser("health", help="Health monitoring")
    health_parser.add_argument("--full-check", action="store_true", help="Run all health checks")
    health_parser.add_argument("--summary", action="store_true", help="Show health summary")
    health_parser.add_argument("--continuous", action="store_true", help="Run continuous monitoring")
    health_parser.add_argument("--no-remediation", action="store_true", help="Disable auto-remediation")
    
    # Run all
    all_parser = subparsers.add_parser("all", help="Run all maintenance")
    all_parser.add_argument("--skip-logs", action="store_true", help="Skip log maintenance")
    all_parser.add_argument("--skip-db", action="store_true", help="Skip database maintenance")
    all_parser.add_argument("--skip-health", action="store_true", help="Skip health monitoring")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Run appropriate command
    if args.command == "logs":
        asyncio.run(run_log_maintenance(args))
    elif args.command == "db":
        asyncio.run(run_db_maintenance(args))
    elif args.command == "cert":
        asyncio.run(run_cert_renewal(args))
    elif args.command == "health":
        asyncio.run(run_health_check(args))
    elif args.command == "all":
        asyncio.run(run_all_maintenance(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()