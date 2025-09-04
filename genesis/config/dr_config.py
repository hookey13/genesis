"""Disaster Recovery Configuration."""

import os
from typing import Any

# Main DR configuration
DR_CONFIG: dict[str, Any] = {
    "failover": {
        "enabled": True,
        "auto_failover": True,
        "rto_target_seconds": 300,  # 5 minutes
        "rpo_target_seconds": 0,  # Zero data loss
        "health_check_interval": 30,
        "failure_threshold": 3,
        "connection_drain_timeout": 30,
        "enable_connection_draining": True,
    },
    "regions": {
        "primary": "us-east-1",
        "secondary": "us-west-2",
        "tertiary": "eu-west-1",
    },
    "failover_regions": ["us-west-2", "eu-west-1"],
    "dns": {
        "hosted_zone_id": "Z123456789ABC",
        "domain": "genesis-trading.com",
        "ttl": 60,
        "health_check_interval": 30,
        "failure_threshold": 3,
    },
    "database": {
        "us-east-1": {
            "host": "db-primary.us-east-1.rds.amazonaws.com",
            "port": 5432,
            "database": "genesis_prod",
            "role": "primary",
            "max_connections": 100,
        },
        "us-west-2": {
            "host": "db-standby.us-west-2.rds.amazonaws.com",
            "port": 5432,
            "database": "genesis_prod",
            "role": "standby",
            "max_connections": 100,
        },
        "eu-west-1": {
            "host": "db-standby.eu-west-1.rds.amazonaws.com",
            "port": 5432,
            "database": "genesis_prod",
            "role": "standby",
            "max_connections": 100,
        },
    },
    "testing": {
        "enabled": True,
        "schedule": "0 0 1 * *",  # Monthly at midnight on the 1st
        "test_environment": "dr-test",
        "test_types": ["failover", "data_integrity", "failback", "partial_failure"],
        "chaos_testing_enabled": False,
        "auto_rollback": True,
        "max_test_duration_minutes": 60,
        "parallel_tests": False,
        "test_data_sample_size": 1000,
        "notification_emails": [
            "ops@genesis-trading.com",
            "devops@genesis-trading.com",
        ],
    },
    "monitoring": {
        "prometheus_endpoints": {
            "us-east-1": "http://prometheus.us-east-1.internal:9090",
            "us-west-2": "http://prometheus.us-west-2.internal:9090",
            "eu-west-1": "http://prometheus.eu-west-1.internal:9090",
        },
        "grafana_dashboards": {
            "failover_metrics": "https://grafana.genesis-trading.com/d/dr-failover",
            "recovery_validation": "https://grafana.genesis-trading.com/d/dr-validation",
        },
        "alert_channels": ["pagerduty", "slack", "email"],
    },
    "backup": {
        "s3_buckets": {
            "us-east-1": "genesis-backups-us-east-1",
            "us-west-2": "genesis-backups-us-west-2",
            "eu-west-1": "genesis-backups-eu-west-1",
        },
        "cross_region_replication": True,
        "retention_days": 30,
    },
    "services": {
        "critical": ["trading_engine", "api_server", "websocket_server", "database"],
        "non_critical": ["monitoring_collector", "backup_service", "log_aggregator"],
        "startup_order": [
            "database",
            "cache",
            "trading_engine",
            "api_server",
            "websocket_server",
            "monitoring_collector",
        ],
    },
    "network": {
        "vpc_peering": {
            "us-east-1_us-west-2": "pcx-12345678",
            "us-east-1_eu-west-1": "pcx-87654321",
            "us-west-2_eu-west-1": "pcx-11223344",
        },
        "security_groups": {
            "dr_failover": "sg-dr-failover",
            "database_replication": "sg-db-replication",
        },
    },
    "notifications": {
        "email": {
            "enabled": True,
            "smtp_host": "smtp.sendgrid.net",
            "smtp_port": 587,
            "from_address": "dr-alerts@genesis-trading.com",
            "recipients": ["ops@genesis-trading.com", "oncall@genesis-trading.com"],
        },
        "slack": {
            "enabled": True,
            "webhook_url": os.environ.get("SLACK_WEBHOOK_URL", ""),
            "channel": "#dr-alerts",
            "username": "DR Bot",
        },
        "pagerduty": {
            "enabled": True,
            "integration_key": os.environ.get("PAGERDUTY_INTEGRATION_KEY", ""),
            "service_id": os.environ.get("PAGERDUTY_SERVICE_ID", "P123456"),
        },
    },
    "validation": {
        "checks": [
            "data_integrity",
            "service_availability",
            "database_sync",
            "api_endpoints",
            "performance_baseline",
            "security_posture",
            "backup_availability",
            "monitoring_coverage",
        ],
        "critical_checks": [
            "data_integrity",
            "service_availability",
            "database_sync",
            "api_endpoints",
            "security_posture",
        ],
        "performance_thresholds": {
            "api_latency_p99_ms": 150,
            "database_query_time_ms": 20,
            "order_execution_time_ms": 100,
            "cpu_usage_percent": 80,
            "memory_usage_percent": 85,
        },
    },
    "rollback": {
        "enabled": True,
        "auto_rollback_on_failure": True,
        "rollback_timeout_seconds": 600,
        "manual_approval_required": False,
    },
    "documentation": {
        "runbook_location": "s3://genesis-docs/runbooks/dr/",
        "auto_generate_runbooks": True,
        "runbook_format": "markdown",
        "include_diagrams": True,
    },
}

# DR test scenarios configuration
DR_TEST_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "complete_regional_failure",
        "description": "Simulate complete failure of primary region",
        "critical": True,
        "timeout_seconds": 600,
        "expected_rto": 300,
        "expected_rpo": 0,
    },
    {
        "name": "database_failure",
        "description": "Simulate primary database failure",
        "critical": True,
        "timeout_seconds": 300,
        "expected_rto": 180,
        "expected_rpo": 0,
    },
    {
        "name": "network_partition",
        "description": "Simulate network split between regions",
        "critical": False,
        "timeout_seconds": 300,
        "expected_behavior": "split_brain_prevention",
    },
    {
        "name": "cascading_failure",
        "description": "Simulate cascading service failures",
        "critical": True,
        "timeout_seconds": 600,
        "expected_behavior": "graceful_degradation",
    },
    {
        "name": "data_corruption",
        "description": "Simulate data corruption scenario",
        "critical": True,
        "timeout_seconds": 900,
        "expected_behavior": "automatic_recovery_from_backup",
    },
]

# Failover decision matrix
FAILOVER_DECISION_MATRIX: dict[str, dict[str, Any]] = {
    "database_unavailable": {"action": "failover", "priority": 1, "automatic": True},
    "api_degraded": {"action": "investigate", "priority": 2, "automatic": False},
    "network_latency_high": {
        "action": "monitor",
        "priority": 3,
        "automatic": False,
        "threshold_ms": 500,
    },
    "multiple_service_failures": {
        "action": "failover",
        "priority": 1,
        "automatic": True,
        "threshold_count": 3,
    },
}

# Recovery validation criteria
RECOVERY_VALIDATION_CRITERIA: dict[str, Any] = {
    "mandatory": {
        "data_integrity": {
            "checksum_match": True,
            "row_count_match": True,
            "transaction_consistency": True,
        },
        "service_health": {
            "all_critical_services_running": True,
            "api_endpoints_responsive": True,
            "database_accessible": True,
        },
        "security": {
            "ssl_certificates_valid": True,
            "firewall_rules_applied": True,
            "access_controls_enforced": True,
        },
    },
    "recommended": {
        "performance": {
            "latency_within_baseline": True,
            "throughput_acceptable": True,
            "resource_utilization_normal": True,
        },
        "monitoring": {
            "metrics_collection_active": True,
            "alerts_configured": True,
            "dashboards_accessible": True,
        },
    },
}


def get_dr_config() -> dict[str, Any]:
    """Get the complete DR configuration."""
    return DR_CONFIG


def get_failover_config() -> dict[str, Any]:
    """Get failover-specific configuration."""
    return DR_CONFIG.get("failover", {})


def get_region_config(region: str) -> dict[str, Any]:
    """Get configuration for a specific region."""
    return {
        "database": DR_CONFIG["database"].get(region, {}),
        "monitoring": {
            "prometheus": DR_CONFIG["monitoring"]["prometheus_endpoints"].get(region)
        },
        "backup": {"s3_bucket": DR_CONFIG["backup"]["s3_buckets"].get(region)},
    }


def get_notification_config() -> dict[str, Any]:
    """Get notification configuration."""
    return DR_CONFIG.get("notifications", {})
