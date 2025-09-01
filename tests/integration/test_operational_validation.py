"""Integration tests for operational validation."""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from genesis.validation.operational import (
    BackupValidator,
    DeploymentValidator,
    DocumentationValidator,
    HealthCheckValidator,
    MonitoringValidator,
)


@pytest.mark.integration
class TestOperationalValidationIntegration:
    """Integration tests for operational validators."""

    @pytest.fixture
    async def complete_environment(self, tmp_path):
        """Set up complete test environment."""
        # Create directory structure
        dirs = [
            "config/prometheus",
            "config/grafana",
            "docker",
            "kubernetes",
            "scripts",
            "docs/api",
            ".genesis/data",
            ".genesis/state",
            ".genesis/backups",
            "genesis/api",
            "genesis/monitoring",
        ]
        for dir_path in dirs:
            (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create all required files
        await self._create_monitoring_files(tmp_path)
        await self._create_backup_files(tmp_path)
        await self._create_documentation_files(tmp_path)
        await self._create_deployment_files(tmp_path)
        await self._create_health_files(tmp_path)
        
        return tmp_path

    async def _create_monitoring_files(self, root_path):
        """Create monitoring configuration files."""
        # Prometheus config
        prometheus_config = {
            "global": {"scrape_interval": "15s"},
            "scrape_configs": [
                {
                    "job_name": "genesis",
                    "static_configs": [{"targets": ["localhost:8000"]}],
                }
            ],
        }
        with open(root_path / "config/prometheus/prometheus.yml", "w") as f:
            yaml.dump(prometheus_config, f)
        
        # Alert rules
        alerts = {
            "groups": [
                {
                    "name": "genesis",
                    "rules": [
                        {
                            "alert": "high_error_rate",
                            "expr": "rate(genesis_errors[5m]) > 0.1",
                        },
                        {
                            "alert": "low_system_health",
                            "expr": "genesis_system_health_score < 70",
                        },
                        {
                            "alert": "high_latency",
                            "expr": "genesis_order_execution_latency_seconds > 1",
                        },
                        {
                            "alert": "position_limit_exceeded",
                            "expr": "genesis_active_positions > 10",
                        },
                        {
                            "alert": "api_rate_limit_warning",
                            "expr": "genesis_api_calls_total > 900",
                        },
                    ],
                }
            ]
        }
        with open(root_path / "config/prometheus/alerts.yml", "w") as f:
            yaml.dump(alerts, f)
        
        # Grafana config
        with open(root_path / "config/grafana/grafana.ini", "w") as f:
            f.write("[server]\nhttp_port = 3000\n[security]\nadmin_password = admin\n")
        
        # Grafana dashboards
        dashboards = [
            "trading-overview",
            "system-health",
            "performance-metrics",
            "error-tracking",
            "api-usage",
        ]
        for dashboard in dashboards:
            dashboard_json = {
                "uid": dashboard,
                "title": dashboard.replace("-", " ").title(),
                "panels": [{"id": 1, "title": "Sample Panel"}],
            }
            with open(root_path / f"config/grafana/{dashboard}.json", "w") as f:
                json.dump(dashboard_json, f)

    async def _create_backup_files(self, root_path):
        """Create backup configuration files."""
        # Create backup script
        backup_script = """#!/bin/bash
set -e
trap 'echo "Backup failed"' ERR

echo "Starting backup..."
logger "Genesis backup started"

# Backup database
restic backup /genesis/.genesis/data/genesis.db

# Backup config
restic backup /genesis/config/

# Backup state
restic backup /genesis/.genesis/state/

# Sync to DigitalOcean Spaces
s3cmd sync /backup/ s3://genesis-backups/

echo "Backup completed"
"""
        script_path = root_path / "scripts/backup.sh"
        script_path.write_text(backup_script)
        script_path.chmod(0o755)
        
        # Create restore script
        restore_script = """#!/bin/bash
set -e
echo "Starting restore..."
restic restore latest --target /restore/
echo "Restore completed"
"""
        script_path = root_path / "scripts/restore.sh"
        script_path.write_text(restore_script)
        script_path.chmod(0o755)
        
        # Create database and state files
        (root_path / ".genesis/data/genesis.db").touch()
        (root_path / ".genesis/state/tier_state.json").write_text('{"tier": "sniper"}')
        (root_path / "config/settings.yaml").write_text("environment: production")
        
        # Create recent backup file
        backup_file = root_path / ".genesis/backups/backup-latest.tar.gz"
        backup_file.touch()
        
        # Create .env with offsite config
        (root_path / ".env").write_text(
            "DO_SPACES_KEY=xxx\n"
            "DO_SPACES_SECRET=yyy\n"
            "DO_SPACES_BUCKET=genesis-backups\n"
        )

    async def _create_documentation_files(self, root_path):
        """Create documentation files."""
        # README
        readme_content = """# Genesis Trading Bot

## Installation
```bash
pip install -r requirements.txt
python setup.py install
```

## Configuration
Copy `.env.example` to `.env` and configure:
- API keys
- Database settings
- Risk parameters

## Usage
```python
python -m genesis
```

## Architecture
See `docs/architecture.md` for system design.

## Testing
```bash
pytest tests/
```

## Deployment
See `docs/deployment.md` for deployment instructions.
""" + "\n" * 100
        (root_path / "README.md").write_text(readme_content)
        
        # Runbook
        runbook_content = """# Genesis Runbook

## System Startup
1. Check prerequisites
2. Start database
3. Start Redis
4. Launch Genesis
5. Verify health checks

## Shutdown Procedures
1. Close all positions
2. Cancel pending orders
3. Save state
4. Stop services
5. Backup data

## Emergency Position Closure
1. Execute `scripts/emergency_close.py`
2. Verify all positions closed
3. Check logs for errors
4. Notify team

## API Rate Limit Handling
1. Check current rate usage
2. Implement backoff
3. Queue requests
4. Monitor recovery

## Database Recovery
1. Stop services
2. Restore from backup
3. Verify data integrity
4. Restart services
5. Run health checks

## Tilt Intervention
1. Review tilt indicators
2. Reduce position sizes
3. Implement cooling period
4. Monitor behavior
5. Gradual re-engagement

## Performance Degradation
1. Check system resources
2. Review logs
3. Identify bottlenecks
4. Scale resources
5. Optimize queries

## Security Incident Response
1. Isolate affected systems
2. Revoke compromised credentials
3. Audit access logs
4. Patch vulnerabilities
5. Report incident

## Rollback Procedures
1. Stop current deployment
2. Checkout previous version
3. Restore database backup
4. Deploy previous version
5. Verify functionality

## Data Corruption Recovery
1. Stop writes
2. Identify corruption extent
3. Restore from clean backup
4. Replay transactions
5. Verify data integrity
""" + "\n" * 150
        (root_path / "docs/runbook.md").write_text(runbook_content)
        
        # Other docs
        (root_path / "docs/architecture.md").write_text(
            "# Architecture\n## System Overview\n## Components\n## Data Flow\n## Security\n## Scalability\n" + "\n" * 150
        )
        (root_path / "docs/deployment.md").write_text(
            "# Deployment\n## Prerequisites\n## Environment Setup\n## Deployment Steps\n## Verification\n## Rollback\n" + "\n" * 50
        )
        (root_path / "docs/troubleshooting.md").write_text(
            "# Troubleshooting\n## Common Issues\n## Error Messages\n## Performance Issues\n## Connectivity Problems\n## Recovery Procedures\n" + "\n" * 100
        )
        (root_path / "docs/monitoring.md").write_text(
            "# Monitoring\n## Metrics\n## Dashboards\n## Alerts\n## Log Analysis\n## Health Checks\n" + "\n" * 75
        )
        
        # API documentation
        api_doc = """# API Documentation

## Authentication:
Use Bearer token in Authorization header.

## Endpoints:

### GET /health
Returns system health status.

Parameters: None

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

Example:
```bash
curl http://localhost:8000/health
```

### POST /orders
Create a new order.

Parameters:
- symbol: Trading pair
- quantity: Order quantity
- side: BUY or SELL

Response:
```json
{
  "order_id": "12345",
  "status": "pending"
}
```
"""
        (root_path / "docs/api/endpoints.md").write_text(api_doc)

    async def _create_deployment_files(self, root_path):
        """Create deployment configuration files."""
        # Dockerfile
        dockerfile = """FROM python:3.11-alpine AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-alpine
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
RUN adduser -D genesis
USER genesis
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"
LABEL maintainer="genesis-team"
CMD ["python", "-m", "genesis"]
"""
        (root_path / "docker/Dockerfile").write_text(dockerfile)
        
        # Docker Compose
        compose = {
            "version": "3.8",
            "services": {
                "genesis": {
                    "build": ".",
                    "ports": ["8000:8000"],
                    "environment": ["ENVIRONMENT=production"],
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "3s",
                        "retries": 3,
                    },
                },
                "nginx": {
                    "image": "nginx:alpine",
                    "ports": ["80:80"],
                    "depends_on": ["genesis"],
                },
            },
        }
        with open(root_path / "docker/docker-compose.yml", "w") as f:
            yaml.dump(compose, f)
        
        # Production compose with blue-green
        compose_prod = {
            "version": "3.8",
            "services": {
                "genesis-blue": {
                    "image": "genesis:blue",
                    "ports": ["8001:8000"],
                },
                "genesis-green": {
                    "image": "genesis:green",
                    "ports": ["8002:8000"],
                },
                "nginx": {
                    "image": "nginx:alpine",
                    "ports": ["80:80"],
                },
            },
        }
        with open(root_path / "docker/docker-compose.prod.yml", "w") as f:
            yaml.dump(compose_prod, f)
        
        # Deployment scripts
        deploy_script = """#!/bin/bash
set -e
trap 'echo "Deployment failed"' ERR

echo "Starting blue-green deployment..."

# Deploy to blue
docker-compose -f docker/docker-compose.prod.yml up -d genesis-blue
sleep 5

# Health check
curl -f http://localhost:8001/health || exit 1

# Switch traffic
echo "Switching traffic to blue..."
docker-compose -f docker/docker-compose.prod.yml stop genesis-green

echo "Deployment completed"
"""
        script_path = root_path / "scripts/deploy.sh"
        script_path.write_text(deploy_script)
        script_path.chmod(0o755)
        
        rollback_script = """#!/bin/bash
set -e
echo "Rolling back..."
git checkout previous_tag
docker-compose down
restore_backup
docker-compose up -d
health_check
"""
        script_path = root_path / "scripts/rollback.sh"
        script_path.write_text(rollback_script)
        script_path.chmod(0o755)
        
        health_script = """#!/bin/bash
curl -f http://localhost:8000/health || exit 1
"""
        script_path = root_path / "scripts/health_check.sh"
        script_path.write_text(health_script)
        script_path.chmod(0o755)
        
        # Kubernetes manifests
        k8s_deployment = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: genesis
spec:
  replicas: 2
  selector:
    matchLabels:
      app: genesis
  template:
    metadata:
      labels:
        app: genesis
    spec:
      containers:
      - name: genesis
        image: genesis:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
"""
        (root_path / "kubernetes/deployment.yaml").write_text(k8s_deployment)
        
        k8s_service = """apiVersion: v1
kind: Service
metadata:
  name: genesis
spec:
  selector:
    app: genesis
  ports:
  - port: 80
    targetPort: 8000
"""
        (root_path / "kubernetes/service.yaml").write_text(k8s_service)
        
        k8s_configmap = """apiVersion: v1
kind: ConfigMap
metadata:
  name: genesis-config
data:
  environment: production
"""
        (root_path / "kubernetes/configmap.yaml").write_text(k8s_configmap)
        
        # Staging environment
        (root_path / ".env.staging").write_text("ENVIRONMENT=staging\nDEBUG=true\n")
        
        # Rollback documentation
        (root_path / "docs/rollback.md").write_text(
            "# Rollback Procedures\n\n"
            "1. Stop current deployment\n"
            "2. Checkout previous version\n"
            "3. Restore database backup\n"
            "4. Deploy previous version\n"
            "5. Verify health checks\n"
        )

    async def _create_health_files(self, root_path):
        """Create health check implementation files."""
        health_implementation = """from fastapi import APIRouter, HTTPException
from datetime import datetime
import asyncio

router = APIRouter()

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "uptime": 3600
    }

@router.get("/health/live")
async def liveness():
    return {"alive": True}

@router.get("/health/ready")
async def readiness():
    checks = await run_readiness_checks()
    if all(checks.values()):
        return {"ready": True, "checks": checks}
    raise HTTPException(status_code=503, detail="Not ready")

@router.get("/health/dependencies")
async def dependencies():
    return {
        "database": await check_database(),
        "redis": await check_redis(),
        "exchange_api": await check_exchange_api(),
        "websocket": await check_websocket()
    }

@router.get("/metrics")
async def metrics():
    return "# Prometheus metrics\\ngenesis_health 1\\n"

async def run_readiness_checks():
    return {
        "database": await check_database(),
        "redis": await check_redis(),
        "config": True
    }

async def check_database():
    # Simulate database check
    await asyncio.sleep(0.01)
    return True

async def check_redis():
    # Simulate Redis check
    await asyncio.sleep(0.01)
    return True

async def check_exchange_api():
    # Simulate exchange API check
    await asyncio.sleep(0.01)
    return True

async def check_websocket():
    # Simulate WebSocket check
    await asyncio.sleep(0.01)
    return True
"""
        (root_path / "genesis/api/health.py").write_text(health_implementation)

    @pytest.mark.asyncio
    async def test_all_validators_integration(self, complete_environment):
        """Test all validators working together."""
        # Initialize all validators
        monitoring = MonitoringValidator(genesis_root=complete_environment)
        backup = BackupValidator(genesis_root=complete_environment)
        docs = DocumentationValidator(genesis_root=complete_environment)
        deployment = DeploymentValidator(genesis_root=complete_environment)
        health = HealthCheckValidator(genesis_root=complete_environment)
        
        # Run all validations
        results = {}
        
        # Mock external service calls
        with patch("aiohttp.ClientSession") as mock_session:
            # Mock Prometheus response
            mock_response = AsyncMock()
            mock_response.status = 404  # Service not running
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            results["monitoring"] = await monitoring.validate()
            results["backup"] = await backup.validate()
            results["documentation"] = await docs.validate()
            results["deployment"] = await deployment.validate()
            results["health"] = await health.validate()
        
        # Verify results
        assert all(r["validator"] for r in results.values())
        assert all(r["score"] >= 0 for r in results.values())
        
        # Check that most validators pass with complete setup
        passing_count = sum(1 for r in results.values() if r["passed"])
        assert passing_count >= 3  # At least 3 validators should pass

    @pytest.mark.asyncio
    async def test_monitoring_with_real_configs(self, complete_environment):
        """Test monitoring validator with real configuration files."""
        validator = MonitoringValidator(genesis_root=complete_environment)
        
        # Test individual checks
        config_result = await validator._validate_monitoring_configs()
        assert config_result["passed"] is True
        assert "prometheus.yml" in config_result["configs_found"]
        
        dashboards_result = await validator._validate_grafana_dashboards()
        assert dashboards_result["passed"] is True
        assert len(dashboards_result["dashboards_found"]) == 5
        
        alerts_result = await validator._validate_alert_rules()
        assert alerts_result["passed"] is True
        assert len(alerts_result["alerts_found"]) == 5

    @pytest.mark.asyncio
    async def test_backup_restore_cycle(self, complete_environment):
        """Test complete backup and restore cycle."""
        validator = BackupValidator(genesis_root=complete_environment)
        
        # Test backup creation
        backup_result = await validator._test_backup_creation()
        assert backup_result["passed"] is True
        assert backup_result["backup_size"] > 0
        
        # Test restore
        restore_result = await validator._test_restore_procedure()
        assert restore_result["passed"] is True
        assert restore_result["restore_time"] < 300
        
        # Test backup schedule and scripts
        scripts_result = await validator._check_backup_scripts()
        assert scripts_result["passed"] is True
        
        offsite_result = await validator._validate_offsite_storage()
        assert offsite_result["passed"] is True

    @pytest.mark.asyncio
    async def test_documentation_completeness(self, complete_environment):
        """Test documentation completeness across all files."""
        validator = DocumentationValidator(genesis_root=complete_environment)
        
        result = await validator.validate()
        
        assert result["passed"] is True
        assert result["score"] >= 80
        
        # Check specific documentation aspects
        assert result["checks"]["required_docs"]["passed"] is True
        assert result["checks"]["runbook_completeness"]["passed"] is True
        assert result["checks"]["api_documentation"]["passed"] is True
        assert result["checks"]["readme_validation"]["passed"] is True

    @pytest.mark.asyncio
    async def test_deployment_readiness(self, complete_environment):
        """Test deployment readiness with all components."""
        validator = DeploymentValidator(genesis_root=complete_environment)
        
        result = await validator.validate()
        
        # Check critical deployment components
        assert result["checks"]["rollback_plan"]["passed"] is True
        assert result["checks"]["blue_green_deployment"]["passed"] is True
        assert result["checks"]["deployment_scripts"]["passed"] is True
        assert result["checks"]["container_configuration"]["passed"] is True
        
        # Kubernetes should be valid if present
        k8s_check = result["checks"]["kubernetes_manifests"]
        if (complete_environment / "kubernetes").exists():
            assert k8s_check["passed"] is True

    @pytest.mark.asyncio
    async def test_health_check_implementation(self, complete_environment):
        """Test health check implementation detection."""
        validator = HealthCheckValidator(genesis_root=complete_environment)
        
        # Test endpoint detection in code
        endpoints_result = await validator._check_health_endpoints_in_code()
        assert "/health" in endpoints_result["endpoints_found"]
        assert "/health/live" in endpoints_result["endpoints_found"]
        assert "/health/ready" in endpoints_result["endpoints_found"]
        
        # Test dependency checks
        deps_result = await validator._check_dependencies_in_code()
        assert "database" in deps_result["dependencies_checked"]
        assert "redis" in deps_result["dependencies_checked"]
        
        # Test implementation check
        impl_result = await validator._check_health_implementation()
        assert impl_result["passed"] is True
        assert impl_result["implementation_found"] is True

    @pytest.mark.asyncio
    async def test_validator_report_generation(self, complete_environment):
        """Test report generation for all validators."""
        validators = [
            MonitoringValidator(genesis_root=complete_environment),
            BackupValidator(genesis_root=complete_environment),
            DocumentationValidator(genesis_root=complete_environment),
            DeploymentValidator(genesis_root=complete_environment),
            HealthCheckValidator(genesis_root=complete_environment),
        ]
        
        for validator in validators:
            # Run validation
            with patch("aiohttp.ClientSession"):
                await validator.validate()
            
            # Generate report
            report = validator.generate_report()
            
            # Verify report content
            assert "VALIDATION REPORT" in report
            assert "Overall Status:" in report
            assert "Score:" in report
            assert "CHECK RESULTS:" in report
            assert "Execution Time:" in report

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test error handling in validators."""
        # Test with non-existent directory
        validators = [
            MonitoringValidator(genesis_root=Path("/non/existent/path")),
            BackupValidator(genesis_root=Path("/non/existent/path")),
            DocumentationValidator(genesis_root=Path("/non/existent/path")),
            DeploymentValidator(genesis_root=Path("/non/existent/path")),
            HealthCheckValidator(genesis_root=Path("/non/existent/path")),
        ]
        
        for validator in validators:
            # Should not raise exceptions
            with patch("aiohttp.ClientSession"):
                result = await validator.validate()
            
            assert result["validator"] is not None
            assert "checks" in result
            assert result["score"] >= 0  # Should have a score even on failure