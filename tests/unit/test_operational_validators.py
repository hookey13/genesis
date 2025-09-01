"""Unit tests for operational validators."""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
import yaml

from genesis.validation.operational import (
    BackupValidator,
    DeploymentValidator,
    DocumentationValidator,
    HealthCheckValidator,
    MonitoringValidator,
)


class TestMonitoringValidator:
    """Test monitoring validator functionality."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create validator instance with temp directory."""
        return MonitoringValidator(genesis_root=tmp_path)

    @pytest.fixture
    def setup_monitoring_configs(self, tmp_path):
        """Set up monitoring configuration files."""
        # Create Prometheus config
        prometheus_dir = tmp_path / "config" / "prometheus"
        prometheus_dir.mkdir(parents=True, exist_ok=True)
        
        prometheus_config = {
            "global": {"scrape_interval": "15s"},
            "scrape_configs": [{"job_name": "genesis"}],
        }
        with open(prometheus_dir / "prometheus.yml", "w") as f:
            yaml.dump(prometheus_config, f)
        
        # Create alert rules
        alert_rules = {
            "groups": [
                {
                    "name": "genesis_alerts",
                    "rules": [
                        {"alert": "high_error_rate"},
                        {"alert": "low_system_health"},
                        {"alert": "high_latency"},
                        {"alert": "position_limit_exceeded"},
                        {"alert": "api_rate_limit_warning"},
                    ],
                }
            ]
        }
        with open(prometheus_dir / "alerts.yml", "w") as f:
            yaml.dump(alert_rules, f)
        
        # Create Grafana config
        grafana_dir = tmp_path / "config" / "grafana"
        grafana_dir.mkdir(parents=True, exist_ok=True)
        
        with open(grafana_dir / "grafana.ini", "w") as f:
            f.write("[server]\nhttp_port = 3000\n")
        
        # Create dashboard files
        dashboards = [
            {"uid": "trading-overview", "title": "Trading Overview"},
            {"uid": "system-health", "title": "System Health"},
            {"uid": "performance-metrics", "title": "Performance Metrics"},
            {"uid": "error-tracking", "title": "Error Tracking"},
            {"uid": "api-usage", "title": "API Usage"},
        ]
        for dashboard in dashboards:
            with open(grafana_dir / f"{dashboard['uid']}.json", "w") as f:
                f.write(f'{{"uid": "{dashboard["uid"]}", "title": "{dashboard["title"]}"}}')

    @pytest.mark.asyncio
    async def test_validate_prometheus_metrics_success(self, validator):
        """Test successful Prometheus metrics validation."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "data": [
                        "genesis_order_execution_latency_seconds",
                        "genesis_api_calls_total",
                        "genesis_active_positions",
                        "genesis_memory_usage_bytes",
                        "genesis_total_pnl_usdt",
                        "genesis_error_rate_per_minute",
                        "genesis_system_health_score",
                    ]
                }
            )
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await validator._validate_prometheus_metrics()
            
            assert result["passed"] is True
            assert len(result["metrics_missing"]) == 0
            assert len(result["metrics_found"]) == 7

    @pytest.mark.asyncio
    async def test_validate_prometheus_metrics_missing(self, validator):
        """Test Prometheus metrics validation with missing metrics."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "data": [
                        "genesis_order_execution_latency_seconds",
                        "genesis_api_calls_total",
                    ]
                }
            )
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await validator._validate_prometheus_metrics()
            
            assert result["passed"] is False
            assert len(result["metrics_missing"]) > 0
            assert "genesis_system_health_score" in result["metrics_missing"]

    @pytest.mark.asyncio
    async def test_validate_monitoring_configs(self, validator, setup_monitoring_configs):
        """Test monitoring configuration files validation."""
        result = await validator._validate_monitoring_configs()
        
        assert result["passed"] is True
        assert "prometheus.yml" in result["configs_found"]
        assert "grafana.ini" in result["configs_found"]
        assert "alerts.yml" in result["configs_found"]
        assert len(result["configs_missing"]) == 0

    @pytest.mark.asyncio
    async def test_full_validation(self, validator, setup_monitoring_configs):
        """Test full monitoring validation."""
        with patch.object(validator, "_validate_prometheus_metrics") as mock_metrics:
            mock_metrics.return_value = {"passed": True, "metrics_found": [], "metrics_missing": []}
            
            result = await validator.validate()
            
            assert "monitoring" in result["validator"]
            assert result["score"] >= 0
            assert "checks" in result
            assert "monitoring_configs" in result["checks"]


class TestBackupValidator:
    """Test backup validator functionality."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create validator instance with temp directory."""
        return BackupValidator(genesis_root=tmp_path)

    @pytest.fixture
    def setup_backup_environment(self, tmp_path):
        """Set up backup environment."""
        # Create backup directory
        backup_dir = tmp_path / ".genesis" / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database file
        db_dir = tmp_path / ".genesis" / "data"
        db_dir.mkdir(parents=True, exist_ok=True)
        (db_dir / "genesis.db").touch()
        
        # Create config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "settings.yaml").touch()
        
        # Create state directory
        state_dir = tmp_path / ".genesis" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "tier_state.json").touch()
        
        # Create backup script
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        backup_script = scripts_dir / "backup.sh"
        backup_script.write_text(
            "#!/bin/bash\n"
            "set -e\n"
            "echo 'Backing up...'\n"
            "restic backup /data\n"
            "s3 sync /backup s3://bucket/\n"
        )
        backup_script.chmod(0o755)
        
        restore_script = scripts_dir / "restore.sh"
        restore_script.write_text(
            "#!/bin/bash\n"
            "set -e\n"
            "echo 'Restoring...'\n"
        )
        restore_script.chmod(0o755)
        
        # Create .env file with offsite storage config
        env_file = tmp_path / ".env"
        env_file.write_text("DO_SPACES_KEY=xxx\nDO_SPACES_SECRET=yyy\n")

    @pytest.mark.asyncio
    async def test_backup_creation(self, validator, setup_backup_environment):
        """Test backup creation validation."""
        result = await validator._test_backup_creation()
        
        assert result["passed"] is True
        assert result["backup_size"] > 0
        assert "database" in result["components_backed_up"]
        assert "configuration" in result["components_backed_up"]

    @pytest.mark.asyncio
    async def test_restore_procedure(self, validator):
        """Test restore procedure validation."""
        result = await validator._test_restore_procedure()
        
        assert result["passed"] is True
        assert result["restore_time"] < 300
        assert len(result["components_restored"]) >= 3

    @pytest.mark.asyncio
    async def test_backup_encryption(self, validator, setup_backup_environment):
        """Test backup encryption validation."""
        result = await validator._verify_backup_encryption()
        
        assert result["passed"] is True
        assert "encryption" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_offsite_storage(self, validator, setup_backup_environment):
        """Test offsite storage validation."""
        result = await validator._validate_offsite_storage()
        
        assert result["passed"] is True
        assert result["storage_type"] == "DigitalOcean Spaces"

    @pytest.mark.asyncio
    async def test_backup_scripts(self, validator, setup_backup_environment):
        """Test backup scripts validation."""
        result = await validator._check_backup_scripts()
        
        assert result["passed"] is True
        assert "backup.sh" in result["scripts_found"]
        assert "restore.sh" in result["scripts_found"]
        assert len(result["scripts_missing"]) == 0


class TestDocumentationValidator:
    """Test documentation validator functionality."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create validator instance with temp directory."""
        return DocumentationValidator(genesis_root=tmp_path)

    @pytest.fixture
    def setup_documentation(self, tmp_path):
        """Set up documentation files."""
        # Create README
        readme = tmp_path / "README.md"
        readme.write_text(
            "# Genesis\n\n"
            "## Installation\n"
            "```bash\npip install -r requirements.txt\n```\n\n"
            "## Configuration\n"
            "Edit config/settings.yaml\n\n"
            "## Usage\n"
            "```python\npython -m genesis\n```\n\n"
            "## Architecture\n"
            "See docs/architecture.md\n\n"
            "## Testing\n"
            "Run pytest\n\n"
            "## Deployment\n"
            "See docs/deployment.md\n" + "\n" * 100
        )
        
        # Create docs directory
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Create runbook
        runbook = docs_dir / "runbook.md"
        runbook_content = "# Runbook\n\n"
        for scenario in [
            "System Startup",
            "Shutdown Procedures",
            "Emergency Position Closure",
            "API Rate Limit Handling",
            "Database Recovery",
            "Tilt Intervention",
            "Performance Degradation",
            "Security Incident Response",
            "Rollback Procedures",
            "Data Corruption Recovery",
        ]:
            runbook_content += f"## {scenario}\n1. Step one\n2. Step two\n3. Step three\n\n"
        runbook_content += "\n" * 150
        runbook.write_text(runbook_content)
        
        # Create other required docs
        (docs_dir / "architecture.md").write_text(
            "# Architecture\n## System Overview\n## Components\n## Data Flow\n## Security\n## Scalability\n" + "\n" * 150
        )
        (docs_dir / "deployment.md").write_text(
            "# Deployment\n## Prerequisites\n## Environment Setup\n## Deployment Steps\n## Verification\n## Rollback\n" + "\n" * 50
        )
        (docs_dir / "troubleshooting.md").write_text(
            "# Troubleshooting\n## Common Issues\n## Error Messages\n## Performance Issues\n## Connectivity Problems\n## Recovery Procedures\n" + "\n" * 100
        )
        (docs_dir / "monitoring.md").write_text(
            "# Monitoring\n## Metrics\n## Dashboards\n## Alerts\n## Log Analysis\n## Health Checks\n" + "\n" * 75
        )
        
        # Create API docs
        api_dir = docs_dir / "api"
        api_dir.mkdir(exist_ok=True)
        (api_dir / "endpoints.md").write_text(
            "# API Endpoints\n\n"
            "## Authentication:\nBearer token required\n\n"
            "## Endpoints:\n"
            "### GET /health\nParameters: None\nResponse: {status: ok}\nExample: curl http://localhost:8000/health\n\n"
            "### POST /orders\nParameters: {symbol, quantity, side}\nResponse: {order_id}\n"
        )

    @pytest.mark.asyncio
    async def test_required_docs(self, validator, setup_documentation):
        """Test required documentation files validation."""
        result = await validator._check_required_docs()
        
        assert result["passed"] is True
        assert "README.md" in result["docs_found"]
        assert len(result["docs_missing"]) == 0
        assert len(result["docs_incomplete"]) == 0

    @pytest.mark.asyncio
    async def test_runbook_completeness(self, validator, setup_documentation):
        """Test runbook completeness validation."""
        result = await validator._check_runbook_completeness()
        
        assert result["passed"] is True
        assert len(result["scenarios_covered"]) == 10
        assert len(result["scenarios_missing"]) == 0

    @pytest.mark.asyncio
    async def test_api_documentation(self, validator, setup_documentation):
        """Test API documentation validation."""
        result = await validator._verify_api_documentation()
        
        assert result["passed"] is True
        assert result["coverage"]["endpoints"] > 0
        assert result["coverage"]["parameters"] > 0
        assert result["coverage"]["responses"] > 0

    @pytest.mark.asyncio
    async def test_readme_validation(self, validator, setup_documentation):
        """Test README validation."""
        result = await validator._validate_readme()
        
        assert result["passed"] is True
        assert "Installation" in result["sections_found"]
        assert "Configuration" in result["sections_found"]
        assert "Usage" in result["sections_found"]
        assert len(result["sections_missing"]) == 0


class TestDeploymentValidator:
    """Test deployment validator functionality."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create validator instance with temp directory."""
        return DeploymentValidator(genesis_root=tmp_path)

    @pytest.fixture
    def setup_deployment(self, tmp_path):
        """Set up deployment configuration."""
        # Create Docker files
        docker_dir = tmp_path / "docker"
        docker_dir.mkdir(exist_ok=True)
        
        dockerfile = docker_dir / "Dockerfile"
        dockerfile.write_text(
            "FROM python:3.11 AS builder\n"
            "WORKDIR /app\n"
            "COPY . .\n"
            "RUN pip install -r requirements.txt\n"
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "USER nobody\n"
            "HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1\n"
            "LABEL version=1.0\n"
            "CMD [\"python\", \"-m\", \"genesis\"]\n"
        )
        
        compose_file = docker_dir / "docker-compose.yml"
        compose_content = {
            "version": "3.8",
            "services": {
                "genesis": {"build": ".", "ports": ["8000:8000"]},
                "nginx": {"image": "nginx:latest"},
            },
        }
        with open(compose_file, "w") as f:
            yaml.dump(compose_content, f)
        
        compose_prod = docker_dir / "docker-compose.prod.yml"
        compose_prod_content = {
            "version": "3.8",
            "services": {
                "genesis-blue": {"image": "genesis:blue"},
                "genesis-green": {"image": "genesis:green"},
                "nginx": {"image": "nginx:latest"},
            },
        }
        with open(compose_prod, "w") as f:
            yaml.dump(compose_prod_content, f)
        
        # Create deployment scripts
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        deploy_script = scripts_dir / "deploy.sh"
        deploy_script.write_text(
            "#!/bin/bash\n"
            "set -e\n"
            "echo 'Deploying...'\n"
            "docker-compose -f docker/docker-compose.prod.yml up -d genesis-blue\n"
            "health_check\n"
            "docker-compose -f docker/docker-compose.prod.yml stop genesis-green\n"
        )
        deploy_script.chmod(0o755)
        
        rollback_script = scripts_dir / "rollback.sh"
        rollback_script.write_text(
            "#!/bin/bash\n"
            "set -e\n"
            "echo 'Rolling back...'\n"
            "git checkout previous_tag\n"
            "docker-compose down\n"
            "restore_backup\n"
            "docker-compose up -d\n"
            "health_check\n"
        )
        rollback_script.chmod(0o755)
        
        # Create rollback documentation
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        (docs_dir / "rollback.md").write_text(
            "# Rollback Procedures\n\n"
            "1. Stop current deployment\n"
            "2. Checkout previous version\n"
            "3. Restore database backup\n"
            "4. Deploy previous version\n"
            "5. Verify health checks\n"
        )
        
        # Create staging config
        (tmp_path / ".env.staging").write_text("ENVIRONMENT=staging\n")

    @pytest.mark.asyncio
    async def test_rollback_plan(self, validator, setup_deployment):
        """Test rollback plan validation."""
        result = await validator._verify_rollback_plan()
        
        assert result["passed"] is True
        assert result["documentation_found"] is True
        assert result["script_found"] is True
        assert len(result["rollback_steps"]) > 0

    @pytest.mark.asyncio
    async def test_blue_green_deployment(self, validator, setup_deployment):
        """Test blue-green deployment validation."""
        result = await validator._check_blue_green_deployment()
        
        assert result["passed"] is True
        assert result["blue_green_configured"] is True

    @pytest.mark.asyncio
    async def test_deployment_scripts(self, validator, setup_deployment):
        """Test deployment scripts validation."""
        result = await validator._validate_deployment_scripts()
        
        assert result["passed"] is True
        assert "deploy.sh" in result["scripts_found"]
        assert "rollback.sh" in result["scripts_found"]

    @pytest.mark.asyncio
    async def test_container_configuration(self, validator, setup_deployment):
        """Test container configuration validation."""
        result = await validator._check_container_configuration()
        
        assert result["passed"] is True
        assert result["dockerfile_found"] is True
        assert len(result["compose_files"]) > 0

    @pytest.mark.asyncio
    async def test_staging_environment(self, validator, setup_deployment):
        """Test staging environment validation."""
        result = await validator._test_staging_environment()
        
        assert result["passed"] is True
        assert result["staging_configured"] is True


class TestHealthCheckValidator:
    """Test health check validator functionality."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create validator instance with temp directory."""
        return HealthCheckValidator(genesis_root=tmp_path)

    @pytest.fixture
    def setup_health_checks(self, tmp_path):
        """Set up health check implementation."""
        # Create API directory
        api_dir = tmp_path / "genesis" / "api"
        api_dir.mkdir(parents=True, exist_ok=True)
        
        # Create health endpoints
        health_file = api_dir / "health.py"
        health_file.write_text(
            "from fastapi import APIRouter\n\n"
            "router = APIRouter()\n\n"
            "@router.get('/health')\n"
            "async def health():\n"
            "    return {'status': 'healthy', 'timestamp': '2024-01-01', 'version': '1.0'}\n\n"
            "@router.get('/health/live')\n"
            "async def liveness():\n"
            "    return {'alive': True}\n\n"
            "@router.get('/health/ready')\n"
            "async def readiness():\n"
            "    db_health = await check_database()\n"
            "    redis_health = await check_redis()\n"
            "    return {'ready': db_health and redis_health}\n\n"
            "@router.get('/health/dependencies')\n"
            "async def dependencies():\n"
            "    return {\n"
            "        'database': await check_database(),\n"
            "        'redis': await check_redis(),\n"
            "        'exchange_api': await check_exchange(),\n"
            "        'websocket': await check_websocket()\n"
            "    }\n\n"
            "async def check_database():\n"
            "    return True\n\n"
            "async def check_redis():\n"
            "    return True\n\n"
            "async def check_exchange():\n"
            "    return True\n\n"
            "async def check_websocket():\n"
            "    return True\n"
        )
        
        # Create Kubernetes deployment
        k8s_dir = tmp_path / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        deployment_yaml = k8s_dir / "deployment.yaml"
        deployment_yaml.write_text(
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "spec:\n"
            "  template:\n"
            "    spec:\n"
            "      containers:\n"
            "      - name: genesis\n"
            "        livenessProbe:\n"
            "          httpGet:\n"
            "            path: /health/live\n"
            "            port: 8000\n"
            "        readinessProbe:\n"
            "          httpGet:\n"
            "            path: /health/ready\n"
            "            port: 8000\n"
        )

    @pytest.mark.asyncio
    async def test_health_endpoints_in_code(self, validator, setup_health_checks):
        """Test health endpoint detection in code."""
        result = await validator._check_health_endpoints_in_code()
        
        assert "/health" in result["endpoints_found"]
        assert "/health/live" in result["endpoints_found"]
        assert "/health/ready" in result["endpoints_found"]
        assert "/health/dependencies" in result["endpoints_found"]

    @pytest.mark.asyncio
    async def test_health_dependencies_in_code(self, validator, setup_health_checks):
        """Test health dependency checks in code."""
        result = await validator._check_dependencies_in_code()
        
        assert "database" in result["dependencies_checked"]
        assert "redis" in result["dependencies_checked"]
        assert "exchange" in result["dependencies_checked"]
        assert "websocket" in result["dependencies_checked"]

    @pytest.mark.asyncio
    async def test_probes_configuration(self, validator, setup_health_checks):
        """Test probe configuration validation."""
        result = await validator._check_probes()
        
        assert result["passed"] is True
        assert result["liveness_configured"] is True
        assert result["readiness_configured"] is True

    @pytest.mark.asyncio
    async def test_health_implementation(self, validator, setup_health_checks):
        """Test health check implementation validation."""
        result = await validator._check_health_implementation()
        
        assert result["passed"] is True
        assert result["implementation_found"] is True
        assert "health endpoint" in result["features"]
        assert "database check" in result["features"]

    @pytest.mark.asyncio
    async def test_response_times_skip(self, validator):
        """Test response time check when service not running."""
        result = await validator._test_response_times()
        
        assert result["passed"] is True
        assert "skipped" in result["message"].lower()