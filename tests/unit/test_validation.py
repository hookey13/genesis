"""Unit tests for validation modules."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from decimal import Decimal

from genesis.validation.test_validator import TestValidator
from genesis.validation.stability_tester import StabilityTester
from genesis.validation.security_scanner import SecurityScanner
from genesis.validation.performance_validator import PerformanceValidator
from genesis.validation.dr_validator import DisasterRecoveryValidator
from genesis.validation.paper_trading_validator import PaperTradingValidator


class TestTestValidator:
    """Tests for TestValidator class."""
    
    @pytest.fixture
    def validator(self):
        return TestValidator()
    
    @pytest.mark.asyncio
    async def test_validate_success(self, validator):
        """Test successful validation."""
        with patch.object(validator, '_run_unit_tests') as mock_unit, \
             patch.object(validator, '_run_integration_tests') as mock_integration, \
             patch.object(validator, '_analyze_coverage') as mock_coverage, \
             patch.object(validator, '_check_critical_path_coverage') as mock_critical, \
             patch.object(validator, '_check_risk_coverage') as mock_risk:
            
            mock_unit.return_value = AsyncMock(return_value={
                "passed": True,
                "total": 100,
                "failed": 0,
                "pass_rate": 100,
            })
            
            mock_integration.return_value = AsyncMock(return_value={
                "passed": True,
                "total": 50,
                "failed": 0,
                "pass_rate": 100,
            })
            
            mock_coverage.return_value = AsyncMock(return_value={
                "overall_coverage": 95,
                "module_coverage": {},
                "lines_covered": 950,
                "lines_total": 1000,
            })
            
            mock_critical.return_value = {
                "passed": True,
                "average": 100,
                "paths_at_100": 5,
                "total_paths": 5,
            }
            
            mock_risk.return_value = {
                "passed": True,
                "average": 92,
                "components_above_90": 4,
                "total_components": 4,
            }
            
            result = await validator.validate()
            
            assert result["passed"] is True
            assert result["details"]["unit_tests_passed"] is True
            assert result["details"]["unit_coverage"] == 95
    
    @pytest.mark.asyncio
    async def test_validate_failure(self, validator):
        """Test validation failure."""
        with patch.object(validator, '_run_unit_tests') as mock_unit:
            mock_unit.return_value = AsyncMock(return_value={
                "passed": False,
                "total": 100,
                "failed": 10,
                "pass_rate": 90,
            })
            
            result = await validator.validate()
            
            assert result["passed"] is False
    
    def test_check_critical_path_coverage(self, validator):
        """Test critical path coverage checking."""
        coverage_analysis = {
            "module_coverage": {
                "engine/risk_engine.py": 100,
                "engine/executor/market.py": 100,
                "utils/math.py": 100,
            }
        }
        
        result = validator._check_critical_path_coverage(coverage_analysis)
        
        assert result["passed"] is True
        assert result["average"] == 100
    
    def test_generate_recommendations(self, validator):
        """Test recommendation generation."""
        unit_results = {"passed": False, "failed": 5}
        integration_results = {"passed": True, "failed": 0}
        coverage_analysis = {"overall_coverage": 85}
        critical_coverage = {"passed": False, "paths_at_100": 3, "total_paths": 5}
        risk_coverage = {"passed": True, "average": 92}
        
        recommendations = validator._generate_recommendations(
            unit_results,
            integration_results,
            coverage_analysis,
            critical_coverage,
            risk_coverage,
        )
        
        assert len(recommendations) > 0
        assert any("Fix 5 failing unit tests" in r for r in recommendations)


class TestStabilityTester:
    """Tests for StabilityTester class."""
    
    @pytest.fixture
    def tester(self):
        return StabilityTester()
    
    @pytest.mark.asyncio
    async def test_validate_success(self, tester, tmp_path):
        """Test successful stability validation."""
        # Create test stability log
        test_log = tmp_path / "stability_test.json"
        test_data = {
            "duration_hours": 50,
            "memory_samples": [
                {"memory_mb": 100},
                {"memory_mb": 105},
            ],
            "error_count": 5,
            "total_events": 1000,
            "restart_count": 1,
            "transactions": 5000,
            "latency_samples": [10, 15, 20],
            "critical_failures": [],
        }
        test_log.write_text(json.dumps(test_data))
        
        tester.stability_log_file = test_log
        
        result = await tester.validate()
        
        assert result["passed"] is True
        assert result["details"]["hours_stable"] == 50
        assert result["details"]["error_rate"] == 0.005
    
    @pytest.mark.asyncio
    async def test_validate_failure_insufficient_hours(self, tester, tmp_path):
        """Test validation failure due to insufficient hours."""
        test_log = tmp_path / "stability_test.json"
        test_data = {
            "duration_hours": 24,  # Less than required 48
            "memory_samples": [{"memory_mb": 100}],
            "error_count": 0,
            "total_events": 1000,
            "restart_count": 0,
        }
        test_log.write_text(json.dumps(test_data))
        
        tester.stability_log_file = test_log
        
        result = await tester.validate()
        
        assert result["passed"] is False
        assert result["details"]["hours_stable"] == 24
    
    def test_analyze_stability_data(self, tester):
        """Test stability data analysis."""
        test_data = {
            "duration_hours": 48,
            "memory_samples": [
                {"memory_mb": 100},
                {"memory_mb": 110},
            ],
            "error_count": 10,
            "total_events": 1000,
            "restart_count": 2,
            "latency_samples": [10, 20, 30],
        }
        
        analysis = tester._analyze_stability_data(test_data)
        
        assert analysis["hours_stable"] == 48
        assert analysis["memory_growth_percent"] == 10
        assert analysis["error_rate"] == 0.01
        assert analysis["restart_count"] == 2
        assert analysis["average_latency_ms"] == 20


class TestSecurityScanner:
    """Tests for SecurityScanner class."""
    
    @pytest.fixture
    def scanner(self):
        return SecurityScanner()
    
    @pytest.mark.asyncio
    async def test_validate_success(self, scanner):
        """Test successful security validation."""
        with patch.object(scanner, '_run_pip_audit') as mock_pip, \
             patch.object(scanner, '_run_bandit') as mock_bandit, \
             patch.object(scanner, '_check_hardcoded_secrets') as mock_secrets, \
             patch.object(scanner, '_check_api_key_management') as mock_api, \
             patch.object(scanner, '_check_file_permissions') as mock_perms:
            
            mock_pip.return_value = AsyncMock(return_value={
                "total_vulnerabilities": 0,
                "critical_count": 0,
                "high_count": 0,
                "moderate_count": 0,
                "low_count": 0,
                "vulnerabilities": [],
            })
            
            mock_bandit.return_value = AsyncMock(return_value={
                "total_issues": 0,
                "high_severity_count": 0,
                "medium_severity_count": 0,
                "low_severity_count": 0,
                "issues": [],
            })
            
            mock_secrets.return_value = AsyncMock(return_value={
                "secrets_found": False,
                "count": 0,
                "locations": [],
            })
            
            mock_api.return_value = {
                "secure": True,
                "issues": [],
            }
            
            mock_perms.return_value = {
                "secure": True,
                "issues": [],
            }
            
            result = await scanner.validate()
            
            assert result["passed"] is True
            assert result["details"]["critical_issues"] == 0
    
    @pytest.mark.asyncio
    async def test_check_hardcoded_secrets(self, scanner, tmp_path):
        """Test hardcoded secrets detection."""
        # Create test file with secret
        test_file = tmp_path / "test.py"
        test_file.write_text('API_KEY = "secret123"')
        
        scanner.code_paths = [str(tmp_path)]
        
        result = await scanner._check_hardcoded_secrets()
        
        assert result["secrets_found"] is True
        assert result["count"] > 0
    
    def test_check_api_key_management(self, scanner, tmp_path):
        """Test API key management checking."""
        # Create .env.example
        env_example = tmp_path / ".env.example"
        env_example.write_text("API_KEY=your_key_here")
        
        # Create .gitignore
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(".env")
        
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.side_effect = lambda: True
            with patch('pathlib.Path.read_text') as mock_read:
                mock_read.return_value = ".env"
                
                result = scanner._check_api_key_management()
                
                assert result["secure"] is True


class TestPerformanceValidator:
    """Tests for PerformanceValidator class."""
    
    @pytest.fixture
    def validator(self):
        return PerformanceValidator()
    
    @pytest.mark.asyncio
    async def test_validate_success(self, validator):
        """Test successful performance validation."""
        with patch.object(validator, '_test_order_latency') as mock_order, \
             patch.object(validator, '_test_websocket_performance') as mock_ws, \
             patch.object(validator, '_test_database_performance') as mock_db, \
             patch.object(validator, '_test_stress_load') as mock_stress:
            
            mock_order.return_value = AsyncMock(return_value={
                "p50": 10,
                "p95": 25,
                "p99": 45,
            })
            
            mock_ws.return_value = AsyncMock(return_value={
                "avg_processing_ms": 3,
            })
            
            mock_db.return_value = AsyncMock(return_value={
                "avg_query_ms": 5,
            })
            
            mock_stress.return_value = AsyncMock(return_value={
                "system_stable": True,
                "max_throughput": 2000,
                "cpu_usage": 60,
                "memory_usage": 500,
            })
            
            result = await validator.validate()
            
            assert result["passed"] is True
            assert result["details"]["p99_latency_ms"] == 45
    
    @pytest.mark.asyncio
    async def test_test_order_latency(self, validator):
        """Test order latency testing."""
        result = await validator._test_order_latency()
        
        assert "p50" in result
        assert "p95" in result
        assert "p99" in result
        assert result["samples"] > 0
    
    def test_generate_recommendations(self, validator):
        """Test performance recommendation generation."""
        order_latency = {"p99": 60, "p95": 40}
        ws_performance = {"avg_processing_ms": 10}
        db_performance = {"avg_query_ms": 15}
        stress_results = {"system_stable": False, "error_rate": 0.05}
        
        recommendations = validator._generate_recommendations(
            order_latency,
            ws_performance,
            db_performance,
            stress_results,
        )
        
        assert len(recommendations) > 0
        assert any("p99 latency" in r for r in recommendations)


class TestDisasterRecoveryValidator:
    """Tests for DisasterRecoveryValidator class."""
    
    @pytest.fixture
    def validator(self):
        return DisasterRecoveryValidator()
    
    @pytest.mark.asyncio
    async def test_validate_success(self, validator):
        """Test successful DR validation."""
        with patch.object(validator, '_test_backup_procedures') as mock_backup, \
             patch.object(validator, '_test_restore_procedures') as mock_restore, \
             patch.object(validator, '_test_failover') as mock_failover, \
             patch.object(validator, '_test_position_recovery') as mock_position, \
             patch.object(validator, '_test_rto') as mock_rto, \
             patch.object(validator, '_test_rpo') as mock_rpo:
            
            mock_backup.return_value = AsyncMock(return_value={
                "passed": True,
                "size_mb": 50,
            })
            
            mock_restore.return_value = AsyncMock(return_value={
                "passed": True,
                "time_seconds": 60,
                "integrity_verified": True,
            })
            
            mock_failover.return_value = AsyncMock(return_value={
                "passed": True,
            })
            
            mock_position.return_value = AsyncMock(return_value={
                "passed": True,
            })
            
            mock_rto.return_value = AsyncMock(return_value={
                "passed": True,
                "minutes": 10,
            })
            
            mock_rpo.return_value = AsyncMock(return_value={
                "passed": True,
                "minutes": 3,
            })
            
            result = await validator.validate()
            
            assert result["passed"] is True
            assert result["details"]["rto_minutes"] == 10
            assert result["details"]["rpo_minutes"] == 3
    
    @pytest.mark.asyncio
    async def test_test_backup_procedures(self, validator, tmp_path):
        """Test backup procedure testing."""
        validator.backup_dir = tmp_path / "backups"
        validator.backup_dir.mkdir()
        
        # Create test backup
        test_backup = validator.backup_dir / "test.tar.gz"
        test_backup.write_text("backup data")
        
        result = await validator._test_backup_procedures()
        
        assert "size_mb" in result
        assert "backup_path" in result
    
    def test_check_encryption_configured(self, validator):
        """Test encryption configuration checking."""
        with patch.dict('os.environ', {'RESTIC_PASSWORD': 'secret'}):
            result = validator._check_encryption_configured()
            assert result is True


class TestPaperTradingValidator:
    """Tests for PaperTradingValidator class."""
    
    @pytest.fixture
    def validator(self):
        return PaperTradingValidator()
    
    @pytest.mark.asyncio
    async def test_validate_success(self, validator):
        """Test successful paper trading validation."""
        with patch.object(validator, '_load_trading_results') as mock_load:
            mock_load.return_value = AsyncMock(return_value={
                "has_data": True,
                "trades": [
                    {
                        "timestamp": "2024-01-01T00:00:00",
                        "pnl": Decimal("500"),
                    } for _ in range(100)
                ] + [
                    {
                        "timestamp": "2024-02-01T00:00:00",
                        "pnl": Decimal("100"),
                    } for _ in range(50)
                ],
            })
            
            result = await validator.validate()
            
            assert result["passed"] is True
            assert result["details"]["total_profit"] >= 10000
    
    def test_calculate_metrics(self, validator):
        """Test metrics calculation."""
        trades = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "pnl": Decimal("100"),
            },
            {
                "timestamp": "2024-01-02T00:00:00",
                "pnl": Decimal("-50"),
            },
            {
                "timestamp": "2024-01-03T00:00:00",
                "pnl": Decimal("200"),
            },
        ]
        
        metrics = validator._calculate_metrics(trades)
        
        assert metrics["total_profit"] == Decimal("250")
        assert metrics["total_trades"] == 3
        assert metrics["win_rate"] > 0
    
    def test_generate_simulated_trades(self, validator):
        """Test simulated trade generation."""
        trades = validator._generate_simulated_trades()
        
        assert len(trades) > 0
        assert all("pnl" in trade for trade in trades)
        assert all("timestamp" in trade for trade in trades)