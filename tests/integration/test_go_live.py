"""Integration tests for go-live validation suite."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

from scripts.production_readiness import ProductionReadinessValidator
from scripts.go_live_dashboard import GoLiveDashboard


class TestProductionReadinessIntegration:
    """Integration tests for production readiness validation."""
    
    @pytest.fixture
    def validator(self):
        return ProductionReadinessValidator()
    
    @pytest.mark.asyncio
    async def test_full_validation_flow(self, validator):
        """Test complete validation flow."""
        # Mock all validators to return passing results
        with patch.object(validator.validators['test_coverage'], 'validate') as mock_test, \
             patch.object(validator.validators['stability'], 'validate') as mock_stability, \
             patch.object(validator.validators['security'], 'validate') as mock_security, \
             patch.object(validator.validators['performance'], 'validate') as mock_perf, \
             patch.object(validator.validators['disaster_recovery'], 'validate') as mock_dr, \
             patch.object(validator.validators['paper_trading'], 'validate') as mock_paper:
            
            # Configure all validators to pass
            mock_test.return_value = {
                "passed": True,
                "details": {
                    "unit_coverage": 95,
                    "integration_pass_rate": 100,
                },
            }
            
            mock_stability.return_value = {
                "passed": True,
                "details": {
                    "hours_stable": 50,
                },
            }
            
            mock_security.return_value = {
                "passed": True,
                "details": {
                    "critical_issues": 0,
                },
            }
            
            mock_perf.return_value = {
                "passed": True,
                "details": {
                    "p99_latency_ms": 45,
                },
            }
            
            mock_dr.return_value = {
                "passed": True,
                "details": {},
            }
            
            mock_paper.return_value = {
                "passed": True,
                "details": {
                    "total_profit": 15000,
                },
            }
            
            result = await validator.run_validation()
            
            assert result["assessment"]["recommendation"] == "PENDING"  # Manual items still pending
            assert result["assessment"]["passed_count"] > 0
    
    @pytest.mark.asyncio
    async def test_validation_with_failures(self, validator):
        """Test validation with some failures."""
        with patch.object(validator.validators['test_coverage'], 'validate') as mock_test:
            mock_test.return_value = {
                "passed": False,
                "details": {
                    "unit_coverage": 70,  # Below threshold
                    "integration_pass_rate": 80,
                },
            }
            
            result = await validator.run_validation()
            
            assert result["assessment"]["recommendation"] != "GO"
            assert result["assessment"]["failed_count"] > 0
    
    def test_check_item_passed(self, validator):
        """Test acceptance criteria checking."""
        # Test AC1 - Unit tests with >90% coverage
        result = validator._check_item_passed("AC1", {
            "passed": True,
            "details": {"unit_coverage": 95},
        })
        assert result is True
        
        result = validator._check_item_passed("AC1", {
            "passed": True,
            "details": {"unit_coverage": 85},
        })
        assert result is False
        
        # Test AC5 - Performance p99 < 50ms
        result = validator._check_item_passed("AC5", {
            "passed": True,
            "details": {"p99_latency_ms": 45},
        })
        assert result is True
        
        result = validator._check_item_passed("AC5", {
            "passed": True,
            "details": {"p99_latency_ms": 55},
        })
        assert result is False
    
    def test_generate_assessment(self, validator):
        """Test assessment generation."""
        checklist_results = [
            {"required": True, "passed": True},
            {"required": True, "passed": True},
            {"required": True, "passed": False},
            {"required": True, "passed": None},  # Manual
            {"required": False, "passed": False},
        ]
        
        assessment = validator._generate_assessment(checklist_results)
        
        assert assessment["recommendation"] == "PENDING"  # Has manual items
        assert assessment["passed_count"] == 2
        assert assessment["failed_count"] == 1
        assert assessment["manual_count"] == 1


class TestGoLiveDashboardIntegration:
    """Integration tests for go-live dashboard."""
    
    @pytest.fixture
    def dashboard(self):
        return GoLiveDashboard()
    
    @pytest.mark.asyncio
    async def test_dashboard_run(self, dashboard):
        """Test dashboard execution."""
        with patch.object(dashboard.validator, 'run_validation') as mock_validate:
            mock_validate.return_value = {
                "assessment": {
                    "recommendation": "GO",
                    "readiness_score": 100,
                    "reason": "All requirements met",
                    "passed_count": 10,
                    "failed_count": 0,
                    "manual_count": 0,
                    "total_required": 10,
                },
                "checklist": [
                    {
                        "id": "AC1",
                        "name": "Unit Tests",
                        "description": "All unit tests passing",
                        "status": "✅",
                        "passed": True,
                        "required": True,
                        "details": {"unit_coverage": 95},
                    },
                ],
                "report": "test_report.html",
            }
            
            with patch.object(dashboard, '_display_dashboard'):
                with patch.object(dashboard, '_generate_html_report'):
                    exit_code = await dashboard.run()
                    
                    assert exit_code == 0  # GO recommendation
    
    @pytest.mark.asyncio
    async def test_dashboard_no_go(self, dashboard):
        """Test dashboard with NO-GO result."""
        with patch.object(dashboard.validator, 'run_validation') as mock_validate:
            mock_validate.return_value = {
                "assessment": {
                    "recommendation": "NO-GO",
                    "readiness_score": 60,
                    "reason": "Critical failures detected",
                    "passed_count": 6,
                    "failed_count": 4,
                    "manual_count": 0,
                    "total_required": 10,
                },
                "checklist": [],
                "report": "test_report.html",
            }
            
            with patch.object(dashboard, '_display_dashboard'):
                with patch.object(dashboard, '_generate_html_report'):
                    exit_code = await dashboard.run()
                    
                    assert exit_code == 1  # NO-GO recommendation


class TestEndToEndValidation:
    """End-to-end validation tests."""
    
    @pytest.mark.asyncio
    async def test_complete_validation_pipeline(self, tmp_path):
        """Test complete validation pipeline."""
        # Setup test environment
        test_dir = tmp_path / "test_genesis"
        test_dir.mkdir()
        
        # Create minimal test files
        (test_dir / ".genesis").mkdir()
        (test_dir / ".genesis" / "logs").mkdir()
        (test_dir / ".genesis" / "data").mkdir()
        (test_dir / ".genesis" / "state").mkdir()
        
        # Create test stability log
        stability_log = test_dir / ".genesis" / "logs" / "stability_test.json"
        stability_log.write_text(json.dumps({
            "duration_hours": 48,
            "error_count": 0,
            "total_events": 1000,
            "memory_samples": [{"memory_mb": 100}],
        }))
        
        # Run validation
        validator = ProductionReadinessValidator()
        
        # Mock validators to use test directory
        with patch.object(validator.validators['stability'], 'stability_log_file', stability_log):
            with patch.object(validator.validators['test_coverage'], 'validate') as mock_test:
                mock_test.return_value = {
                    "passed": True,
                    "details": {
                        "unit_coverage": 95,
                        "integration_pass_rate": 100,
                    },
                }
                
                # Run validation
                result = await validator.run_validation()
                
                assert "assessment" in result
                assert "checklist" in result
                assert len(result["checklist"]) > 0
    
    @pytest.mark.asyncio
    async def test_report_generation(self, tmp_path):
        """Test HTML report generation."""
        validator = ProductionReadinessValidator()
        
        # Mock validation results
        with patch.object(validator, 'run_validation') as mock_validate:
            mock_validate.return_value = {
                "assessment": {
                    "recommendation": "GO",
                    "readiness_score": 95,
                    "reason": "System ready",
                    "passed_count": 9,
                    "failed_count": 1,
                    "manual_count": 0,
                    "total_required": 10,
                },
                "checklist": [
                    {
                        "id": "AC1",
                        "name": "Tests",
                        "description": "Unit tests",
                        "status": "✅",
                        "passed": True,
                        "required": True,
                        "details": {},
                    },
                ],
                "report": str(tmp_path / "report.html"),
                "timestamp": "2024-01-01T00:00:00",
            }
            
            dashboard = GoLiveDashboard()
            dashboard.results = await mock_validate.return_value
            
            # Generate report
            with patch('pathlib.Path.mkdir'):
                with patch('pathlib.Path.write_text') as mock_write:
                    dashboard._generate_html_report()
                    
                    # Verify HTML was generated
                    mock_write.assert_called()
                    html_content = mock_write.call_args[0][0]
                    assert "Genesis Go-Live Dashboard" in html_content
                    assert "GO" in html_content


@pytest.mark.integration
class TestValidationWithRealFiles:
    """Integration tests with real file system."""
    
    @pytest.mark.asyncio
    async def test_security_scanner_with_files(self, tmp_path):
        """Test security scanner with actual files."""
        from genesis.validation.security_scanner import SecurityScanner
        
        # Create test code files
        code_dir = tmp_path / "genesis"
        code_dir.mkdir()
        
        # File with hardcoded secret
        bad_file = code_dir / "bad.py"
        bad_file.write_text('API_KEY = "sk_live_12345"')
        
        # Good file
        good_file = code_dir / "good.py"
        good_file.write_text('API_KEY = os.environ.get("API_KEY")')
        
        scanner = SecurityScanner()
        scanner.code_paths = [str(code_dir)]
        
        result = await scanner._check_hardcoded_secrets()
        
        assert result["secrets_found"] is True
        assert result["count"] > 0
        assert any("bad.py" in loc["file"] for loc in result["locations"])
    
    @pytest.mark.asyncio
    async def test_dr_validator_with_backups(self, tmp_path):
        """Test DR validator with backup files."""
        from genesis.validation.dr_validator import DisasterRecoveryValidator
        
        # Create backup directory
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()
        
        # Create test backup
        import tarfile
        backup_file = backup_dir / "test_backup.tar.gz"
        with tarfile.open(backup_file, "w:gz") as tar:
            test_file = tmp_path / "test.txt"
            test_file.write_text("test data")
            tar.add(test_file, arcname="test.txt")
        
        validator = DisasterRecoveryValidator()
        validator.backup_dir = backup_dir
        
        result = await validator._test_backup_procedures()
        
        assert "size_mb" in result
        assert result["integrity_verified"] is True