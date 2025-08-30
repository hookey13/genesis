"""Unit tests for TestValidator."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from genesis.validation.test_validator import TestValidator


@pytest.fixture
def test_validator(tmp_path):
    """Create TestValidator instance."""
    return TestValidator(genesis_root=tmp_path)


@pytest.fixture
def mock_test_results():
    """Create mock test results."""
    return {
        "passed": True,
        "total": 100,
        "failed": 0,
        "pass_rate": 100.0,
        "output": "All tests passed"
    }


@pytest.fixture
def mock_coverage_data():
    """Create mock coverage data."""
    return {
        "totals": {
            "percent_covered": 85.5,
            "covered_lines": 1500,
            "num_statements": 1754
        },
        "files": {
            "genesis/engine/risk_engine.py": {
                "summary": {"percent_covered": 100.0}
            },
            "genesis/engine/executor/market.py": {
                "summary": {"percent_covered": 95.0}
            },
            "genesis/utils/math.py": {
                "summary": {"percent_covered": 100.0}
            },
            "genesis/core/models.py": {
                "summary": {"percent_covered": 98.0}
            },
            "genesis/exchange/gateway.py": {
                "summary": {"percent_covered": 92.0}
            },
            "genesis/tilt/detector.py": {
                "summary": {"percent_covered": 88.0}
            },
            "genesis/ui/dashboard.py": {
                "summary": {"percent_covered": 65.0}
            }
        }
    }


class TestTestValidator:
    """Test TestValidator class."""
    
    def test_initialization(self, tmp_path):
        """Test TestValidator initialization."""
        validator = TestValidator(genesis_root=tmp_path)
        
        assert validator.genesis_root == tmp_path
        assert "money_paths" in validator.coverage_thresholds
        assert "risk_components" in validator.coverage_thresholds
        assert "core_modules" in validator.coverage_thresholds
        assert validator.coverage_thresholds["money_paths"]["threshold"] == 100
        assert validator.coverage_thresholds["risk_components"]["threshold"] == 90
        assert validator.coverage_history == []
        
    @pytest.mark.asyncio
    async def test_validate_success(self, test_validator, mock_test_results, mock_coverage_data):
        """Test successful validation."""
        # Mock subprocess calls
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock successful test runs
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(b"Tests passed", b""))
            mock_subprocess.return_value = mock_process
            
            # Mock coverage file
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=json.dumps(mock_coverage_data))):
                    result = await test_validator.validate()
        
        assert "passed" in result
        assert "score" in result
        assert "checks" in result
        assert "details" in result
        assert "recommendations" in result
        
    @pytest.mark.asyncio
    async def test_run_unit_tests(self, test_validator):
        """Test running unit tests."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(
                return_value=(b"10 passed in 5.00s", b"")
            )
            mock_subprocess.return_value = mock_process
            
            # Mock test results file
            test_results = {
                "exitcode": 0,
                "summary": {"total": 10, "passed": 10, "failed": 0}
            }
            
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=json.dumps(test_results))):
                    result = await test_validator._run_unit_tests()
        
        assert result["passed"] is True
        assert result["total"] == 10
        assert result["failed"] == 0
        assert result["pass_rate"] == 100.0
        
    @pytest.mark.asyncio
    async def test_run_unit_tests_with_failures(self, test_validator):
        """Test running unit tests with failures."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = 1
            mock_process.communicate = AsyncMock(
                return_value=(b"8 passed, 2 failed in 5.00s", b"")
            )
            mock_subprocess.return_value = mock_process
            
            # Mock test results file
            test_results = {
                "exitcode": 1,
                "summary": {"total": 10, "passed": 8, "failed": 2}
            }
            
            with patch("pathlib.Path.exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=json.dumps(test_results))):
                    result = await test_validator._run_unit_tests()
        
        assert result["passed"] is False
        assert result["total"] == 10
        assert result["failed"] == 2
        assert result["pass_rate"] == 80.0
        
    @pytest.mark.asyncio
    async def test_run_unit_tests_timeout(self, test_validator):
        """Test unit tests timeout."""
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_subprocess.return_value = mock_process
            
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                result = await test_validator._run_unit_tests()
        
        assert result["passed"] is False
        assert "error" in result
        assert "timed out" in result["error"]
        
    @pytest.mark.asyncio
    async def test_analyze_coverage(self, test_validator, mock_coverage_data):
        """Test coverage analysis."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_coverage_data))):
                result = await test_validator._analyze_coverage()
        
        assert result["overall_coverage"] == 85.5
        assert result["lines_covered"] == 1500
        assert result["lines_total"] == 1754
        assert "module_coverage" in result
        assert len(result["module_coverage"]) > 0
        
    @pytest.mark.asyncio
    async def test_analyze_coverage_no_file(self, test_validator):
        """Test coverage analysis when file doesn't exist."""
        with patch("pathlib.Path.exists", return_value=False):
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate = AsyncMock(return_value=(b"", b""))
                mock_subprocess.return_value = mock_process
                
                result = await test_validator._analyze_coverage()
        
        assert result["overall_coverage"] == 0
        assert result["module_coverage"] == {}
        
    def test_check_path_thresholds(self, test_validator, mock_coverage_data):
        """Test path-specific threshold checking."""
        coverage_analysis = {
            "overall_coverage": 85.5,
            "module_coverage": {
                "engine/risk_engine.py": 100.0,
                "engine/executor/market.py": 95.0,
                "utils/math.py": 100.0,
                "core/models.py": 98.0,
                "exchange/gateway.py": 92.0,
                "tilt/detector.py": 88.0,
                "ui/dashboard.py": 65.0
            }
        }
        
        results = test_validator._check_path_thresholds(coverage_analysis)
        
        assert "money_paths" in results
        assert "risk_components" in results
        
        # Money paths should fail if not all at 100%
        money_result = results["money_paths"]
        assert money_result["threshold"] == 100
        assert "average" in money_result
        assert "paths_meeting_threshold" in money_result
        
        # Risk components should pass if all >= 90%
        risk_result = results["risk_components"]
        assert risk_result["threshold"] == 90
        
    def test_track_coverage_trend(self, test_validator):
        """Test coverage trend tracking."""
        coverage_analysis = {
            "overall_coverage": 85.0,
            "lines_covered": 1000,
            "lines_total": 1176
        }
        
        # Track first entry
        test_validator._track_coverage_trend(coverage_analysis)
        assert len(test_validator.coverage_history) == 1
        
        # Track second entry
        coverage_analysis["overall_coverage"] = 87.0
        test_validator._track_coverage_trend(coverage_analysis)
        assert len(test_validator.coverage_history) == 2
        
        # Test history limit
        for i in range(30):
            test_validator._track_coverage_trend(coverage_analysis)
        assert len(test_validator.coverage_history) == 30
        
    def test_get_coverage_trend(self, test_validator):
        """Test getting coverage trend."""
        # Insufficient data
        trend = test_validator._get_coverage_trend()
        assert trend["trend"] == "insufficient_data"
        
        # Add history
        test_validator.coverage_history = [
            {"overall_coverage": 80.0, "timestamp": "2024-01-01T00:00:00"},
            {"overall_coverage": 85.0, "timestamp": "2024-01-02T00:00:00"}
        ]
        
        trend = test_validator._get_coverage_trend()
        assert trend["trend"] == "improving"
        assert trend["current"] == 85.0
        assert trend["previous"] == 80.0
        assert trend["change"] == 5.0
        
        # Declining trend
        test_validator.coverage_history.append(
            {"overall_coverage": 82.0, "timestamp": "2024-01-03T00:00:00"}
        )
        trend = test_validator._get_coverage_trend()
        assert trend["trend"] == "declining"
        
    def test_generate_module_breakdown(self, test_validator):
        """Test module breakdown generation."""
        coverage_analysis = {
            "module_coverage": {
                "core/models.py": 95.0,
                "engine/risk_engine.py": 100.0,
                "ui/dashboard.py": 65.0,
                "utils/math.py": 100.0,
                "tilt/detector.py": 88.0
            }
        }
        
        breakdown = test_validator._generate_module_breakdown(coverage_analysis)
        
        assert "core" in breakdown
        assert "engine" in breakdown
        assert "ui" in breakdown
        assert "utils" in breakdown
        assert "tilt" in breakdown
        
        # Check sorting
        if breakdown["core"]:
            coverages = [m["coverage"] for m in breakdown["core"]]
            assert coverages == sorted(coverages, reverse=True)
            
        # Check status indicators
        for module in breakdown["ui"]:
            if module["coverage"] >= 80:
                assert module["status"] == "✓"
            elif module["coverage"] >= 60:
                assert module["status"] == "⚠"
            else:
                assert module["status"] == "✗"
                
    def test_calculate_overall_score(self, test_validator):
        """Test overall score calculation."""
        unit_results = {"total": 100, "pass_rate": 95.0}
        integration_results = {"total": 50, "pass_rate": 90.0}
        threshold_results = {
            "money_paths": {"average": 97.0},
            "risk_components": {"average": 88.0},
            "core_modules": {"average": 85.0}
        }
        coverage_analysis = {"overall_coverage": 85.0}
        
        score = test_validator._calculate_overall_score(
            unit_results,
            integration_results,
            threshold_results,
            coverage_analysis
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        
    def test_generate_summary(self, test_validator):
        """Test summary generation."""
        # Passing summary
        summary = test_validator._generate_summary(
            passed=True,
            score=95.0,
            threshold_results={}
        )
        assert "passed" in summary
        assert "95.0%" in summary
        
        # Failing summary
        threshold_results = {
            "money_paths": {"passed": False},
            "risk_components": {"passed": True}
        }
        summary = test_validator._generate_summary(
            passed=False,
            score=75.0,
            threshold_results=threshold_results
        )
        assert "failed" in summary
        assert "money_paths" in summary
        
    def test_generate_recommendations(self, test_validator):
        """Test recommendation generation."""
        unit_results = {"passed": False, "failed": 5}
        integration_results = {"passed": True, "failed": 0}
        coverage_analysis = {
            "overall_coverage": 75.0,
            "module_coverage": {
                "engine/risk_engine.py": 85.0,
                "utils/math.py": 50.0
            }
        }
        threshold_results = {
            "money_paths": {
                "passed": False,
                "threshold": 100,
                "average": 85.0,
                "total_paths": 5,
                "paths_meeting_threshold": 2,
                "details": ["path1: 85.0% ✗", "path2: 100.0% ✓"]
            }
        }
        
        recommendations = test_validator._generate_recommendations(
            unit_results,
            integration_results,
            coverage_analysis,
            threshold_results
        )
        
        assert len(recommendations) > 0
        assert any("Fix 5 failing unit tests" in r for r in recommendations)
        assert any("money_paths" in r for r in recommendations)
        assert any("75.0%" in r for r in recommendations)
        
    @pytest.mark.asyncio
    async def test_analyze_coverage_xml(self, test_validator):
        """Test XML coverage analysis."""
        xml_content = """<?xml version="1.0" ?>
        <coverage>
            <packages>
                <package name="genesis.core" line-rate="0.95" branch-rate="0.90" complexity="10">
                </package>
                <package name="genesis.engine" line-rate="1.0" branch-rate="0.95" complexity="15">
                </package>
            </packages>
        </coverage>"""
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=xml_content)):
                # Mock ET.parse to work with the mock file
                with patch("xml.etree.ElementTree.parse") as mock_parse:
                    mock_tree = MagicMock()
                    mock_root = MagicMock()
                    mock_tree.getroot.return_value = mock_root
                    
                    # Mock packages
                    mock_package1 = MagicMock()
                    mock_package1.get.side_effect = lambda x, default=None: {
                        "name": "genesis.core",
                        "line-rate": "0.95",
                        "branch-rate": "0.90",
                        "complexity": "10"
                    }.get(x, default)
                    
                    mock_package2 = MagicMock()
                    mock_package2.get.side_effect = lambda x, default=None: {
                        "name": "genesis.engine",
                        "line-rate": "1.0",
                        "branch-rate": "0.95",
                        "complexity": "15"
                    }.get(x, default)
                    
                    mock_root.findall.return_value = [mock_package1, mock_package2]
                    mock_parse.return_value = mock_tree
                    
                    result = await test_validator._analyze_coverage_xml()
        
        assert "genesis.core" in result
        assert "genesis.engine" in result
        assert result["genesis.core"]["line_coverage"] == 95
        assert result["genesis.engine"]["line_coverage"] == 100