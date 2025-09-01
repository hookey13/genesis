"""Unit tests for quality validators."""

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from genesis.validation.quality.test_validator import TestCoverageValidator


class TestTestCoverageValidator:
    """Test the TestCoverageValidator class."""

    @pytest.fixture
    def validator(self, tmp_path):
        """Create a test validator instance."""
        return TestCoverageValidator(project_root=tmp_path)

    @pytest.fixture
    def sample_coverage_xml(self, tmp_path):
        """Create a sample coverage.xml file."""
        coverage_xml = """<?xml version="1.0" ?>
        <coverage version="7.3.2" timestamp="1704067200000" lines-valid="1000" lines-covered="850" line-rate="0.85" branches-covered="0" branches-valid="0" branch-rate="0" complexity="0">
            <packages>
                <package name="genesis" line-rate="0.85" branch-rate="0">
                    <classes>
                        <class name="genesis.engine.risk_engine" filename="genesis/engine/risk_engine.py" line-rate="1.0" branch-rate="0">
                            <lines>
                                <line number="1" hits="1"/>
                                <line number="2" hits="1"/>
                                <line number="3" hits="1"/>
                                <line number="4" hits="1"/>
                                <line number="5" hits="1"/>
                            </lines>
                        </class>
                        <class name="genesis.utils.math" filename="genesis/utils/math.py" line-rate="0.95" branch-rate="0">
                            <lines>
                                <line number="1" hits="1"/>
                                <line number="2" hits="1"/>
                                <line number="3" hits="1"/>
                                <line number="4" hits="1"/>
                                <line number="5" hits="0"/>
                            </lines>
                        </class>
                        <class name="genesis.ui.dashboard" filename="genesis/ui/dashboard.py" line-rate="0.70" branch-rate="0">
                            <lines>
                                <line number="1" hits="1"/>
                                <line number="2" hits="1"/>
                                <line number="3" hits="1"/>
                                <line number="4" hits="0"/>
                                <line number="5" hits="0"/>
                            </lines>
                        </class>
                        <class name="genesis.core.models" filename="genesis/core/models.py" line-rate="0.80" branch-rate="0">
                            <lines>
                                <line number="1" hits="1"/>
                                <line number="2" hits="1"/>
                                <line number="3" hits="1"/>
                                <line number="4" hits="1"/>
                                <line number="5" hits="0"/>
                            </lines>
                        </class>
                    </classes>
                </package>
            </packages>
        </coverage>"""
        
        coverage_file = tmp_path / "coverage.xml"
        coverage_file.write_text(coverage_xml)
        return coverage_file

    @pytest.mark.asyncio
    async def test_run_validation_success(self, validator, sample_coverage_xml):
        """Test successful validation run."""
        results = await validator.run_validation()
        
        assert results["validator"] == "TestCoverageValidator"
        assert results["status"] in ["passed", "failed"]
        assert "coverage_analysis" in results
        assert "threshold_violations" in results
        assert "evidence" in results
        assert "metadata" in results

    @pytest.mark.asyncio
    async def test_run_validation_no_coverage_file(self, validator):
        """Test validation when coverage file doesn't exist."""
        results = await validator.run_validation()
        
        assert results["status"] == "failed"
        assert results["error"] == "Coverage report not found or invalid"
        assert results["passed"] is False

    @pytest.mark.asyncio
    async def test_parse_coverage_xml_valid(self, validator, sample_coverage_xml):
        """Test parsing valid coverage XML."""
        coverage_data = await validator._parse_coverage_xml()
        
        assert coverage_data is not None
        assert "overall" in coverage_data
        assert "files" in coverage_data
        assert len(coverage_data["files"]) == 4
        assert coverage_data["overall"]["line-rate"] == 85.0

    @pytest.mark.asyncio
    async def test_parse_coverage_xml_invalid(self, validator, tmp_path):
        """Test parsing invalid coverage XML."""
        invalid_xml = tmp_path / "coverage.xml"
        invalid_xml.write_text("invalid xml content")
        
        coverage_data = await validator._parse_coverage_xml()
        assert coverage_data is None

    def test_analyze_path_coverage(self, validator):
        """Test path coverage analysis."""
        coverage_data = {
            "overall": {"line-rate": 85.0},
            "files": {
                "genesis/engine/risk_engine.py": {
                    "line_rate": 100.0,
                    "total_lines": 100,
                    "covered_lines": 100,
                    "uncovered_lines": 0,
                },
                "genesis/utils/math.py": {
                    "line_rate": 95.0,
                    "total_lines": 100,
                    "covered_lines": 95,
                    "uncovered_lines": 5,
                },
                "genesis/ui/dashboard.py": {
                    "line_rate": 70.0,
                    "total_lines": 100,
                    "covered_lines": 70,
                    "uncovered_lines": 30,
                },
                "genesis/core/models.py": {
                    "line_rate": 80.0,
                    "total_lines": 100,
                    "covered_lines": 80,
                    "uncovered_lines": 20,
                },
            }
        }
        
        analysis = validator._analyze_path_coverage(coverage_data)
        
        assert "money_paths" in analysis
        assert "risk_paths" in analysis
        assert "core_paths" in analysis
        assert "ui_paths" in analysis
        
        # Check money paths categorization
        money_files = [f["path"] for f in analysis["money_paths"]["files"]]
        assert "genesis/engine/risk_engine.py" in money_files
        assert "genesis/utils/math.py" in money_files
        
        # Check UI paths categorization
        ui_files = [f["path"] for f in analysis["ui_paths"]["files"]]
        assert "genesis/ui/dashboard.py" in ui_files

    def test_check_thresholds_with_violations(self, validator):
        """Test threshold checking with violations."""
        path_analysis = {
            "money_paths": {
                "coverage": 95.0,  # Below 100% threshold
                "threshold": 100.0,
                "files": [
                    {"path": "genesis/utils/math.py", "coverage": 95.0, "lines": 100, "covered": 95}
                ]
            },
            "risk_paths": {
                "coverage": 85.0,  # Below 90% threshold
                "threshold": 90.0,
                "files": [
                    {"path": "genesis/tilt/detector.py", "coverage": 85.0, "lines": 100, "covered": 85}
                ]
            },
            "core_paths": {
                "coverage": 90.0,  # Above 85% threshold
                "threshold": 85.0,
                "files": []
            },
            "ui_paths": {
                "coverage": 75.0,  # Above 70% threshold
                "threshold": 70.0,
                "files": []
            }
        }
        
        violations = validator._check_thresholds(path_analysis)
        
        assert len(violations) == 2
        
        # Check money path violation
        money_violation = next((v for v in violations if v["category"] == "money_paths"), None)
        assert money_violation is not None
        assert money_violation["severity"] == "critical"
        assert money_violation["gap"] == 5.0
        
        # Check risk path violation
        risk_violation = next((v for v in violations if v["category"] == "risk_paths"), None)
        assert risk_violation is not None
        assert risk_violation["severity"] == "high"
        assert risk_violation["gap"] == 5.0

    def test_check_thresholds_no_violations(self, validator):
        """Test threshold checking with no violations."""
        path_analysis = {
            "money_paths": {
                "coverage": 100.0,
                "threshold": 100.0,
                "files": []
            },
            "risk_paths": {
                "coverage": 95.0,
                "threshold": 90.0,
                "files": []
            },
            "core_paths": {
                "coverage": 90.0,
                "threshold": 85.0,
                "files": []
            },
            "ui_paths": {
                "coverage": 75.0,
                "threshold": 70.0,
                "files": []
            }
        }
        
        violations = validator._check_thresholds(path_analysis)
        assert len(violations) == 0

    def test_generate_evidence_report(self, validator):
        """Test evidence report generation."""
        coverage_data = {"overall": {"line-rate": 85.0}}
        path_analysis = {
            "overall_coverage": 85.0,
            "money_paths": {
                "coverage": 95.0,
                "threshold": 100.0,
                "files": [
                    {"path": "genesis/utils/math.py", "coverage": 95.0, "lines": 100, "covered": 95}
                ]
            },
            "risk_paths": {
                "coverage": 85.0,
                "threshold": 90.0,
                "files": [
                    {"path": "genesis/tilt/detector.py", "coverage": 85.0, "lines": 100, "covered": 85}
                ]
            },
            "core_paths": {"coverage": 90.0, "threshold": 85.0, "files": []},
            "ui_paths": {"coverage": 75.0, "threshold": 70.0, "files": []},
        }
        
        evidence = validator._generate_evidence_report(coverage_data, path_analysis)
        
        assert "summary" in evidence
        assert "critical_files" in evidence
        assert "improvement_opportunities" in evidence
        assert "coverage_gaps" in evidence
        
        # Check summary
        assert evidence["summary"]["overall_coverage"] == 85.0
        assert evidence["summary"]["money_path_coverage"] == 95.0
        
        # Check critical files
        assert len(evidence["critical_files"]) == 2
        math_file = next((f for f in evidence["critical_files"] if "math.py" in f["file"]), None)
        assert math_file is not None
        assert math_file["gap"] == 5.0

    def test_update_coverage_trends(self, validator):
        """Test coverage trend tracking."""
        # Add historical data
        validator.coverage_history = [
            {"timestamp": "2024-01-01T00:00:00", "overall": 80.0, "money_paths": 90.0, "risk_paths": 85.0, "core_paths": 85.0, "ui_paths": 70.0},
            {"timestamp": "2024-01-02T00:00:00", "overall": 82.0, "money_paths": 92.0, "risk_paths": 86.0, "core_paths": 86.0, "ui_paths": 71.0},
        ]
        
        path_analysis = {
            "overall_coverage": 85.0,
            "money_paths": {"coverage": 95.0},
            "risk_paths": {"coverage": 88.0},
            "core_paths": {"coverage": 87.0},
            "ui_paths": {"coverage": 72.0},
        }
        
        trends = validator._update_coverage_trends(path_analysis)
        
        assert "overall" in trends
        assert "money_paths" in trends
        
        # Check overall trend
        overall_trend = trends["overall"]
        assert overall_trend["current"] == 85.0
        assert overall_trend["previous"] == 82.0
        assert overall_trend["change"] == 3.0
        assert overall_trend["trend"] == "improving"
        
        # Check history is updated
        assert len(validator.coverage_history) == 3

    def test_update_coverage_trends_baseline(self, validator):
        """Test coverage trends with no history."""
        path_analysis = {
            "overall_coverage": 85.0,
            "money_paths": {"coverage": 95.0},
            "risk_paths": {"coverage": 88.0},
            "core_paths": {"coverage": 87.0},
            "ui_paths": {"coverage": 72.0},
        }
        
        trends = validator._update_coverage_trends(path_analysis)
        
        # All should show baseline trend
        for category in trends.values():
            assert category["trend"] == "baseline"
            assert category["history_points"] == 1

    @pytest.mark.asyncio
    async def test_generate_report(self, validator):
        """Test report generation."""
        validation_results = {
            "timestamp": "2024-01-01T00:00:00",
            "status": "failed",
            "coverage_analysis": {
                "money_paths": {"coverage": 95.0, "threshold": 100.0},
                "risk_paths": {"coverage": 88.0, "threshold": 90.0},
                "core_paths": {"coverage": 87.0, "threshold": 85.0},
                "ui_paths": {"coverage": 72.0, "threshold": 70.0},
            },
            "threshold_violations": [
                {
                    "category": "money_paths",
                    "coverage": 95.0,
                    "threshold": 100.0,
                    "gap": 5.0,
                    "severity": "critical",
                    "files_below_threshold": [
                        {"path": "genesis/utils/math.py", "coverage": 95.0}
                    ]
                }
            ],
            "evidence": {
                "critical_files": [
                    {
                        "file": "genesis/utils/math.py",
                        "current_coverage": 95.0,
                        "gap": 5.0,
                        "uncovered_lines": 5,
                    }
                ]
            },
            "coverage_trends": {
                "overall": {"current": 85.0, "change": 3.0, "trend": "improving"}
            }
        }
        
        report = await validator.generate_report(validation_results)
        
        assert "TEST COVERAGE VALIDATION REPORT" in report
        assert "Status: FAILED" in report
        assert "THRESHOLD VIOLATIONS" in report
        assert "genesis/utils/math.py" in report
        assert "CRITICAL FILES REQUIRING ATTENTION" in report
        assert "COVERAGE TRENDS" in report

    def test_money_paths_configuration(self, validator):
        """Test that money paths are properly configured."""
        assert validator.MONEY_PATH_THRESHOLD == 100.0
        assert "genesis/engine/risk_engine.py" in validator.MONEY_PATHS
        assert "genesis/utils/math.py" in validator.MONEY_PATHS
        assert "genesis/exchange/gateway.py" in validator.MONEY_PATHS

    def test_threshold_values(self, validator):
        """Test that threshold values are set correctly."""
        assert validator.MONEY_PATH_THRESHOLD == 100.0
        assert validator.RISK_PATH_THRESHOLD == 90.0
        assert validator.CORE_PATH_THRESHOLD == 85.0
        assert validator.UI_PATH_THRESHOLD == 70.0
        assert validator.DEFAULT_THRESHOLD == 80.0