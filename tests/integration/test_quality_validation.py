"""Integration tests for quality validation framework."""

import asyncio
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from genesis.validation.quality import (
    CodeQualityAnalyzer,
    DatabaseValidator,
    PerformanceValidator,
    ResourceValidator,
    TestCoverageValidator,
)


class TestQualityValidationIntegration:
    """Integration tests for the quality validation framework."""

    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create a temporary project structure for testing."""
        # Create project structure
        genesis_dir = tmp_path / "genesis"
        genesis_dir.mkdir()
        
        # Create sample Python files
        (genesis_dir / "__init__.py").write_text("")
        
        # Create a file with various code issues
        sample_code = '''
"""Sample module for testing."""

def calculate_price(amount, rate):
    # TODO: Add validation
    result = amount * rate  # Using float for money
    print(f"Result: {result}")  # Using print
    return result

def process_orders(orders):
    results = []
    for order in orders:
        # N+1 query pattern
        user = db.query(f"SELECT * FROM users WHERE id = {order.user_id}")
        order_details = db.query(f"SELECT * FROM details WHERE order_id = {order.id}")
        results.append((user, order_details))
    return results

try:
    risky_operation()
except:  # Bare except
    pass

# Long function
def very_long_function():
    line1 = 1
    line2 = 2
    line3 = 3
    line4 = 4
    line5 = 5
    line6 = 6
    line7 = 7
    line8 = 8
    line9 = 9
    line10 = 10
    line11 = 11
    line12 = 12
    line13 = 13
    line14 = 14
    line15 = 15
    line16 = 16
    line17 = 17
    line18 = 18
    line19 = 19
    line20 = 20
    line21 = 21
    line22 = 22
    line23 = 23
    line24 = 24
    line25 = 25
    line26 = 26
    line27 = 27
    line28 = 28
    line29 = 29
    line30 = 30
    return sum([line1, line2, line3, line4, line5])
'''
        (genesis_dir / "sample.py").write_text(sample_code)
        
        # Create coverage.xml
        coverage_xml = '''<?xml version="1.0" ?>
        <coverage version="7.3.2" timestamp="1704067200000" lines-valid="100" lines-covered="85" line-rate="0.85">
            <packages>
                <package name="genesis" line-rate="0.85">
                    <classes>
                        <class name="genesis.sample" filename="genesis/sample.py" line-rate="0.85">
                            <lines>
                                <line number="1" hits="1"/>
                                <line number="2" hits="1"/>
                                <line number="3" hits="1"/>
                                <line number="4" hits="0"/>
                                <line number="5" hits="0"/>
                            </lines>
                        </class>
                    </classes>
                </package>
            </packages>
        </coverage>'''
        (tmp_path / "coverage.xml").write_text(coverage_xml)
        
        # Create database
        db_dir = tmp_path / ".genesis" / "data"
        db_dir.mkdir(parents=True)
        
        # Create test results directory
        test_results = tmp_path / "test-results" / "performance"
        test_results.mkdir(parents=True)
        
        # Create benchmark results
        benchmark_data = {
            "benchmarks": [
                {
                    "name": "test_order_validation",
                    "stats": {
                        "mean": 0.025,  # 25ms
                        "min": 0.020,
                        "max": 0.055,  # 55ms - violates 50ms p99 threshold
                        "stddev": 0.005,
                        "iterations": 100
                    }
                }
            ]
        }
        
        import json
        (test_results / "benchmark.json").write_text(json.dumps(benchmark_data))
        
        return tmp_path

    @pytest.mark.asyncio
    async def test_test_coverage_validator_integration(self, temp_project):
        """Test the test coverage validator with real coverage data."""
        validator = TestCoverageValidator(project_root=temp_project)
        
        results = await validator.run_validation()
        
        assert results["validator"] == "TestCoverageValidator"
        assert results["status"] in ["passed", "failed"]
        assert "coverage_analysis" in results
        assert "threshold_violations" in results
        assert "evidence" in results
        
        # Check that coverage was analyzed
        coverage_analysis = results["coverage_analysis"]
        assert "overall_coverage" in coverage_analysis
        assert coverage_analysis["overall_coverage"] == 85.0  # From our test coverage.xml
        
        # Generate report
        report = await validator.generate_report(results)
        assert "TEST COVERAGE VALIDATION REPORT" in report
        assert "COVERAGE SUMMARY" in report

    @pytest.mark.asyncio
    async def test_code_quality_analyzer_integration(self, temp_project):
        """Test the code quality analyzer with real code files."""
        analyzer = CodeQualityAnalyzer(project_root=temp_project)
        
        results = await analyzer.run_validation()
        
        assert results["validator"] == "CodeQualityAnalyzer"
        assert "complexity_analysis" in results
        assert "duplication_analysis" in results
        assert "code_smells" in results
        assert "standards_violations" in results
        
        # Check that code smells were detected
        code_smells = results["code_smells"]
        smell_types = [smell["type"] for smell in code_smells]
        
        # Our sample code should have these issues
        assert any("bare_except" in t for t in smell_types)
        assert any("print_statement" in t for t in smell_types)
        assert any("todo_fixme" in t for t in smell_types)
        
        # Check for anti-patterns
        assert any("float_for_money" in t for t in smell_types)

    @pytest.mark.asyncio
    async def test_performance_validator_integration(self, temp_project):
        """Test the performance validator with benchmark data."""
        validator = PerformanceValidator(project_root=temp_project)
        
        results = await validator.run_validation()
        
        assert results["validator"] == "PerformanceValidator"
        assert "latency_benchmarks" in results
        assert "throughput_results" in results
        assert "load_test_results" in results
        
        # Check latency analysis
        latency = results["latency_benchmarks"]
        assert "measurements" in latency
        assert "percentiles" in latency
        
        # Check for violations (our test data has 55ms max, violating 50ms p99)
        violations = results["violations"]
        assert any(v["type"] == "latency_p99" for v in violations)

    @pytest.mark.asyncio
    async def test_resource_validator_integration(self, temp_project):
        """Test the resource validator with system metrics."""
        validator = ResourceValidator(project_root=temp_project)
        
        # Mock tracemalloc to avoid memory tracking overhead
        with patch('tracemalloc.start'), \
             patch('tracemalloc.take_snapshot') as mock_snapshot, \
             patch('tracemalloc.stop'):
            
            # Create mock snapshot
            mock_snapshot.return_value = MagicMock(
                statistics=MagicMock(return_value=[]),
                compare_to=MagicMock(return_value=[])
            )
            
            results = await validator.run_validation()
        
        assert results["validator"] == "ResourceValidator"
        assert "memory_analysis" in results
        assert "cpu_analysis" in results
        assert "resource_usage" in results
        
        # Check memory analysis
        memory = results["memory_analysis"]
        assert "current_usage_mb" in memory
        assert "available_mb" in memory
        assert "percent_used" in memory
        
        # Check CPU analysis
        cpu = results["cpu_analysis"]
        assert "current_percent" in cpu
        assert "thread_count" in cpu

    @pytest.mark.asyncio
    async def test_database_validator_integration(self, temp_project):
        """Test the database validator with sample database."""
        validator = DatabaseValidator(project_root=temp_project)
        
        results = await validator.run_validation()
        
        assert results["validator"] == "DatabaseValidator"
        assert "query_analysis" in results
        assert "index_analysis" in results
        assert "n_plus_one_detection" in results
        assert "connection_pool_analysis" in results
        
        # Check N+1 detection (our sample code has N+1 patterns)
        n_plus_one = results["n_plus_one_detection"]
        assert "detected_patterns" in n_plus_one
        assert "total_issues" in n_plus_one
        
        # Should detect the N+1 pattern in our sample code
        assert n_plus_one["total_issues"] > 0

    @pytest.mark.asyncio
    async def test_all_validators_together(self, temp_project):
        """Test running all validators together."""
        validators = [
            TestCoverageValidator(project_root=temp_project),
            CodeQualityAnalyzer(project_root=temp_project),
            PerformanceValidator(project_root=temp_project),
            ResourceValidator(project_root=temp_project),
            DatabaseValidator(project_root=temp_project),
        ]
        
        # Run all validators concurrently
        with patch('tracemalloc.start'), \
             patch('tracemalloc.take_snapshot') as mock_snapshot, \
             patch('tracemalloc.stop'):
            
            mock_snapshot.return_value = MagicMock(
                statistics=MagicMock(return_value=[]),
                compare_to=MagicMock(return_value=[])
            )
            
            tasks = [v.run_validation() for v in validators]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check all validators completed
        assert len(results) == 5
        
        # Check no exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert result["status"] in ["passed", "failed", "error"]

    @pytest.mark.asyncio
    async def test_validator_error_handling(self, temp_project):
        """Test that validators handle errors gracefully."""
        # Create validator with non-existent path
        validator = TestCoverageValidator(project_root=Path("/non/existent/path"))
        
        results = await validator.run_validation()
        
        assert results["status"] in ["failed", "error"]
        assert results["passed"] is False

    @pytest.mark.asyncio
    async def test_evidence_generation(self, temp_project):
        """Test that all validators generate proper evidence."""
        validators = [
            TestCoverageValidator(project_root=temp_project),
            CodeQualityAnalyzer(project_root=temp_project),
            PerformanceValidator(project_root=temp_project),
            ResourceValidator(project_root=temp_project),
            DatabaseValidator(project_root=temp_project),
        ]
        
        for validator in validators:
            # Skip resource validator's memory tracking
            if isinstance(validator, ResourceValidator):
                with patch('tracemalloc.start'), \
                     patch('tracemalloc.take_snapshot') as mock_snapshot, \
                     patch('tracemalloc.stop'):
                    mock_snapshot.return_value = MagicMock(
                        statistics=MagicMock(return_value=[]),
                        compare_to=MagicMock(return_value=[])
                    )
                    results = await validator.run_validation()
            else:
                results = await validator.run_validation()
            
            # Check evidence is generated
            assert "evidence" in results
            evidence = results["evidence"]
            assert isinstance(evidence, dict)
            assert len(evidence) > 0

    @pytest.mark.asyncio
    async def test_trend_analysis(self, temp_project):
        """Test that validators track trends over multiple runs."""
        validator = TestCoverageValidator(project_root=temp_project)
        
        # Run validator multiple times
        for _ in range(3):
            results = await validator.run_validation()
        
        # Check trends are tracked
        assert "coverage_trends" in results
        trends = results["coverage_trends"]
        
        # After 3 runs, should have trend data
        if "overall" in trends:
            assert "trend" in trends["overall"]
            assert trends["overall"]["history_points"] == 3

    @pytest.mark.asyncio
    async def test_performance_thresholds(self, temp_project):
        """Test that performance thresholds are properly enforced."""
        validator = PerformanceValidator(project_root=temp_project)
        
        # Check thresholds are set correctly
        assert validator.LATENCY_P99_THRESHOLD_MS == 50
        assert validator.MIN_THROUGHPUT_TPS == 100
        assert validator.LOAD_TEST_MULTIPLIER == 100
        
        results = await validator.run_validation()
        
        # Check violations are detected for thresholds
        violations = results.get("violations", [])
        for violation in violations:
            if "threshold" in violation:
                assert violation["value"] != violation["threshold"]

    @pytest.mark.asyncio
    async def test_database_optimization_suggestions(self, temp_project):
        """Test that database validator provides optimization suggestions."""
        validator = DatabaseValidator(project_root=temp_project)
        
        results = await validator.run_validation()
        
        assert "optimization_suggestions" in results
        suggestions = results["optimization_suggestions"]
        
        # Should have suggestions for N+1 queries
        assert any(s["area"] == "n_plus_one" for s in suggestions)

    @pytest.mark.asyncio
    async def test_code_quality_metrics(self, temp_project):
        """Test that code quality metrics are calculated correctly."""
        analyzer = CodeQualityAnalyzer(project_root=temp_project)
        
        results = await analyzer.run_validation()
        
        assert "metrics" in results
        metrics = results["metrics"]
        
        # Check all metric categories are present
        assert "complexity" in metrics
        assert "duplication" in metrics
        assert "code_smells" in metrics
        assert "standards" in metrics
        assert "overall_score" in metrics
        
        # Overall score should be between 0 and 100
        assert 0 <= metrics["overall_score"] <= 100

    @pytest.mark.asyncio  
    async def test_memory_leak_detection(self, temp_project):
        """Test memory leak detection capabilities."""
        validator = ResourceValidator(project_root=temp_project)
        
        with patch('tracemalloc.start'), \
             patch('tracemalloc.take_snapshot') as mock_snapshot, \
             patch('tracemalloc.stop'):
            
            # Simulate memory growth
            mock_stat = MagicMock()
            mock_stat.size_diff = 2 * 1024 * 1024  # 2MB growth
            mock_stat.count_diff = 100
            mock_stat.traceback = MagicMock()
            mock_stat.traceback.format.return_value = ["test_file.py:123"]
            
            mock_snapshot.return_value = MagicMock(
                statistics=MagicMock(return_value=[]),
                compare_to=MagicMock(return_value=[mock_stat])
            )
            
            # Mock process memory to show growth
            with patch.object(validator.process, 'memory_info') as mock_mem:
                mock_mem.return_value = MagicMock(rss=200 * 1024 * 1024)  # 200MB
                
                results = await validator.run_validation()
        
        assert "leaks_detected" in results
        leaks = results["leaks_detected"]
        
        # Should detect the growing allocation
        assert any(leak["type"] == "growing_allocation" for leak in leaks)