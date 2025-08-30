"""Integration tests for stability testing framework."""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from genesis.validation.stability_tester import StabilityTester


@pytest.fixture
def stability_tester(tmp_path):
    """Create StabilityTester instance."""
    return StabilityTester(genesis_root=tmp_path)


@pytest.fixture
def mock_stability_data():
    """Create mock stability test data."""
    return {
        "start_time": (datetime.utcnow() - timedelta(hours=48)).isoformat(),
        "end_time": datetime.utcnow().isoformat(),
        "duration_hours": 48.5,
        "memory_samples": [
            {"timestamp": datetime.utcnow().isoformat(), "memory_mb": 512 + i * 0.5}
            for i in range(100)
        ],
        "cpu_samples": [
            {"timestamp": datetime.utcnow().isoformat(), "cpu_percent": 30 + (i % 20)}
            for i in range(100)
        ],
        "connection_samples": [
            {"timestamp": datetime.utcnow().isoformat(), "connected": i % 10 != 0}
            for i in range(100)
        ],
        "latency_samples": [45 + (i % 10) for i in range(100)],
        "error_count": 5,
        "total_events": 10000,
        "restart_count": 1,
        "reconnection_count": 3,
        "recovery_attempts": 2,
        "recovery_successes": 2,
        "transactions": 5000,
        "critical_failures": []
    }


class TestStabilityTester:
    """Test StabilityTester class."""
    
    def test_initialization(self, tmp_path):
        """Test StabilityTester initialization."""
        tester = StabilityTester(genesis_root=tmp_path)
        
        assert tester.genesis_root == tmp_path
        assert tester.required_hours == 48
        assert tester.max_memory_growth == 10
        assert tester.max_cpu_usage == 80
        assert tester.max_error_rate == 0.01
        assert tester.max_restart_count == 3
        assert tester.min_connection_uptime == 0.99
        assert tester.memory_leak_threshold == 5
        
    @pytest.mark.asyncio
    async def test_validate_with_passing_data(self, stability_tester, mock_stability_data):
        """Test validation with passing stability data."""
        # Create stability log file
        log_file = stability_tester.stability_log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, "w") as f:
            json.dump(mock_stability_data, f)
        
        result = await stability_tester.validate()
        
        assert "passed" in result
        assert "score" in result
        assert "checks" in result
        assert "recommendations" in result
        
        # Check individual checks
        assert "duration" in result["checks"]
        assert "memory" in result["checks"]
        assert "cpu" in result["checks"]
        assert "errors" in result["checks"]
        assert "stability" in result["checks"]
        assert "connections" in result["checks"]
        
    @pytest.mark.asyncio
    async def test_validate_with_memory_leak(self, stability_tester, mock_stability_data):
        """Test validation with memory leak detected."""
        # Modify data to simulate memory leak
        for i, sample in enumerate(mock_stability_data["memory_samples"]):
            sample["memory_mb"] = 512 + i * 10  # Rapid memory growth
        
        log_file = stability_tester.stability_log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, "w") as f:
            json.dump(mock_stability_data, f)
        
        result = await stability_tester.validate()
        
        assert result["passed"] is False
        assert result["checks"]["memory"]["passed"] is False
        assert any("memory leak" in r.lower() for r in result["recommendations"])
        
    @pytest.mark.asyncio
    async def test_validate_with_high_error_rate(self, stability_tester, mock_stability_data):
        """Test validation with high error rate."""
        # Increase error rate above threshold
        mock_stability_data["error_count"] = 200
        mock_stability_data["total_events"] = 1000
        
        log_file = stability_tester.stability_log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, "w") as f:
            json.dump(mock_stability_data, f)
        
        result = await stability_tester.validate()
        
        assert result["passed"] is False
        assert result["checks"]["errors"]["passed"] is False
        assert any("error rate" in r.lower() for r in result["recommendations"])
        
    @pytest.mark.asyncio
    async def test_validate_with_insufficient_duration(self, stability_tester, mock_stability_data):
        """Test validation with insufficient test duration."""
        # Reduce duration below required
        mock_stability_data["duration_hours"] = 24
        
        log_file = stability_tester.stability_log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, "w") as f:
            json.dump(mock_stability_data, f)
        
        result = await stability_tester.validate()
        
        assert result["passed"] is False
        assert result["checks"]["duration"]["passed"] is False
        assert any("48 hours" in r for r in result["recommendations"])
        
    @pytest.mark.asyncio
    async def test_validate_with_critical_failures(self, stability_tester, mock_stability_data):
        """Test validation with critical failures."""
        mock_stability_data["critical_failures"] = [
            {"timestamp": datetime.utcnow().isoformat(), "error": "Critical error 1"},
            {"timestamp": datetime.utcnow().isoformat(), "error": "Critical error 2"}
        ]
        
        log_file = stability_tester.stability_log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, "w") as f:
            json.dump(mock_stability_data, f)
        
        result = await stability_tester.validate()
        
        assert result["passed"] is False
        assert result["checks"]["stability"]["passed"] is False
        assert any("critical failures" in r.lower() for r in result["recommendations"])
        
    @pytest.mark.asyncio
    async def test_validate_no_data(self, stability_tester):
        """Test validation with no stability data."""
        result = await stability_tester.validate()
        
        assert result["passed"] is False
        assert "note" in result["details"] or "error" in result
        
    @pytest.mark.asyncio
    async def test_check_paper_trading_stability(self, stability_tester):
        """Test checking paper trading logs for stability."""
        # Create mock trading log
        log_dir = stability_tester.genesis_root / ".genesis" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        trading_log = log_dir / "trading.log"
        
        # Create log entries spanning 48+ hours
        start_time = datetime.utcnow() - timedelta(hours=50)
        log_entries = []
        
        for i in range(1000):
            timestamp = start_time + timedelta(minutes=i * 3)
            log_entry = {
                "timestamp": timestamp.isoformat(),
                "level": "INFO" if i % 100 != 0 else "ERROR",
                "message": "Trading event" if i != 0 else "Starting Genesis"
            }
            log_entries.append(json.dumps(log_entry) + "\n")
        
        with open(trading_log, "w") as f:
            f.writelines(log_entries)
        
        result = await stability_tester._check_paper_trading_stability()
        
        assert "passed" in result
        assert "details" in result
        assert result["details"]["hours_stable"] > 48
        
    def test_calculate_stability_score(self, stability_tester):
        """Test stability score calculation."""
        analysis = {
            "hours_stable": 48,
            "memory_growth_percent": 5,
            "memory_leak_detected": False,
            "average_cpu_percent": 60,
            "peak_cpu_percent": 75,
            "error_rate": 0.005,
            "error_count": 50,
            "restart_count": 1,
            "critical_failures": [],
            "recovery_success_rate": 1.0,
            "connection_uptime": 0.995,
            "reconnection_count": 2
        }
        
        score = stability_tester._calculate_stability_score(analysis)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score > 80  # Should be a good score
        
    def test_calculate_stability_score_poor(self, stability_tester):
        """Test stability score calculation with poor metrics."""
        analysis = {
            "hours_stable": 24,  # Only half the required time
            "memory_growth_percent": 20,  # High memory growth
            "memory_leak_detected": True,
            "average_cpu_percent": 90,  # High CPU
            "peak_cpu_percent": 100,
            "error_rate": 0.05,  # 5% error rate
            "error_count": 500,
            "restart_count": 5,  # Many restarts
            "critical_failures": ["error1", "error2"],
            "recovery_success_rate": 0.5,
            "connection_uptime": 0.90,  # Poor uptime
            "reconnection_count": 20
        }
        
        score = stability_tester._calculate_stability_score(analysis)
        
        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert score < 50  # Should be a poor score
        
    def test_analyze_stability_data(self, stability_tester, mock_stability_data):
        """Test analyzing stability data."""
        analysis = stability_tester._analyze_stability_data(mock_stability_data)
        
        assert analysis["hours_stable"] == 48.5
        assert "memory_growth_percent" in analysis
        assert "memory_leak_detected" in analysis
        assert "average_cpu_percent" in analysis
        assert "peak_cpu_percent" in analysis
        assert analysis["error_rate"] == 0.0005  # 5/10000
        assert analysis["restart_count"] == 1
        assert analysis["recovery_success_rate"] == 1.0  # 2/2
        assert analysis["connection_uptime"] == 0.9  # 90/100 samples connected
        
    def test_generate_recommendations(self, stability_tester):
        """Test recommendation generation."""
        analysis = {
            "hours_stable": 24,
            "memory_growth_percent": 15,
            "memory_leak_detected": True,
            "average_cpu_percent": 85,
            "peak_cpu_percent": 95,
            "error_rate": 0.02,
            "error_count": 200,
            "restart_count": 5,
            "critical_failures": ["error1"],
            "recovery_success_rate": 0.8,
            "connection_uptime": 0.95,
            "reconnection_count": 10,
            "average_latency_ms": 150
        }
        
        recommendations = stability_tester._generate_recommendations(analysis)
        
        assert len(recommendations) > 0
        assert any("48 hours" in r for r in recommendations)
        assert any("memory leak" in r.lower() for r in recommendations)
        assert any("error rate" in r.lower() for r in recommendations)
        assert any("restarts" in r.lower() for r in recommendations)
        assert any("critical failures" in r.lower() for r in recommendations)
        assert any("latency" in r.lower() for r in recommendations)
        
    @pytest.mark.asyncio
    async def test_run_stability_test_short(self, stability_tester):
        """Test running a short stability test."""
        # Run a very short test (1 second)
        with patch("time.time") as mock_time:
            # Simulate time passing quickly
            mock_time.side_effect = [0, 0.5, 1, 61, 62, 3700]  # Simulate > 1 hour quickly
            
            test_data = await stability_tester.run_stability_test(duration_hours=0.001)
        
        assert "start_time" in test_data
        assert "end_time" in test_data
        assert "duration_hours" in test_data
        assert "memory_samples" in test_data
        assert "cpu_samples" in test_data
        assert "connection_samples" in test_data
        
        # Check that log file was created
        assert stability_tester.stability_log_file.exists()
        
        # Verify saved data
        with open(stability_tester.stability_log_file) as f:
            saved_data = json.load(f)
        
        assert saved_data["duration_hours"] > 0