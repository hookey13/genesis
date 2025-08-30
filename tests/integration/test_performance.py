"""Integration tests for performance validator."""

import asyncio
import time
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock

import pytest

from genesis.validation.performance_validator import PerformanceValidator


@pytest.fixture
def performance_validator():
    """Create a performance validator instance."""
    return PerformanceValidator()


class TestPerformanceValidatorIntegration:
    """Integration tests for performance validation."""
    
    @pytest.mark.asyncio
    async def test_full_performance_validation(self, performance_validator):
        """Test complete performance validation workflow."""
        result = await performance_validator.validate()
        
        # Check required fields
        assert "passed" in result
        assert "details" in result
        assert "benchmarks" in result
        assert "recommendations" in result
        
        # Check metrics are calculated
        details = result["details"]
        assert "p50_latency_ms" in details
        assert "p95_latency_ms" in details
        assert "p99_latency_ms" in details
        assert "ws_processing_ms" in details
        assert "db_query_ms" in details
        assert "stress_test_passed" in details
    
    @pytest.mark.asyncio
    async def test_order_latency_benchmarking(self, performance_validator):
        """Test order execution latency benchmarking."""
        result = await performance_validator._test_order_latency()
        
        assert result["samples"] == 100
        assert result["p50"] > 0
        assert result["p95"] > result["p50"]
        assert result["p99"] > result["p95"]
        assert result["min"] <= result["mean"] <= result["max"]
    
    @pytest.mark.asyncio
    async def test_websocket_performance(self, performance_validator):
        """Test WebSocket message processing performance."""
        result = await performance_validator._test_websocket_performance()
        
        assert result["samples"] == 100
        assert result["avg_processing_ms"] > 0
        assert result["max_processing_ms"] >= result["avg_processing_ms"]
        assert result["messages_per_second"] > 0
    
    @pytest.mark.asyncio
    async def test_database_performance(self, performance_validator):
        """Test database query performance."""
        result = await performance_validator._test_database_performance()
        
        assert result["samples"] == 50
        assert result["avg_query_ms"] > 0
        assert result["max_query_ms"] >= result["avg_query_ms"]
        assert result["queries_per_second"] > 0
    
    @pytest.mark.asyncio
    async def test_stress_load_handling(self, performance_validator):
        """Test system behavior under stress load."""
        # Reduce stress multiplier for testing
        performance_validator.stress_multiplier = 10
        
        result = await performance_validator._test_stress_load()
        
        assert "system_stable" in result
        assert "messages_processed" in result
        assert "errors" in result
        assert "max_throughput" in result
        assert "cpu_usage" in result
        assert "memory_usage" in result
        
        # Check error rate is reasonable
        if result["messages_processed"] > 0:
            error_rate = result["errors"] / result["messages_processed"]
            assert error_rate < 0.1  # Less than 10% errors
    
    @pytest.mark.asyncio
    async def test_latency_thresholds(self, performance_validator):
        """Test that latency thresholds are properly enforced."""
        # Mock order latency to exceed thresholds
        with patch.object(performance_validator, "_test_order_latency") as mock_latency:
            mock_latency.return_value = {
                "samples": 100,
                "p50": 15,  # Exceeds 10ms target
                "p95": 40,  # Exceeds 30ms target
                "p99": 60,  # Exceeds 50ms target
                "min": 5,
                "max": 100,
                "mean": 25
            }
            
            # Mock other tests to pass
            with patch.object(performance_validator, "_test_websocket_performance") as mock_ws:
                mock_ws.return_value = {
                    "samples": 100,
                    "avg_processing_ms": 3,
                    "max_processing_ms": 5,
                    "messages_per_second": 300
                }
                
                with patch.object(performance_validator, "_test_database_performance") as mock_db:
                    mock_db.return_value = {
                        "samples": 50,
                        "avg_query_ms": 5,
                        "max_query_ms": 10,
                        "queries_per_second": 200
                    }
                    
                    with patch.object(performance_validator, "_test_stress_load") as mock_stress:
                        mock_stress.return_value = {
                            "system_stable": True,
                            "messages_processed": 1000,
                            "errors": 5,
                            "max_throughput": 100,
                            "cpu_usage": 50,
                            "memory_usage": 512
                        }
                        
                        result = await performance_validator.validate()
                        
                        assert result["passed"] is False
                        assert "Optimize order execution" in str(result["recommendations"])
    
    @pytest.mark.asyncio
    async def test_stress_worker_resilience(self, performance_validator):
        """Test individual stress worker resilience."""
        result = await performance_validator._stress_worker(1)
        
        assert result["messages"] > 0
        assert result["max_latency"] >= 0
    
    @pytest.mark.asyncio
    async def test_performance_with_executor_import(self, performance_validator):
        """Test performance validation with executor module available."""
        # Mock the executor import
        with patch("genesis.validation.performance_validator.MarketOrderExecutor"):
            result = await performance_validator._test_order_latency()
            
            assert result["samples"] == 100
            assert "error" not in result
    
    @pytest.mark.asyncio
    async def test_performance_with_database_import(self, performance_validator):
        """Test performance validation with database module available."""
        # Mock the repository import
        with patch("genesis.validation.performance_validator.SQLiteRepository"):
            result = await performance_validator._test_database_performance()
            
            assert result["samples"] == 50
            assert "error" not in result
    
    @pytest.mark.asyncio
    async def test_recommendations_generation(self, performance_validator):
        """Test generation of performance recommendations."""
        order_latency = {
            "p50": 15,
            "p95": 40,
            "p99": 60
        }
        
        ws_performance = {
            "avg_processing_ms": 10
        }
        
        db_performance = {
            "avg_query_ms": 15
        }
        
        stress_results = {
            "system_stable": False,
            "error_rate": 0.02,
            "memory_growth_mb": 150,
            "max_latency_ms": 1500,
            "max_throughput": 500
        }
        
        recommendations = performance_validator._generate_recommendations(
            order_latency,
            ws_performance,
            db_performance,
            stress_results
        )
        
        assert len(recommendations) > 0
        assert any("Optimize order execution" in r for r in recommendations)
        assert any("WebSocket processing" in r for r in recommendations)
        assert any("database queries" in r for r in recommendations)
        assert any("stability issues" in r for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_psutil_not_available(self, performance_validator):
        """Test handling when psutil is not available."""
        with patch("genesis.validation.performance_validator.psutil", side_effect=ImportError()):
            result = await performance_validator._test_stress_load()
            
            assert result["system_stable"] is False
            assert "error" in result
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(15)
    async def test_stress_test_timeout(self, performance_validator):
        """Test stress test completes within timeout."""
        # Reduce multiplier for faster test
        performance_validator.stress_multiplier = 5
        
        start_time = time.time()
        result = await performance_validator._test_stress_load()
        duration = time.time() - start_time
        
        assert duration < 15  # Should complete within 15 seconds
        assert "messages_processed" in result


@pytest.mark.asyncio
async def test_performance_validator_with_real_load():
    """Test performance validator with simulated real load."""
    validator = PerformanceValidator()
    
    # Simulate some load
    async def generate_load():
        tasks = []
        for _ in range(10):
            tasks.append(asyncio.create_task(asyncio.sleep(0.001)))
        await asyncio.gather(*tasks)
    
    # Run load generation in background
    load_task = asyncio.create_task(generate_load())
    
    # Run validation
    result = await validator.validate()
    
    # Wait for load to complete
    await load_task
    
    assert "passed" in result
    assert "details" in result
    assert result["details"]["p99_latency_ms"] is not None
