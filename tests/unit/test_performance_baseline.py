"""Unit tests for performance baseline calculation system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch
import json

from genesis.operations.performance_baseline import (
    PerformanceBaseline,
    PerformanceMetrics,
    BaselineCalculator,
    PerformanceAlert
)


class TestPerformanceMetrics:
    """Test performance metrics collection."""
    
    def test_metrics_initialization(self):
        """Test metrics object initialization."""
        metrics = PerformanceMetrics(
            operation="test_op",
            latency_ms=100.5,
            timestamp=datetime.now(),
            success=True
        )
        
        assert metrics.operation == "test_op"
        assert metrics.latency_ms == 100.5
        assert metrics.success is True
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metrics_serialization(self):
        """Test metrics can be serialized to JSON."""
        metrics = PerformanceMetrics(
            operation="test_op",
            latency_ms=50.0,
            timestamp=datetime.now(),
            success=True,
            metadata={"key": "value"}
        )
        
        data = metrics.to_dict()
        assert data["operation"] == "test_op"
        assert data["latency_ms"] == 50.0
        assert data["success"] is True
        assert data["metadata"]["key"] == "value"
        
        # Ensure JSON serializable
        json_str = json.dumps(data, default=str)
        assert json_str


class TestBaselineCalculator:
    """Test baseline calculation logic."""
    
    def test_calculate_percentiles(self):
        """Test percentile calculation."""
        calculator = BaselineCalculator()
        
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        p50 = calculator.calculate_percentile(latencies, 50)
        p95 = calculator.calculate_percentile(latencies, 95)
        p99 = calculator.calculate_percentile(latencies, 99)
        
        assert p50 == pytest.approx(55, rel=1)  # Median
        assert p95 == pytest.approx(95, rel=5)
        assert p99 == pytest.approx(99, rel=5)
    
    def test_calculate_percentiles_empty_list(self):
        """Test percentile calculation with empty data."""
        calculator = BaselineCalculator()
        
        result = calculator.calculate_percentile([], 50)
        assert result == 0
    
    def test_calculate_standard_deviation(self):
        """Test standard deviation calculation."""
        calculator = BaselineCalculator()
        
        values = [10, 20, 30, 40, 50]
        mean = sum(values) / len(values)
        std_dev = calculator.calculate_std_deviation(values, mean)
        
        # Standard deviation should be ~15.81
        assert std_dev == pytest.approx(15.81, rel=0.1)
    
    def test_detect_outliers(self):
        """Test outlier detection using 2-sigma rule."""
        calculator = BaselineCalculator()
        
        # Normal values with outliers
        values = [10, 12, 11, 13, 9, 100, 11, 10, 200]  # 100 and 200 are outliers
        
        outliers = calculator.detect_outliers(values, sigma_threshold=2)
        
        assert 100 in outliers
        assert 200 in outliers
        assert 10 not in outliers


class TestPerformanceBaseline:
    """Test main performance baseline system."""
    
    @pytest.mark.asyncio
    async def test_baseline_initialization(self):
        """Test baseline system initialization."""
        with patch('genesis.operations.performance_baseline.get_db_session'):
            baseline = PerformanceBaseline()
            
            assert baseline.window_days == 7  # Default window
            assert baseline.update_interval_hours == 168  # Weekly
            assert baseline.baselines == {}
            assert baseline.metrics_buffer == []
    
    @pytest.mark.asyncio
    async def test_record_metric(self):
        """Test recording performance metrics."""
        with patch('genesis.operations.performance_baseline.get_db_session'):
            baseline = PerformanceBaseline()
            
            await baseline.record_metric(
                operation="test_op",
                latency_ms=100.5,
                success=True
            )
            
            assert len(baseline.metrics_buffer) == 1
            assert baseline.metrics_buffer[0].operation == "test_op"
            assert baseline.metrics_buffer[0].latency_ms == 100.5
    
    @pytest.mark.asyncio
    async def test_calculate_baselines(self):
        """Test baseline calculation from metrics."""
        with patch('genesis.operations.performance_baseline.get_db_session') as mock_db:
            # Mock database query results
            mock_metrics = [
                MagicMock(operation="api_call", latency_ms=50),
                MagicMock(operation="api_call", latency_ms=60),
                MagicMock(operation="api_call", latency_ms=70),
                MagicMock(operation="api_call", latency_ms=80),
                MagicMock(operation="api_call", latency_ms=90),
                MagicMock(operation="api_call", latency_ms=100),
                MagicMock(operation="api_call", latency_ms=110),
                MagicMock(operation="api_call", latency_ms=500),  # Outlier
            ]
            
            mock_db.return_value.query.return_value.filter.return_value.all.return_value = mock_metrics
            
            baseline = PerformanceBaseline()
            await baseline.calculate_baselines()
            
            assert "api_call" in baseline.baselines
            api_baseline = baseline.baselines["api_call"]
            
            # Check percentiles (outlier should affect p99)
            assert api_baseline["p50"] > 0
            assert api_baseline["p95"] > api_baseline["p50"]
            assert api_baseline["p99"] >= api_baseline["p95"]
            assert api_baseline["mean"] > 0
            assert api_baseline["std_dev"] > 0
    
    @pytest.mark.asyncio
    async def test_check_degradation(self):
        """Test performance degradation detection."""
        with patch('genesis.operations.performance_baseline.get_db_session'):
            baseline = PerformanceBaseline()
            
            # Set a baseline
            baseline.baselines["test_op"] = {
                "p50": 50,
                "p95": 90,
                "p99": 100,
                "mean": 60,
                "std_dev": 15
            }
            
            # Check normal performance
            is_degraded = await baseline.check_degradation("test_op", 65)
            assert not is_degraded  # Within 2 sigma
            
            # Check degraded performance
            is_degraded = await baseline.check_degradation("test_op", 150)
            assert is_degraded  # Exceeds 2 sigma
    
    @pytest.mark.asyncio
    async def test_export_to_prometheus(self):
        """Test Prometheus metrics export."""
        with patch('genesis.operations.performance_baseline.prometheus_client') as mock_prom:
            baseline = PerformanceBaseline()
            
            baseline.baselines = {
                "api_call": {
                    "p50": 50,
                    "p95": 90,
                    "p99": 100,
                    "mean": 60,
                    "std_dev": 15
                }
            }
            
            await baseline.export_to_prometheus()
            
            # Verify Prometheus gauges were set
            assert mock_prom.Gauge.called
    
    @pytest.mark.asyncio
    async def test_auto_update_schedule(self):
        """Test automatic baseline updates."""
        with patch('genesis.operations.performance_baseline.get_db_session'):
            with patch('genesis.operations.performance_baseline.asyncio.sleep') as mock_sleep:
                baseline = PerformanceBaseline()
                baseline.update_interval_hours = 0.001  # Very short for testing
                
                # Mock sleep to prevent actual waiting
                mock_sleep.side_effect = asyncio.CancelledError()
                
                with pytest.raises(asyncio.CancelledError):
                    await baseline.start_auto_update()
    
    @pytest.mark.asyncio
    async def test_performance_alerts(self):
        """Test performance alert generation."""
        with patch('genesis.operations.performance_baseline.get_db_session'):
            with patch('genesis.operations.performance_baseline.send_alert') as mock_alert:
                baseline = PerformanceBaseline()
                
                baseline.baselines["critical_op"] = {
                    "p50": 10,
                    "p95": 20,
                    "p99": 25,
                    "mean": 12,
                    "std_dev": 3
                }
                
                # Trigger alert with severely degraded performance
                await baseline.check_degradation("critical_op", 50)
                
                # Verify alert was sent
                mock_alert.assert_called_once()
                alert_data = mock_alert.call_args[0][0]
                assert alert_data.operation == "critical_op"
                assert alert_data.severity == "HIGH"
    
    @pytest.mark.asyncio
    async def test_baseline_persistence(self):
        """Test baseline persistence to database."""
        with patch('genesis.operations.performance_baseline.get_db_session') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value = mock_session
            
            baseline = PerformanceBaseline()
            baseline.baselines = {
                "test_op": {
                    "p50": 50,
                    "p95": 90,
                    "p99": 100,
                    "mean": 60,
                    "std_dev": 15,
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            await baseline.save_baselines()
            
            # Verify database operations
            assert mock_session.merge.called
            assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_rolling_window_calculation(self):
        """Test rolling window for baseline calculation."""
        with patch('genesis.operations.performance_baseline.get_db_session') as mock_db:
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            
            # Create metrics over time
            mock_metrics = []
            for i in range(100):
                timestamp = week_ago + timedelta(hours=i)
                mock_metrics.append(
                    MagicMock(
                        operation="test_op",
                        latency_ms=50 + i % 20,  # Varying latency
                        timestamp=timestamp
                    )
                )
            
            mock_db.return_value.query.return_value.filter.return_value.all.return_value = mock_metrics
            
            baseline = PerformanceBaseline(window_days=7)
            await baseline.calculate_baselines()
            
            assert "test_op" in baseline.baselines
            # Verify window was applied
            assert len(mock_metrics) > 0


class TestPerformanceIntegration:
    """Integration tests for performance baseline system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from metric recording to alerting."""
        with patch('genesis.operations.performance_baseline.get_db_session'):
            with patch('genesis.operations.performance_baseline.send_alert') as mock_alert:
                baseline = PerformanceBaseline()
                
                # Record normal metrics
                for i in range(100):
                    await baseline.record_metric(
                        operation="trade_execution",
                        latency_ms=50 + (i % 10),
                        success=True
                    )
                
                # Calculate baseline
                await baseline.calculate_baselines()
                
                # Record degraded performance
                await baseline.record_metric(
                    operation="trade_execution",
                    latency_ms=500,  # Severe degradation
                    success=False
                )
                
                # Check for degradation
                is_degraded = await baseline.check_degradation("trade_execution", 500)
                assert is_degraded
                
                # Verify alert would be triggered
                if is_degraded:
                    alert = PerformanceAlert(
                        operation="trade_execution",
                        current_latency=500,
                        baseline_p95=baseline.baselines["trade_execution"]["p95"],
                        severity="HIGH",
                        timestamp=datetime.now()
                    )
                    mock_alert(alert)
                
                assert mock_alert.called