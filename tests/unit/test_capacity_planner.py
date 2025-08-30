"""Unit tests for Capacity Planner."""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock
import numpy as np
from typing import List, Dict

from genesis.monitoring.capacity_planner import (
    ResourceType,
    ResourceMetric,
    CapacityThreshold,
    ForecastModel,
    CapacityPlan,
    CapacityPlanner,
    ResourceUtilization,
    ScalingRecommendation,
    TrendAnalyzer
)


class TestResourceMetric:
    """Test ResourceMetric class."""
    
    def test_resource_metric_creation(self):
        """Test creating a resource metric."""
        metric = ResourceMetric(
            resource_type=ResourceType.CPU,
            current_usage=65.5,
            max_capacity=100.0,
            unit="percent"
        )
        
        assert metric.resource_type == ResourceType.CPU
        assert metric.current_usage == 65.5
        assert metric.max_capacity == 100.0
        assert metric.utilization_percent == 65.5
    
    def test_resource_metric_utilization_calculation(self):
        """Test utilization percentage calculation."""
        metric = ResourceMetric(
            resource_type=ResourceType.MEMORY,
            current_usage=4096,
            max_capacity=8192,
            unit="MB"
        )
        
        assert metric.utilization_percent == 50.0
    
    def test_resource_metric_headroom(self):
        """Test calculating resource headroom."""
        metric = ResourceMetric(
            resource_type=ResourceType.DISK,
            current_usage=750,
            max_capacity=1000,
            unit="GB"
        )
        
        assert metric.get_headroom() == 250  # 250 GB available
        assert metric.get_headroom_percent() == 25.0  # 25% available


class TestCapacityThreshold:
    """Test CapacityThreshold class."""
    
    def test_threshold_creation(self):
        """Test creating capacity thresholds."""
        threshold = CapacityThreshold(
            resource_type=ResourceType.CPU,
            warning_percent=70,
            critical_percent=85,
            emergency_percent=95
        )
        
        assert threshold.warning_percent == 70
        assert threshold.critical_percent == 85
        assert threshold.emergency_percent == 95
    
    def test_threshold_check_level(self):
        """Test checking threshold levels."""
        threshold = CapacityThreshold(
            resource_type=ResourceType.MEMORY,
            warning_percent=60,
            critical_percent=80,
            emergency_percent=90
        )
        
        assert threshold.check_level(50) == "normal"
        assert threshold.check_level(65) == "warning"
        assert threshold.check_level(82) == "critical"
        assert threshold.check_level(92) == "emergency"
    
    def test_threshold_validation(self):
        """Test threshold validation."""
        # Invalid thresholds (not in ascending order)
        with pytest.raises(ValueError):
            CapacityThreshold(
                resource_type=ResourceType.CPU,
                warning_percent=80,
                critical_percent=70,  # Less than warning
                emergency_percent=90
            )


class TestTrendAnalyzer:
    """Test TrendAnalyzer class."""
    
    def test_trend_analyzer_initialization(self):
        """Test trend analyzer initialization."""
        analyzer = TrendAnalyzer(window_size=7)
        
        assert analyzer.window_size == 7
        assert len(analyzer.data_points) == 0
    
    def test_trend_analyzer_add_data(self):
        """Test adding data points."""
        analyzer = TrendAnalyzer(window_size=5)
        
        for i in range(10):
            analyzer.add_data_point(i * 10)
        
        # Should only keep last 5 points
        assert len(analyzer.data_points) == 5
        assert analyzer.data_points == [50, 60, 70, 80, 90]
    
    def test_trend_analyzer_calculate_trend(self):
        """Test calculating trend."""
        analyzer = TrendAnalyzer()
        
        # Add increasing data
        data = [10, 20, 30, 40, 50]
        for value in data:
            analyzer.add_data_point(value)
        
        trend = analyzer.calculate_trend()
        
        assert trend["direction"] == "increasing"
        assert trend["slope"] == 10.0  # Increase of 10 per step
        assert trend["r_squared"] == 1.0  # Perfect linear fit
    
    def test_trend_analyzer_decreasing_trend(self):
        """Test detecting decreasing trend."""
        analyzer = TrendAnalyzer()
        
        # Add decreasing data
        data = [100, 90, 80, 70, 60]
        for value in data:
            analyzer.add_data_point(value)
        
        trend = analyzer.calculate_trend()
        
        assert trend["direction"] == "decreasing"
        assert trend["slope"] == -10.0
    
    def test_trend_analyzer_stable_trend(self):
        """Test detecting stable trend."""
        analyzer = TrendAnalyzer()
        
        # Add stable data with small variations
        data = [50, 51, 49, 50, 51, 50]
        for value in data:
            analyzer.add_data_point(value)
        
        trend = analyzer.calculate_trend()
        
        assert trend["direction"] == "stable"
        assert abs(trend["slope"]) < 1.0
    
    def test_trend_analyzer_seasonality(self):
        """Test detecting seasonality."""
        analyzer = TrendAnalyzer(window_size=24)  # 24 hours
        
        # Add data with daily pattern
        for hour in range(48):  # 2 days
            # Peak at noon (hour 12), low at midnight
            value = 50 + 30 * np.sin((hour % 24) * np.pi / 12)
            analyzer.add_data_point(value)
        
        seasonality = analyzer.detect_seasonality(period=24)
        
        assert seasonality["has_seasonality"] is True
        assert seasonality["period"] == 24


class TestForecastModel:
    """Test ForecastModel class."""
    
    def test_forecast_model_initialization(self):
        """Test forecast model initialization."""
        model = ForecastModel(model_type="linear")
        
        assert model.model_type == "linear"
        assert model.is_trained is False
    
    def test_forecast_model_train_linear(self):
        """Test training linear forecast model."""
        model = ForecastModel(model_type="linear")
        
        # Generate linear data
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(10, 0, -1)]
        values = [10 * i for i in range(1, 11)]
        
        model.train(timestamps, values)
        
        assert model.is_trained is True
        assert model.coefficients is not None
    
    def test_forecast_model_predict(self):
        """Test making predictions."""
        model = ForecastModel(model_type="linear")
        
        # Train with simple linear data
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(5, 0, -1)]
        values = [20, 30, 40, 50, 60]
        
        model.train(timestamps, values)
        
        # Predict next value
        future_time = datetime.now() + timedelta(hours=1)
        prediction = model.predict(future_time)
        
        assert prediction > 60  # Should continue the trend
    
    def test_forecast_model_moving_average(self):
        """Test moving average forecast model."""
        model = ForecastModel(model_type="moving_average", window=3)
        
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(5, 0, -1)]
        values = [10, 20, 30, 25, 35]
        
        model.train(timestamps, values)
        
        # Predict should be average of last 3 values
        prediction = model.predict(datetime.now())
        expected = (30 + 25 + 35) / 3
        
        assert abs(prediction - expected) < 0.1
    
    def test_forecast_model_exponential_smoothing(self):
        """Test exponential smoothing model."""
        model = ForecastModel(model_type="exponential", alpha=0.3)
        
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(5, 0, -1)]
        values = [10, 12, 14, 13, 15]
        
        model.train(timestamps, values)
        prediction = model.predict(datetime.now())
        
        assert prediction > 0
        assert prediction < 20  # Reasonable range
    
    def test_forecast_model_accuracy_metrics(self):
        """Test calculating model accuracy metrics."""
        model = ForecastModel(model_type="linear")
        
        # Train model
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(10, 0, -1)]
        values = [10 * i + np.random.normal(0, 2) for i in range(1, 11)]
        
        model.train(timestamps, values)
        
        # Calculate accuracy metrics
        metrics = model.calculate_accuracy_metrics(timestamps[-3:], values[-3:])
        
        assert "mae" in metrics  # Mean Absolute Error
        assert "rmse" in metrics  # Root Mean Square Error
        assert "mape" in metrics  # Mean Absolute Percentage Error


class TestScalingRecommendation:
    """Test ScalingRecommendation class."""
    
    def test_scaling_recommendation_creation(self):
        """Test creating scaling recommendation."""
        recommendation = ScalingRecommendation(
            resource_type=ResourceType.CPU,
            action="scale_up",
            target_capacity=8,
            reason="CPU utilization above 80% for 10 minutes",
            urgency="high"
        )
        
        assert recommendation.action == "scale_up"
        assert recommendation.target_capacity == 8
        assert recommendation.urgency == "high"
    
    def test_scaling_recommendation_cost_estimate(self):
        """Test cost estimation for scaling."""
        recommendation = ScalingRecommendation(
            resource_type=ResourceType.MEMORY,
            action="scale_up",
            target_capacity=16,
            current_capacity=8,
            cost_per_unit=10.0
        )
        
        cost = recommendation.calculate_cost()
        assert cost == 80.0  # (16 - 8) * 10.0
    
    def test_scaling_recommendation_validation(self):
        """Test recommendation validation."""
        recommendation = ScalingRecommendation(
            resource_type=ResourceType.DISK,
            action="scale_down",
            target_capacity=500,
            current_capacity=1000,
            min_capacity=600
        )
        
        assert recommendation.is_valid() is False  # Below minimum


class TestCapacityPlan:
    """Test CapacityPlan class."""
    
    def test_capacity_plan_creation(self):
        """Test creating capacity plan."""
        plan = CapacityPlan(
            forecast_period_days=30,
            confidence_level=0.95
        )
        
        assert plan.forecast_period_days == 30
        assert plan.confidence_level == 0.95
        assert len(plan.recommendations) == 0
    
    def test_capacity_plan_add_recommendation(self):
        """Test adding recommendations to plan."""
        plan = CapacityPlan(forecast_period_days=7)
        
        rec1 = ScalingRecommendation(
            resource_type=ResourceType.CPU,
            action="scale_up",
            target_capacity=4
        )
        
        rec2 = ScalingRecommendation(
            resource_type=ResourceType.MEMORY,
            action="scale_up",
            target_capacity=16
        )
        
        plan.add_recommendation(rec1)
        plan.add_recommendation(rec2)
        
        assert len(plan.recommendations) == 2
    
    def test_capacity_plan_total_cost(self):
        """Test calculating total plan cost."""
        plan = CapacityPlan(forecast_period_days=30)
        
        plan.add_recommendation(ScalingRecommendation(
            resource_type=ResourceType.CPU,
            action="scale_up",
            target_capacity=4,
            current_capacity=2,
            cost_per_unit=50.0
        ))
        
        plan.add_recommendation(ScalingRecommendation(
            resource_type=ResourceType.MEMORY,
            action="scale_up",
            target_capacity=16,
            current_capacity=8,
            cost_per_unit=25.0
        ))
        
        total_cost = plan.calculate_total_cost()
        assert total_cost == 300.0  # (2*50) + (8*25)
    
    def test_capacity_plan_prioritization(self):
        """Test prioritizing recommendations."""
        plan = CapacityPlan(forecast_period_days=7)
        
        # Add recommendations with different urgencies
        plan.add_recommendation(ScalingRecommendation(
            resource_type=ResourceType.CPU,
            action="scale_up",
            target_capacity=4,
            urgency="low"
        ))
        
        plan.add_recommendation(ScalingRecommendation(
            resource_type=ResourceType.MEMORY,
            action="scale_up",
            target_capacity=16,
            urgency="high"
        ))
        
        plan.add_recommendation(ScalingRecommendation(
            resource_type=ResourceType.DISK,
            action="scale_up",
            target_capacity=1000,
            urgency="medium"
        ))
        
        prioritized = plan.get_prioritized_recommendations()
        
        assert prioritized[0].urgency == "high"
        assert prioritized[1].urgency == "medium"
        assert prioritized[2].urgency == "low"


class TestCapacityPlanner:
    """Test main CapacityPlanner class."""
    
    def test_capacity_planner_initialization(self):
        """Test capacity planner initialization."""
        planner = CapacityPlanner()
        
        assert len(planner.resource_metrics) == 0
        assert len(planner.thresholds) > 0  # Should have default thresholds
        assert len(planner.historical_data) == 0
    
    def test_capacity_planner_record_metric(self):
        """Test recording resource metrics."""
        planner = CapacityPlanner()
        
        planner.record_metric(ResourceType.CPU, 65.5)
        planner.record_metric(ResourceType.MEMORY, 4096, max_capacity=8192)
        planner.record_metric(ResourceType.DISK, 500, max_capacity=1000)
        
        assert ResourceType.CPU in planner.resource_metrics
        assert planner.resource_metrics[ResourceType.CPU].current_usage == 65.5
    
    def test_capacity_planner_check_thresholds(self):
        """Test checking threshold violations."""
        planner = CapacityPlanner()
        
        # Set custom thresholds
        planner.set_threshold(ResourceType.CPU, warning=60, critical=80)
        
        # Record metrics
        planner.record_metric(ResourceType.CPU, 75)
        
        violations = planner.check_threshold_violations()
        
        assert len(violations) == 1
        assert violations[0]["resource"] == ResourceType.CPU
        assert violations[0]["level"] == "warning"
    
    @pytest.mark.asyncio
    async def test_capacity_planner_collect_metrics(self):
        """Test collecting system metrics."""
        planner = CapacityPlanner()
        
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 60.0
                with patch('psutil.disk_usage') as mock_disk:
                    mock_disk.return_value.percent = 70.0
                    
                    await planner.collect_system_metrics()
        
        assert ResourceType.CPU in planner.resource_metrics
        assert planner.resource_metrics[ResourceType.CPU].current_usage == 50.0
    
    def test_capacity_planner_forecast_usage(self):
        """Test forecasting resource usage."""
        planner = CapacityPlanner()
        
        # Add historical data
        now = datetime.now()
        for i in range(10):
            timestamp = now - timedelta(hours=10-i)
            planner.add_historical_data(ResourceType.CPU, timestamp, 50 + i*2)
        
        # Forecast future usage
        forecast = planner.forecast_usage(
            ResourceType.CPU,
            forecast_days=7
        )
        
        assert len(forecast) > 0
        assert all(isinstance(v, (int, float)) for v in forecast)
    
    def test_capacity_planner_growth_rate(self):
        """Test calculating resource growth rate."""
        planner = CapacityPlanner()
        
        # Add historical data showing growth
        now = datetime.now()
        for day in range(30):
            timestamp = now - timedelta(days=30-day)
            usage = 100 + day * 5  # 5 units per day growth
            planner.add_historical_data(ResourceType.DISK, timestamp, usage)
        
        growth_rate = planner.calculate_growth_rate(ResourceType.DISK, days=30)
        
        assert growth_rate > 0  # Positive growth
        assert abs(growth_rate - 5.0) < 1.0  # Approximately 5 units/day
    
    def test_capacity_planner_time_to_limit(self):
        """Test calculating time until capacity limit."""
        planner = CapacityPlanner()
        
        # Set current usage and growth rate
        planner.record_metric(ResourceType.DISK, 800, max_capacity=1000)
        
        # Add historical data showing linear growth
        now = datetime.now()
        for day in range(10):
            timestamp = now - timedelta(days=10-day)
            usage = 700 + day * 10  # 10 GB per day
            planner.add_historical_data(ResourceType.DISK, timestamp, usage)
        
        days_to_limit = planner.calculate_time_to_limit(ResourceType.DISK)
        
        assert days_to_limit is not None
        assert days_to_limit < 30  # Should hit limit within a month
    
    def test_capacity_planner_generate_plan(self):
        """Test generating capacity plan."""
        planner = CapacityPlanner()
        
        # Set up current state
        planner.record_metric(ResourceType.CPU, 85, max_capacity=100)
        planner.record_metric(ResourceType.MEMORY, 7000, max_capacity=8192)
        planner.record_metric(ResourceType.DISK, 900, max_capacity=1000)
        
        # Generate plan
        plan = planner.generate_capacity_plan(forecast_days=30)
        
        assert isinstance(plan, CapacityPlan)
        assert len(plan.recommendations) > 0
    
    def test_capacity_planner_optimization(self):
        """Test resource optimization recommendations."""
        planner = CapacityPlanner()
        
        # Set underutilized resources
        planner.record_metric(ResourceType.CPU, 20, max_capacity=100)
        planner.record_metric(ResourceType.MEMORY, 1000, max_capacity=8192)
        
        optimizations = planner.get_optimization_opportunities()
        
        assert len(optimizations) > 0
        assert any(o["action"] == "scale_down" for o in optimizations)
    
    def test_capacity_planner_alert_generation(self):
        """Test generating capacity alerts."""
        planner = CapacityPlanner()
        
        # Set critical usage
        planner.record_metric(ResourceType.CPU, 95, max_capacity=100)
        
        alerts = planner.generate_alerts()
        
        assert len(alerts) > 0
        assert any("critical" in alert["severity"] for alert in alerts)
    
    def test_capacity_planner_report_generation(self):
        """Test generating capacity report."""
        planner = CapacityPlanner()
        
        # Set up metrics
        planner.record_metric(ResourceType.CPU, 65)
        planner.record_metric(ResourceType.MEMORY, 4096, max_capacity=8192)
        planner.record_metric(ResourceType.DISK, 500, max_capacity=1000)
        
        report = planner.generate_report()
        
        assert "summary" in report
        assert "resources" in report
        assert "recommendations" in report
        assert len(report["resources"]) == 3
    
    @pytest.mark.asyncio
    async def test_capacity_planner_auto_scaling(self):
        """Test auto-scaling functionality."""
        planner = CapacityPlanner(auto_scale_enabled=True)
        
        # Set critical CPU usage
        planner.record_metric(ResourceType.CPU, 92, max_capacity=100)
        
        # Mock scaling action
        with patch.object(planner, 'execute_scaling') as mock_scale:
            await planner.check_and_scale()
            
            mock_scale.assert_called_once()
            call_args = mock_scale.call_args[0]
            assert call_args[0].resource_type == ResourceType.CPU
            assert call_args[0].action == "scale_up"