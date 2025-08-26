"""
Unit tests for behavior-P&L correlation analysis.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from genesis.analytics.behavior_correlation import BehaviorPnLCorrelator
from genesis.core.exceptions import ValidationError


class TestBehaviorPnLCorrelator:
    """Tests for behavior-P&L correlation."""
    
    def test_initialization(self):
        """Test correlator initialization."""
        correlator = BehaviorPnLCorrelator(
            min_sample_size=20,
            significance_threshold=0.01
        )
        
        assert correlator.min_sample_size == 20
        assert correlator.significance_threshold == 0.01
        assert len(correlator.behavior_data) == 0
        assert len(correlator.pnl_data) == 0
    
    def test_add_behavior_data(self):
        """Test adding behavioral data."""
        correlator = BehaviorPnLCorrelator()
        
        now = datetime.utcnow()
        correlator.add_behavior_data(
            "click_latency",
            now,
            150.5,
            {"action": "buy"}
        )
        
        assert "click_latency" in correlator.behavior_data
        assert len(correlator.behavior_data["click_latency"]) == 1
        
        data = correlator.behavior_data["click_latency"][0]
        assert data["timestamp"] == now
        assert data["value"] == 150.5
        assert data["metadata"]["action"] == "buy"
    
    def test_add_pnl_data(self):
        """Test adding P&L data."""
        correlator = BehaviorPnLCorrelator()
        
        now = datetime.utcnow()
        correlator.add_pnl_data(
            now,
            Decimal("100.50"),
            "position_123"
        )
        
        assert len(correlator.pnl_data) == 1
        
        data = correlator.pnl_data[0]
        assert data["timestamp"] == now
        assert data["pnl"] == 100.50
        assert data["position_id"] == "position_123"
    
    def test_cache_invalidation(self):
        """Test that cache is invalidated on new data."""
        correlator = BehaviorPnLCorrelator()
        
        # Add to cache
        correlator.correlation_cache["test_30"] = MagicMock()
        
        # Add new behavior data
        correlator.add_behavior_data("test", datetime.utcnow(), 100)
        
        # Cache for that behavior should be cleared
        assert "test_30" not in correlator.correlation_cache
        
        # Add to cache again
        correlator.correlation_cache["test_30"] = MagicMock()
        correlator.correlation_cache["other_30"] = MagicMock()
        
        # Add P&L data
        correlator.add_pnl_data(datetime.utcnow(), Decimal("50"))
        
        # All cache should be cleared
        assert len(correlator.correlation_cache) == 0
    
    def test_correlate_no_data_raises(self):
        """Test correlation with no data raises error."""
        correlator = BehaviorPnLCorrelator()
        
        with pytest.raises(ValidationError, match="No data for behavior"):
            correlator.correlate_behavior_with_pnl("unknown", 30)
        
        # Add behavior but no P&L
        correlator.add_behavior_data("test", datetime.utcnow(), 100)
        
        with pytest.raises(ValidationError, match="No P&L data"):
            correlator.correlate_behavior_with_pnl("test", 30)
    
    def test_align_time_series(self):
        """Test time series alignment."""
        correlator = BehaviorPnLCorrelator()
        
        base_time = datetime.utcnow()
        
        # Add behavior data
        for i in range(5):
            correlator.add_behavior_data(
                "latency",
                base_time - timedelta(minutes=i*10),
                100 + i*10
            )
        
        # Add P&L data
        for i in range(3):
            correlator.pnl_data.append({
                "timestamp": base_time - timedelta(minutes=i*10),
                "pnl": 50 - i*10
            })
        
        # Align with 30 minute window
        aligned = correlator._align_time_series("latency", 30)
        
        assert len(aligned) > 0
        assert all("behavior_value" in d for d in aligned)
        assert all("pnl" in d for d in aligned)
    
    def test_correlate_behavior_with_pnl(self):
        """Test correlation calculation."""
        correlator = BehaviorPnLCorrelator(min_sample_size=5)
        
        base_time = datetime.utcnow()
        
        # Add correlated data (high latency -> low P&L)
        for i in range(10):
            latency = 100 + i*20
            pnl = 100 - i*15
            
            correlator.add_behavior_data(
                "latency",
                base_time - timedelta(minutes=i*5),
                latency
            )
            
            correlator.add_pnl_data(
                base_time - timedelta(minutes=i*5),
                Decimal(str(pnl))
            )
        
        # Calculate correlation
        correlation = correlator.correlate_behavior_with_pnl("latency", 10)
        
        # Should be negative correlation
        assert correlation < 0
        assert -1 <= correlation <= 1
    
    def test_calculate_p_value(self):
        """Test p-value calculation."""
        correlator = BehaviorPnLCorrelator()
        
        # Strongly correlated data
        x = list(range(10))
        y = [i*2 for i in range(10)]
        
        p_value = correlator._calculate_p_value(x, y, n_permutations=100)
        
        # Should have low p-value for strong correlation
        assert p_value < 0.1
        
        # Random data
        x_random = list(range(10))
        y_random = np.random.random(10).tolist()
        
        p_value_random = correlator._calculate_p_value(x_random, y_random, n_permutations=100)
        
        # Should have higher p-value for random data
        assert p_value_random > p_value
    
    def test_determine_significance(self):
        """Test significance level determination."""
        correlator = BehaviorPnLCorrelator()
        
        assert correlator._determine_significance(0.005) == "high"
        assert correlator._determine_significance(0.03) == "medium"
        assert correlator._determine_significance(0.07) == "low"
        assert correlator._determine_significance(0.15) == "none"
    
    def test_get_all_correlations(self):
        """Test getting all correlations."""
        correlator = BehaviorPnLCorrelator(min_sample_size=3)
        
        base_time = datetime.utcnow()
        
        # Add data for multiple behaviors
        for behavior in ["latency", "cancel_rate", "switch_count"]:
            for i in range(5):
                correlator.add_behavior_data(
                    behavior,
                    base_time - timedelta(minutes=i*5),
                    100 + i*10
                )
        
        # Add P&L data
        for i in range(5):
            correlator.add_pnl_data(
                base_time - timedelta(minutes=i*5),
                Decimal(str(50 + i*5))
            )
        
        # Get all correlations
        results = correlator.get_all_correlations(time_window=30)
        
        assert len(results) == 3
        assert all(hasattr(r, "correlation_coefficient") for r in results)
        assert all(hasattr(r, "significance_level") for r in results)
    
    def test_identify_loss_behaviors(self):
        """Test identification of loss-associated behaviors."""
        correlator = BehaviorPnLCorrelator(min_sample_size=3)
        
        # Mock correlation results
        from genesis.analytics.behavior_correlation import CorrelationResult
        
        correlator.correlation_cache = {
            "high_latency_30": CorrelationResult(
                behavior_type="high_latency",
                correlation_coefficient=-0.6,
                p_value=0.01,
                sample_size=50,
                significance_level="high",
                impact_direction="negative"
            ),
            "revenge_trading_30": CorrelationResult(
                behavior_type="revenge_trading",
                correlation_coefficient=-0.4,
                p_value=0.03,
                sample_size=40,
                significance_level="medium",
                impact_direction="negative"
            ),
            "focus_30": CorrelationResult(
                behavior_type="focus",
                correlation_coefficient=0.3,
                p_value=0.02,
                sample_size=45,
                significance_level="medium",
                impact_direction="positive"
            )
        }
        
        # Mock behavior data to avoid validation errors
        correlator.behavior_data = {
            "high_latency": [{"timestamp": datetime.utcnow(), "value": 100}],
            "revenge_trading": [{"timestamp": datetime.utcnow(), "value": 100}],
            "focus": [{"timestamp": datetime.utcnow(), "value": 100}]
        }
        
        correlator.pnl_data = [{"timestamp": datetime.utcnow(), "pnl": 100}]
        
        loss_behaviors = correlator.identify_loss_behaviors(threshold=-0.3)
        
        assert len(loss_behaviors) == 2
        assert all(b.correlation_coefficient < -0.3 for b in loss_behaviors)
        assert all(b.significance_level in ["high", "medium"] for b in loss_behaviors)
    
    def test_calculate_behavior_impact(self):
        """Test behavior impact calculation."""
        correlator = BehaviorPnLCorrelator()
        
        base_time = datetime.utcnow()
        
        # Add behavior data with varying values
        for i in range(20):
            # High latency for first 10, low for next 10
            latency = 200 if i < 10 else 50
            correlator.add_behavior_data(
                "latency",
                base_time - timedelta(minutes=i),
                latency
            )
            
            # Low P&L with high latency, high P&L with low latency
            pnl = -50 if i < 10 else 100
            correlator.add_pnl_data(
                base_time - timedelta(minutes=i),
                Decimal(str(pnl))
            )
        
        impact = correlator.calculate_behavior_impact("latency", threshold_percentile=50)
        
        assert impact.behavior == "latency"
        assert impact.average_pnl_with < impact.average_pnl_without
        assert impact.pnl_difference < 0
        assert impact.occurrences > 0
        assert "recommendation" in impact.recommendation
    
    def test_calculate_behavior_impact_no_data_raises(self):
        """Test impact calculation with no data raises error."""
        correlator = BehaviorPnLCorrelator()
        
        with pytest.raises(ValidationError, match="No data for behavior"):
            correlator.calculate_behavior_impact("unknown")
    
    def test_generate_impact_recommendation(self):
        """Test impact recommendation generation."""
        correlator = BehaviorPnLCorrelator()
        
        # Critical loss pattern
        rec = correlator._generate_impact_recommendation(
            "rapid_clicking",
            Decimal("-150"),
            15
        )
        assert "Critical" in rec
        
        # Warning level
        rec = correlator._generate_impact_recommendation(
            "high_cancel_rate",
            Decimal("-75"),
            8
        )
        assert "Warning" in rec
        
        # Positive pattern
        rec = correlator._generate_impact_recommendation(
            "steady_pace",
            Decimal("75"),
            12
        )
        assert "Positive" in rec
        
        # Neutral
        rec = correlator._generate_impact_recommendation(
            "some_metric",
            Decimal("10"),
            5
        )
        assert "Neutral" in rec
    
    def test_get_correlation_summary(self):
        """Test correlation summary generation."""
        correlator = BehaviorPnLCorrelator(min_sample_size=2)
        
        # Add some test data
        base_time = datetime.utcnow()
        
        for i in range(5):
            correlator.add_behavior_data("test1", base_time - timedelta(minutes=i), 100)
            correlator.add_behavior_data("test2", base_time - timedelta(minutes=i), 200)
            correlator.add_pnl_data(base_time - timedelta(minutes=i), Decimal("50"))
        
        summary = correlator.get_correlation_summary()
        
        assert "total_behaviors_analyzed" in summary
        assert "significant_correlations" in summary
        assert "loss_associated_behaviors" in summary
        assert "top_risk_behaviors" in summary
        assert "recommendation" in summary
    
    def test_generate_summary_recommendation(self):
        """Test summary recommendation generation."""
        correlator = BehaviorPnLCorrelator()
        
        from genesis.analytics.behavior_correlation import CorrelationResult
        
        # No loss behaviors
        rec = correlator._generate_summary_recommendation([])
        assert "No concerning" in rec
        
        # Multiple loss behaviors
        loss_behaviors = [
            CorrelationResult("b1", -0.4, 0.01, 50, "high", "negative"),
            CorrelationResult("b2", -0.35, 0.02, 40, "medium", "negative"),
            CorrelationResult("b3", -0.32, 0.03, 35, "medium", "negative"),
        ]
        rec = correlator._generate_summary_recommendation(loss_behaviors)
        assert "Multiple risk behaviors" in rec
        
        # Strong correlation
        strong_loss = [
            CorrelationResult("critical", -0.7, 0.001, 100, "high", "negative")
        ]
        rec = correlator._generate_summary_recommendation(strong_loss)
        assert "Strong loss correlation" in rec
        
        # Moderate correlation
        moderate_loss = [
            CorrelationResult("watch", -0.35, 0.04, 30, "medium", "negative")
        ]
        rec = correlator._generate_summary_recommendation(moderate_loss)
        assert "Monitor" in rec