"""Unit tests for validation criteria module."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from genesis.paper_trading.validation_criteria import (
    CriteriaResult,
    ValidationCriteria,
    ValidationResult,
)


class TestValidationCriteria:
    """Test ValidationCriteria class."""

    @pytest.fixture
    def criteria(self):
        """Create test validation criteria."""
        return ValidationCriteria(
            min_trades=100,
            min_days=7,
            min_sharpe_ratio=1.5,
            max_drawdown=Decimal("0.10"),
            min_win_rate=Decimal("0.55"),
            min_profit_factor=2.0,
            min_avg_win_loss_ratio=1.5,
        )

    def test_initialization(self, criteria):
        """Test criteria initialization."""
        assert criteria.min_trades == 100
        assert criteria.min_days == 7
        assert criteria.min_sharpe_ratio == 1.5
        assert criteria.max_drawdown == Decimal("0.10")
        assert criteria.min_win_rate == Decimal("0.55")
        assert criteria.min_profit_factor == 2.0
        assert criteria.min_avg_win_loss_ratio == 1.5

    def test_default_values(self):
        """Test default criteria values."""
        criteria = ValidationCriteria()
        assert criteria.min_trades == 100
        assert criteria.min_days == 7
        assert criteria.min_sharpe_ratio == 1.5
        assert criteria.max_drawdown == Decimal("0.10")
        assert criteria.min_win_rate == Decimal("0.55")
        assert criteria.min_profit_factor is None
        assert criteria.min_avg_win_loss_ratio is None

    def test_validate_trades(self, criteria):
        """Test trade count validation."""
        # Test with insufficient trades
        result = criteria.validate_trades(50)
        assert isinstance(result, CriteriaResult)
        assert result.passed is False
        assert result.name == "Minimum Trades"
        assert result.required == 100
        assert result.actual == 50
        assert "50 < 100" in result.reason
        
        # Test with sufficient trades
        result = criteria.validate_trades(150)
        assert result.passed is True
        assert result.actual == 150

    def test_validate_days(self, criteria):
        """Test trading days validation."""
        # Test with insufficient days
        start_date = datetime.now() - timedelta(days=3)
        result = criteria.validate_days(start_date)
        assert result.passed is False
        assert result.name == "Minimum Days"
        assert result.required == 7
        assert result.actual == 3
        
        # Test with sufficient days
        start_date = datetime.now() - timedelta(days=10)
        result = criteria.validate_days(start_date)
        assert result.passed is True
        assert result.actual >= 7

    def test_validate_sharpe_ratio(self, criteria):
        """Test Sharpe ratio validation."""
        # Test with low Sharpe
        result = criteria.validate_sharpe_ratio(0.8)
        assert result.passed is False
        assert result.name == "Sharpe Ratio"
        assert result.required == 1.5
        assert result.actual == 0.8
        assert "0.8 < 1.5" in result.reason
        
        # Test with good Sharpe
        result = criteria.validate_sharpe_ratio(2.0)
        assert result.passed is True
        assert result.actual == 2.0
        
        # Test with None (no validation)
        criteria.min_sharpe_ratio = None
        result = criteria.validate_sharpe_ratio(0.5)
        assert result.passed is True

    def test_validate_drawdown(self, criteria):
        """Test maximum drawdown validation."""
        # Test with excessive drawdown
        result = criteria.validate_drawdown(Decimal("0.15"))
        assert result.passed is False
        assert result.name == "Maximum Drawdown"
        assert result.required == Decimal("0.10")
        assert result.actual == Decimal("0.15")
        assert "15.00% > 10.00%" in result.reason
        
        # Test with acceptable drawdown
        result = criteria.validate_drawdown(Decimal("0.08"))
        assert result.passed is True
        assert result.actual == Decimal("0.08")
        
        # Test with None (no validation)
        criteria.max_drawdown = None
        result = criteria.validate_drawdown(Decimal("0.20"))
        assert result.passed is True

    def test_validate_win_rate(self, criteria):
        """Test win rate validation."""
        # Test with low win rate
        result = criteria.validate_win_rate(Decimal("0.45"))
        assert result.passed is False
        assert result.name == "Win Rate"
        assert result.required == Decimal("0.55")
        assert result.actual == Decimal("0.45")
        assert "45.00% < 55.00%" in result.reason
        
        # Test with good win rate
        result = criteria.validate_win_rate(Decimal("0.60"))
        assert result.passed is True
        assert result.actual == Decimal("0.60")
        
        # Test with None (no validation)
        criteria.min_win_rate = None
        result = criteria.validate_win_rate(Decimal("0.30"))
        assert result.passed is True

    def test_validate_profit_factor(self, criteria):
        """Test profit factor validation."""
        # Test with low profit factor
        result = criteria.validate_profit_factor(1.2)
        assert result.passed is False
        assert result.name == "Profit Factor"
        assert result.required == 2.0
        assert result.actual == 1.2
        assert "1.2 < 2.0" in result.reason
        
        # Test with good profit factor
        result = criteria.validate_profit_factor(2.5)
        assert result.passed is True
        assert result.actual == 2.5
        
        # Test with None criteria
        criteria.min_profit_factor = None
        result = criteria.validate_profit_factor(1.0)
        assert result.passed is True

    def test_validate_avg_win_loss_ratio(self, criteria):
        """Test average win/loss ratio validation."""
        # Test with low ratio
        result = criteria.validate_avg_win_loss_ratio(1.2)
        assert result.passed is False
        assert result.name == "Avg Win/Loss Ratio"
        assert result.required == 1.5
        assert result.actual == 1.2
        assert "1.2 < 1.5" in result.reason
        
        # Test with good ratio
        result = criteria.validate_avg_win_loss_ratio(2.0)
        assert result.passed is True
        assert result.actual == 2.0
        
        # Test with None criteria
        criteria.min_avg_win_loss_ratio = None
        result = criteria.validate_avg_win_loss_ratio(1.0)
        assert result.passed is True

    def test_validate_all(self, criteria):
        """Test validating all criteria together."""
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 1.8,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
            "profit_factor": 2.5,
            "avg_win_loss_ratio": 1.8,
        }
        
        result = criteria.validate_all(metrics)
        
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert len(result.failed_criteria) == 0
        assert len(result.passed_criteria) == 7
        assert result.overall_score > 0

    def test_validate_all_with_failures(self, criteria):
        """Test validation with some failures."""
        metrics = {
            "total_trades": 50,  # Fail
            "start_date": datetime.now() - timedelta(days=3),  # Fail
            "sharpe_ratio": 2.0,  # Pass
            "max_drawdown": Decimal("0.15"),  # Fail
            "win_rate": Decimal("0.60"),  # Pass
            "profit_factor": 2.5,  # Pass
            "avg_win_loss_ratio": 1.8,  # Pass
        }
        
        result = criteria.validate_all(metrics)
        
        assert result.passed is False
        assert len(result.failed_criteria) == 3
        assert len(result.passed_criteria) == 4
        assert result.overall_score < 1.0
        
        # Check specific failures
        failed_names = [c.name for c in result.failed_criteria]
        assert "Minimum Trades" in failed_names
        assert "Minimum Days" in failed_names
        assert "Maximum Drawdown" in failed_names

    def test_calculate_score(self, criteria):
        """Test overall score calculation."""
        metrics = {
            "total_trades": 100,  # Exactly at minimum
            "start_date": datetime.now() - timedelta(days=7),  # Exactly at minimum
            "sharpe_ratio": 1.5,  # Exactly at minimum
            "max_drawdown": Decimal("0.10"),  # Exactly at maximum
            "win_rate": Decimal("0.55"),  # Exactly at minimum
            "profit_factor": 2.0,  # Exactly at minimum
            "avg_win_loss_ratio": 1.5,  # Exactly at minimum
        }
        
        result = criteria.validate_all(metrics)
        
        # All criteria met but at minimum levels
        assert result.passed is True
        assert result.overall_score >= 0.5  # Should be around baseline

    def test_score_with_excellent_metrics(self, criteria):
        """Test score with excellent metrics."""
        metrics = {
            "total_trades": 500,  # Well above minimum
            "start_date": datetime.now() - timedelta(days=30),  # Well above minimum
            "sharpe_ratio": 3.0,  # Double the minimum
            "max_drawdown": Decimal("0.03"),  # Much better than maximum
            "win_rate": Decimal("0.75"),  # High win rate
            "profit_factor": 4.0,  # Double the minimum
            "avg_win_loss_ratio": 3.0,  # Double the minimum
        }
        
        result = criteria.validate_all(metrics)
        
        assert result.passed is True
        assert result.overall_score > 0.9  # Should be high score

    def test_get_summary(self, criteria):
        """Test getting validation summary."""
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 1.8,
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
            "profit_factor": 2.5,
            "avg_win_loss_ratio": 1.8,
        }
        
        result = criteria.validate_all(metrics)
        summary = result.get_summary()
        
        assert isinstance(summary, str)
        assert "PASSED" in summary
        assert "Score:" in summary
        assert "✓" in summary  # Check marks for passed criteria

    def test_get_summary_with_failures(self, criteria):
        """Test summary with failures."""
        metrics = {
            "total_trades": 50,
            "start_date": datetime.now() - timedelta(days=3),
            "sharpe_ratio": 1.0,
            "max_drawdown": Decimal("0.15"),
            "win_rate": Decimal("0.45"),
            "profit_factor": 1.5,
            "avg_win_loss_ratio": 1.2,
        }
        
        result = criteria.validate_all(metrics)
        summary = result.get_summary()
        
        assert "FAILED" in summary
        assert "✗" in summary  # X marks for failed criteria
        assert "Failed Criteria:" in summary

    def test_custom_criteria(self):
        """Test custom criteria configuration."""
        custom = ValidationCriteria(
            min_trades=50,
            min_days=3,
            min_sharpe_ratio=1.0,
            max_drawdown=Decimal("0.20"),
            min_win_rate=Decimal("0.50"),
        )
        
        metrics = {
            "total_trades": 60,
            "start_date": datetime.now() - timedelta(days=4),
            "sharpe_ratio": 1.2,
            "max_drawdown": Decimal("0.15"),
            "win_rate": Decimal("0.52"),
        }
        
        result = custom.validate_all(metrics)
        assert result.passed is True

    def test_partial_metrics(self, criteria):
        """Test validation with partial metrics."""
        # Only provide some metrics
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 2.0,
            # Missing drawdown, win_rate, profit_factor, avg_win_loss_ratio
        }
        
        result = criteria.validate_all(metrics)
        
        # Should handle missing metrics gracefully
        assert result.passed is False  # Failed due to missing required metrics

    def test_criteria_weights(self, criteria):
        """Test that criteria have appropriate weights in scoring."""
        # Test that critical metrics have higher weight
        metrics = {
            "total_trades": 150,
            "start_date": datetime.now() - timedelta(days=10),
            "sharpe_ratio": 0.5,  # Very bad Sharpe
            "max_drawdown": Decimal("0.08"),
            "win_rate": Decimal("0.60"),
            "profit_factor": 2.5,
            "avg_win_loss_ratio": 1.8,
        }
        
        result = criteria.validate_all(metrics)
        
        # Should fail due to poor Sharpe ratio (critical metric)
        assert result.passed is False
        assert any("Sharpe" in c.name for c in result.failed_criteria)

    def test_edge_cases(self, criteria):
        """Test edge cases in validation."""
        # Test with exactly meeting criteria
        metrics = {
            "total_trades": 100,
            "start_date": datetime.now() - timedelta(days=7),
            "sharpe_ratio": 1.5,
            "max_drawdown": Decimal("0.10"),
            "win_rate": Decimal("0.55"),
            "profit_factor": 2.0,
            "avg_win_loss_ratio": 1.5,
        }
        
        result = criteria.validate_all(metrics)
        assert result.passed is True
        
        # Test with slightly below criteria
        metrics["total_trades"] = 99
        result = criteria.validate_all(metrics)
        assert result.passed is False
        
        # Test with zero/negative values
        metrics["total_trades"] = 0
        result = criteria.validate_all(metrics)
        assert result.passed is False
        
        metrics["sharpe_ratio"] = -1.0
        result = criteria.validate_all(metrics)
        assert result.passed is False

    def test_to_dict(self, criteria):
        """Test converting criteria to dictionary."""
        criteria_dict = criteria.to_dict()
        
        assert isinstance(criteria_dict, dict)
        assert criteria_dict["min_trades"] == 100
        assert criteria_dict["min_days"] == 7
        assert criteria_dict["min_sharpe_ratio"] == 1.5
        assert criteria_dict["max_drawdown"] == "0.10"
        assert criteria_dict["min_win_rate"] == "0.55"

    def test_from_dict(self):
        """Test creating criteria from dictionary."""
        config = {
            "min_trades": 200,
            "min_days": 14,
            "min_sharpe_ratio": 2.0,
            "max_drawdown": "0.05",
            "min_win_rate": "0.60",
            "min_profit_factor": 3.0,
        }
        
        criteria = ValidationCriteria.from_dict(config)
        
        assert criteria.min_trades == 200
        assert criteria.min_days == 14
        assert criteria.min_sharpe_ratio == 2.0
        assert criteria.max_drawdown == Decimal("0.05")
        assert criteria.min_win_rate == Decimal("0.60")
        assert criteria.min_profit_factor == 3.0