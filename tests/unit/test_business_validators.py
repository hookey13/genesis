"""Unit tests for business validators."""

import asyncio
import json
from datetime import datetime, timedelta, UTC
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.validation.business import (
    MetricsValidator,
    PaperTradingValidator,
    RiskValidator,
    StabilityValidator,
    TierGateValidator,
)


class TestPaperTradingValidator:
    """Test paper trading profit validation."""
    
    @pytest.fixture
    def validator(self):
        """Create paper trading validator instance."""
        return PaperTradingValidator()
    
    @pytest.fixture
    def sample_trades(self):
        """Generate sample paper trading data."""
        trades = []
        for i in range(150):
            # Create profitable trades to meet $10k requirement
            pnl = Decimal("100") if i % 2 == 0 else Decimal("-30")
            trades.append({
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 0.01,
                "entry_price": 50000,
                "exit_price": 50000 + float(pnl * 10),
                "pnl": float(pnl),
                "timestamp": (datetime.now() - timedelta(days=30-i)).isoformat(),
                "duration": 3600
            })
        return trades
    
    @pytest.mark.asyncio
    async def test_validate_no_history(self, validator):
        """Test validation with no trading history."""
        with patch.object(validator, "_load_trading_history", return_value=[]):
            result = await validator.validate()
            
            assert result["status"] == "failed"
            assert "No paper trading history" in result["message"]
            assert result["evidence"]["trades"] == 0
    
    @pytest.mark.asyncio
    async def test_validate_insufficient_profit(self, validator):
        """Test validation with insufficient profit."""
        trades = [
            {"pnl": 100, "timestamp": datetime.now().isoformat()}
            for _ in range(10)
        ]
        
        with patch.object(validator, "_load_trading_history", return_value=trades):
            result = await validator.validate()
            
            assert result["status"] == "failed"
            assert "$1000.00 < required $10000.00" in result["message"]
    
    @pytest.mark.asyncio
    async def test_validate_successful(self, validator, sample_trades):
        """Test successful validation with sufficient profit."""
        # Adjust trades to ensure $10k+ profit
        for i in range(100):
            sample_trades[i]["pnl"] = 150  # $150 per trade
        
        with patch.object(validator, "_load_trading_history", return_value=sample_trades):
            result = await validator.validate()
            
            assert result["status"] == "passed"
            assert "Paper trading validation passed" in result["message"]
            assert float(result["evidence"]["total_pnl"]) >= 10000
    
    @pytest.mark.asyncio
    async def test_consecutive_profitable_days(self, validator):
        """Test consecutive profitable days calculation."""
        trades = []
        base_date = datetime.now() - timedelta(days=5)
        
        # Create 3 consecutive profitable days
        for day in range(3):
            for _ in range(5):
                trades.append({
                    "pnl": 100,
                    "timestamp": (base_date + timedelta(days=day)).isoformat()
                })
        
        result = await validator._check_consecutive_profitable_days(trades)
        assert result >= 3
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, validator, sample_trades):
        """Test trading metrics calculation."""
        metrics = await validator._calculate_metrics(sample_trades)
        
        assert "win_rate" in metrics
        assert "max_drawdown" in metrics
        assert "sharpe_ratio" in metrics
        assert "profit_factor" in metrics
        
        assert metrics["win_rate"] >= 0
        assert metrics["win_rate"] <= 1


class TestStabilityValidator:
    """Test system stability validation."""
    
    @pytest.fixture
    def validator(self):
        """Create stability validator instance."""
        return StabilityValidator()
    
    @pytest.fixture
    def stability_results(self):
        """Generate sample stability test results."""
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "duration_hours": 48.5,
            "initial_memory": 100 * 1024 * 1024,  # 100 MB
            "final_memory": 105 * 1024 * 1024,  # 105 MB (5% growth)
            "initial_latency": 10.0,
            "final_latency": 11.0,
            "total_errors": 5,
            "total_operations": 10000,
            "error_rate": 0.0005,
            "avg_cpu_percent": 45.0,
            "peak_cpu_percent": 75.0,
            "uptime_percent": 99.95,
            "total_downtime": 86.4  # seconds
        }
    
    @pytest.mark.asyncio
    async def test_validate_no_test_results(self, validator):
        """Test validation with no stability test results."""
        with patch.object(validator, "_load_stability_test_results", return_value=None):
            result = await validator.validate()
            
            assert result["status"] == "failed"
            assert "Stability test not completed" in result["message"]
    
    @pytest.mark.asyncio
    async def test_validate_insufficient_duration(self, validator, stability_results):
        """Test validation with insufficient test duration."""
        stability_results["duration_hours"] = 24
        
        with patch.object(validator, "_load_stability_test_results", return_value=stability_results):
            result = await validator.validate()
            
            assert result["status"] == "failed"
            assert "only ran for 24.0 hours" in result["message"]
    
    @pytest.mark.asyncio
    async def test_validate_memory_leak(self, validator, stability_results):
        """Test detection of memory leaks."""
        stability_results["final_memory"] = 120 * 1024 * 1024  # 20% growth
        
        with patch.object(validator, "_load_stability_test_results", return_value=stability_results):
            result = await validator.validate()
            
            assert result["status"] == "failed"
            assert "Memory leak detected" in result["message"]
    
    @pytest.mark.asyncio
    async def test_validate_high_error_rate(self, validator, stability_results):
        """Test detection of high error rate."""
        stability_results["error_rate"] = 0.01  # 1% error rate
        
        with patch.object(validator, "_load_stability_test_results", return_value=stability_results):
            result = await validator.validate()
            
            assert result["status"] == "failed"
            assert "Error rate" in result["message"]
    
    @pytest.mark.asyncio
    async def test_validate_successful(self, validator, stability_results):
        """Test successful stability validation."""
        with patch.object(validator, "_load_stability_test_results", return_value=stability_results):
            result = await validator.validate()
            
            assert result["status"] == "passed"
            assert "System stable for" in result["message"]


class TestRiskValidator:
    """Test risk configuration validation."""
    
    @pytest.fixture
    def validator(self):
        """Create risk validator instance."""
        return RiskValidator()
    
    @pytest.mark.asyncio
    async def test_position_limits_validation(self, validator):
        """Test position limits validation."""
        result = await validator._check_position_limits()
        
        assert result["status"] in ["passed", "warning"]
        if result["status"] == "passed":
            assert "Position limits properly configured" in result["message"]
    
    @pytest.mark.asyncio
    async def test_drawdown_limits_validation(self, validator):
        """Test drawdown limits validation."""
        result = await validator._check_drawdown_limits()
        
        assert result["status"] == "passed"
        assert "sniper_drawdown" in result["evidence"]
        assert "hunter_drawdown" in result["evidence"]
        assert "strategist_drawdown" in result["evidence"]
    
    @pytest.mark.asyncio
    async def test_emergency_stops_validation(self, validator):
        """Test emergency stop mechanisms validation."""
        result = await validator._check_emergency_stops()
        
        assert result["status"] in ["passed", "warning"]
        assert "stop_losses_configured" in result["evidence"] or "script_exists" in result["evidence"]
    
    @pytest.mark.asyncio
    async def test_circuit_breakers_validation(self, validator):
        """Test circuit breaker validation."""
        result = await validator._check_circuit_breakers()
        
        assert result["status"] in ["passed", "warning"]
        if result["status"] == "passed":
            assert "config" in result["evidence"]
    
    @pytest.mark.asyncio
    async def test_validate_complete(self, validator):
        """Test complete risk validation."""
        with patch.object(validator, "_check_position_limits") as mock_position, \
             patch.object(validator, "_check_drawdown_limits") as mock_drawdown, \
             patch.object(validator, "_check_emergency_stops") as mock_emergency, \
             patch.object(validator, "_check_circuit_breakers") as mock_circuit:
            
            mock_position.return_value = {"status": "passed"}
            mock_drawdown.return_value = {"status": "passed"}
            mock_emergency.return_value = {"status": "passed"}
            mock_circuit.return_value = {"status": "passed"}
            
            result = await validator.validate()
            
            assert result["status"] in ["passed", "warning"]
            assert "Risk configuration" in result["message"]


class TestMetricsValidator:
    """Test trading metrics validation."""
    
    @pytest.fixture
    def validator(self):
        """Create metrics validator instance."""
        return MetricsValidator()
    
    @pytest.fixture
    def sample_metrics_data(self):
        """Generate sample trading data with good metrics."""
        trades = []
        for i in range(100):
            # 60% win rate with better profit/loss ratio for good Sharpe
            if i % 10 < 6:
                pnl = 150  # Win - larger wins
            else:
                pnl = -40  # Loss - smaller losses
            
            trades.append({
                "symbol": "BTC/USDT",
                "pnl": pnl,
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat()
            })
        return trades
    
    @pytest.mark.asyncio
    async def test_validate_no_data(self, validator):
        """Test validation with no trading data."""
        with patch.object(validator, "_load_trading_data", return_value=[]):
            result = await validator.validate()
            
            assert result["status"] == "failed"
            assert "No trading data available" in result["message"]
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self, validator, sample_metrics_data):
        """Test metrics calculation."""
        metrics = await validator._calculate_all_metrics(sample_metrics_data)
        
        assert metrics["total_trades"] == 100
        assert metrics["win_rate"] == Decimal("0.6")
        assert metrics["sharpe_ratio"] > 0
        assert metrics["profit_factor"] > 1
    
    @pytest.mark.asyncio
    async def test_validate_poor_metrics(self, validator):
        """Test validation with poor trading metrics."""
        # Create losing trades
        trades = [{"pnl": -100, "timestamp": datetime.now().isoformat()} for _ in range(100)]
        
        with patch.object(validator, "_load_trading_data", return_value=trades):
            result = await validator.validate()
            
            assert result["status"] == "failed"
            assert "Win rate" in result["message"] or "Sharpe ratio" in result["message"]
    
    @pytest.mark.asyncio
    async def test_validate_successful(self, validator, sample_metrics_data):
        """Test successful metrics validation."""
        with patch.object(validator, "_load_trading_data", return_value=sample_metrics_data):
            result = await validator.validate()
            
            assert result["status"] == "passed"
            assert "All trading metrics meet requirements" in result["message"]
            assert result["evidence"]["win_rate"] >= 0.55


class TestTierGateValidator:
    """Test tier progression validation."""
    
    @pytest.fixture
    def validator(self):
        """Create tier gate validator instance."""
        return TierGateValidator()
    
    @pytest.fixture
    def tier_state(self):
        """Generate sample tier state."""
        return {
            "current_tier": "sniper",
            "tier_start_date": (datetime.utcnow() - timedelta(days=10)).isoformat(),
            "capital": 2500,
            "total_profit": 600,
            "total_trades": 150,
            "win_rate": 0.58,
            "days_at_tier": 10
        }
    
    @pytest.mark.asyncio
    async def test_load_tier_state(self, validator):
        """Test loading tier state."""
        state = await validator._load_tier_state()
        
        assert "current_tier" in state
        assert "capital" in state
        assert state["current_tier"] in ["sniper", "hunter", "strategist"]
    
    @pytest.mark.asyncio
    async def test_progression_readiness(self, validator, tier_state):
        """Test tier progression readiness check."""
        with patch.object(validator, "_load_tier_state", return_value=tier_state):
            result = await validator._check_progression_readiness(tier_state)
            
            assert result["status"] in ["passed", "warning"]
            if result["status"] == "passed":
                assert "Ready for progression" in result["message"]
    
    @pytest.mark.asyncio
    async def test_demotion_triggers(self, validator, tier_state):
        """Test demotion trigger detection."""
        tier_state["current_tier"] = "hunter"
        tier_state["daily_loss_percent"] = 0.15  # 15% loss
        
        result = await validator._check_demotion_triggers(tier_state)
        
        assert result["status"] == "warning"
        assert "Demotion triggers detected" in result["message"]
    
    @pytest.mark.asyncio
    async def test_capital_requirements(self, validator, tier_state):
        """Test capital requirements validation."""
        result = await validator._check_capital_requirements(tier_state)
        
        assert result["status"] == "passed"
        assert "Capital requirements met" in result["message"]
    
    @pytest.mark.asyncio
    async def test_validate_complete(self, validator, tier_state):
        """Test complete tier validation."""
        with patch.object(validator, "_load_tier_state", return_value=tier_state):
            result = await validator.validate()
            
            assert result["status"] in ["passed", "warning"]
            assert "Tier validation complete" in result["message"]
            assert result["evidence"]["current_tier"] == "sniper"