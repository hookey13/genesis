"""Integration tests for business validation framework."""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from genesis.validation.business import (
    MetricsValidator,
    PaperTradingValidator,
    RiskValidator,
    StabilityValidator,
    TierGateValidator,
)


@pytest.fixture
def temp_genesis_dir():
    """Create temporary Genesis directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        genesis_dir = Path(tmpdir)
        
        # Create directory structure
        (genesis_dir / ".genesis" / "logs").mkdir(parents=True)
        (genesis_dir / ".genesis" / "data").mkdir(parents=True)
        (genesis_dir / ".genesis" / "state").mkdir(parents=True)
        (genesis_dir / ".genesis" / "tests").mkdir(parents=True)
        (genesis_dir / "config").mkdir(parents=True)
        
        yield genesis_dir


@pytest.fixture
def setup_paper_trading_data(temp_genesis_dir):
    """Setup paper trading test data."""
    # Create paper trading log
    trades = []
    total_pnl = Decimal("0")
    
    for i in range(200):
        # Generate trades to exceed $10k profit
        pnl = Decimal("75") if i % 3 != 0 else Decimal("-25")
        total_pnl += pnl
        
        trades.append({
            "symbol": "BTC/USDT",
            "side": "buy" if i % 2 == 0 else "sell",
            "quantity": 0.01,
            "entry_price": 50000,
            "exit_price": 50000 + float(pnl * 10),
            "pnl": float(pnl),
            "timestamp": (datetime.now() - timedelta(days=30-i%30, hours=i%24)).isoformat(),
            "duration": 3600
        })
    
    log_file = temp_genesis_dir / ".genesis" / "logs" / "paper_trading.json"
    with open(log_file, "w") as f:
        json.dump({"trades": trades, "total_pnl": float(total_pnl)}, f)
    
    return log_file, trades


@pytest.fixture
def setup_stability_results(temp_genesis_dir):
    """Setup stability test results."""
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "duration_hours": 48.2,
        "initial_memory": 100 * 1024 * 1024,
        "final_memory": 108 * 1024 * 1024,  # 8% growth
        "initial_latency": 10.0,
        "final_latency": 11.5,  # 15% degradation
        "total_errors": 10,
        "total_operations": 50000,
        "error_rate": 0.0002,  # 0.02% error rate
        "avg_cpu_percent": 55.0,
        "peak_cpu_percent": 78.0,
        "avg_memory": 104 * 1024 * 1024,
        "peak_memory": 110 * 1024 * 1024,
        "avg_latency": 10.8,
        "uptime_percent": 99.95,
        "total_downtime": 86.4
    }
    
    results_file = temp_genesis_dir / ".genesis" / "tests" / "stability_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f)
    
    return results_file, results


@pytest.fixture
def setup_tier_config(temp_genesis_dir):
    """Setup tier configuration files."""
    # Create tier gates config
    tier_gates = {
        "tier_gates": {
            "sniper_to_hunter": {
                "capital_required": 2000,
                "profit_required": 500,
                "trades_required": 100,
                "win_rate_required": 0.55,
                "days_at_tier": 7
            },
            "hunter_to_strategist": {
                "capital_required": 10000,
                "profit_required": 3000,
                "trades_required": 500,
                "win_rate_required": 0.60,
                "days_at_tier": 30
            }
        }
    }
    
    config_file = temp_genesis_dir / "config" / "tier_gates.yaml"
    with open(config_file, "w") as f:
        yaml.dump(tier_gates, f)
    
    # Create tier state
    tier_state = {
        "current_tier": "sniper",
        "tier_start_date": (datetime.utcnow() - timedelta(days=10)).isoformat(),
        "capital": 2500,
        "total_profit": 600,
        "total_trades": 150,
        "win_rate": 0.58,
        "daily_loss_percent": 0.02,
        "consecutive_loss_days": 0,
        "current_drawdown": 0.05,
        "tilt_score": 0.3,
        "emergency_stop_triggered": False
    }
    
    state_file = temp_genesis_dir / ".genesis" / "state" / "tier_state.json"
    with open(state_file, "w") as f:
        json.dump(tier_state, f)
    
    return config_file, state_file


class TestBusinessValidationIntegration:
    """Integration tests for complete business validation workflow."""
    
    @pytest.mark.asyncio
    async def test_paper_trading_validation_integration(
        self, temp_genesis_dir, setup_paper_trading_data
    ):
        """Test paper trading validation with real data files."""
        log_file, trades = setup_paper_trading_data
        
        validator = PaperTradingValidator()
        validator.trading_log = log_file
        
        result = await validator.validate()
        
        assert result["status"] == "passed"
        assert float(result["evidence"]["total_pnl"]) >= 10000
        assert result["evidence"]["trades"] == len(trades)
        assert "report" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_stability_validation_integration(
        self, temp_genesis_dir, setup_stability_results
    ):
        """Test stability validation with real test results."""
        results_file, results = setup_stability_results
        
        validator = StabilityValidator()
        validator.test_results_file = results_file
        
        result = await validator.validate()
        
        assert result["status"] == "passed"
        assert "48.2 hours" in result["message"]
        assert result["evidence"]["memory_growth"] == "8.0%"
        assert result["evidence"]["error_rate"] == "0.020%"
    
    @pytest.mark.asyncio
    async def test_risk_validation_integration(self, temp_genesis_dir):
        """Test risk validation with configuration files."""
        # Create trading rules config
        trading_rules = {
            "risk_limits": {
                "max_position_size": 1000,
                "max_daily_loss": 200,
                "max_drawdown": 0.20
            }
        }
        
        config_file = temp_genesis_dir / "config" / "trading_rules.yaml"
        with open(config_file, "w") as f:
            yaml.dump(trading_rules, f)
        
        validator = RiskValidator()
        validator.config_file = config_file
        
        result = await validator.validate()
        
        assert result["status"] in ["passed", "warning"]
        assert "Risk configuration" in result["message"]
        assert "report" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_metrics_validation_integration(self, temp_genesis_dir):
        """Test metrics validation with trading data."""
        # Create metrics file
        trades = []
        for i in range(150):
            # 60% win rate with good metrics
            if i % 10 < 6:
                pnl = 120
            else:
                pnl = -60
            
            trades.append({
                "symbol": "BTC/USDT",
                "pnl": pnl,
                "timestamp": (datetime.now() - timedelta(hours=i*2)).isoformat()
            })
        
        metrics_file = temp_genesis_dir / ".genesis" / "data" / "trading_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({"trades": trades}, f)
        
        validator = MetricsValidator()
        validator.metrics_file = metrics_file
        
        result = await validator.validate()
        
        assert result["status"] == "passed"
        assert result["evidence"]["win_rate"] >= 0.55
        assert result["evidence"]["sharpe_ratio"] > 0
        assert "report" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_tier_validation_integration(
        self, temp_genesis_dir, setup_tier_config
    ):
        """Test tier gate validation with configuration."""
        config_file, state_file = setup_tier_config
        
        validator = TierGateValidator()
        validator.tier_config = config_file
        validator.tier_state = state_file
        
        result = await validator.validate()
        
        assert result["status"] in ["passed", "warning"]
        assert result["evidence"]["current_tier"] == "sniper"
        assert "report" in result["metadata"]
        
        # Check report structure
        report = result["metadata"]["report"]
        assert "current_tier" in report
        assert "performance" in report
        assert "risk_status" in report
        assert "available_features" in report
    
    @pytest.mark.asyncio
    async def test_complete_validation_workflow(
        self, temp_genesis_dir, 
        setup_paper_trading_data,
        setup_stability_results,
        setup_tier_config
    ):
        """Test complete business validation workflow."""
        # Setup all validators
        paper_validator = PaperTradingValidator()
        paper_validator.trading_log = setup_paper_trading_data[0]
        
        stability_validator = StabilityValidator()
        stability_validator.test_results_file = setup_stability_results[0]
        
        risk_validator = RiskValidator()
        
        metrics_validator = MetricsValidator()
        
        tier_validator = TierGateValidator()
        tier_validator.tier_config = setup_tier_config[0]
        tier_validator.tier_state = setup_tier_config[1]
        
        # Run all validators
        results = {}
        
        results["paper_trading"] = await paper_validator.validate()
        results["stability"] = await stability_validator.validate()
        results["risk"] = await risk_validator.validate()
        results["metrics"] = await metrics_validator.validate()
        results["tier"] = await tier_validator.validate()
        
        # Check overall results
        assert all(r["status"] in ["passed", "warning"] for r in results.values())
        
        # Generate summary report
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "validators_run": len(results),
            "validators_passed": sum(1 for r in results.values() if r["status"] == "passed"),
            "validators_warning": sum(1 for r in results.values() if r["status"] == "warning"),
            "validators_failed": sum(1 for r in results.values() if r["status"] == "failed"),
            "details": results
        }
        
        assert summary["validators_run"] == 5
        assert summary["validators_failed"] == 0
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, temp_genesis_dir):
        """Test error handling in validators."""
        # Test with corrupted data file
        bad_file = temp_genesis_dir / ".genesis" / "logs" / "paper_trading.json"
        with open(bad_file, "w") as f:
            f.write("not valid json{")
        
        validator = PaperTradingValidator()
        validator.trading_log = bad_file
        
        result = await validator.validate()
        
        assert result["status"] == "failed"
        assert "error" in result["evidence"] or "No paper trading history" in result["message"]
    
    @pytest.mark.asyncio
    async def test_concurrent_validation(
        self, temp_genesis_dir,
        setup_paper_trading_data,
        setup_stability_results,
        setup_tier_config
    ):
        """Test running multiple validators concurrently."""
        validators = [
            PaperTradingValidator(),
            StabilityValidator(),
            RiskValidator(),
            MetricsValidator(),
            TierGateValidator()
        ]
        
        # Configure validators
        validators[0].trading_log = setup_paper_trading_data[0]
        validators[1].test_results_file = setup_stability_results[0]
        validators[4].tier_config = setup_tier_config[0]
        validators[4].tier_state = setup_tier_config[1]
        
        # Run concurrently
        tasks = [v.validate() for v in validators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        assert len(results) == 5
        for result in results:
            if not isinstance(result, Exception):
                assert result["status"] in ["passed", "warning", "failed"]
                assert "message" in result
                assert "evidence" in result