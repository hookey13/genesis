"""Risk engine validation for Genesis trading system."""

import asyncio
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from . import BaseValidator, ValidationIssue, ValidationSeverity


class RiskValidator(BaseValidator):
    """Validates risk engine configuration and enforcement."""
    
    @property
    def name(self) -> str:
        return "risk"
    
    @property
    def description(self) -> str:
        return "Validates risk engine, position limits, and safety mechanisms"
    
    async def _validate(self, mode: str):
        """Perform risk validation."""
        # Check risk configuration
        await self._validate_risk_config()
        
        # Validate position limits by tier
        await self._validate_position_limits()
        
        # Test stop-loss enforcement
        await self._validate_stop_loss()
        
        # Validate Kelly criterion sizing
        await self._validate_kelly_sizing()
        
        # Check correlation limits
        if mode in ["standard", "thorough"]:
            await self._validate_correlation_limits()
        
        # Test circuit breakers
        if mode == "thorough":
            await self._test_circuit_breakers()
    
    async def _validate_risk_config(self):
        """Validate risk configuration files."""
        config_path = Path("genesis/config/trading_rules.yaml")
        
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    risk_config = yaml.safe_load(f)
                
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Risk configuration loaded: {config_path}"
                ))
                
                # Validate tier configurations
                tiers = ["sniper", "hunter", "strategist"]
                for tier in tiers:
                    if tier in risk_config:
                        await self._validate_tier_config(tier, risk_config[tier])
                    else:
                        self.result.add_issue(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            message=f"Missing risk configuration for {tier} tier",
                            recommendation=f"Add {tier} configuration to trading_rules.yaml"
                        ))
                
            except yaml.YAMLError as e:
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    message="Invalid YAML in risk configuration",
                    details={"error": str(e)},
                    recommendation="Fix YAML syntax in trading_rules.yaml"
                ))
        else:
            # Create default risk configuration
            default_config = self._get_default_risk_config()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Created default risk configuration: {config_path}"
            ))
    
    async def _validate_tier_config(self, tier: str, config: Dict[str, Any]):
        """Validate tier-specific risk configuration."""
        required_fields = [
            "max_positions",
            "max_position_size",
            "max_daily_loss",
            "max_drawdown",
            "stop_loss_percent",
            "take_profit_percent",
            "max_leverage",
            "min_capital",
            "max_capital"
        ]
        
        for field in required_fields:
            self.check_condition(
                field in config,
                f"{tier}: Required field '{field}' present",
                f"{tier}: Missing required field '{field}'",
                ValidationSeverity.ERROR,
                details={"tier": tier, "field": field}
            )
        
        # Validate value ranges
        if "max_positions" in config:
            expected_max = {"sniper": 1, "hunter": 3, "strategist": 10}
            self.check_condition(
                config["max_positions"] == expected_max.get(tier, 1),
                f"{tier}: Correct max_positions ({config['max_positions']})",
                f"{tier}: Invalid max_positions ({config['max_positions']} != {expected_max.get(tier)})",
                ValidationSeverity.ERROR,
                details={"tier": tier, "actual": config["max_positions"], "expected": expected_max.get(tier)}
            )
        
        if "max_leverage" in config:
            self.check_threshold(
                config["max_leverage"],
                3,
                "<=",
                f"{tier}: Max leverage",
                "x",
                ValidationSeverity.WARNING
            )
        
        if "max_daily_loss" in config:
            self.check_threshold(
                config["max_daily_loss"],
                0.05,
                "<=",
                f"{tier}: Max daily loss",
                "",
                ValidationSeverity.INFO
            )
    
    async def _validate_position_limits(self):
        """Validate position limits enforcement."""
        try:
            from genesis.engine.risk_engine import RiskEngine
            from genesis.config.settings import Settings
            
            settings = Settings()
            risk_engine = RiskEngine(settings)
            
            # Test position limits for each tier
            test_cases = [
                ("sniper", 1, 500, True),
                ("sniper", 2, 500, False),  # Should fail - too many positions
                ("hunter", 3, 3000, True),
                ("hunter", 4, 3000, False),  # Should fail - too many positions
                ("strategist", 10, 15000, True),
                ("strategist", 11, 15000, False),  # Should fail - too many positions
            ]
            
            for tier, positions, capital, should_pass in test_cases:
                result = risk_engine.check_position_limit(tier, positions, capital)
                
                if should_pass:
                    self.check_condition(
                        result,
                        f"{tier}: Position limit check passed ({positions} positions, ${capital})",
                        f"{tier}: Position limit check failed unexpectedly",
                        ValidationSeverity.ERROR,
                        details={"tier": tier, "positions": positions, "capital": capital}
                    )
                else:
                    self.check_condition(
                        not result,
                        f"{tier}: Position limit correctly rejected ({positions} positions)",
                        f"{tier}: Position limit not enforced properly",
                        ValidationSeverity.CRITICAL,
                        details={"tier": tier, "positions": positions, "capital": capital},
                        recommendation="Fix position limit enforcement in RiskEngine"
                    )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="RiskEngine not implemented yet",
                recommendation="Implement genesis/engine/risk_engine.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Position limit validation failed",
                details={"error": str(e)},
                recommendation="Fix RiskEngine implementation"
            ))
    
    async def _validate_stop_loss(self):
        """Validate stop-loss enforcement."""
        try:
            from genesis.engine.risk_engine import RiskEngine
            from genesis.core.models import Position
            from decimal import Decimal
            
            risk_engine = RiskEngine()
            
            # Test stop-loss calculation
            test_positions = [
                Position(
                    symbol="BTC/USDT",
                    entry_price=Decimal("50000"),
                    quantity=Decimal("0.1"),
                    stop_loss_percent=Decimal("0.02")
                ),
                Position(
                    symbol="ETH/USDT",
                    entry_price=Decimal("3000"),
                    quantity=Decimal("1.0"),
                    stop_loss_percent=Decimal("0.03")
                )
            ]
            
            for position in test_positions:
                stop_loss_price = risk_engine.calculate_stop_loss(position)
                expected = position.entry_price * (Decimal("1") - position.stop_loss_percent)
                
                self.check_condition(
                    abs(stop_loss_price - expected) < Decimal("0.01"),
                    f"Stop-loss calculation correct for {position.symbol}",
                    f"Stop-loss calculation error for {position.symbol}",
                    ValidationSeverity.ERROR,
                    details={
                        "symbol": position.symbol,
                        "calculated": float(stop_loss_price),
                        "expected": float(expected)
                    }
                )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Stop-loss components not fully implemented",
                recommendation="Complete Position model and RiskEngine stop-loss methods"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Stop-loss validation failed",
                details={"error": str(e)}
            ))
    
    async def _validate_kelly_sizing(self):
        """Validate Kelly criterion position sizing."""
        try:
            from genesis.engine.risk_engine import KellyCriterion
            
            kelly = KellyCriterion()
            
            # Test Kelly sizing with different win rates and ratios
            test_cases = [
                (0.60, 2.0, 0.20),  # 60% win rate, 2:1 ratio = 20% Kelly
                (0.55, 1.5, 0.117),  # 55% win rate, 1.5:1 ratio = 11.7% Kelly
                (0.45, 2.0, 0.0),    # 45% win rate, 2:1 ratio = 0% (don't trade)
            ]
            
            for win_rate, win_loss_ratio, expected_kelly in test_cases:
                kelly_fraction = kelly.calculate(win_rate, win_loss_ratio)
                
                self.check_condition(
                    abs(kelly_fraction - expected_kelly) < 0.01,
                    f"Kelly sizing correct: {win_rate:.0%} WR, {win_loss_ratio}:1 = {kelly_fraction:.1%}",
                    f"Kelly sizing error: got {kelly_fraction:.1%}, expected {expected_kelly:.1%}",
                    ValidationSeverity.ERROR,
                    details={
                        "win_rate": win_rate,
                        "win_loss_ratio": win_loss_ratio,
                        "calculated": kelly_fraction,
                        "expected": expected_kelly
                    }
                )
            
            # Test Kelly with safety factor
            kelly_safe = kelly.calculate(0.60, 2.0, safety_factor=0.25)
            expected_safe = 0.20 * 0.25
            
            self.check_condition(
                abs(kelly_safe - expected_safe) < 0.01,
                f"Kelly with safety factor correct: {kelly_safe:.1%}",
                f"Kelly safety factor error",
                ValidationSeverity.WARNING,
                details={"calculated": kelly_safe, "expected": expected_safe}
            )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Kelly criterion not implemented yet",
                recommendation="Implement KellyCriterion in risk_engine.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Kelly sizing validation error",
                details={"error": str(e)}
            ))
    
    async def _validate_correlation_limits(self):
        """Validate correlation limits between positions."""
        try:
            from genesis.engine.risk_engine import CorrelationManager
            import numpy as np
            
            correlation_mgr = CorrelationManager()
            
            # Test correlation calculation
            test_pairs = [
                (["BTC/USDT", "ETH/USDT"], 0.8),  # High correlation
                (["BTC/USDT", "DOGE/USDT"], 0.5),  # Medium correlation
                (["BTC/USDT", "USDT/USD"], -0.2),  # Negative correlation
            ]
            
            for pairs, expected_corr in test_pairs:
                # This would normally fetch real data
                correlation = correlation_mgr.calculate_correlation(pairs[0], pairs[1])
                
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Correlation {pairs[0]}/{pairs[1]}: {correlation:.2f}",
                    details={"pair1": pairs[0], "pair2": pairs[1], "correlation": correlation}
                ))
                
                # Check if correlation limit is enforced
                max_correlation = 0.7
                if abs(correlation) > max_correlation:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High correlation detected: {pairs[0]}/{pairs[1]} = {correlation:.2f}",
                        details={"pairs": pairs, "correlation": correlation, "limit": max_correlation},
                        recommendation="Consider position sizing adjustment for correlated pairs"
                    ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Correlation manager not implemented",
                recommendation="Implement CorrelationManager for advanced risk management"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Correlation validation error",
                details={"error": str(e)}
            ))
    
    async def _test_circuit_breakers(self):
        """Test circuit breaker mechanisms."""
        try:
            from genesis.exchange.circuit_breaker import CircuitBreaker
            
            # Test different circuit breaker scenarios
            breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                half_open_requests=2
            )
            
            # Simulate failures
            for i in range(5):
                breaker.record_failure()
            
            self.check_condition(
                breaker.is_open(),
                "Circuit breaker opens after threshold failures",
                "Circuit breaker failed to open",
                ValidationSeverity.CRITICAL,
                recommendation="Fix circuit breaker logic"
            )
            
            # Test half-open state
            await asyncio.sleep(0.1)  # Simulate time passing
            breaker.record_success()
            
            self.check_condition(
                breaker.is_half_open(),
                "Circuit breaker transitions to half-open",
                "Circuit breaker half-open transition failed",
                ValidationSeverity.ERROR
            )
            
            # Test recovery
            breaker.record_success()
            breaker.record_success()
            
            self.check_condition(
                breaker.is_closed(),
                "Circuit breaker recovers after successful requests",
                "Circuit breaker recovery failed",
                ValidationSeverity.ERROR
            )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="Circuit breaker not implemented",
                recommendation="Implement genesis/exchange/circuit_breaker.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Circuit breaker test failed",
                details={"error": str(e)}
            ))
    
    def _get_default_risk_config(self) -> Dict[str, Any]:
        """Get default risk configuration."""
        return {
            "sniper": {
                "max_positions": 1,
                "max_position_size": 0.10,  # 10% of capital
                "max_daily_loss": 0.02,  # 2% daily loss limit
                "max_drawdown": 0.05,  # 5% max drawdown
                "stop_loss_percent": 0.02,  # 2% stop loss
                "take_profit_percent": 0.04,  # 4% take profit
                "max_leverage": 1,  # No leverage
                "min_capital": 500,
                "max_capital": 2000,
                "risk_per_trade": 0.01  # 1% risk per trade
            },
            "hunter": {
                "max_positions": 3,
                "max_position_size": 0.15,
                "max_daily_loss": 0.03,
                "max_drawdown": 0.08,
                "stop_loss_percent": 0.025,
                "take_profit_percent": 0.05,
                "max_leverage": 2,
                "min_capital": 2000,
                "max_capital": 10000,
                "risk_per_trade": 0.015,
                "max_correlated_positions": 2,
                "max_correlation": 0.7
            },
            "strategist": {
                "max_positions": 10,
                "max_position_size": 0.20,
                "max_daily_loss": 0.05,
                "max_drawdown": 0.10,
                "stop_loss_percent": 0.03,
                "take_profit_percent": 0.06,
                "max_leverage": 3,
                "min_capital": 10000,
                "max_capital": None,
                "risk_per_trade": 0.02,
                "max_correlated_positions": 5,
                "max_correlation": 0.8,
                "use_kelly_sizing": True,
                "kelly_safety_factor": 0.25
            },
            "global": {
                "emergency_stop_loss": 0.10,  # 10% emergency stop
                "max_daily_trades": 200,
                "min_trade_size_usdt": 10,
                "max_slippage_percent": 0.005,
                "require_stop_loss": True,
                "allow_market_orders": True,
                "allow_leverage": False  # Disabled until Strategist
            }
        }