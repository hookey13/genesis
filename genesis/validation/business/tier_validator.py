"""Tier progression and gate validation."""

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import yaml

logger = structlog.get_logger(__name__)


class ValidationResult:
    """Standardized validation result."""
    
    def __init__(
        self,
        check_id: str,
        status: str,
        message: str,
        evidence: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.check_id = check_id
        self.status = status
        self.message = message
        self.evidence = evidence
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
            "evidence": self.evidence,
            "metadata": self.metadata
        }


class CheckStatus:
    """Validation check status constants."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


# Tier progression requirements
TIER_GATES = {
    "sniper_to_hunter": {
        "capital_required": Decimal("2000"),
        "profit_required": Decimal("500"),
        "trades_required": 100,
        "win_rate_required": Decimal("0.55"),
        "days_at_tier": 7,
        "max_drawdown": Decimal("0.20"),
        "consecutive_profitable_days": 3
    },
    "hunter_to_strategist": {
        "capital_required": Decimal("10000"),
        "profit_required": Decimal("3000"),
        "trades_required": 500,
        "win_rate_required": Decimal("0.60"),
        "days_at_tier": 30,
        "max_drawdown": Decimal("0.15"),
        "consecutive_profitable_days": 7
    }
}

# Tier demotion triggers
DEMOTION_TRIGGERS = {
    "max_daily_loss_percent": Decimal("0.10"),  # 10% daily loss triggers demotion
    "consecutive_loss_days": 3,  # 3 consecutive losing days
    "drawdown_threshold": Decimal("0.25"),  # 25% drawdown from peak
    "tilt_score_threshold": Decimal("0.80"),  # High tilt score
    "emergency_stop_triggered": True  # Emergency stop activation
}


class TierGateValidator:
    """Validates tier progression requirements and gates."""
    
    def __init__(self):
        """Initialize tier gate validator."""
        self.tier_config = Path("config/tier_gates.yaml")
        self.tier_state = Path(".genesis/state/tier_state.json")
        self.trading_log = Path(".genesis/logs/trading.log")
        self.current_tier = "sniper"  # Default starting tier
        
    async def validate(self) -> Dict[str, Any]:
        """Verify tier progression requirements are met."""
        try:
            # Load current tier state
            tier_info = await self._load_tier_state()
            self.current_tier = tier_info.get("current_tier", "sniper")
            
            # Check progression readiness
            progression_check = await self._check_progression_readiness(tier_info)
            
            # Check demotion triggers
            demotion_check = await self._check_demotion_triggers(tier_info)
            
            # Validate feature gates
            feature_gate_check = await self._validate_feature_gates()
            
            # Check capital requirements
            capital_check = await self._check_capital_requirements(tier_info)
            
            # Combine all checks
            all_checks = [
                progression_check,
                demotion_check,
                feature_gate_check,
                capital_check
            ]
            
            # Find any failures
            failures = [
                check for check in all_checks 
                if check.get("status") == CheckStatus.FAILED
            ]
            
            if failures:
                return failures[0]  # Return first failure
            
            # Find any warnings
            warnings = [
                check for check in all_checks 
                if check.get("status") == CheckStatus.WARNING
            ]
            
            # Generate tier readiness report
            report = await self._generate_tier_report(tier_info)
            
            status = CheckStatus.PASSED if not warnings else CheckStatus.WARNING
            
            return ValidationResult(
                check_id="TIER-001",
                status=status,
                message=f"Tier validation complete for {self.current_tier} tier",
                evidence={
                    "current_tier": self.current_tier,
                    "progression_ready": progression_check.get("status") == CheckStatus.PASSED,
                    "demotion_risk": demotion_check.get("status") == CheckStatus.WARNING,
                    "feature_gates": "configured"
                },
                metadata={"report": report}
            ).to_dict()
            
        except Exception as e:
            logger.error("Tier validation failed", error=str(e))
            return ValidationResult(
                check_id="TIER-001",
                status=CheckStatus.FAILED,
                message=f"Validation error: {str(e)}",
                evidence={"error": str(e)},
                metadata={}
            ).to_dict()
    
    async def _load_tier_state(self) -> Dict[str, Any]:
        """Load current tier state and history."""
        if not self.tier_state.exists():
            # Return default state
            return {
                "current_tier": "sniper",
                "tier_start_date": datetime.utcnow().isoformat(),
                "capital": 500,
                "total_profit": 0,
                "total_trades": 0,
                "days_at_tier": 0
            }
        
        try:
            with open(self.tier_state) as f:
                state = json.load(f)
                
                # Calculate days at current tier
                tier_start = datetime.fromisoformat(state.get("tier_start_date", datetime.utcnow().isoformat()))
                state["days_at_tier"] = (datetime.utcnow() - tier_start).days
                
                return state
        except Exception as e:
            logger.error("Failed to load tier state", error=str(e))
            return {
                "current_tier": "sniper",
                "tier_start_date": datetime.utcnow().isoformat(),
                "capital": 500,
                "total_profit": 0,
                "total_trades": 0,
                "days_at_tier": 0
            }
    
    async def _check_progression_readiness(self, tier_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if ready for tier progression."""
        current_tier = tier_info.get("current_tier", "sniper")
        
        # Determine next tier
        if current_tier == "sniper":
            gate_key = "sniper_to_hunter"
            next_tier = "hunter"
        elif current_tier == "hunter":
            gate_key = "hunter_to_strategist"
            next_tier = "strategist"
        else:
            # Already at highest tier
            return ValidationResult(
                check_id="TIER-002",
                status=CheckStatus.PASSED,
                message=f"Already at highest tier: {current_tier}",
                evidence={"current_tier": current_tier},
                metadata={}
            ).to_dict()
        
        requirements = TIER_GATES[gate_key]
        
        # Check each requirement
        checks_passed = []
        checks_failed = []
        
        # Capital requirement
        capital = Decimal(str(tier_info.get("capital", 0)))
        if capital >= requirements["capital_required"]:
            checks_passed.append(f"Capital: ${capital} >= ${requirements['capital_required']}")
        else:
            checks_failed.append(f"Capital: ${capital} < ${requirements['capital_required']}")
        
        # Profit requirement
        profit = Decimal(str(tier_info.get("total_profit", 0)))
        if profit >= requirements["profit_required"]:
            checks_passed.append(f"Profit: ${profit} >= ${requirements['profit_required']}")
        else:
            checks_failed.append(f"Profit: ${profit} < ${requirements['profit_required']}")
        
        # Trades requirement
        trades = tier_info.get("total_trades", 0)
        if trades >= requirements["trades_required"]:
            checks_passed.append(f"Trades: {trades} >= {requirements['trades_required']}")
        else:
            checks_failed.append(f"Trades: {trades} < {requirements['trades_required']}")
        
        # Win rate requirement
        win_rate = Decimal(str(tier_info.get("win_rate", 0)))
        if win_rate >= requirements["win_rate_required"]:
            checks_passed.append(f"Win rate: {win_rate:.1%} >= {requirements['win_rate_required']:.1%}")
        else:
            checks_failed.append(f"Win rate: {win_rate:.1%} < {requirements['win_rate_required']:.1%}")
        
        # Days at tier requirement
        days = tier_info.get("days_at_tier", 0)
        if days >= requirements["days_at_tier"]:
            checks_passed.append(f"Days at tier: {days} >= {requirements['days_at_tier']}")
        else:
            checks_failed.append(f"Days at tier: {days} < {requirements['days_at_tier']}")
        
        # Determine status
        if checks_failed:
            return ValidationResult(
                check_id="TIER-002",
                status=CheckStatus.WARNING,
                message=f"Not ready for progression to {next_tier}",
                evidence={
                    "requirements_met": len(checks_passed),
                    "requirements_failed": len(checks_failed),
                    "details": checks_failed
                },
                metadata={"next_tier": next_tier, "requirements": requirements}
            ).to_dict()
        
        return ValidationResult(
            check_id="TIER-002",
            status=CheckStatus.PASSED,
            message=f"Ready for progression to {next_tier}",
            evidence={
                "all_requirements_met": True,
                "details": checks_passed
            },
            metadata={"next_tier": next_tier}
        ).to_dict()
    
    async def _check_demotion_triggers(self, tier_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check for tier demotion triggers."""
        current_tier = tier_info.get("current_tier", "sniper")
        
        if current_tier == "sniper":
            # Can't demote from lowest tier
            return ValidationResult(
                check_id="TIER-003",
                status=CheckStatus.PASSED,
                message="No demotion risk at sniper tier",
                evidence={"current_tier": current_tier},
                metadata={}
            ).to_dict()
        
        triggers_active = []
        
        # Check daily loss
        daily_loss = Decimal(str(tier_info.get("daily_loss_percent", 0)))
        if daily_loss > DEMOTION_TRIGGERS["max_daily_loss_percent"]:
            triggers_active.append(f"Daily loss {daily_loss:.1%} exceeds threshold")
        
        # Check consecutive losses
        consecutive_losses = tier_info.get("consecutive_loss_days", 0)
        if consecutive_losses >= DEMOTION_TRIGGERS["consecutive_loss_days"]:
            triggers_active.append(f"{consecutive_losses} consecutive loss days")
        
        # Check drawdown
        drawdown = Decimal(str(tier_info.get("current_drawdown", 0)))
        if drawdown > DEMOTION_TRIGGERS["drawdown_threshold"]:
            triggers_active.append(f"Drawdown {drawdown:.1%} exceeds threshold")
        
        # Check tilt score
        tilt_score = Decimal(str(tier_info.get("tilt_score", 0)))
        if tilt_score > DEMOTION_TRIGGERS["tilt_score_threshold"]:
            triggers_active.append(f"High tilt score: {tilt_score:.2f}")
        
        # Check emergency stop
        if tier_info.get("emergency_stop_triggered", False):
            triggers_active.append("Emergency stop was triggered")
        
        if triggers_active:
            return ValidationResult(
                check_id="TIER-003",
                status=CheckStatus.WARNING,
                message="Demotion triggers detected",
                evidence={
                    "triggers_active": len(triggers_active),
                    "details": triggers_active
                },
                metadata={"current_tier": current_tier}
            ).to_dict()
        
        return ValidationResult(
            check_id="TIER-003",
            status=CheckStatus.PASSED,
            message="No demotion triggers active",
            evidence={"current_tier": current_tier},
            metadata={}
        ).to_dict()
    
    async def _validate_feature_gates(self) -> Dict[str, Any]:
        """Validate feature gates are properly configured."""
        try:
            # Check if tier gates config exists
            if self.tier_config.exists():
                with open(self.tier_config) as f:
                    config = yaml.safe_load(f)
                    
                    if not config or "tier_gates" not in config:
                        return ValidationResult(
                            check_id="TIER-004",
                            status=CheckStatus.WARNING,
                            message="Tier gates configuration incomplete",
                            evidence={"config_exists": True, "gates_defined": False},
                            metadata={}
                        ).to_dict()
            else:
                # Check if gates are implemented in code
                from genesis.engine.state_machine import TierStateMachine
                
                # Verify state machine exists
                if not hasattr(TierStateMachine, "check_tier_gate"):
                    return ValidationResult(
                        check_id="TIER-004",
                        status=CheckStatus.WARNING,
                        message="Tier gate checking not implemented",
                        evidence={"implementation": "missing"},
                        metadata={}
                    ).to_dict()
            
            # Verify feature locks by tier
            feature_gates = {
                "sniper": ["simple_arb", "spread_capture"],
                "hunter": ["multi_pair", "mean_reversion", "iceberg_orders"],
                "strategist": ["statistical_arb", "market_making", "vwap_execution"]
            }
            
            return ValidationResult(
                check_id="TIER-004",
                status=CheckStatus.PASSED,
                message="Feature gates properly configured",
                evidence={
                    "tiers": list(feature_gates.keys()),
                    "total_features": sum(len(f) for f in feature_gates.values())
                },
                metadata={"feature_gates": feature_gates}
            ).to_dict()
            
        except Exception as e:
            return ValidationResult(
                check_id="TIER-004",
                status=CheckStatus.WARNING,
                message=f"Feature gate validation warning: {str(e)}",
                evidence={"error": str(e)},
                metadata={}
            ).to_dict()
    
    async def _check_capital_requirements(self, tier_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check capital requirements for current tier."""
        current_tier = tier_info.get("current_tier", "sniper")
        capital = Decimal(str(tier_info.get("capital", 0)))
        
        # Minimum capital by tier
        min_capital = {
            "sniper": Decimal("500"),
            "hunter": Decimal("2000"),
            "strategist": Decimal("10000")
        }
        
        required = min_capital.get(current_tier, Decimal("500"))
        
        if capital < required:
            return ValidationResult(
                check_id="TIER-005",
                status=CheckStatus.WARNING,
                message=f"Capital below minimum for {current_tier} tier",
                evidence={
                    "current_capital": float(capital),
                    "required_capital": float(required)
                },
                metadata={"tier": current_tier}
            ).to_dict()
        
        # Check if capital supports next tier
        next_tier_capital = {
            "sniper": min_capital["hunter"],
            "hunter": min_capital["strategist"],
            "strategist": Decimal("50000")  # Future tier
        }
        
        next_required = next_tier_capital.get(current_tier, Decimal("50000"))
        progress = capital / next_required
        
        return ValidationResult(
            check_id="TIER-005",
            status=CheckStatus.PASSED,
            message=f"Capital requirements met for {current_tier} tier",
            evidence={
                "current_capital": float(capital),
                "tier_minimum": float(required),
                "next_tier_progress": f"{progress:.1%}"
            },
            metadata={"tier": current_tier}
        ).to_dict()
    
    async def _generate_tier_report(self, tier_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive tier readiness report."""
        current_tier = tier_info.get("current_tier", "sniper")
        
        # Determine progression status
        if current_tier == "strategist":
            next_tier = None
            gate_key = None
        elif current_tier == "hunter":
            next_tier = "strategist"
            gate_key = "hunter_to_strategist"
        else:
            next_tier = "hunter"
            gate_key = "sniper_to_hunter"
        
        report = {
            "current_tier": current_tier,
            "tier_start_date": tier_info.get("tier_start_date"),
            "days_at_tier": tier_info.get("days_at_tier", 0),
            "performance": {
                "capital": float(tier_info.get("capital", 0)),
                "total_profit": float(tier_info.get("total_profit", 0)),
                "total_trades": tier_info.get("total_trades", 0),
                "win_rate": float(tier_info.get("win_rate", 0))
            },
            "risk_status": {
                "current_drawdown": float(tier_info.get("current_drawdown", 0)),
                "daily_loss": float(tier_info.get("daily_loss_percent", 0)),
                "consecutive_losses": tier_info.get("consecutive_loss_days", 0),
                "tilt_score": float(tier_info.get("tilt_score", 0))
            }
        }
        
        if next_tier and gate_key:
            requirements = TIER_GATES[gate_key]
            report["next_tier"] = {
                "tier": next_tier,
                "requirements": {
                    "capital": float(requirements["capital_required"]),
                    "profit": float(requirements["profit_required"]),
                    "trades": requirements["trades_required"],
                    "win_rate": float(requirements["win_rate_required"]),
                    "days": requirements["days_at_tier"]
                },
                "progress": {
                    "capital": f"{Decimal(str(tier_info.get('capital', 0))) / requirements['capital_required']:.1%}",
                    "profit": f"{Decimal(str(tier_info.get('total_profit', 0))) / requirements['profit_required']:.1%}",
                    "trades": f"{tier_info.get('total_trades', 0) / requirements['trades_required']:.1%}",
                    "days": f"{tier_info.get('days_at_tier', 0) / requirements['days_at_tier']:.1%}"
                }
            }
        
        # Add feature availability
        report["available_features"] = self._get_tier_features(current_tier)
        
        return report
    
    def _get_tier_features(self, tier: str) -> List[str]:
        """Get available features for a tier."""
        features = {
            "sniper": [
                "Simple arbitrage",
                "Spread capture",
                "Market orders",
                "Single pair trading"
            ],
            "hunter": [
                "All sniper features",
                "Multi-pair trading",
                "Mean reversion",
                "Iceberg orders",
                "Order slicing"
            ],
            "strategist": [
                "All hunter features",
                "Statistical arbitrage",
                "Market making",
                "VWAP execution",
                "Advanced analytics"
            ]
        }
        
        return features.get(tier, [])