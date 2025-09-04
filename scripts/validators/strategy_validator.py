"""Strategy validation for Genesis trading system."""

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from . import BaseValidator, ValidationIssue, ValidationSeverity


class StrategyValidator(BaseValidator):
    """Validates strategy configuration and implementation."""
    
    @property
    def name(self) -> str:
        return "strategy"
    
    @property
    def description(self) -> str:
        return "Validates all tier strategies are properly configured and operational"
    
    async def _validate(self, mode: str):
        """Perform strategy validation."""
        # Check strategy directory structure
        await self._check_directory_structure()
        
        # Validate strategy configurations
        await self._validate_strategy_configs()
        
        # Check tier strategies
        await self._validate_tier_strategies()
        
        # Test strategy initialization
        if mode in ["standard", "thorough"]:
            await self._test_strategy_initialization()
        
        # Validate strategy state machines
        if mode == "thorough":
            await self._validate_state_machines()
    
    async def _check_directory_structure(self):
        """Check if strategy directories exist and are properly structured."""
        base_path = Path("genesis/strategies")
        
        # Check base directory
        self.check_condition(
            base_path.exists(),
            f"Strategy base directory exists: {base_path}",
            f"Strategy base directory not found: {base_path}",
            ValidationSeverity.CRITICAL,
            recommendation="Ensure genesis/strategies directory exists"
        )
        
        if not base_path.exists():
            return
        
        # Check tier directories
        tiers = ["sniper", "hunter", "strategist"]
        for tier in tiers:
            tier_path = base_path / tier
            self.check_condition(
                tier_path.exists(),
                f"Tier directory exists: {tier_path}",
                f"Tier directory missing: {tier_path}",
                ValidationSeverity.ERROR,
                details={"tier": tier},
                recommendation=f"Create {tier_path} directory for {tier} strategies"
            )
            
            if tier_path.exists():
                # Check for __init__.py
                init_path = tier_path / "__init__.py"
                self.check_condition(
                    init_path.exists(),
                    f"Tier init file exists: {init_path}",
                    f"Tier init file missing: {init_path}",
                    ValidationSeverity.WARNING,
                    details={"tier": tier}
                )
                
                # Check for strategy files
                py_files = list(tier_path.glob("*.py"))
                strategy_files = [f for f in py_files if f.name != "__init__.py"]
                
                self.check_condition(
                    len(strategy_files) > 0,
                    f"Found {len(strategy_files)} strategies in {tier}",
                    f"No strategies found in {tier} tier",
                    ValidationSeverity.WARNING if tier == "strategist" else ValidationSeverity.ERROR,
                    details={"tier": tier, "files": [f.name for f in strategy_files]},
                    recommendation=f"Implement strategies for {tier} tier"
                )
    
    async def _validate_strategy_configs(self):
        """Validate strategy configuration files."""
        config_path = Path("config/strategies")
        
        if not config_path.exists():
            config_path.mkdir(parents=True, exist_ok=True)
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message=f"Created strategy config directory: {config_path}"
            ))
        
        # Expected configuration files
        expected_configs = [
            "sniper_config.yaml",
            "hunter_config.yaml",
            "strategist_config.yaml",
            "common_config.yaml"
        ]
        
        for config_file in expected_configs:
            file_path = config_path / config_file
            
            if file_path.exists():
                # Validate YAML syntax
                try:
                    with open(file_path, "r") as f:
                        config_data = yaml.safe_load(f)
                    
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Valid configuration file: {config_file}",
                        details={"path": str(file_path)}
                    ))
                    
                    # Validate required fields
                    await self._validate_config_contents(config_file, config_data)
                    
                except yaml.YAMLError as e:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid YAML in {config_file}",
                        details={"error": str(e)},
                        recommendation=f"Fix YAML syntax in {file_path}"
                    ))
            else:
                # Create default config
                tier = config_file.replace("_config.yaml", "")
                default_config = self._get_default_config(tier)
                
                with open(file_path, "w") as f:
                    yaml.dump(default_config, f, default_flow_style=False)
                
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Created default config: {config_file}",
                    details={"path": str(file_path)}
                ))
    
    async def _validate_config_contents(self, config_file: str, config_data: Dict):
        """Validate configuration contents."""
        required_fields = {
            "sniper_config.yaml": ["enabled", "max_positions", "risk_per_trade", "strategies"],
            "hunter_config.yaml": ["enabled", "max_positions", "risk_per_trade", "strategies", "slicing"],
            "strategist_config.yaml": ["enabled", "max_positions", "risk_per_trade", "strategies", "vwap", "market_making"],
            "common_config.yaml": ["timeframes", "indicators", "risk_limits"]
        }
        
        if config_file in required_fields:
            for field in required_fields[config_file]:
                self.check_condition(
                    field in config_data,
                    f"Required field '{field}' present in {config_file}",
                    f"Missing required field '{field}' in {config_file}",
                    ValidationSeverity.ERROR,
                    recommendation=f"Add '{field}' to {config_file}"
                )
    
    async def _validate_tier_strategies(self):
        """Validate strategies for each tier."""
        tier_requirements = {
            "sniper": {
                "min_strategies": 2,
                "required_methods": ["calculate_signal", "calculate_position_size", "should_enter", "should_exit"],
                "max_complexity": "simple"
            },
            "hunter": {
                "min_strategies": 3,
                "required_methods": ["calculate_signal", "calculate_position_size", "should_enter", "should_exit", "slice_order"],
                "max_complexity": "medium"
            },
            "strategist": {
                "min_strategies": 2,
                "required_methods": ["calculate_signal", "calculate_position_size", "should_enter", "should_exit", "optimize_execution"],
                "max_complexity": "advanced"
            }
        }
        
        for tier, requirements in tier_requirements.items():
            tier_path = Path(f"genesis/strategies/{tier}")
            
            if not tier_path.exists():
                continue
            
            # Load and validate each strategy
            strategy_count = 0
            for py_file in tier_path.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                try:
                    # Import the module
                    module_name = f"genesis.strategies.{tier}.{py_file.stem}"
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Find strategy classes
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and name.endswith("Strategy"):
                                strategy_count += 1
                                
                                # Check required methods
                                for method in requirements["required_methods"]:
                                    has_method = hasattr(obj, method)
                                    self.check_condition(
                                        has_method,
                                        f"{tier}.{name} has method '{method}'",
                                        f"{tier}.{name} missing required method '{method}'",
                                        ValidationSeverity.ERROR,
                                        details={"tier": tier, "strategy": name, "method": method}
                                    )
                
                except Exception as e:
                    self.result.add_issue(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to load strategy: {py_file.name}",
                        details={"tier": tier, "file": str(py_file), "error": str(e)},
                        recommendation=f"Fix syntax errors in {py_file}"
                    ))
            
            # Check minimum strategy count
            self.check_condition(
                strategy_count >= requirements["min_strategies"],
                f"{tier} tier has {strategy_count} strategies (min: {requirements['min_strategies']})",
                f"{tier} tier has insufficient strategies: {strategy_count} < {requirements['min_strategies']}",
                ValidationSeverity.WARNING if tier == "strategist" else ValidationSeverity.ERROR,
                details={"tier": tier, "count": strategy_count, "required": requirements["min_strategies"]}
            )
    
    async def _test_strategy_initialization(self):
        """Test strategy initialization with mock data."""
        from genesis.config.settings import Settings
        
        # Try to initialize settings
        try:
            settings = Settings()
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Settings initialized successfully"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Failed to initialize settings",
                details={"error": str(e)},
                recommendation="Check environment variables and settings.py"
            ))
            return
        
        # Test strategy loader
        try:
            from genesis.strategies.loader import StrategyLoader
            
            loader = StrategyLoader(settings)
            strategies = loader.load_tier_strategies("sniper")
            
            self.check_condition(
                len(strategies) > 0,
                f"Strategy loader successfully loaded {len(strategies)} sniper strategies",
                "Strategy loader failed to load any strategies",
                ValidationSeverity.ERROR,
                details={"strategies_loaded": len(strategies)}
            )
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="StrategyLoader not implemented yet",
                recommendation="Implement genesis/strategies/loader.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Strategy loader initialization failed",
                details={"error": str(e)},
                recommendation="Fix strategy loader implementation"
            ))
    
    async def _validate_state_machines(self):
        """Validate strategy state machines."""
        try:
            from genesis.engine.state_machine import StateMachine, TierState
            
            # Test tier state transitions
            sm = StateMachine()
            
            # Test valid transitions
            transitions = [
                ("SNIPER", "HUNTER", 2000),
                ("HUNTER", "STRATEGIST", 10000),
                ("STRATEGIST", "HUNTER", 8000),  # Demotion
                ("HUNTER", "SNIPER", 1500)  # Demotion
            ]
            
            for from_tier, to_tier, capital in transitions:
                can_transition = sm.can_transition(from_tier, to_tier, capital)
                self.result.add_issue(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"State transition {from_tier} -> {to_tier} at ${capital}: {'✓' if can_transition else '✗'}",
                    details={"from": from_tier, "to": to_tier, "capital": capital}
                ))
            
        except ImportError:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message="State machine not implemented",
                recommendation="Implement genesis/engine/state_machine.py"
            ))
        except Exception as e:
            self.result.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="State machine validation failed",
                details={"error": str(e)},
                recommendation="Fix state machine implementation"
            ))
    
    def _get_default_config(self, tier: str) -> Dict[str, Any]:
        """Get default configuration for a tier."""
        if tier == "sniper":
            return {
                "enabled": True,
                "max_positions": 1,
                "risk_per_trade": 0.01,
                "max_daily_trades": 10,
                "strategies": {
                    "simple_arbitrage": {"enabled": True, "min_spread": 0.002},
                    "momentum_breakout": {"enabled": True, "threshold": 0.015}
                }
            }
        elif tier == "hunter":
            return {
                "enabled": True,
                "max_positions": 3,
                "risk_per_trade": 0.015,
                "max_daily_trades": 50,
                "slicing": {
                    "enabled": True,
                    "max_slices": 10,
                    "min_slice_size": 100
                },
                "strategies": {
                    "mean_reversion": {"enabled": True, "zscore_threshold": 2.0},
                    "pairs_trading": {"enabled": True, "correlation_threshold": 0.8}
                }
            }
        elif tier == "strategist":
            return {
                "enabled": True,
                "max_positions": 10,
                "risk_per_trade": 0.02,
                "max_daily_trades": 200,
                "vwap": {
                    "enabled": True,
                    "participation_rate": 0.1,
                    "urgency": "normal"
                },
                "market_making": {
                    "enabled": False,
                    "spread": 0.001,
                    "depth": 5
                },
                "strategies": {
                    "statistical_arbitrage": {"enabled": True},
                    "smart_routing": {"enabled": True}
                }
            }
        else:  # common
            return {
                "timeframes": ["1m", "5m", "15m", "1h"],
                "indicators": {
                    "sma": [20, 50, 200],
                    "ema": [12, 26],
                    "rsi": {"period": 14, "overbought": 70, "oversold": 30},
                    "macd": {"fast": 12, "slow": 26, "signal": 9}
                },
                "risk_limits": {
                    "max_drawdown": 0.1,
                    "max_correlation": 0.7,
                    "max_leverage": 3
                }
            }