"""
Feature flags system for graceful degradation and controlled rollouts.

Provides dynamic feature toggling based on system health, error rates,
and manual configuration for safe degradation during failures.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

import structlog


class FeatureStatus(Enum):
    """Status of a feature flag."""
    
    ENABLED = "enabled"  # Feature is active
    DISABLED = "disabled"  # Feature is disabled
    DEGRADED = "degraded"  # Feature running in degraded mode
    EXPERIMENTAL = "experimental"  # Feature in testing


class DegradationLevel(Enum):
    """System degradation levels."""
    
    NORMAL = "normal"  # All features enabled
    MINOR = "minor"  # Non-critical features disabled
    MAJOR = "major"  # Most features disabled
    CRITICAL = "critical"  # Only essential features enabled
    EMERGENCY = "emergency"  # Minimal functionality only


@dataclass
class FeatureFlag:
    """Represents a feature flag configuration."""
    
    name: str
    description: str
    status: FeatureStatus = FeatureStatus.ENABLED
    critical: bool = False  # Is this a critical feature?
    dependencies: List[str] = field(default_factory=list)
    degradation_level: DegradationLevel = DegradationLevel.NORMAL
    enabled_at_levels: Set[DegradationLevel] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_toggled: Optional[datetime] = None
    toggle_count: int = 0
    error_threshold: float = 0.1  # Error rate threshold for auto-disable
    current_error_rate: float = 0.0
    
    def is_enabled_at_level(self, level: DegradationLevel) -> bool:
        """Check if feature is enabled at given degradation level."""
        if self.critical:
            # Critical features always enabled except in emergency
            return level != DegradationLevel.EMERGENCY
        
        # Check if feature is enabled at this level
        level_hierarchy = {
            DegradationLevel.NORMAL: 0,
            DegradationLevel.MINOR: 1,
            DegradationLevel.MAJOR: 2,
            DegradationLevel.CRITICAL: 3,
            DegradationLevel.EMERGENCY: 4,
        }
        
        # Feature disabled at higher degradation levels
        if level_hierarchy[level] >= level_hierarchy[self.degradation_level]:
            return False
        
        return self.status == FeatureStatus.ENABLED


@dataclass
class FallbackStrategy:
    """Defines fallback behavior for degraded features."""
    
    feature_name: str
    fallback_function: Optional[Callable] = None
    fallback_value: Any = None
    cache_duration: int = 60  # Seconds to cache fallback result
    log_degradation: bool = True
    alert_on_fallback: bool = False


class FeatureManager:
    """
    Manages feature flags for graceful degradation.
    
    Provides:
    - Dynamic feature toggling based on error rates
    - Degradation levels for system-wide feature control
    - Fallback strategies for degraded features
    - Dependency management between features
    """
    
    def __init__(
        self,
        config_file: Optional[Path] = None,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        self.config_file = config_file or Path(".genesis/feature_flags.json")
        
        # Feature registry
        self._features: Dict[str, FeatureFlag] = {}
        self._fallback_strategies: Dict[str, FallbackStrategy] = {}
        
        # System state
        self._degradation_level = DegradationLevel.NORMAL
        self._auto_degrade_enabled = True
        self._error_counts: Dict[str, int] = {}
        self._success_counts: Dict[str, int] = {}
        
        # Load configuration
        self._load_config()
        self._initialize_default_features()
    
    def _initialize_default_features(self):
        """Initialize default feature flags."""
        default_features = [
            FeatureFlag(
                name="order_execution",
                description="Core order execution",
                critical=True,
                status=FeatureStatus.ENABLED,
            ),
            FeatureFlag(
                name="risk_management",
                description="Risk limit enforcement",
                critical=True,
                status=FeatureStatus.ENABLED,
            ),
            FeatureFlag(
                name="market_data_streaming",
                description="Real-time market data",
                critical=False,
                status=FeatureStatus.ENABLED,
                degradation_level=DegradationLevel.MINOR,
            ),
            FeatureFlag(
                name="advanced_analytics",
                description="Advanced trading analytics",
                critical=False,
                status=FeatureStatus.ENABLED,
                degradation_level=DegradationLevel.MINOR,
            ),
            FeatureFlag(
                name="multi_pair_trading",
                description="Multiple pair trading",
                critical=False,
                status=FeatureStatus.ENABLED,
                degradation_level=DegradationLevel.MAJOR,
                dependencies=["market_data_streaming"],
            ),
            FeatureFlag(
                name="ui_dashboard",
                description="Terminal UI dashboard",
                critical=False,
                status=FeatureStatus.ENABLED,
                degradation_level=DegradationLevel.MINOR,
            ),
            FeatureFlag(
                name="automated_recovery",
                description="Automated error recovery",
                critical=False,
                status=FeatureStatus.ENABLED,
                degradation_level=DegradationLevel.MAJOR,
            ),
            FeatureFlag(
                name="performance_monitoring",
                description="Performance metrics collection",
                critical=False,
                status=FeatureStatus.ENABLED,
                degradation_level=DegradationLevel.MINOR,
            ),
            FeatureFlag(
                name="tilt_detection",
                description="Psychological tilt detection",
                critical=False,
                status=FeatureStatus.ENABLED,
                degradation_level=DegradationLevel.MAJOR,
            ),
            FeatureFlag(
                name="backtesting",
                description="Strategy backtesting",
                critical=False,
                status=FeatureStatus.ENABLED,
                degradation_level=DegradationLevel.MAJOR,
            ),
        ]
        
        for feature in default_features:
            if feature.name not in self._features:
                self._features[feature.name] = feature
    
    def _load_config(self):
        """Load feature flags from configuration file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    
                # Load features
                for feature_data in config.get("features", []):
                    feature = FeatureFlag(
                        name=feature_data["name"],
                        description=feature_data["description"],
                        status=FeatureStatus(feature_data.get("status", "enabled")),
                        critical=feature_data.get("critical", False),
                        dependencies=feature_data.get("dependencies", []),
                        degradation_level=DegradationLevel(
                            feature_data.get("degradation_level", "normal")
                        ),
                        metadata=feature_data.get("metadata", {}),
                    )
                    self._features[feature.name] = feature
                
                # Load system state
                self._degradation_level = DegradationLevel(
                    config.get("degradation_level", "normal")
                )
                self._auto_degrade_enabled = config.get("auto_degrade", True)
                
                self.logger.info(
                    "Loaded feature flags configuration",
                    feature_count=len(self._features),
                    degradation_level=self._degradation_level.value,
                )
                
            except Exception as e:
                self.logger.error(
                    "Failed to load feature flags config",
                    error=str(e),
                )
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            config = {
                "degradation_level": self._degradation_level.value,
                "auto_degrade": self._auto_degrade_enabled,
                "features": [
                    {
                        "name": f.name,
                        "description": f.description,
                        "status": f.status.value,
                        "critical": f.critical,
                        "dependencies": f.dependencies,
                        "degradation_level": f.degradation_level.value,
                        "metadata": f.metadata,
                    }
                    for f in self._features.values()
                ],
            }
            
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            self.logger.error(
                "Failed to save feature flags config",
                error=str(e),
            )
    
    def register_feature(
        self,
        name: str,
        description: str,
        critical: bool = False,
        dependencies: Optional[List[str]] = None,
        degradation_level: DegradationLevel = DegradationLevel.NORMAL,
    ) -> FeatureFlag:
        """
        Register a new feature flag.
        
        Args:
            name: Unique feature name
            description: Feature description
            critical: Whether feature is critical
            dependencies: List of dependent feature names
            degradation_level: Level at which feature degrades
            
        Returns:
            The created feature flag
        """
        feature = FeatureFlag(
            name=name,
            description=description,
            critical=critical,
            dependencies=dependencies or [],
            degradation_level=degradation_level,
        )
        
        self._features[name] = feature
        self._save_config()
        
        self.logger.info(
            "Registered feature flag",
            feature=name,
            critical=critical,
        )
        
        return feature
    
    def register_fallback(
        self,
        feature_name: str,
        fallback_function: Optional[Callable] = None,
        fallback_value: Any = None,
        cache_duration: int = 60,
    ):
        """
        Register fallback strategy for a feature.
        
        Args:
            feature_name: Name of the feature
            fallback_function: Function to call when degraded
            fallback_value: Static value to return when degraded
            cache_duration: Seconds to cache fallback result
        """
        strategy = FallbackStrategy(
            feature_name=feature_name,
            fallback_function=fallback_function,
            fallback_value=fallback_value,
            cache_duration=cache_duration,
        )
        
        self._fallback_strategies[feature_name] = strategy
        
        self.logger.info(
            "Registered fallback strategy",
            feature=feature_name,
        )
    
    def is_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            True if feature is enabled
        """
        feature = self._features.get(feature_name)
        if not feature:
            self.logger.warning(
                "Unknown feature flag",
                feature=feature_name,
            )
            return False
        
        # Check dependencies
        for dep in feature.dependencies:
            if not self.is_enabled(dep):
                return False
        
        # Check degradation level
        if not feature.is_enabled_at_level(self._degradation_level):
            return False
        
        # Check feature status
        return feature.status == FeatureStatus.ENABLED
    
    def toggle_feature(
        self,
        feature_name: str,
        enabled: Optional[bool] = None,
    ) -> bool:
        """
        Toggle a feature on/off.
        
        Args:
            feature_name: Name of the feature
            enabled: Explicit state (None = toggle)
            
        Returns:
            New state of the feature
        """
        feature = self._features.get(feature_name)
        if not feature:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        if feature.critical:
            self.logger.warning(
                "Attempting to toggle critical feature",
                feature=feature_name,
            )
            # Require additional confirmation for critical features
            
        # Determine new status
        if enabled is None:
            # Toggle
            new_status = (
                FeatureStatus.DISABLED
                if feature.status == FeatureStatus.ENABLED
                else FeatureStatus.ENABLED
            )
        else:
            new_status = FeatureStatus.ENABLED if enabled else FeatureStatus.DISABLED
        
        # Update feature
        feature.status = new_status
        feature.last_toggled = datetime.utcnow()
        feature.toggle_count += 1
        
        self._save_config()
        
        self.logger.info(
            "Toggled feature flag",
            feature=feature_name,
            new_status=new_status.value,
        )
        
        return new_status == FeatureStatus.ENABLED
    
    def set_degradation_level(self, level: DegradationLevel):
        """
        Set system degradation level.
        
        Args:
            level: New degradation level
        """
        old_level = self._degradation_level
        self._degradation_level = level
        
        # Log features affected
        affected = []
        for feature in self._features.values():
            was_enabled = feature.is_enabled_at_level(old_level)
            is_enabled = feature.is_enabled_at_level(level)
            
            if was_enabled != is_enabled:
                affected.append({
                    "feature": feature.name,
                    "was_enabled": was_enabled,
                    "is_enabled": is_enabled,
                })
        
        self._save_config()
        
        self.logger.warning(
            "Changed system degradation level",
            old_level=old_level.value,
            new_level=level.value,
            affected_features=len(affected),
        )
    
    def record_feature_error(self, feature_name: str):
        """Record an error for a feature."""
        self._error_counts[feature_name] = self._error_counts.get(feature_name, 0) + 1
        
        # Check if auto-degradation should trigger
        if self._auto_degrade_enabled:
            self._check_auto_degrade(feature_name)
    
    def record_feature_success(self, feature_name: str):
        """Record a success for a feature."""
        self._success_counts[feature_name] = self._success_counts.get(feature_name, 0) + 1
    
    def _check_auto_degrade(self, feature_name: str):
        """Check if feature should be auto-degraded based on error rate."""
        feature = self._features.get(feature_name)
        if not feature or feature.critical:
            return
        
        errors = self._error_counts.get(feature_name, 0)
        successes = self._success_counts.get(feature_name, 0)
        total = errors + successes
        
        if total < 10:  # Need minimum samples
            return
        
        error_rate = errors / total
        feature.current_error_rate = error_rate
        
        if error_rate > feature.error_threshold:
            # Auto-disable feature
            feature.status = FeatureStatus.DEGRADED
            self._save_config()
            
            self.logger.error(
                "Auto-degraded feature due to high error rate",
                feature=feature_name,
                error_rate=error_rate,
                threshold=feature.error_threshold,
            )
    
    def get_fallback(self, feature_name: str) -> Any:
        """
        Get fallback value/function for a degraded feature.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Fallback value or result of fallback function
        """
        strategy = self._fallback_strategies.get(feature_name)
        if not strategy:
            return None
        
        if strategy.log_degradation:
            self.logger.info(
                "Using fallback for degraded feature",
                feature=feature_name,
            )
        
        if strategy.fallback_function:
            return strategy.fallback_function()
        
        return strategy.fallback_value
    
    def get_enabled_features(self) -> List[str]:
        """Get list of currently enabled features."""
        return [
            name for name, feature in self._features.items()
            if self.is_enabled(name)
        ]
    
    def get_degraded_features(self) -> List[str]:
        """Get list of degraded features."""
        return [
            name for name, feature in self._features.items()
            if feature.status == FeatureStatus.DEGRADED
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature flag statistics."""
        total = len(self._features)
        enabled = sum(1 for f in self._features.values() if f.status == FeatureStatus.ENABLED)
        disabled = sum(1 for f in self._features.values() if f.status == FeatureStatus.DISABLED)
        degraded = sum(1 for f in self._features.values() if f.status == FeatureStatus.DEGRADED)
        
        return {
            "degradation_level": self._degradation_level.value,
            "total_features": total,
            "enabled": enabled,
            "disabled": disabled,
            "degraded": degraded,
            "critical_features": sum(1 for f in self._features.values() if f.critical),
            "auto_degrade_enabled": self._auto_degrade_enabled,
        }
    
    def reset_error_counts(self):
        """Reset error and success counts."""
        self._error_counts.clear()
        self._success_counts.clear()
        
        for feature in self._features.values():
            feature.current_error_rate = 0.0