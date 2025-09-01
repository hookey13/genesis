"""
Error budget tracking system for SLO monitoring.

Tracks error rates against defined Service Level Objectives (SLOs) and
provides alerting when error budgets are exhausted.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Deque, Tuple

import structlog

from genesis.core.error_handler import ErrorSeverity


class SLOTarget(Enum):
    """Service Level Objective targets."""
    
    FIVE_NINES = 0.99999  # 99.999% availability
    FOUR_NINES = 0.9999  # 99.99% availability
    THREE_NINES = 0.999  # 99.9% availability
    TWO_NINES = 0.99  # 99% availability
    ONE_NINE = 0.9  # 90% availability


@dataclass
class SLO:
    """Service Level Objective definition."""
    
    name: str
    description: str
    target: float  # Success rate target (0.0 to 1.0)
    measurement_window: timedelta = timedelta(days=30)
    critical: bool = False
    categories: List[str] = field(default_factory=list)  # Error categories to track
    severity_weights: Dict[ErrorSeverity, float] = field(default_factory=dict)


@dataclass
class ErrorBudgetStatus:
    """Current status of an error budget."""
    
    slo_name: str
    target: float
    current_success_rate: float
    error_budget_total: float  # Total allowed errors
    error_budget_consumed: float  # Errors already consumed
    error_budget_remaining: float  # Remaining error budget
    time_window_start: datetime
    time_window_end: datetime
    is_exhausted: bool
    burn_rate: float  # Rate of budget consumption
    time_to_exhaustion: Optional[timedelta] = None
    error_categories: Dict[str, int] = field(default_factory=dict)  # Error counts by category
    error_trend: str = "stable"  # "improving", "degrading", "stable"
    confidence_level: float = 0.0  # Confidence in the trend (0.0 to 1.0)


class ErrorBudget:
    """
    Error budget tracking for SLO monitoring.
    
    Tracks error rates against defined SLOs and provides:
    - Real-time error budget calculation
    - Burn rate monitoring
    - Alerting when budget thresholds exceeded
    - Historical tracking for trend analysis
    """
    
    def __init__(
        self,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.logger = logger or structlog.get_logger(__name__)
        
        # SLO definitions
        self._slos: Dict[str, SLO] = {}
        
        # Error tracking
        self._error_counts: Dict[str, Dict[str, int]] = {}  # slo_name -> category -> count
        self._success_counts: Dict[str, int] = {}
        self._window_start: Dict[str, datetime] = {}
        
        # Historical tracking for trends (last 60 minutes)
        self._error_history: Dict[str, Deque[Tuple[datetime, int]]] = {}
        self._success_history: Dict[str, Deque[Tuple[datetime, int]]] = {}
        self._minute_buckets: Dict[str, Dict[datetime, Dict[str, int]]] = {}  # Per-minute error counts
        
        # Alert callbacks
        self._alert_callbacks: List[Callable] = []
        self._threshold_alerts: Dict[str, float] = {}  # slo_name -> threshold (0.0 to 1.0)
        
        # Initialize default SLOs
        self._initialize_default_slos()
    
    def _initialize_default_slos(self):
        """Initialize default SLO definitions."""
        default_slos = [
            SLO(
                name="overall_availability",
                description="Overall system availability",
                target=SLOTarget.THREE_NINES.value,
                critical=True,
                severity_weights={
                    ErrorSeverity.CRITICAL: 10.0,
                    ErrorSeverity.HIGH: 5.0,
                    ErrorSeverity.MEDIUM: 2.0,
                    ErrorSeverity.LOW: 1.0,
                },
            ),
            SLO(
                name="order_execution",
                description="Order execution success rate",
                target=SLOTarget.FOUR_NINES.value,
                critical=True,
                categories=["EXCHANGE", "TRADING"],
                measurement_window=timedelta(days=7),
            ),
            SLO(
                name="data_accuracy",
                description="Market data accuracy",
                target=SLOTarget.THREE_NINES.value,
                categories=["MARKET_DATA", "NETWORK"],
                measurement_window=timedelta(days=1),
            ),
            SLO(
                name="risk_compliance",
                description="Risk limit compliance",
                target=SLOTarget.FIVE_NINES.value,
                critical=True,
                categories=["RISK", "BUSINESS"],
                measurement_window=timedelta(days=30),
            ),
            SLO(
                name="api_latency",
                description="API response time",
                target=0.95,  # 95% of requests under threshold
                categories=["NETWORK", "EXCHANGE"],
                measurement_window=timedelta(hours=1),
            ),
        ]
        
        for slo in default_slos:
            self.register_slo(slo)
    
    def register_slo(self, slo: SLO):
        """
        Register a new SLO.
        
        Args:
            slo: SLO definition
        """
        self._slos[slo.name] = slo
        self._error_counts[slo.name] = {}
        self._success_counts[slo.name] = 0
        self._window_start[slo.name] = datetime.utcnow()
        
        # Initialize history tracking
        self._error_history[slo.name] = deque(maxlen=60)  # 60 minutes
        self._success_history[slo.name] = deque(maxlen=60)
        self._minute_buckets[slo.name] = {}
        
        # Set default alert threshold at 80% consumed
        self._threshold_alerts[slo.name] = 0.8
        
        self.logger.info(
            "Registered SLO",
            slo_name=slo.name,
            target=slo.target,
            window_days=slo.measurement_window.days,
        )
    
    def record_error(
        self,
        slo_name: str,
        category: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        """
        Record an error against an SLO.
        
        Args:
            slo_name: Name of the SLO
            category: Error category
            severity: Error severity
        """
        if slo_name not in self._slos:
            # Record against overall availability if SLO not found
            slo_name = "overall_availability"
        
        slo = self._slos[slo_name]
        
        # Check if we should track this category
        if slo.categories and category not in slo.categories:
            return
        
        # Apply severity weighting
        weight = slo.severity_weights.get(severity, 1.0)
        
        # Update counts
        if category not in self._error_counts[slo_name]:
            self._error_counts[slo_name][category] = 0
        
        self._error_counts[slo_name][category] += weight
        
        # Update per-minute buckets
        now = datetime.utcnow()
        minute_key = now.replace(second=0, microsecond=0)
        
        if minute_key not in self._minute_buckets[slo_name]:
            self._minute_buckets[slo_name][minute_key] = {}
        
        if category not in self._minute_buckets[slo_name][minute_key]:
            self._minute_buckets[slo_name][minute_key][category] = 0
        
        self._minute_buckets[slo_name][minute_key][category] += weight
        
        # Update error history
        self._error_history[slo_name].append((now, weight))
        
        # Clean old minute buckets (keep last 60 minutes)
        cutoff = now - timedelta(minutes=60)
        old_keys = [k for k in self._minute_buckets[slo_name] if k < cutoff]
        for key in old_keys:
            del self._minute_buckets[slo_name][key]
        
        # Check measurement window
        self._check_window_rollover(slo_name)
        
        # Check budget status
        status = self.get_budget_status(slo_name)
        if status and status.is_exhausted:
            self._trigger_exhaustion_alert(slo_name, status)
        elif status:
            # Check threshold alerts
            consumption_rate = status.error_budget_consumed / status.error_budget_total
            threshold = self._threshold_alerts.get(slo_name, 0.8)
            
            if consumption_rate >= threshold:
                self._trigger_threshold_alert(slo_name, status, consumption_rate)
    
    def record_success(self, slo_name: str):
        """
        Record a successful operation against an SLO.
        
        Args:
            slo_name: Name of the SLO
        """
        if slo_name not in self._slos:
            slo_name = "overall_availability"
        
        self._success_counts[slo_name] = self._success_counts.get(slo_name, 0) + 1
        
        # Update success history
        now = datetime.utcnow()
        self._success_history[slo_name].append((now, 1))
        
        # Check measurement window
        self._check_window_rollover(slo_name)
    
    def _check_window_rollover(self, slo_name: str):
        """Check if measurement window should roll over."""
        slo = self._slos[slo_name]
        window_start = self._window_start[slo_name]
        
        if datetime.utcnow() - window_start > slo.measurement_window:
            # Roll over to new window
            self._window_start[slo_name] = datetime.utcnow()
            self._error_counts[slo_name] = {}
            self._success_counts[slo_name] = 0
            
            self.logger.info(
                "Rolled over SLO measurement window",
                slo_name=slo_name,
            )
    
    def get_budget_status(self, slo_name: str) -> Optional[ErrorBudgetStatus]:
        """
        Get current error budget status for an SLO.
        
        Args:
            slo_name: Name of the SLO
            
        Returns:
            Error budget status or None if SLO not found
        """
        if slo_name not in self._slos:
            return None
        
        slo = self._slos[slo_name]
        
        # Calculate total operations
        total_errors = sum(self._error_counts[slo_name].values())
        total_successes = self._success_counts.get(slo_name, 0)
        total_operations = total_errors + total_successes
        
        if total_operations == 0:
            # No operations yet
            return ErrorBudgetStatus(
                slo_name=slo_name,
                target=slo.target,
                current_success_rate=1.0,
                error_budget_total=0,
                error_budget_consumed=0,
                error_budget_remaining=0,
                time_window_start=self._window_start[slo_name],
                time_window_end=self._window_start[slo_name] + slo.measurement_window,
                is_exhausted=False,
                burn_rate=0.0,
            )
        
        # Calculate success rate
        current_success_rate = total_successes / total_operations
        
        # Calculate error budget
        allowed_error_rate = 1.0 - slo.target
        error_budget_total = total_operations * allowed_error_rate
        error_budget_consumed = total_errors
        error_budget_remaining = max(0, error_budget_total - error_budget_consumed)
        
        # Calculate burn rate
        elapsed = (datetime.utcnow() - self._window_start[slo_name]).total_seconds()
        window_seconds = slo.measurement_window.total_seconds()
        time_fraction = elapsed / window_seconds if window_seconds > 0 else 1.0
        
        if time_fraction > 0 and error_budget_total > 0:
            expected_consumption = error_budget_total * time_fraction
            actual_consumption = error_budget_consumed
            burn_rate = actual_consumption / expected_consumption if expected_consumption > 0 else 0.0
        else:
            burn_rate = 0.0
        
        # Calculate time to exhaustion
        time_to_exhaustion = None
        if burn_rate > 1.0 and error_budget_remaining > 0:
            current_rate = error_budget_consumed / elapsed if elapsed > 0 else 0
            if current_rate > 0:
                seconds_remaining = error_budget_remaining / current_rate
                time_to_exhaustion = timedelta(seconds=seconds_remaining)
        
        # Calculate error trend
        error_trend, confidence = self._calculate_trend(slo_name)
        
        return ErrorBudgetStatus(
            slo_name=slo_name,
            target=slo.target,
            current_success_rate=current_success_rate,
            error_budget_total=error_budget_total,
            error_budget_consumed=error_budget_consumed,
            error_budget_remaining=error_budget_remaining,
            time_window_start=self._window_start[slo_name],
            time_window_end=self._window_start[slo_name] + slo.measurement_window,
            is_exhausted=error_budget_remaining <= 0,
            burn_rate=burn_rate,
            time_to_exhaustion=time_to_exhaustion,
            error_categories=dict(self._error_counts[slo_name]),
            error_trend=error_trend,
            confidence_level=confidence,
        )
    
    def get_all_budgets(self) -> Dict[str, ErrorBudgetStatus]:
        """Get status of all error budgets."""
        return {
            slo_name: self.get_budget_status(slo_name)
            for slo_name in self._slos.keys()
        }
    
    def register_alert_callback(self, callback: Callable):
        """
        Register callback for budget alerts.
        
        Args:
            callback: Function to call on alerts
        """
        self._alert_callbacks.append(callback)
    
    def set_alert_threshold(self, slo_name: str, threshold: float):
        """
        Set alert threshold for an SLO.
        
        Args:
            slo_name: Name of the SLO
            threshold: Threshold (0.0 to 1.0) of budget consumption
        """
        if slo_name in self._slos:
            self._threshold_alerts[slo_name] = threshold
    
    def _trigger_exhaustion_alert(self, slo_name: str, status: ErrorBudgetStatus):
        """Trigger alert for exhausted error budget."""
        slo = self._slos[slo_name]
        
        self.logger.critical(
            "Error budget exhausted",
            slo_name=slo_name,
            target=slo.target,
            current_success_rate=status.current_success_rate,
            critical=slo.critical,
        )
        
        # Execute callbacks
        for callback in self._alert_callbacks:
            try:
                callback({
                    "type": "budget_exhausted",
                    "slo_name": slo_name,
                    "status": status,
                    "critical": slo.critical,
                })
            except Exception as e:
                self.logger.error(
                    "Failed to execute alert callback",
                    error=str(e),
                )
    
    def _trigger_threshold_alert(
        self,
        slo_name: str,
        status: ErrorBudgetStatus,
        consumption_rate: float,
    ):
        """Trigger alert for threshold exceeded."""
        slo = self._slos[slo_name]
        
        self.logger.warning(
            "Error budget threshold exceeded",
            slo_name=slo_name,
            consumption_rate=consumption_rate,
            threshold=self._threshold_alerts[slo_name],
            time_to_exhaustion=str(status.time_to_exhaustion) if status.time_to_exhaustion else None,
        )
        
        # Execute callbacks
        for callback in self._alert_callbacks:
            try:
                callback({
                    "type": "threshold_exceeded",
                    "slo_name": slo_name,
                    "status": status,
                    "consumption_rate": consumption_rate,
                })
            except Exception as e:
                self.logger.error(
                    "Failed to execute alert callback",
                    error=str(e),
                )
    
    def reset_budget(self, slo_name: str):
        """
        Reset error budget for an SLO.
        
        Args:
            slo_name: Name of the SLO
        """
        if slo_name in self._slos:
            self._error_counts[slo_name] = {}
            self._success_counts[slo_name] = 0
            self._window_start[slo_name] = datetime.utcnow()
            
            self.logger.info(
                "Reset error budget",
                slo_name=slo_name,
            )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get error budget statistics."""
        stats = {
            "total_slos": len(self._slos),
            "critical_slos": sum(1 for s in self._slos.values() if s.critical),
            "exhausted_budgets": 0,
            "at_risk_budgets": 0,
            "healthy_budgets": 0,
            "slo_details": {},
        }
        
        for slo_name in self._slos:
            status = self.get_budget_status(slo_name)
            if status:
                if status.is_exhausted:
                    stats["exhausted_budgets"] += 1
                elif status.error_budget_consumed / status.error_budget_total > 0.8:
                    stats["at_risk_budgets"] += 1
                else:
                    stats["healthy_budgets"] += 1
                
                stats["slo_details"][slo_name] = {
                    "success_rate": status.current_success_rate,
                    "budget_consumed": status.error_budget_consumed / status.error_budget_total
                    if status.error_budget_total > 0 else 0,
                    "burn_rate": status.burn_rate,
                }
        
        return stats
    
    def _calculate_trend(self, slo_name: str) -> Tuple[str, float]:
        """
        Calculate error trend based on historical data.
        
        Returns:
            Tuple of (trend, confidence) where trend is "improving", "degrading", or "stable"
        """
        if not self._error_history.get(slo_name):
            return "stable", 0.0
        
        history = list(self._error_history[slo_name])
        if len(history) < 10:
            return "stable", 0.0  # Not enough data
        
        # Calculate moving averages
        recent_errors = sum(e[1] for e in history[-10:]) / 10
        older_errors = sum(e[1] for e in history[-20:-10]) / 10 if len(history) >= 20 else recent_errors
        
        if older_errors == 0:
            if recent_errors > 0:
                return "degrading", 0.8
            return "stable", 0.5
        
        change_ratio = (recent_errors - older_errors) / older_errors
        
        # Calculate confidence based on data points
        confidence = min(1.0, len(history) / 30)
        
        if change_ratio < -0.2:
            return "improving", confidence
        elif change_ratio > 0.2:
            return "degrading", confidence
        else:
            return "stable", confidence * 0.5
    
    def get_error_rate_per_minute(self, slo_name: str, minutes: int = 60) -> List[Dict[str, any]]:
        """
        Get error rate per minute for the last N minutes.
        
        Args:
            slo_name: Name of the SLO
            minutes: Number of minutes to retrieve (default 60)
            
        Returns:
            List of per-minute error data
        """
        if slo_name not in self._slos:
            return []
        
        now = datetime.utcnow()
        result = []
        
        for i in range(minutes):
            minute_time = now - timedelta(minutes=i)
            minute_key = minute_time.replace(second=0, microsecond=0)
            
            if minute_key in self._minute_buckets.get(slo_name, {}):
                minute_data = self._minute_buckets[slo_name][minute_key]
                total_errors = sum(minute_data.values())
                result.append({
                    "timestamp": minute_key.isoformat(),
                    "error_count": total_errors,
                    "categories": dict(minute_data),
                })
            else:
                result.append({
                    "timestamp": minute_key.isoformat(),
                    "error_count": 0,
                    "categories": {},
                })
        
        return list(reversed(result))
    
    def get_error_categories_breakdown(self, slo_name: str) -> Dict[str, float]:
        """
        Get percentage breakdown of errors by category.
        
        Args:
            slo_name: Name of the SLO
            
        Returns:
            Dictionary of category to percentage
        """
        if slo_name not in self._error_counts:
            return {}
        
        total_errors = sum(self._error_counts[slo_name].values())
        if total_errors == 0:
            return {}
        
        return {
            category: (count / total_errors) * 100
            for category, count in self._error_counts[slo_name].items()
        }
    
    def get_historical_trend(self, slo_name: str, hours: int = 24) -> List[Dict[str, any]]:
        """
        Get historical error trend for charting.
        
        Args:
            slo_name: Name of the SLO
            hours: Number of hours of history to retrieve
            
        Returns:
            List of hourly aggregated error data
        """
        if slo_name not in self._minute_buckets:
            return []
        
        now = datetime.utcnow()
        hourly_data = {}
        
        # Aggregate minute data into hourly buckets
        for minute_key, categories in self._minute_buckets[slo_name].items():
            hour_key = minute_key.replace(minute=0)
            
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {"error_count": 0, "categories": {}}
            
            hourly_data[hour_key]["error_count"] += sum(categories.values())
            
            for category, count in categories.items():
                if category not in hourly_data[hour_key]["categories"]:
                    hourly_data[hour_key]["categories"][category] = 0
                hourly_data[hour_key]["categories"][category] += count
        
        # Create result for requested hours
        result = []
        for i in range(hours):
            hour_time = now - timedelta(hours=i)
            hour_key = hour_time.replace(minute=0, second=0, microsecond=0)
            
            if hour_key in hourly_data:
                result.append({
                    "timestamp": hour_key.isoformat(),
                    "error_count": hourly_data[hour_key]["error_count"],
                    "categories": hourly_data[hour_key]["categories"],
                })
            else:
                result.append({
                    "timestamp": hour_key.isoformat(),
                    "error_count": 0,
                    "categories": {},
                })
        
        return list(reversed(result))