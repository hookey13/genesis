"""
SLO (Service Level Objective) tracking and error budget calculation.

This module implements SLI collection, SLO evaluation, and error budget
tracking using the multi-window, multi-burn-rate methodology from Google SRE.
"""

import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import structlog
import yaml
from prometheus_client import Gauge, Counter, Histogram

from genesis.core.exceptions import ValidationError

logger = structlog.get_logger(__name__)

# Prometheus metrics for SLO tracking
slo_compliance_gauge = Gauge(
    'genesis_slo_compliance_ratio',
    'Current SLO compliance ratio',
    ['service', 'sli_type', 'window']
)

error_budget_remaining_gauge = Gauge(
    'genesis_error_budget_remaining_ratio',
    'Remaining error budget as a ratio',
    ['service', 'window']
)

error_budget_burn_rate_gauge = Gauge(
    'genesis_error_budget_burn_rate',
    'Current error budget burn rate',
    ['service', 'window']
)

sli_calculation_duration = Histogram(
    'genesis_sli_calculation_duration_seconds',
    'Time taken to calculate SLIs',
    ['service', 'sli_type']
)

slo_evaluation_counter = Counter(
    'genesis_slo_evaluations_total',
    'Total number of SLO evaluations',
    ['service', 'result']
)


class SLIType(Enum):
    """Types of Service Level Indicators."""
    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    SATURATION = "saturation"


class AlertSeverity(Enum):
    """Alert severity levels for burn rate thresholds."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class BurnRateThreshold:
    """Burn rate threshold configuration."""
    rate: float
    window_minutes: int
    severity: AlertSeverity


@dataclass
class SLIConfig:
    """Configuration for a single SLI."""
    type: SLIType
    metric_query: str
    threshold: float
    aggregation: str = "avg"  # avg, min, max, p50, p95, p99
    unit: str = ""


@dataclass
class SLOConfig:
    """Configuration for a service's SLO."""
    service: str
    slis: Dict[str, SLIConfig]
    error_budget_window_days: int = 30
    burn_rate_thresholds: List[BurnRateThreshold] = field(default_factory=list)


@dataclass
class SLIResult:
    """Result of an SLI calculation."""
    timestamp: datetime
    service: str
    sli_type: SLIType
    value: float
    threshold: float
    is_good: bool
    calculation_time_ms: float


@dataclass
class ErrorBudget:
    """Error budget calculation result."""
    service: str
    window: timedelta
    total_budget: float
    consumed_budget: float
    remaining_budget: float
    remaining_ratio: float
    burn_rate: float
    time_until_exhaustion: Optional[timedelta] = None


class SLOTracker:
    """
    Tracks Service Level Objectives and calculates error budgets.
    
    Implements the multi-window, multi-burn-rate alerting methodology
    to balance between quick detection and alert fatigue reduction.
    """
    
    def __init__(self, config_path: str = "config/slo_definitions.yaml"):
        self.config_path = config_path
        self.slo_configs: Dict[str, SLOConfig] = {}
        self.sli_history: Dict[str, List[SLIResult]] = {}
        self.prometheus_client = None  # Will be injected
        self._running = False
        self._task = None
        
    async def initialize(self) -> None:
        """Initialize the SLO tracker."""
        await self.load_config()
        logger.info("SLO tracker initialized", services=list(self.slo_configs.keys()))
        
    async def load_config(self) -> None:
        """Load SLO configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            for service_name, service_config in config_data.get('services', {}).items():
                slis = {}
                for sli_name, sli_config in service_config.get('slis', {}).items():
                    slis[sli_name] = SLIConfig(
                        type=SLIType(sli_config['type']),
                        metric_query=sli_config['metric'],
                        threshold=sli_config['threshold'],
                        aggregation=sli_config.get('aggregation', 'avg'),
                        unit=sli_config.get('unit', '')
                    )
                
                burn_thresholds = []
                for threshold in service_config.get('error_budget', {}).get('burn_rate_thresholds', []):
                    burn_thresholds.append(BurnRateThreshold(
                        rate=threshold['rate'],
                        window_minutes=threshold.get('window_minutes', 60),
                        severity=AlertSeverity(threshold['severity'])
                    ))
                
                self.slo_configs[service_name] = SLOConfig(
                    service=service_name,
                    slis=slis,
                    error_budget_window_days=service_config.get('error_budget', {}).get('window_days', 30),
                    burn_rate_thresholds=burn_thresholds
                )
                
        except Exception as e:
            logger.error("Failed to load SLO configuration", error=str(e))
            raise ValidationError(f"Invalid SLO configuration: {e}")
    
    async def calculate_sli(self, service: str, sli_name: str, sli_config: SLIConfig) -> SLIResult:
        """
        Calculate a single SLI value.
        
        Args:
            service: Service name
            sli_name: SLI identifier
            sli_config: SLI configuration
            
        Returns:
            SLIResult with calculated value
        """
        start_time = time.time()
        
        try:
            # Query Prometheus for metric value
            # In production, this would use the actual Prometheus client
            # For now, we'll simulate with realistic values
            value = await self._query_prometheus(sli_config.metric_query)
            
            # Determine if SLI is meeting objective
            is_good = self._evaluate_sli(value, sli_config.threshold, sli_config.type)
            
            calculation_time = (time.time() - start_time) * 1000
            
            result = SLIResult(
                timestamp=datetime.utcnow(),
                service=service,
                sli_type=sli_config.type,
                value=value,
                threshold=sli_config.threshold,
                is_good=is_good,
                calculation_time_ms=calculation_time
            )
            
            # Update metrics
            with sli_calculation_duration.labels(
                service=service,
                sli_type=sli_config.type.value
            ).time():
                pass  # Metric is recorded automatically
            
            # Store in history
            if service not in self.sli_history:
                self.sli_history[service] = []
            self.sli_history[service].append(result)
            
            # Trim history to last 30 days
            cutoff = datetime.utcnow() - timedelta(days=30)
            self.sli_history[service] = [
                r for r in self.sli_history[service]
                if r.timestamp > cutoff
            ]
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to calculate SLI",
                service=service,
                sli=sli_name,
                error=str(e)
            )
            raise
    
    def _evaluate_sli(self, value: float, threshold: float, sli_type: SLIType) -> bool:
        """
        Evaluate if an SLI value meets its objective.
        
        Args:
            value: Current SLI value
            threshold: SLI threshold
            sli_type: Type of SLI
            
        Returns:
            True if SLI is meeting objective
        """
        if sli_type == SLIType.AVAILABILITY:
            return value >= threshold
        elif sli_type == SLIType.LATENCY:
            return value <= threshold
        elif sli_type == SLIType.ERROR_RATE:
            return value <= threshold
        elif sli_type == SLIType.THROUGHPUT:
            return value >= threshold
        elif sli_type == SLIType.SATURATION:
            return value <= threshold
        else:
            return False
    
    async def _query_prometheus(self, query: str) -> float:
        """
        Query Prometheus for metric value.
        
        This is a placeholder that would be replaced with actual
        Prometheus client query in production.
        """
        # Simulate realistic values for different metric types
        import random
        
        if "availability" in query or "up{" in query:
            return 0.995 + random.uniform(-0.01, 0.005)
        elif "latency" in query or "duration_seconds" in query:
            return 0.05 + random.uniform(-0.02, 0.1)
        elif "error" in query or "status=~\"5..\"" in query:
            return 0.001 + random.uniform(-0.0005, 0.005)
        elif "throughput" in query or "rate" in query:
            return 1000 + random.uniform(-100, 200)
        else:
            return random.uniform(0, 1)
    
    def calculate_error_budget(
        self,
        service: str,
        window: timedelta
    ) -> ErrorBudget:
        """
        Calculate error budget for a service over a time window.
        
        Args:
            service: Service name
            window: Time window for calculation
            
        Returns:
            ErrorBudget with consumption and burn rate
        """
        if service not in self.sli_history:
            raise ValueError(f"No SLI history for service: {service}")
        
        slo_config = self.slo_configs[service]
        cutoff = datetime.utcnow() - window
        
        # Get relevant SLI results
        relevant_results = [
            r for r in self.sli_history[service]
            if r.timestamp > cutoff
        ]
        
        if not relevant_results:
            return ErrorBudget(
                service=service,
                window=window,
                total_budget=1.0,
                consumed_budget=0.0,
                remaining_budget=1.0,
                remaining_ratio=1.0,
                burn_rate=0.0
            )
        
        # Calculate good/bad events
        total_events = len(relevant_results)
        bad_events = sum(1 for r in relevant_results if not r.is_good)
        
        # Assume 99.9% SLO for simplicity (would come from config)
        slo_target = 0.999
        allowed_failures = total_events * (1 - slo_target)
        
        consumed_budget = bad_events / allowed_failures if allowed_failures > 0 else 0
        remaining_budget = max(0, 1 - consumed_budget)
        
        # Calculate burn rate (rate of budget consumption)
        # Burn rate = (bad events / time) / (allowed rate)
        time_hours = window.total_seconds() / 3600
        if time_hours > 0:
            actual_failure_rate = bad_events / time_hours
            allowed_failure_rate = allowed_failures / (window.days * 24)
            burn_rate = actual_failure_rate / allowed_failure_rate if allowed_failure_rate > 0 else 0
        else:
            burn_rate = 0
        
        # Calculate time until exhaustion
        time_until_exhaustion = None
        if burn_rate > 1:
            hours_remaining = (remaining_budget * window.total_seconds() / 3600) / (burn_rate - 1)
            if hours_remaining > 0:
                time_until_exhaustion = timedelta(hours=hours_remaining)
        
        budget = ErrorBudget(
            service=service,
            window=window,
            total_budget=1.0,
            consumed_budget=consumed_budget,
            remaining_budget=remaining_budget,
            remaining_ratio=remaining_budget,
            burn_rate=burn_rate,
            time_until_exhaustion=time_until_exhaustion
        )
        
        # Update Prometheus metrics
        error_budget_remaining_gauge.labels(
            service=service,
            window=f"{window.days}d"
        ).set(remaining_budget)
        
        error_budget_burn_rate_gauge.labels(
            service=service,
            window=f"{window.days}d"
        ).set(burn_rate)
        
        return budget
    
    def check_burn_rate_alerts(self, service: str) -> List[Tuple[AlertSeverity, str]]:
        """
        Check if any burn rate thresholds are exceeded.
        
        Args:
            service: Service name
            
        Returns:
            List of triggered alerts with severity and message
        """
        alerts = []
        slo_config = self.slo_configs[service]
        
        for threshold in slo_config.burn_rate_thresholds:
            window = timedelta(minutes=threshold.window_minutes)
            budget = self.calculate_error_budget(service, window)
            
            if budget.burn_rate >= threshold.rate:
                message = (
                    f"Error budget burn rate for {service} is {budget.burn_rate:.2f}x "
                    f"over {threshold.window_minutes} minutes (threshold: {threshold.rate}x). "
                    f"Remaining budget: {budget.remaining_ratio:.1%}"
                )
                
                if budget.time_until_exhaustion:
                    hours = budget.time_until_exhaustion.total_seconds() / 3600
                    message += f" Time until exhaustion: {hours:.1f} hours"
                
                alerts.append((threshold.severity, message))
                
                logger.warning(
                    "Burn rate threshold exceeded",
                    service=service,
                    burn_rate=budget.burn_rate,
                    threshold=threshold.rate,
                    severity=threshold.severity.value
                )
        
        return alerts
    
    async def evaluate_slos(self) -> Dict[str, List[SLIResult]]:
        """
        Evaluate all configured SLOs.
        
        Returns:
            Dictionary of service -> SLI results
        """
        results = {}
        
        for service, slo_config in self.slo_configs.items():
            service_results = []
            
            for sli_name, sli_config in slo_config.slis.items():
                try:
                    result = await self.calculate_sli(service, sli_name, sli_config)
                    service_results.append(result)
                    
                    # Update compliance metric
                    compliance = 1.0 if result.is_good else 0.0
                    slo_compliance_gauge.labels(
                        service=service,
                        sli_type=sli_config.type.value,
                        window="current"
                    ).set(compliance)
                    
                    # Track evaluation
                    slo_evaluation_counter.labels(
                        service=service,
                        result="success" if result.is_good else "failure"
                    ).inc()
                    
                except Exception as e:
                    logger.error(
                        "Failed to evaluate SLI",
                        service=service,
                        sli=sli_name,
                        error=str(e)
                    )
                    
                    slo_evaluation_counter.labels(
                        service=service,
                        result="error"
                    ).inc()
            
            results[service] = service_results
            
            # Check burn rate alerts
            self.check_burn_rate_alerts(service)
        
        return results
    
    async def start(self) -> None:
        """Start the SLO tracking loop."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._tracking_loop())
        logger.info("SLO tracker started")
    
    async def stop(self) -> None:
        """Stop the SLO tracking loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SLO tracker stopped")
    
    async def _tracking_loop(self) -> None:
        """Main tracking loop that evaluates SLOs periodically."""
        while self._running:
            try:
                await self.evaluate_slos()
                await asyncio.sleep(60)  # Evaluate every minute
            except Exception as e:
                logger.error("Error in SLO tracking loop", error=str(e))
                await asyncio.sleep(60)
    
    def get_slo_summary(self, service: str) -> Dict[str, Any]:
        """
        Get a summary of SLO status for a service.
        
        Args:
            service: Service name
            
        Returns:
            Dictionary with SLO summary data
        """
        if service not in self.sli_history:
            return {"error": "No data available"}
        
        # Calculate compliance over different windows
        windows = [
            ("1h", timedelta(hours=1)),
            ("24h", timedelta(days=1)),
            ("7d", timedelta(days=7)),
            ("30d", timedelta(days=30))
        ]
        
        summary = {
            "service": service,
            "compliance": {},
            "error_budgets": {},
            "current_slis": {}
        }
        
        for window_name, window_delta in windows:
            cutoff = datetime.utcnow() - window_delta
            relevant_results = [
                r for r in self.sli_history[service]
                if r.timestamp > cutoff
            ]
            
            if relevant_results:
                good_count = sum(1 for r in relevant_results if r.is_good)
                compliance_ratio = good_count / len(relevant_results)
                summary["compliance"][window_name] = compliance_ratio
                
                # Calculate error budget for this window
                budget = self.calculate_error_budget(service, window_delta)
                summary["error_budgets"][window_name] = {
                    "remaining": budget.remaining_ratio,
                    "burn_rate": budget.burn_rate,
                    "time_until_exhaustion": (
                        budget.time_until_exhaustion.total_seconds() / 3600
                        if budget.time_until_exhaustion else None
                    )
                }
        
        # Get latest SLI values
        latest_by_type = {}
        for result in self.sli_history[service]:
            sli_type = result.sli_type.value
            if sli_type not in latest_by_type or result.timestamp > latest_by_type[sli_type].timestamp:
                latest_by_type[sli_type] = result
        
        for sli_type, result in latest_by_type.items():
            summary["current_slis"][sli_type] = {
                "value": result.value,
                "threshold": result.threshold,
                "is_good": result.is_good,
                "timestamp": result.timestamp.isoformat()
            }
        
        return summary