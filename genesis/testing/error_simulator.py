"""
Error simulation framework for chaos engineering and testing.

Provides controlled failure injection for testing error handling, recovery
procedures, and system resilience under various failure conditions.
"""

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type

import structlog

from genesis.core.exceptions import (
    BaseError,
    NetworkError,
    ConnectionTimeout,
    RateLimitError,
    DatabaseLocked,
    OrderRejected,
    ValidationError,
    TiltInterventionRequired,
    RiskLimitExceeded,
)


class SimulationMode(Enum):
    """Modes for error simulation."""
    
    DISABLED = "disabled"  # No simulation
    DETERMINISTIC = "deterministic"  # Fixed sequence of failures
    PROBABILISTIC = "probabilistic"  # Random failures based on probability
    SCHEDULED = "scheduled"  # Failures at specific times
    CHAOS = "chaos"  # Maximum chaos mode


class FailureType(Enum):
    """Types of failures that can be simulated."""
    
    NETWORK_TIMEOUT = "network_timeout"
    RATE_LIMIT = "rate_limit"
    DATABASE_LOCK = "database_lock"
    ORDER_REJECTION = "order_rejection"
    VALIDATION_ERROR = "validation_error"
    MEMORY_PRESSURE = "memory_pressure"
    CPU_SPIKE = "cpu_spike"
    DISK_FULL = "disk_full"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_CORRUPTION = "data_corruption"
    TILT_TRIGGER = "tilt_trigger"
    RISK_VIOLATION = "risk_violation"
    WEBSOCKET_DISCONNECT = "websocket_disconnect"
    AUTHENTICATION_FAILURE = "authentication_failure"
    PARTIAL_FAILURE = "partial_failure"  # Some operations succeed, some fail


@dataclass
class FailureScenario:
    """Defines a failure scenario for simulation."""
    
    name: str
    description: str
    failure_type: FailureType
    exception_type: Type[Exception]
    probability: float = 0.1  # Probability of failure (0.0 to 1.0)
    duration: Optional[timedelta] = None  # How long failure persists
    cooldown: Optional[timedelta] = None  # Time before can trigger again
    max_occurrences: Optional[int] = None  # Max times to trigger
    conditions: Dict[str, Any] = field(default_factory=dict)  # Trigger conditions
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    occurrence_count: int = 0
    last_triggered: Optional[datetime] = None
    active_until: Optional[datetime] = None


@dataclass
class SimulationResult:
    """Result of a simulation attempt."""
    
    triggered: bool
    scenario_name: Optional[str] = None
    failure_type: Optional[FailureType] = None
    exception: Optional[Exception] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorSimulator:
    """
    Error simulation framework for controlled failure injection.
    
    Provides:
    - Multiple simulation modes (deterministic, probabilistic, chaos)
    - Configurable failure scenarios
    - Conditional failure triggers
    - Failure injection hooks
    - Simulation metrics and reporting
    """
    
    def __init__(
        self,
        mode: SimulationMode = SimulationMode.DISABLED,
        logger: Optional[structlog.BoundLogger] = None,
    ):
        self.mode = mode
        self.logger = logger or structlog.get_logger(__name__)
        
        # Scenario registry
        self._scenarios: Dict[str, FailureScenario] = {}
        self._active_scenarios: Set[str] = set()
        
        # Injection points
        self._injection_points: Dict[str, List[Callable]] = {}
        
        # Simulation history
        self._simulation_history: List[SimulationResult] = []
        self._failure_counts: Dict[FailureType, int] = {}
        
        # Chaos mode settings
        self._chaos_intensity: float = 0.1  # Base probability in chaos mode
        
        # Initialize default scenarios
        self._initialize_default_scenarios()
    
    def _initialize_default_scenarios(self):
        """Initialize default failure scenarios."""
        
        # Network timeout scenario
        self.register_scenario(
            FailureScenario(
                name="network_timeout",
                description="Simulate network timeout",
                failure_type=FailureType.NETWORK_TIMEOUT,
                exception_type=ConnectionTimeout,
                probability=0.05,
                duration=timedelta(seconds=30),
                cooldown=timedelta(minutes=1),
            )
        )
        
        # Rate limit scenario
        self.register_scenario(
            FailureScenario(
                name="rate_limit_burst",
                description="Simulate rate limit during burst",
                failure_type=FailureType.RATE_LIMIT,
                exception_type=RateLimitError,
                probability=0.1,
                duration=timedelta(seconds=60),
                cooldown=timedelta(minutes=5),
                conditions={"request_rate": "high"},
            )
        )
        
        # Database lock scenario
        self.register_scenario(
            FailureScenario(
                name="database_contention",
                description="Simulate database lock contention",
                failure_type=FailureType.DATABASE_LOCK,
                exception_type=DatabaseLocked,
                probability=0.02,
                duration=timedelta(seconds=5),
                max_occurrences=10,
            )
        )
        
        # Order rejection scenario
        self.register_scenario(
            FailureScenario(
                name="order_rejection_random",
                description="Random order rejections",
                failure_type=FailureType.ORDER_REJECTION,
                exception_type=OrderRejected,
                probability=0.01,
                metadata={"reason": "SIMULATED_REJECTION"},
            )
        )
        
        # Tilt trigger scenario
        self.register_scenario(
            FailureScenario(
                name="tilt_detection",
                description="Trigger tilt intervention",
                failure_type=FailureType.TILT_TRIGGER,
                exception_type=TiltInterventionRequired,
                probability=0.001,
                cooldown=timedelta(hours=1),
                metadata={
                    "tilt_score": 0.85,
                    "threshold": 0.75,
                },
            )
        )
        
        # Risk violation scenario
        self.register_scenario(
            FailureScenario(
                name="risk_limit_breach",
                description="Simulate risk limit violation",
                failure_type=FailureType.RISK_VIOLATION,
                exception_type=RiskLimitExceeded,
                probability=0.005,
                conditions={"position_size": "large"},
            )
        )
        
        # WebSocket disconnect scenario
        self.register_scenario(
            FailureScenario(
                name="websocket_flap",
                description="WebSocket connection flapping",
                failure_type=FailureType.WEBSOCKET_DISCONNECT,
                exception_type=ConnectionError,
                probability=0.02,
                duration=timedelta(seconds=10),
                cooldown=timedelta(minutes=10),
            )
        )
    
    def register_scenario(self, scenario: FailureScenario):
        """
        Register a failure scenario.
        
        Args:
            scenario: Failure scenario to register
        """
        self._scenarios[scenario.name] = scenario
        
        self.logger.info(
            "Registered failure scenario",
            name=scenario.name,
            failure_type=scenario.failure_type.value,
            probability=scenario.probability,
        )
    
    def activate_scenario(self, scenario_name: str):
        """
        Activate a failure scenario.
        
        Args:
            scenario_name: Name of scenario to activate
        """
        if scenario_name in self._scenarios:
            self._active_scenarios.add(scenario_name)
            self.logger.info(
                "Activated failure scenario",
                scenario=scenario_name,
            )
    
    def deactivate_scenario(self, scenario_name: str):
        """
        Deactivate a failure scenario.
        
        Args:
            scenario_name: Name of scenario to deactivate
        """
        self._active_scenarios.discard(scenario_name)
        self.logger.info(
            "Deactivated failure scenario",
            scenario=scenario_name,
        )
    
    def set_mode(self, mode: SimulationMode):
        """
        Set simulation mode.
        
        Args:
            mode: New simulation mode
        """
        old_mode = self.mode
        self.mode = mode
        
        self.logger.warning(
            "Changed simulation mode",
            old_mode=old_mode.value,
            new_mode=mode.value,
        )
        
        if mode == SimulationMode.CHAOS:
            self.logger.warning(
                "CHAOS MODE ACTIVATED - Expect random failures",
                intensity=self._chaos_intensity,
            )
    
    def should_fail(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """
        Check if an operation should fail based on simulation settings.
        
        Args:
            operation: Name of the operation
            context: Operation context for conditional checks
            
        Returns:
            Simulation result with failure details if triggered
        """
        if self.mode == SimulationMode.DISABLED:
            return SimulationResult(triggered=False)
        
        context = context or {}
        
        # Check each active scenario
        for scenario_name in self._active_scenarios:
            scenario = self._scenarios.get(scenario_name)
            if not scenario:
                continue
            
            # Check if scenario should trigger
            if self._should_trigger_scenario(scenario, operation, context):
                # Create exception
                exception = self._create_exception(scenario)
                
                # Update tracking
                scenario.occurrence_count += 1
                scenario.last_triggered = datetime.utcnow()
                
                if scenario.duration:
                    scenario.active_until = datetime.utcnow() + scenario.duration
                
                # Record simulation
                result = SimulationResult(
                    triggered=True,
                    scenario_name=scenario.name,
                    failure_type=scenario.failure_type,
                    exception=exception,
                    metadata={
                        "operation": operation,
                        "context": context,
                    },
                )
                
                self._record_simulation(result)
                
                self.logger.warning(
                    "Simulated failure triggered",
                    scenario=scenario.name,
                    operation=operation,
                    failure_type=scenario.failure_type.value,
                )
                
                return result
        
        # Chaos mode - random failures
        if self.mode == SimulationMode.CHAOS:
            if random.random() < self._chaos_intensity:
                # Pick random failure type
                failure_type = random.choice(list(FailureType))
                exception = self._create_random_exception(failure_type)
                
                result = SimulationResult(
                    triggered=True,
                    scenario_name="chaos",
                    failure_type=failure_type,
                    exception=exception,
                    metadata={
                        "operation": operation,
                        "chaos_mode": True,
                    },
                )
                
                self._record_simulation(result)
                
                self.logger.warning(
                    "Chaos mode failure triggered",
                    operation=operation,
                    failure_type=failure_type.value,
                )
                
                return result
        
        return SimulationResult(triggered=False)
    
    def _should_trigger_scenario(
        self,
        scenario: FailureScenario,
        operation: str,
        context: Dict[str, Any],
    ) -> bool:
        """Check if a scenario should trigger."""
        
        # Check if still active from previous trigger
        if scenario.active_until and datetime.utcnow() < scenario.active_until:
            return True
        
        # Check cooldown
        if scenario.last_triggered and scenario.cooldown:
            if datetime.utcnow() < scenario.last_triggered + scenario.cooldown:
                return False
        
        # Check max occurrences
        if scenario.max_occurrences and scenario.occurrence_count >= scenario.max_occurrences:
            return False
        
        # Check conditions
        for key, expected_value in scenario.conditions.items():
            if context.get(key) != expected_value:
                return False
        
        # Check probability
        if self.mode == SimulationMode.DETERMINISTIC:
            # In deterministic mode, trigger in sequence
            return scenario.occurrence_count == 0
        elif self.mode == SimulationMode.PROBABILISTIC:
            return random.random() < scenario.probability
        elif self.mode == SimulationMode.SCHEDULED:
            # Would check schedule here
            return False
        
        return False
    
    def _create_exception(self, scenario: FailureScenario) -> Exception:
        """Create exception for a scenario."""
        exception_class = scenario.exception_type
        
        # Create appropriate exception based on type
        if exception_class == ConnectionTimeout:
            return ConnectionTimeout(
                "Simulated connection timeout",
                timeout_seconds=30.0,
            )
        elif exception_class == RateLimitError:
            return RateLimitError(
                "Simulated rate limit",
                retry_after_seconds=60,
            )
        elif exception_class == DatabaseLocked:
            return DatabaseLocked(
                "Simulated database lock",
                table="simulated_table",
            )
        elif exception_class == OrderRejected:
            return OrderRejected(
                "Simulated order rejection",
                reason=scenario.metadata.get("reason", "SIMULATION"),
            )
        elif exception_class == TiltInterventionRequired:
            return TiltInterventionRequired(
                "Simulated tilt detection",
                tilt_score=scenario.metadata.get("tilt_score", 0.85),
                threshold=scenario.metadata.get("threshold", 0.75),
            )
        elif exception_class == RiskLimitExceeded:
            from decimal import Decimal
            return RiskLimitExceeded(
                "Simulated risk limit exceeded",
                limit_type="position_size",
                current_value=Decimal("1000"),
                limit_value=Decimal("500"),
            )
        else:
            # Generic exception
            return exception_class(f"Simulated {scenario.failure_type.value}")
    
    def _create_random_exception(self, failure_type: FailureType) -> Exception:
        """Create random exception for chaos mode."""
        exception_map = {
            FailureType.NETWORK_TIMEOUT: ConnectionTimeout(
                "Chaos: Network timeout",
                timeout_seconds=random.randint(10, 60),
            ),
            FailureType.RATE_LIMIT: RateLimitError(
                "Chaos: Rate limit",
                retry_after_seconds=random.randint(30, 300),
            ),
            FailureType.DATABASE_LOCK: DatabaseLocked(
                "Chaos: Database locked",
            ),
            FailureType.ORDER_REJECTION: OrderRejected(
                "Chaos: Order rejected",
            ),
            FailureType.VALIDATION_ERROR: ValidationError(
                "Chaos: Validation failed",
            ),
        }
        
        return exception_map.get(
            failure_type,
            BaseError(f"Chaos: {failure_type.value}"),
        )
    
    def _record_simulation(self, result: SimulationResult):
        """Record simulation result."""
        self._simulation_history.append(result)
        
        # Keep history bounded
        if len(self._simulation_history) > 10000:
            self._simulation_history = self._simulation_history[-10000:]
        
        # Update counters
        if result.failure_type:
            self._failure_counts[result.failure_type] = (
                self._failure_counts.get(result.failure_type, 0) + 1
            )
    
    def inject_failure(
        self,
        injection_point: str,
        failure_callback: Callable,
    ):
        """
        Register a failure injection point.
        
        Args:
            injection_point: Name of injection point
            failure_callback: Callback to trigger failure
        """
        if injection_point not in self._injection_points:
            self._injection_points[injection_point] = []
        
        self._injection_points[injection_point].append(failure_callback)
        
        self.logger.debug(
            "Registered failure injection point",
            point=injection_point,
        )
    
    async def trigger_injection_point(
        self,
        injection_point: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Trigger failure at injection point.
        
        Args:
            injection_point: Name of injection point
            context: Context for failure injection
        """
        if injection_point not in self._injection_points:
            return
        
        for callback in self._injection_points[injection_point]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context)
                else:
                    callback(context)
            except Exception as e:
                self.logger.error(
                    "Failed to trigger injection point",
                    point=injection_point,
                    error=str(e),
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        total_simulations = len(self._simulation_history)
        
        if total_simulations == 0:
            return {
                "mode": self.mode.value,
                "active_scenarios": len(self._active_scenarios),
                "total_simulations": 0,
                "failure_distribution": {},
            }
        
        # Calculate failure distribution
        failure_distribution = {
            ft.value: count / total_simulations
            for ft, count in self._failure_counts.items()
        }
        
        # Get scenario statistics
        scenario_stats = {}
        for name, scenario in self._scenarios.items():
            if scenario.occurrence_count > 0:
                scenario_stats[name] = {
                    "occurrences": scenario.occurrence_count,
                    "last_triggered": scenario.last_triggered.isoformat()
                    if scenario.last_triggered else None,
                }
        
        return {
            "mode": self.mode.value,
            "active_scenarios": len(self._active_scenarios),
            "total_simulations": total_simulations,
            "failure_distribution": failure_distribution,
            "scenario_statistics": scenario_stats,
            "chaos_intensity": self._chaos_intensity if self.mode == SimulationMode.CHAOS else None,
        }
    
    def reset(self):
        """Reset simulation state."""
        # Reset scenario tracking
        for scenario in self._scenarios.values():
            scenario.occurrence_count = 0
            scenario.last_triggered = None
            scenario.active_until = None
        
        # Clear history
        self._simulation_history.clear()
        self._failure_counts.clear()
        
        self.logger.info("Reset error simulator")
    
    def create_test_harness(self) -> "TestHarness":
        """
        Create a test harness for structured testing.
        
        Returns:
            Test harness instance
        """
        return TestHarness(self)


class TestHarness:
    """
    Test harness for structured error simulation testing.
    
    Provides utilities for creating and running test scenarios.
    """
    
    def __init__(self, simulator: ErrorSimulator):
        self.simulator = simulator
        self.logger = structlog.get_logger(__name__)
        self.test_results: List[Dict[str, Any]] = []
    
    async def run_scenario_test(
        self,
        scenario_name: str,
        test_function: Callable,
        iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Run test with specific scenario.
        
        Args:
            scenario_name: Scenario to test
            test_function: Function to test
            iterations: Number of test iterations
            
        Returns:
            Test results
        """
        # Activate scenario
        self.simulator.activate_scenario(scenario_name)
        
        failures = 0
        successes = 0
        errors = []
        
        for i in range(iterations):
            try:
                if asyncio.iscoroutinefunction(test_function):
                    await test_function()
                else:
                    test_function()
                successes += 1
            except Exception as e:
                failures += 1
                errors.append(str(e))
        
        # Deactivate scenario
        self.simulator.deactivate_scenario(scenario_name)
        
        result = {
            "scenario": scenario_name,
            "iterations": iterations,
            "successes": successes,
            "failures": failures,
            "failure_rate": failures / iterations,
            "unique_errors": len(set(errors)),
        }
        
        self.test_results.append(result)
        
        return result
    
    async def run_chaos_test(
        self,
        test_function: Callable,
        duration_seconds: int = 60,
        intensity: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run chaos test for specified duration.
        
        Args:
            test_function: Function to test
            duration_seconds: Test duration
            intensity: Chaos intensity (0.0 to 1.0)
            
        Returns:
            Test results
        """
        # Set chaos mode
        original_mode = self.simulator.mode
        original_intensity = self.simulator._chaos_intensity
        
        self.simulator.set_mode(SimulationMode.CHAOS)
        self.simulator._chaos_intensity = intensity
        
        start_time = asyncio.get_event_loop().time()
        iterations = 0
        failures = 0
        
        while asyncio.get_event_loop().time() - start_time < duration_seconds:
            iterations += 1
            try:
                if asyncio.iscoroutinefunction(test_function):
                    await test_function()
                else:
                    test_function()
            except Exception:
                failures += 1
            
            await asyncio.sleep(0.1)  # Small delay between iterations
        
        # Restore original mode
        self.simulator.set_mode(original_mode)
        self.simulator._chaos_intensity = original_intensity
        
        result = {
            "test_type": "chaos",
            "duration_seconds": duration_seconds,
            "intensity": intensity,
            "iterations": iterations,
            "failures": failures,
            "failure_rate": failures / iterations if iterations > 0 else 0,
        }
        
        self.test_results.append(result)
        
        return result