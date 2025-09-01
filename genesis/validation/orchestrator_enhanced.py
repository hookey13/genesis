"""Enhanced validation orchestrator with full dependency management."""

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
import yaml

from genesis.validation.base import (
    CheckStatus,
    ValidationContext,
    ValidationMetadata,
    ValidationResult,
    Validator,
    ValidationEvidence
)
from genesis.validation.exceptions import (
    ValidationConfigError,
    ValidationDependencyError,
    ValidationOverrideError,
    ValidationPipelineError,
    ValidationRetryExhausted,
    ValidationTimeout
)
from genesis.validation.report import ReportGenerator

logger = structlog.get_logger(__name__)


class DependencyGraph:
    """Manages validator dependencies with topological sorting."""
    
    def __init__(self):
        """Initialize dependency graph."""
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def add_dependency(self, validator_id: str, depends_on: str) -> None:
        """Add a dependency relationship.
        
        Args:
            validator_id: Validator that has the dependency
            depends_on: Validator that is depended upon
        """
        self.graph[validator_id].add(depends_on)
        self.reverse_graph[depends_on].add(validator_id)
    
    def remove_dependency(self, validator_id: str, depends_on: str) -> None:
        """Remove a dependency relationship.
        
        Args:
            validator_id: Validator that has the dependency
            depends_on: Validator that is depended upon
        """
        if depends_on in self.graph.get(validator_id, set()):
            self.graph[validator_id].discard(depends_on)
        if validator_id in self.reverse_graph.get(depends_on, set()):
            self.reverse_graph[depends_on].discard(validator_id)
    
    def get_dependencies(self, validator_id: str) -> Set[str]:
        """Get direct dependencies of a validator.
        
        Args:
            validator_id: Validator ID
            
        Returns:
            Set of validator IDs this validator depends on
        """
        return self.graph.get(validator_id, set())
    
    def get_dependents(self, validator_id: str) -> Set[str]:
        """Get validators that depend on this one.
        
        Args:
            validator_id: Validator ID
            
        Returns:
            Set of validator IDs that depend on this validator
        """
        return self.reverse_graph.get(validator_id, set())
    
    def topological_sort(self, validator_ids: Set[str]) -> List[List[str]]:
        """Sort validators by dependencies into execution levels.
        
        Args:
            validator_ids: Set of validator IDs to sort
            
        Returns:
            List of levels, each containing validator IDs that can run in parallel
            
        Raises:
            ValidationDependencyError: If there's a circular dependency
        """
        # Build in-degree map for the subgraph
        in_degree = {}
        for vid in validator_ids:
            deps = self.graph.get(vid, set()) & validator_ids
            in_degree[vid] = len(deps)
        
        # Find validators with no dependencies
        queue = [vid for vid, degree in in_degree.items() if degree == 0]
        levels = []
        processed = set()
        
        while queue:
            current_level = queue[:]
            levels.append(current_level)
            processed.update(current_level)
            
            next_queue = []
            for vid in current_level:
                # Check dependents
                for dependent in self.reverse_graph.get(vid, set()):
                    if dependent in validator_ids and dependent not in processed:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            next_queue.append(dependent)
            
            queue = next_queue
        
        # Check for circular dependencies
        if len(processed) != len(validator_ids):
            missing = validator_ids - processed
            raise ValidationDependencyError(
                f"Circular dependency detected involving validators: {missing}",
                missing_dependencies=list(missing)
            )
        
        return levels
    
    def has_circular_dependency(self, validator_ids: Set[str]) -> bool:
        """Check if there's a circular dependency in the given validators.
        
        Args:
            validator_ids: Set of validator IDs to check
            
        Returns:
            True if circular dependency exists
        """
        try:
            self.topological_sort(validator_ids)
            return False
        except ValidationDependencyError:
            return True


class ValidationOrchestrator:
    """Orchestrates validation execution with full dependency management."""
    
    def __init__(self, genesis_root: Optional[Path] = None):
        """Initialize orchestrator.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.validators: Dict[str, Validator] = {}
        self.dependency_graph = DependencyGraph()
        self.results: Dict[str, ValidationResult] = {}
        self.pipeline_config: Dict[str, Any] = {}
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self.run_id = str(uuid.uuid4())
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.report_generator = ReportGenerator(self.genesis_root)
        self.historical_results: List[Dict[str, Any]] = []
    
    def register_validator(self, validator: Validator) -> None:
        """Register a validator with the orchestrator.
        
        Args:
            validator: Validator instance to register
        """
        self.validators[validator.validator_id] = validator
        
        # Register dependencies
        for dep_id in validator.dependencies:
            self.dependency_graph.add_dependency(validator.validator_id, dep_id)
        
        logger.info(
            "Registered validator",
            validator_id=validator.validator_id,
            dependencies=list(validator.dependencies)
        )
    
    def unregister_validator(self, validator_id: str) -> None:
        """Unregister a validator.
        
        Args:
            validator_id: ID of validator to unregister
        """
        if validator_id in self.validators:
            # Remove from dependency graph
            deps = self.dependency_graph.get_dependencies(validator_id)
            for dep in deps:
                self.dependency_graph.remove_dependency(validator_id, dep)
            
            # Remove validators that depend on this one
            dependents = self.dependency_graph.get_dependents(validator_id)
            for dependent in dependents:
                if dependent in self.validators:
                    self.validators[dependent].remove_dependency(validator_id)
            
            del self.validators[validator_id]
            logger.info("Unregistered validator", validator_id=validator_id)
    
    async def load_pipeline_config(self, config_path: Path) -> None:
        """Load pipeline configuration from YAML file.
        
        Args:
            config_path: Path to pipeline configuration file
            
        Raises:
            ValidationConfigError: If configuration is invalid
        """
        try:
            with open(config_path, 'r') as f:
                self.pipeline_config = yaml.safe_load(f)
            
            # Validate configuration
            self._validate_pipeline_config()
            
            logger.info("Loaded pipeline configuration", path=str(config_path))
        except Exception as e:
            raise ValidationConfigError(
                f"Failed to load pipeline configuration: {e}",
                config_key="pipeline_config",
                config_value=str(config_path)
            )
    
    def _validate_pipeline_config(self) -> None:
        """Validate pipeline configuration."""
        required_keys = ["version", "pipelines"]
        for key in required_keys:
            if key not in self.pipeline_config:
                raise ValidationConfigError(
                    f"Missing required configuration key: {key}",
                    config_key=key
                )
        
        # Check for circular dependencies in configured validators
        all_validator_ids = set()
        for pipeline in self.pipeline_config.get("pipelines", {}).values():
            if "validators" in pipeline:
                if isinstance(pipeline["validators"], list):
                    all_validator_ids.update(pipeline["validators"])
        
        if self.dependency_graph.has_circular_dependency(all_validator_ids):
            raise ValidationConfigError(
                "Circular dependency detected in pipeline configuration"
            )
    
    async def run_full_validation(
        self,
        mode: str = "standard",
        parallel: bool = True,
        dry_run: bool = False,
        force_continue: bool = False
    ) -> Dict[str, Any]:
        """Run full validation pipeline with dependency management.
        
        Args:
            mode: Validation mode ('quick', 'standard', 'comprehensive')
            parallel: Whether to run validators in parallel where possible
            dry_run: Whether to run in dry-run mode
            force_continue: Whether to continue on non-blocking failures
            
        Returns:
            Comprehensive validation report
        """
        self.start_time = datetime.utcnow()
        self.run_id = str(uuid.uuid4())
        
        logger.info(
            "Starting full validation",
            run_id=self.run_id,
            mode=mode,
            parallel=parallel,
            dry_run=dry_run
        )
        
        # Create validation context
        context = ValidationContext(
            genesis_root=str(self.genesis_root),
            environment="production",
            run_mode=mode,
            dry_run=dry_run,
            force_continue=force_continue,
            metadata=ValidationMetadata(
                version="1.0.0",
                environment="production",
                run_id=self.run_id,
                started_at=self.start_time
            )
        )
        
        # Get validators for the mode
        validator_ids = self._get_validators_for_mode(mode)
        
        # Run validators
        if parallel:
            await self._run_parallel(validator_ids, context)
        else:
            await self._run_sequential(validator_ids, context)
        
        self.end_time = datetime.utcnow()
        
        # Generate report
        report = self._generate_report()
        
        # Store for historical tracking
        self.historical_results.append(report)
        
        # Save report
        await self.save_results()
        
        logger.info(
            "Completed full validation",
            run_id=self.run_id,
            duration=(self.end_time - self.start_time).total_seconds()
        )
        
        return report
    
    def _get_validators_for_mode(self, mode: str) -> Set[str]:
        """Get validator IDs for a given mode.
        
        Args:
            mode: Validation mode
            
        Returns:
            Set of validator IDs to run
        """
        if mode == "quick":
            # Only critical validators
            return {
                vid for vid, v in self.validators.items()
                if v.is_critical
            }
        elif mode == "comprehensive":
            # All validators
            return set(self.validators.keys())
        else:  # standard
            # All non-optional validators
            return {
                vid for vid in self.validators.keys()
                if vid not in self.pipeline_config.get("optional_validators", [])
            }
    
    async def _run_parallel(
        self,
        validator_ids: Set[str],
        context: ValidationContext
    ) -> None:
        """Run validators in parallel respecting dependencies.
        
        Args:
            validator_ids: Set of validator IDs to run
            context: Validation context
        """
        # Get execution levels from dependency graph
        levels = self.dependency_graph.topological_sort(validator_ids)
        
        for level_num, level_validators in enumerate(levels):
            logger.info(
                f"Running validation level {level_num + 1}/{len(levels)}",
                validators=level_validators
            )
            
            # Run all validators in this level in parallel
            tasks = []
            for vid in level_validators:
                if vid in self.validators:
                    validator = self.validators[vid]
                    tasks.append(self._run_single_validator(validator, context))
            
            # Wait for all validators in this level to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check for blocking failures
            for vid, result in zip(level_validators, results):
                if isinstance(result, Exception):
                    logger.error(f"Validator {vid} failed with exception", error=str(result))
                    if self.validators[vid].is_critical and not context.force_continue:
                        raise ValidationPipelineError(
                            f"Critical validator {vid} failed",
                            stage=f"level_{level_num}",
                            failed_validators=[vid]
                        )
                elif isinstance(result, ValidationResult):
                    if result.has_blocking_failures() and not context.force_continue:
                        raise ValidationPipelineError(
                            f"Blocking failure in validator {vid}",
                            stage=f"level_{level_num}",
                            failed_validators=[vid]
                        )
    
    async def _run_sequential(
        self,
        validator_ids: Set[str],
        context: ValidationContext
    ) -> None:
        """Run validators sequentially respecting dependencies.
        
        Args:
            validator_ids: Set of validator IDs to run
            context: Validation context
        """
        # Get execution order from dependency graph
        levels = self.dependency_graph.topological_sort(validator_ids)
        
        for level_validators in levels:
            for vid in level_validators:
                if vid in self.validators:
                    validator = self.validators[vid]
                    result = await self._run_single_validator(validator, context)
                    
                    # Check for blocking failures
                    if result.has_blocking_failures() and not context.force_continue:
                        raise ValidationPipelineError(
                            f"Blocking failure in validator {vid}",
                            failed_validators=[vid]
                        )
    
    async def _run_single_validator(
        self,
        validator: Validator,
        context: ValidationContext
    ) -> ValidationResult:
        """Run a single validator with retry logic.
        
        Args:
            validator: Validator to run
            context: Validation context
            
        Returns:
            Validation result
        """
        vid = validator.validator_id
        logger.info(f"Running validator: {vid}")
        
        # Check for overrides
        if vid in self.overrides:
            logger.info(f"Validator {vid} has been overridden", override=self.overrides[vid])
            # Create override result
            result = ValidationResult(
                validator_id=vid,
                validator_name=validator.name,
                status=CheckStatus.SKIPPED,
                message=f"Overridden: {self.overrides[vid].get('reason', 'No reason provided')}",
                checks=[],
                evidence=ValidationEvidence(),
                metadata=context.metadata
            )
            self.results[vid] = result
            return result
        
        # Run with retry logic
        attempts = 0
        last_error = None
        
        while attempts <= validator.retry_count:
            try:
                # Run pre-validation hook
                await validator.pre_validation(context)
                
                # Run validation with timeout
                result = await asyncio.wait_for(
                    validator.run_validation(context),
                    timeout=validator.timeout_seconds
                )
                
                # Update counts and score
                result.update_counts()
                
                # Run post-validation hook
                await validator.post_validation(context, result)
                
                # Store result
                self.results[vid] = result
                
                logger.info(
                    f"Validator {vid} completed",
                    status=result.status.value,
                    score=str(result.score)
                )
                
                return result
                
            except asyncio.TimeoutError:
                last_error = ValidationTimeout(
                    f"Validator {vid} timed out after {validator.timeout_seconds}s",
                    validator_id=vid,
                    timeout_seconds=validator.timeout_seconds
                )
                logger.warning(f"Validator {vid} timed out, attempt {attempts + 1}")
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Validator {vid} failed, attempt {attempts + 1}",
                    error=str(e)
                )
            
            attempts += 1
            if attempts <= validator.retry_count:
                await asyncio.sleep(validator.retry_delay_seconds)
        
        # All retries exhausted
        raise ValidationRetryExhausted(
            f"Validator {vid} failed after {attempts} attempts",
            validator_id=vid,
            attempts=attempts,
            last_error=last_error
        )
    
    def add_override(
        self,
        validator_id: str,
        reason: str,
        authorized_by: str,
        authorization_level: str = "admin",
        required_level: str = "admin"
    ) -> None:
        """Add an override for a validator with authorization check.
        
        Args:
            validator_id: ID of validator to override
            reason: Reason for override
            authorized_by: Who authorized the override
            authorization_level: Level of authorization
            required_level: Required authorization level
            
        Raises:
            ValidationOverrideError: If authorization level insufficient
        """
        # Check authorization
        auth_levels = {"viewer": 0, "operator": 1, "admin": 2, "super_admin": 3}
        if auth_levels.get(authorization_level, 0) < auth_levels.get(required_level, 2):
            raise ValidationOverrideError(
                f"Insufficient authorization level for override",
                validator_id=validator_id,
                authorization_level=authorization_level,
                required_level=required_level
            )
        
        self.overrides[validator_id] = {
            "reason": reason,
            "authorized_by": authorized_by,
            "authorization_level": authorization_level,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info(
            f"Added override for validator {validator_id}",
            override=self.overrides[validator_id]
        )
    
    def remove_override(self, validator_id: str) -> None:
        """Remove an override.
        
        Args:
            validator_id: ID of validator to remove override for
        """
        if validator_id in self.overrides:
            del self.overrides[validator_id]
            logger.info(f"Removed override for validator {validator_id}")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report.
        
        Returns:
            Validation report dictionary
        """
        # Calculate overall status
        all_passed = all(
            r.status == CheckStatus.PASSED
            for r in self.results.values()
        )
        has_failures = any(
            r.status == CheckStatus.FAILED
            for r in self.results.values()
        )
        has_warnings = any(
            r.status == CheckStatus.WARNING
            for r in self.results.values()
        )
        
        if all_passed:
            overall_status = "passed"
        elif has_failures:
            overall_status = "failed"
        elif has_warnings:
            overall_status = "warning"
        else:
            overall_status = "unknown"
        
        # Calculate overall score
        total_score = sum(r.score for r in self.results.values())
        avg_score = total_score / len(self.results) if self.results else 0
        
        # Count blocking issues
        blocking_issues = []
        for vid, result in self.results.items():
            if result.has_blocking_failures():
                for check in result.checks:
                    if check.is_blocking and check.status in [CheckStatus.FAILED, CheckStatus.ERROR]:
                        blocking_issues.append({
                            "validator_id": vid,
                            "check_id": check.id,
                            "message": check.details
                        })
        
        return {
            "run_id": self.run_id,
            "timestamp": self.start_time.isoformat() if self.start_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time else None
            ),
            "overall_status": overall_status,
            "overall_score": float(avg_score),
            "validators_run": len(self.results),
            "validators_passed": sum(
                1 for r in self.results.values()
                if r.status == CheckStatus.PASSED
            ),
            "validators_failed": sum(
                1 for r in self.results.values()
                if r.status == CheckStatus.FAILED
            ),
            "validators_warning": sum(
                1 for r in self.results.values()
                if r.status == CheckStatus.WARNING
            ),
            "validators_skipped": sum(
                1 for r in self.results.values()
                if r.status == CheckStatus.SKIPPED
            ),
            "blocking_issues": blocking_issues,
            "overrides": self.overrides,
            "results": {
                vid: result.to_dict()
                for vid, result in self.results.items()
            }
        }
    
    async def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Save validation results to file.
        
        Args:
            output_path: Optional output path
            
        Returns:
            Path to saved results
        """
        if output_path is None:
            reports_dir = self.genesis_root / "docs" / "validation_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = reports_dir / f"validation_{self.run_id}_{timestamp}.yaml"
        
        report = self._generate_report()
        
        with open(output_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)
        
        logger.info("Saved validation results", path=str(output_path))
        return output_path
    
    def get_historical_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical validation results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of historical validation results
        """
        return self.historical_results[-limit:]
    
    def clear_results(self) -> None:
        """Clear current validation results."""
        self.results.clear()
        logger.info("Cleared validation results")
    
    def get_validator_status(self, validator_id: str) -> Optional[str]:
        """Get the status of a specific validator.
        
        Args:
            validator_id: Validator ID
            
        Returns:
            Status string or None if not found
        """
        if validator_id in self.results:
            return self.results[validator_id].status.value
        return None
    
    def get_blocking_validators(self) -> List[str]:
        """Get list of validators with blocking failures.
        
        Returns:
            List of validator IDs with blocking failures
        """
        blocking = []
        for vid, result in self.results.items():
            if result.has_blocking_failures():
                blocking.append(vid)
        return blocking