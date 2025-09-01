"""Configuration management for validation framework."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ValidatorConfig:
    """Configuration for a single validator."""

    id: str
    name: str
    enabled: bool = True
    timeout_seconds: int = 60
    retry_count: int = 0
    retry_delay_seconds: int = 5
    is_critical: bool = False
    is_blocking: bool = False
    dependencies: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStage:
    """Configuration for a pipeline stage."""

    name: str
    validators: list[str]
    parallel: bool = True
    continue_on_failure: bool = False
    timeout_minutes: int = 30


@dataclass
class PipelineConfig:
    """Configuration for a validation pipeline."""

    name: str
    description: str
    stages: list[PipelineStage]
    mode: str = "standard"  # quick, standard, comprehensive
    required_score: float = 80.0
    blocking_on_failure: bool = True
    allow_overrides: bool = True
    override_authorization_level: str = "admin"
    environment_specific: dict[str, Any] = field(default_factory=dict)


class ValidationConfig:
    """Manages validation framework configuration."""

    def __init__(self, config_path: Path | None = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or Path("config/validation_pipeline.yaml")
        self.validators: dict[str, ValidatorConfig] = {}
        self.pipelines: dict[str, PipelineConfig] = {}
        self.global_settings: dict[str, Any] = {}

        if self.config_path.exists():
            self.load_config()
        else:
            self.load_defaults()

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path) as f:
            data = yaml.safe_load(f)

        # Load global settings
        self.global_settings = data.get('global', {})

        # Load validator configurations
        validators_data = data.get('validators', {})
        for vid, vconfig in validators_data.items():
            self.validators[vid] = ValidatorConfig(
                id=vid,
                name=vconfig.get('name', vid),
                enabled=vconfig.get('enabled', True),
                timeout_seconds=vconfig.get('timeout_seconds', 60),
                retry_count=vconfig.get('retry_count', 0),
                retry_delay_seconds=vconfig.get('retry_delay_seconds', 5),
                is_critical=vconfig.get('is_critical', False),
                is_blocking=vconfig.get('is_blocking', False),
                dependencies=vconfig.get('dependencies', []),
                parameters=vconfig.get('parameters', {})
            )

        # Load pipeline configurations
        pipelines_data = data.get('pipelines', {})
        for pname, pconfig in pipelines_data.items():
            stages = []
            for stage_data in pconfig.get('stages', []):
                stages.append(PipelineStage(
                    name=stage_data.get('name'),
                    validators=stage_data.get('validators', []),
                    parallel=stage_data.get('parallel', True),
                    continue_on_failure=stage_data.get('continue_on_failure', False),
                    timeout_minutes=stage_data.get('timeout_minutes', 30)
                ))

            self.pipelines[pname] = PipelineConfig(
                name=pname,
                description=pconfig.get('description', ''),
                stages=stages,
                mode=pconfig.get('mode', 'standard'),
                required_score=pconfig.get('required_score', 80.0),
                blocking_on_failure=pconfig.get('blocking_on_failure', True),
                allow_overrides=pconfig.get('allow_overrides', True),
                override_authorization_level=pconfig.get('override_authorization_level', 'admin'),
                environment_specific=pconfig.get('environment_specific', {})
            )

    def load_defaults(self) -> None:
        """Load default configuration."""
        # Default global settings
        self.global_settings = {
            'version': '1.0.0',
            'default_timeout': 60,
            'max_parallel_validators': 10,
            'report_retention_days': 30,
            'enable_historical_tracking': True
        }

        # Default validator configurations
        self._add_default_validators()

        # Default pipeline configurations
        self._add_default_pipelines()

    def _add_default_validators(self) -> None:
        """Add default validator configurations."""
        default_validators = [
            ValidatorConfig(
                id='test_coverage',
                name='Test Coverage',
                is_critical=True,
                timeout_seconds=120
            ),
            ValidatorConfig(
                id='security_scan',
                name='Security Scanner',
                is_critical=True,
                is_blocking=True,
                timeout_seconds=300
            ),
            ValidatorConfig(
                id='performance',
                name='Performance Validator',
                is_critical=False,
                timeout_seconds=180
            ),
            ValidatorConfig(
                id='compliance',
                name='Compliance Checker',
                is_critical=True,
                is_blocking=True,
                timeout_seconds=60
            ),
            ValidatorConfig(
                id='operational',
                name='Operational Readiness',
                is_critical=True,
                timeout_seconds=120,
                dependencies=['test_coverage', 'security_scan']
            ),
            ValidatorConfig(
                id='disaster_recovery',
                name='Disaster Recovery',
                is_critical=False,
                timeout_seconds=300
            ),
            ValidatorConfig(
                id='paper_trading',
                name='Paper Trading Validator',
                is_critical=False,
                timeout_seconds=600,
                retry_count=2,
                retry_delay_seconds=10
            ),
            ValidatorConfig(
                id='stability',
                name='Stability Tester',
                is_critical=False,
                timeout_seconds=900,
                retry_count=1
            )
        ]

        for validator in default_validators:
            self.validators[validator.id] = validator

    def _add_default_pipelines(self) -> None:
        """Add default pipeline configurations."""
        # Quick pipeline
        self.pipelines['quick'] = PipelineConfig(
            name='quick',
            description='Quick validation for development',
            stages=[
                PipelineStage(
                    name='critical',
                    validators=['test_coverage', 'security_scan'],
                    parallel=True,
                    timeout_minutes=5
                )
            ],
            mode='quick',
            required_score=70.0,
            blocking_on_failure=False
        )

        # Standard pipeline
        self.pipelines['standard'] = PipelineConfig(
            name='standard',
            description='Standard validation',
            stages=[
                PipelineStage(
                    name='foundation',
                    validators=['test_coverage', 'security_scan'],
                    parallel=True,
                    timeout_minutes=10
                ),
                PipelineStage(
                    name='quality',
                    validators=['performance', 'compliance'],
                    parallel=True,
                    timeout_minutes=15
                ),
                PipelineStage(
                    name='operational',
                    validators=['operational'],
                    parallel=False,
                    timeout_minutes=10
                )
            ],
            mode='standard',
            required_score=80.0,
            blocking_on_failure=True
        )

        # Comprehensive pipeline
        self.pipelines['comprehensive'] = PipelineConfig(
            name='comprehensive',
            description='Full validation suite',
            stages=[
                PipelineStage(
                    name='foundation',
                    validators=['test_coverage', 'security_scan'],
                    parallel=True,
                    timeout_minutes=15
                ),
                PipelineStage(
                    name='quality',
                    validators=['performance', 'compliance'],
                    parallel=True,
                    timeout_minutes=20
                ),
                PipelineStage(
                    name='operational',
                    validators=['operational', 'disaster_recovery'],
                    parallel=True,
                    timeout_minutes=30
                ),
                PipelineStage(
                    name='business',
                    validators=['paper_trading', 'stability'],
                    parallel=True,
                    continue_on_failure=True,
                    timeout_minutes=60
                )
            ],
            mode='comprehensive',
            required_score=85.0,
            blocking_on_failure=True
        )

        # Go-live pipeline
        self.pipelines['go_live'] = PipelineConfig(
            name='go_live',
            description='Go-live readiness validation',
            stages=[
                PipelineStage(
                    name='critical',
                    validators=['test_coverage', 'security_scan', 'compliance'],
                    parallel=False,
                    timeout_minutes=20
                ),
                PipelineStage(
                    name='operational',
                    validators=['operational', 'disaster_recovery', 'performance'],
                    parallel=False,
                    timeout_minutes=40
                ),
                PipelineStage(
                    name='validation',
                    validators=['paper_trading', 'stability'],
                    parallel=False,
                    timeout_minutes=90
                )
            ],
            mode='comprehensive',
            required_score=95.0,
            blocking_on_failure=True,
            allow_overrides=False
        )

    def get_validator_config(self, validator_id: str) -> ValidatorConfig | None:
        """Get configuration for a specific validator.
        
        Args:
            validator_id: Validator identifier
            
        Returns:
            Validator configuration or None
        """
        return self.validators.get(validator_id)

    def get_pipeline_config(self, pipeline_name: str) -> PipelineConfig | None:
        """Get configuration for a specific pipeline.
        
        Args:
            pipeline_name: Pipeline name
            
        Returns:
            Pipeline configuration or None
        """
        return self.pipelines.get(pipeline_name)

    def save_config(self, output_path: Path | None = None) -> None:
        """Save current configuration to YAML file.
        
        Args:
            output_path: Optional output path (uses config_path if not provided)
        """
        output_path = output_path or self.config_path

        # Prepare data for YAML
        data = {
            'global': self.global_settings,
            'validators': {},
            'pipelines': {}
        }

        # Convert validators
        for vid, vconfig in self.validators.items():
            data['validators'][vid] = {
                'name': vconfig.name,
                'enabled': vconfig.enabled,
                'timeout_seconds': vconfig.timeout_seconds,
                'retry_count': vconfig.retry_count,
                'retry_delay_seconds': vconfig.retry_delay_seconds,
                'is_critical': vconfig.is_critical,
                'is_blocking': vconfig.is_blocking,
                'dependencies': vconfig.dependencies,
                'parameters': vconfig.parameters
            }

        # Convert pipelines
        for pname, pconfig in self.pipelines.items():
            stages_data = []
            for stage in pconfig.stages:
                stages_data.append({
                    'name': stage.name,
                    'validators': stage.validators,
                    'parallel': stage.parallel,
                    'continue_on_failure': stage.continue_on_failure,
                    'timeout_minutes': stage.timeout_minutes
                })

            data['pipelines'][pname] = {
                'description': pconfig.description,
                'stages': stages_data,
                'mode': pconfig.mode,
                'required_score': pconfig.required_score,
                'blocking_on_failure': pconfig.blocking_on_failure,
                'allow_overrides': pconfig.allow_overrides,
                'override_authorization_level': pconfig.override_authorization_level,
                'environment_specific': pconfig.environment_specific
            }

        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def validate_pipeline(self, pipeline_name: str) -> list[str]:
        """Validate a pipeline configuration.
        
        Args:
            pipeline_name: Pipeline name to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        pipeline = self.pipelines.get(pipeline_name)
        if not pipeline:
            errors.append(f"Pipeline '{pipeline_name}' not found")
            return errors

        # Check all validators exist
        for stage in pipeline.stages:
            for validator_id in stage.validators:
                if validator_id not in self.validators:
                    errors.append(f"Validator '{validator_id}' in stage '{stage.name}' not found")

        # Check dependency graph for cycles
        visited = set()
        rec_stack = set()

        def has_cycle(vid: str) -> bool:
            visited.add(vid)
            rec_stack.add(vid)

            validator = self.validators.get(vid)
            if validator:
                for dep in validator.dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(vid)
            return False

        for validator_id in self.validators:
            if validator_id not in visited:
                if has_cycle(validator_id):
                    errors.append(f"Circular dependency detected involving '{validator_id}'")

        return errors
