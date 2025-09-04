"""Configuration Validation Module.

This module provides comprehensive validation for strategy configurations,
including type checking, constraint validation, and schema enforcement.
"""

import re
from decimal import Decimal
from typing import Any

import structlog
from pydantic import BaseModel, Field, validator

logger = structlog.get_logger(__name__)


class ValidationResult(BaseModel):
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    validated_config: dict[str, Any] | None = None


class FieldConstraint(BaseModel):
    """Defines constraints for a configuration field."""

    field_type: str  # int, float, decimal, str, bool, list, dict
    required: bool = True
    min_value: int | float | Decimal | None = None
    max_value: int | float | Decimal | None = None
    min_length: int | None = None
    max_length: int | None = None
    regex_pattern: str | None = None
    allowed_values: list[Any] | None = None
    default_value: Any | None = None

    @validator("regex_pattern")
    def validate_regex(cls, v):
        if v:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return v


class SchemaDefinition(BaseModel):
    """Defines the schema for a configuration section."""

    fields: dict[str, FieldConstraint]
    allow_extra_fields: bool = False
    required_sections: list[str] = Field(default_factory=list)


class ConfigValidator:
    """Validates strategy configurations against defined schemas."""

    # Pre-defined schemas for different configuration sections
    PARAMETER_SCHEMA = SchemaDefinition(
        fields={
            "min_profit_pct": FieldConstraint(
                field_type="decimal", min_value=Decimal(0), max_value=Decimal(100)
            ),
            "max_position_pct": FieldConstraint(
                field_type="decimal", min_value=Decimal(0), max_value=Decimal(100)
            ),
            "stop_loss_pct": FieldConstraint(
                field_type="decimal", min_value=Decimal(0), max_value=Decimal(100)
            ),
            "take_profit_pct": FieldConstraint(
                field_type="decimal", min_value=Decimal(0), max_value=Decimal(100)
            ),
            "min_order_size": FieldConstraint(
                field_type="decimal", min_value=Decimal(0)
            ),
            "max_order_size": FieldConstraint(
                field_type="decimal", min_value=Decimal(0)
            ),
        }
    )

    RISK_LIMITS_SCHEMA = SchemaDefinition(
        fields={
            "max_positions": FieldConstraint(
                field_type="int", min_value=1, max_value=100
            ),
            "max_daily_loss_pct": FieldConstraint(
                field_type="decimal", min_value=Decimal(0), max_value=Decimal(100)
            ),
            "max_correlation": FieldConstraint(
                field_type="decimal", min_value=Decimal(0), max_value=Decimal(1)
            ),
        }
    )

    EXECUTION_SCHEMA = SchemaDefinition(
        fields={
            "order_type": FieldConstraint(
                field_type="str",
                allowed_values=["market", "limit", "stop", "stop_limit"],
                default_value="market",
            ),
            "time_in_force": FieldConstraint(
                field_type="str",
                allowed_values=["IOC", "FOK", "GTC", "GTX"],
                default_value="IOC",
            ),
            "retry_attempts": FieldConstraint(
                field_type="int", min_value=0, max_value=10, default_value=3
            ),
            "retry_delay_ms": FieldConstraint(
                field_type="int", min_value=0, max_value=60000, default_value=100
            ),
        }
    )

    MONITORING_SCHEMA = SchemaDefinition(
        fields={
            "log_level": FieldConstraint(
                field_type="str",
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                default_value="INFO",
            ),
            "metrics_interval_seconds": FieldConstraint(
                field_type="int", min_value=1, max_value=3600, default_value=60
            ),
            "alert_on_loss": FieldConstraint(field_type="bool", default_value=True),
        }
    )

    STRATEGY_SCHEMA = SchemaDefinition(
        fields={
            "name": FieldConstraint(
                field_type="str",
                required=True,
                min_length=1,
                max_length=100,
                regex_pattern=r"^[A-Za-z][A-Za-z0-9_]*$",
            ),
            "version": FieldConstraint(
                field_type="str", required=True, regex_pattern=r"^\d+\.\d+\.\d+$"
            ),
            "tier": FieldConstraint(
                field_type="str",
                required=True,
                allowed_values=["sniper", "hunter", "strategist"],
            ),
            "enabled": FieldConstraint(field_type="bool", default_value=True),
        }
    )

    def __init__(self):
        """Initialize the configuration validator."""
        self.custom_schemas: dict[str, SchemaDefinition] = {}
        self.tier_constraints: dict[str, dict[str, Any]] = self._load_tier_constraints()

        logger.info("ConfigValidator initialized")

    def _load_tier_constraints(self) -> dict[str, dict[str, Any]]:
        """Load tier-specific constraints.

        Returns:
            Dictionary of tier constraints
        """
        return {
            "sniper": {
                "max_positions": 1,
                "max_order_size": Decimal(100),
                "max_daily_loss_pct": Decimal(5),
            },
            "hunter": {
                "max_positions": 5,
                "max_order_size": Decimal(500),
                "max_daily_loss_pct": Decimal(10),
            },
            "strategist": {
                "max_positions": 20,
                "max_order_size": Decimal(5000),
                "max_daily_loss_pct": Decimal(20),
            },
        }

    def validate_config(
        self,
        config: dict[str, Any],
        tier: str | None = None,
        strict_mode: bool = False,
    ) -> ValidationResult:
        """Validate a complete configuration.

        Args:
            config: Configuration dictionary to validate
            tier: Optional tier for tier-specific validation
            strict_mode: If True, treat warnings as errors

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=True)

        try:
            # Validate required sections
            required_sections = [
                "strategy",
                "parameters",
                "risk_limits",
                "execution",
                "monitoring",
            ]
            for section in required_sections:
                if section not in config:
                    result.errors.append(f"Missing required section: {section}")
                    result.is_valid = False

            if not result.is_valid:
                return result

            # Validate each section
            self._validate_section(
                config.get("strategy", {}), self.STRATEGY_SCHEMA, "strategy", result
            )
            self._validate_section(
                config.get("parameters", {}),
                self.PARAMETER_SCHEMA,
                "parameters",
                result,
            )
            self._validate_section(
                config.get("risk_limits", {}),
                self.RISK_LIMITS_SCHEMA,
                "risk_limits",
                result,
            )
            self._validate_section(
                config.get("execution", {}), self.EXECUTION_SCHEMA, "execution", result
            )
            self._validate_section(
                config.get("monitoring", {}),
                self.MONITORING_SCHEMA,
                "monitoring",
                result,
            )

            # Extract tier from strategy section if not provided
            if not tier and "strategy" in config:
                tier = config["strategy"].get("tier")

            # Validate tier-specific constraints
            if tier:
                self._validate_tier_constraints(config, tier, result)

            # Validate cross-field constraints
            self._validate_cross_field_constraints(config, result)

            # Apply defaults and type conversions
            if result.is_valid:
                validated_config = self._apply_defaults_and_conversions(config)
                result.validated_config = validated_config

            # In strict mode, treat warnings as errors
            if strict_mode and result.warnings:
                result.errors.extend(result.warnings)
                result.warnings = []
                result.is_valid = False

        except Exception as e:
            result.errors.append(f"Validation error: {e!s}")
            result.is_valid = False

        # Log validation result
        if result.is_valid:
            logger.info(
                "Configuration validation passed", warnings_count=len(result.warnings)
            )
        else:
            logger.error(
                "Configuration validation failed",
                errors=result.errors,
                warnings=result.warnings,
            )

        return result

    def _validate_section(
        self,
        section_data: dict[str, Any],
        schema: SchemaDefinition,
        section_name: str,
        result: ValidationResult,
    ) -> None:
        """Validate a configuration section against its schema.

        Args:
            section_data: Section data to validate
            schema: Schema definition for the section
            section_name: Name of the section being validated
            result: ValidationResult to update
        """
        # Check for unknown fields
        if not schema.allow_extra_fields:
            extra_fields = set(section_data.keys()) - set(schema.fields.keys())
            if extra_fields:
                result.warnings.append(
                    f"Unknown fields in {section_name}: {', '.join(extra_fields)}"
                )

        # Validate each field
        for field_name, constraint in schema.fields.items():
            field_value = section_data.get(field_name)

            # Check required fields
            if field_value is None:
                if constraint.required:
                    if constraint.default_value is not None:
                        section_data[field_name] = constraint.default_value
                    else:
                        result.errors.append(
                            f"Required field missing in {section_name}: {field_name}"
                        )
                        result.is_valid = False
                continue

            # Validate field type
            if not self._validate_field_type(field_value, constraint.field_type):
                result.errors.append(
                    f"Invalid type for {section_name}.{field_name}: "
                    f"expected {constraint.field_type}, got {type(field_value).__name__}"
                )
                result.is_valid = False
                continue

            # Validate numeric constraints
            if constraint.field_type in ["int", "float", "decimal"]:
                self._validate_numeric_constraints(
                    field_value, constraint, f"{section_name}.{field_name}", result
                )

            # Validate string constraints
            if constraint.field_type == "str":
                self._validate_string_constraints(
                    field_value, constraint, f"{section_name}.{field_name}", result
                )

            # Validate allowed values
            if (
                constraint.allowed_values
                and field_value not in constraint.allowed_values
            ):
                result.errors.append(
                    f"Invalid value for {section_name}.{field_name}: "
                    f"{field_value} not in {constraint.allowed_values}"
                )
                result.is_valid = False

    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type.

        Args:
            value: Value to validate
            expected_type: Expected type name

        Returns:
            True if type is valid, False otherwise
        """
        type_map = {
            "int": int,
            "float": (int, float),
            "decimal": (int, float, Decimal),
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        if expected_type not in type_map:
            return False

        expected_types = type_map[expected_type]
        return isinstance(value, expected_types)

    def _validate_numeric_constraints(
        self,
        value: int | float | Decimal,
        constraint: FieldConstraint,
        field_path: str,
        result: ValidationResult,
    ) -> None:
        """Validate numeric field constraints.

        Args:
            value: Numeric value to validate
            constraint: Field constraint
            field_path: Full path to the field
            result: ValidationResult to update
        """
        if constraint.min_value is not None and value < constraint.min_value:
            result.errors.append(
                f"Value too small for {field_path}: {value} < {constraint.min_value}"
            )
            result.is_valid = False

        if constraint.max_value is not None and value > constraint.max_value:
            result.errors.append(
                f"Value too large for {field_path}: {value} > {constraint.max_value}"
            )
            result.is_valid = False

    def _validate_string_constraints(
        self,
        value: str,
        constraint: FieldConstraint,
        field_path: str,
        result: ValidationResult,
    ) -> None:
        """Validate string field constraints.

        Args:
            value: String value to validate
            constraint: Field constraint
            field_path: Full path to the field
            result: ValidationResult to update
        """
        if constraint.min_length is not None and len(value) < constraint.min_length:
            result.errors.append(
                f"String too short for {field_path}: length {len(value)} < {constraint.min_length}"
            )
            result.is_valid = False

        if constraint.max_length is not None and len(value) > constraint.max_length:
            result.errors.append(
                f"String too long for {field_path}: length {len(value)} > {constraint.max_length}"
            )
            result.is_valid = False

        if constraint.regex_pattern:
            pattern = re.compile(constraint.regex_pattern)
            if not pattern.match(value):
                result.errors.append(
                    f"String format invalid for {field_path}: '{value}' doesn't match pattern '{constraint.regex_pattern}'"
                )
                result.is_valid = False

    def _validate_tier_constraints(
        self, config: dict[str, Any], tier: str, result: ValidationResult
    ) -> None:
        """Validate tier-specific constraints.

        Args:
            config: Configuration dictionary
            tier: Trading tier
            result: ValidationResult to update
        """
        if tier not in self.tier_constraints:
            result.warnings.append(f"Unknown tier: {tier}")
            return

        constraints = self.tier_constraints[tier]

        # Check max positions
        if "risk_limits" in config:
            max_positions = config["risk_limits"].get("max_positions", 0)
            tier_max = constraints.get("max_positions")
            if tier_max and max_positions > tier_max:
                result.errors.append(
                    f"Max positions ({max_positions}) exceeds tier limit ({tier_max}) for {tier}"
                )
                result.is_valid = False

        # Check max order size
        if "parameters" in config:
            max_order_size = Decimal(str(config["parameters"].get("max_order_size", 0)))
            tier_max = constraints.get("max_order_size")
            if tier_max and max_order_size > tier_max:
                result.errors.append(
                    f"Max order size ({max_order_size}) exceeds tier limit ({tier_max}) for {tier}"
                )
                result.is_valid = False

        # Check max daily loss
        if "risk_limits" in config:
            max_daily_loss = Decimal(
                str(config["risk_limits"].get("max_daily_loss_pct", 0))
            )
            tier_max = constraints.get("max_daily_loss_pct")
            if tier_max and max_daily_loss > tier_max:
                result.errors.append(
                    f"Max daily loss ({max_daily_loss}%) exceeds tier limit ({tier_max}%) for {tier}"
                )
                result.is_valid = False

    def _validate_cross_field_constraints(
        self, config: dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate constraints that involve multiple fields.

        Args:
            config: Configuration dictionary
            result: ValidationResult to update
        """
        # Validate min/max order sizes
        if "parameters" in config:
            params = config["parameters"]
            min_order = params.get("min_order_size")
            max_order = params.get("max_order_size")

            if min_order and max_order and min_order > max_order:
                result.errors.append(
                    f"Min order size ({min_order}) > max order size ({max_order})"
                )
                result.is_valid = False

        # Validate stop loss vs take profit
        if "parameters" in config:
            params = config["parameters"]
            stop_loss = params.get("stop_loss_pct")
            take_profit = params.get("take_profit_pct")

            if stop_loss and take_profit and stop_loss <= take_profit:
                result.warnings.append(
                    f"Stop loss ({stop_loss}%) <= take profit ({take_profit}%), "
                    "this might be intentional for certain strategies"
                )

    def _apply_defaults_and_conversions(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply default values and type conversions.

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with defaults and conversions applied
        """
        validated_config = config.copy()

        # Convert numeric strings to Decimal for parameters
        if "parameters" in validated_config:
            for field in [
                "min_profit_pct",
                "max_position_pct",
                "stop_loss_pct",
                "take_profit_pct",
                "min_order_size",
                "max_order_size",
            ]:
                if field in validated_config["parameters"]:
                    value = validated_config["parameters"][field]
                    if not isinstance(value, Decimal):
                        validated_config["parameters"][field] = Decimal(str(value))

        # Convert numeric strings to Decimal for risk limits
        if "risk_limits" in validated_config:
            for field in ["max_daily_loss_pct", "max_correlation"]:
                if field in validated_config["risk_limits"]:
                    value = validated_config["risk_limits"][field]
                    if not isinstance(value, Decimal):
                        validated_config["risk_limits"][field] = Decimal(str(value))

        return validated_config

    def add_custom_schema(self, name: str, schema: SchemaDefinition) -> None:
        """Add a custom schema for validation.

        Args:
            name: Name of the custom schema
            schema: Schema definition
        """
        self.custom_schemas[name] = schema
        logger.info(f"Added custom schema: {name}")

    def validate_partial_config(
        self, partial_config: dict[str, Any], section_name: str
    ) -> ValidationResult:
        """Validate a partial configuration (single section).

        Args:
            partial_config: Partial configuration to validate
            section_name: Name of the section being validated

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True)

        # Get schema for section
        schema_map = {
            "strategy": self.STRATEGY_SCHEMA,
            "parameters": self.PARAMETER_SCHEMA,
            "risk_limits": self.RISK_LIMITS_SCHEMA,
            "execution": self.EXECUTION_SCHEMA,
            "monitoring": self.MONITORING_SCHEMA,
        }

        schema = schema_map.get(section_name)
        if not schema:
            # Check custom schemas
            schema = self.custom_schemas.get(section_name)
            if not schema:
                result.errors.append(f"Unknown section: {section_name}")
                result.is_valid = False
                return result

        # Validate the section
        self._validate_section(partial_config, schema, section_name, result)

        if result.is_valid:
            result.validated_config = self._apply_defaults_and_conversions(
                {section_name: partial_config}
            )[section_name]

        return result

    def get_schema_info(self, section_name: str | None = None) -> dict[str, Any]:
        """Get information about available schemas.

        Args:
            section_name: Optional section to get info for

        Returns:
            Dictionary with schema information
        """
        if section_name:
            schema_map = {
                "strategy": self.STRATEGY_SCHEMA,
                "parameters": self.PARAMETER_SCHEMA,
                "risk_limits": self.RISK_LIMITS_SCHEMA,
                "execution": self.EXECUTION_SCHEMA,
                "monitoring": self.MONITORING_SCHEMA,
            }

            schema = schema_map.get(section_name) or self.custom_schemas.get(
                section_name
            )
            if not schema:
                return {"error": f"Unknown section: {section_name}"}

            return {
                "section": section_name,
                "fields": {
                    name: {
                        "type": constraint.field_type,
                        "required": constraint.required,
                        "constraints": {
                            k: v
                            for k, v in constraint.dict().items()
                            if v is not None and k not in ["field_type", "required"]
                        },
                    }
                    for name, constraint in schema.fields.items()
                },
                "allow_extra_fields": schema.allow_extra_fields,
            }

        # Return all schemas
        all_schemas = [
            "strategy",
            "parameters",
            "risk_limits",
            "execution",
            "monitoring",
        ]
        all_schemas.extend(self.custom_schemas.keys())

        return {
            "available_schemas": all_schemas,
            "tier_constraints": self.tier_constraints,
        }
