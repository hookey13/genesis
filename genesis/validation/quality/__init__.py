"""Technical quality validators for comprehensive code and performance validation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_analyzer import CodeQualityAnalyzer
    from .database_validator import DatabaseValidator
    from .performance_validator import PerformanceValidator
    from .resource_validator import ResourceValidator
    from .test_validator import TestCoverageValidator

__all__ = [
    "TestCoverageValidator",
    "CodeQualityAnalyzer",
    "PerformanceValidator",
    "ResourceValidator",
    "DatabaseValidator",
]


def __getattr__(name: str):
    """Lazy import validators to avoid circular dependencies."""
    if name == "TestCoverageValidator":
        from .test_validator import TestCoverageValidator
        return TestCoverageValidator
    elif name == "CodeQualityAnalyzer":
        from .code_analyzer import CodeQualityAnalyzer
        return CodeQualityAnalyzer
    elif name == "PerformanceValidator":
        from .performance_validator import PerformanceValidator
        return PerformanceValidator
    elif name == "ResourceValidator":
        from .resource_validator import ResourceValidator
        return ResourceValidator
    elif name == "DatabaseValidator":
        from .database_validator import DatabaseValidator
        return DatabaseValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")