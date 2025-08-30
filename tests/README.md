# Test Structure

This directory contains all tests for the Genesis trading system.

## Organization

- **unit/** - Unit tests for individual components
- **integration/** - Integration tests for component interactions
- **performance/** - Performance and benchmark tests
- **stress/** - Stress tests for system limits
- **chaos/** - Chaos engineering tests
- **dr/** - Disaster recovery tests
- **mocks/** - Mock objects and test helpers
- **fixtures/** - Test data fixtures
- **load/** - Load testing scenarios

## Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit

# Run integration tests
pytest tests/integration

# Run with coverage
pytest --cov=genesis --cov-report=html

# Run specific test file
pytest tests/unit/test_risk_engine.py

# Run tests in parallel
pytest -n auto

# Run with timeout enforcement
pytest --timeout=300

# Run specific marker
pytest -m "not slow"
```

## Test Requirements

- Python 3.11.8+
- pytest 7.0+
- pytest-asyncio
- pytest-cov
- pytest-mock
- pytest-benchmark
- pytest-timeout
- hypothesis

## Test Markers

Tests can be marked with the following decorators:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.requires_api` - Tests requiring API access
- `@pytest.mark.timeout(30)` - Test with specific timeout in seconds

## Writing Tests

1. Follow naming convention: `test_*.py`
2. Use fixtures from `conftest.py`
3. Mock external dependencies
4. Test both success and failure cases
5. Maintain 100% coverage for money paths
6. Add timeout markers for long-running tests
7. Handle file encoding properly (UTF-8 with errors='ignore')

## Known Issues

- Some files in the codebase contain Unicode characters that may cause encoding issues
- Tests should use `encoding='utf-8', errors='ignore'` when reading files
- Mock file system paths when testing file operations to avoid real file system dependencies