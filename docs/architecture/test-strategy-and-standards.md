# Test Strategy and Standards

## Testing Philosophy

- **Approach:** Test-After-Development with critical-path Test-First
- **Coverage Goals:** 100% for money paths, 90% for risk/tilt, 70% for UI/analytics
- **Test Pyramid:** 70% unit (fast/isolated), 20% integration (workflows), 10% end-to-end (full system)

## Test Types and Organization

### Unit Tests
- **Framework:** pytest 8.0.0 with pytest-asyncio
- **File Convention:** `test_{module}.py` in same structure as source
- **Location:** `tests/unit/` mirrors `genesis/` structure
- **Mocking Library:** pytest-mock with faker for test data
- **Coverage Requirement:** 100% for risk_engine, executor, math utils

### Integration Tests
- **Scope:** Component interactions, database operations, API calls
- **Location:** `tests/integration/`
- **Test Infrastructure:**
  - **Database:** In-memory SQLite for speed
  - **Message Queue:** Python queues instead of Redis
  - **External APIs:** VCR.py for recorded responses

### End-to-End Tests
- **Framework:** pytest with asyncio
- **Scope:** Full system with mocked exchange
- **Environment:** Docker compose with all services
- **Test Data:** Fixtures with known market conditions

## Test Data Management

- **Strategy:** Builder pattern for test objects
- **Fixtures:** `tests/fixtures/` with market data snapshots
- **Factories:** Object mothers for common test scenarios
- **Cleanup:** Automatic cleanup after each test

## Continuous Testing

- **CI Integration:** GitHub Actions runs all tests on push
- **Performance Tests:** Benchmark critical paths
- **Security Tests:** Check for common vulnerabilities
