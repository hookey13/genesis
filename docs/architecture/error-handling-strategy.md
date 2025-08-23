# Error Handling Strategy

## General Approach

- **Error Model:** Hierarchical exceptions with specific handling per level
- **Exception Hierarchy:** BaseError → DomainError → TradingError → OrderError
- **Error Propagation:** Errors bubble up with context, handled at appropriate level

## Logging Standards

- **Library:** structlog 24.1.0
- **Format:** JSON with mandatory fields: timestamp, level, component, error_type, context
- **Levels:** DEBUG (dev only), INFO (normal operation), WARNING (degraded), ERROR (failure), CRITICAL (money at risk)
- **Required Context:**
  - Correlation ID: UUID per operation chain
  - Service Context: Component, function, line number
  - User Context: No PII, only account_id and tier

## Error Handling Patterns

### External API Errors
- **Retry Policy:** Exponential backoff: 1s, 2s, 4s, 8s, max 30s
- **Circuit Breaker:** Open after 5 failures in 30 seconds, half-open after 60s
- **Timeout Configuration:** Connect: 5s, Read: 10s, Total: 30s
- **Error Translation:** Map Binance errors to domain exceptions

### Business Logic Errors
- **Custom Exceptions:** TierViolation, RiskLimitExceeded, TiltInterventionRequired
- **User-Facing Errors:** Translated to calm, helpful messages
- **Error Codes:** GENESIS-XXXX format for tracking

### Data Consistency
- **Transaction Strategy:** All-or-nothing database operations
- **Compensation Logic:** Rollback procedures for partial failures
- **Idempotency:** Client order IDs prevent duplicate processing
