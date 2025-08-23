# Security

## Input Validation

- **Validation Library:** Pydantic 2.5.3 with custom validators
- **Validation Location:** At API boundary AND before execution (defense in depth)
- **Required Rules:**
  - All external inputs MUST be validated
  - Validation at API boundary before processing
  - Whitelist approach preferred over blacklist

## Authentication & Authorization

- **Auth Method:** API key with HMAC signatures (no JWT - stateless is dangerous for trading)
- **Session Management:** Server-side sessions with Redis, 4-hour timeout
- **Required Patterns:**
  - API keys never in code, only environment variables
  - Separate keys for read vs trade permissions
  - IP whitelist enforcement at application level

## Secrets Management

- **Development:** `.env` file with strict `.gitignore` (never committed)
- **Production:** Environment variables injected at runtime
- **Code Requirements:**
  - NEVER hardcode secrets
  - Access via configuration service only
  - No secrets in logs or error messages

## API Security

- **Rate Limiting:** 10 requests per second per endpoint (prevent self-DoS)
- **CORS Policy:** Not applicable (no browser access)
- **Security Headers:** X-Request-ID for tracing, X-API-Version for compatibility
- **HTTPS Enforcement:** SSH tunnel only, no public HTTPS endpoint

## Data Protection

- **Encryption at Rest:** SQLite database encrypted with SQLCipher
- **Encryption in Transit:** TLS 1.3 for all external communication
- **PII Handling:** No personal data stored (account_id only)
- **Logging Restrictions:** No keys, passwords, or full order details in logs

## Dependency Security

- **Scanning Tool:** pip-audit for Python dependencies
- **Update Policy:** Security updates within 24 hours, others monthly
- **Approval Process:** New dependencies require security review

## Security Testing

- **SAST Tool:** Bandit for Python static analysis
- **DAST Tool:** Not applicable (no web interface)
- **Penetration Testing:** Self-audit quarterly, focus on API keys and order manipulation
