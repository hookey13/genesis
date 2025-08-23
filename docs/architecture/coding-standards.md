# Coding Standards

These standards are MANDATORY for all development on Project GENESIS. They are extracted to `docs/architecture/coding-standards.md` and loaded by AI agents.

## Core Standards

- **Languages & Runtimes:** Python 3.11.8 exclusively - no Python 3.12 features, no backports
- **Style & Linting:** Black formatter (line length 88), Ruff linter with config, Type hints mandatory for all functions
- **Test Organization:** `tests/unit/test_{module}.py`, `tests/integration/test_{workflow}.py`

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase | `OrderExecutor`, `TiltDetector` |
| Functions | snake_case | `execute_market_order`, `calculate_position_size` |
| Constants | UPPER_SNAKE | `MAX_POSITION_SIZE`, `TIER_LIMITS` |
| Private methods | Leading underscore | `_validate_order`, `_check_risk` |
| Tier-locked functions | Suffix with tier | `execute_vwap_strategist`, `multi_pair_hunter` |
| Money variables | Suffix with currency | `balance_usdt`, `pnl_dollars` |
| Time variables | Suffix with unit | `timeout_seconds`, `delay_ms` |

## Critical Rules

- **NEVER use float for money:** Always use Decimal from decimal module
- **NEVER use print() or console.log:** Always use structlog
- **NEVER catch bare exceptions:** Always catch specific exceptions
- **NEVER hard-code tier limits:** All limits must come from configuration
- **ALWAYS use idempotency keys:** Every order must have a client_order_id
- **ALWAYS validate state after restart:** Never assume in-memory state matches reality
- **ALWAYS use database transactions for multi-step operations:** Partial writes cause inconsistent state
- **NEVER modify tier without state machine:** Tier changes must go through the state machine

## Python-Specific Guidelines

- **Use asyncio for all I/O:** Blocking I/O freezes the event loop
- **Type hints are mandatory:** Catches errors at development time
- **Use dataclasses for domain models:** Provides validation, immutability
- **Context managers for resource cleanup:** Ensures connections are closed
