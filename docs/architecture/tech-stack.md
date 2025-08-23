# Tech Stack

This is the DEFINITIVE technology selection for Project GENESIS. Every choice here directly impacts our ability to survive the valley of death and reach $100k. These decisions are final and all other documentation must reference these exact versions.

## Cloud Infrastructure

- **Provider:** DigitalOcean
- **Key Services:** Droplets (Ubuntu 22.04 LTS), Spaces (S3-compatible backup)
- **Deployment Regions:** Singapore (SGP1 - primary, closest to Binance)

## Technology Stack Table - PHASED APPROACH

| Category | Technology | Version | Phase | Purpose | Rationale |
|----------|------------|---------|-------|---------|-----------|
| **Language** | Python | 3.11.8 | MVP | Primary development language | Asyncio maturity, sufficient speed with proper architecture |
| **Runtime** | CPython | 3.11.8 | MVP | Python interpreter | Standard, stable, well-tested with asyncio |
| **Async Framework** | asyncio | stdlib | MVP | Concurrent I/O operations | Native Python, no extra dependencies |
| **Exchange Library** | ccxt | 4.2.25 | MVP | Binance API abstraction | Battle-tested, handles rate limits, automatic retry logic |
| **Terminal UI** | Rich | 13.7.0 | MVP | Terminal interface rendering | Beautiful TUI for "Digital Zen Garden" aesthetic |
| **Terminal Framework** | Textual | 0.47.1 | MVP | Interactive TUI framework | Event-driven UI, reactive updates without flicker |
| **Database** | SQLite | 3.45.0 | MVP | Trade logging, state persistence | Zero-config, atomic transactions, single file backup |
| **Database** | PostgreSQL | 16.1 | $2k+ | Scalable data storage | Time-series optimization, concurrent access |
| **Cache/Queue** | Python dicts + asyncio.Queue | stdlib | MVP | In-memory state and queuing | Zero additional complexity, sufficient for single-pair trading |
| **Cache/Queue** | Redis | 7.2.4 | $2k+ | Distributed state, order queuing | <1ms latency when multi-pair trading requires it |
| **Decimal Math** | decimal | stdlib | MVP | Financial calculations | Prevents float rounding errors in position sizing |
| **Data Analysis** | pandas | 2.2.0 | MVP | Statistical calculations | Required for spread analysis and arbitrage detection |
| **Numerical Computing** | numpy | 1.26.3 | MVP | Array operations | Correlation matrices, fast calculations |
| **HTTP Client** | aiohttp | 3.9.3 | MVP | Async HTTP for REST API | Connection pooling, timeout handling |
| **WebSocket Client** | websockets | 12.0 | MVP | Market data streams | Pure Python, asyncio native |
| **Configuration** | pydantic | 2.5.3 | MVP | Config validation | Runtime validation, environment variable parsing |
| **Logging** | structlog | 24.1.0 | MVP | Structured logging | JSON output for forensic analysis |
| **Schema Migrations** | Alembic | 1.13.1 | MVP | Database migrations | Critical for SQLite→PostgreSQL transition |
| **Backup Tool** | restic | 0.16.3 | MVP | Encrypted backups | Automated backups to DigitalOcean Spaces |
| **Testing** | pytest | 8.0.0 | MVP | Test framework | Async test support, fixtures |
| **Test Coverage** | pytest-cov | 4.1.0 | MVP | Coverage reporting | Ensures critical paths tested |
| **Code Formatting** | black | 24.1.1 | MVP | Code style enforcement | Consistent formatting |
| **Linting** | ruff | 0.1.14 | MVP | Fast Python linter | 10-100x faster than pylint |
| **Type Checking** | mypy | 1.8.0 | MVP | Static type checking | Catch errors before runtime |
| **Process Manager** | supervisor | 4.2.5 | MVP | Process monitoring/restart | Automatic recovery from crashes |
| **Monitoring** | Log files + alerts | N/A | MVP | Basic monitoring | Parse logs for critical errors, email alerts |
| **Monitoring** | Prometheus | 2.48.1 | $5k+ | Metrics collection | When capital justifies infrastructure |
| **Visualization** | Grafana | 10.3.1 | $5k+ | Metrics dashboards | When worth the complexity |
| **Message Format** | JSON | stdlib | MVP | Serialization | Simple, debuggable, sufficient |
| **Message Format** | MessagePack | 1.0.7 | $10k+ | Binary serialization | When microseconds matter |
| **Container** | Docker | 25.0.0 | MVP | Deployment consistency | Identical dev/prod environments |
| **IaC** | Bash scripts | N/A | MVP | Simple deployment | Reduce complexity at start |
| **IaC** | Terraform | 1.7.0 | $2k+ | Infrastructure automation | When managing multiple servers |
| **VPN** | SSH + tmux | N/A | MVP | Remote access | Simple, secure, no additional services |
| **VPN** | Tailscale | 1.56.1 | $5k+ | Team access | When backup operator needed |
| **Version Control** | Git | 2.43.0 | MVP | Source control | Critical from day one |
| **CI/CD** | GitHub Actions | N/A | MVP | Automated testing | Free tier, test before deploy |

## Phased Technology Evolution

**Phase 1: MVP ($500-$2k)**
- Pure Python with asyncio - no external state management
- SQLite with Alembic for future migration path
- In-memory everything (dicts, queues, calculations)
- Structured logging to files with log rotation
- Automated backups every 4 hours with restic

**Phase 2: Hunter ($2k-$10k)**
- Add Redis for multi-pair state management
- Migrate to PostgreSQL (via Alembic migrations)
- Keep JSON serialization (it's working)
- Basic monitoring with log aggregation

**Phase 3: Strategist ($10k+)**
- Consider Rust for execution engine
- Add Prometheus/Grafana for observability
- MessagePack if event volume demands it
- Full infrastructure automation with Terraform
