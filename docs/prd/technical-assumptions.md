# Technical Assumptions

## Repository Structure: Monorepo

Single repository containing all system components with clear modular separation:
- `/engine` - Core trading execution and order management  
- `/strategies` - Pluggable strategy modules (arbitrage, liquidity provision)
- `/risk` - Risk management and position control
- `/tilt` - Psychological monitoring and intervention systems
- `/analytics` - Performance tracking and reporting
- `/infrastructure` - Deployment, monitoring, logging utilities

## Service Architecture

**Phase 1 (MVP, $500-$2k): Monolith**
- Single Python 3.11+ process with asyncio for concurrent operations
- All components in-memory for sub-10ms internal communication  
- Supervisor process for automatic restart on failure
- SQLite for initial trade logging (migrate at $2k)

**Phase 2 ($2k-$10k): Modular Monolith**  
- Core process with hot-swappable strategy modules
- PostgreSQL for trade history and analytics
- Redis for real-time state and order queuing
- Separate monitoring process for tilt detection

**Phase 3 ($10k+): Service-Oriented Architecture**
- Execution service (Rust) for <10ms critical path
- Risk service (Python) for portfolio management
- Analytics service (Python) for reporting
- Message queue (Redis Streams) for service communication

## Testing Requirements

**Critical Testing Pyramid:**
- **Unit Tests (70%):** Every calculation, especially position sizing and risk limits. Zero tolerance for math errors
- **Integration Tests (20%):** API interaction, order flow, database writes. Mock Binance responses for edge cases
- **End-to-End Tests (10%):** Full trading scenarios including tilt detection and tier transitions
- **Chaos Testing:** Weekly "kill -9" random component during paper trading to verify recovery
- **Psychological Testing:** Simulated drawdown scenarios to verify intervention triggers

**Performance Benchmarks:**
- Order decision to API call: <10ms required
- Market data ingestion to signal: <5ms required  
- Database write: <10ms required (async, non-blocking)
- Tilt detection calculation: <50ms acceptable

## Additional Technical Assumptions and Requests

**Language & Framework Stack:**
- **Python 3.11+:** Primary language for all business logic (asyncio for concurrency)
- **Rust:** Planned for execution engine at $10k+ (not MVP)
- **PostgreSQL 15+:** Time-series optimized for trade data
- **Redis 7+:** Pub/sub for market data, sorted sets for order books
- **Rich/Textual:** Terminal UI framework for "Digital Zen Garden" interface
- **ccxt:** Exchange abstraction library (Binance initially, multi-exchange ready)
- **pandas/numpy:** Statistical calculations for arbitrage detection
- **Prometheus + Grafana:** Metrics and monitoring
- **Docker:** Containerization for consistent deployment

**Infrastructure & Deployment:**
- **Primary VPS:** DigitalOcean Singapore (closest to Binance servers) - $20/month initially
- **Backup VPS:** Vultr Tokyo (failover ready) - activated at $5k capital
- **GitHub Actions:** CI/CD pipeline with mandatory test passage
- **Terraform:** Infrastructure as code for reproducible deployment
- **tmux:** Persistent terminal sessions for 24/7 monitoring
- **Tailscale:** Secure VPN for remote access without exposed ports

**Network & Latency Optimization:**
- **WebSocket streams:** For real-time price feeds (never polling)
- **Connection pooling:** Reuse HTTPS connections to minimize handshake overhead
- **Binary protocols:** MessagePack for internal communication at $10k+
- **Route optimization:** Direct peering to Binance IP ranges where possible
- **Rate limit management:** 80% utilization maximum with exponential backoff

**Data Architecture:**
- **Hot data (Redis):** Last 1000 trades, current positions, order book snapshots
- **Warm data (PostgreSQL):** 30-day trade history, hourly aggregates
- **Cold data (S3-compatible):** Historical data for backtesting and analysis
- **Data retention:** Full tick data for 90 days, aggregates forever
- **Backup strategy:** Hourly snapshots, daily offsite backup, weekly full backup

**Security Architecture:**
- **API Key Management:** HashiCorp Vault for production (env vars for development)
- **Key Rotation:** Monthly automated rotation with zero-downtime deployment
- **Network Security:** Wireguard VPN only access, no public endpoints
- **Audit Logging:** Every trade decision logged with full context
- **Encryption:** TLS 1.3 for all external communication, AES-256 for data at rest

**Development Workflow:**
- **Git branching:** GitFlow with protection on main branch
- **Code review:** All changes require review (even as solo developer - next day review)
- **Documentation:** Docstrings mandatory, architecture decisions in ADRs
- **Versioning:** Semantic versioning with detailed changelogs
- **Rollback capability:** One-command rollback to any previous version

## Monitoring & Alerting Thresholds

**System Health Metrics (Technical)**

**Critical - Immediate PagerDuty (Phone Call + SMS):**
- API connectivity lost >30 seconds
- Order execution latency >500ms (5 consecutive orders)
- Database write failures >2 in 60 seconds
- Memory usage >90% of available
- Disk space <500MB remaining
- Position calculation mismatch detected (any deviation)
- API rate limit usage >95%
- WebSocket disconnection >10 seconds
- Supervisor process restart >3 times in 5 minutes

**Warning - Slack/Email Alert:**
- API latency >200ms (rolling 1-minute average)
- Memory usage >70%
- Disk space <2GB
- Database query time >100ms
- Redis latency >10ms
- Network packet loss >0.1%
- CPU usage >80% sustained for 5 minutes
- Log write failures (any)
- Backup failure (any)

**Trading Performance Metrics (Financial)**

**Tier-Specific Drawdown Alerts:**

*Sniper Tier ($500-$2k):*
- Info: -$10 in single trade (2% of minimum)
- Warning: -$25 daily loss (5% of minimum)
- Critical: -$50 total drawdown (10% of minimum)
- HALT: -$75 drawdown (15% - automatic trading stop)

*Hunter Tier ($2k-$10k):*
- Info: -$50 in single trade (2.5% of minimum)
- Warning: -$100 daily loss (5% of minimum)  
- Critical: -$300 total drawdown (15% of minimum)
- HALT: -$400 drawdown (20% - requires manual reset)

*Strategist Tier ($10k+):*
- Info: -$200 in single trade (2% of minimum)
- Warning: -$500 daily loss (5% of minimum)
- Critical: -$1,500 total drawdown (15% of minimum)
- HALT: -$2,000 drawdown (20% - lockout protocol)
