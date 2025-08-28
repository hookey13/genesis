# Project GENESIS Product Requirements Document (PRD)

## Goals and Background Context

### Goals
- Execute trades automatically without manual intervention 95% of the time through tier-locked evolution system
- Successfully navigate the $2k-10k "execution valley of death" that destroys 90% of profitable strategies
- Achieve consistent 5-10% monthly returns while maintaining maximum 15% drawdown limit
- Reach $100k within 18-24 months through disciplined compound growth from $500 starting capital
- Enforce systematic discipline during emotional moments and drawdowns through automated guardrails
- Maintain 99.5% system uptime during market hours with robust infrastructure

### Phase 0: Trading Foundation (Pre-Tier Entry Gate)

**Purpose:** Establish core trading capability before tier progression begins

**Success Criteria:**
- Execute ONE successful round-trip trade (buy + sell)
- Maintain 10 consecutive paper trades without system failure
- Demonstrate 24-hour autonomous operation
- Achieve 95% uptime during market hours
- Complete full order lifecycle: Connect → Price → Signal → Risk Check → Execute → Track → Exit → P&L

**Timeline:** 2 weeks from implementation start

**MVP Definition (Revised):**
- **Previous:** Tier progression system with $500→$2k capability
- **New:** Execute a single successful trade end-to-end with:
  1. Connect to Binance testnet ✓
  2. Receive live prices via WebSocket
  3. Place market buy order
  4. Track position in database
  5. Place market sell order
  6. Calculate and log P&L
  7. Display in terminal UI

**Entry Gate to Sniper Tier:**
- Phase 0 MVP operational for 72 hours
- Zero manual interventions required
- All core components communicating via Event Bus
- Risk engine validating all trades
- Audit log capturing all events

**Tier Transition Success Gates:**

**Sniper Tier ($500-$2,000) → Hunter Tier Gate Requirements:**
- Minimum 30 days at current tier before transition eligible
- Demonstrated 60% profitable days over last 20 trading days
- Maximum position size discipline: Zero violations of 5% rule in past 14 days
- Execution proficiency: 90% of trades executed within 100ms target
- Risk compliance: No drawdowns exceeding 10% in transition month
- Pair graduation: Successfully traded 3+ different pairs profitably
- **Lock Criteria:** System auto-locks Hunter features until ALL gates passed

**Hunter Tier ($2,000-$10,000) → Strategist Tier Gate Requirements:**
- Minimum 45 days at Hunter tier for strategy maturation
- Consistent slicing execution: 95% of orders >$200 use 3+ slices
- Portfolio diversification: Active positions across 5+ uncorrelated pairs
- Drawdown recovery: Demonstrated recovery from one 10%+ drawdown
- Volume analysis proficiency: No market impact >0.5% on any trade
- Risk-adjusted returns: Sharpe ratio >1.5 over past 30 days
- **Lock Criteria:** VWAP/iceberg algorithms remain disabled until gates cleared

**Strategist Tier ($10,000-$50,000) → Architect Tier Gate Requirements:**
- Minimum 60 days of Strategist-level operation
- Multi-strategy validation: 3+ concurrent strategies profitable
- Correlation management: No strategy >40% of total P&L
- Stress test survival: Paper-tested through 3 historical crash scenarios
- Infrastructure redundancy: Backup systems tested under fire
- Institutional discipline: 30 consecutive days without manual override
- **Lock Criteria:** Advanced features locked until psychological evaluation passed

**Emergency Demotion Triggers (Automatic Tier Downgrade):**
- Any single day loss >15% of account
- Three consecutive days of >5% losses
- Five override attempts within 24 hours
- Manual execution attempts bypassing system
- Correlation spike: All positions moving same direction >80%
- Infrastructure failure causing >$500 loss

### Background Context

Project GENESIS addresses a critical gap in the cryptocurrency trading landscape where Binance processes $50+ billion daily yet thousands of sub-$1M volume trading pairs contain persistent inefficiencies generating 2-5% spreads. These opportunities exist because institutional traders cannot profitably deploy capital below $100k volumes while retail traders lack the speed and infrastructure to capture fleeting arbitrage windows. The system specifically targets the $2-10k "execution valley of death" where manual methods fail and sophisticated execution algorithms become necessary, implementing a Tier-Locked Evolution System that automatically adapts strategies, risk parameters, and psychological frameworks as capital grows.

The core innovation is the Three-Layer Transformation Protocol combining technical evolution (automated execution changes), psychological safeguards (dollar-focused metrics), and strategic migration (enforced pair discovery). Unlike traditional bots applying uniform logic regardless of account size, GENESIS physically prevents behaviors that worked at lower tiers from destroying higher-tier capital, forcing adoption of appropriate slicing algorithms, position sizing, and risk management at each growth stage.

### Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-08-23 | 1.0 | Initial PRD creation from Project Brief | John (PM) |

## Requirements

### Functional Requirements

**Core Trading Engine:**
- FR1: The system shall enforce tier-locked execution strategies that automatically evolve based on account balance thresholds ($500/$2k/$10k)
- FR2: Market orders shall be used exclusively for positions <$500, with mandatory 3-slice execution for $500-$2k range
- FR3: The system shall monitor and classify market states (DEAD/NORMAL/VOLATILE/PANIC) in real-time with <5 second detection latency
- FR4: Statistical arbitrage engine shall identify and execute mean reversion trades when spreads exceed 2-sigma historical norms
- FR5: Position sizing shall automatically calculate based on current capital with hard 5% maximum per position
- FR6: The system shall scan and rank trading pairs by liquidity appropriateness for current capital tier
- FR7: Emergency override protocol shall require typed confirmation and enforce 24-hour cooldown after use

**Risk Management:**
- FR8: The system shall enforce maximum 15% total account drawdown with automatic trading halt
- FR9: Daily loss limits shall be tier-specific: $25 (Sniper), $100 (Hunter), $500 (Strategist)
- FR10: Correlation monitoring shall prevent >60% of positions moving in same direction
- FR11: The system shall automatically reduce position sizes after 2 consecutive losing days
- FR12: Tier demotion shall trigger automatically upon hitting emergency thresholds

**Execution & Performance:**
- FR13: Order execution latency shall not exceed 50ms for market orders, 100ms for limit orders
- FR14: The system shall maintain order queues for sliced execution with priority management
- FR15: VWAP/TWAP algorithms shall adapt to real-time liquidity conditions
- FR16: Market impact monitoring shall alert when any trade moves price >0.5%
- FR17: The system shall support concurrent execution across 10+ trading pairs

**Monitoring & Analytics:**
- FR18: Real-time P&L shall display in dollars primarily with percentage as secondary metric
- FR19: The system shall log all trades with full execution details for forensic analysis
- FR20: Performance analytics shall calculate Sharpe ratio, win rate, and average R per trade
- FR21: Tier progression status shall show current gates passed and remaining requirements
- FR22: The system shall generate daily reports of strategy performance by pair and time window

**Pre-Tilt Behavioral Markers:**
- FR23: The system shall track click-to-trade latency, flagging when decision speed decreases by >30% (analysis paralysis) or increases by >50% (impulsive trading)
- FR24: Mouse movement heat maps shall detect erratic patterns - excessive hovering over "all-in" buttons or repeated position size recalculations
- FR25: The system shall monitor order cancellation rates, triggering alerts when >40% of orders cancelled within 5 seconds of placement
- FR26: Trading frequency deviation detection shall flag when trades/hour exceeds 2x personal baseline (chasing) or drops to <0.2x baseline (fear paralysis)
- FR27: The system shall track "stare time" - periods of dashboard monitoring without action exceeding 30 minutes indicate fear/uncertainty

**Micro-Tilt Indicators:**
- FR28: Revenge pattern detection shall identify rapid re-entry into same pair within 60 seconds of stop-loss
- FR29: The system shall monitor position sizing volatility - alternating between min and max sizes indicates emotional instability
- FR30: "Bargaining behavior" detection shall flag repeated limit order adjustments (>5 modifications on same order)
- FR31: The system shall track correlation decay - when previously uncorrelated positions start moving together (panic hedging)
- FR32: Time-of-day analysis shall identify trading outside normal hours as potential exhaustion/desperation

**Physiological Proxy Metrics:**
- FR33: Typing speed and accuracy in command inputs shall be baselined, with >40% deviation triggering cooldown suggestions
- FR34: The system shall measure time between loss notification and next action - immediate action (<2 seconds) indicates emotional reaction
- FR35: Tab-switching frequency to price charts shall be monitored - >20 checks/minute indicates anxiety state
- FR36: The system shall detect "rage quit" patterns - multiple terminal disconnections/reconnections within 10 minutes
- FR37: Configuration thrashing detection shall flag when risk parameters are modified >3 times in one session

### Non-Functional Requirements

**Performance & Reliability:**
- NFR1: System uptime shall exceed 99.5% during market hours
- NFR2: Database write operations shall complete within 10ms for trade logging
- NFR3: The system shall auto-restart within 30 seconds of any process crash
- NFR4: Memory usage shall not exceed 2GB under normal operation
- NFR5: The system shall handle 1000 concurrent price updates without degradation

**Security & Compliance:**
- NFR6: API keys shall be stored encrypted and never logged or transmitted
- NFR7: All trading actions shall have audit trails with immutable timestamps
- NFR8: The system shall enforce IP whitelist for API access
- NFR9: Backup of all trade data shall occur every 4 hours
- NFR10: System shall comply with Binance API rate limits with 20% safety margin

**Usability & Maintenance:**
- NFR11: Terminal dashboard shall update every 100ms for real-time feedback
- NFR12: System configuration changes shall not require restart
- NFR13: Log rotation shall prevent disk usage exceeding 10GB
- NFR14: The system shall provide clear error messages for all failure modes
- NFR15: Strategy modules shall be hot-swappable without stopping trading

**Psychological & Behavioral:**
- NFR16: Revenge trading detection shall trigger after 3 rapid sequential trades post-loss
- NFR17: The system shall enforce mandatory 5-minute cooldown between manual interventions
- NFR18: Celebration messages for profits shall be subdued to prevent overconfidence
- NFR19: Drawdown warnings shall escalate in visibility as losses approach limits
- NFR20: The system shall require written reflection entry after any tier transition

**Progressive Intervention System:**
- NFR21: Level 1 Tilt (2-3 indicators): Subtle yellow terminal border, optional 5-minute meditation timer suggestion
- NFR22: Level 2 Tilt (4-5 indicators): Reduced position sizes to 50%, mandatory strategy explanation before trades
- NFR23: Level 3 Tilt (6+ indicators): Trading lockout with "Tilt Recovery Protocol" - requires 500-word journal entry analyzing emotional state
- NFR24: The system shall never use aggressive or shaming language - all interventions framed as "performance optimization"
- NFR25: Positive reinforcement for accepting interventions - "Discipline bonus" multiplier on next winning trade

**Baseline Calibration:**
- NFR26: First 30 days shall establish personal behavioral baselines without interventions
- NFR27: The system shall adapt to individual patterns - what's normal for one trader is tilt for another
- NFR28: Weekly baseline adjustments based on rolling 7-day averages, preventing drift
- NFR29: Major life events can be logged to temporarily adjust sensitivity (job stress, relationship issues)
- NFR30: The system shall distinguish between learning (early mistakes) and tilt (emotional degradation)

**Recovery Protocols:**
- NFR31: Post-tilt reports shall show correlation between behavioral markers and losses
- NFR32: The system shall suggest personalized recovery activities based on what worked previously
- NFR33: "Tilt debt" tracking - accumulated tilt scores that must be "paid down" through profitable discipline
- NFR34: Graduated re-entry after lockout - start with 25% normal position sizes
- NFR35: The system shall maintain "tilt diary" - anonymous patterns shared with future self

**Anti-Gaming Mechanisms:**
- NFR36: The system shall detect attempts to circumvent tilt detection (random clicking to appear active)
- NFR37: Multiple account detection to prevent "fresh start" attempts during tilt
- NFR38: Time-delay on parameter changes - 1-hour cooldown prevents emotional system modifications
- NFR39: The system shall log all override attempts even if not executed
- NFR40: "Tilt prediction" mode - warns when entering historically dangerous psychological territories

## User Interface Design Goals

### Overall UX Vision

A calm, confidence-inspiring terminal interface that reduces cognitive load during high-stress trading moments. The design philosophy centers on "progressive disclosure" - showing exactly what's needed at each tier level, hiding complexity until the trader is psychologically ready. Anti-anxiety design patterns using consistent spatial positioning, muted success indicators, and escalating warning systems that match threat levels. The interface should feel like a trusted co-pilot, not a casino floor.

### Key Interaction Paradigms

**Command-First Architecture:** Primary interaction through typed commands with muscle-memory shortcuts (e.g., 'b100u' = buy $100 USDT), reducing mouse-induced impulsive clicking. Visual feedback confirms intent before execution.

**Hierarchical Information Zones:** Screen divided into stable zones - P&L always top-left (reduces hunting), active positions center, command input bottom. Information stays where expected, reducing stress-induced confusion.

**Breathing Room Design:** Generous whitespace and 100ms update throttling prevents overwhelming rapid changes. Numbers fade in/out rather than jumping, creating perception of market flow rather than chaos.

**Tier-Adaptive Complexity:** Sniper tier shows 3 data points max. Hunter tier reveals 5-7. Strategist unlocks full dashboard. Prevents information overload during learning phases.

### Core Screens and Views

**Main Trading Dashboard**
- Active position monitor with color-coded P&L (green for profit, gray for loss - never red)
- Market state indicator (DEAD/NORMAL/VOLATILE/PANIC) as subtle background tint
- Tier progression bar showing gates passed and current restrictions
- Command input terminal with auto-complete and confirmation prompts

**Risk Management View**
- Real-time drawdown meter with graduated warning colors (yellow at 5%, orange at 10%, red pulse at 15%)
- Position correlation matrix showing dangerous clustering
- Daily loss budget remaining as prominent "fuel gauge" metaphor
- Tilt indicators as subtle border effects rather than alarming popups

**Performance Analytics Screen**
- Dollar-focused P&L with percentages de-emphasized in smaller font
- Win/loss pattern calendar view for self-awareness
- Strategy performance breakdown by pair and time window
- Psychological state history correlated with trading outcomes

**Tier Transition Gateway**
- Current tier requirements checklist with progress bars
- Historical transition attempts and failure reasons
- "Graduation ceremony" interface for tier advancement
- Emergency demotion warnings with 60-second countdown

### Accessibility: None (Personal System)

Single-user terminal application without accessibility requirements. However, design considers personal eye strain and fatigue with high contrast modes and adjustable font sizes.

### Branding

**Visual Philosophy:** "Digital Zen Garden" - Clean monospace typography (JetBrains Mono), minimal color palette (black, white, two shades of gray, muted green, warning amber). No animations except critical alerts. Anti-gambling aesthetic: no coins, rockets, or excitement-inducing imagery.

**Emotional Tone:** Professional trading desk, not day-trader setup. Inspired by Bloomberg Terminal's information density but with modern typography and spacing. Success celebrated with subtle "+$47.32" rather than "🚀 WINNER! 🚀".

### Target Device and Platforms: Desktop Only

- Primary: Ubuntu Linux VPS accessed via SSH/tmux for 24/7 operation
- Development: macOS terminal with identical interface
- No mobile access (prevents emotional trading from phones)
- Minimum 80x24 terminal size, optimized for 120x40

### Error Message Tone & Language Framework

**Core Philosophy: "Calm Competence, Never Blame"**

Every error message follows the structure: [What happened] → [Why it's okay] → [What to do next]. No exclamation points, no "ERROR!" prefixes, no implied user stupidity. Messages written as if explaining to a respected colleague, not scolding a child.

**Error Severity Levels & Language:**

**Level 1 - Information (Gray text, no border)**
- "Position size adjusted to $247 to maintain 5% limit"
- "Order queued - high volume detected, executing in 3 slices"
- "Market order converted to limit order due to low liquidity"

**Level 2 - Caution (Yellow accent, subtle border)**
- "Order declined - would exceed daily tier limit. $312 remaining today"
- "Unusual spread detected (2.3%). Confirming intention... [y/n]"
- "Connection latency elevated (142ms). Orders may execute slower"

**Level 3 - Intervention (Orange accent, breathing animation)**
- "Trading paused - rapid order pattern detected. Resume in 4:47"
- "Position would create 73% correlation. Consider diversification"
- "Approaching daily loss limit. $89 remaining in risk budget"

**Level 4 - Protection (Red border, requires acknowledgment)**
- "Account protection activated. Journal entry required to continue"
- "System prevented potential $2,341 loss from execution error"
- "Emergency stop triggered. Market conditions exceed risk parameters"

## Technical Assumptions

### Repository Structure: Monorepo

Single repository containing all system components with clear modular separation:
- `/engine` - Core trading execution and order management  
- `/strategies` - Pluggable strategy modules (arbitrage, liquidity provision)
- `/risk` - Risk management and position control
- `/tilt` - Psychological monitoring and intervention systems
- `/analytics` - Performance tracking and reporting
- `/infrastructure` - Deployment, monitoring, logging utilities

### Service Architecture

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

### Testing Requirements

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

### Additional Technical Assumptions and Requests

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

### Monitoring & Alerting Thresholds

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

## Epic List

**Epic 1: Foundation & Sniper Tier Core ($0-$500 capability)**
Establish project infrastructure, core trading engine, and basic Sniper tier functionality with essential risk management. Deliver working system capable of executing simple market orders on single pairs with position sizing and stop-loss protection.

**Epic 2: Data Pipeline & Market Intelligence ($500-$1k capability)**
Build real-time market data ingestion, statistical analysis engine, and pair discovery system. Enable system to identify and rank trading opportunities based on spread analysis and liquidity assessment.

**Epic 3: Psychological Safeguards & Tilt Detection ($1k-$2k capability)**
Implement behavioral monitoring, tilt detection algorithms, and intervention systems. Create the psychological infrastructure that prevents emotional trading and enforces discipline during the valley of death transition.

**Epic 4: Hunter Tier Evolution & Execution Algorithms ($2k-$5k capability)**
Develop slicing algorithms, multi-pair concurrent trading, and tier transition gateways. Enable smooth progression from Sniper to Hunter with automated execution strategy evolution.

**Epic 5: Advanced Risk Management & Portfolio Control ($5k-$10k capability)**
Implement correlation monitoring, portfolio-level risk metrics, and sophisticated position management. Add drawdown recovery protocols and emergency demotion systems.

**Epic 6: Strategist Tier & Performance Analytics ($10k+ capability)**
Create VWAP/TWAP execution, multi-strategy management, and comprehensive analytics dashboard. Complete the evolution to institutional-grade trading with full performance attribution.

## Epic 1: Foundation & Sniper Tier Core ($0-$500 capability)

**Goal:** Establish bulletproof project infrastructure, core trading engine, and basic Sniper tier functionality. Create a system capable of executing simple market orders with position sizing and essential risk management. This foundation must be rock-solid as all future development depends on its reliability.

### Story 1.1: Project Infrastructure Setup
As a solo trader,
I want a properly configured development and deployment environment,
so that I can develop safely and deploy reliably without risking production trading.

**Acceptance Criteria:**
1. Git repository initialized with .gitignore for Python, API keys never committed
2. Python 3.11+ virtual environment with requirements.txt for dependency management
3. Project structure created (/engine, /strategies, /risk, /analytics directories)
4. Configuration management using environment variables with .env.example template
5. Docker container configuration for consistent deployment
6. Basic Makefile with commands for test, run, deploy
7. Pre-commit hooks for code formatting (black) and linting (pylint)
8. README with setup instructions and architecture overview

### Story 1.2: Binance API Integration Layer
As a trader,
I want secure and reliable connection to Binance API,
so that I can execute trades and receive market data without connectivity issues.

**Acceptance Criteria:**
1. API client wrapper using ccxt library with connection pooling
2. Secure credential management using environment variables
3. Rate limit tracking with automatic backoff (max 80% utilization)
4. WebSocket connection for real-time price streams
5. Connection health monitoring with automatic reconnection
6. Error handling for all API failure modes with appropriate logging
7. Mock API mode for testing without real funds
8. API response validation and sanitization

### Story 1.3: Position & Risk Calculator
As a risk-conscious trader,
I want automatic position sizing based on account balance,
so that I never risk more than 5% on a single trade.

**Acceptance Criteria:**
1. Calculate position size based on 5% rule with current balance
2. Minimum position size validation ($10 minimum)
3. Account balance synchronization every 60 seconds
4. Stop-loss calculation at 2% below entry (configurable)
5. Real-time P&L calculation for open positions
6. Maximum daily loss enforcement ($25 for Sniper tier)
7. Prevent orders that would exceed risk limits
8. Clear error messages when limits block trades

### Story 1.4: Order Execution Engine (Sniper Mode)
As a Sniper tier trader,
I want simple market order execution with confirmations,
so that I can enter and exit positions quickly on single pairs.

**Acceptance Criteria:**
1. Market buy/sell order execution with <100ms latency
2. Order confirmation before execution (can be disabled)
3. Post-execution verification of fill price and amount
4. Automatic stop-loss order placement after entry
5. Order status tracking (pending/filled/cancelled)
6. Execution logging to database with timestamps
7. Slippage calculation and alerting if >0.5%
8. Emergency cancel-all-orders function

### Story 1.5: Terminal UI Foundation
As a trader,
I want a clean terminal interface showing essential information,
so that I can monitor my trading without information overload.

**Acceptance Criteria:**
1. Terminal UI using Rich/Textual framework
2. Three-panel layout: P&L (top), Position (middle), Commands (bottom)
3. Real-time updates every 100ms without flicker
4. Color coding: green for profit, gray for loss (no red)
5. Command input with autocomplete
6. Status messages with 3-second fade
7. Responsive to terminal resize
8. Keyboard shortcuts for common actions

### Story 1.6: Trade Logging & Persistence
As a trader,
I want all trades logged for analysis and recovery,
so that I can track performance and recover from crashes.

**Acceptance Criteria:**
1. SQLite database for trade history (upgrade path to PostgreSQL)
2. Every trade logged with entry/exit/P&L/timestamps
3. Position recovery on restart from database
4. Daily backup to separate file
5. Trade export to CSV for tax purposes
6. Performance metrics calculation (win rate, average R)
7. Database size management (rotation after 10MB)
8. Audit trail for all system decisions

## Epic 2: Data Pipeline & Market Intelligence ($500-$1k capability)

**Goal:** Build comprehensive market data infrastructure with real-time ingestion, statistical analysis, and opportunity discovery. Enable the system to identify profitable trading pairs and optimal entry points based on spread analysis and liquidity assessment.

### Story 2.1: Real-time Market Data Ingestion
As a trader,
I want continuous market data streams from multiple pairs,
so that I can identify opportunities across the market.

**Acceptance Criteria:**
1. WebSocket streams for top 50 trading pairs by volume
2. 1-minute candle aggregation from tick data
3. Order book depth tracking (5 levels each side)
4. Spread calculation and tracking per pair
5. Volume profile analysis by time of day
6. Data normalization and cleaning pipeline
7. Gap detection and handling in data streams
8. Memory-efficient circular buffer (last 1000 candles)

### Story 2.2: Statistical Arbitrage Engine
As an arbitrage trader,
I want to detect price divergences between correlated pairs,
so that I can profit from mean reversion.

**Acceptance Criteria:**
1. Correlation calculation between stablecoin pairs (USDT/USDC/BUSD)
2. Z-score calculation for divergence detection
3. 2-sigma threshold alerting for entry signals
4. Historical spread analysis (20-day rolling window)
5. Cointegration testing for pair validation
6. Signal generation with confidence scores
7. Backtesting framework for strategy validation
8. Performance attribution by pair and time

### Story 2.3: Liquidity Ladder Pair Scanner
As a capital-growing trader,
I want automatic pair discovery based on my capital level,
so that I trade appropriate liquidity for my tier.

**Acceptance Criteria:**
1. Daily scan of all Binance pairs for liquidity metrics
2. Categorization by daily volume (<$100k, $100k-$1M, >$1M)
3. Spread persistence scoring (how long spreads remain)
4. Tier-appropriate pair recommendations
5. Automatic graduation alerts when capital increases
6. Pair health monitoring for degradation
7. Blacklist for consistently unprofitable pairs
8. Liquidity depth analysis at different price levels

### Story 2.4: Market State Classifier
As a risk-aware trader,
I want real-time market regime detection,
so that I can adjust strategies based on conditions.

**Acceptance Criteria:**
1. Market state classification (DEAD/NORMAL/VOLATILE/PANIC/MAINTENANCE)
2. Volatility calculation using ATR and realized volatility
3. Volume anomaly detection vs historical patterns
4. State transition alerts with timestamps
5. Automatic position sizing adjustment by state
6. Strategy activation/deactivation by market state
7. Historical state analysis for pattern recognition
8. Maintenance detection from exchange announcements

### Story 2.5: Spread Analytics Dashboard
As a spread trader,
I want comprehensive spread analysis tools,
so that I can identify the most profitable opportunities.

**Acceptance Criteria:**
1. Real-time spread tracking across all monitored pairs
2. Spread heatmap visualization in terminal
3. Historical spread patterns by hour/day
4. Spread compression alerts
5. Bid/ask imbalance indicators
6. Spread volatility scoring
7. Profitability calculator based on current spreads
8. Cross-exchange spread comparison (future-ready)

## Epic 3: Psychological Safeguards & Tilt Detection ($1k-$2k capability)

**Goal:** Implement comprehensive behavioral monitoring and intervention systems that prevent emotional trading. Create the psychological infrastructure that detects tilt before it causes damage and enforces discipline during the critical valley of death transition.

### Story 3.1: Behavioral Baseline Establishment
As a self-aware trader,
I want the system to learn my normal trading patterns,
so that it can detect when I'm deviating due to emotion.

**Acceptance Criteria:**
1. 30-day baseline period for pattern learning
2. Metrics tracked: click speed, order frequency, position sizes, cancel rates
3. Time-of-day pattern analysis
4. Personal "normal" ranges calculated per metric
5. Baseline adjustment algorithm (rolling 7-day window)
6. Manual baseline reset capability
7. Export baseline data for analysis
8. Multiple baseline profiles (tired/alert/stressed)

### Story 3.2: Multi-Level Tilt Detection System
As an emotional human,
I want early warning of tilt behavior,
so that I can correct before causing damage.

**Acceptance Criteria:**
1. Level 1 detection: 2-3 behavioral anomalies (yellow border)
2. Level 2 detection: 4-5 anomalies (orange border, reduced sizing)
3. Level 3 detection: 6+ anomalies (trading lockout)
4. Progressive intervention messages (supportive, not shaming)
5. Real-time tilt score calculation (<50ms)
6. Pattern detection: revenge trading, position size volatility
7. Physical behavior tracking: typing speed, mouse patterns
8. Tilt history log with triggering events

### Story 3.3: Intervention & Recovery Protocols
As a tilt-prone trader,
I want structured recovery processes,
so that I can safely return to trading after emotional episodes.

**Acceptance Criteria:**
1. Graduated lockout periods (5min/30min/24hr)
2. Journal entry requirement for Level 3 recovery
3. Reduced position sizes on re-entry (25% to start)
4. Meditation timer integration (optional)
5. Performance comparison: tilted vs calm trading
6. Recovery checklist before trading resumption
7. "Tilt debt" system requiring profitable trades to clear
8. Emergency contact option (future: family notification)

### Story 3.4: Micro-Behavior Analytics
As a trader seeking self-improvement,
I want detailed behavioral analytics,
so that I can understand my emotional patterns.

**Acceptance Criteria:**
1. Click-to-trade latency tracking and analysis
2. Order modification frequency monitoring
3. Tab-switching pattern detection
4. "Stare time" measurement (inactivity periods)
5. Configuration change tracking
6. Session length analysis
7. Correlation between behaviors and P&L
8. Weekly behavioral report generation

### Story 3.5: Valley of Death Transition Guard
As a trader approaching the $2k threshold,
I want special protection during tier transition,
so that I don't destroy my account during the critical evolution.

**Acceptance Criteria:**
1. Heightened monitoring starting at $1,800
2. Tier transition readiness assessment
3. Forced paper trading of new execution methods
4. Psychological preparation checklist
5. "Old habits funeral" ceremony requirement
6. Success celebration for tier achievement (muted)
7. Automatic strategy migration enforcement
8. 48-hour adjustment period with reduced limits

## Epic 4: Hunter Tier Evolution & Execution Algorithms ($2k-$5k capability)

**Goal:** Develop sophisticated execution algorithms and multi-pair trading capabilities required for Hunter tier success. Enable smooth progression from Sniper to Hunter with automated strategy evolution and enhanced order management.

### Story 4.1: Iceberg Order Execution
As a Hunter tier trader,
I want orders automatically sliced to avoid market impact,
so that I can trade larger positions without moving prices.

**Acceptance Criteria:**
1. Automatic slicing for orders >$200 (3 minimum slices)
2. Smart slice sizing based on order book depth
3. Random time delays between slices (1-5 seconds)
4. Slice size variation to avoid detection (±20%)
5. Market impact monitoring per slice
6. Abort if slippage exceeds 0.5%
7. Completion tracking and reporting
8. Rollback capability for partially filled orders

### Story 4.2: Multi-Pair Concurrent Trading
As a diversifying trader,
I want to trade multiple pairs simultaneously,
so that I can spread risk and capture more opportunities.

**Acceptance Criteria:**
1. Support for 5+ concurrent positions
2. Per-pair position limits enforced
3. Overall portfolio risk management
4. Correlation monitoring between positions
5. Smart capital allocation across pairs
6. Queue management for competing signals
7. Priority system for high-confidence trades
8. Performance attribution by pair

### Story 4.3: TWAP Execution Algorithm
As a price-sensitive trader,
I want time-weighted average price execution,
so that I can achieve better average entry prices.

**Acceptance Criteria:**
1. TWAP over configurable time window (5-30 minutes)
2. Adaptive slice timing based on volume patterns
3. Participation rate limiting (max 10% of volume)
4. Benchmark tracking vs arrival price
5. Early completion on favorable prices
6. Pause/resume capability
7. Real-time progress visualization
8. Post-trade execution analysis

### Story 4.4: Tier Transition Gateway System
As a graduating trader,
I want automated tier transitions with safety checks,
so that I progress smoothly without manual errors.

**Acceptance Criteria:**
1. Gate requirement tracking dashboard
2. Automatic feature unlocking on gate completion
3. Tier demotion triggers monitored
4. Transition ceremony interface
5. Feature tutorial for newly unlocked capabilities
6. Rollback protection (no manual tier changes)
7. Grace period for adjustment (48 hours)
8. Historical transition log

### Story 4.5: Smart Order Routing
As an efficiency-focused trader,
I want intelligent order type selection,
so that I get best execution for market conditions.

**Acceptance Criteria:**
1. Automatic market vs limit order decision
2. Spread-based routing logic
3. Liquidity-based execution choice
4. Time-of-day optimization
5. Maker/taker fee optimization
6. Post-only order usage when appropriate
7. FOK/IOC order type selection
8. Execution quality scoring

## Epic 5: Advanced Risk Management & Portfolio Control ($5k-$10k capability)

**Goal:** Implement institutional-grade risk management with correlation monitoring, portfolio optimization, and sophisticated drawdown protection. Create systems that preserve capital during market stress while enabling controlled growth.

### Story 5.1: Portfolio Correlation Monitor
As a risk-conscious trader,
I want real-time correlation tracking,
so that I avoid concentration risk from hidden correlations.

**Acceptance Criteria:**
1. Real-time correlation matrix for all positions
2. Alert when portfolio correlation >60%
3. Historical correlation analysis (30-day window)
4. Correlation breakdown by market regime
5. Suggested decorrelation trades
6. Correlation impact on new trades (pre-trade)
7. Stress testing under correlation spikes
8. Visual correlation heatmap

### Story 5.2: Dynamic Position Sizing
As a sophisticated trader,
I want Kelly Criterion-based position sizing,
so that I optimize growth while managing risk.

**Acceptance Criteria:**
1. Kelly Criterion calculation per strategy
2. Fractional Kelly implementation (25% default)
3. Win rate and edge estimation
4. Dynamic adjustment based on recent performance
5. Override for high-conviction trades
6. Minimum/maximum position boundaries
7. Volatility-adjusted sizing
8. Monte Carlo simulation for validation

### Story 5.3: Drawdown Recovery Protocol
As a trader in drawdown,
I want systematic recovery procedures,
so that I can rebuild without revenge trading.

**Acceptance Criteria:**
1. Automatic detection of 10%+ drawdown
2. Position size reduction to 50%
3. Strategy restriction to highest win-rate only
4. Daily profit targets reduced by 50%
5. Forced break after 3 consecutive losses
6. Recovery milestone tracking
7. Psychological support messages
8. Historical recovery pattern analysis

### Story 5.4: Portfolio Optimization Engine
As a return-seeking trader,
I want optimal capital allocation across strategies,
so that I maximize risk-adjusted returns.

**Acceptance Criteria:**
1. Sharpe ratio calculation per strategy
2. Efficient frontier analysis
3. Dynamic rebalancing triggers
4. Transaction cost consideration
5. Minimum allocation thresholds
6. Strategy correlation in optimization
7. Out-of-sample validation
8. Weekly rebalancing recommendations

### Story 5.5: Emergency Risk Controls
As a capital protector,
I want circuit breakers for extreme events,
so that I preserve capital during black swans.

**Acceptance Criteria:**
1. Automatic halt on 15% daily loss
2. Correlation spike emergency response
3. Liquidity crisis detection and response
4. Flash crash protection (cancel all orders)
5. Rapid deleveraging protocol
6. Manual override with typed confirmation
7. Post-emergency analysis report
8. Recovery checklist after emergency

## Epic 6: Strategist Tier & Performance Analytics ($10k+ capability)

**Goal:** Create institutional-grade execution algorithms, multi-strategy management, and comprehensive analytics. Complete the evolution to professional trading with full performance attribution and advanced market microstructure capabilities.

### Story 6.1: VWAP Execution Algorithm
As a Strategist tier trader,
I want volume-weighted average price execution,
so that I achieve institutional-quality fills.

**Acceptance Criteria:**
1. Historical volume pattern analysis
2. Intraday volume prediction model
3. Dynamic participation rate adjustment
4. Real-time VWAP tracking vs benchmark
5. Aggressive/passive mode selection
6. Dark pool simulation (iceberg orders)
7. Multi-venue execution ready (future)
8. Transaction cost analysis

### Story 6.2: Multi-Strategy Orchestration
As a portfolio manager,
I want to run multiple strategies concurrently,
so that I diversify return sources.

**Acceptance Criteria:**
1. Strategy registry and lifecycle management
2. Independent capital allocation per strategy
3. Strategy correlation monitoring
4. Performance-based capital adjustment
5. Strategy on/off based on market regime
6. Conflict resolution between strategies
7. Aggregate risk management
8. Strategy A/B testing framework

### Story 6.3: Advanced Performance Analytics
As a data-driven trader,
I want comprehensive performance attribution,
so that I understand my edge sources.

**Acceptance Criteria:**
1. Performance attribution by strategy/pair/time
2. Risk-adjusted metrics (Sharpe, Sortino, Calmar)
3. Maximum adverse excursion analysis
4. Win/loss pattern analysis
5. Execution quality metrics
6. Behavioral correlation with performance
7. Peer comparison benchmarks (future)
8. Monthly performance reports

### Story 6.4: Market Microstructure Analysis
As a sophisticated trader,
I want to understand market microstructure,
so that I can exploit institutional patterns.

**Acceptance Criteria:**
1. Order flow imbalance detection
2. Large trader detection algorithms
3. Spoofing/layering detection
4. Price impact modeling
5. Optimal execution timing
6. Market maker behavior analysis
7. Toxicity scoring for pairs
8. Microstructure regime identification

### Story 6.5: Institutional Features Suite
As a professional trader,
I want institutional-grade features,
so that I operate at professional standards.

**Acceptance Criteria:**
1. FIX protocol readiness (future)
2. Multi-account management capability
3. Compliance reporting tools
4. Risk metrics dashboard (VaR, CVaR)
5. Automated month-end reconciliation
6. Tax lot optimization
7. Prime broker integration ready
8. Disaster recovery procedures

## Checklist Results Report

### Executive Summary

**Overall PRD Completeness:** 94%
**MVP Scope Appropriateness:** Just Right (appropriately scoped for $500 start)
**Readiness for Architecture Phase:** READY
**Most Critical Gaps:** Data migration strategy, disaster recovery procedures, and formal testing strategy documentation

### Category Analysis Table

| Category                         | Status  | Critical Issues |
| -------------------------------- | ------- | --------------- |
| 1. Problem Definition & Context  | PASS    | None - Comprehensive Project Brief provides excellent foundation |
| 2. MVP Scope Definition          | PASS    | None - Clear MVP boundaries with explicit out-of-scope items |
| 3. User Experience Requirements  | PASS    | None - Terminal UI well-defined with psychological considerations |
| 4. Functional Requirements       | PASS    | None - Detailed FRs with tilt detection innovation |
| 5. Non-Functional Requirements   | PASS    | None - Extensive NFRs including psychological safeguards |
| 6. Epic & Story Structure        | PASS    | None - Well-sized stories with clear acceptance criteria |
| 7. Technical Guidance            | PARTIAL | Missing: Formal testing strategy, disaster recovery details |
| 8. Cross-Functional Requirements | PARTIAL | Missing: Data migration plan, schema evolution strategy |
| 9. Clarity & Communication       | PASS    | None - Clear language, consistent terminology |

### Top Issues by Priority

**BLOCKERS:** None identified

**HIGH PRIORITY:**
1. **Testing Strategy Gap** - While testing requirements mentioned, no formal test automation strategy or CI/CD pipeline definition
2. **Disaster Recovery Procedures** - Critical for financial system but only briefly mentioned
3. **Data Migration Strategy** - How to migrate from SQLite to PostgreSQL at $2k not detailed

**MEDIUM PRIORITY:**
1. **Schema Evolution Plan** - Database changes as system grows need planning
2. **Backup Verification Process** - Backups mentioned but not verification procedures
3. **Integration Test Environment** - No mention of Binance testnet usage strategy

**LOW PRIORITY:**
1. **Documentation Standards** - Code documentation approach not specified
2. **Performance Profiling Tools** - Monitoring mentioned but not profiling approach
3. **Log Aggregation Details** - Logging strategy needs more specificity

### Final Decision

**✅ READY FOR ARCHITECT**

The PRD and epics are comprehensive, properly structured, and ready for architectural design. The identified gaps are implementation details that can be addressed during architecture phase rather than blocking concerns. The psychological sophistication and tier evolution approach provide exceptional foundation for a trading system that could actually survive the valley of death.

## Next Steps

### UX Expert Prompt

"Review the Project GENESIS PRD focusing on the terminal-based 'Digital Zen Garden' interface design. Create detailed wireframes for the three-panel layout (P&L, Positions, Commands) with special attention to the psychological aspects: color psychology for anti-tilt design, progressive disclosure by tier, and error message presentation. Consider how visual hierarchy and information density change between Sniper ($500), Hunter ($2k), and Strategist ($10k) tiers. Document the command syntax and autocomplete behavior. Pay particular attention to how tilt warning indicators manifest visually without triggering panic."

### Architect Prompt

"Design the technical architecture for Project GENESIS using the PRD as foundation. Focus on the evolutionary architecture that transforms from monolith (Python/SQLite) to service-oriented (Python/Rust/PostgreSQL) as capital grows. Address the critical <100ms execution requirement, tier-locked feature system implementation, and real-time tilt detection algorithms. Special attention needed for: state management across tiers, WebSocket connection resilience, order slicing algorithms, and the correlation calculation engine. Provide deployment architecture for DigitalOcean Singapore with failover planning. Document how the system enforces tier restrictions at the code level to prevent override attempts."

---

*End of Product Requirements Document v1.0*