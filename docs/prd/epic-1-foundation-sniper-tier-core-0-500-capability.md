# Epic 1: Foundation & Sniper Tier Core ($0-$500 capability)

**Goal:** Establish bulletproof project infrastructure, core trading engine, and basic Sniper tier functionality. Create a system capable of executing simple market orders with position sizing and essential risk management. This foundation must be rock-solid as all future development depends on its reliability.

## Story 1.1: Project Infrastructure Setup
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

## Story 1.2: Binance API Integration Layer
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

## Story 1.3: Position & Risk Calculator
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

## Story 1.4: Order Execution Engine (Sniper Mode)
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

## Story 1.5: Terminal UI Foundation
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

## Story 1.6: Trade Logging & Persistence
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
