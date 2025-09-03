# Requirements

## Functional Requirements

**Core Trading Engine:**
- FR1: The system shall enforce tier-locked execution strategies that automatically evolve based on account balance thresholds ($500/$2k/$10k) ✅ (Epic 10)
- FR2: Market orders shall be used exclusively for positions <$500, with mandatory 3-slice execution for $500-$2k range ✅ (Epic 10)
- FR3: The system shall monitor and classify market states (DEAD/NORMAL/VOLATILE/PANIC) in real-time with <5 second detection latency ✅ (Epic 10)
- FR4: Statistical arbitrage engine shall identify and execute mean reversion trades when spreads exceed 2-sigma historical norms ✅ (Epic 10)
- FR5: Position sizing shall automatically calculate based on current capital with hard 5% maximum per position ✅ (Epic 10)
- FR6: The system shall scan and rank trading pairs by liquidity appropriateness for current capital tier ✅ (Epic 10)
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

## Non-Functional Requirements

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
