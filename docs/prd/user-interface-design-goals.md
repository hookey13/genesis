# User Interface Design Goals

## Overall UX Vision

A calm, confidence-inspiring terminal interface that reduces cognitive load during high-stress trading moments. The design philosophy centers on "progressive disclosure" - showing exactly what's needed at each tier level, hiding complexity until the trader is psychologically ready. Anti-anxiety design patterns using consistent spatial positioning, muted success indicators, and escalating warning systems that match threat levels. The interface should feel like a trusted co-pilot, not a casino floor.

## Key Interaction Paradigms

**Command-First Architecture:** Primary interaction through typed commands with muscle-memory shortcuts (e.g., 'b100u' = buy $100 USDT), reducing mouse-induced impulsive clicking. Visual feedback confirms intent before execution.

**Hierarchical Information Zones:** Screen divided into stable zones - P&L always top-left (reduces hunting), active positions center, command input bottom. Information stays where expected, reducing stress-induced confusion.

**Breathing Room Design:** Generous whitespace and 100ms update throttling prevents overwhelming rapid changes. Numbers fade in/out rather than jumping, creating perception of market flow rather than chaos.

**Tier-Adaptive Complexity:** Sniper tier shows 3 data points max. Hunter tier reveals 5-7. Strategist unlocks full dashboard. Prevents information overload during learning phases.

## Core Screens and Views

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

## Accessibility: None (Personal System)

Single-user terminal application without accessibility requirements. However, design considers personal eye strain and fatigue with high contrast modes and adjustable font sizes.

## Branding

**Visual Philosophy:** "Digital Zen Garden" - Clean monospace typography (JetBrains Mono), minimal color palette (black, white, two shades of gray, muted green, warning amber). No animations except critical alerts. Anti-gambling aesthetic: no coins, rockets, or excitement-inducing imagery.

**Emotional Tone:** Professional trading desk, not day-trader setup. Inspired by Bloomberg Terminal's information density but with modern typography and spacing. Success celebrated with subtle "+$47.32" rather than "🚀 WINNER! 🚀".

## Target Device and Platforms: Desktop Only

- Primary: Ubuntu Linux VPS accessed via SSH/tmux for 24/7 operation
- Development: macOS terminal with identical interface
- No mobile access (prevents emotional trading from phones)
- Minimum 80x24 terminal size, optimized for 120x40

## Error Message Tone & Language Framework

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
