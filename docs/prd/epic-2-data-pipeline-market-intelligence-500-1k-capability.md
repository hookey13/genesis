# Epic 2: Data Pipeline & Market Intelligence ($500-$1k capability)

**Goal:** Build comprehensive market data infrastructure with real-time ingestion, statistical analysis, and opportunity discovery. Enable the system to identify profitable trading pairs and optimal entry points based on spread analysis and liquidity assessment.

## Story 2.1: Real-time Market Data Ingestion
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

## Story 2.2: Statistical Arbitrage Engine
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

## Story 2.3: Liquidity Ladder Pair Scanner
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

## Story 2.4: Market State Classifier
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

## Story 2.5: Spread Analytics Dashboard
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
