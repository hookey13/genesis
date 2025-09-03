# Epic 10: Core Trading Brain Implementation ($100k+ Production Ready)

**Goal:** Transform Genesis from an exceptional infrastructure framework into a fully operational algorithmic trading system by implementing the core trading strategies, market analysis engine, and live trading loop. This epic addresses the critical 60% gap identified in the production readiness audit, delivering the actual "trading brain" that transforms market data into profitable trading decisions across all three tiers (Sniper, Hunter, Strategist).

## Story 10.1: Market Analysis Engine & Arbitrage Detection
As a trading system architect,
I want to implement the core market analysis engine with arbitrage opportunity detection,
So that the system can identify and rank profitable trading opportunities in real-time across multiple pairs and exchanges.

**Acceptance Criteria:**
1. Real-time order book analysis with depth aggregation
2. Cross-exchange arbitrage opportunity detection (>0.3% profit threshold)
3. Triangular arbitrage calculation for multi-hop opportunities
4. Spread analysis with historical baseline comparison
5. Liquidity depth assessment for position sizing
6. Market microstructure anomaly detection
7. Opportunity ranking by risk-adjusted profit potential
8. Latency-optimized calculations (<10ms per pair)
9. Statistical significance testing for opportunities
10. Integration with existing WebSocket market data streams

**Implementation Details:**
```python
# genesis/analytics/market_analyzer.py
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta
import numpy as np
from collections import deque

import structlog

logger = structlog.get_logger()

@dataclass
class ArbitrageOpportunity:
    symbol: str
    exchange_buy: str
    exchange_sell: str
    buy_price: Decimal
    sell_price: Decimal
    max_quantity: Decimal
    profit_pct: Decimal
    confidence: float
    latency_ms: float
    expires_at: datetime
    execution_path: List[str]
    
class MarketAnalyzer:
    """Core market analysis engine for opportunity detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_profit_pct = Decimal(str(config.get('min_profit_pct', 0.3)))
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_latency_ms = config.get('max_latency_ms', 100)
        self.orderbook_cache = {}
        self.spread_history = defaultdict(lambda: deque(maxlen=1000))
        self.opportunity_tracker = {}
        
    async def analyze_market_data(
        self,
        market_data: Dict[str, Any]
    ) -> List[ArbitrageOpportunity]:
        """Analyze market data for trading opportunities."""
        start_time = asyncio.get_event_loop().time()
        opportunities = []
        
        try:
            # Update orderbook cache
            await self._update_orderbook_cache(market_data)
            
            # Detect direct arbitrage
            direct_arb = await self._find_direct_arbitrage()
            opportunities.extend(direct_arb)
            
            # Detect triangular arbitrage
            triangular_arb = await self._find_triangular_arbitrage()
            opportunities.extend(triangular_arb)
            
            # Detect statistical arbitrage
            stat_arb = await self._find_statistical_arbitrage()
            opportunities.extend(stat_arb)
            
            # Filter and rank opportunities
            filtered = self._filter_opportunities(opportunities)
            ranked = self._rank_opportunities(filtered)
            
            # Track processing latency
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
            if latency > self.max_latency_ms:
                logger.warning(
                    "market_analysis_slow",
                    latency_ms=latency,
                    opportunities_found=len(ranked)
                )
            
            return ranked[:10]  # Return top 10 opportunities
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return []
    
    async def _find_direct_arbitrage(self) -> List[ArbitrageOpportunity]:
        """Find direct arbitrage between exchanges."""
        opportunities = []
        
        for symbol, books in self.orderbook_cache.items():
            if len(books) < 2:
                continue
                
            # Find best bid and ask across exchanges
            best_bid = max(books.items(), key=lambda x: x[1]['bid_price'])
            best_ask = min(books.items(), key=lambda x: x[1]['ask_price'])
            
            if best_bid[1]['bid_price'] > best_ask[1]['ask_price']:
                profit_pct = ((best_bid[1]['bid_price'] - best_ask[1]['ask_price']) 
                             / best_ask[1]['ask_price']) * 100
                
                if profit_pct >= self.min_profit_pct:
                    opportunities.append(
                        ArbitrageOpportunity(
                            symbol=symbol,
                            exchange_buy=best_ask[0],
                            exchange_sell=best_bid[0],
                            buy_price=best_ask[1]['ask_price'],
                            sell_price=best_bid[1]['bid_price'],
                            max_quantity=min(
                                best_ask[1]['ask_size'],
                                best_bid[1]['bid_size']
                            ),
                            profit_pct=profit_pct,
                            confidence=self._calculate_confidence(profit_pct),
                            latency_ms=0,  # Will be updated
                            expires_at=datetime.utcnow() + timedelta(seconds=5),
                            execution_path=[best_ask[0], best_bid[0]]
                        )
                    )
        
        return opportunities
```

**File Locations:**
- `genesis/analytics/market_analyzer.py` (new file)
- `genesis/analytics/arbitrage_detector.py` (new file)
- `genesis/analytics/spread_tracker.py` (new file)
- `tests/unit/test_market_analyzer.py` (new test)
- `tests/integration/test_arbitrage_detection.py` (new test)

## Story 10.2: Sniper Tier Trading Strategy Implementation
As a strategy developer,
I want to implement the complete Sniper tier trading strategy with simple arbitrage and momentum detection,
So that the system can execute profitable trades in the $500-$2k capital range with appropriate risk management.

**Acceptance Criteria:**
1. Simple arbitrage strategy for spread capture
2. Momentum breakout detection on single pairs
3. Position sizing using Kelly criterion (capped at 2% risk)
4. Entry signal generation with confidence scoring
5. Exit signal generation (take profit and stop loss)
6. Strategy state persistence and recovery
7. Performance tracking per strategy instance
8. Integration with existing risk engine
9. Backtesting validation showing positive expectancy
10. Paper trading mode for strategy validation

**Implementation Details:**
```python
# genesis/strategies/sniper/simple_arbitrage.py
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from genesis.strategies.base import BaseStrategy, StrategyState
from genesis.core.models import Signal, SignalType, Order, OrderType
from genesis.analytics.market_analyzer import MarketAnalyzer
from genesis.risk.position_sizer import KellySizer

class SniperArbitrageStrategy(BaseStrategy):
    """Simple arbitrage strategy for Sniper tier."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.analyzer = MarketAnalyzer(config)
        self.position_sizer = KellySizer(config)
        self.min_profit_pct = Decimal(str(config.get('min_profit_pct', 0.3)))
        self.max_position_pct = Decimal(str(config.get('max_position_pct', 0.02)))
        self.stop_loss_pct = Decimal(str(config.get('stop_loss_pct', 1.0)))
        self.take_profit_pct = Decimal(str(config.get('take_profit_pct', 0.5)))
        self.active_positions = {}
        self.performance_tracker = PerformanceTracker()
        
    async def analyze(self, market_data: Dict) -> Optional[Signal]:
        """Generate trading signals from market data."""
        try:
            # Update strategy state
            self.state = StrategyState.ANALYZING
            
            # Find arbitrage opportunities
            opportunities = await self.analyzer.analyze_market_data(market_data)
            
            if not opportunities:
                return None
            
            # Select best opportunity
            best = self._select_best_opportunity(opportunities)
            
            if not best:
                return None
            
            # Calculate position size
            position_size = await self.position_sizer.calculate_size(
                opportunity=best,
                account_balance=await self._get_account_balance(),
                existing_positions=self.active_positions
            )
            
            if position_size < self.min_order_size:
                logger.debug(
                    "position_too_small",
                    calculated_size=position_size,
                    min_size=self.min_order_size
                )
                return None
            
            # Generate entry signal
            signal = Signal(
                type=SignalType.BUY,
                symbol=best.symbol,
                price=best.buy_price,
                quantity=position_size,
                confidence=best.confidence,
                strategy_name=self.__class__.__name__,
                metadata={
                    'opportunity_type': 'arbitrage',
                    'profit_pct': float(best.profit_pct),
                    'exchange_buy': best.exchange_buy,
                    'exchange_sell': best.exchange_sell,
                    'stop_loss': float(best.buy_price * (1 - self.stop_loss_pct / 100)),
                    'take_profit': float(best.buy_price * (1 + self.take_profit_pct / 100)),
                    'expires_at': best.expires_at.isoformat()
                }
            )
            
            # Update state
            self.state = StrategyState.SIGNAL_GENERATED
            
            # Track signal generation
            self.performance_tracker.record_signal(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Strategy analysis failed: {e}")
            self.state = StrategyState.ERROR
            return None
    
    async def manage_positions(self) -> List[Signal]:
        """Generate exit signals for existing positions."""
        exit_signals = []
        
        for position_id, position in self.active_positions.items():
            # Check stop loss
            if position.unrealized_pnl_pct <= -self.stop_loss_pct:
                exit_signals.append(
                    self._create_exit_signal(position, "stop_loss")
                )
            
            # Check take profit
            elif position.unrealized_pnl_pct >= self.take_profit_pct:
                exit_signals.append(
                    self._create_exit_signal(position, "take_profit")
                )
            
            # Check expiration
            elif datetime.utcnow() > position.expires_at:
                exit_signals.append(
                    self._create_exit_signal(position, "expired")
                )
        
        return exit_signals

# genesis/strategies/sniper/momentum_breakout.py
class MomentumBreakoutStrategy(BaseStrategy):
    """Momentum breakout strategy for trending markets."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_periods = config.get('lookback_periods', 20)
        self.breakout_threshold = Decimal(str(config.get('breakout_threshold', 2.0)))
        self.volume_multiplier = config.get('volume_multiplier', 1.5)
        self.price_history = deque(maxlen=self.lookback_periods)
        self.volume_history = deque(maxlen=self.lookback_periods)
```

**File Locations:**
- `genesis/strategies/sniper/simple_arbitrage.py` (new file)
- `genesis/strategies/sniper/momentum_breakout.py` (new file)
- `genesis/strategies/base.py` (update with StrategyState enum)
- `genesis/risk/position_sizer.py` (new file - Kelly criterion)
- `tests/unit/test_sniper_strategies.py` (new test)
- `tests/integration/test_strategy_execution.py` (new test)

## Story 10.3: Hunter Tier Advanced Strategy Suite
As a strategy developer,
I want to implement Hunter tier strategies including mean reversion and pairs trading,
So that the system can handle $2k-$10k capital with multi-pair concurrent execution and advanced order types.

**Acceptance Criteria:**
1. Mean reversion strategy with Bollinger Bands
2. Statistical pairs trading with cointegration testing
3. Multi-pair portfolio management (up to 5 concurrent)
4. Iceberg order execution for large positions
5. Dynamic position sizing based on volatility
6. Correlation-based risk management
7. Strategy performance attribution
8. Automated strategy switching based on market regime
9. Integration with slicing algorithms
10. Real-time P&L tracking per strategy

**Implementation Details:**
```python
# genesis/strategies/hunter/mean_reversion.py
class MeanReversionStrategy(BaseStrategy):
    """Mean reversion using statistical indicators."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bb_periods = config.get('bb_periods', 20)
        self.bb_std = config.get('bb_std', 2.0)
        self.rsi_period = config.get('rsi_period', 14)
        self.min_zscore = config.get('min_zscore', 2.0)
        self.indicators = TechnicalIndicators()
        self.regime_detector = MarketRegimeDetector()
        
    async def analyze(self, market_data: Dict) -> Optional[Signal]:
        """Detect mean reversion opportunities."""
        # Check market regime
        regime = await self.regime_detector.detect(market_data)
        
        if regime != MarketRegime.RANGING:
            return None  # Only trade in ranging markets
        
        # Calculate indicators
        bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(
            market_data['prices'], 
            self.bb_periods, 
            self.bb_std
        )
        
        current_price = market_data['prices'][-1]
        rsi = self.indicators.rsi(market_data['prices'], self.rsi_period)
        
        # Generate signals
        if current_price < bb_lower and rsi < 30:
            return self._generate_buy_signal(market_data)
        elif current_price > bb_upper and rsi > 70:
            return self._generate_sell_signal(market_data)
        
        return None

# genesis/strategies/hunter/pairs_trading.py
class PairsTradingStrategy(BaseStrategy):
    """Statistical arbitrage between correlated pairs."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_window = config.get('lookback_window', 100)
        self.entry_zscore = config.get('entry_zscore', 2.0)
        self.exit_zscore = config.get('exit_zscore', 0.5)
        self.min_correlation = config.get('min_correlation', 0.8)
        self.cointegration_tester = CointegrationTester()
        self.spread_tracker = SpreadTracker()
```

**File Locations:**
- `genesis/strategies/hunter/mean_reversion.py` (new file)
- `genesis/strategies/hunter/pairs_trading.py` (new file)
- `genesis/strategies/hunter/portfolio_manager.py` (new file)
- `genesis/analytics/technical_indicators.py` (new file)
- `genesis/analytics/cointegration.py` (new file)

## Story 10.4: Strategist Tier Institutional Algorithms
As a strategy developer,
I want to implement Strategist tier algorithms including VWAP/TWAP execution and market making,
So that the system can handle $10k+ capital with institutional-grade execution quality.

**Acceptance Criteria:**
1. VWAP execution algorithm with intraday volume curves
2. TWAP execution with adaptive scheduling
3. Implementation shortfall minimization
4. Market making with dynamic spread adjustment
5. Smart order routing across multiple venues
6. Pre-trade TCA (Transaction Cost Analysis)
7. Post-trade performance attribution
8. Liquidity seeking algorithms
9. Dark pool integration simulation
10. Regulatory compliance reporting

**Implementation Details:**
```python
# genesis/strategies/strategist/vwap_execution.py
class VWAPExecutionStrategy(BaseStrategy):
    """Volume-Weighted Average Price execution."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.volume_curve = VolumeCurveEstimator()
        self.slicer = OrderSlicer()
        self.scheduler = ExecutionScheduler()
        
    async def execute_parent_order(
        self,
        parent_order: Order,
        market_data: Dict
    ) -> List[Order]:
        """Slice and schedule parent order for VWAP execution."""
        # Estimate intraday volume curve
        volume_profile = await self.volume_curve.estimate(
            symbol=parent_order.symbol,
            lookback_days=20
        )
        
        # Create execution schedule
        schedule = self.scheduler.create_vwap_schedule(
            parent_order=parent_order,
            volume_profile=volume_profile,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=1)
        )
        
        # Generate child orders
        child_orders = []
        for slot in schedule:
            child_order = self.slicer.create_child_order(
                parent_order=parent_order,
                quantity=slot.quantity,
                time_slot=slot.time,
                urgency=slot.urgency
            )
            child_orders.append(child_order)
        
        return child_orders

# genesis/strategies/strategist/market_maker.py
class MarketMakingStrategy(BaseStrategy):
    """Two-sided market making with inventory management."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.spread_model = SpreadModel()
        self.inventory_manager = InventoryManager()
        self.quote_generator = QuoteGenerator()
```

**File Locations:**
- `genesis/strategies/strategist/vwap_execution.py` (new file)
- `genesis/strategies/strategist/market_maker.py` (new file)
- `genesis/strategies/strategist/smart_router.py` (new file)
- `genesis/execution/volume_curve.py` (new file)
- `genesis/execution/tca_analyzer.py` (new file)

## Story 10.5: Trading Engine Integration & Orchestration
As a system architect,
I want to connect all trading components into a cohesive live trading loop,
So that strategies can execute trades in real-time with proper orchestration and state management.

**Acceptance Criteria:**
1. Main trading loop initialization in `__main__.py`
2. Strategy loading based on current tier
3. Real-time market data feed integration
4. Signal generation and validation pipeline
5. Order execution with proper error handling
6. Position tracking and P&L calculation
7. State persistence and recovery
8. Graceful shutdown procedures
9. Health monitoring and alerting
10. Performance metrics collection

**Implementation Details:**
```python
# genesis/__main__.py (update lines 218-250)
async def main():
    """Main application entry point."""
    # ... existing initialization code ...
    
    # Initialize trading engine components
    logger.info("initializing_trading_engine")
    
    # Create strategy registry and load strategies
    strategy_registry = StrategyRegistry(config)
    strategy_registry.load_tier_strategies(config.trading_tier)
    
    # Initialize trading engine
    engine = TradingEngine(
        config=config,
        gateway=gateway,
        risk_engine=risk_engine,
        strategy_registry=strategy_registry,
        state_machine=state_machine,
        tilt_detector=tilt_detector
    )
    
    # Register shutdown handlers
    def signal_handler(sig, frame):
        logger.info("shutdown_signal_received", signal=sig)
        asyncio.create_task(engine.graceful_shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start trading engine
    logger.info(
        "starting_trading_engine",
        tier=config.trading_tier,
        strategies=len(strategy_registry.get_active_strategies())
    )
    
    try:
        # Start market data feeds
        market_feed = MarketDataFeed(gateway)
        await market_feed.start()
        
        # Start trading loop
        await engine.start()
        
        # Monitor engine health
        health_monitor = HealthMonitor(engine)
        monitor_task = asyncio.create_task(health_monitor.run())
        
        # Keep application running
        while engine.is_running():
            await asyncio.sleep(1)
            
            # Periodic status update
            if int(time.time()) % 60 == 0:
                status = await engine.get_status()
                logger.info("engine_status", **status)
        
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt_received")
    except Exception as e:
        logger.error(f"Fatal error in trading engine: {e}")
        await engine.emergency_stop()
    finally:
        # Cleanup
        await market_feed.stop()
        monitor_task.cancel()
        await engine.cleanup()
        logger.info("trading_engine_stopped")

# genesis/engine/engine.py (complete implementation)
class TradingEngine:
    """Core trading engine orchestrator."""
    
    async def start(self) -> None:
        """Start the trading engine."""
        self.running = True
        self.state = EngineState.RUNNING
        
        # Start component tasks
        self.tasks = [
            asyncio.create_task(self._trading_loop()),
            asyncio.create_task(self._risk_monitor_loop()),
            asyncio.create_task(self._position_manager_loop()),
            asyncio.create_task(self._performance_tracker_loop())
        ]
        
        logger.info("trading_engine_started", tasks=len(self.tasks))
        
    async def _trading_loop(self) -> None:
        """Main trading loop."""
        while self.running:
            try:
                # Get market data
                market_data = await self.gateway.get_market_snapshot()
                
                # Process each active strategy
                strategies = self.strategy_registry.get_active_strategies()
                
                for strategy in strategies:
                    # Check if strategy should run
                    if not await self._should_run_strategy(strategy):
                        continue
                    
                    # Generate signals
                    signal = await strategy.analyze(market_data)
                    
                    if signal:
                        # Validate with risk engine
                        if await self.risk_engine.validate_signal(signal):
                            # Execute trade
                            await self._execute_signal(signal)
                        else:
                            logger.warning(
                                "signal_rejected",
                                strategy=strategy.name,
                                reason="risk_validation_failed"
                            )
                
                # Check for exit signals
                exit_signals = await self._check_exit_conditions()
                for signal in exit_signals:
                    await self._execute_signal(signal)
                
                # Rate limiting
                await asyncio.sleep(self.config.loop_interval)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await self._handle_error(e)
```

**File Locations:**
- `genesis/__main__.py` (update existing)
- `genesis/engine/engine.py` (update existing)
- `genesis/engine/orchestrator.py` (new file)
- `genesis/engine/health_monitor.py` (new file)
- `genesis/data/market_feed.py` (new file)

## Story 10.6: Strategy Backtesting & Validation Framework
As a quant researcher,
I want a comprehensive backtesting framework to validate strategies before live deployment,
So that we can ensure positive expectancy and proper risk-adjusted returns.

**Acceptance Criteria:**
1. Historical data replay with realistic slippage
2. Order fill simulation with market impact
3. Transaction cost modeling (fees, spreads)
4. Performance metrics calculation (Sharpe, Sortino, Calmar)
5. Maximum drawdown analysis
6. Monte Carlo simulation for confidence intervals
7. Walk-forward optimization
8. Out-of-sample validation
9. Strategy comparison framework
10. Automated report generation

**Implementation Details:**
```python
# genesis/backtesting/engine.py
class BacktestEngine:
    """Historical simulation engine for strategy validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.data_provider = HistoricalDataProvider()
        self.execution_simulator = ExecutionSimulator()
        self.performance_calculator = PerformanceCalculator()
        self.report_generator = BacktestReportGenerator()
        
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal = Decimal("1000")
    ) -> BacktestResult:
        """Run historical backtest for a strategy."""
        # Load historical data
        data = await self.data_provider.load_data(
            symbols=strategy.symbols,
            start_date=start_date,
            end_date=end_date,
            resolution='1m'
        )
        
        # Initialize portfolio
        portfolio = Portfolio(initial_capital)
        
        # Replay market data
        for timestamp, market_snapshot in data:
            # Generate signals
            signal = await strategy.analyze(market_snapshot)
            
            if signal:
                # Simulate execution
                fill = await self.execution_simulator.simulate_fill(
                    signal=signal,
                    market_data=market_snapshot,
                    slippage_model=self.slippage_model
                )
                
                # Update portfolio
                portfolio.process_fill(fill)
            
            # Update portfolio valuation
            portfolio.mark_to_market(market_snapshot)
            
            # Check risk limits
            if portfolio.drawdown > strategy.max_drawdown:
                break
        
        # Calculate performance metrics
        metrics = self.performance_calculator.calculate(portfolio)
        
        # Generate report
        report = self.report_generator.generate(
            strategy=strategy,
            portfolio=portfolio,
            metrics=metrics
        )
        
        return BacktestResult(
            strategy=strategy,
            metrics=metrics,
            report=report,
            portfolio=portfolio
        )
```

**File Locations:**
- `genesis/backtesting/engine.py` (new file)
- `genesis/backtesting/data_provider.py` (new file)
- `genesis/backtesting/execution_simulator.py` (new file)
- `genesis/backtesting/performance_metrics.py` (new file)
- `tests/backtesting/test_strategies.py` (new test)

## Story 10.7: Live Strategy Performance Monitoring
As a portfolio manager,
I want real-time monitoring of strategy performance with attribution analysis,
So that I can make informed decisions about strategy allocation and risk management.

**Acceptance Criteria:**
1. Real-time P&L tracking per strategy
2. Performance attribution by strategy component
3. Risk metrics dashboard (VaR, CVaR, Beta)
4. Correlation matrix between strategies
5. Drawdown tracking with recovery time
6. Win rate and profit factor calculation
7. Slippage analysis vs expected fills
8. Strategy capacity estimation
9. Alert system for performance degradation
10. Automated strategy rotation based on performance

**Implementation Details:**
```python
# genesis/monitoring/strategy_monitor.py
class StrategyPerformanceMonitor:
    """Real-time strategy performance tracking."""
    
    def __init__(self):
        self.metrics_cache = {}
        self.performance_history = defaultdict(list)
        self.alert_manager = AlertManager()
        
    async def update_metrics(
        self,
        strategy_name: str,
        trade_result: TradeResult
    ) -> None:
        """Update performance metrics for a strategy."""
        metrics = self.metrics_cache.get(strategy_name, StrategyMetrics())
        
        # Update trade statistics
        metrics.total_trades += 1
        metrics.pnl += trade_result.pnl
        
        if trade_result.pnl > 0:
            metrics.winning_trades += 1
            metrics.gross_profit += trade_result.pnl
        else:
            metrics.losing_trades += 1
            metrics.gross_loss += abs(trade_result.pnl)
        
        # Calculate ratios
        metrics.win_rate = metrics.winning_trades / metrics.total_trades
        metrics.profit_factor = (
            metrics.gross_profit / metrics.gross_loss 
            if metrics.gross_loss > 0 else float('inf')
        )
        
        # Update drawdown
        metrics.update_drawdown()
        
        # Check for alerts
        await self._check_performance_alerts(strategy_name, metrics)
        
        # Store metrics
        self.metrics_cache[strategy_name] = metrics
        self.performance_history[strategy_name].append({
            'timestamp': datetime.utcnow(),
            'metrics': metrics.to_dict()
        })
```

**File Locations:**
- `genesis/monitoring/strategy_monitor.py` (new file)
- `genesis/monitoring/performance_attribution.py` (new file)
- `genesis/monitoring/risk_metrics.py` (new file)
- `genesis/ui/widgets/strategy_dashboard.py` (new file)

## Story 10.8: Paper Trading & Strategy Validation
As a risk manager,
I want a paper trading mode that validates strategies with simulated execution,
So that new strategies can be tested in live market conditions without capital risk.

**Acceptance Criteria:**
1. Parallel paper trading alongside live trading
2. Simulated order execution with realistic fills
3. Virtual portfolio tracking
4. Side-by-side performance comparison
5. Automatic promotion to live trading after validation
6. Configurable validation criteria (min trades, min days, min Sharpe)
7. A/B testing framework for strategy variants
8. Gradual capital allocation for validated strategies
9. Performance regression detection
10. Audit trail for strategy promotion decisions

**Implementation Details:**
```python
# genesis/paper_trading/simulator.py
class PaperTradingSimulator:
    """Simulated trading for strategy validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.virtual_portfolio = VirtualPortfolio()
        self.execution_engine = SimulatedExecutor()
        self.validation_criteria = ValidationCriteria(config)
        self.promotion_manager = StrategyPromotionManager()
        
    async def run_paper_trade(
        self,
        strategy: BaseStrategy,
        signal: Signal
    ) -> SimulatedResult:
        """Execute paper trade for validation."""
        # Simulate order execution
        simulated_fill = await self.execution_engine.simulate(
            signal=signal,
            market_data=await self.get_current_market_data(),
            latency_ms=random.gauss(10, 2),
            slippage_bps=random.gauss(2, 1)
        )
        
        # Update virtual portfolio
        self.virtual_portfolio.process_fill(simulated_fill)
        
        # Track performance
        await self.track_performance(strategy, simulated_fill)
        
        # Check promotion criteria
        if await self.validation_criteria.is_met(strategy):
            await self.promotion_manager.promote_to_live(strategy)
        
        return simulated_fill
```

**File Locations:**
- `genesis/paper_trading/simulator.py` (new file)
- `genesis/paper_trading/virtual_portfolio.py` (new file)
- `genesis/paper_trading/promotion_manager.py` (new file)
- `tests/integration/test_paper_trading.py` (update existing)

## Story 10.9: Strategy Configuration & Parameter Management
As a system operator,
I want centralized strategy configuration with hot-reload capability,
So that strategy parameters can be adjusted without system restart.

**Acceptance Criteria:**
1. YAML/JSON strategy configuration files
2. Parameter validation with type checking
3. Hot-reload without trading interruption
4. Configuration versioning with rollback
5. A/B testing parameter sets
6. Environment-specific overrides (dev/staging/prod)
7. Secure storage for sensitive parameters
8. Configuration drift detection
9. Automated parameter optimization
10. Audit logging for all changes

**Implementation Details:**
```python
# genesis/config/strategy_config.py
class StrategyConfigManager:
    """Centralized strategy configuration management."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.configs = {}
        self.version_history = []
        self.file_watcher = FileWatcher()
        
    async def load_strategy_configs(self) -> Dict[str, Any]:
        """Load all strategy configurations."""
        config_files = glob.glob(f"{self.config_path}/*.yaml")
        
        for file_path in config_files:
            strategy_name = Path(file_path).stem
            config = self._load_and_validate(file_path)
            self.configs[strategy_name] = config
            
        # Start watching for changes
        self.file_watcher.watch(
            self.config_path,
            callback=self._on_config_change
        )
        
        return self.configs
    
    async def _on_config_change(self, file_path: str) -> None:
        """Handle configuration file changes."""
        logger.info(f"Config change detected: {file_path}")
        
        # Load new configuration
        new_config = self._load_and_validate(file_path)
        
        # Create version snapshot
        strategy_name = Path(file_path).stem
        self.version_history.append({
            'strategy': strategy_name,
            'timestamp': datetime.utcnow(),
            'old_config': self.configs.get(strategy_name),
            'new_config': new_config
        })
        
        # Hot-reload configuration
        await self._hot_reload_strategy(strategy_name, new_config)
```

**File Locations:**
- `genesis/config/strategy_config.py` (new file)
- `config/strategies/sniper_arbitrage.yaml` (new config)
- `config/strategies/hunter_mean_reversion.yaml` (new config)
- `config/strategies/strategist_vwap.yaml` (new config)

## Story 10.10: Production Deployment Validation
As a DevOps engineer,
I want comprehensive validation that all trading components are properly integrated,
So that the system can be deployed to production with confidence.

**Acceptance Criteria:**
1. End-to-end integration tests for complete trading flow
2. Load testing with 1000+ orders/second
3. Failover testing for all critical components
4. Data integrity validation
5. Performance benchmarks meeting latency requirements
6. Security scanning passing all checks
7. Compliance validation for audit requirements
8. Monitoring and alerting verification
9. Disaster recovery drill successful
10. Production smoke tests passing

**Implementation Details:**
```python
# tests/e2e/test_trading_system.py
class EndToEndTradingTests:
    """Complete system validation tests."""
    
    async def test_complete_trading_flow(self):
        """Test entire trading pipeline from signal to execution."""
        # Initialize system
        system = await self.initialize_test_system()
        
        # Start market data feed
        await system.market_feed.start()
        
        # Load test strategy
        strategy = TestArbitrageStrategy()
        system.strategy_registry.register('test_arb', strategy)
        
        # Start trading engine
        await system.engine.start()
        
        # Wait for signal generation
        signal = await self.wait_for_signal(timeout=30)
        assert signal is not None
        
        # Verify risk validation
        assert await system.risk_engine.validate_signal(signal)
        
        # Verify order execution
        order = await self.wait_for_order(timeout=10)
        assert order.status == OrderStatus.FILLED
        
        # Verify position update
        position = await system.get_position(order.symbol)
        assert position.quantity == order.quantity
        
        # Verify P&L calculation
        await asyncio.sleep(5)
        pnl = await system.calculate_pnl()
        assert pnl is not None
        
        # Test graceful shutdown
        await system.engine.stop()
        assert system.engine.state == EngineState.STOPPED

# scripts/validate_production.py
async def validate_production_readiness():
    """Comprehensive production validation script."""
    validators = [
        StrategyValidator(),
        RiskEngineValidator(),
        ExecutionValidator(),
        DatabaseValidator(),
        MonitoringValidator(),
        SecurityValidator()
    ]
    
    results = []
    for validator in validators:
        result = await validator.validate()
        results.append(result)
        print(f"{validator.name}: {'✅ PASS' if result.passed else '❌ FAIL'}")
        
        if not result.passed:
            print(f"  Failures: {result.failures}")
    
    all_passed = all(r.passed for r in results)
    
    if all_passed:
        print("\n🎉 System is PRODUCTION READY!")
    else:
        print("\n⚠️ System is NOT ready for production")
        print("Run 'make fix-production-issues' to address failures")
    
    return all_passed
```

**File Locations:**
- `tests/e2e/test_trading_system.py` (new file)
- `tests/load/test_performance.py` (new file)
- `scripts/validate_production.py` (new file)
- `tests/integration/test_complete_flow.py` (new file)

## Technical Specifications

### Performance Requirements
- Market analysis latency: <10ms per pair
- Order execution latency: <50ms
- Strategy calculation: <100ms per cycle
- WebSocket message processing: <5ms
- Database query response: <10ms
- Memory usage: <2GB under normal load
- CPU usage: <50% with 10 active strategies

### Risk Constraints
- Maximum position size: 2% of capital (Sniper), 5% (Hunter), 10% (Strategist)
- Maximum daily loss: 5% of capital
- Maximum correlation between positions: 0.7
- Minimum confidence score for signal: 0.6
- Maximum slippage tolerance: 0.1%
- Stop loss mandatory for all positions
- Maximum leverage: 1x (no margin trading initially)

### Data Requirements
- Order book depth: minimum 20 levels
- Trade history: minimum 1000 trades
- Tick data resolution: 100ms
- Historical data: 90 days minimum
- Real-time data latency: <100ms
- Data persistence: 1 year minimum
- Backup frequency: every 5 minutes

### Testing Coverage
- Unit test coverage: >95%
- Integration test coverage: >90%
- Strategy test coverage: 100%
- Risk engine test coverage: 100%
- Backtesting validation: 1000+ trades per strategy
- Paper trading validation: 7 days minimum
- Load testing: 10x normal volume

## Dependencies

### Prerequisites
- Epic 1: Foundation (must be complete)
- Epic 2: Data Pipeline (must be complete)
- Epic 7: Production Deployment (infrastructure ready)
- Epic 9: Security Infrastructure (authentication ready)

### Integration Points
- `genesis/exchange/gateway.py` - Market data connection
- `genesis/risk/risk_engine.py` - Risk validation
- `genesis/engine/state_machine.py` - State management
- `genesis/data/repositories/` - Data persistence
- `genesis/tilt/detector.py` - Behavioral monitoring
- `genesis/ui/dashboard.py` - UI updates

### External Services
- Binance WebSocket API for market data
- PostgreSQL for production database
- Redis for caching and session management
- AWS S3 for backup storage (optional)
- CloudWatch for monitoring (optional)

## Risk Mitigation

### Technical Risks
1. **Strategy Bugs**: Comprehensive backtesting and paper trading
2. **Market Data Gaps**: Multiple data source fallbacks
3. **Execution Failures**: Retry logic with exponential backoff
4. **System Crashes**: State persistence and recovery procedures
5. **Capital Loss**: Strict risk limits and circuit breakers

### Mitigation Strategies
- Gradual rollout starting with minimum position sizes
- Mandatory paper trading before live deployment
- Real-time monitoring with automated alerts
- Daily backup and recovery drills
- Code review by multiple developers
- Automated testing for all strategies

## Success Metrics

### Functional Metrics
- All strategies generating valid signals: ✓
- Orders executing successfully: >99%
- Risk limits enforced: 100%
- P&L tracking accurate: ±0.01%
- System uptime: >99.9%

### Performance Metrics
- Sharpe Ratio: >1.5
- Win Rate: >55%
- Profit Factor: >1.5
- Maximum Drawdown: <10%
- Recovery Time: <30 days

### Operational Metrics
- Mean Time To Recovery: <5 minutes
- Deployment Success Rate: 100%
- Alert Response Time: <1 minute
- Backup Success Rate: 100%
- Security Scan Pass Rate: 100%

## Migration Plan

### Phase 1: Development (Week 1-2)
1. Implement market analyzer (Story 10.1)
2. Implement Sniper strategies (Story 10.2)
3. Connect trading engine (Story 10.5)
4. Basic integration testing

### Phase 2: Validation (Week 3)
1. Implement backtesting (Story 10.6)
2. Validate all strategies show positive expectancy
3. Paper trading deployment (Story 10.8)
4. 7-day paper trading validation

### Phase 3: Production Rollout (Week 4)
1. Deploy to production with minimum capital
2. Monitor for 24 hours
3. Gradually increase position sizes
4. Full deployment after 7 days stable operation

### Rollback Plan
1. Immediate engine stop on critical errors
2. Position liquidation if needed
3. Revert to previous version
4. Post-mortem analysis
5. Fix and re-validate before retry

## Documentation Updates

### Files to Update
- `docs/architecture.md` - Add strategy architecture section
- `docs/prd/requirements.md` - Mark trading requirements complete
- `docs/api/strategies.md` - Document strategy interfaces
- `docs/operations/runbook.md` - Add strategy operation procedures
- `README.md` - Update feature completion status

### New Documentation
- `docs/strategies/README.md` - Strategy development guide
- `docs/backtesting/guide.md` - Backtesting tutorial
- `docs/trading/parameters.md` - Parameter tuning guide
- `docs/monitoring/metrics.md` - Performance metrics guide

## Completion Checklist

- [ ] All 10 stories implemented
- [ ] Unit tests passing (>95% coverage)
- [ ] Integration tests passing
- [ ] Backtesting shows positive expectancy
- [ ] 7-day paper trading successful
- [ ] Performance benchmarks met
- [ ] Security scan passing
- [ ] Documentation updated
- [ ] Code review completed
- [ ] Production validation passing
- [ ] Deployment runbook tested
- [ ] Monitoring alerts configured
- [ ] Team training completed
- [ ] Go-live approval obtained
- [ ] Post-deployment monitoring plan activated

## Post-Implementation Tasks

1. **Continuous Optimization**
   - Weekly strategy parameter tuning
   - Monthly performance review
   - Quarterly strategy additions

2. **Scaling Preparation**
   - Hunter tier strategy development
   - Strategist tier algorithm implementation
   - Multi-exchange integration

3. **Advanced Features**
   - Machine learning signal generation
   - Sentiment analysis integration
   - Social trading features

---

**Epic Status:** 🔴 NOT STARTED
**Priority:** CRITICAL (P0)
**Estimated Effort:** 80-120 hours
**Target Completion:** End of current sprint
**Success Criteria:** Live trading with positive P&L over 30 days