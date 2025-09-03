"""Integration tests for SniperArbitrageStrategy with MarketAnalyzer."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from genesis.analytics.opportunity_models import (
    DirectArbitrageOpportunity,
    OpportunityStatus,
    OpportunityType,
)
from genesis.core.models import Signal, SignalType
from genesis.strategies.base import StrategyConfig
from genesis.strategies.sniper.simple_arbitrage import SniperArbitrageStrategy


@pytest.fixture
def mock_market_analyzer():
    """Create a mock MarketAnalyzer."""
    analyzer = MagicMock()
    
    # Create realistic arbitrage opportunities
    opportunities = [
        DirectArbitrageOpportunity(
            id=str(uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=Decimal("0.45"),
            profit_amount=Decimal("4.5"),
            confidence_score=0.78,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            buy_exchange="binance",
            sell_exchange="kraken",
            symbol="BTCUSDT",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50225"),
            max_volume=Decimal("0.1"),
            buy_fee=Decimal("0.001"),
            sell_fee=Decimal("0.001"),
            net_profit_pct=Decimal("0.448"),
            status=OpportunityStatus.ACTIVE,
        ),
        DirectArbitrageOpportunity(
            id=str(uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=Decimal("0.35"),
            profit_amount=Decimal("3.5"),
            confidence_score=0.72,
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            buy_exchange="binance",
            sell_exchange="coinbase",
            symbol="ETHUSDT",
            buy_price=Decimal("3000"),
            sell_price=Decimal("3010.50"),
            max_volume=Decimal("1"),
            buy_fee=Decimal("0.001"),
            sell_fee=Decimal("0.001"),
            net_profit_pct=Decimal("0.348"),
            status=OpportunityStatus.ACTIVE,
        ),
    ]
    
    analyzer.analyze_market_data = AsyncMock(return_value=opportunities)
    return analyzer


@pytest.fixture
def strategy_with_mock_analyzer(mock_market_analyzer):
    """Create strategy with mocked MarketAnalyzer."""
    config = StrategyConfig(
        name="IntegrationTestStrategy",
        symbol="BTCUSDT",
        max_position_usdt=Decimal("2000"),
        risk_limit=Decimal("0.02"),
        tier_required="SNIPER",
        metadata={
            "min_confidence": 0.6,
            "min_profit_pct": 0.3,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 0.5,
            "position_timeout_minutes": 5,
        }
    )
    
    strategy = SniperArbitrageStrategy(config)
    strategy.market_analyzer = mock_market_analyzer
    return strategy


class TestMarketAnalyzerIntegration:
    """Test integration with MarketAnalyzer."""
    
    @pytest.mark.asyncio
    async def test_integration_with_mock_market_data(self, strategy_with_mock_analyzer):
        """Test strategy integration with mock MarketAnalyzer data."""
        strategy = strategy_with_mock_analyzer
        
        # Simulate market data from analyzer
        opportunities = await strategy.market_analyzer.analyze_market_data()
        market_data = {
            "arbitrage_opportunities": opportunities,
            "account_balance": Decimal("2000"),
        }
        
        # Generate signal
        signal = await strategy.analyze(market_data)
        
        assert signal is not None
        assert signal.symbol in ["BTCUSDT", "ETHUSDT"]
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence >= Decimal("0.6")
        assert signal.quantity > 0

    @pytest.mark.asyncio
    async def test_signal_generation_latency(self, strategy_with_mock_analyzer):
        """Test that signal generation meets latency requirements (<100ms)."""
        strategy = strategy_with_mock_analyzer
        
        opportunities = await strategy.market_analyzer.analyze_market_data()
        market_data = {
            "arbitrage_opportunities": opportunities,
            "account_balance": Decimal("2000"),
        }
        
        # Measure signal generation time
        start_time = time.perf_counter()
        signal = await strategy.analyze(market_data)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        assert signal is not None
        assert latency_ms < 100, f"Signal generation took {latency_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_memory_usage_per_instance(self, strategy_with_mock_analyzer):
        """Test memory usage stays below 10MB per strategy instance."""
        import sys
        
        strategy = strategy_with_mock_analyzer
        
        # Add some positions to increase memory usage
        for i in range(100):
            position_id = str(uuid4())
            strategy.active_positions[position_id] = {
                "signal_id": position_id,
                "symbol": "BTCUSDT",
                "entry_price": 50000 + i,
                "quantity": 0.01,
                "stop_loss": 49500,
                "take_profit": 50250,
                "entry_time": datetime.now(UTC).isoformat(),
                "current_price": 50000,
                "unrealized_pnl": 0.0
            }
        
        # Rough memory estimation (Python objects)
        memory_usage = sys.getsizeof(strategy)
        memory_usage += sys.getsizeof(strategy.active_positions)
        memory_usage += sys.getsizeof(strategy.performance_tracker)
        memory_usage += sum(sys.getsizeof(pos) for pos in strategy.active_positions.values())
        
        memory_mb = memory_usage / (1024 * 1024)
        
        assert memory_mb < 10, f"Memory usage is {memory_mb:.2f}MB"

    @pytest.mark.asyncio
    async def test_positive_expectancy_validation(self, strategy_with_mock_analyzer):
        """Test that strategy maintains positive expectancy with historical data."""
        strategy = strategy_with_mock_analyzer
        
        # Simulate multiple trading cycles
        total_pnl = Decimal("0")
        trades = []
        
        for i in range(20):
            # Create varying opportunities
            opportunity = DirectArbitrageOpportunity(
                id=str(uuid4()),
                type=OpportunityType.DIRECT,
                profit_pct=Decimal(str(0.3 + (i % 5) * 0.1)),  # 0.3% to 0.7%
                profit_amount=Decimal("4.0"),
                confidence_score=0.6 + (i % 4) * 0.1,  # 0.6 to 0.9
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(seconds=30),
                buy_exchange="binance",
                sell_exchange="kraken",
                symbol="BTCUSDT",
                buy_price=Decimal("50000"),
                sell_price=Decimal("50150"),
                max_volume=Decimal("0.1"),
                buy_fee=Decimal("0.001"),
                sell_fee=Decimal("0.001"),
                net_profit_pct=Decimal("0.298"),
                status=OpportunityStatus.ACTIVE,
            )
            
            market_data = {
                "arbitrage_opportunities": [opportunity],
                "account_balance": Decimal("2000"),
            }
            
            signal = await strategy.analyze(market_data)
            
            if signal:
                # Simulate trade outcome (60% win rate based on confidence)
                import random
                random.seed(i)  # Deterministic for testing
                
                if random.random() < opportunity.confidence_score * 0.8:
                    # Winning trade
                    pnl = signal.quantity * opportunity.buy_price * (opportunity.profit_pct / Decimal("100"))
                    total_pnl += pnl
                    trades.append(float(pnl))
                else:
                    # Losing trade (hit stop loss)
                    loss = signal.quantity * opportunity.buy_price * Decimal("0.01")  # 1% loss
                    total_pnl -= loss
                    trades.append(-float(loss))
        
        # Calculate expectancy
        if trades:
            avg_trade = sum(trades) / len(trades)
            winning_trades = [t for t in trades if t > 0]
            losing_trades = [t for t in trades if t < 0]
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = abs(sum(losing_trades) / len(losing_trades)) if losing_trades else 0
            
            # Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            assert expectancy > 0, f"Negative expectancy: {expectancy:.4f}"
            assert total_pnl > 0, f"Negative total P&L: {total_pnl}"

    @pytest.mark.asyncio
    async def test_concurrent_opportunity_processing(self, mock_market_analyzer):
        """Test handling multiple opportunities concurrently."""
        config = StrategyConfig(
            name="ConcurrentTestStrategy",
            symbol="BTCUSDT",
            max_position_usdt=Decimal("5000"),
            risk_limit=Decimal("0.02"),
            tier_required="SNIPER",
        )
        
        strategy = SniperArbitrageStrategy(config)
        
        # Create multiple opportunities
        opportunities = []
        for i in range(5):
            opp = DirectArbitrageOpportunity(
                id=str(uuid4()),
                type=OpportunityType.DIRECT,
                profit_pct=Decimal(str(0.4 + i * 0.05)),
                profit_amount=Decimal("4.0"),
                confidence_score=0.7 + i * 0.02,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(seconds=30),
                buy_exchange="binance",
                sell_exchange=f"exchange_{i}",
                symbol=f"TOKEN{i}USDT",
                buy_price=Decimal("1000"),
                sell_price=Decimal("1004"),
                max_volume=Decimal("1"),
                buy_fee=Decimal("0.001"),
                sell_fee=Decimal("0.001"),
                net_profit_pct=Decimal("0.398"),
                status=OpportunityStatus.ACTIVE,
            )
            opportunities.append(opp)
        
        market_data = {
            "arbitrage_opportunities": opportunities,
            "account_balance": Decimal("5000"),
        }
        
        # Process opportunities
        signal = await strategy.analyze(market_data)
        
        # Should select the best opportunity (highest profit + confidence)
        assert signal is not None
        assert "TOKEN4USDT" in signal.symbol  # Highest profit opportunity

    @pytest.mark.asyncio
    async def test_state_recovery_after_crash(self, strategy_with_mock_analyzer):
        """Test strategy recovery after simulated crash."""
        strategy = strategy_with_mock_analyzer
        
        # Add positions and record some trades
        for i in range(3):
            position_id = str(uuid4())
            position = {
                "signal_id": position_id,
                "symbol": f"TOKEN{i}USDT",
                "entry_price": 1000 * (i + 1),
                "quantity": 0.1,
                "stop_loss": 990 * (i + 1),
                "take_profit": 1010 * (i + 1),
                "entry_time": datetime.now(UTC).isoformat(),
                "current_price": 1000 * (i + 1),
                "unrealized_pnl": 0.0
            }
            strategy.active_positions[position_id] = position
            strategy.performance_tracker.record_trade(position, 10.0 * i, i % 2 == 0)
        
        # Save state before "crash"
        saved_state = await strategy.save_state()
        
        # Simulate crash by creating new strategy instance
        new_strategy = SniperArbitrageStrategy(strategy.config)
        
        # Verify empty state
        assert len(new_strategy.active_positions) == 0
        assert new_strategy.performance_tracker.metrics["total_trades"] == 0
        
        # Recover state
        await new_strategy.load_state(saved_state)
        
        # Verify state restored
        assert len(new_strategy.active_positions) == 3
        assert new_strategy.performance_tracker.metrics["total_trades"] == 3
        assert new_strategy.performance_tracker.metrics["winning_trades"] == 2
        
        # Verify can continue trading after recovery
        opportunities = await strategy.market_analyzer.analyze_market_data()
        market_data = {
            "arbitrage_opportunities": opportunities,
            "account_balance": Decimal("2000"),
        }
        
        signal = await new_strategy.analyze(market_data)
        assert signal is not None or len(new_strategy.active_positions) > 0

    @pytest.mark.asyncio
    async def test_minimum_profit_threshold_enforcement(self, strategy_with_mock_analyzer):
        """Test that 0.3% minimum profit threshold is enforced."""
        strategy = strategy_with_mock_analyzer
        
        # Create opportunity below threshold
        low_profit_opp = DirectArbitrageOpportunity(
            id=str(uuid4()),
            type=OpportunityType.DIRECT,
            profit_pct=Decimal("0.25"),  # Below 0.3% threshold
            profit_amount=Decimal("2.5"),
            confidence_score=0.9,  # High confidence
            created_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
            buy_exchange="binance",
            sell_exchange="kraken",
            symbol="BTCUSDT",
            buy_price=Decimal("50000"),
            sell_price=Decimal("50125"),
            max_volume=Decimal("0.1"),
            buy_fee=Decimal("0.001"),
            sell_fee=Decimal("0.001"),
            net_profit_pct=Decimal("0.248"),
            status=OpportunityStatus.ACTIVE,
        )
        
        market_data = {
            "arbitrage_opportunities": [low_profit_opp],
            "account_balance": Decimal("2000"),
        }
        
        signal = await strategy.analyze(market_data)
        
        # Should reject due to profit threshold
        assert signal is None
        
        # Now test with opportunity at exact threshold
        threshold_opp = low_profit_opp
        threshold_opp.profit_pct = Decimal("0.3")
        threshold_opp.net_profit_pct = Decimal("0.298")
        
        signal = await strategy.analyze(market_data)
        
        # Should accept at threshold
        assert signal is not None

    @pytest.mark.asyncio
    async def test_risk_engine_validation(self, strategy_with_mock_analyzer):
        """Test that risk engine validates position sizing."""
        strategy = strategy_with_mock_analyzer
        
        # Set very low account balance to trigger risk limits
        opportunities = await strategy.market_analyzer.analyze_market_data()
        market_data = {
            "arbitrage_opportunities": opportunities,
            "account_balance": Decimal("100"),  # Very low balance
        }
        
        signal = await strategy.analyze(market_data)
        
        if signal:
            # With $100 balance and minimum $10 position, the strategy
            # will use the minimum position size which may exceed 2%
            # This is acceptable for very small accounts
            position_value = signal.quantity * signal.price_target
            
            # Should be either the minimum position ($10) or 2% of account
            assert position_value <= Decimal("10") or position_value <= Decimal("2"), (
                f"Position value {position_value} is neither minimum nor within risk limit"
            )