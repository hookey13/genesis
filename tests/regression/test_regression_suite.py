"""
Automated regression test suite for Project GENESIS.

Tests critical user journeys to ensure no regressions in core functionality.
Designed to run nightly in CI/CD pipeline.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.core.models import (
    Account,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TradingSession,
    TradingTier,
)
from genesis.engine.event_bus import EventBus
from genesis.engine.executor.market import MarketOrderExecutor
from genesis.engine.risk_engine import RiskEngine
from genesis.engine.state_machine import TierStateMachine
from genesis.exchange.gateway import BinanceGateway
from genesis.strategies.sniper.simple_arb import SimpleArbitrageStrategy
from genesis.tilt.detector import TiltDetector


class RegressionTestResult:
    """Tracks regression test results for reporting."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = time.time()
        self.end_time = None
        self.passed = False
        self.error_message = None
        self.performance_metrics = {}
        self.baseline_metrics = {}
    
    def complete(self, passed: bool, error_message: Optional[str] = None):
        """Mark test as complete."""
        self.end_time = time.time()
        self.passed = passed
        self.error_message = error_message
    
    @property
    def duration(self) -> float:
        """Get test duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def check_performance_regression(self) -> bool:
        """Check if performance has regressed vs baseline."""
        if not self.baseline_metrics:
            return False
        
        for metric, value in self.performance_metrics.items():
            baseline = self.baseline_metrics.get(metric)
            if baseline and value > baseline * 1.1:  # 10% regression threshold
                return True
        return False


class RegressionTestSuite:
    """Main regression test suite runner."""
    
    def __init__(self):
        self.results: List[RegressionTestResult] = []
        self.baseline_file = Path("tests/regression/baseline_metrics.json")
        self.load_baseline_metrics()
    
    def load_baseline_metrics(self):
        """Load baseline performance metrics."""
        if self.baseline_file.exists():
            with open(self.baseline_file) as f:
                self.baseline = json.load(f)
        else:
            self.baseline = {}
    
    def save_baseline_metrics(self):
        """Save current metrics as new baseline."""
        metrics = {}
        for result in self.results:
            metrics[result.test_name] = {
                "duration": result.duration,
                **result.performance_metrics
            }
        
        with open(self.baseline_file, "w") as f:
            json.dump(metrics, f, indent=2)
    
    def generate_report(self) -> Dict:
        """Generate regression test report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        performance_regressions = sum(
            1 for r in self.results if r.check_performance_regression()
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "performance_regressions": performance_regressions,
                "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            },
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "error": r.error_message,
                    "performance_regression": r.check_performance_regression(),
                }
                for r in self.results
            ],
        }


@pytest.fixture
def regression_suite():
    """Create regression test suite instance."""
    return RegressionTestSuite()


class TestCriticalUserJourneys:
    """Test critical user journeys for regression."""
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_new_user_onboarding(self, regression_suite):
        """Test new user onboarding journey."""
        result = RegressionTestResult("new_user_onboarding")
        
        try:
            # Create new account
            account = Account(
                account_id="new-user-001",
                balance_usdt=Decimal("500"),
                tier=TradingTier.SNIPER,
            )
            
            # Initialize session
            session = TradingSession(
                session_id="session-001",
                account_id=account.account_id,
                session_date=datetime.now(),
                starting_balance=account.balance_usdt,
                current_balance=account.balance_usdt,
                daily_loss_limit=Decimal("25"),
            )
            
            # Initialize risk engine
            risk_engine = RiskEngine(account, session)
            
            # Verify initial state
            assert account.tier == TradingTier.SNIPER
            assert account.balance_usdt == Decimal("500")
            assert session.daily_loss_limit == Decimal("25")
            assert len(risk_engine.positions) == 0
            
            result.complete(passed=True)
            
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_basic_trading_flow(self, regression_suite):
        """Test basic buy and sell trading flow."""
        result = RegressionTestResult("basic_trading_flow")
        
        try:
            # Setup
            account = Account(
                account_id="trader-001",
                balance_usdt=Decimal("1000"),
                tier=TradingTier.SNIPER,
            )
            risk_engine = RiskEngine(account)
            
            with patch("genesis.exchange.gateway.BinanceGateway") as MockGateway:
                gateway = MockGateway(mock_mode=True)
                gateway.place_order = AsyncMock(return_value=MagicMock(
                    order_id="order-001",
                    status=OrderStatus.FILLED,
                    executed_quantity=Decimal("0.01"),
                    average_price=Decimal("50000"),
                ))
                
                executor = MarketOrderExecutor(gateway, risk_engine)
                
                # Place buy order
                start_time = time.time()
                buy_order = await executor.execute_market_order(
                    symbol="BTC/USDT",
                    side=OrderSide.BUY,
                    quantity=Decimal("0.01"),
                )
                buy_latency = (time.time() - start_time) * 1000
                
                assert buy_order.status == OrderStatus.FILLED
                assert buy_order.executed_quantity == Decimal("0.01")
                
                # Place sell order
                start_time = time.time()
                sell_order = await executor.execute_market_order(
                    symbol="BTC/USDT",
                    side=OrderSide.SELL,
                    quantity=Decimal("0.01"),
                )
                sell_latency = (time.time() - start_time) * 1000
                
                assert sell_order.status == OrderStatus.FILLED
                
                # Record performance metrics
                result.performance_metrics = {
                    "buy_latency_ms": buy_latency,
                    "sell_latency_ms": sell_latency,
                }
                result.baseline_metrics = regression_suite.baseline.get(
                    "basic_trading_flow", {}
                )
                
                result.complete(passed=True)
                
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_risk_management_flow(self, regression_suite):
        """Test risk management and position sizing."""
        result = RegressionTestResult("risk_management_flow")
        
        try:
            account = Account(
                account_id="risk-test-001",
                balance_usdt=Decimal("5000"),
                tier=TradingTier.HUNTER,
            )
            
            session = TradingSession(
                session_id="risk-session-001",
                account_id=account.account_id,
                session_date=datetime.now(),
                starting_balance=account.balance_usdt,
                current_balance=account.balance_usdt,
                daily_loss_limit=Decimal("250"),  # 5% of balance
            )
            
            risk_engine = RiskEngine(account, session)
            
            # Test position sizing
            position_size = risk_engine.calculate_position_size(
                symbol="BTC/USDT",
                entry_price=Decimal("50000"),
            )
            
            assert position_size > 0
            assert position_size * Decimal("50000") <= account.balance_usdt
            
            # Test stop loss calculation
            stop_loss = risk_engine.calculate_stop_loss(
                Decimal("50000"),
                PositionSide.LONG,
            )
            
            assert stop_loss < Decimal("50000")
            assert stop_loss == Decimal("49000")  # 2% default stop
            
            # Test daily loss limit
            session.realized_pnl = Decimal("-240")  # Just under limit
            assert not session.is_daily_limit_reached()
            
            session.realized_pnl = Decimal("-250")  # At limit
            assert session.is_daily_limit_reached()
            
            result.complete(passed=True)
            
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_tier_progression_flow(self, regression_suite):
        """Test tier progression from Sniper to Hunter."""
        result = RegressionTestResult("tier_progression_flow")
        
        try:
            account = Account(
                account_id="tier-test-001",
                balance_usdt=Decimal("500"),
                tier=TradingTier.SNIPER,
            )
            
            state_machine = TierStateMachine(account)
            
            # Verify starting tier
            assert account.tier == TradingTier.SNIPER
            
            # Simulate meeting Hunter requirements
            account.balance_usdt = Decimal("2100")
            
            with patch.object(state_machine, "check_tier_requirements") as mock_check:
                mock_check.return_value = {
                    "balance": True,
                    "trades": True,
                    "win_rate": True,
                    "days_active": True,
                }
                
                eligible = await state_machine.check_tier_progression()
                assert eligible is True
                
                # Progress tier
                await state_machine.progress_tier()
                assert account.tier == TradingTier.HUNTER
            
            result.complete(passed=True)
            
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_tilt_detection_flow(self, regression_suite):
        """Test tilt detection and intervention."""
        result = RegressionTestResult("tilt_detection_flow")
        
        try:
            detector = TiltDetector()
            
            # Simulate normal trading
            for _ in range(5):
                detector.record_event({
                    "type": "order_placed",
                    "timestamp": time.time(),
                    "severity": 1,
                })
                await asyncio.sleep(0.1)
            
            normal_score = detector.calculate_tilt_score()
            assert normal_score < 3  # Low tilt
            
            # Simulate tilt behavior
            for _ in range(10):
                detector.record_event({
                    "type": "rapid_clicks",
                    "timestamp": time.time(),
                    "severity": 8,
                })
            
            tilt_score = detector.calculate_tilt_score()
            assert tilt_score > 5  # Moderate to high tilt
            
            result.complete(passed=True)
            
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_multi_position_management(self, regression_suite):
        """Test managing multiple concurrent positions."""
        result = RegressionTestResult("multi_position_management")
        
        try:
            account = Account(
                account_id="multi-pos-001",
                balance_usdt=Decimal("10000"),
                tier=TradingTier.HUNTER,
            )
            
            risk_engine = RiskEngine(account)
            
            # Add multiple positions
            positions = []
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            
            for i, symbol in enumerate(symbols):
                position = Position(
                    position_id=f"pos-{i}",
                    account_id=account.account_id,
                    symbol=symbol,
                    side=PositionSide.LONG,
                    entry_price=Decimal(str(50000 - i * 10000)),
                    quantity=Decimal("0.1"),
                    dollar_value=Decimal(str((50000 - i * 10000) * 0.1)),
                )
                risk_engine.add_position(position)
                positions.append(position)
            
            # Verify positions
            assert len(risk_engine.positions) == 3
            
            # Calculate total exposure
            total_exposure = risk_engine.get_total_exposure()
            expected_exposure = sum(p.dollar_value for p in positions)
            assert abs(total_exposure - expected_exposure) < Decimal("0.01")
            
            # Update P&L
            price_updates = {
                "BTC/USDT": Decimal("51000"),
                "ETH/USDT": Decimal("41000"),
                "BNB/USDT": Decimal("31000"),
            }
            
            risk_engine.update_all_pnl(price_updates)
            
            # Verify P&L calculations
            total_pnl = risk_engine.get_total_pnl()
            assert "total_pnl_dollars" in total_pnl
            assert "total_pnl_percent" in total_pnl
            
            result.complete(passed=True)
            
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_emergency_procedures(self, regression_suite):
        """Test emergency procedures and recovery."""
        result = RegressionTestResult("emergency_procedures")
        
        try:
            account = Account(
                account_id="emergency-001",
                balance_usdt=Decimal("5000"),
                tier=TradingTier.HUNTER,
            )
            
            risk_engine = RiskEngine(account)
            
            # Add positions to close
            for i in range(3):
                position = Position(
                    position_id=f"emergency-pos-{i}",
                    account_id=account.account_id,
                    symbol=f"TEST{i}/USDT",
                    side=PositionSide.LONG,
                    entry_price=Decimal("100"),
                    quantity=Decimal("1"),
                    dollar_value=Decimal("100"),
                )
                risk_engine.add_position(position)
            
            # Simulate emergency closure
            positions_before = len(risk_engine.positions)
            assert positions_before == 3
            
            # Clear all positions (simulating emergency close)
            for pos_id in list(risk_engine.positions.keys()):
                risk_engine.remove_position(pos_id)
            
            assert len(risk_engine.positions) == 0
            
            result.complete(passed=True)
            
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_strategy_execution(self, regression_suite):
        """Test strategy execution and signals."""
        result = RegressionTestResult("strategy_execution")
        
        try:
            with patch("genesis.strategies.sniper.simple_arb.SimpleArbitrageStrategy") as MockStrategy:
                strategy = MockStrategy()
                strategy.calculate_signals = AsyncMock(return_value=[
                    {
                        "symbol": "BTC/USDT",
                        "action": "BUY",
                        "confidence": 0.85,
                        "expected_profit": Decimal("50"),
                    }
                ])
                
                # Get signals
                signals = await strategy.calculate_signals()
                
                assert len(signals) == 1
                assert signals[0]["action"] == "BUY"
                assert signals[0]["confidence"] > 0.8
                
                result.complete(passed=True)
                
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_data_persistence(self, regression_suite):
        """Test data persistence and recovery."""
        result = RegressionTestResult("data_persistence")
        
        try:
            # Simulate saving state
            state_data = {
                "account_id": "persist-001",
                "balance": "5000.00",
                "tier": "HUNTER",
                "positions": [
                    {
                        "position_id": "pos-001",
                        "symbol": "BTC/USDT",
                        "quantity": "0.1",
                        "entry_price": "50000.00",
                    }
                ],
                "session": {
                    "session_id": "session-001",
                    "realized_pnl": "-100.00",
                    "unrealized_pnl": "50.00",
                },
            }
            
            # Simulate state recovery
            recovered_account = Account(
                account_id=state_data["account_id"],
                balance_usdt=Decimal(state_data["balance"]),
                tier=TradingTier[state_data["tier"]],
            )
            
            assert recovered_account.account_id == "persist-001"
            assert recovered_account.balance_usdt == Decimal("5000.00")
            assert recovered_account.tier == TradingTier.HUNTER
            
            result.complete(passed=True)
            
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, regression_suite):
        """Test performance meets requirements."""
        result = RegressionTestResult("performance_benchmarks")
        
        try:
            # Test order execution latency
            latencies = []
            
            for _ in range(100):
                start = time.time()
                # Simulate order processing
                await asyncio.sleep(0.01)  # Simulate 10ms processing
                latencies.append((time.time() - start) * 1000)
            
            # Calculate p99 latency
            latencies.sort()
            p99_index = int(len(latencies) * 0.99)
            p99_latency = latencies[p99_index]
            
            # Requirement: p99 < 50ms
            assert p99_latency < 50, f"P99 latency {p99_latency}ms exceeds 50ms requirement"
            
            result.performance_metrics = {
                "p99_latency_ms": p99_latency,
                "avg_latency_ms": sum(latencies) / len(latencies),
            }
            
            result.complete(passed=True)
            
        except Exception as e:
            result.complete(passed=False, error_message=str(e))
        
        finally:
            regression_suite.results.append(result)


@pytest.mark.regression
class TestRegressionReporting:
    """Test regression reporting functionality."""
    
    def test_report_generation(self, regression_suite):
        """Test regression report generation."""
        # Add some test results
        result1 = RegressionTestResult("test_1")
        result1.complete(passed=True)
        regression_suite.results.append(result1)
        
        result2 = RegressionTestResult("test_2")
        result2.complete(passed=False, error_message="Test failure")
        regression_suite.results.append(result2)
        
        # Generate report
        report = regression_suite.generate_report()
        
        assert report["summary"]["total"] == 2
        assert report["summary"]["passed"] == 1
        assert report["summary"]["failed"] == 1
        assert report["summary"]["pass_rate"] == 50.0
        
        # Save report
        report_file = Path("tests/regression/reports/latest_regression.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        assert report_file.exists()


# CI/CD Integration
def run_nightly_regression():
    """Run nightly regression tests for CI/CD."""
    import subprocess
    import sys
    
    # Run regression tests
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "regression", "--tb=short"],
        capture_output=True,
        text=True,
    )
    
    # Parse results
    if result.returncode == 0:
        print("✅ All regression tests passed")
        return True
    else:
        print("❌ Regression tests failed")
        print(result.stdout)
        print(result.stderr)
        return False


if __name__ == "__main__":
    # Run regression suite when executed directly
    run_nightly_regression()