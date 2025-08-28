"""
Unit tests for Behavioral Correlation Analysis module.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from genesis.analytics.behavioral_correlation import (
    BehavioralCorrelation,
    BehavioralCorrelationAnalyzer,
    InterventionEffect,
)
from genesis.core.models import Trade


class TestBehavioralCorrelation:
    """Test BehavioralCorrelation dataclass."""

    def test_behavioral_correlation_creation(self):
        """Test creating a BehavioralCorrelation instance."""
        correlation = BehavioralCorrelation(
            tilt_score_correlation=Decimal("0.75"),
            performance_after_intervention=Decimal("0.15"),
            average_loss_with_high_tilt=Decimal("-500"),
            average_win_with_low_tilt=Decimal("300"),
            tilt_threshold_for_losses=Decimal("65"),
            improvement_after_journal=Decimal("0.25"),
            recovery_time_hours=Decimal("48"),
            behavioral_patterns=[
                "High tilt correlated with increased losses",
                "Performance improves after meditation",
            ],
            intervention_effectiveness={
                "meditation": Decimal("0.8"),
                "forced_break": Decimal("0.6"),
            },
        )

        assert correlation.tilt_score_correlation == Decimal("0.75")
        assert correlation.performance_after_intervention == Decimal("0.15")
        assert correlation.improvement_after_journal == Decimal("0.25")
        assert len(correlation.behavioral_patterns) == 2
        assert correlation.intervention_effectiveness["meditation"] == Decimal("0.8")

    def test_behavioral_correlation_to_dict(self):
        """Test converting BehavioralCorrelation to dictionary."""
        correlation = BehavioralCorrelation(
            tilt_score_correlation=Decimal("0.5"),
            performance_after_intervention=Decimal("0.2"),
            average_loss_with_high_tilt=Decimal("-100"),
            average_win_with_low_tilt=Decimal("150"),
            tilt_threshold_for_losses=Decimal("70"),
            improvement_after_journal=Decimal("0.3"),
            recovery_time_hours=Decimal("24"),
            behavioral_patterns=["Pattern 1"],
            intervention_effectiveness={"break": Decimal("0.7")},
        )

        result = correlation.to_dict()

        assert result["tilt_score_correlation"] == "0.5"
        assert result["performance_after_intervention"] == "0.2"
        assert result["recovery_time_hours"] == "24"
        assert result["behavioral_patterns"] == ["Pattern 1"]
        assert result["intervention_effectiveness"]["break"] == "0.7"


class TestInterventionEffect:
    """Test InterventionEffect dataclass."""

    def test_intervention_effect_creation(self):
        """Test creating an InterventionEffect instance."""
        effect = InterventionEffect(
            intervention_type="meditation",
            timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            tilt_before=Decimal("75"),
            tilt_after=Decimal("45"),
            performance_change=Decimal("0.3"),
            time_to_recovery_hours=Decimal("2.5"),
            subsequent_win_rate=Decimal("0.65"),
        )

        assert effect.intervention_type == "meditation"
        assert effect.tilt_before == Decimal("75")
        assert effect.tilt_after == Decimal("45")
        assert effect.performance_change == Decimal("0.3")

    def test_intervention_effect_to_dict(self):
        """Test converting InterventionEffect to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        effect = InterventionEffect(
            intervention_type="journal_entry",
            timestamp=timestamp,
            tilt_before=Decimal("80"),
            tilt_after=Decimal("50"),
            performance_change=Decimal("0.25"),
            time_to_recovery_hours=Decimal("3"),
            subsequent_win_rate=Decimal("0.6"),
        )

        result = effect.to_dict()

        assert result["intervention_type"] == "journal_entry"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["tilt_before"] == "80"
        assert result["tilt_after"] == "50"
        assert result["performance_change"] == "0.25"


class TestBehavioralCorrelationAnalyzer:
    """Test BehavioralCorrelationAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a BehavioralCorrelationAnalyzer instance with mocked dependencies."""
        mock_tilt_detector = MagicMock()
        mock_intervention_logger = MagicMock()
        return BehavioralCorrelationAnalyzer(
            tilt_detector=mock_tilt_detector,
            intervention_logger=mock_intervention_logger,
        )

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        trades = []

        for i in range(10):
            trades.append(
                Trade(
                    trade_id=f"trade_{i}",
                    order_id=f"order_{i}",
                    strategy_id="test_strategy",
                    symbol="BTC/USDT",
                    side="BUY",
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("51000") if i % 2 == 0 else Decimal("49000"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=Decimal("100") if i % 2 == 0 else Decimal("-50"),
                    pnl_percent=Decimal("2") if i % 2 == 0 else Decimal("-1"),
                    timestamp=base_time + timedelta(hours=i),
                )
            )

        return trades

    @pytest.fixture
    def sample_tilt_events(self):
        """Create sample tilt events."""
        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        return [
            {
                "timestamp": base_time,
                "tilt_score": 30,
                "indicators": {"revenge_trading": 0.2},
            },
            {
                "timestamp": base_time + timedelta(hours=2),
                "tilt_score": 60,
                "indicators": {"revenge_trading": 0.6},
            },
            {
                "timestamp": base_time + timedelta(hours=4),
                "tilt_score": 80,
                "indicators": {"revenge_trading": 0.8},
            },
            {
                "timestamp": base_time + timedelta(hours=6),
                "tilt_score": 45,
                "indicators": {"revenge_trading": 0.4},
            },
            {
                "timestamp": base_time + timedelta(hours=8),
                "tilt_score": 25,
                "indicators": {"revenge_trading": 0.2},
            },
        ]

    @pytest.fixture
    def sample_interventions(self):
        """Create sample intervention events."""
        base_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
        return [
            {
                "type": "meditation",
                "timestamp": base_time + timedelta(hours=4, minutes=30),
                "duration_minutes": 10,
                "tilt_before": 80,
                "tilt_after": 45,
            },
            {
                "type": "journal_entry",
                "timestamp": base_time + timedelta(hours=7),
                "content": "Reflected on losses",
                "tilt_before": 45,
                "tilt_after": 25,
            },
        ]

    async def test_analyze_correlation_no_data(self, analyzer):
        """Test correlation analysis with no data."""
        analyzer.tilt_detector.get_tilt_events = AsyncMock(return_value=[])
        analyzer.intervention_logger.get_interventions = AsyncMock(return_value=[])

        result = await analyzer.analyze_correlation(
            trades=[],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
        )

        assert result.tilt_score_correlation == Decimal("0")
        assert result.performance_after_intervention == Decimal("0")
        assert len(result.behavioral_patterns) == 0

    async def test_calculate_tilt_correlation(
        self, analyzer, sample_trades, sample_tilt_events
    ):
        """Test tilt score correlation calculation."""
        analyzer.tilt_detector.get_tilt_events = AsyncMock(
            return_value=sample_tilt_events
        )

        correlation = analyzer._calculate_tilt_correlation(
            sample_trades, sample_tilt_events
        )

        # Correlation should be negative (higher tilt = worse performance)
        assert correlation < Decimal("0")
        assert abs(correlation) <= Decimal("1")

    async def test_analyze_intervention_effects(
        self, analyzer, sample_trades, sample_interventions
    ):
        """Test intervention effect analysis."""
        analyzer.intervention_logger.get_interventions = AsyncMock(
            return_value=sample_interventions
        )

        effects = analyzer._analyze_intervention_effects(
            sample_trades, sample_interventions
        )

        assert len(effects) == 2
        assert effects[0].intervention_type == "meditation"
        assert effects[0].tilt_before == Decimal("80")
        assert effects[0].tilt_after == Decimal("45")
        assert effects[1].intervention_type == "journal_entry"

    async def test_identify_behavioral_patterns(self, analyzer, sample_tilt_events):
        """Test behavioral pattern identification."""
        # Create trades with specific patterns
        trades = []

        # High tilt = losses
        for i, event in enumerate(sample_tilt_events):
            pnl = Decimal("-100") if event["tilt_score"] > 60 else Decimal("50")

            trades.append(
                Trade(
                    trade_id=f"trade_{i}",
                    order_id=f"order_{i}",
                    strategy_id="test",
                    symbol="BTC/USDT",
                    side="BUY",
                    entry_price=Decimal("50000"),
                    exit_price=Decimal("51000"),
                    quantity=Decimal("0.1"),
                    pnl_dollars=pnl,
                    pnl_percent=Decimal("1"),
                    timestamp=event["timestamp"],
                )
            )

        patterns = analyzer._identify_behavioral_patterns(
            trades, sample_tilt_events, []
        )

        assert len(patterns) > 0
        assert any("tilt > 70" in p for p in patterns)

    async def test_calculate_performance_metrics(
        self, analyzer, sample_trades, sample_tilt_events
    ):
        """Test performance metrics calculation."""
        metrics = analyzer._calculate_performance_metrics(
            sample_trades, sample_tilt_events
        )

        assert "average_loss_with_high_tilt" in metrics
        assert "average_win_with_low_tilt" in metrics
        assert "tilt_threshold_for_losses" in metrics
        assert metrics["tilt_threshold_for_losses"] >= Decimal("0")
        assert metrics["tilt_threshold_for_losses"] <= Decimal("100")

    async def test_analyze_journal_impact(self, analyzer, sample_trades):
        """Test journal entry impact analysis."""
        journal_entries = [
            {
                "timestamp": datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
                "content": "Reflected on morning losses",
                "mood": "calm",
            }
        ]

        improvement = analyzer._analyze_journal_impact(sample_trades, journal_entries)

        assert improvement >= Decimal("0")
        assert improvement <= Decimal("1")

    async def test_full_correlation_analysis(
        self, analyzer, sample_trades, sample_tilt_events, sample_interventions
    ):
        """Test complete correlation analysis workflow."""
        analyzer.tilt_detector.get_tilt_events = AsyncMock(
            return_value=sample_tilt_events
        )
        analyzer.intervention_logger.get_interventions = AsyncMock(
            return_value=sample_interventions
        )

        result = await analyzer.analyze_correlation(
            trades=sample_trades,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
        )

        assert isinstance(result, BehavioralCorrelation)
        assert result.tilt_score_correlation != Decimal("0")
        assert len(result.behavioral_patterns) > 0
        assert len(result.intervention_effectiveness) > 0

    async def test_edge_case_single_trade(self, analyzer):
        """Test with only one trade."""
        single_trade = [
            Trade(
                trade_id="single",
                order_id="order_1",
                strategy_id="test",
                symbol="BTC/USDT",
                side="BUY",
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                quantity=Decimal("0.1"),
                pnl_dollars=Decimal("100"),
                pnl_percent=Decimal("2"),
                timestamp=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            )
        ]

        tilt_event = [
            {
                "timestamp": datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                "tilt_score": 50,
                "indicators": {},
            }
        ]

        analyzer.tilt_detector.get_tilt_events = AsyncMock(return_value=tilt_event)
        analyzer.intervention_logger.get_interventions = AsyncMock(return_value=[])

        result = await analyzer.analyze_correlation(
            trades=single_trade,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
        )

        # With single data point, correlation should be 0 or 1
        assert abs(result.tilt_score_correlation) <= Decimal("1")

    async def test_no_interventions_scenario(
        self, analyzer, sample_trades, sample_tilt_events
    ):
        """Test scenario with no interventions."""
        analyzer.tilt_detector.get_tilt_events = AsyncMock(
            return_value=sample_tilt_events
        )
        analyzer.intervention_logger.get_interventions = AsyncMock(return_value=[])

        result = await analyzer.analyze_correlation(
            trades=sample_trades,
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 31, tzinfo=UTC),
        )

        assert result.performance_after_intervention == Decimal("0")
        assert result.improvement_after_journal == Decimal("0")
        assert len(result.intervention_effectiveness) == 0
