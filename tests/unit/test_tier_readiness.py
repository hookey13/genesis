"""Unit tests for tier readiness assessment."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from genesis.core.exceptions import ValidationError
from genesis.tilt.tier_readiness_assessor import (
    READINESS_REQUIREMENTS,
    ReadinessReport,
    TierReadinessAssessor,
)


class TestReadinessReport:
    """Test ReadinessReport dataclass."""

    def test_to_dict(self):
        """Test converting report to dictionary."""
        report = ReadinessReport(
            account_id='test-account',
            current_tier='SNIPER',
            target_tier='HUNTER',
            readiness_score=85,
            is_ready=True,
            assessment_timestamp=datetime.utcnow(),
            behavioral_stability_score=90,
            profitability_score=80,
            consistency_score=85,
            risk_management_score=88,
            experience_score=82,
            days_at_current_tier=45,
            current_tilt_score=20,
            recent_tilt_events=1,
            profitability_ratio=Decimal('0.62'),
            risk_adjusted_return=Decimal('1.25'),
            max_drawdown=Decimal('500'),
            trade_count=150,
            failure_reasons=[],
            recommendations=['Continue current performance']
        )

        result = report.to_dict()

        assert result['account_id'] == 'test-account'
        assert result['readiness_score'] == 85
        assert result['is_ready'] is True
        assert result['component_scores']['behavioral_stability'] == 90
        assert result['metrics']['trade_count'] == 150
        assert len(result['recommendations']) == 1


class TestTierReadinessAssessor:
    """Test TierReadinessAssessor class."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def assessor(self, mock_session):
        """Create TierReadinessAssessor instance."""
        return TierReadinessAssessor(session=mock_session)

    @pytest.mark.asyncio
    async def test_assess_readiness_profile_not_found(self, assessor, mock_session):
        """Test assessment when profile not found."""
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        with pytest.raises(ValidationError, match="Profile not found"):
            await assessor.assess_readiness('invalid-profile', 'HUNTER')

    @pytest.mark.asyncio
    async def test_assess_readiness_successful(self, assessor, mock_session):
        """Test successful readiness assessment."""
        # Mock profile
        mock_profile = MagicMock()
        mock_profile.profile_id = 'test-profile'
        mock_profile.account_id = 'test-account'
        mock_profile.current_tilt_score = 25

        # Mock account
        mock_account = MagicMock()
        mock_account.account_id = 'test-account'
        mock_account.current_tier = 'SNIPER'
        mock_account.balance_usdt = Decimal('1900')
        mock_account.tier_started_at = datetime.utcnow() - timedelta(days=45)
        mock_account.created_at = datetime.utcnow() - timedelta(days=60)

        # Setup query mocks
        def query_side_effect(model):
            mock_query = MagicMock()
            if model.__name__ == 'TiltProfile':
                mock_query.filter_by.return_value.first.return_value = mock_profile
            elif model.__name__ == 'Account':
                mock_query.filter_by.return_value.first.return_value = mock_account
            elif model.__name__ == 'TiltEvent':
                mock_query.filter.return_value.all.return_value = []
                mock_query.filter.return_value.count.return_value = 1
            elif model.__name__ == 'Trade':
                # Mock trades for various queries
                mock_trades = [MagicMock(pnl_usdt=Decimal('10'), closed_at=datetime.utcnow()) for _ in range(20)]
                mock_query.filter.return_value.all.return_value = mock_trades
                mock_query.filter.return_value.count.return_value = 100
            elif model.__name__ == 'TierTransition':
                mock_query.filter_by.return_value.first.return_value = None
            return mock_query

        mock_session.query.side_effect = query_side_effect

        # Assess readiness
        report = await assessor.assess_readiness('test-profile', 'HUNTER')

        assert report.account_id == 'test-account'
        assert report.current_tier == 'SNIPER'
        assert report.target_tier == 'HUNTER'
        assert report.readiness_score >= 0
        assert report.readiness_score <= 100
        assert isinstance(report.is_ready, bool)

    @pytest.mark.asyncio
    async def test_assess_behavioral_stability_high_score(self, assessor, mock_session):
        """Test behavioral stability assessment with good behavior."""
        mock_profile = MagicMock()
        mock_profile.profile_id = 'test-profile'
        mock_profile.current_tilt_score = 10
        mock_profile.recovery_required = False
        mock_profile.journal_entries_required = 0

        # No recent tilt events
        mock_session.query.return_value.filter.return_value.all.return_value = []

        score = await assessor._assess_behavioral_stability(mock_profile)

        assert score == 90  # 100 - 10 (tilt score)

    @pytest.mark.asyncio
    async def test_assess_behavioral_stability_with_tilt_events(self, assessor, mock_session):
        """Test behavioral stability with recent tilt events."""
        mock_profile = MagicMock()
        mock_profile.profile_id = 'test-profile'
        mock_profile.current_tilt_score = 20
        mock_profile.recovery_required = False

        # Mock tilt events
        mock_events = [
            MagicMock(severity='HIGH'),
            MagicMock(severity='MEDIUM'),
            MagicMock(severity='LOW')
        ]
        mock_session.query.return_value.filter.return_value.all.return_value = mock_events

        score = await assessor._assess_behavioral_stability(mock_profile)

        # 100 - 20 (tilt score) - 15 (HIGH) - 10 (MEDIUM) - 5 (LOW) = 50
        assert score == 50

    @pytest.mark.asyncio
    async def test_assess_profitability_insufficient_trades(self, assessor, mock_session):
        """Test profitability assessment with insufficient trades."""
        mock_account = MagicMock()
        mock_account.account_id = 'test-account'

        # Less than 10 trades
        mock_session.query.return_value.filter.return_value.all.return_value = [
            MagicMock(pnl_usdt=Decimal('10')) for _ in range(5)
        ]

        score = await assessor._assess_profitability(mock_account)

        assert score == 0  # Not enough data

    @pytest.mark.asyncio
    async def test_assess_profitability_good_performance(self, assessor, mock_session):
        """Test profitability assessment with good performance."""
        mock_account = MagicMock()
        mock_account.account_id = 'test-account'

        # 70% profitable trades
        trades = []
        for i in range(10):
            trade = MagicMock()
            trade.pnl_usdt = Decimal('10') if i < 7 else Decimal('-5')
            trade.closed_at = datetime.utcnow()
            trades.append(trade)

        mock_session.query.return_value.filter.return_value.all.return_value = trades

        score = await assessor._assess_profitability(mock_account)

        assert score >= 70  # At least 70% profitability

    @pytest.mark.asyncio
    async def test_assess_consistency_high_variation(self, assessor, mock_session):
        """Test consistency assessment with high P&L variation."""
        mock_account = MagicMock()
        mock_account.account_id = 'test-account'

        # High variation trades
        trades = []
        for i in range(20):
            trade = MagicMock()
            # Alternating high gains and losses
            trade.pnl_usdt = Decimal('100') if i % 2 == 0 else Decimal('-80')
            trade.closed_at = datetime.utcnow() - timedelta(days=i)
            trades.append(trade)

        mock_session.query.return_value.filter.return_value.all.return_value = trades

        score = await assessor._assess_consistency(mock_account)

        assert score <= 60  # High variation should give low score

    @pytest.mark.asyncio
    async def test_assess_risk_management_good_sharpe(self, assessor, mock_session):
        """Test risk management assessment with good Sharpe ratio."""
        mock_account = MagicMock()
        mock_account.account_id = 'test-account'
        mock_account.balance_usdt = Decimal('2000')

        # Consistent small gains
        trades = []
        for i in range(20):
            trade = MagicMock()
            trade.pnl_usdt = Decimal('10')  # Consistent gains
            trade.closed_at = datetime.utcnow() - timedelta(hours=i)
            trades.append(trade)

        mock_session.query.return_value.filter.return_value.all.return_value = trades

        score = await assessor._assess_risk_management(mock_account)

        assert score >= 80  # Good risk management

    @pytest.mark.asyncio
    async def test_assess_experience_high_score(self, assessor, mock_session):
        """Test experience assessment with extensive history."""
        mock_account = MagicMock()
        mock_account.account_id = 'test-account'
        mock_account.tier_started_at = datetime.utcnow() - timedelta(days=100)
        mock_account.created_at = datetime.utcnow() - timedelta(days=200)

        # Many trades
        mock_session.query.return_value.filter.return_value.count.return_value = 300

        score = await assessor._assess_experience(mock_account)

        assert score >= 60  # Good experience

    def test_calculate_readiness_score(self, assessor):
        """Test overall readiness score calculation."""
        score = assessor._calculate_readiness_score(
            behavioral=80,
            profitability=70,
            consistency=75,
            risk=85,
            experience=60
        )

        # Weighted average
        expected = int(
            80 * 0.30 +  # behavioral
            70 * 0.20 +  # profitability
            75 * 0.20 +  # consistency
            85 * 0.20 +  # risk
            60 * 0.10    # experience
        )

        assert score == expected

    def test_check_requirements_all_pass(self, assessor):
        """Test requirement checking when all pass."""
        metrics = {
            'days_at_tier': 35,
            'tilt_score': 25,
            'profitability_ratio': Decimal('0.60'),
            'recent_tilt_events': 1,
            'trade_count': 100,
            'risk_adjusted_return': Decimal('1.0'),
            'max_drawdown': Decimal('100')
        }

        requirements = READINESS_REQUIREMENTS['HUNTER']

        is_ready, failures = assessor._check_requirements(
            metrics, requirements, readiness_score=85
        )

        assert is_ready is True
        assert len(failures) == 0

    def test_check_requirements_multiple_failures(self, assessor):
        """Test requirement checking with multiple failures."""
        metrics = {
            'days_at_tier': 10,  # Too few days
            'tilt_score': 40,  # Too high
            'profitability_ratio': Decimal('0.40'),  # Too low
            'recent_tilt_events': 5,  # Too many
            'trade_count': 20,  # Too few trades
            'risk_adjusted_return': Decimal('0.5'),
            'max_drawdown': Decimal('500')
        }

        requirements = READINESS_REQUIREMENTS['HUNTER']

        is_ready, failures = assessor._check_requirements(
            metrics, requirements, readiness_score=70
        )

        assert is_ready is False
        assert len(failures) >= 5

    def test_generate_recommendations_for_high_tilt(self, assessor):
        """Test recommendation generation for high tilt score."""
        metrics = {
            'tilt_score': 40,
            'profitability_ratio': Decimal('0.55'),
            'recent_tilt_events': 3,
            'risk_adjusted_return': Decimal('0.8'),
            'max_drawdown': Decimal('500')
        }

        recommendations = assessor._generate_recommendations(
            metrics, {}, []
        )

        assert any('emotional regulation' in r for r in recommendations)
        assert any('tilt triggers' in r for r in recommendations)

    def test_generate_recommendations_ready(self, assessor):
        """Test recommendations when ready for transition."""
        metrics = {
            'tilt_score': 20,
            'profitability_ratio': Decimal('0.65'),
            'recent_tilt_events': 0,
            'risk_adjusted_return': Decimal('1.5'),
            'max_drawdown': Decimal('200')
        }

        recommendations = assessor._generate_recommendations(
            metrics, {}, []  # No failures
        )

        assert any('ready for tier transition' in r for r in recommendations)

    @pytest.mark.asyncio
    async def test_store_assessment_update_existing(self, assessor, mock_session):
        """Test storing assessment with existing transition."""
        report = ReadinessReport(
            account_id='test-account',
            current_tier='SNIPER',
            target_tier='HUNTER',
            readiness_score=85,
            is_ready=True,
            assessment_timestamp=datetime.utcnow(),
            behavioral_stability_score=90,
            profitability_score=80,
            consistency_score=85,
            risk_management_score=88,
            experience_score=82,
            days_at_current_tier=45,
            current_tilt_score=20,
            recent_tilt_events=1,
            profitability_ratio=Decimal('0.62'),
            risk_adjusted_return=Decimal('1.25'),
            max_drawdown=Decimal('500'),
            trade_count=150
        )

        # Mock existing transition
        mock_transition = MagicMock()
        mock_transition.transition_id = 'test-transition'
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_transition

        await assessor._store_assessment(report)

        assert mock_transition.readiness_score == 85
        assert mock_transition.transition_status == 'READY'
        assert mock_session.commit.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
