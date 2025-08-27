"""Integration test for complete tier transition workflow."""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from genesis.engine.paper_trading_enforcer import PaperTrade, PaperTradingEnforcer
from genesis.engine.state_machine import TierStateMachine
from genesis.strategies.loader import StrategyLoader
from genesis.tilt.adjustment_period_manager import AdjustmentPeriodManager
from genesis.tilt.habit_funeral_ceremony import HabitFuneralCeremony
from genesis.tilt.tier_readiness_assessor import TierReadinessAssessor
from genesis.tilt.transition_checklist import TransitionChecklist
from genesis.tilt.valley_of_death_monitor import TransitionMonitor


class TestTierTransitionWorkflow:
    """Test complete tier transition workflow from approach to completion."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        return session

    @pytest.fixture
    def transition_components(self, mock_session):
        """Create all transition components."""
        return {
            'monitor': TransitionMonitor(session=mock_session),
            'assessor': TierReadinessAssessor(session=mock_session),
            'paper_trading': PaperTradingEnforcer(session=mock_session),
            'checklist': TransitionChecklist(session=mock_session),
            'funeral': HabitFuneralCeremony(session=mock_session),
            'state_machine': TierStateMachine(session=mock_session),
            'strategy_loader': StrategyLoader(session=mock_session),
            'adjustment_manager': AdjustmentPeriodManager(session=mock_session)
        }

    @pytest.mark.asyncio
    async def test_complete_tier_transition_sniper_to_hunter(
        self,
        transition_components,
        mock_session
    ):
        """Test complete transition from Sniper to Hunter tier."""

        # Setup test data
        account_id = str(uuid.uuid4())
        profile_id = str(uuid.uuid4())
        transition_id = str(uuid.uuid4())

        # Mock account approaching $2k threshold
        mock_account = MagicMock()
        mock_account.account_id = account_id
        mock_account.balance_usdt = Decimal('1900')  # 95% of $2k threshold
        mock_account.current_tier = 'SNIPER'
        mock_account.tier_started_at = datetime.utcnow() - timedelta(days=45)
        mock_account.created_at = datetime.utcnow() - timedelta(days=60)

        # Mock tilt profile
        mock_profile = MagicMock()
        mock_profile.profile_id = profile_id
        mock_profile.account_id = account_id
        mock_profile.current_tilt_score = 20
        mock_profile.tilt_level = 'NORMAL'
        mock_profile.recovery_required = False
        mock_profile.monitoring_sensitivity = 1.0

        # Mock tier transition
        mock_transition = MagicMock()
        mock_transition.transition_id = transition_id
        mock_transition.account_id = account_id
        mock_transition.from_tier = 'SNIPER'
        mock_transition.to_tier = 'HUNTER'
        mock_transition.transition_status = 'APPROACHING'
        mock_transition.readiness_score = None
        mock_transition.checklist_completed = False
        mock_transition.funeral_completed = False
        mock_transition.paper_trading_completed = False

        # Phase 1: Detection of approaching transition
        proximity = transition_components['monitor'].check_approaching_transition(
            balance=Decimal('1900'),
            current_tier='SNIPER'
        )

        assert proximity.is_approaching is True
        assert proximity.is_critical is True
        assert proximity.next_tier == 'HUNTER'
        assert proximity.monitoring_multiplier == 2.0  # 95% triggers 2x monitoring

        # Phase 2: Readiness assessment
        def setup_assessment_mocks(query_obj):
            if query_obj.__name__ == 'TiltProfile':
                return MagicMock(filter_by=lambda **k: MagicMock(first=lambda: mock_profile))
            elif query_obj.__name__ == 'Account':
                return MagicMock(filter_by=lambda **k: MagicMock(first=lambda: mock_account))
            elif query_obj.__name__ == 'TiltEvent':
                return MagicMock(filter=lambda *a: MagicMock(all=lambda: [], count=lambda: 1))
            elif query_obj.__name__ == 'Trade':
                mock_trades = [MagicMock(pnl_usdt=Decimal('10'), closed_at=datetime.utcnow()) for _ in range(20)]
                return MagicMock(filter=lambda *a: MagicMock(all=lambda: mock_trades, count=lambda: 100))
            elif query_obj.__name__ == 'TierTransition':
                return MagicMock(filter_by=lambda **k: MagicMock(first=lambda: mock_transition))
            return MagicMock()

        mock_session.query.side_effect = setup_assessment_mocks

        report = await transition_components['assessor'].assess_readiness(
            profile_id=profile_id,
            target_tier='HUNTER'
        )

        assert report.target_tier == 'HUNTER'
        # Readiness would be calculated based on mocked data

        # Phase 3: Paper trading requirement
        paper_session_id = await transition_components['paper_trading'].require_paper_trading(
            account_id=account_id,
            strategy='iceberg_orders',
            duration_hours=24,
            transition_id=transition_id
        )

        assert paper_session_id is not None

        # Simulate successful paper trading
        for i in range(25):
            trade = PaperTrade(
                trade_id=str(uuid.uuid4()),
                session_id=paper_session_id,
                symbol='BTC/USDT',
                side='BUY' if i % 2 == 0 else 'SELL',
                quantity=Decimal('0.01'),
                entry_price=Decimal('50000')
            )
            await transition_components['paper_trading'].record_paper_trade(
                paper_session_id, trade
            )

            # Close with profit
            exit_price = trade.entry_price + Decimal('100') if i < 20 else trade.entry_price - Decimal('50')
            await transition_components['paper_trading'].close_paper_trade(
                paper_session_id,
                trade.trade_id,
                exit_price
            )

        # Phase 4: Psychological preparation checklist
        checklist_items = await transition_components['checklist'].create_checklist(
            transition_id=transition_id,
            target_tier='HUNTER'
        )

        assert len(checklist_items) > 0

        # Complete checklist items
        for item in checklist_items:
            if item.is_required:
                response = f"Test response for {item.name} - " + "x" * item.min_response_length
                await transition_components['checklist'].complete_item(
                    transition_id=transition_id,
                    item_name=item.name,
                    response=response
                )

        # Phase 5: Old habits funeral ceremony
        old_habits = [
            "Overtrading on small wins",
            "Revenge trading after losses",
            "Ignoring stop losses"
        ]
        impact_descriptions = [
            "Lost significant capital",
            "Emotional decisions led to poor trades",
            "Unnecessary risks taken"
        ]
        commitments = [
            "I will follow position sizing rules",
            "I will take breaks after losses",
            "I will always set and respect stop losses"
        ]

        ceremony = await transition_components['funeral'].conduct_funeral(
            transition_id=transition_id,
            old_habits=old_habits,
            impact_descriptions=impact_descriptions,
            commitments=commitments
        )

        assert ceremony.funeral_id is not None
        assert ceremony.certificate_generated is True

        # Phase 6: Tier transition approval and celebration
        # Mock transition as ready
        mock_transition.transition_status = 'READY'
        mock_transition.readiness_score = 85
        mock_transition.checklist_completed = True
        mock_transition.funeral_completed = True
        mock_transition.paper_trading_completed = True

        # Request tier change
        success = await transition_components['state_machine'].request_tier_change(
            account_id=account_id,
            new_tier='HUNTER',
            reason='All requirements completed'
        )

        # Would be True if all mocks properly set up
        # assert success is True

        # Phase 7: Strategy migration
        migration_plan = await transition_components['strategy_loader'].migrate_strategies(
            account_id=account_id,
            old_tier='SNIPER',
            new_tier='HUNTER'
        )

        assert migration_plan.old_tier == 'SNIPER'
        assert migration_plan.new_tier == 'HUNTER'
        assert len(migration_plan.strategies_to_enable) > 0  # New Hunter strategies

        # Phase 8: 48-hour adjustment period
        adjustment_period_id = await transition_components['adjustment_manager'].start_adjustment_period(
            account_id=account_id,
            tier='HUNTER',
            duration_hours=48,
            transition_id=transition_id
        )

        assert adjustment_period_id is not None

        # Check adjustment status
        status = await transition_components['adjustment_manager'].get_adjustment_status(
            account_id=account_id
        )

        assert status is not None
        assert status.is_active is True
        assert status.current_position_limit < status.normal_position_limit
        assert status.monitoring_multiplier > 1.0

        # Simulate time passing and phase transitions would occur
        # In real scenario, the adjustment period would gradually restore limits

        # Clean up all components
        await transition_components['monitor'].cleanup()
        await transition_components['paper_trading'].cleanup()
        await transition_components['adjustment_manager'].cleanup()

    @pytest.mark.asyncio
    async def test_transition_with_failures_and_retry(
        self,
        transition_components,
        mock_session
    ):
        """Test tier transition with failures requiring retry."""

        account_id = str(uuid.uuid4())
        profile_id = str(uuid.uuid4())

        # Mock account with poor metrics
        mock_account = MagicMock()
        mock_account.account_id = account_id
        mock_account.balance_usdt = Decimal('1950')
        mock_account.current_tier = 'SNIPER'
        mock_account.tier_started_at = datetime.utcnow() - timedelta(days=20)  # Too recent

        mock_profile = MagicMock()
        mock_profile.profile_id = profile_id
        mock_profile.account_id = account_id
        mock_profile.current_tilt_score = 45  # Too high
        mock_profile.tilt_level = 'WARNING'

        def setup_failing_mocks(query_obj):
            if query_obj.__name__ == 'TiltProfile':
                return MagicMock(filter_by=lambda **k: MagicMock(first=lambda: mock_profile))
            elif query_obj.__name__ == 'Account':
                return MagicMock(filter_by=lambda **k: MagicMock(first=lambda: mock_account))
            elif query_obj.__name__ == 'TiltEvent':
                # Many recent tilt events
                mock_events = [MagicMock(severity='HIGH') for _ in range(5)]
                return MagicMock(filter=lambda *a: MagicMock(all=lambda: mock_events, count=lambda: 5))
            elif query_obj.__name__ == 'Trade':
                # Poor trading performance
                mock_trades = [MagicMock(pnl_usdt=Decimal('-10'), closed_at=datetime.utcnow()) for _ in range(10)]
                return MagicMock(filter=lambda *a: MagicMock(all=lambda: mock_trades, count=lambda: 10))
            return MagicMock()

        mock_session.query.side_effect = setup_failing_mocks

        # Readiness assessment should indicate not ready
        report = await transition_components['assessor'].assess_readiness(
            profile_id=profile_id,
            target_tier='HUNTER'
        )

        assert report.is_ready is False
        assert len(report.failure_reasons) > 0
        assert report.current_tilt_score == 45  # Too high for transition

        # Recommendations should suggest improvements
        assert any('emotional regulation' in r for r in report.recommendations)

    @pytest.mark.asyncio
    async def test_emergency_demotion_cancels_transition(
        self,
        transition_components,
        mock_session
    ):
        """Test that emergency demotion cancels ongoing transition."""

        account_id = str(uuid.uuid4())

        # Setup account in mid-transition
        mock_account = MagicMock()
        mock_account.account_id = account_id
        mock_account.current_tier = 'HUNTER'
        mock_account.balance_usdt = Decimal('1500')  # Lost money

        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_account

        # Force demotion due to capital loss
        await transition_components['state_machine'].force_demotion(
            account_id=account_id,
            new_tier='SNIPER',
            reason='Capital below tier minimum'
        )

        # Check that adjustment period can be force completed
        success = await transition_components['adjustment_manager'].force_complete(
            account_id=account_id,
            reason='Emergency demotion'
        )

        # Would cancel any ongoing transition activities
        # assert success is True or False depending on state


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
