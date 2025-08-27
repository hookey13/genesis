"""Unit tests for tier gate progress widgets."""

from unittest.mock import Mock

import pytest

from genesis.engine.state_machine import TierStateMachine
from genesis.ui.widgets.tier_gate_progress import (
    FeatureTutorial,
    GateProgressBar,
    TierCeremonyAnimation,
    TierGateProgressWidget,
    TierTransitionHistory,
)


class TestGateProgressBar:
    """Test suite for GateProgressBar widget."""

    def test_initialization(self):
        """Test progress bar initialization."""
        bar = GateProgressBar(
            gate_name="Minimum Balance",
            current_value=1500,
            required_value=2000,
            unit="USDT"
        )

        assert bar.gate_name == "Minimum Balance"
        assert bar.current_value == 1500
        assert bar.required_value == 2000
        assert bar.unit == "USDT"
        assert bar.progress == 75  # 1500/2000 * 100

    def test_progress_calculation(self):
        """Test progress calculation."""
        bar = GateProgressBar("Test", 50, 100)
        assert bar.progress == 50

        bar = GateProgressBar("Test", 150, 100)
        assert bar.progress == 100  # Capped at 100

        bar = GateProgressBar("Test", 0, 0)
        assert bar.progress == 0  # Handle division by zero

    def test_update_progress(self):
        """Test updating progress value."""
        bar = GateProgressBar("Test", 50, 100)
        assert bar.progress == 50

        bar.update_progress(75)
        assert bar.current_value == 75
        assert bar.progress == 75

        bar.update_progress(200)
        assert bar.current_value == 200
        assert bar.progress == 100  # Capped


class TestTierCeremonyAnimation:
    """Test suite for TierCeremonyAnimation widget."""

    def test_initialization(self):
        """Test ceremony animation initialization."""
        animation = TierCeremonyAnimation(
            from_tier="SNIPER",
            to_tier="HUNTER"
        )

        assert animation.from_tier == "SNIPER"
        assert animation.to_tier == "HUNTER"
        assert animation.animation_frame == 0
        assert animation.timer is None

    def test_animate(self):
        """Test animation frame progression."""
        animation = TierCeremonyAnimation("SNIPER", "HUNTER")

        # Simulate animation frames
        for _ in range(5):
            animation.animate()

        assert animation.animation_frame == 5

    def test_animation_completion(self):
        """Test animation stops after completion."""
        animation = TierCeremonyAnimation("SNIPER", "HUNTER")
        animation.timer = Mock()

        # Animate past completion threshold
        animation.animation_frame = 30
        animation.animate()

        animation.timer.stop.assert_called_once()


class TestFeatureTutorial:
    """Test suite for FeatureTutorial widget."""

    def test_initialization(self):
        """Test tutorial initialization."""
        tutorial = FeatureTutorial(
            feature_name="Iceberg Orders",
            description="Split large orders into smaller chunks."
        )

        assert tutorial.feature_name == "Iceberg Orders"
        assert tutorial.description == "Split large orders into smaller chunks."


class TestTierGateProgressWidget:
    """Test suite for TierGateProgressWidget."""

    @pytest.fixture
    def mock_state_machine(self):
        """Create mock state machine."""
        machine = Mock(spec=TierStateMachine)
        machine.get_tier_requirements.return_value = {
            'min_balance': 2000,
            'min_trades': 50,
            'max_tilt_events': 2,
            'paper_trading_required': True
        }
        machine.get_next_tier.return_value = "HUNTER"
        machine.get_available_features.return_value = [
            "iceberg_orders",
            "multi_pair_trading"
        ]
        return machine

    def test_initialization(self, mock_state_machine):
        """Test widget initialization."""
        widget = TierGateProgressWidget(mock_state_machine)

        assert widget.state_machine == mock_state_machine
        assert widget.current_tier == "SNIPER"
        assert widget.next_tier == "HUNTER"
        assert widget.gates_completed == 0
        assert widget.total_gates == 4
        assert len(widget.gate_bars) == 0

    @pytest.mark.asyncio
    async def test_update_progress(self, mock_state_machine):
        """Test updating gate progress."""
        widget = TierGateProgressWidget(mock_state_machine)

        # Mock the container query
        mock_container = Mock()
        widget.query_one = Mock(return_value=mock_container)

        await widget.update_progress()

        # Should query requirements
        mock_state_machine.get_tier_requirements.assert_called_with("HUNTER")

    @pytest.mark.asyncio
    async def test_trigger_ceremony(self, mock_state_machine):
        """Test triggering tier ceremony."""
        widget = TierGateProgressWidget(mock_state_machine)
        widget.mount = Mock()

        # Add some progress bars to clear
        widget.gate_bars = {
            "balance": Mock(),
            "trades": Mock()
        }

        await widget.trigger_ceremony("SNIPER", "HUNTER")

        # Should update tiers
        assert widget.current_tier == "HUNTER"

        # Should clear old bars
        assert len(widget.gate_bars) == 0

        # Should mount ceremony animation
        assert widget.mount.called

    @pytest.mark.asyncio
    async def test_show_feature_tutorials(self, mock_state_machine):
        """Test showing feature tutorials."""
        widget = TierGateProgressWidget(mock_state_machine)
        widget.mount = Mock()

        features = ["iceberg_orders", "multi_pair_trading"]
        await widget.show_feature_tutorials(features)

        # Should mount at least one tutorial
        assert widget.mount.called


class TestTierTransitionHistory:
    """Test suite for TierTransitionHistory widget."""

    def test_initialization(self):
        """Test history widget initialization."""
        transitions = [
            {
                "date": "2024-01-01",
                "from_tier": "SNIPER",
                "to_tier": "HUNTER",
                "reason": "Balance threshold met",
                "duration": "30 days"
            }
        ]

        widget = TierTransitionHistory(transitions)
        assert widget.transitions == transitions

    def test_render(self):
        """Test rendering transition history."""
        transitions = [
            {
                "date": "2024-01-01",
                "from_tier": "SNIPER",
                "to_tier": "HUNTER",
                "reason": "Balance threshold met",
                "duration": "30 days"
            },
            {
                "date": "2024-02-01",
                "from_tier": "HUNTER",
                "to_tier": "STRATEGIST",
                "reason": "All gates passed",
                "duration": "31 days"
            }
        ]

        widget = TierTransitionHistory(transitions)
        rendered = widget.render()

        # Should return a Panel with Table
        assert rendered is not None
