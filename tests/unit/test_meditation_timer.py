"""Unit tests for meditation timer."""
import pytest
import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from genesis.tilt.meditation_timer import (
    MeditationTimer,
    MeditationStatus,
    MeditationSession,
)


@pytest.fixture
def meditation_timer():
    """Create a meditation timer for testing."""
    return MeditationTimer()


@pytest.fixture
def timer_with_callback():
    """Create a meditation timer with tick callback."""
    callback = MagicMock()
    return MeditationTimer(on_tick=callback), callback


class TestSessionManagement:
    """Test meditation session management."""

    @pytest.mark.asyncio
    async def test_start_meditation_session(self, meditation_timer):
        """Test starting a meditation session."""
        profile_id = "test_profile"
        duration = 10  # minutes

        session = await meditation_timer.start_meditation_session(profile_id, duration)

        assert session.profile_id == profile_id
        assert session.duration_minutes == duration
        assert session.status == MeditationStatus.IN_PROGRESS
        assert session.started_at is not None
        assert session.remaining_seconds == 600  # 10 * 60

        # Check session is tracked
        assert profile_id in meditation_timer.active_sessions
        assert profile_id in meditation_timer.timer_tasks

    @pytest.mark.asyncio
    async def test_cancel_existing_session_on_new_start(self, meditation_timer):
        """Test that starting new session cancels existing one."""
        profile_id = "test_profile"

        # Start first session
        session1 = await meditation_timer.start_meditation_session(profile_id, 10)
        task1 = meditation_timer.timer_tasks[profile_id]

        # Start second session
        session2 = await meditation_timer.start_meditation_session(profile_id, 5)

        # First task should be cancelled
        assert task1.cancelled() or task1.done()
        # New session should be active
        assert meditation_timer.active_sessions[profile_id] == session2

    def test_skip_meditation(self, meditation_timer):
        """Test skipping meditation (optional requirement)."""
        profile_id = "test_profile"

        session = meditation_timer.skip_meditation(profile_id)

        assert session.profile_id == profile_id
        assert session.status == MeditationStatus.SKIPPED
        assert session.duration_minutes == 0

        # Should be marked as complete (optional satisfied)
        assert meditation_timer.is_meditation_complete(profile_id) is True

    @pytest.mark.asyncio
    async def test_cancel_session(self, meditation_timer):
        """Test cancelling an active session."""
        profile_id = "test_profile"

        session = await meditation_timer.start_meditation_session(profile_id, 10)
        task = meditation_timer.timer_tasks[profile_id]

        # Cancel session
        cancelled = await meditation_timer.cancel_session(profile_id)

        assert cancelled is True
        assert session.status == MeditationStatus.CANCELLED
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session(self, meditation_timer):
        """Test cancelling when no session exists."""
        cancelled = await meditation_timer.cancel_session("unknown_profile")
        assert cancelled is False


class TestTimerFunctionality:
    """Test timer countdown functionality."""

    @pytest.mark.asyncio
    async def test_timer_countdown(self, timer_with_callback):
        """Test that timer counts down correctly."""
        timer, callback = timer_with_callback
        profile_id = "test_profile"

        # Start 1-second session for testing
        session = await timer.start_meditation_session(profile_id, 1/60)  # 1 second

        # Wait for completion
        await asyncio.sleep(1.5)

        # Should be completed
        assert session.status == MeditationStatus.COMPLETED
        assert session.completed_at is not None

        # Callback should have been called
        assert callback.called

    @pytest.mark.asyncio
    async def test_timer_cancellation_during_countdown(self, meditation_timer):
        """Test cancelling timer during countdown."""
        profile_id = "test_profile"

        session = await meditation_timer.start_meditation_session(profile_id, 10)

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel
        await meditation_timer.cancel_session(profile_id)

        assert session.status == MeditationStatus.CANCELLED


class TestSessionRetrieval:
    """Test retrieving session information."""

    @pytest.mark.asyncio
    async def test_get_session(self, meditation_timer):
        """Test getting current session."""
        profile_id = "test_profile"

        # No session initially
        assert meditation_timer.get_session(profile_id) is None

        # Start session
        session = await meditation_timer.start_meditation_session(profile_id, 10)

        # Should retrieve same session
        retrieved = meditation_timer.get_session(profile_id)
        assert retrieved == session

    @pytest.mark.asyncio
    async def test_get_remaining_time(self, meditation_timer):
        """Test getting remaining meditation time."""
        profile_id = "test_profile"

        # No session - should be 0
        assert meditation_timer.get_remaining_time(profile_id) == 0

        # Start session
        session = await meditation_timer.start_meditation_session(profile_id, 10)

        # Should have full time initially
        assert meditation_timer.get_remaining_time(profile_id) == 600

        # Simulate countdown
        session.remaining_seconds = 300
        assert meditation_timer.get_remaining_time(profile_id) == 300

    def test_is_meditation_complete(self, meditation_timer):
        """Test checking if meditation is complete."""
        profile_id = "test_profile"

        # No session - not complete
        assert meditation_timer.is_meditation_complete(profile_id) is False

        # Skipped session - complete (optional satisfied)
        meditation_timer.skip_meditation(profile_id)
        assert meditation_timer.is_meditation_complete(profile_id) is True

        # Completed session
        profile_id2 = "test_profile2"
        session = MeditationSession(
            profile_id=profile_id2,
            duration_minutes=10,
            status=MeditationStatus.COMPLETED,
        )
        meditation_timer.active_sessions[profile_id2] = session
        assert meditation_timer.is_meditation_complete(profile_id2) is True


class TestTimeFormatting:
    """Test time formatting utilities."""

    def test_format_time(self, meditation_timer):
        """Test formatting seconds to MM:SS."""
        assert meditation_timer.format_time(0) == "00:00"
        assert meditation_timer.format_time(30) == "00:30"
        assert meditation_timer.format_time(60) == "01:00"
        assert meditation_timer.format_time(90) == "01:30"
        assert meditation_timer.format_time(600) == "10:00"
        assert meditation_timer.format_time(-5) == "00:00"  # Negative


class TestCleanup:
    """Test cleanup functionality."""

    @pytest.mark.asyncio
    async def test_clear_session(self, meditation_timer):
        """Test clearing a session."""
        profile_id = "test_profile"

        await meditation_timer.start_meditation_session(profile_id, 10)

        # Clear session
        meditation_timer.clear_session(profile_id)

        assert profile_id not in meditation_timer.active_sessions
        assert profile_id not in meditation_timer.timer_tasks

    @pytest.mark.asyncio
    async def test_cleanup_all_sessions(self, meditation_timer):
        """Test cleaning up all active sessions."""
        # Start multiple sessions
        await meditation_timer.start_meditation_session("profile1", 10)
        await meditation_timer.start_meditation_session("profile2", 15)
        await meditation_timer.start_meditation_session("profile3", 20)

        assert len(meditation_timer.active_sessions) == 3
        assert len(meditation_timer.timer_tasks) == 3

        # Cleanup all
        await meditation_timer.cleanup()

        assert len(meditation_timer.active_sessions) == 0
        assert len(meditation_timer.timer_tasks) == 0