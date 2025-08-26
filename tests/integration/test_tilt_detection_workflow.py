"""Integration tests for complete tilt detection workflow."""
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genesis.analytics.behavioral_metrics import (
    ClickLatencyTracker,
    InactivityTracker,
    OrderModificationTracker,
    SessionAnalyzer,
)
from genesis.analytics.config_tracker import ConfigurationChangeTracker
from genesis.core.events import EventType, Event
from genesis.engine.event_bus import EventBus
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus as RealEventBus
from genesis.tilt.baseline import BehavioralBaseline, BehavioralMetric
from genesis.tilt.detector import TiltDetector, TiltLevel
from genesis.tilt.indicators.focus_patterns import FocusPatternDetector
from genesis.tilt.interventions import InterventionManager
from genesis.tilt.profile_manager import ProfileManager


class TestTiltDetectionWorkflow:
    """Test the complete tilt detection workflow."""
    
    @pytest.fixture
    async def repo(self, tmp_path):
        """Create test repository."""
        db_path = tmp_path / "test.db"
        repo = SQLiteRepository(str(db_path))
        await repo.initialize()
        yield repo
        await repo.close()
    
    @pytest.fixture
    def event_bus(self):
        """Create real event bus."""
        return RealEventBus()
    
    @pytest.fixture
    async def profile_manager(self, repo):
        """Create profile manager with test repository."""
        manager = ProfileManager(repo)
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def tilt_detector(self, profile_manager, event_bus):
        """Create tilt detector with behavioral components."""
        return TiltDetector(
            profile_manager=profile_manager,
            event_bus=event_bus,
            click_tracker=ClickLatencyTracker(),
            modification_tracker=OrderModificationTracker(),
            focus_detector=FocusPatternDetector(),
            inactivity_tracker=InactivityTracker(),
            session_analyzer=SessionAnalyzer(),
            config_tracker=ConfigurationChangeTracker()
        )
    
    @pytest.fixture
    def intervention_manager(self, event_bus):
        """Create intervention manager."""
        return InterventionManager(event_bus=event_bus)
    
    @pytest.mark.asyncio
    async def test_normal_behavior_workflow(
        self,
        repo,
        event_bus,
        profile_manager,
        tilt_detector,
        intervention_manager
    ):
        """Test workflow with normal behavioral metrics."""
        # Create account and profile
        account_id = await repo.create_account(Decimal("1000"))
        profile_id = await repo.create_tilt_profile(account_id)
        
        # Create a behavioral profile with baseline
        profile = await profile_manager.get_profile(profile_id)
        assert profile is not None
        
        # Setup baseline data
        baseline_metrics = []
        for i in range(20):
            baseline_metrics.append(
                BehavioralMetric(
                    metric_name="click_speed",
                    value=50.0 + i % 10 - 5,  # Values around 50
                    timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
                    context={}
                )
            )
        
        # Calculate baseline
        await profile.baseline.calculate_baseline(baseline_metrics)
        
        # Create normal current metrics
        current_metrics = [
            BehavioralMetric(
                metric_name="click_speed",
                value=52.0,  # Close to baseline
                timestamp=datetime.now(timezone.utc),
                context={}
            )
        ]
        
        # Detect tilt level
        result = await tilt_detector.detect_tilt_level(profile_id, current_metrics)
        
        # Assert normal behavior detected
        assert result.tilt_level == TiltLevel.NORMAL
        assert result.tilt_score == 0
        assert len(result.anomalies) == 0
        
        # Check no intervention applied
        active_interventions = intervention_manager.get_active_interventions(profile_id)
        assert len(active_interventions) == 0
    
    @pytest.mark.asyncio
    async def test_progressive_tilt_escalation(
        self,
        repo,
        event_bus,
        profile_manager,
        tilt_detector,
        intervention_manager
    ):
        """Test progressive escalation through tilt levels."""
        # Create account and profile
        account_id = await repo.create_account(Decimal("1000"))
        profile_id = await repo.create_tilt_profile(account_id)
        
        # Mock baseline for testing
        mock_baseline = MagicMock(spec=BehavioralBaseline)
        mock_baseline.get_metric_stats.return_value = {
            'median': 50.0,
            'iqr': 10.0,
            'min': 30.0,
            'max': 70.0
        }
        
        profile = MagicMock()
        profile.baseline = mock_baseline
        profile_manager.get_profile = AsyncMock(return_value=profile)
        
        # Track events
        events_received = []
        
        async def event_handler(event_type, data):
            events_received.append((event_type, data))
        
        # Subscribe to tilt events
        event_bus.subscribe(EventType.TILT_LEVEL1_DETECTED, event_handler)
        event_bus.subscribe(EventType.TILT_LEVEL2_DETECTED, event_handler)
        event_bus.subscribe(EventType.TILT_LEVEL3_DETECTED, event_handler)
        event_bus.subscribe(EventType.INTERVENTION_APPLIED, event_handler)
        
        # Stage 1: Level 1 tilt (2-3 anomalies)
        metrics_level1 = [
            BehavioralMetric("metric_1", 100.0, datetime.now(timezone.utc), {}),  # Anomaly
            BehavioralMetric("metric_2", 5.0, datetime.now(timezone.utc), {}),    # Anomaly
            BehavioralMetric("metric_3", 52.0, datetime.now(timezone.utc), {}),   # Normal
        ]
        
        result = await tilt_detector.detect_tilt_level(profile_id, metrics_level1)
        assert result.tilt_level == TiltLevel.LEVEL1
        assert len(result.anomalies) == 2
        
        # Apply Level 1 intervention
        await intervention_manager.apply_intervention(
            profile_id,
            TiltLevel.LEVEL1,
            result.tilt_score
        )
        
        # Stage 2: Level 2 tilt (4-5 anomalies)
        metrics_level2 = [
            BehavioralMetric(f"metric_{i}", 100.0 + i*10, datetime.now(timezone.utc), {})
            for i in range(4)
        ]
        
        result = await tilt_detector.detect_tilt_level(profile_id, metrics_level2)
        assert result.tilt_level == TiltLevel.LEVEL2
        assert len(result.anomalies) == 4
        
        # Apply Level 2 intervention
        intervention = await intervention_manager.apply_intervention(
            profile_id,
            TiltLevel.LEVEL2,
            result.tilt_score
        )
        assert intervention.position_size_multiplier == Decimal("0.5")
        
        # Stage 3: Level 3 tilt (6+ anomalies)
        metrics_level3 = [
            BehavioralMetric(f"metric_{i}", 100.0 + i*10, datetime.now(timezone.utc), {})
            for i in range(6)
        ]
        
        result = await tilt_detector.detect_tilt_level(profile_id, metrics_level3)
        assert result.tilt_level == TiltLevel.LEVEL3
        assert len(result.anomalies) == 6
        
        # Apply Level 3 intervention
        intervention = await intervention_manager.apply_intervention(
            profile_id,
            TiltLevel.LEVEL3,
            result.tilt_score
        )
        assert intervention.position_size_multiplier == Decimal("0")
        assert intervention_manager.is_trading_locked(profile_id)
        
        # Wait for events to process
        await asyncio.sleep(0.1)
        
        # Verify events were published
        event_types = [e[0] for e in events_received]
        assert EventType.TILT_LEVEL1_DETECTED in event_types
        assert EventType.TILT_LEVEL2_DETECTED in event_types
        assert EventType.TILT_LEVEL3_DETECTED in event_types
        assert EventType.INTERVENTION_APPLIED in event_types
    
    @pytest.mark.asyncio
    async def test_tilt_recovery_workflow(
        self,
        repo,
        event_bus,
        profile_manager,
        tilt_detector,
        intervention_manager
    ):
        """Test recovery from tilt state."""
        # Create account and profile
        account_id = await repo.create_account(Decimal("1000"))
        profile_id = await repo.create_tilt_profile(account_id)
        
        # Apply Level 2 intervention
        await intervention_manager.apply_intervention(
            profile_id,
            TiltLevel.LEVEL2,
            60
        )
        
        # Verify intervention is active
        assert len(intervention_manager.get_active_interventions(profile_id)) == 1
        assert not intervention_manager.is_trading_locked(profile_id)
        assert intervention_manager.get_position_size_multiplier(profile_id) == Decimal("0.5")
        
        # Manually expire intervention
        for intervention in intervention_manager.active_interventions[profile_id]:
            intervention.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        
        # Check recovery
        recovered = await intervention_manager.check_recovery(profile_id)
        assert recovered
        
        # Verify no active interventions
        assert len(intervention_manager.get_active_interventions(profile_id)) == 0
        assert intervention_manager.get_position_size_multiplier(profile_id) == Decimal("1.0")
    
    @pytest.mark.asyncio
    async def test_database_persistence(
        self,
        repo,
        event_bus,
        profile_manager,
        tilt_detector,
        intervention_manager
    ):
        """Test that tilt events are persisted to database."""
        # Create account and profile
        account_id = await repo.create_account(Decimal("1000"))
        profile_id = await repo.create_tilt_profile(account_id)
        
        # Create tilt event
        event_data = {
            "profile_id": profile_id,
            "event_type": "TILT_LEVEL2_DETECTED",
            "tilt_indicators": ["rapid_clicking", "position_size_variance"],
            "tilt_score_before": 30,
            "tilt_score_after": 60,
            "intervention_message": "Position sizes reduced for safety",
            "timestamp": datetime.now(timezone.utc)
        }
        
        # Save event
        event_id = await repo.save_tilt_event(event_data)
        assert event_id is not None
        
        # Retrieve event history
        history = await repo.get_tilt_history(profile_id, days=1)
        assert len(history) == 1
        assert history[0]["event_id"] == event_id
        assert history[0]["tilt_score_after"] == 60
        
        # Update profile level
        await repo.update_tilt_profile_level(profile_id, "LEVEL2", 60)
        
        # Verify update
        profile_data = await repo.get_tilt_profile(account_id)
        assert profile_data["tilt_level"] == "LEVEL2"
        assert profile_data["current_tilt_score"] == 60
    
    @pytest.mark.asyncio
    async def test_performance_under_load(
        self,
        repo,
        event_bus,
        profile_manager,
        tilt_detector
    ):
        """Test performance with many concurrent detections."""
        # Create account and profile
        account_id = await repo.create_account(Decimal("1000"))
        profile_id = await repo.create_tilt_profile(account_id)
        
        # Mock baseline
        mock_baseline = MagicMock(spec=BehavioralBaseline)
        mock_baseline.get_metric_stats.return_value = {
            'median': 50.0,
            'iqr': 10.0,
            'min': 30.0,
            'max': 70.0
        }
        
        profile = MagicMock()
        profile.baseline = mock_baseline
        profile_manager.get_profile = AsyncMock(return_value=profile)
        
        # Create many metrics
        metrics = [
            BehavioralMetric(f"metric_{i}", 50.0 + i % 20, datetime.now(timezone.utc), {})
            for i in range(100)
        ]
        
        # Run multiple detections concurrently
        start_time = asyncio.get_event_loop().time()
        
        tasks = [
            tilt_detector.detect_tilt_level(profile_id, metrics[i:i+10])
            for i in range(0, 100, 10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # All detections should complete
        assert len(results) == 10
        
        # Each individual detection should be fast
        for result in results:
            assert result.detection_time_ms < 50
        
        # Total time should be reasonable (parallel execution)
        assert total_time_ms < 500  # Should complete in under 500ms total
    
    @pytest.mark.asyncio
    async def test_event_driven_ui_updates(
        self,
        event_bus,
        intervention_manager
    ):
        """Test that UI components can subscribe to tilt events."""
        # Mock UI component
        ui_updates = []
        
        async def ui_handler(event_type, data):
            ui_updates.append({
                'type': event_type,
                'level': data.get('tilt_level'),
                'score': data.get('tilt_score'),
                'message': data.get('message')
            })
        
        # Subscribe UI handler
        event_bus.subscribe(EventType.TILT_LEVEL1_DETECTED, ui_handler)
        event_bus.subscribe(EventType.TILT_LEVEL2_DETECTED, ui_handler)
        event_bus.subscribe(EventType.INTERVENTION_APPLIED, ui_handler)
        
        # Publish tilt events
        await event_bus.publish(
            EventType.TILT_LEVEL1_DETECTED,
            {
                'profile_id': 'test',
                'tilt_level': 'LEVEL1',
                'tilt_score': 30,
                'anomaly_count': 2
            }
        )
        
        await event_bus.publish(
            EventType.INTERVENTION_APPLIED,
            {
                'profile_id': 'test',
                'tilt_level': 'LEVEL1',
                'message': 'Take a moment to breathe'
            }
        )
        
        # Wait for events to process
        await asyncio.sleep(0.1)
        
        # Verify UI received updates
        assert len(ui_updates) == 2
        assert ui_updates[0]['type'] == EventType.TILT_LEVEL1_DETECTED
        assert ui_updates[0]['score'] == 30
        assert ui_updates[1]['type'] == EventType.INTERVENTION_APPLIED
        assert 'breathe' in ui_updates[1]['message']