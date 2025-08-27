"""
Integration tests for baseline establishment workflow.
"""

import asyncio
import json
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio

from genesis.core.events import Event, EventType
from genesis.data.sqlite_repo import SQLiteRepository
from genesis.engine.event_bus import EventBus
from genesis.tilt.profile_manager import ProfileManager


@pytest.mark.asyncio
class TestBaselineEstablishmentWorkflow:
    """Test the complete baseline establishment workflow."""

    @pytest_asyncio.fixture
    async def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        repo = SQLiteRepository(db_path)
        await repo.initialize()

        yield repo

        await repo.shutdown()
        Path(db_path).unlink(missing_ok=True)

    @pytest_asyncio.fixture
    async def profile_manager(self, temp_db):
        """Create profile manager with temporary database."""
        return ProfileManager(temp_db)

    @pytest_asyncio.fixture
    async def event_bus(self):
        """Create event bus for testing."""
        return EventBus()

    async def test_full_baseline_learning_process(self, profile_manager, event_bus):
        """Test the full 30-day baseline learning process."""
        account_id = "test_account_001"

        # First create an account (required for foreign key)
        await profile_manager.repository.connection.execute("""
            INSERT INTO accounts (account_id, balance_usdt, tier, created_at)
            VALUES (?, '1000', 'SNIPER', datetime('now'))
        """, (account_id,))
        await profile_manager.repository.connection.commit()

        # Create profile
        profile = await profile_manager.create_baseline_profile(account_id, "normal")
        assert profile is not None
        assert profile.is_mature is False

        # Simulate 30 days of metrics
        base_time = datetime.now(UTC) - timedelta(days=30)

        for day in range(31):  # 31 days to ensure maturity
            timestamp = base_time + timedelta(days=day)

            # Simulate daily trading patterns
            for hour in [9, 10, 11, 14, 15, 16]:  # Trading hours
                # Click speed metrics
                await profile_manager.repository.save_behavioral_metric({
                    "profile_id": profile.profile_id,
                    "metric_type": "click_speed",
                    "value": Decimal(str(100 + hour * 5 + (day % 7) * 10)),
                    "timestamp": timestamp.replace(hour=hour),
                    "time_of_day_bucket": hour
                })

                # Order frequency
                if hour % 3 == 0:
                    await profile_manager.repository.save_behavioral_metric({
                        "profile_id": profile.profile_id,
                        "metric_type": "order_frequency",
                        "value": Decimal(str(10 + day % 10)),
                        "timestamp": timestamp.replace(hour=hour),
                        "time_of_day_bucket": hour
                    })

                # Cancel rate
                if hour % 2 == 0:
                    await profile_manager.repository.save_behavioral_metric({
                        "profile_id": profile.profile_id,
                        "metric_type": "cancel_rate",
                        "value": Decimal(str(0.1 + (day % 5) * 0.05)),
                        "timestamp": timestamp.replace(hour=hour),
                        "time_of_day_bucket": hour
                    })

                # Position sizing
                await profile_manager.repository.save_behavioral_metric({
                    "profile_id": profile.profile_id,
                    "metric_type": "position_size_variance",
                    "value": Decimal(str(0.2 + (hour % 3) * 0.1)),
                    "timestamp": timestamp.replace(hour=hour),
                    "time_of_day_bucket": hour
                })

        # Update baseline from collected metrics
        updated_profile = await profile_manager.update_baseline_from_metrics(
            profile.profile_id,
            force_recalculation=True
        )

        # Verify baseline is mature
        assert updated_profile.is_mature is True
        assert updated_profile.total_samples > 100  # MIN_SAMPLES_FOR_BASELINE
        assert len(updated_profile.metric_ranges) == 4  # All metric types

        # Verify time patterns exist
        assert len(updated_profile.time_of_day_patterns) > 0

        # Fire baseline complete event
        event = Event(
            event_type=EventType.BASELINE_CALCULATION_COMPLETE,
            aggregate_id=profile.profile_id,
            event_data={
                "profile_id": profile.profile_id,
                "is_mature": updated_profile.is_mature,
                "total_samples": updated_profile.total_samples
            }
        )
        await event_bus.publish(event)

    async def test_database_persistence_and_retrieval(self, profile_manager):
        """Test database persistence and retrieval of behavioral metrics."""
        account_id = "test_account_002"

        # First create an account (required for foreign key)
        await profile_manager.repository.connection.execute("""
            INSERT INTO accounts (account_id, balance_usdt, tier, created_at)
            VALUES (?, '1000', 'SNIPER', datetime('now'))
        """, (account_id,))
        await profile_manager.repository.connection.commit()

        # Create profile
        profile = await profile_manager.create_baseline_profile(account_id)

        # Save metrics
        metrics_to_save = []
        for i in range(10):
            metric = {
                "profile_id": profile.profile_id,
                "metric_type": "click_speed",
                "value": Decimal(str(100 + i * 10)),
                "timestamp": datetime.now(UTC) - timedelta(hours=i),
                "time_of_day_bucket": (12 + i) % 24
            }
            metric_id = await profile_manager.repository.save_behavioral_metric(metric)
            metrics_to_save.append(metric_id)

        # Retrieve metrics
        retrieved = await profile_manager.repository.get_metrics_for_baseline(
            profile.profile_id,
            days=1
        )

        assert len(retrieved) == 10
        assert all(m["profile_id"] == profile.profile_id for m in retrieved)
        assert all(m["metric_type"] == "click_speed" for m in retrieved)

    async def test_profile_context_switching(self, profile_manager):
        """Test switching between different profile contexts."""
        account_id = "test_account_003"

        # First create an account (required for foreign key)
        await profile_manager.repository.connection.execute("""
            INSERT INTO accounts (account_id, balance_usdt, tier, created_at)
            VALUES (?, '1000', 'SNIPER', datetime('now'))
        """, (account_id,))
        await profile_manager.repository.connection.commit()

        # Create initial profile
        profile = await profile_manager.create_baseline_profile(account_id, "normal")
        assert profile.context == "normal"

        # Switch contexts
        contexts = ["tired", "alert", "stressed", "normal"]

        for context in contexts:
            updated = await profile_manager.switch_profile_context(
                profile.profile_id,
                context
            )
            assert updated.context == context

        # Verify profile is cached
        assert profile.profile_id in profile_manager.active_profiles

    async def test_baseline_reset_functionality(self, profile_manager):
        """Test baseline reset clears all learned patterns."""
        account_id = "test_account_004"

        # First create an account (required for foreign key)
        await profile_manager.repository.connection.execute("""
            INSERT INTO accounts (account_id, balance_usdt, tier, created_at)
            VALUES (?, '1000', 'SNIPER', datetime('now'))
        """, (account_id,))
        await profile_manager.repository.connection.commit()

        # Create and populate profile
        profile = await profile_manager.create_baseline_profile(account_id)

        # Add some metrics
        for i in range(50):
            await profile_manager.repository.save_behavioral_metric({
                "profile_id": profile.profile_id,
                "metric_type": "order_frequency",
                "value": Decimal(str(10 + i)),
                "timestamp": datetime.now(UTC) - timedelta(hours=i)
            })

        # Calculate baseline
        baseline = await profile_manager.update_baseline_from_metrics(profile.profile_id)
        assert len(baseline.metric_ranges) > 0

        # Reset baseline
        reset_profile = await profile_manager.reset_baseline(profile.profile_id)

        assert reset_profile.is_mature is False
        assert reset_profile.total_samples == 0
        assert len(reset_profile.metric_ranges) == 0

    async def test_export_functionality(self, profile_manager):
        """Test exporting baseline data for analysis."""
        account_id = "test_account_005"

        # First create an account (required for foreign key)
        await profile_manager.repository.connection.execute("""
            INSERT INTO accounts (account_id, balance_usdt, tier, created_at)
            VALUES (?, '1000', 'SNIPER', datetime('now'))
        """, (account_id,))
        await profile_manager.repository.connection.commit()

        # Create profile
        profile = await profile_manager.create_baseline_profile(account_id)

        # Add metrics
        for i in range(20):
            timestamp = datetime.now(UTC) - timedelta(hours=i)
            await profile_manager.repository.save_behavioral_metric({
                "profile_id": profile.profile_id,
                "metric_type": "cancel_rate",
                "value": Decimal(str(0.1 + i * 0.01)),
                "timestamp": timestamp,
                "session_context": "alert" if i % 2 == 0 else "tired",
                "time_of_day_bucket": timestamp.hour
            })

        # Export data
        export_data = await profile_manager.repository.export_baseline_data(profile.profile_id)

        assert "profile_id" in export_data
        assert "baseline" in export_data
        assert "metrics" in export_data
        assert "export_timestamp" in export_data

        # Verify metrics are grouped by type
        assert "cancel_rate" in export_data["metrics"]
        assert len(export_data["metrics"]["cancel_rate"]) == 20

    async def test_event_flow_integration(self, event_bus):
        """Test event flow from metric collection to baseline update."""
        # Start the event bus
        await event_bus.start()

        events_received = []

        async def event_handler(event):
            events_received.append(event)

        # Subscribe to baseline events
        event_bus.subscribe(EventType.BEHAVIORAL_METRIC_RECORDED, event_handler)
        event_bus.subscribe(EventType.BASELINE_CALCULATION_STARTED, event_handler)
        event_bus.subscribe(EventType.BASELINE_CALCULATION_COMPLETE, event_handler)

        # Simulate metric recording
        await event_bus.publish(Event(
            event_type=EventType.BEHAVIORAL_METRIC_RECORDED,
            event_data={"metric_type": "click_speed", "value": "150"}
        ))

        # Simulate baseline calculation
        await event_bus.publish(Event(
            event_type=EventType.BASELINE_CALCULATION_STARTED,
            event_data={"profile_id": "test"}
        ))

        await event_bus.publish(Event(
            event_type=EventType.BASELINE_CALCULATION_COMPLETE,
            event_data={"profile_id": "test", "is_mature": True}
        ))

        # Allow events to process
        await asyncio.sleep(0.2)

        assert len(events_received) == 3
        assert events_received[0].event_type == EventType.BEHAVIORAL_METRIC_RECORDED
        assert events_received[1].event_type == EventType.BASELINE_CALCULATION_STARTED
        assert events_received[2].event_type == EventType.BASELINE_CALCULATION_COMPLETE

        # Stop the event bus
        await event_bus.stop()

    async def test_profile_validation_and_consistency(self, profile_manager):
        """Test profile validation and consistency checks."""
        account_id = "test_account_006"

        # First create an account (required for foreign key)
        await profile_manager.repository.connection.execute("""
            INSERT INTO accounts (account_id, balance_usdt, tier, created_at)
            VALUES (?, '1000', 'SNIPER', datetime('now'))
        """, (account_id,))
        await profile_manager.repository.connection.commit()

        # Create profile
        profile = await profile_manager.create_baseline_profile(account_id)

        # Check validation with insufficient data
        issues = profile_manager.validate_profile_consistency(profile)

        assert len(issues) > 0
        assert any("Insufficient samples" in issue for issue in issues)
        assert any("not mature" in issue for issue in issues)

        # Add sufficient metrics
        for i in range(150):
            for metric_type in ["click_speed", "order_frequency", "cancel_rate", "position_size_variance"]:
                await profile_manager.repository.save_behavioral_metric({
                    "profile_id": profile.profile_id,
                    "metric_type": metric_type,
                    "value": Decimal(str(10 + i)),
                    "timestamp": datetime.now(UTC) - timedelta(days=35-i//10)
                })

        # Update baseline
        updated = await profile_manager.update_baseline_from_metrics(profile.profile_id)

        # Validate again
        issues = profile_manager.validate_profile_consistency(updated)

        # Should have fewer or no issues now
        assert len(issues) == 0 or not any("Insufficient samples" in issue for issue in issues)

    async def test_multiple_profile_management(self, profile_manager):
        """Test managing multiple profiles simultaneously."""
        accounts = ["account_001", "account_002", "account_003"]
        profiles = []

        # Create accounts and profiles
        for account_id in accounts:
            # First create an account (required for foreign key)
            await profile_manager.repository.connection.execute("""
                INSERT INTO accounts (account_id, balance_usdt, tier, created_at)
                VALUES (?, '1000', 'SNIPER', datetime('now'))
            """, (account_id,))
            await profile_manager.repository.connection.commit()

            profile = await profile_manager.create_baseline_profile(account_id)
            profiles.append(profile)

        # Verify all profiles are cached
        assert len(profile_manager.active_profiles) >= len(accounts)

        # Export all profiles
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            export_path = temp_file.name

        summary = await profile_manager.export_all_profiles(export_path)

        assert summary["profiles_exported"] >= len(accounts)

        # Verify export file
        with open(export_path) as f:
            export_data = json.load(f)

        assert "profiles" in export_data
        assert len(export_data["profiles"]) >= len(accounts)

        # Cleanup
        Path(export_path).unlink(missing_ok=True)

    async def test_performance_with_large_dataset(self, profile_manager):
        """Test performance with large number of metrics."""
        account_id = "test_account_performance"

        # First create an account (required for foreign key)
        await profile_manager.repository.connection.execute("""
            INSERT INTO accounts (account_id, balance_usdt, tier, created_at)
            VALUES (?, '1000', 'SNIPER', datetime('now'))
        """, (account_id,))
        await profile_manager.repository.connection.commit()

        profile = await profile_manager.create_baseline_profile(account_id)

        # Add 10,000 metrics
        start_time = datetime.now(UTC)

        for i in range(10000):
            if i % 1000 == 0:
                print(f"Added {i} metrics...")

            await profile_manager.repository.save_behavioral_metric({
                "profile_id": profile.profile_id,
                "metric_type": "click_speed",
                "value": Decimal(str(100 + (i % 100))),
                "timestamp": datetime.now(UTC) - timedelta(seconds=i),
                "time_of_day_bucket": i % 24
            })

        # Calculate baseline
        baseline_start = datetime.now(UTC)
        updated = await profile_manager.update_baseline_from_metrics(profile.profile_id)
        baseline_time = (datetime.now(UTC) - baseline_start).total_seconds()

        # Should complete in reasonable time
        assert baseline_time < 5  # Less than 5 seconds
        assert updated.total_samples > 0

        total_time = (datetime.now(UTC) - start_time).total_seconds()
        print(f"Total time for 10,000 metrics: {total_time:.2f} seconds")
        print(f"Baseline calculation time: {baseline_time:.2f} seconds")
