"""
Profile management for behavioral baselines.

This module provides functionality for managing multiple behavioral
profile contexts and operations.
"""

import json
from datetime import UTC, datetime

import structlog

from genesis.tilt.baseline import BaselineProfile, BehavioralBaseline, BehavioralMetric

logger = structlog.get_logger(__name__)


class ProfileManager:
    """Manages behavioral profiles and their contexts."""

    def __init__(self, repository):
        """
        Initialize the profile manager.
        
        Args:
            repository: Data repository for persistence
        """
        self.repository = repository
        self.baseline_calculator = BehavioralBaseline()
        self.active_profiles: dict[str, BaselineProfile] = {}

    async def create_baseline_profile(
        self,
        account_id: str,
        context: str = "normal"
    ) -> BaselineProfile:
        """
        Create a new baseline profile.
        
        Args:
            account_id: Account ID
            context: Profile context (normal/tired/alert/stressed)
            
        Returns:
            New baseline profile
        """
        # Check if profile already exists
        existing_profile = await self.repository.get_tilt_profile(account_id)

        if not existing_profile:
            # Create new tilt profile in database
            profile_id = await self.repository.create_tilt_profile(account_id)
        else:
            profile_id = existing_profile["profile_id"]

        # Create baseline profile object
        profile = BaselineProfile(
            profile_id=profile_id,
            learning_start_date=datetime.now(UTC),
            is_mature=False,
            context=context
        )

        # Cache in memory
        self.active_profiles[profile_id] = profile

        logger.info(
            "Baseline profile created",
            profile_id=profile_id,
            account_id=account_id,
            context=context
        )

        return profile

    async def reset_baseline(self, profile_id: str) -> BaselineProfile:
        """
        Reset a profile's baseline, clearing all learned patterns.
        
        Args:
            profile_id: Profile ID to reset
            
        Returns:
            Reset baseline profile
        """
        # Reset using baseline calculator
        new_baseline = self.baseline_calculator.reset_baseline(profile_id)

        # Update database
        await self.repository.update_tilt_profile_baseline(
            profile_id,
            {"metric_ranges": {}}  # Empty baseline
        )

        # Update cache
        self.active_profiles[profile_id] = new_baseline

        logger.info("Profile baseline reset", profile_id=profile_id)

        return new_baseline

    async def switch_profile_context(
        self,
        profile_id: str,
        new_context: str
    ) -> BaselineProfile:
        """
        Switch profile to a different context.
        
        Args:
            profile_id: Profile ID
            new_context: New context (tired/alert/stressed/normal)
            
        Returns:
            Updated profile
        """
        if new_context not in ["tired", "alert", "stressed", "normal"]:
            raise ValueError(f"Invalid context: {new_context}")

        # Get or load profile
        profile = await self._get_or_load_profile(profile_id)

        if not profile:
            raise ValueError(f"Profile {profile_id} not found")

        # Update context
        profile.context = new_context

        # Save to cache
        self.active_profiles[profile_id] = profile

        logger.info(
            "Profile context switched",
            profile_id=profile_id,
            new_context=new_context
        )

        return profile

    async def update_baseline_from_metrics(
        self,
        profile_id: str,
        force_recalculation: bool = False
    ) -> BaselineProfile:
        """
        Update profile baseline from collected metrics.
        
        Args:
            profile_id: Profile ID
            force_recalculation: Force full recalculation instead of rolling update
            
        Returns:
            Updated baseline profile
        """
        # Get metrics from database
        metrics_data = await self.repository.get_metrics_for_baseline(profile_id, days=30)

        if not metrics_data:
            logger.warning("No metrics found for baseline update", profile_id=profile_id)
            return await self._get_or_load_profile(profile_id)

        # Convert to BehavioralMetric objects
        metrics = []
        for data in metrics_data:
            metric = BehavioralMetric(
                metric_type=data["metric_type"],
                value=data["value"],
                timestamp=data["timestamp"],
                session_context=data.get("session_context"),
                time_of_day_bucket=data.get("time_of_day_bucket"),
                profile_id=profile_id
            )
            metrics.append(metric)

        # Calculate or update baseline
        # Check if we need full recalculation
        needs_recalculation = (
            force_recalculation or
            profile_id not in self.active_profiles or
            (profile_id in self.active_profiles and
             len(self.active_profiles[profile_id].metric_ranges) == 0)
        )

        if needs_recalculation:
            # Full recalculation
            logger.debug(f"Full recalculation for profile {profile_id}, metrics count: {len(metrics)}")
            baseline = self.baseline_calculator.calculate_baseline(metrics)
            # Ensure profile_id is set correctly
            baseline.profile_id = profile_id
            logger.debug(f"Calculated baseline has {len(baseline.metric_ranges)} metric ranges")
        else:
            # Rolling update
            current_baseline = self.active_profiles[profile_id]
            logger.debug(f"Rolling update for profile {profile_id}")
            baseline = self.baseline_calculator.update_rolling_baseline(
                current_baseline,
                metrics
            )

        # Update database
        baseline_dict = {
            "metric_ranges": {
                k: {
                    "mean": float(v.mean),  # Convert Decimal to float for JSON
                    "std_dev": float(v.std_dev),
                    "lower_bound": float(v.lower_bound),
                    "upper_bound": float(v.upper_bound)
                }
                for k, v in baseline.metric_ranges.items()
            }
        }
        await self.repository.update_tilt_profile_baseline(profile_id, baseline_dict)

        # Update cache
        self.active_profiles[profile_id] = baseline

        logger.info(
            "Baseline updated from metrics",
            profile_id=profile_id,
            metric_count=len(metrics),
            is_mature=baseline.is_mature
        )

        return baseline

    async def get_profile_by_account(self, account_id: str) -> BaselineProfile | None:
        """
        Get profile for an account.
        
        Args:
            account_id: Account ID
            
        Returns:
            Baseline profile or None
        """
        # Get from database
        profile_data = await self.repository.get_tilt_profile(account_id)

        if not profile_data:
            return None

        profile_id = profile_data["profile_id"]

        # Check cache first
        if profile_id in self.active_profiles:
            return self.active_profiles[profile_id]

        # Load from database
        return await self._load_profile_from_db(profile_id)

    async def _get_or_load_profile(self, profile_id: str) -> BaselineProfile | None:
        """
        Get profile from cache or load from database.
        
        Args:
            profile_id: Profile ID
            
        Returns:
            Baseline profile or None
        """
        # Check cache
        if profile_id in self.active_profiles:
            return self.active_profiles[profile_id]

        # Load from database
        return await self._load_profile_from_db(profile_id)

    async def _load_profile_from_db(self, profile_id: str) -> BaselineProfile | None:
        """
        Load profile from database.
        
        Args:
            profile_id: Profile ID
            
        Returns:
            Baseline profile or None
        """
        # Get metrics for baseline calculation
        metrics_data = await self.repository.get_metrics_for_baseline(profile_id, days=30)

        if not metrics_data:
            # Create empty profile
            profile = BaselineProfile(
                profile_id=profile_id,
                learning_start_date=datetime.now(UTC),
                is_mature=False
            )
        else:
            # Convert to BehavioralMetric objects
            metrics = []
            for data in metrics_data:
                metric = BehavioralMetric(
                    metric_type=data["metric_type"],
                    value=data["value"],
                    timestamp=data["timestamp"],
                    session_context=data.get("session_context"),
                    time_of_day_bucket=data.get("time_of_day_bucket"),
                    profile_id=profile_id
                )
                metrics.append(metric)

            # Calculate baseline
            profile = self.baseline_calculator.calculate_baseline(metrics)

        # Cache it
        self.active_profiles[profile_id] = profile

        return profile

    def validate_profile_consistency(self, profile: BaselineProfile) -> list[str]:
        """
        Validate profile consistency and identify issues.
        
        Args:
            profile: Profile to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check if profile has minimum data
        if profile.total_samples < 100:
            issues.append(f"Insufficient samples: {profile.total_samples} < 100")

        # Check if learning period is complete
        if not profile.is_mature:
            issues.append("Profile not mature - still in learning period")

        # Check for missing metric types
        expected_metrics = ["click_speed", "order_frequency", "position_size_variance", "cancel_rate"]
        missing_metrics = [m for m in expected_metrics if m not in profile.metric_ranges]

        if missing_metrics:
            issues.append(f"Missing metrics: {', '.join(missing_metrics)}")

        # Check for suspicious ranges
        for metric_type, metric_range in profile.metric_ranges.items():
            if metric_range.std_dev == 0:
                issues.append(f"{metric_type} has zero variance - may be stale")

            if metric_range.upper_bound == metric_range.lower_bound:
                issues.append(f"{metric_type} has identical bounds - needs recalculation")

        return issues

    async def export_all_profiles(self, output_path: str) -> dict:
        """
        Export all profiles for backup or analysis.
        
        Args:
            output_path: Path to export file
            
        Returns:
            Export summary
        """
        export_data = {
            "export_timestamp": datetime.now(UTC).isoformat(),
            "profiles": {}
        }

        # Export each cached profile
        for profile_id, profile in self.active_profiles.items():
            # Get full data from repository
            baseline_data = await self.repository.export_baseline_data(profile_id)
            export_data["profiles"][profile_id] = baseline_data

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        summary = {
            "profiles_exported": len(export_data["profiles"]),
            "export_path": output_path,
            "timestamp": export_data["export_timestamp"]
        }

        logger.info(
            "Profiles exported",
            count=summary["profiles_exported"],
            path=output_path
        )

        return summary

    async def cleanup_stale_profiles(self, inactive_days: int = 90) -> int:
        """
        Clean up profiles that haven't been used recently.
        
        Args:
            inactive_days: Days of inactivity before cleanup
            
        Returns:
            Number of profiles cleaned up
        """
        # This would be implemented when needed
        # For now, just clear the cache of old entries
        initial_count = len(self.active_profiles)

        # Clear profiles not accessed recently
        cutoff_time = datetime.now(UTC).timestamp() - (inactive_days * 86400)
        profiles_to_remove = []

        for profile_id, profile in self.active_profiles.items():
            if profile.last_updated:
                if profile.last_updated.timestamp() < cutoff_time:
                    profiles_to_remove.append(profile_id)

        for profile_id in profiles_to_remove:
            del self.active_profiles[profile_id]

        cleaned_count = len(profiles_to_remove)

        if cleaned_count > 0:
            logger.info(
                "Stale profiles cleaned from cache",
                cleaned=cleaned_count,
                remaining=len(self.active_profiles)
            )

        return cleaned_count
