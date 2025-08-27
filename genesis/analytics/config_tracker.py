"""
Configuration change tracking for behavioral analysis.

Monitors changes to trading settings and configuration,
which can indicate emotional states or tilt behavior.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Optional, TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from genesis.data.repository import Repository


logger = structlog.get_logger(__name__)


@dataclass
class ConfigChangeMetrics:
    """Metrics for configuration change patterns."""

    total_changes: int
    change_frequency: float  # Changes per hour
    frequent_settings: list[str]  # Most frequently changed settings
    revert_count: int  # Number of settings reverted
    stability_score: float  # 0-100, higher is more stable


class ConfigurationChangeTracker:
    """
    Tracks configuration and settings changes.

    Monitors when traders change their settings, which can indicate
    uncertainty, frustration, or system gaming attempts.
    """

    def __init__(
        self,
        window_size: int = 100,
        frequent_change_threshold: int = 3,
        repository: Optional[Repository] = None,
        profile_id: Optional[str] = None,
    ) -> None:
        """
        Initialize configuration tracker.

        Args:
            window_size: Size of rolling window for analysis
            frequent_change_threshold: Changes to consider frequent
            repository: Repository for persistence
            profile_id: Profile ID for tracking
        """
        self.window_size = window_size
        self.frequent_change_threshold = frequent_change_threshold
        self.repository = repository
        self.profile_id = profile_id

        # Change history
        self.changes: deque[dict[str, Any]] = deque(maxlen=window_size * 2)

        # Setting-specific tracking
        self.setting_history: dict[str, list[dict]] = defaultdict(list)

        # Revert detection
        self.reverts: list[dict[str, Any]] = []

        # Baseline change rate
        self.baseline_change_rate: float = 1.0  # Changes per hour

        logger.info(
            "configuration_tracker_initialized",
            window_size=window_size,
            frequent_change_threshold=frequent_change_threshold,
            has_repository=repository is not None,
        )

    async def track_config_change(
        self, setting: str, old_value: Any, new_value: Any
    ) -> None:
        """
        Track a configuration change.

        Args:
            setting: Name of the setting changed
            old_value: Previous value
            new_value: New value
        """
        timestamp = datetime.now(UTC)

        change = {
            "setting": setting,
            "old_value": old_value,
            "new_value": new_value,
            "timestamp": timestamp,
        }

        # Add to general history
        self.changes.append(change)

        # Persist to database if repository available
        if self.repository and self.profile_id:
            try:
                await self.repository.save_config_change(
                    {
                        "profile_id": self.profile_id,
                        "setting_name": setting,
                        "old_value": str(old_value) if old_value is not None else None,
                        "new_value": str(new_value) if new_value is not None else None,
                        "changed_at": timestamp,
                    }
                )
            except Exception as e:
                logger.error("Failed to save config change", error=str(e))

        # Add to setting-specific history
        self.setting_history[setting].append(
            {"value": new_value, "timestamp": timestamp}
        )

        # Check for revert (changing back to previous value)
        if self._is_revert(setting, old_value, new_value):
            self.reverts.append(change)
            logger.warning(
                "configuration_revert_detected", setting=setting, reverted_to=new_value
            )

        # Check for frequent changes
        recent_changes = self._count_recent_changes(setting, hours=1)
        if recent_changes >= self.frequent_change_threshold:
            logger.warning(
                "frequent_configuration_changes",
                setting=setting,
                changes_in_hour=recent_changes,
                threshold=self.frequent_change_threshold,
            )

        # Clean old data
        self._cleanup_old_data()

    def _is_revert(self, setting: str, old_value: Any, new_value: Any) -> bool:
        """
        Check if a change is reverting to a previous value.

        Args:
            setting: Setting name
            old_value: Previous value
            new_value: New value

        Returns:
            True if this is a revert
        """
        history = self.setting_history.get(setting, [])

        # Check if new_value appeared before old_value
        for i, entry in enumerate(history[:-1]):  # Exclude the just-added entry
            if entry["value"] == new_value:
                # Found new_value in history, check if old_value came after
                for j in range(i + 1, len(history) - 1):
                    if history[j]["value"] == old_value:
                        return True

        return False

    def _count_recent_changes(self, setting: str, hours: float = 1) -> int:
        """
        Count recent changes to a specific setting.

        Args:
            setting: Setting name
            hours: Time window in hours

        Returns:
            Number of changes
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        history = self.setting_history.get(setting, [])

        return sum(1 for entry in history if entry["timestamp"] > cutoff)

    def _cleanup_old_data(self) -> None:
        """Remove old data beyond retention window."""
        cutoff = datetime.utcnow() - timedelta(hours=24)

        # Clean reverts
        self.reverts = [r for r in self.reverts if r["timestamp"] > cutoff]

        # Clean setting history
        for setting in self.setting_history:
            self.setting_history[setting] = [
                entry
                for entry in self.setting_history[setting]
                if entry["timestamp"] > cutoff
            ]

    def get_change_metrics(self, hours: float = 1) -> ConfigChangeMetrics:
        """
        Calculate configuration change metrics.

        Args:
            hours: Time window for analysis

        Returns:
            ConfigChangeMetrics with statistics
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Filter recent changes
        recent_changes = [c for c in self.changes if c["timestamp"] > cutoff]

        if not recent_changes:
            return ConfigChangeMetrics(
                total_changes=0,
                change_frequency=0.0,
                frequent_settings=[],
                revert_count=0,
                stability_score=100.0,
            )

        # Calculate frequency
        total_changes = len(recent_changes)
        change_frequency = total_changes / hours

        # Find most frequently changed settings
        setting_counts = defaultdict(int)
        for change in recent_changes:
            setting_counts[change["setting"]] += 1

        frequent_settings = sorted(
            setting_counts.keys(), key=lambda s: setting_counts[s], reverse=True
        )[
            :5
        ]  # Top 5

        # Count recent reverts
        recent_reverts = [r for r in self.reverts if r["timestamp"] > cutoff]
        revert_count = len(recent_reverts)

        # Calculate stability score
        stability_score = self._calculate_stability_score(
            change_frequency, revert_count, total_changes
        )

        return ConfigChangeMetrics(
            total_changes=total_changes,
            change_frequency=change_frequency,
            frequent_settings=frequent_settings,
            revert_count=revert_count,
            stability_score=stability_score,
        )

    def _calculate_stability_score(
        self, change_frequency: float, revert_count: int, total_changes: int
    ) -> float:
        """
        Calculate configuration stability score.

        Args:
            change_frequency: Changes per hour
            revert_count: Number of reverts
            total_changes: Total number of changes

        Returns:
            Stability score from 0 (unstable) to 100 (stable)
        """
        score = 100.0

        # Penalize high change frequency
        if change_frequency > self.baseline_change_rate:
            excess = change_frequency - self.baseline_change_rate
            score -= min(40, excess * 10)

        # Penalize reverts heavily
        if total_changes > 0:
            revert_ratio = revert_count / total_changes
            score -= min(40, revert_ratio * 100)

        # Penalize absolute number of changes
        score -= min(20, total_changes * 2)

        return max(0.0, score)

    def is_configuration_unstable(self, threshold: float = 50.0) -> bool:
        """
        Check if configuration changes indicate instability.

        Args:
            threshold: Stability score threshold

        Returns:
            True if stability score is below threshold
        """
        metrics = self.get_change_metrics()
        return metrics.stability_score < threshold

    def get_risky_changes(self) -> list[dict[str, Any]]:
        """
        Identify potentially risky configuration changes.

        Returns:
            List of risky changes
        """
        risky_changes = []
        risky_settings = [
            "position_size",
            "max_leverage",
            "stop_loss",
            "risk_limit",
            "max_positions",
        ]

        for change in self.changes:
            setting = change["setting"].lower()

            # Check if it's a risk-related setting
            if any(risky in setting for risky in risky_settings):
                # Check if value increased risk
                if self._is_risk_increase(
                    setting, change["old_value"], change["new_value"]
                ):
                    risky_changes.append(change)

        return risky_changes

    def _is_risk_increase(self, setting: str, old_value: Any, new_value: Any) -> bool:
        """
        Check if a change increases risk.

        Args:
            setting: Setting name
            old_value: Previous value
            new_value: New value

        Returns:
            True if risk increased
        """
        try:
            # Convert to comparable types
            old_num = float(old_value) if old_value is not None else 0
            new_num = float(new_value) if new_value is not None else 0

            # Settings where increase = more risk
            increase_risk = ["position_size", "max_leverage", "max_positions"]
            # Settings where decrease = more risk
            decrease_risk = ["stop_loss", "risk_limit"]

            if any(risk in setting for risk in increase_risk):
                return new_num > old_num
            elif any(risk in setting for risk in decrease_risk):
                return new_num < old_num

        except (ValueError, TypeError):
            # Can't convert to numbers, assume any change is risky
            return True

        return False

    def get_analysis_summary(self) -> dict[str, Any]:
        """
        Get comprehensive configuration analysis.

        Returns:
            Dictionary with analysis results
        """
        metrics_1h = self.get_change_metrics(1)
        metrics_24h = self.get_change_metrics(24)

        risky_changes = self.get_risky_changes()

        # Determine configuration state
        state = "stable"
        if metrics_1h.stability_score < 30:
            state = "highly_unstable"
        elif metrics_1h.stability_score < 50:
            state = "unstable"
        elif metrics_1h.stability_score < 70:
            state = "somewhat_unstable"

        return {
            "state": state,
            "stability_score": metrics_1h.stability_score,
            "changes_last_hour": metrics_1h.total_changes,
            "changes_last_24h": metrics_24h.total_changes,
            "revert_count": metrics_1h.revert_count,
            "frequent_settings": metrics_1h.frequent_settings,
            "risky_changes": len(risky_changes),
            "recommendation": self._get_recommendation(metrics_1h, risky_changes),
        }

    def _get_recommendation(
        self, metrics: ConfigChangeMetrics, risky_changes: list[dict]
    ) -> str:
        """
        Generate recommendation based on configuration patterns.

        Args:
            metrics: Current metrics
            risky_changes: List of risky changes

        Returns:
            Recommendation string
        """
        if metrics.stability_score < 30:
            return "Stop changing settings - stick to your plan"
        elif risky_changes:
            return "Review risk settings - recent changes increased exposure"
        elif metrics.revert_count > 2:
            return "Commit to settings - frequent reverts indicate indecision"
        elif metrics.change_frequency > 5:
            return "Too many adjustments - let strategies run"
        else:
            return "Configuration stable"
