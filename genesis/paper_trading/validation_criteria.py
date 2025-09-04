"""Validation criteria for strategy promotion."""

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CriteriaResult:
    """Result of a single criteria check."""

    name: str
    passed: bool
    value: Any
    required: Any
    message: str = ""


@dataclass
class ValidationResult:
    """Result of validation against all criteria."""

    passed: bool
    criteria_results: list[CriteriaResult]
    confidence_score: float
    message: str = ""


@dataclass
class ValidationCriteria:
    """Criteria for validating paper trading strategies."""

    min_trades: int = 100
    min_days: int = 7
    min_sharpe: float = 1.5
    max_drawdown: float = 0.10
    min_win_rate: float = 0.55
    min_profit_factor: float = 1.2
    regression_threshold: float = 0.20
    confidence_level: float = 0.95

    def is_eligible(self, metrics: dict[str, Any]) -> bool:
        """Check if metrics meet validation criteria.

        Args:
            metrics: Performance metrics dictionary

        Returns:
            True if all criteria are met
        """
        eligibility_checks = {
            "min_trades": metrics.get("total_trades", 0) >= self.min_trades,
            "min_days": metrics.get("days_running", 0) >= self.min_days,
            "min_sharpe": metrics.get("sharpe_ratio", 0) >= self.min_sharpe,
            "max_drawdown": metrics.get("max_drawdown", 1.0) <= self.max_drawdown,
            "min_win_rate": metrics.get("win_rate", 0) >= self.min_win_rate,
            "min_profit_factor": metrics.get("profit_factor", 0)
            >= self.min_profit_factor,
        }

        all_passed = all(eligibility_checks.values())

        if not all_passed:
            failed_criteria = [
                criterion
                for criterion, passed in eligibility_checks.items()
                if not passed
            ]
            logger.info(
                "Strategy failed validation criteria",
                strategy_id=metrics.get("strategy_id"),
                failed_criteria=failed_criteria,
                checks=eligibility_checks,
            )
        else:
            logger.info(
                "Strategy passed all validation criteria",
                strategy_id=metrics.get("strategy_id"),
                metrics=metrics,
            )

        return all_passed

    def get_eligibility_report(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Generate detailed eligibility report.

        Args:
            metrics: Performance metrics dictionary

        Returns:
            Detailed report of criteria status
        """
        report = {
            "eligible": self.is_eligible(metrics),
            "criteria_status": {
                "trades": {
                    "required": self.min_trades,
                    "actual": metrics.get("total_trades", 0),
                    "passed": metrics.get("total_trades", 0) >= self.min_trades,
                },
                "days": {
                    "required": self.min_days,
                    "actual": metrics.get("days_running", 0),
                    "passed": metrics.get("days_running", 0) >= self.min_days,
                },
                "sharpe_ratio": {
                    "required": self.min_sharpe,
                    "actual": metrics.get("sharpe_ratio", 0),
                    "passed": metrics.get("sharpe_ratio", 0) >= self.min_sharpe,
                },
                "max_drawdown": {
                    "required": self.max_drawdown,
                    "actual": metrics.get("max_drawdown", 1.0),
                    "passed": metrics.get("max_drawdown", 1.0) <= self.max_drawdown,
                },
                "win_rate": {
                    "required": self.min_win_rate,
                    "actual": metrics.get("win_rate", 0),
                    "passed": metrics.get("win_rate", 0) >= self.min_win_rate,
                },
                "profit_factor": {
                    "required": self.min_profit_factor,
                    "actual": metrics.get("profit_factor", 0),
                    "passed": metrics.get("profit_factor", 0) >= self.min_profit_factor,
                },
            },
            "confidence_score": self._calculate_confidence_score(metrics),
        }

        return report

    def _calculate_confidence_score(self, metrics: dict[str, Any]) -> float:
        """Calculate confidence score based on how well criteria are exceeded.

        Args:
            metrics: Performance metrics dictionary

        Returns:
            Confidence score between 0 and 1
        """
        scores = []

        if self.min_trades > 0:
            trade_score = (
                min(metrics.get("total_trades", 0) / self.min_trades, 2.0) / 2.0
            )
            scores.append(trade_score)

        if self.min_days > 0:
            days_score = min(metrics.get("days_running", 0) / self.min_days, 2.0) / 2.0
            scores.append(days_score)

        if self.min_sharpe > 0:
            sharpe_score = (
                min(metrics.get("sharpe_ratio", 0) / self.min_sharpe, 2.0) / 2.0
            )
            scores.append(sharpe_score)

        if self.max_drawdown > 0:
            drawdown = metrics.get("max_drawdown", 1.0)
            drawdown_score = max(0, (self.max_drawdown - drawdown) / self.max_drawdown)
            scores.append(drawdown_score)

        if self.min_win_rate > 0:
            win_rate_score = (
                min(metrics.get("win_rate", 0) / self.min_win_rate, 1.5) / 1.5
            )
            scores.append(win_rate_score)

        if self.min_profit_factor > 0:
            pf_score = (
                min(metrics.get("profit_factor", 0) / self.min_profit_factor, 2.0) / 2.0
            )
            scores.append(pf_score)

        return sum(scores) / len(scores) if scores else 0.0

    def update_criteria(self, updates: dict[str, Any]) -> None:
        """Update validation criteria.

        Args:
            updates: Dictionary of criteria to update
        """
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(
                    "Validation criteria updated", criterion=key, new_value=value
                )
            else:
                logger.warning("Unknown validation criterion", criterion=key)

    def to_dict(self) -> dict[str, Any]:
        """Convert criteria to dictionary.

        Returns:
            Dictionary representation of criteria
        """
        return {
            "min_trades": self.min_trades,
            "min_days": self.min_days,
            "min_sharpe": self.min_sharpe,
            "max_drawdown": self.max_drawdown,
            "min_win_rate": self.min_win_rate,
            "min_profit_factor": self.min_profit_factor,
            "regression_threshold": self.regression_threshold,
            "confidence_level": self.confidence_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationCriteria":
        """Create ValidationCriteria from dictionary.

        Args:
            data: Dictionary of criteria values

        Returns:
            ValidationCriteria instance
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def check_regression(
        self, current_metrics: dict[str, Any], baseline_metrics: dict[str, Any]
    ) -> bool:
        """Check if current performance has regressed from baseline.

        Args:
            current_metrics: Current performance metrics
            baseline_metrics: Baseline performance metrics

        Returns:
            True if regression detected
        """
        regression_detected = False
        regression_details = []

        if baseline_metrics.get("sharpe_ratio", 0) > 0:
            sharpe_decline = (
                baseline_metrics["sharpe_ratio"]
                - current_metrics.get("sharpe_ratio", 0)
            ) / baseline_metrics["sharpe_ratio"]
            if sharpe_decline > self.regression_threshold:
                regression_detected = True
                regression_details.append(
                    f"Sharpe ratio declined by {sharpe_decline:.2%}"
                )

        if baseline_metrics.get("win_rate", 0) > 0:
            win_rate_decline = (
                baseline_metrics["win_rate"] - current_metrics.get("win_rate", 0)
            ) / baseline_metrics["win_rate"]
            if win_rate_decline > self.regression_threshold:
                regression_detected = True
                regression_details.append(
                    f"Win rate declined by {win_rate_decline:.2%}"
                )

        if baseline_metrics.get("profit_factor", 0) > 0:
            pf_decline = (
                baseline_metrics["profit_factor"]
                - current_metrics.get("profit_factor", 0)
            ) / baseline_metrics["profit_factor"]
            if pf_decline > self.regression_threshold:
                regression_detected = True
                regression_details.append(f"Profit factor declined by {pf_decline:.2%}")

        drawdown_increase = current_metrics.get(
            "max_drawdown", 0
        ) - baseline_metrics.get("max_drawdown", 0)
        if drawdown_increase > self.regression_threshold:
            regression_detected = True
            regression_details.append(f"Drawdown increased by {drawdown_increase:.2%}")

        if regression_detected:
            logger.warning(
                "Performance regression detected",
                strategy_id=current_metrics.get("strategy_id"),
                regression_details=regression_details,
            )

        return regression_detected
