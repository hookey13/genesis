"""Tier transition readiness assessment system.

Evaluates whether a trader is psychologically and technically ready
to advance to the next tier, preventing premature transitions that
could lead to account destruction.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import structlog

from genesis.core.exceptions import ValidationError
from genesis.data.models_db import (
    AccountDB,
    Session,
    TierTransition,
    TiltEventDB,
    TiltProfile,
    Trade,
    get_session,
)

logger = structlog.get_logger(__name__)


# Minimum requirements for tier readiness
READINESS_REQUIREMENTS = {
    "HUNTER": {
        "min_days_at_tier": 30,
        "max_tilt_score": 30,
        "min_profitability_ratio": 0.55,
        "max_recent_tilt_events": 2,
        "min_trades": 50,
        "min_readiness_score": 80,
    },
    "STRATEGIST": {
        "min_days_at_tier": 60,
        "max_tilt_score": 25,
        "min_profitability_ratio": 0.60,
        "max_recent_tilt_events": 1,
        "min_trades": 200,
        "min_readiness_score": 85,
    },
    "ARCHITECT": {
        "min_days_at_tier": 90,
        "max_tilt_score": 20,
        "min_profitability_ratio": 0.65,
        "max_recent_tilt_events": 0,
        "min_trades": 500,
        "min_readiness_score": 90,
    },
    "EMPEROR": {
        "min_days_at_tier": 180,
        "max_tilt_score": 15,
        "min_profitability_ratio": 0.70,
        "max_recent_tilt_events": 0,
        "min_trades": 1000,
        "min_readiness_score": 95,
    },
}


@dataclass
class ReadinessReport:
    """Comprehensive readiness assessment report."""

    account_id: str
    current_tier: str
    target_tier: str
    readiness_score: int  # 0-100
    is_ready: bool
    assessment_timestamp: datetime

    # Component scores
    behavioral_stability_score: int
    profitability_score: int
    consistency_score: int
    risk_management_score: int
    experience_score: int

    # Detailed metrics
    days_at_current_tier: int
    current_tilt_score: int
    recent_tilt_events: int
    profitability_ratio: Decimal
    risk_adjusted_return: Decimal
    max_drawdown: Decimal
    trade_count: int

    # Failure reasons
    failure_reasons: list[str] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "account_id": self.account_id,
            "current_tier": self.current_tier,
            "target_tier": self.target_tier,
            "readiness_score": self.readiness_score,
            "is_ready": self.is_ready,
            "assessment_timestamp": self.assessment_timestamp.isoformat(),
            "component_scores": {
                "behavioral_stability": self.behavioral_stability_score,
                "profitability": self.profitability_score,
                "consistency": self.consistency_score,
                "risk_management": self.risk_management_score,
                "experience": self.experience_score,
            },
            "metrics": {
                "days_at_tier": self.days_at_current_tier,
                "tilt_score": self.current_tilt_score,
                "recent_tilt_events": self.recent_tilt_events,
                "profitability_ratio": float(self.profitability_ratio),
                "risk_adjusted_return": float(self.risk_adjusted_return),
                "max_drawdown": float(self.max_drawdown),
                "trade_count": self.trade_count,
            },
            "failure_reasons": self.failure_reasons,
            "recommendations": self.recommendations,
        }


class TierReadinessAssessor:
    """Assesses trader readiness for tier transitions."""

    def __init__(self, session: Session | None = None):
        """Initialize readiness assessor.

        Args:
            session: Optional database session
        """
        self.session = session or get_session()

    async def assess_readiness(
        self, profile_id: str, target_tier: str
    ) -> ReadinessReport:
        """Assess readiness for tier transition.

        Args:
            profile_id: Tilt profile ID to assess
            target_tier: Target tier to transition to

        Returns:
            ReadinessReport with comprehensive assessment
        """
        # Get profile and account
        profile = (
            self.session.query(TiltProfile).filter_by(profile_id=profile_id).first()
        )

        if not profile:
            raise ValidationError(f"Profile not found: {profile_id}")

        account = (
            self.session.query(AccountDB)
            .filter_by(account_id=profile.account_id)
            .first()
        )

        if not account:
            raise ValidationError(f"Account not found: {profile.account_id}")

        # Calculate all component scores
        behavioral_score = await self._assess_behavioral_stability(profile)
        profitability_score = await self._assess_profitability(account)
        consistency_score = await self._assess_consistency(account)
        risk_score = await self._assess_risk_management(account)
        experience_score = await self._assess_experience(account)

        # Calculate overall readiness score
        readiness_score = self._calculate_readiness_score(
            behavioral_score,
            profitability_score,
            consistency_score,
            risk_score,
            experience_score,
        )

        # Get detailed metrics
        metrics = await self._gather_detailed_metrics(account, profile)

        # Check requirements
        requirements = READINESS_REQUIREMENTS.get(target_tier, {})
        is_ready, failure_reasons = self._check_requirements(
            metrics, requirements, readiness_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, requirements, failure_reasons
        )

        # Create report
        report = ReadinessReport(
            account_id=account.account_id,
            current_tier=account.current_tier,
            target_tier=target_tier,
            readiness_score=readiness_score,
            is_ready=is_ready,
            assessment_timestamp=datetime.utcnow(),
            behavioral_stability_score=behavioral_score,
            profitability_score=profitability_score,
            consistency_score=consistency_score,
            risk_management_score=risk_score,
            experience_score=experience_score,
            days_at_current_tier=metrics["days_at_tier"],
            current_tilt_score=metrics["tilt_score"],
            recent_tilt_events=metrics["recent_tilt_events"],
            profitability_ratio=metrics["profitability_ratio"],
            risk_adjusted_return=metrics["risk_adjusted_return"],
            max_drawdown=metrics["max_drawdown"],
            trade_count=metrics["trade_count"],
            failure_reasons=failure_reasons,
            recommendations=recommendations,
        )

        # Store assessment in database
        await self._store_assessment(report)

        logger.info(
            "Readiness assessment completed",
            profile_id=profile_id,
            target_tier=target_tier,
            readiness_score=readiness_score,
            is_ready=is_ready,
        )

        return report

    async def _assess_behavioral_stability(self, profile: TiltProfile) -> int:
        """Assess behavioral stability over past 30 days.

        Args:
            profile: Tilt profile to assess

        Returns:
            Behavioral stability score (0-100)
        """
        # Get recent tilt events
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_events = (
            self.session.query(TiltEventDB)
            .filter(
                TiltEventDB.profile_id == profile.profile_id,
                TiltEventDB.timestamp >= thirty_days_ago,
            )
            .all()
        )

        # Base score starts at 100
        score = 100

        # Deduct for current tilt score
        score -= min(profile.current_tilt_score, 50)

        # Deduct for recent tilt events
        for event in recent_events:
            if event.severity == "HIGH":
                score -= 15
            elif event.severity == "MEDIUM":
                score -= 10
            else:
                score -= 5

        # Bonus for recovery completion
        if profile.recovery_required and profile.journal_entries_required == 0:
            score += 10

        return max(0, min(100, score))

    async def _assess_profitability(self, account: AccountDB) -> int:
        """Assess consistent profitability.

        Args:
            account: AccountDB to assess

        Returns:
            Profitability score (0-100)
        """
        # Get trades from last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        trades = (
            self.session.query(Trade)
            .filter(
                Trade.account_id == account.account_id,
                Trade.closed_at >= thirty_days_ago,
                Trade.closed_at.isnot(None),
            )
            .all()
        )

        if len(trades) < 10:
            return 0  # Not enough data

        # Calculate profitability metrics
        profitable_trades = [t for t in trades if t.pnl_usdt > 0]
        profitability_ratio = len(profitable_trades) / len(trades)

        # Calculate average profit/loss
        total_pnl = sum(t.pnl_usdt for t in trades)
        avg_pnl = total_pnl / len(trades)

        # Base score on profitability ratio
        score = int(profitability_ratio * 100)

        # Adjust for average P&L
        if avg_pnl > 0:
            score = min(100, score + 10)

        return max(0, min(100, score))

    async def _assess_consistency(self, account: AccountDB) -> int:
        """Assess trading consistency.

        Args:
            account: AccountDB to assess

        Returns:
            Consistency score (0-100)
        """
        # Get daily P&L for last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        trades = (
            self.session.query(Trade)
            .filter(
                Trade.account_id == account.account_id,
                Trade.closed_at >= thirty_days_ago,
                Trade.closed_at.isnot(None),
            )
            .all()
        )

        if len(trades) < 10:
            return 0

        # Group trades by day
        daily_pnl = {}
        for trade in trades:
            day = trade.closed_at.date()
            if day not in daily_pnl:
                daily_pnl[day] = Decimal("0")
            daily_pnl[day] += trade.pnl_usdt

        if len(daily_pnl) < 5:
            return 0  # Not enough trading days

        # Calculate standard deviation of daily P&L
        pnl_values = list(daily_pnl.values())
        avg_daily_pnl = sum(pnl_values) / len(pnl_values)

        if len(pnl_values) > 1:
            std_dev = statistics.stdev([float(p) for p in pnl_values])
            coefficient_of_variation = (
                std_dev / float(abs(avg_daily_pnl))
                if avg_daily_pnl != 0
                else float("inf")
            )
        else:
            coefficient_of_variation = 0

        # Lower variation = higher consistency
        if coefficient_of_variation < 0.5:
            score = 100
        elif coefficient_of_variation < 1.0:
            score = 80
        elif coefficient_of_variation < 1.5:
            score = 60
        elif coefficient_of_variation < 2.0:
            score = 40
        else:
            score = 20

        return max(0, min(100, score))

    async def _assess_risk_management(self, account: AccountDB) -> int:
        """Assess risk management discipline.

        Args:
            account: AccountDB to assess

        Returns:
            Risk management score (0-100)
        """
        # Get recent trades
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        trades = (
            self.session.query(Trade)
            .filter(
                Trade.account_id == account.account_id,
                Trade.closed_at >= thirty_days_ago,
                Trade.closed_at.isnot(None),
            )
            .all()
        )

        if not trades:
            return 50  # No data, neutral score

        # Calculate max drawdown
        cumulative_pnl = Decimal("0")
        peak = Decimal("0")
        max_drawdown = Decimal("0")

        for trade in sorted(trades, key=lambda t: t.closed_at):
            cumulative_pnl += trade.pnl_usdt
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = peak - cumulative_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate risk-adjusted return (simplified Sharpe ratio)
        returns = [float(t.pnl_usdt) for t in trades]
        if len(returns) > 1:
            avg_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)
            sharpe = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe = 0

        # Score based on drawdown and Sharpe
        score = 100

        # Penalize for high drawdown
        if max_drawdown > account.balance_usdt * Decimal("0.20"):
            score -= 30
        elif max_drawdown > account.balance_usdt * Decimal("0.10"):
            score -= 15

        # Reward for good Sharpe ratio
        if sharpe > 1.5:
            score += 10
        elif sharpe < 0.5:
            score -= 20

        return max(0, min(100, score))

    async def _assess_experience(self, account: AccountDB) -> int:
        """Assess trading experience.

        Args:
            account: AccountDB to assess

        Returns:
            Experience score (0-100)
        """
        # Calculate days at current tier
        tier_started_at = account.tier_started_at or account.created_at
        days_at_tier = (datetime.utcnow() - tier_started_at).days

        # Count total trades
        total_trades = (
            self.session.query(Trade)
            .filter(Trade.account_id == account.account_id)
            .count()
        )

        # Base score on days and trades
        score = 0

        # Days component (max 50 points)
        if days_at_tier >= 180:
            score += 50
        elif days_at_tier >= 90:
            score += 40
        elif days_at_tier >= 60:
            score += 30
        elif days_at_tier >= 30:
            score += 20
        else:
            score += int(days_at_tier / 30 * 20)

        # Trades component (max 50 points)
        if total_trades >= 1000:
            score += 50
        elif total_trades >= 500:
            score += 40
        elif total_trades >= 200:
            score += 30
        elif total_trades >= 100:
            score += 20
        else:
            score += int(total_trades / 100 * 20)

        return max(0, min(100, score))

    def _calculate_readiness_score(
        self,
        behavioral: int,
        profitability: int,
        consistency: int,
        risk: int,
        experience: int,
    ) -> int:
        """Calculate overall readiness score.

        Args:
            behavioral: Behavioral stability score
            profitability: Profitability score
            consistency: Consistency score
            risk: Risk management score
            experience: Experience score

        Returns:
            Overall readiness score (0-100)
        """
        # Weighted average with behavioral stability having highest weight
        weights = {
            "behavioral": 0.30,
            "profitability": 0.20,
            "consistency": 0.20,
            "risk": 0.20,
            "experience": 0.10,
        }

        weighted_sum = (
            behavioral * weights["behavioral"]
            + profitability * weights["profitability"]
            + consistency * weights["consistency"]
            + risk * weights["risk"]
            + experience * weights["experience"]
        )

        return int(weighted_sum)

    async def _gather_detailed_metrics(
        self, account: AccountDB, profile: TiltProfile
    ) -> dict[str, Any]:
        """Gather detailed metrics for assessment.

        Args:
            account: AccountDB to assess
            profile: Tilt profile

        Returns:
            Dictionary of detailed metrics
        """
        # Days at current tier
        tier_started_at = account.tier_started_at or account.created_at
        days_at_tier = (datetime.utcnow() - tier_started_at).days

        # Recent tilt events (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_tilt_events = (
            self.session.query(TiltEventDB)
            .filter(
                TiltEventDB.profile_id == profile.profile_id,
                TiltEventDB.timestamp >= thirty_days_ago,
                TiltEventDB.severity.in_(["MEDIUM", "HIGH"]),
            )
            .count()
        )

        # Trade metrics
        trades = (
            self.session.query(Trade)
            .filter(
                Trade.account_id == account.account_id,
                Trade.closed_at >= thirty_days_ago,
                Trade.closed_at.isnot(None),
            )
            .all()
        )

        if trades:
            profitable_trades = [t for t in trades if t.pnl_usdt > 0]
            profitability_ratio = Decimal(len(profitable_trades)) / Decimal(len(trades))

            # Risk-adjusted return
            returns = [float(t.pnl_usdt) for t in trades]
            if len(returns) > 1:
                avg_return = statistics.mean(returns)
                std_return = statistics.stdev(returns)
                risk_adjusted_return = (
                    Decimal(avg_return / std_return) if std_return > 0 else Decimal("0")
                )
            else:
                risk_adjusted_return = Decimal("0")

            # Max drawdown
            cumulative_pnl = Decimal("0")
            peak = Decimal("0")
            max_drawdown = Decimal("0")

            for trade in sorted(trades, key=lambda t: t.closed_at):
                cumulative_pnl += trade.pnl_usdt
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                drawdown = peak - cumulative_pnl
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        else:
            profitability_ratio = Decimal("0")
            risk_adjusted_return = Decimal("0")
            max_drawdown = Decimal("0")

        # Total trade count
        trade_count = (
            self.session.query(Trade)
            .filter(Trade.account_id == account.account_id)
            .count()
        )

        return {
            "days_at_tier": days_at_tier,
            "tilt_score": profile.current_tilt_score,
            "recent_tilt_events": recent_tilt_events,
            "profitability_ratio": profitability_ratio,
            "risk_adjusted_return": risk_adjusted_return,
            "max_drawdown": max_drawdown,
            "trade_count": trade_count,
        }

    def _check_requirements(
        self,
        metrics: dict[str, Any],
        requirements: dict[str, Any],
        readiness_score: int,
    ) -> tuple[bool, list[str]]:
        """Check if metrics meet requirements.

        Args:
            metrics: Calculated metrics
            requirements: Tier requirements
            readiness_score: Overall readiness score

        Returns:
            Tuple of (is_ready, failure_reasons)
        """
        failure_reasons = []

        # Check each requirement
        if metrics["days_at_tier"] < requirements.get("min_days_at_tier", 0):
            failure_reasons.append(
                f"Insufficient time at tier: {metrics['days_at_tier']} days "
                f"(required: {requirements['min_days_at_tier']})"
            )

        if metrics["tilt_score"] > requirements.get("max_tilt_score", 100):
            failure_reasons.append(
                f"Tilt score too high: {metrics['tilt_score']} "
                f"(maximum: {requirements['max_tilt_score']})"
            )

        if metrics["profitability_ratio"] < Decimal(
            str(requirements.get("min_profitability_ratio", 0))
        ):
            failure_reasons.append(
                f"Profitability ratio too low: {float(metrics['profitability_ratio']):.2%} "
                f"(required: {requirements['min_profitability_ratio']:.2%})"
            )

        if metrics["recent_tilt_events"] > requirements.get(
            "max_recent_tilt_events", 999
        ):
            failure_reasons.append(
                f"Too many recent tilt events: {metrics['recent_tilt_events']} "
                f"(maximum: {requirements['max_recent_tilt_events']})"
            )

        if metrics["trade_count"] < requirements.get("min_trades", 0):
            failure_reasons.append(
                f"Insufficient trading experience: {metrics['trade_count']} trades "
                f"(required: {requirements['min_trades']})"
            )

        if readiness_score < requirements.get("min_readiness_score", 0):
            failure_reasons.append(
                f"Overall readiness score too low: {readiness_score} "
                f"(required: {requirements['min_readiness_score']})"
            )

        is_ready = len(failure_reasons) == 0

        return is_ready, failure_reasons

    def _generate_recommendations(
        self,
        metrics: dict[str, Any],
        requirements: dict[str, Any],
        failure_reasons: list[str],
    ) -> list[str]:
        """Generate improvement recommendations.

        Args:
            metrics: Current metrics
            requirements: Tier requirements
            failure_reasons: Reasons for failure

        Returns:
            List of recommendations
        """
        recommendations = []

        if metrics["tilt_score"] > 30:
            recommendations.append(
                "Focus on emotional regulation and complete recovery protocols"
            )

        if metrics["profitability_ratio"] < Decimal("0.60"):
            recommendations.append(
                "Review and refine your trading strategy for better win rate"
            )

        if metrics["recent_tilt_events"] > 2:
            recommendations.append(
                "Identify and address tilt triggers through journaling"
            )

        if metrics["risk_adjusted_return"] < Decimal("0.5"):
            recommendations.append(
                "Improve risk management to achieve better risk-adjusted returns"
            )

        if metrics["max_drawdown"] > Decimal("1000"):
            recommendations.append(
                "Implement stricter position sizing to reduce drawdowns"
            )

        if not failure_reasons:
            recommendations.append(
                "You are ready for tier transition! Proceed with paper trading"
            )

        return recommendations

    async def _store_assessment(self, report: ReadinessReport) -> None:
        """Store assessment report in database.

        Args:
            report: Readiness report to store
        """
        try:
            # Update or create tier transition record
            transition = (
                self.session.query(TierTransition)
                .filter_by(
                    account_id=report.account_id,
                    from_tier=report.current_tier,
                    to_tier=report.target_tier,
                    transition_status="APPROACHING",
                )
                .first()
            )

            if transition:
                transition.readiness_score = report.readiness_score
                transition.updated_at = datetime.utcnow()

                if report.is_ready:
                    transition.transition_status = "READY"

                self.session.commit()

                logger.info(
                    "Updated transition record with readiness",
                    transition_id=transition.transition_id,
                    readiness_score=report.readiness_score,
                )

        except Exception as e:
            logger.error(
                "Failed to store assessment", account_id=report.account_id, error=str(e)
            )
            self.session.rollback()
