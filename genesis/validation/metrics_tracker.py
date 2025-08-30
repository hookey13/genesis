"""Historical metrics tracking for validation results."""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MetricsTracker:
    """Tracks and analyzes historical validation metrics."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize metrics tracker.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.db_path = self.genesis_root / ".genesis" / "data" / "validation_metrics.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for metrics storage."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create validation_runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    overall_passed BOOLEAN,
                    overall_score REAL,
                    duration_seconds REAL,
                    validator_count INTEGER,
                    passed_count INTEGER,
                    failed_count INTEGER,
                    summary TEXT
                )
            """)

            # Create validator_results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validator_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    validator_name TEXT,
                    passed BOOLEAN,
                    score REAL,
                    error_count INTEGER,
                    warning_count INTEGER,
                    details TEXT,
                    FOREIGN KEY (run_id) REFERENCES validation_runs (id)
                )
            """)

            # Create metrics_trends table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    value REAL,
                    validator_name TEXT,
                    trend TEXT
                )
            """)

            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_validation_runs_timestamp 
                ON validation_runs (timestamp DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_validator_results_run_id 
                ON validator_results (run_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_trends_timestamp 
                ON metrics_trends (timestamp DESC)
            """)

            conn.commit()

    def store_validation_results(self, results: dict[str, Any]) -> int:
        """Store validation results in database.
        
        Args:
            results: Validation results from orchestrator
            
        Returns:
            Run ID of stored results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Extract overall metrics
                overall_passed = results.get("overall_passed", False)
                overall_score = results.get("overall_score", 0)
                duration_seconds = results.get("duration_seconds", 0)
                summary = results.get("summary", "")

                # Count validators
                validators = results.get("validators", {})
                validator_count = len(validators)
                passed_count = sum(1 for v in validators.values() if v.get("passed"))
                failed_count = validator_count - passed_count

                # Insert run record
                cursor.execute("""
                    INSERT INTO validation_runs 
                    (overall_passed, overall_score, duration_seconds, 
                     validator_count, passed_count, failed_count, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    overall_passed, overall_score, duration_seconds,
                    validator_count, passed_count, failed_count, summary
                ))

                run_id = cursor.lastrowid

                # Insert validator results
                for name, validator in validators.items():
                    passed = validator.get("passed", False)
                    score = validator.get("score", 0)
                    errors = validator.get("errors", [])
                    warnings = validator.get("warnings", [])
                    details = json.dumps(validator.get("details", {}))

                    cursor.execute("""
                        INSERT INTO validator_results
                        (run_id, validator_name, passed, score, 
                         error_count, warning_count, details)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        run_id, name, passed, score,
                        len(errors), len(warnings), details
                    ))

                conn.commit()

                # Update trends
                self._update_trends(run_id, results)

                logger.info("Stored validation results", run_id=run_id)
                return run_id

        except Exception as e:
            logger.error("Failed to store validation results", error=str(e))
            return -1

    def _update_trends(self, run_id: int, results: dict[str, Any]) -> None:
        """Update trend analysis for metrics.
        
        Args:
            run_id: ID of current run
            results: Validation results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Calculate trends for overall score
                overall_score = results.get("overall_score", 0)
                trend = self._calculate_trend("overall_score", overall_score)

                cursor.execute("""
                    INSERT INTO metrics_trends (metric_name, value, trend)
                    VALUES (?, ?, ?)
                """, ("overall_score", overall_score, trend))

                # Calculate trends for each validator
                validators = results.get("validators", {})
                for name, validator in validators.items():
                    score = validator.get("score", 0)
                    trend = self._calculate_trend(f"{name}_score", score, name)

                    cursor.execute("""
                        INSERT INTO metrics_trends 
                        (metric_name, value, validator_name, trend)
                        VALUES (?, ?, ?, ?)
                    """, (f"{name}_score", score, name, trend))

                conn.commit()

        except Exception as e:
            logger.error("Failed to update trends", error=str(e))

    def _calculate_trend(
        self,
        metric_name: str,
        current_value: float,
        validator_name: str | None = None
    ) -> str:
        """Calculate trend for a metric.
        
        Args:
            metric_name: Name of metric
            current_value: Current metric value
            validator_name: Optional validator name
            
        Returns:
            Trend indicator (improving/declining/stable)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get recent values for this metric
                if validator_name:
                    cursor.execute("""
                        SELECT value FROM metrics_trends
                        WHERE metric_name = ? AND validator_name = ?
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """, (metric_name, validator_name))
                else:
                    cursor.execute("""
                        SELECT value FROM metrics_trends
                        WHERE metric_name = ?
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """, (metric_name,))

                recent_values = [row[0] for row in cursor.fetchall()]

                if not recent_values:
                    return "new"

                # Calculate trend
                avg_recent = sum(recent_values) / len(recent_values)

                if current_value > avg_recent * 1.05:
                    return "improving"
                elif current_value < avg_recent * 0.95:
                    return "declining"
                else:
                    return "stable"

        except Exception as e:
            logger.error("Failed to calculate trend", error=str(e))
            return "unknown"

    def get_historical_data(
        self,
        days: int = 30,
        validator_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Get historical validation data.
        
        Args:
            days: Number of days of history to retrieve
            validator_name: Optional specific validator to filter
            
        Returns:
            List of historical validation results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                since = datetime.utcnow() - timedelta(days=days)

                if validator_name:
                    # Get specific validator history
                    cursor.execute("""
                        SELECT 
                            vr.timestamp,
                            r.validator_name,
                            r.passed,
                            r.score,
                            r.error_count,
                            r.warning_count
                        FROM validator_results r
                        JOIN validation_runs vr ON r.run_id = vr.id
                        WHERE r.validator_name = ? AND vr.timestamp >= ?
                        ORDER BY vr.timestamp DESC
                    """, (validator_name, since))
                else:
                    # Get overall history
                    cursor.execute("""
                        SELECT 
                            timestamp,
                            overall_passed,
                            overall_score,
                            duration_seconds,
                            validator_count,
                            passed_count,
                            failed_count
                        FROM validation_runs
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                    """, (since,))

                return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error("Failed to get historical data", error=str(e))
            return []

    def get_trend_analysis(self, days: int = 7) -> dict[str, Any]:
        """Get trend analysis for recent period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Trend analysis results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                since = datetime.utcnow() - timedelta(days=days)

                # Get overall trend
                cursor.execute("""
                    SELECT 
                        AVG(value) as avg_value,
                        MIN(value) as min_value,
                        MAX(value) as max_value,
                        COUNT(*) as data_points
                    FROM metrics_trends
                    WHERE metric_name = 'overall_score' AND timestamp >= ?
                """, (since,))

                overall_stats = cursor.fetchone()

                # Get validator-specific trends
                cursor.execute("""
                    SELECT 
                        validator_name,
                        AVG(value) as avg_score,
                        trend,
                        COUNT(*) as runs
                    FROM metrics_trends
                    WHERE validator_name IS NOT NULL AND timestamp >= ?
                    GROUP BY validator_name
                """, (since,))

                validator_trends = {}
                for row in cursor.fetchall():
                    validator_trends[row[0]] = {
                        "avg_score": row[1],
                        "trend": row[2],
                        "runs": row[3]
                    }

                # Get failure patterns
                cursor.execute("""
                    SELECT 
                        validator_name,
                        COUNT(*) as failure_count
                    FROM validator_results r
                    JOIN validation_runs vr ON r.run_id = vr.id
                    WHERE r.passed = 0 AND vr.timestamp >= ?
                    GROUP BY validator_name
                    ORDER BY failure_count DESC
                    LIMIT 5
                """, (since,))

                top_failures = [
                    {"validator": row[0], "failures": row[1]}
                    for row in cursor.fetchall()
                ]

                return {
                    "period_days": days,
                    "overall": {
                        "avg_score": overall_stats[0] if overall_stats else 0,
                        "min_score": overall_stats[1] if overall_stats else 0,
                        "max_score": overall_stats[2] if overall_stats else 0,
                        "data_points": overall_stats[3] if overall_stats else 0
                    },
                    "validator_trends": validator_trends,
                    "top_failures": top_failures
                }

        except Exception as e:
            logger.error("Failed to get trend analysis", error=str(e))
            return {}

    def generate_weekly_report(self) -> dict[str, Any]:
        """Generate weekly validation report.
        
        Returns:
            Weekly report data
        """
        # Get 7-day trend analysis
        trends = self.get_trend_analysis(days=7)

        # Get historical data for comparison
        this_week = self.get_historical_data(days=7)
        last_week = self.get_historical_data(days=14)

        # Calculate week-over-week changes
        if this_week and last_week:
            this_week_avg = sum(r.get("overall_score", 0) for r in this_week[:7]) / min(7, len(this_week))
            last_week_avg = sum(r.get("overall_score", 0) for r in last_week[7:14]) / min(7, len(last_week[7:14]))
            week_over_week_change = this_week_avg - last_week_avg
        else:
            week_over_week_change = 0

        # Find most improved and declined validators
        validator_trends = trends.get("validator_trends", {})

        most_improved = []
        most_declined = []

        for name, data in validator_trends.items():
            if data["trend"] == "improving":
                most_improved.append(name)
            elif data["trend"] == "declining":
                most_declined.append(name)

        report = {
            "week_ending": datetime.utcnow().isoformat(),
            "summary": {
                "avg_score": trends["overall"]["avg_score"],
                "min_score": trends["overall"]["min_score"],
                "max_score": trends["overall"]["max_score"],
                "total_runs": trends["overall"]["data_points"],
                "week_over_week_change": week_over_week_change
            },
            "highlights": {
                "most_improved": most_improved[:3],
                "most_declined": most_declined[:3],
                "top_failures": trends["top_failures"]
            },
            "recommendations": self._generate_weekly_recommendations(trends),
            "validator_details": validator_trends
        }

        # Store the report
        self._store_weekly_report(report)

        return report

    def _generate_weekly_recommendations(self, trends: dict[str, Any]) -> list[str]:
        """Generate recommendations based on weekly trends.
        
        Args:
            trends: Trend analysis data
            
        Returns:
            List of recommendations
        """
        recommendations = []

        # Check overall score
        overall_avg = trends["overall"]["avg_score"]
        if overall_avg < 80:
            recommendations.append(
                f"Overall validation score is below 80% ({overall_avg:.1f}%). "
                "Focus on improving failing validators."
            )

        # Check for declining validators
        declining = [
            name for name, data in trends["validator_trends"].items()
            if data["trend"] == "declining"
        ]
        if declining:
            recommendations.append(
                f"The following validators are declining: {', '.join(declining[:3])}. "
                "Investigate and address issues."
            )

        # Check for consistent failures
        top_failures = trends.get("top_failures", [])
        if top_failures and top_failures[0]["failures"] > 3:
            recommendations.append(
                f"{top_failures[0]['validator']} has failed {top_failures[0]['failures']} times. "
                "This requires immediate attention."
            )

        # Positive feedback
        if overall_avg >= 90:
            recommendations.append(
                "Excellent validation performance! Keep up the good work."
            )

        return recommendations

    def _store_weekly_report(self, report: dict[str, Any]) -> None:
        """Store weekly report to file.
        
        Args:
            report: Weekly report data
        """
        try:
            reports_dir = self.genesis_root / "docs" / "reports" / "weekly"
            reports_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d")
            report_path = reports_dir / f"weekly_validation_report_{timestamp}.json"

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info("Stored weekly report", path=str(report_path))

        except Exception as e:
            logger.error("Failed to store weekly report", error=str(e))

    def create_alert(
        self,
        validator_name: str,
        threshold: float,
        condition: str = "below"
    ) -> None:
        """Create an alert for a validator metric.
        
        Args:
            validator_name: Name of validator to monitor
            threshold: Threshold value
            condition: Alert condition (below/above)
        """
        # This would integrate with the alerting system
        # For now, just log the alert configuration
        logger.info(
            "Alert configured",
            validator=validator_name,
            threshold=threshold,
            condition=condition
        )
