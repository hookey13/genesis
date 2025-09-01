"""Validation history tracking and analysis system."""

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
from rich.console import Console
from rich.table import Table

from genesis.validation.orchestrator import ValidationReport

logger = structlog.get_logger(__name__)


@dataclass
class ValidationHistoryEntry:
    """Single validation history entry."""
    id: int | None = None
    pipeline_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0
    overall_score: float = 0.0
    overall_passed: bool = False
    ready: bool = False
    validator_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    blocking_issues: int = 0
    report_json: str = ""

    @classmethod
    def from_report(cls, report: ValidationReport) -> "ValidationHistoryEntry":
        """Create history entry from validation report.
        
        Args:
            report: Validation report
            
        Returns:
            History entry
        """
        passed_count = sum(1 for r in report.results if r.passed)
        failed_count = len(report.results) - passed_count

        return cls(
            pipeline_name=report.pipeline_name,
            timestamp=report.timestamp,
            duration_seconds=report.duration_seconds,
            overall_score=report.overall_score,
            overall_passed=report.overall_passed,
            ready=report.ready,
            validator_count=len(report.results),
            passed_count=passed_count,
            failed_count=failed_count,
            blocking_issues=len(report.blocking_issues),
            report_json=json.dumps(report.to_dict(), default=str)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ValidationHistory:
    """Manages validation history persistence and analysis."""

    def __init__(self, genesis_root: Path | None = None):
        """Initialize validation history.
        
        Args:
            genesis_root: Root directory of Genesis project
        """
        self.genesis_root = genesis_root or Path.cwd()
        self.db_path = self.genesis_root / ".genesis" / "validation_history.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pipeline_name TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    duration_seconds REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    overall_passed BOOLEAN NOT NULL,
                    ready BOOLEAN NOT NULL,
                    validator_count INTEGER NOT NULL,
                    passed_count INTEGER NOT NULL,
                    failed_count INTEGER NOT NULL,
                    blocking_issues INTEGER NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_timestamp 
                ON validation_history(pipeline_name, timestamp DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ready_score 
                ON validation_history(ready, overall_score DESC)
            """)

            # Create validator results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validator_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id INTEGER NOT NULL,
                    validator_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    score REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    error_count INTEGER NOT NULL,
                    warning_count INTEGER NOT NULL,
                    FOREIGN KEY (history_id) REFERENCES validation_history(id)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_validator_history 
                ON validator_results(history_id, validator_name)
            """)

            conn.commit()

    def save_report(self, report: ValidationReport) -> int:
        """Save validation report to history.
        
        Args:
            report: Validation report to save
            
        Returns:
            History entry ID
        """
        entry = ValidationHistoryEntry.from_report(report)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert main history entry
            cursor.execute("""
                INSERT INTO validation_history (
                    pipeline_name, timestamp, duration_seconds,
                    overall_score, overall_passed, ready,
                    validator_count, passed_count, failed_count,
                    blocking_issues, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.pipeline_name,
                entry.timestamp,
                entry.duration_seconds,
                entry.overall_score,
                entry.overall_passed,
                entry.ready,
                entry.validator_count,
                entry.passed_count,
                entry.failed_count,
                entry.blocking_issues,
                entry.report_json
            ))

            history_id = cursor.lastrowid

            # Insert individual validator results
            for result in report.results:
                cursor.execute("""
                    INSERT INTO validator_results (
                        history_id, validator_name, category,
                        passed, score, duration_seconds,
                        error_count, warning_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    history_id,
                    result.validator_name,
                    result.category,
                    result.passed,
                    result.score,
                    result.duration_seconds,
                    len(result.errors),
                    len(result.warnings)
                ))

            conn.commit()

        logger.info(
            "Saved validation report to history",
            history_id=history_id,
            pipeline=report.pipeline_name
        )

        return history_id

    def get_latest(self, pipeline_name: str | None = None) -> ValidationHistoryEntry | None:
        """Get latest validation entry.
        
        Args:
            pipeline_name: Optional pipeline filter
            
        Returns:
            Latest validation entry or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if pipeline_name:
                cursor.execute("""
                    SELECT * FROM validation_history
                    WHERE pipeline_name = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, (pipeline_name,))
            else:
                cursor.execute("""
                    SELECT * FROM validation_history
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)

            row = cursor.fetchone()

            if row:
                return self._row_to_entry(row)

        return None

    def get_history(
        self,
        limit: int = 10,
        pipeline_name: str | None = None,
        days_back: int | None = None
    ) -> list[ValidationHistoryEntry]:
        """Get validation history.
        
        Args:
            limit: Maximum number of entries
            pipeline_name: Optional pipeline filter
            days_back: Optional days to look back
            
        Returns:
            List of history entries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM validation_history WHERE 1=1"
            params = []

            if pipeline_name:
                query += " AND pipeline_name = ?"
                params.append(pipeline_name)

            if days_back:
                cutoff = datetime.utcnow() - timedelta(days=days_back)
                query += " AND timestamp >= ?"
                params.append(cutoff)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [self._row_to_entry(row) for row in rows]

    def compare_reports(
        self,
        report_id1: int,
        report_id2: int
    ) -> dict[str, Any]:
        """Compare two validation reports.
        
        Args:
            report_id1: First report ID
            report_id2: Second report ID
            
        Returns:
            Comparison results
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get both reports
            cursor.execute("""
                SELECT * FROM validation_history
                WHERE id IN (?, ?)
                ORDER BY id
            """, (report_id1, report_id2))

            rows = cursor.fetchall()

            if len(rows) != 2:
                raise ValueError("Could not find both reports")

            entry1 = self._row_to_entry(rows[0])
            entry2 = self._row_to_entry(rows[1])

            # Get validator results for both
            cursor.execute("""
                SELECT * FROM validator_results
                WHERE history_id IN (?, ?)
                ORDER BY history_id, validator_name
            """, (report_id1, report_id2))

            validator_rows = cursor.fetchall()

        # Build comparison
        comparison = {
            "report1": entry1.to_dict(),
            "report2": entry2.to_dict(),
            "score_diff": entry2.overall_score - entry1.overall_score,
            "duration_diff": entry2.duration_seconds - entry1.duration_seconds,
            "passed_diff": entry2.passed_count - entry1.passed_count,
            "failed_diff": entry2.failed_count - entry1.failed_count,
            "blocking_diff": entry2.blocking_issues - entry1.blocking_issues,
            "ready_changed": entry1.ready != entry2.ready,
            "validators": self._compare_validators(validator_rows, report_id1, report_id2)
        }

        return comparison

    def _compare_validators(
        self,
        rows: list[sqlite3.Row],
        id1: int,
        id2: int
    ) -> dict[str, Any]:
        """Compare validator results between two reports.
        
        Args:
            rows: Validator result rows
            id1: First report ID
            id2: Second report ID
            
        Returns:
            Validator comparison
        """
        validators1 = {}
        validators2 = {}

        for row in rows:
            data = {
                "name": row["validator_name"],
                "passed": row["passed"],
                "score": row["score"],
                "duration": row["duration_seconds"]
            }

            if row["history_id"] == id1:
                validators1[row["validator_name"]] = data
            else:
                validators2[row["validator_name"]] = data

        # Find changes
        improved = []
        degraded = []
        unchanged = []
        added = []
        removed = []

        all_validators = set(validators1.keys()) | set(validators2.keys())

        for name in all_validators:
            if name in validators1 and name in validators2:
                v1 = validators1[name]
                v2 = validators2[name]

                score_diff = v2["score"] - v1["score"]

                if score_diff > 1:
                    improved.append({
                        "name": name,
                        "old_score": v1["score"],
                        "new_score": v2["score"],
                        "improvement": score_diff
                    })
                elif score_diff < -1:
                    degraded.append({
                        "name": name,
                        "old_score": v1["score"],
                        "new_score": v2["score"],
                        "degradation": abs(score_diff)
                    })
                else:
                    unchanged.append(name)

            elif name in validators2:
                added.append(name)
            else:
                removed.append(name)

        return {
            "improved": improved,
            "degraded": degraded,
            "unchanged": unchanged,
            "added": added,
            "removed": removed
        }

    def get_trend_analysis(
        self,
        pipeline_name: str,
        days: int = 7
    ) -> pd.DataFrame:
        """Get trend analysis for a pipeline.
        
        Args:
            pipeline_name: Pipeline to analyze
            days: Number of days to analyze
            
        Returns:
            Pandas DataFrame with trend data
        """
        cutoff = datetime.utcnow() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as runs,
                    AVG(overall_score) as avg_score,
                    MAX(overall_score) as max_score,
                    MIN(overall_score) as min_score,
                    AVG(duration_seconds) as avg_duration,
                    SUM(CASE WHEN ready = 1 THEN 1 ELSE 0 END) as ready_count,
                    SUM(CASE WHEN overall_passed = 1 THEN 1 ELSE 0 END) as passed_count
                FROM validation_history
                WHERE pipeline_name = ? AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """

            df = pd.read_sql_query(
                query,
                conn,
                params=(pipeline_name, cutoff),
                parse_dates=["date"]
            )

        return df

    def generate_trend_graph(
        self,
        pipeline_name: str,
        days: int = 7
    ) -> str:
        """Generate ASCII trend graph.
        
        Args:
            pipeline_name: Pipeline to analyze
            days: Number of days to analyze
            
        Returns:
            ASCII graph string
        """
        df = self.get_trend_analysis(pipeline_name, days)

        if df.empty:
            return "No data available for trend analysis"

        # Simple ASCII graph
        graph_lines = []
        graph_lines.append(f"Validation Score Trend - {pipeline_name} (Last {days} days)")
        graph_lines.append("=" * 60)

        max_score = df["max_score"].max()
        min_score = df["min_score"].min()

        for _, row in df.iterrows():
            date_str = row["date"].strftime("%m/%d")
            avg_score = row["avg_score"]

            # Calculate bar length (0-50 characters)
            bar_length = int((avg_score / 100) * 50)
            bar = "█" * bar_length

            line = f"{date_str:5} | {bar:50} | {avg_score:5.1f}%"
            graph_lines.append(line)

        graph_lines.append("=" * 60)
        graph_lines.append(f"Average: {df['avg_score'].mean():.1f}%")
        graph_lines.append(f"Best: {max_score:.1f}%")
        graph_lines.append(f"Worst: {min_score:.1f}%")

        return "\n".join(graph_lines)

    def display_history_table(self, entries: list[ValidationHistoryEntry]):
        """Display history entries in a rich table.
        
        Args:
            entries: List of history entries
        """
        console = Console()

        table = Table(title="Validation History")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Pipeline", style="magenta")
        table.add_column("Timestamp", style="blue")
        table.add_column("Score", style="yellow")
        table.add_column("Ready", style="green")
        table.add_column("Passed", style="green")
        table.add_column("Duration", style="cyan")

        for entry in entries:
            ready_icon = "✅" if entry.ready else "❌"
            passed_text = f"{entry.passed_count}/{entry.validator_count}"

            table.add_row(
                str(entry.id),
                entry.pipeline_name,
                entry.timestamp.strftime("%Y-%m-%d %H:%M"),
                f"{entry.overall_score:.1f}%",
                ready_icon,
                passed_text,
                f"{entry.duration_seconds:.1f}s"
            )

        console.print(table)

    def _row_to_entry(self, row: sqlite3.Row) -> ValidationHistoryEntry:
        """Convert database row to history entry.
        
        Args:
            row: Database row
            
        Returns:
            History entry
        """
        return ValidationHistoryEntry(
            id=row["id"],
            pipeline_name=row["pipeline_name"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            duration_seconds=row["duration_seconds"],
            overall_score=row["overall_score"],
            overall_passed=bool(row["overall_passed"]),
            ready=bool(row["ready"]),
            validator_count=row["validator_count"],
            passed_count=row["passed_count"],
            failed_count=row["failed_count"],
            blocking_issues=row["blocking_issues"],
            report_json=row["report_json"]
        )

    def cleanup_old_entries(self, days_to_keep: int = 30):
        """Clean up old history entries.
        
        Args:
            days_to_keep: Number of days of history to keep
        """
        cutoff = datetime.utcnow() - timedelta(days=days_to_keep)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get IDs to delete
            cursor.execute("""
                SELECT id FROM validation_history
                WHERE timestamp < ?
            """, (cutoff,))

            ids_to_delete = [row[0] for row in cursor.fetchall()]

            if ids_to_delete:
                # Delete validator results first
                cursor.execute("""
                    DELETE FROM validator_results
                    WHERE history_id IN ({})
                """.format(",".join("?" * len(ids_to_delete))), ids_to_delete)

                # Delete history entries
                cursor.execute("""
                    DELETE FROM validation_history
                    WHERE id IN ({})
                """.format(",".join("?" * len(ids_to_delete))), ids_to_delete)

                conn.commit()

                logger.info(
                    "Cleaned up old validation history",
                    deleted_count=len(ids_to_delete),
                    cutoff=cutoff.isoformat()
                )
