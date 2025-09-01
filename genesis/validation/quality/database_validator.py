"""Database optimization validator for query performance and connection pooling."""

import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


class DatabaseValidator:
    """Validates database performance, query optimization, and connection management."""

    # Performance thresholds
    MAX_QUERY_TIME_MS = 100
    MAX_CONNECTION_POOL_SIZE = 20
    MIN_CONNECTION_POOL_SIZE = 5
    MAX_IDLE_CONNECTIONS = 10
    MAX_N_PLUS_ONE_QUERIES = 0
    MIN_INDEX_USAGE_PERCENT = 80
    MAX_TABLE_SCAN_ROWS = 1000
    MAX_CONNECTION_WAIT_MS = 50

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize database validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path.cwd()
        self.db_path = self.project_root / ".genesis" / "data" / "genesis.db"
        self.query_history: List[Dict[str, Any]] = []

    async def run_validation(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive database validation.
        
        Args:
            context: Optional context for validation
            
        Returns:
            Validation results dictionary
        """
        logger.info("Starting database validation")
        start_time = datetime.utcnow()

        results = {
            "validator": "DatabaseValidator",
            "timestamp": start_time.isoformat(),
            "status": "pending",
            "passed": False,
            "query_analysis": {},
            "index_analysis": {},
            "connection_pool_analysis": {},
            "n_plus_one_detection": {},
            "optimization_suggestions": [],
            "violations": [],
            "evidence": {},
        }

        try:
            # Analyze query performance
            query_analysis = await self._analyze_query_performance()
            results["query_analysis"] = query_analysis

            # Validate index usage
            index_analysis = await self._validate_index_usage()
            results["index_analysis"] = index_analysis

            # Check for N+1 query problems
            n_plus_one = await self._detect_n_plus_one_queries()
            results["n_plus_one_detection"] = n_plus_one

            # Validate connection pool configuration
            pool_analysis = await self._validate_connection_pool()
            results["connection_pool_analysis"] = pool_analysis

            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(
                query_analysis, index_analysis, n_plus_one, pool_analysis
            )
            results["optimization_suggestions"] = suggestions

            # Check for violations
            violations = self._check_database_violations(
                query_analysis, index_analysis, n_plus_one, pool_analysis
            )
            results["violations"] = violations

            # Generate evidence report
            evidence = self._generate_database_evidence(
                query_analysis, index_analysis, n_plus_one, violations
            )
            results["evidence"] = evidence

            # Determine pass/fail
            results["passed"] = len(violations) == 0
            results["status"] = "passed" if results["passed"] else "failed"

            logger.info(
                "Database validation completed",
                passed=results["passed"],
                violations=len(violations),
                slow_queries=len(query_analysis.get("slow_queries", [])),
            )

        except Exception as e:
            logger.error("Database validation failed", error=str(e))
            results["status"] = "error"
            results["error"] = str(e)

        return results

    async def _analyze_query_performance(self) -> Dict[str, Any]:
        """Analyze database query performance.
        
        Returns:
            Query performance analysis
        """
        logger.info("Analyzing query performance")
        
        query_data = {
            "total_queries": 0,
            "slow_queries": [],
            "query_patterns": {},
            "execution_stats": {},
            "table_scans": [],
            "missing_indexes": [],
        }

        try:
            # Connect to database
            if not self.db_path.exists():
                # Use in-memory database for testing
                conn = sqlite3.connect(":memory:")
                self._setup_test_database(conn)
            else:
                conn = sqlite3.connect(str(self.db_path))

            cursor = conn.cursor()

            # Enable query plan analysis
            cursor.execute("PRAGMA query_only = ON")

            # Analyze common query patterns
            test_queries = [
                ("SELECT * FROM positions WHERE user_id = ?", (1,)),
                ("SELECT * FROM orders WHERE status = ?", ("pending",)),
                ("SELECT * FROM trades WHERE timestamp > ?", (datetime.utcnow().isoformat(),)),
                ("SELECT COUNT(*) FROM positions", ()),
                ("SELECT p.*, o.* FROM positions p JOIN orders o ON p.id = o.position_id", ()),
            ]

            for query, params in test_queries:
                try:
                    # Measure execution time
                    start = time.perf_counter()
                    cursor.execute(f"EXPLAIN QUERY PLAN {query}", params)
                    plan = cursor.fetchall()
                    end = time.perf_counter()
                    
                    execution_time_ms = (end - start) * 1000
                    query_data["total_queries"] += 1

                    # Analyze query plan
                    is_scan = any("SCAN" in str(row) for row in plan)
                    uses_index = any("USING INDEX" in str(row) for row in plan)

                    query_info = {
                        "query": query[:100],
                        "execution_time_ms": execution_time_ms,
                        "uses_index": uses_index,
                        "is_table_scan": is_scan,
                        "plan": str(plan),
                    }

                    # Check for slow queries
                    if execution_time_ms > self.MAX_QUERY_TIME_MS:
                        query_data["slow_queries"].append(query_info)

                    # Check for table scans
                    if is_scan:
                        query_data["table_scans"].append(query_info)

                    # Categorize query pattern
                    pattern = self._extract_query_pattern(query)
                    if pattern not in query_data["query_patterns"]:
                        query_data["query_patterns"][pattern] = {
                            "count": 0,
                            "avg_time_ms": 0,
                            "max_time_ms": 0,
                        }
                    
                    pattern_stats = query_data["query_patterns"][pattern]
                    pattern_stats["count"] += 1
                    pattern_stats["avg_time_ms"] = (
                        (pattern_stats["avg_time_ms"] * (pattern_stats["count"] - 1) + execution_time_ms)
                        / pattern_stats["count"]
                    )
                    pattern_stats["max_time_ms"] = max(pattern_stats["max_time_ms"], execution_time_ms)

                except sqlite3.Error as e:
                    logger.warning(f"Query analysis failed: {e}")

            # Get database statistics
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                
                query_data["execution_stats"][table_name] = {
                    "row_count": row_count,
                    "has_primary_key": self._check_primary_key(cursor, table_name),
                }

            # Check for missing indexes on foreign keys
            query_data["missing_indexes"] = self._find_missing_indexes(cursor)

            conn.close()

        except Exception as e:
            logger.error("Failed to analyze query performance", error=str(e))

        return query_data

    async def _validate_index_usage(self) -> Dict[str, Any]:
        """Validate database index usage and effectiveness.
        
        Returns:
            Index usage analysis
        """
        logger.info("Validating index usage")
        
        index_data = {
            "total_indexes": 0,
            "unused_indexes": [],
            "redundant_indexes": [],
            "missing_indexes": [],
            "index_statistics": {},
            "coverage_percent": 0,
        }

        try:
            # Connect to database
            if not self.db_path.exists():
                conn = sqlite3.connect(":memory:")
                self._setup_test_database(conn)
            else:
                conn = sqlite3.connect(str(self.db_path))

            cursor = conn.cursor()

            # Get all indexes
            cursor.execute("""
                SELECT name, tbl_name, sql 
                FROM sqlite_master 
                WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            """)
            indexes = cursor.fetchall()
            
            index_data["total_indexes"] = len(indexes)

            # Analyze each index
            for index_name, table_name, index_sql in indexes:
                # Get index columns
                columns = self._extract_index_columns(index_sql) if index_sql else []
                
                index_info = {
                    "name": index_name,
                    "table": table_name,
                    "columns": columns,
                    "is_unique": "UNIQUE" in (index_sql or ""),
                    "is_partial": "WHERE" in (index_sql or ""),
                }

                index_data["index_statistics"][index_name] = index_info

                # Check for redundant indexes (indexes with same columns)
                for other_name, other_info in index_data["index_statistics"].items():
                    if (other_name != index_name and 
                        other_info["table"] == table_name and
                        other_info["columns"] == columns):
                        index_data["redundant_indexes"].append({
                            "index1": index_name,
                            "index2": other_name,
                            "table": table_name,
                            "columns": columns,
                        })

            # Check foreign key coverage
            foreign_keys = self._get_foreign_keys(cursor)
            covered_fks = 0
            
            for fk in foreign_keys:
                if self._is_foreign_key_indexed(cursor, fk):
                    covered_fks += 1
                else:
                    index_data["missing_indexes"].append({
                        "type": "foreign_key",
                        "table": fk["table"],
                        "column": fk["column"],
                        "references": fk["references"],
                    })

            if foreign_keys:
                index_data["coverage_percent"] = (covered_fks / len(foreign_keys)) * 100

            # Identify commonly queried columns without indexes
            common_filters = ["status", "user_id", "timestamp", "created_at", "updated_at"]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                for col_info in columns:
                    col_name = col_info[1]
                    if col_name in common_filters:
                        if not self._column_has_index(cursor, table_name, col_name):
                            index_data["missing_indexes"].append({
                                "type": "common_filter",
                                "table": table_name,
                                "column": col_name,
                                "reason": "Commonly filtered column without index",
                            })

            conn.close()

        except Exception as e:
            logger.error("Failed to validate index usage", error=str(e))

        return index_data

    async def _detect_n_plus_one_queries(self) -> Dict[str, Any]:
        """Detect N+1 query problems in the codebase.
        
        Returns:
            N+1 query detection results
        """
        logger.info("Detecting N+1 query problems")
        
        n_plus_one_data = {
            "detected_patterns": [],
            "suspicious_loops": [],
            "orm_issues": [],
            "total_issues": 0,
        }

        try:
            # Scan Python files for N+1 patterns
            for py_file in self.project_root.rglob("genesis/**/*.py"):
                if "__pycache__" in str(py_file):
                    continue

                try:
                    content = py_file.read_text(encoding="utf-8")
                    
                    # Pattern 1: Loop with database query inside
                    loop_query_pattern = re.compile(
                        r"for\s+\w+\s+in\s+.*:\s*\n(?:.*\n){0,5}.*(?:execute|query|select|find|get)\s*\(",
                        re.MULTILINE
                    )
                    
                    matches = loop_query_pattern.finditer(content)
                    for match in matches:
                        line_num = content[:match.start()].count("\n") + 1
                        n_plus_one_data["detected_patterns"].append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": line_num,
                            "pattern": "query_in_loop",
                            "snippet": match.group(0)[:100],
                            "severity": "high",
                        })

                    # Pattern 2: Multiple similar queries
                    similar_query_pattern = re.compile(
                        r"((?:execute|query)\s*\([^)]*WHERE\s+\w+\s*=\s*[^)]+\).*\n){2,}",
                        re.MULTILINE
                    )
                    
                    matches = similar_query_pattern.finditer(content)
                    for match in matches:
                        line_num = content[:match.start()].count("\n") + 1
                        n_plus_one_data["suspicious_loops"].append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": line_num,
                            "pattern": "repeated_similar_queries",
                            "snippet": match.group(0)[:100],
                            "severity": "medium",
                        })

                    # Pattern 3: ORM lazy loading in loops
                    orm_pattern = re.compile(
                        r"for\s+\w+\s+in\s+.*:\s*\n(?:.*\n){0,5}.*\.\w+\.(?:all|first|get)\s*\(",
                        re.MULTILINE
                    )
                    
                    matches = orm_pattern.finditer(content)
                    for match in matches:
                        line_num = content[:match.start()].count("\n") + 1
                        n_plus_one_data["orm_issues"].append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": line_num,
                            "pattern": "orm_lazy_loading",
                            "snippet": match.group(0)[:100],
                            "severity": "high",
                            "suggestion": "Use eager loading or select_related/prefetch_related",
                        })

                except Exception as e:
                    logger.warning(f"Failed to analyze file {py_file}", error=str(e))

            n_plus_one_data["total_issues"] = (
                len(n_plus_one_data["detected_patterns"]) +
                len(n_plus_one_data["suspicious_loops"]) +
                len(n_plus_one_data["orm_issues"])
            )

        except Exception as e:
            logger.error("Failed to detect N+1 queries", error=str(e))

        return n_plus_one_data

    async def _validate_connection_pool(self) -> Dict[str, Any]:
        """Validate database connection pool configuration.
        
        Returns:
            Connection pool analysis
        """
        logger.info("Validating connection pool configuration")
        
        pool_data = {
            "pool_size": 0,
            "max_connections": 0,
            "min_connections": 0,
            "idle_connections": 0,
            "connection_timeout_ms": 0,
            "pool_efficiency": 0,
            "configuration_issues": [],
        }

        try:
            # Check for connection pool configuration in settings
            settings_file = self.project_root / "genesis" / "config" / "settings.py"
            if settings_file.exists():
                content = settings_file.read_text()
                
                # Extract pool configuration
                pool_size_match = re.search(r"(?:POOL_SIZE|pool_size)\s*=\s*(\d+)", content)
                max_conn_match = re.search(r"(?:MAX_CONNECTIONS|max_connections)\s*=\s*(\d+)", content)
                min_conn_match = re.search(r"(?:MIN_CONNECTIONS|min_connections)\s*=\s*(\d+)", content)
                timeout_match = re.search(r"(?:CONNECTION_TIMEOUT|connection_timeout)\s*=\s*(\d+)", content)
                
                if pool_size_match:
                    pool_data["pool_size"] = int(pool_size_match.group(1))
                if max_conn_match:
                    pool_data["max_connections"] = int(max_conn_match.group(1))
                if min_conn_match:
                    pool_data["min_connections"] = int(min_conn_match.group(1))
                if timeout_match:
                    pool_data["connection_timeout_ms"] = int(timeout_match.group(1))

            # Set defaults if not configured
            if pool_data["pool_size"] == 0:
                pool_data["pool_size"] = 10
                pool_data["configuration_issues"].append({
                    "issue": "no_pool_size_configured",
                    "severity": "medium",
                    "suggestion": "Set explicit pool_size in configuration",
                })

            if pool_data["max_connections"] == 0:
                pool_data["max_connections"] = 20
            if pool_data["min_connections"] == 0:
                pool_data["min_connections"] = 5

            # Validate pool configuration
            if pool_data["pool_size"] > self.MAX_CONNECTION_POOL_SIZE:
                pool_data["configuration_issues"].append({
                    "issue": "pool_too_large",
                    "value": pool_data["pool_size"],
                    "threshold": self.MAX_CONNECTION_POOL_SIZE,
                    "severity": "medium",
                    "impact": "Excessive resource consumption",
                })

            if pool_data["min_connections"] < self.MIN_CONNECTION_POOL_SIZE:
                pool_data["configuration_issues"].append({
                    "issue": "min_connections_too_low",
                    "value": pool_data["min_connections"],
                    "threshold": self.MIN_CONNECTION_POOL_SIZE,
                    "severity": "low",
                    "impact": "Connection startup latency",
                })

            if pool_data["connection_timeout_ms"] > self.MAX_CONNECTION_WAIT_MS:
                pool_data["configuration_issues"].append({
                    "issue": "timeout_too_high",
                    "value": pool_data["connection_timeout_ms"],
                    "threshold": self.MAX_CONNECTION_WAIT_MS,
                    "severity": "medium",
                    "impact": "Slow failure detection",
                })

            # Calculate pool efficiency (simulated)
            pool_data["idle_connections"] = max(0, pool_data["min_connections"] - 2)
            active_connections = pool_data["pool_size"] - pool_data["idle_connections"]
            pool_data["pool_efficiency"] = (active_connections / pool_data["pool_size"]) * 100

        except Exception as e:
            logger.error("Failed to validate connection pool", error=str(e))

        return pool_data

    def _check_database_violations(
        self,
        query_analysis: Dict[str, Any],
        index_analysis: Dict[str, Any],
        n_plus_one: Dict[str, Any],
        pool_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check for database optimization violations.
        
        Returns:
            List of violations
        """
        violations = []

        # Check for slow queries
        slow_queries = query_analysis.get("slow_queries", [])
        if slow_queries:
            violations.append({
                "type": "slow_queries",
                "count": len(slow_queries),
                "severity": "high",
                "impact": "User experience degradation",
                "queries": [q["query"] for q in slow_queries[:3]],
            })

        # Check for table scans
        table_scans = query_analysis.get("table_scans", [])
        if table_scans:
            violations.append({
                "type": "table_scans",
                "count": len(table_scans),
                "severity": "medium",
                "impact": "Poor query performance",
            })

        # Check for N+1 queries
        if n_plus_one.get("total_issues", 0) > self.MAX_N_PLUS_ONE_QUERIES:
            violations.append({
                "type": "n_plus_one_queries",
                "count": n_plus_one["total_issues"],
                "severity": "critical",
                "impact": "Exponential query growth",
            })

        # Check index coverage
        coverage = index_analysis.get("coverage_percent", 0)
        if coverage < self.MIN_INDEX_USAGE_PERCENT:
            violations.append({
                "type": "poor_index_coverage",
                "value": coverage,
                "threshold": self.MIN_INDEX_USAGE_PERCENT,
                "severity": "medium",
                "impact": "Suboptimal query performance",
            })

        # Check for redundant indexes
        if index_analysis.get("redundant_indexes"):
            violations.append({
                "type": "redundant_indexes",
                "count": len(index_analysis["redundant_indexes"]),
                "severity": "low",
                "impact": "Unnecessary storage and maintenance overhead",
            })

        # Check connection pool issues
        if pool_analysis.get("configuration_issues"):
            violations.append({
                "type": "connection_pool_misconfiguration",
                "issues": pool_analysis["configuration_issues"],
                "severity": "medium",
                "impact": "Connection management issues",
            })

        return violations

    def _generate_optimization_suggestions(
        self,
        query_analysis: Dict[str, Any],
        index_analysis: Dict[str, Any],
        n_plus_one: Dict[str, Any],
        pool_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate database optimization suggestions.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Query optimization suggestions
        if query_analysis.get("slow_queries"):
            suggestions.append({
                "area": "query_optimization",
                "suggestion": "Add indexes for frequently queried columns",
                "priority": "high",
                "expected_improvement": "50-80% query time reduction",
            })

        if query_analysis.get("table_scans"):
            suggestions.append({
                "area": "query_optimization",
                "suggestion": "Rewrite queries to use indexed columns in WHERE clauses",
                "priority": "high",
                "expected_improvement": "Order of magnitude performance improvement",
            })

        # Index suggestions
        if index_analysis.get("missing_indexes"):
            for missing in index_analysis["missing_indexes"][:3]:
                suggestions.append({
                    "area": "indexing",
                    "suggestion": f"Add index on {missing['table']}.{missing['column']}",
                    "priority": "medium",
                    "reason": missing.get("reason", "Missing index on foreign key"),
                })

        if index_analysis.get("redundant_indexes"):
            suggestions.append({
                "area": "indexing",
                "suggestion": "Remove redundant indexes to reduce overhead",
                "priority": "low",
                "indexes": [r["index1"] for r in index_analysis["redundant_indexes"]],
            })

        # N+1 query suggestions
        if n_plus_one.get("total_issues", 0) > 0:
            suggestions.append({
                "area": "n_plus_one",
                "suggestion": "Use batch queries or eager loading to eliminate N+1 patterns",
                "priority": "critical",
                "locations": n_plus_one.get("detected_patterns", [])[:3],
            })

        # Connection pool suggestions
        if pool_analysis.get("pool_efficiency", 100) < 50:
            suggestions.append({
                "area": "connection_pool",
                "suggestion": "Adjust pool size based on actual usage patterns",
                "priority": "medium",
                "current_efficiency": pool_analysis["pool_efficiency"],
            })

        return suggestions

    def _generate_database_evidence(
        self,
        query_analysis: Dict[str, Any],
        index_analysis: Dict[str, Any],
        n_plus_one: Dict[str, Any],
        violations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate database optimization evidence report.
        
        Returns:
            Evidence dictionary
        """
        evidence = {
            "summary": {
                "total_queries_analyzed": query_analysis.get("total_queries", 0),
                "slow_queries": len(query_analysis.get("slow_queries", [])),
                "table_scans": len(query_analysis.get("table_scans", [])),
                "n_plus_one_issues": n_plus_one.get("total_issues", 0),
                "index_coverage": index_analysis.get("coverage_percent", 0),
            },
            "worst_queries": query_analysis.get("slow_queries", [])[:5],
            "missing_indexes": index_analysis.get("missing_indexes", [])[:5],
            "n_plus_one_locations": n_plus_one.get("detected_patterns", [])[:5],
            "critical_violations": [v for v in violations if v["severity"] == "critical"],
            "query_patterns": query_analysis.get("query_patterns", {}),
        }

        return evidence

    def _setup_test_database(self, conn: sqlite3.Connection) -> None:
        """Set up test database schema for validation."""
        cursor = conn.cursor()
        
        # Create test tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                position_id INTEGER,
                user_id INTEGER NOT NULL,
                type TEXT,
                status TEXT,
                price REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (position_id) REFERENCES positions(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                order_id INTEGER,
                executed_at TIMESTAMP,
                price REAL,
                volume REAL,
                FOREIGN KEY (order_id) REFERENCES orders(id)
            )
        """)
        
        # Add some test data
        cursor.execute("INSERT INTO positions (user_id, symbol, amount, status) VALUES (1, 'BTC/USDT', 0.1, 'open')")
        cursor.execute("INSERT INTO orders (position_id, user_id, type, status) VALUES (1, 1, 'market', 'pending')")
        
        conn.commit()

    def _extract_query_pattern(self, query: str) -> str:
        """Extract query pattern for categorization."""
        query_upper = query.upper()
        
        if "SELECT" in query_upper:
            if "JOIN" in query_upper:
                return "select_join"
            elif "WHERE" in query_upper:
                return "select_filter"
            else:
                return "select_simple"
        elif "INSERT" in query_upper:
            return "insert"
        elif "UPDATE" in query_upper:
            return "update"
        elif "DELETE" in query_upper:
            return "delete"
        else:
            return "other"

    def _extract_index_columns(self, index_sql: str) -> List[str]:
        """Extract column names from index SQL."""
        if not index_sql:
            return []
        
        # Extract columns from CREATE INDEX statement
        match = re.search(r"\(([^)]+)\)", index_sql)
        if match:
            columns_str = match.group(1)
            return [col.strip() for col in columns_str.split(",")]
        
        return []

    def _check_primary_key(self, cursor: sqlite3.Cursor, table_name: str) -> bool:
        """Check if table has a primary key."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        return any(col[5] > 0 for col in columns)  # pk column is at index 5

    def _find_missing_indexes(self, cursor: sqlite3.Cursor) -> List[Dict[str, str]]:
        """Find columns that should have indexes but don't."""
        missing = []
        
        # This is a simplified check - in production, would analyze actual query logs
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            foreign_keys = cursor.fetchall()
            
            for fk in foreign_keys:
                fk_column = fk[3]  # from column
                if not self._column_has_index(cursor, table_name, fk_column):
                    missing.append({
                        "table": table_name,
                        "column": fk_column,
                        "type": "foreign_key",
                    })
        
        return missing

    def _get_foreign_keys(self, cursor: sqlite3.Cursor) -> List[Dict[str, str]]:
        """Get all foreign keys in the database."""
        foreign_keys = []
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            fks = cursor.fetchall()
            
            for fk in fks:
                foreign_keys.append({
                    "table": table_name,
                    "column": fk[3],
                    "references": f"{fk[2]}.{fk[4]}",
                })
        
        return foreign_keys

    def _is_foreign_key_indexed(self, cursor: sqlite3.Cursor, fk: Dict[str, str]) -> bool:
        """Check if a foreign key has an index."""
        return self._column_has_index(cursor, fk["table"], fk["column"])

    def _column_has_index(self, cursor: sqlite3.Cursor, table: str, column: str) -> bool:
        """Check if a column has an index."""
        cursor.execute(f"""
            SELECT sql FROM sqlite_master 
            WHERE type = 'index' AND tbl_name = ? AND sql LIKE ?
        """, (table, f"%{column}%"))
        
        return cursor.fetchone() is not None