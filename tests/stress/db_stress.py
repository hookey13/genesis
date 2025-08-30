"""
Database stress testing suite for performance validation.

Tests database performance with 1M+ records and various query patterns.
"""

import asyncio
import sqlite3
import time
import random
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DBMetrics:
    """Database performance metrics."""
    
    total_records: int = 0
    insert_time_seconds: float = 0
    query_times_ms: List[float] = None
    index_effectiveness: float = 0
    backup_time_seconds: float = 0
    restore_time_seconds: float = 0
    
    def __post_init__(self):
        if self.query_times_ms is None:
            self.query_times_ms = []


class DatabaseStressTest:
    """Database stress testing suite."""
    
    def __init__(self, db_path: str = "test_stress.db"):
        self.db_path = db_path
        self.metrics = DBMetrics()
        self.conn = None
        
    async def run_stress_test(self, num_records: int = 1000000):
        """Run complete database stress test."""
        logger.info(f"Starting database stress test with {num_records:,} records")
        
        try:
            # Setup database
            await self.setup_database()
            
            # Generate and insert data
            await self.populate_database(num_records)
            
            # Run query benchmarks
            await self.benchmark_queries()
            
            # Test index effectiveness
            await self.test_index_effectiveness()
            
            # Test backup/restore
            await self.test_backup_restore()
            
            # Generate report
            await self.generate_report()
            
        finally:
            if self.conn:
                self.conn.close()
    
    async def setup_database(self):
        """Setup database schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        
        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                fee REAL NOT NULL,
                pnl REAL
            );
            
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                amount REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                unrealized_pnl REAL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_order_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                type TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
        """)
        
        self.conn.commit()
    
    async def populate_database(self, num_records: int):
        """Populate database with test data."""
        logger.info("Populating database...")
        
        start_time = time.time()
        batch_size = 10000
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "MATIC/USDT"]
        
        for i in range(0, num_records, batch_size):
            trades = []
            orders = []
            
            for j in range(min(batch_size, num_records - i)):
                timestamp = int(time.time() * 1000) - random.randint(0, 86400000)
                
                # Generate trade
                trades.append((
                    timestamp,
                    random.choice(symbols),
                    random.choice(["buy", "sell"]),
                    random.uniform(100, 50000),
                    random.uniform(0.001, 10),
                    random.uniform(0.0001, 0.01),
                    random.uniform(-100, 100)
                ))
                
                # Generate order
                orders.append((
                    f"order_{i+j}",
                    random.choice(symbols),
                    random.choice(["limit", "market"]),
                    random.choice(["buy", "sell"]),
                    random.uniform(100, 50000) if random.random() > 0.3 else None,
                    random.uniform(0.001, 10),
                    random.choice(["open", "filled", "cancelled"]),
                    timestamp
                ))
            
            # Bulk insert
            self.conn.executemany(
                "INSERT INTO trades (timestamp, symbol, side, price, amount, fee, pnl) VALUES (?, ?, ?, ?, ?, ?, ?)",
                trades
            )
            self.conn.executemany(
                "INSERT INTO orders (client_order_id, symbol, type, side, price, amount, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                orders
            )
            
            if (i + batch_size) % 100000 == 0:
                logger.info(f"Inserted {i + batch_size:,} records...")
                self.conn.commit()
        
        self.conn.commit()
        self.metrics.total_records = num_records
        self.metrics.insert_time_seconds = time.time() - start_time
        
        logger.info(f"Populated {num_records:,} records in {self.metrics.insert_time_seconds:.2f} seconds")
    
    async def benchmark_queries(self):
        """Benchmark various query types."""
        logger.info("Benchmarking queries...")
        
        queries = [
            ("Simple SELECT", "SELECT * FROM trades LIMIT 100"),
            ("Aggregation", "SELECT symbol, COUNT(*), AVG(price) FROM trades GROUP BY symbol"),
            ("Range query", "SELECT * FROM trades WHERE timestamp > ? LIMIT 1000", (int(time.time() * 1000) - 3600000,)),
            ("JOIN query", "SELECT t.*, o.status FROM trades t JOIN orders o ON t.symbol = o.symbol LIMIT 100"),
            ("Complex filter", "SELECT * FROM trades WHERE symbol = ? AND side = ? AND price > ? LIMIT 100", ("BTC/USDT", "buy", 40000))
        ]
        
        for name, query, *params in queries:
            start = time.time()
            cursor = self.conn.execute(query, params[0] if params else ())
            cursor.fetchall()
            elapsed_ms = (time.time() - start) * 1000
            
            self.metrics.query_times_ms.append(elapsed_ms)
            logger.info(f"{name}: {elapsed_ms:.2f}ms")
    
    async def test_index_effectiveness(self):
        """Test index effectiveness."""
        logger.info("Testing index effectiveness...")
        
        # Query with index
        start = time.time()
        self.conn.execute("SELECT * FROM trades WHERE symbol = 'BTC/USDT' LIMIT 1000").fetchall()
        with_index = time.time() - start
        
        # Query without index (on non-indexed column)
        start = time.time()
        self.conn.execute("SELECT * FROM trades WHERE pnl > 50 LIMIT 1000").fetchall()
        without_index = time.time() - start
        
        self.metrics.index_effectiveness = (without_index - with_index) / without_index * 100 if without_index > 0 else 0
        
        logger.info(f"Index effectiveness: {self.metrics.index_effectiveness:.1f}% improvement")
    
    async def test_backup_restore(self):
        """Test backup and restore performance."""
        logger.info("Testing backup/restore...")
        
        backup_path = Path("test_backup.db")
        
        # Backup
        start = time.time()
        backup_conn = sqlite3.connect(str(backup_path))
        self.conn.backup(backup_conn)
        backup_conn.close()
        self.metrics.backup_time_seconds = time.time() - start
        
        # Restore
        start = time.time()
        restore_conn = sqlite3.connect(":memory:")
        backup_conn = sqlite3.connect(str(backup_path))
        backup_conn.backup(restore_conn)
        backup_conn.close()
        restore_conn.close()
        self.metrics.restore_time_seconds = time.time() - start
        
        # Cleanup
        backup_path.unlink(missing_ok=True)
        
        logger.info(f"Backup: {self.metrics.backup_time_seconds:.2f}s, Restore: {self.metrics.restore_time_seconds:.2f}s")
    
    async def generate_report(self):
        """Generate stress test report."""
        import json
        
        report_path = Path("tests/stress/reports")
        report_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"db_stress_{timestamp}.json"
        
        avg_query_time = sum(self.metrics.query_times_ms) / len(self.metrics.query_times_ms) if self.metrics.query_times_ms else 0
        
        report = {
            "test_type": "database_stress",
            "metrics": {
                "total_records": self.metrics.total_records,
                "insert_time_seconds": self.metrics.insert_time_seconds,
                "records_per_second": self.metrics.total_records / self.metrics.insert_time_seconds if self.metrics.insert_time_seconds > 0 else 0,
                "average_query_time_ms": avg_query_time,
                "max_query_time_ms": max(self.metrics.query_times_ms) if self.metrics.query_times_ms else 0,
                "index_effectiveness_percent": self.metrics.index_effectiveness,
                "backup_time_seconds": self.metrics.backup_time_seconds,
                "restore_time_seconds": self.metrics.restore_time_seconds
            },
            "success": avg_query_time < 100  # Queries should be < 100ms
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report generated: {report_file}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("DATABASE STRESS TEST SUMMARY")
        print("=" * 60)
        print(f"Total Records: {self.metrics.total_records:,}")
        print(f"Insert Rate: {report['metrics']['records_per_second']:.0f} records/sec")
        print(f"Average Query Time: {avg_query_time:.2f}ms")
        print(f"Max Query Time: {report['metrics']['max_query_time_ms']:.2f}ms")
        print(f"Index Effectiveness: {self.metrics.index_effectiveness:.1f}%")
        print(f"Backup Time: {self.metrics.backup_time_seconds:.2f}s")
        print(f"Test Result: {'PASS' if report['success'] else 'FAIL'}")
        print("=" * 60)


async def main():
    """Run database stress test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database stress testing")
    parser.add_argument(
        "--records",
        type=int,
        default=1000000,
        help="Number of records to generate (default: 1000000)"
    )
    
    args = parser.parse_args()
    
    test = DatabaseStressTest()
    
    try:
        await test.run_stress_test(num_records=args.records)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        Path("test_stress.db").unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())