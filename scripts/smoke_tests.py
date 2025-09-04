#!/usr/bin/env python3
"""
Smoke test suite for Genesis trading system.

Quick health checks to verify basic system functionality.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


class SmokeTestResult:
    """Result of a smoke test."""
    
    def __init__(self, name: str, passed: bool, duration: float, message: str = "", details: Dict = None):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "duration": self.duration,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class SmokeTestSuite:
    """Quick smoke tests for production validation."""
    
    def __init__(self):
        self.results: List[SmokeTestResult] = []
        self.start_time = None
        self.end_time = None
    
    async def run(self) -> bool:
        """Run all smoke tests."""
        print("\n" + "="*60)
        print("Genesis Smoke Test Suite")
        print("="*60 + "\n")
        
        self.start_time = time.time()
        
        # Run tests
        tests = [
            self.test_python_version,
            self.test_required_directories,
            self.test_configuration_files,
            self.test_database_connection,
            self.test_import_core_modules,
            self.test_environment_variables,
            self.test_log_directory,
            self.test_backup_directory,
            self.test_network_connectivity,
            self.test_critical_dependencies
        ]
        
        for test in tests:
            await self._run_test(test)
        
        self.end_time = time.time()
        
        # Print summary
        self._print_summary()
        
        # Return overall pass/fail
        return all(r.passed for r in self.results)
    
    async def _run_test(self, test_func):
        """Run a single test and record result."""
        test_name = test_func.__name__.replace("test_", "").replace("_", " ").title()
        print(f"Running: {test_name}...", end=" ")
        
        start = time.perf_counter()
        try:
            result = await test_func()
            duration = time.perf_counter() - start
            
            if result is None or result is True:
                print("✅ PASSED")
                self.results.append(SmokeTestResult(test_name, True, duration))
            else:
                print("❌ FAILED")
                message = result if isinstance(result, str) else "Test failed"
                self.results.append(SmokeTestResult(test_name, False, duration, message))
        except Exception as e:
            duration = time.perf_counter() - start
            print("❌ ERROR")
            self.results.append(SmokeTestResult(
                test_name,
                False,
                duration,
                str(e),
                {"exception": str(e)}
            ))
    
    async def test_python_version(self) -> bool:
        """Test Python version is 3.11.8."""
        version = sys.version_info
        if version.major == 3 and version.minor == 11:
            return True
        return f"Python {version.major}.{version.minor} (expected 3.11)"
    
    async def test_required_directories(self) -> bool:
        """Test required directories exist."""
        required_dirs = [
            "genesis",
            "scripts",
            "tests",
            "config",
            "alembic"
        ]
        
        missing = []
        for dir_name in required_dirs:
            if not Path(dir_name).exists():
                missing.append(dir_name)
        
        if missing:
            return f"Missing directories: {', '.join(missing)}"
        return True
    
    async def test_configuration_files(self) -> bool:
        """Test configuration files exist."""
        config_files = [
            "genesis/config/settings.py",
            "scripts/config/validation_criteria.yaml",
            ".env.example"
        ]
        
        missing = []
        for file_path in config_files:
            if not Path(file_path).exists():
                missing.append(file_path)
        
        if missing:
            return f"Missing config files: {', '.join(missing)}"
        return True
    
    async def test_database_connection(self) -> bool:
        """Test database connectivity."""
        import sqlite3
        
        db_path = Path(".genesis/data/genesis.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            conn.close()
            return True
        except Exception as e:
            return f"Database error: {e}"
    
    async def test_import_core_modules(self) -> bool:
        """Test core modules can be imported."""
        modules_to_test = [
            "genesis",
            "genesis.core",
            "genesis.config",
            "genesis.engine",
            "scripts.validators"
        ]
        
        failed_imports = []
        for module in modules_to_test:
            try:
                __import__(module)
            except ImportError as e:
                failed_imports.append(f"{module}: {e}")
        
        if failed_imports:
            return f"Import errors: {', '.join(failed_imports)}"
        return True
    
    async def test_environment_variables(self) -> bool:
        """Test critical environment variables."""
        # Check for .env file
        env_file = Path(".env")
        if not env_file.exists():
            # Check for environment variables
            critical_vars = [
                "BINANCE_API_KEY",
                "BINANCE_API_SECRET"
            ]
            
            missing = []
            for var in critical_vars:
                if not os.getenv(var):
                    missing.append(var)
            
            if missing:
                return f"Missing env vars (create .env file): {', '.join(missing)}"
        return True
    
    async def test_log_directory(self) -> bool:
        """Test log directory is writable."""
        log_dir = Path(".genesis/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = log_dir / "smoke_test.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception as e:
            return f"Log directory not writable: {e}"
    
    async def test_backup_directory(self) -> bool:
        """Test backup directory exists."""
        backup_dir = Path(".genesis/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        return True
    
    async def test_network_connectivity(self) -> bool:
        """Test network connectivity."""
        import socket
        
        try:
            # Test DNS resolution for Binance
            socket.gethostbyname("api.binance.com")
            return True
        except socket.gaierror:
            return "Cannot resolve api.binance.com"
        except Exception as e:
            return f"Network error: {e}"
    
    async def test_critical_dependencies(self) -> bool:
        """Test critical Python dependencies."""
        required_packages = [
            "ccxt",
            "pydantic",
            "structlog",
            "asyncio",
            "decimal",
            "yaml"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            return f"Missing packages: {', '.join(missing)}"
        return True
    
    def _print_summary(self):
        """Print test summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        duration = self.end_time - self.start_time
        
        print("\n" + "="*60)
        print("Smoke Test Summary")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Result: {'PASSED' if failed == 0 else 'FAILED'}")
        
        if failed > 0:
            print("\nFailed Tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.message}")
        
        print("="*60 + "\n")
    
    def save_results(self, output_file: str = None):
        """Save test results to file."""
        if not output_file:
            output_file = f".genesis/logs/smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_data = {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time,
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": sum(1 for r in self.results if not r.passed),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to: {output_path}")


async def main():
    """Run smoke test suite."""
    suite = SmokeTestSuite()
    success = await suite.run()
    
    # Save results
    suite.save_results()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())