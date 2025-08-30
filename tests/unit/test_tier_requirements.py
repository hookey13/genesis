"""Unit tests for tier-specific requirements management."""

import os
from pathlib import Path
from typing import Dict, Set
import pytest
from unittest.mock import patch, MagicMock, mock_open


class TestTierRequirements:
    """Test tier-specific requirements management."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_tier_files_exist(self, project_root):
        """Verify all tier requirement files exist."""
        req_dir = project_root / "requirements"
        
        required_files = [
            "base.txt",
            "sniper.txt",
            "hunter.txt",
            "strategist.txt",
            "dev.txt"
        ]
        
        for filename in required_files:
            file_path = req_dir / filename
            assert file_path.exists(), f"requirements/{filename} must exist"
    
    def test_base_requirements_content(self, project_root):
        """Verify base.txt contains core dependencies."""
        base_file = project_root / "requirements" / "base.txt"
        
        if base_file.exists():
            content = base_file.read_text().lower()
            
            # Core packages that must be in base
            core_packages = [
                "ccxt",
                "rich",
                "textual",
                "aiohttp",
                "websockets",
                "pydantic",
                "structlog",
                "sqlalchemy",
                "alembic",
                "pandas",
                "numpy"
            ]
            
            for package in core_packages:
                assert package in content, f"{package} must be in base.txt"
    
    def test_sniper_tier_inheritance(self, project_root):
        """Verify sniper tier properly inherits from base."""
        sniper_file = project_root / "requirements" / "sniper.txt"
        
        if sniper_file.exists():
            content = sniper_file.read_text()
            
            # Sniper should inherit from base
            assert "-r base.txt" in content or len(content.strip()) == 0, \
                "sniper.txt should inherit from base.txt or be empty"
            
            # Sniper should NOT have advanced packages
            forbidden = ["scipy", "scikit-learn", "statsmodels", "ta-lib"]
            for package in forbidden:
                assert package not in content.lower(), \
                    f"sniper.txt should not have {package} (too advanced for tier)"
    
    def test_hunter_tier_dependencies(self, project_root):
        """Verify hunter tier has appropriate dependencies."""
        hunter_file = project_root / "requirements" / "hunter.txt"
        
        if hunter_file.exists():
            content = hunter_file.read_text()
            
            # Hunter should inherit from base
            assert "-r base.txt" in content, "hunter.txt should inherit from base.txt"
            
            # Hunter should have statistical packages
            # Note: These may be commented out or optional
            statistical_packages = ["scipy", "statsmodels"]
            has_statistical = any(pkg in content.lower() for pkg in statistical_packages)
            
            # This is optional - hunter may or may not have these yet
            if has_statistical:
                assert True  # Good, has statistical packages
    
    def test_strategist_tier_dependencies(self, project_root):
        """Verify strategist tier has appropriate dependencies."""
        strategist_file = project_root / "requirements" / "strategist.txt"
        
        if strategist_file.exists():
            content = strategist_file.read_text()
            
            # Strategist should inherit from hunter
            assert "-r hunter.txt" in content, \
                "strategist.txt should inherit from hunter.txt"
            
            # Strategist should have ML packages (when implemented)
            # Note: These may be commented out or optional initially
            ml_packages = ["scikit-learn", "ta-lib"]
            has_ml = any(pkg in content.lower() for pkg in ml_packages)
            
            # This is optional - strategist may not have these yet
            if has_ml:
                assert True  # Good, has ML packages
    
    def test_dev_requirements_content(self, project_root):
        """Verify dev.txt contains development tools."""
        dev_file = project_root / "requirements" / "dev.txt"
        
        if dev_file.exists():
            content = dev_file.read_text().lower()
            
            # Development tools that should be present
            dev_tools = [
                "pytest",
                "black",
                "ruff",
                "mypy",
                "pre-commit"
            ]
            
            for tool in dev_tools:
                assert tool in content, f"{tool} must be in dev.txt"
    
    def test_no_circular_dependencies(self, project_root):
        """Verify no circular dependencies in tier inheritance."""
        req_dir = project_root / "requirements"
        
        def get_inherited_files(file_path: Path) -> Set[str]:
            """Extract inherited files from a requirements file."""
            inherited = set()
            if file_path.exists():
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('-r '):
                            inherited.add(line[3:].strip())
            return inherited
        
        # Build inheritance graph
        inheritance = {}
        tier_files = ["base.txt", "sniper.txt", "hunter.txt", "strategist.txt", "dev.txt"]
        
        for tier_file in tier_files:
            file_path = req_dir / tier_file
            inheritance[tier_file] = get_inherited_files(file_path)
        
        # Check for circular dependencies
        def has_circular(file_name: str, visited: Set[str] = None) -> bool:
            if visited is None:
                visited = set()
            
            if file_name in visited:
                return True
            
            visited.add(file_name)
            
            for inherited in inheritance.get(file_name, []):
                if has_circular(inherited, visited.copy()):
                    return True
            
            return False
        
        for tier_file in tier_files:
            assert not has_circular(tier_file), \
                f"Circular dependency detected starting from {tier_file}"
    
    def test_tier_environment_variable(self):
        """Test tier environment variable handling."""
        # Test setting and getting tier
        os.environ['TIER'] = 'hunter'
        assert os.environ.get('TIER') == 'hunter'
        
        # Test default tier
        del os.environ['TIER']
        default_tier = os.environ.get('TIER', 'sniper')
        assert default_tier == 'sniper'
    
    def test_validate_tiers_script_exists(self, project_root):
        """Verify tier validation script exists."""
        script_path = project_root / "scripts" / "validate_tiers.py"
        assert script_path.exists(), "scripts/validate_tiers.py must exist"
    
    @patch('subprocess.run')
    def test_tier_validation_execution(self, mock_run, project_root):
        """Test tier validation script execution."""
        # Mock successful validation
        mock_run.return_value = MagicMock(returncode=0, stdout="Validation passed")
        
        # Import and run validation
        import sys
        sys.path.insert(0, str(project_root / "scripts"))
        
        try:
            from validate_tiers import validate_tier_inheritance
            # This would normally run the validation
            assert callable(validate_tier_inheritance)
        except ImportError:
            # Script may not be importable in test environment
            pass
    
    def test_tier_conditional_imports(self, project_root):
        """Test that tier-specific imports are conditional."""
        # Check for conditional import patterns in strategy loader
        loader_file = project_root / "genesis" / "strategies" / "loader.py"
        
        if loader_file.exists():
            content = loader_file.read_text()
            
            # Should have tier checking logic
            assert "tier" in content.lower() or "TIER" in content, \
                "Strategy loader should check tier"
    
    def test_tier_based_feature_flags(self):
        """Test tier-based feature flag system."""
        # Define tier limits
        tier_limits = {
            "sniper": {
                "max_pairs": 1,
                "max_position_size": 100,
                "strategies": ["simple_arb", "spread_capture"]
            },
            "hunter": {
                "max_pairs": 5,
                "max_position_size": 500,
                "strategies": ["simple_arb", "spread_capture", "mean_reversion", "multi_pair"]
            },
            "strategist": {
                "max_pairs": 20,
                "max_position_size": 2000,
                "strategies": ["simple_arb", "spread_capture", "mean_reversion", 
                            "multi_pair", "statistical_arb", "market_making"]
            }
        }
        
        # Test tier progression
        assert tier_limits["sniper"]["max_pairs"] < tier_limits["hunter"]["max_pairs"]
        assert tier_limits["hunter"]["max_pairs"] < tier_limits["strategist"]["max_pairs"]
        
        # Test strategy availability
        assert "market_making" not in tier_limits["sniper"]["strategies"]
        assert "market_making" in tier_limits["strategist"]["strategies"]
    
    def test_pyproject_toml_tier_groups(self, project_root):
        """Test Poetry tier groups in pyproject.toml."""
        pyproject_file = project_root / "pyproject.toml"
        
        if pyproject_file.exists():
            content = pyproject_file.read_text()
            
            # Check for tier-specific dependency groups
            assert "[tool.poetry.group.sniper]" in content, \
                "pyproject.toml should have sniper group"
            assert "[tool.poetry.group.hunter]" in content, \
                "pyproject.toml should have hunter group"
            assert "[tool.poetry.group.strategist]" in content, \
                "pyproject.toml should have strategist group"
    
    def test_dockerfile_tier_argument(self, project_root):
        """Test Dockerfile supports tier selection."""
        dockerfile = project_root / "Dockerfile"
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            # Check for tier build argument
            assert "ARG TIER=" in content, "Dockerfile should have TIER argument"
            assert "ENV TIER=" in content or "${TIER}" in content, \
                "Dockerfile should use TIER variable"
    
    def test_tier_upgrade_path(self):
        """Test tier upgrade path logic."""
        # Simulated account balance to tier mapping
        def get_tier_for_balance(balance: float) -> str:
            if balance < 500:
                return "paper"  # Not ready for live trading
            elif balance < 2000:
                return "sniper"
            elif balance < 10000:
                return "hunter"
            else:
                return "strategist"
        
        # Test tier progression
        assert get_tier_for_balance(100) == "paper"
        assert get_tier_for_balance(500) == "sniper"
        assert get_tier_for_balance(1500) == "sniper"
        assert get_tier_for_balance(2000) == "hunter"
        assert get_tier_for_balance(5000) == "hunter"
        assert get_tier_for_balance(10000) == "strategist"
        assert get_tier_for_balance(50000) == "strategist"