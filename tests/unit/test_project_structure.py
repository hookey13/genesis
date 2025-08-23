"""
Unit tests for project structure and setup validation.
"""

from pathlib import Path


def test_project_directories_exist():
    """Test that all required project directories exist."""
    project_root = Path(__file__).parent.parent.parent

    required_dirs = [
        "genesis",
        "genesis/core",
        "genesis/engine",
        "genesis/tilt",
        "genesis/strategies",
        "genesis/strategies/sniper",
        "genesis/strategies/hunter",
        "genesis/strategies/strategist",
        "genesis/exchange",
        "genesis/data",
        "genesis/analytics",
        "genesis/ui",
        "genesis/ui/widgets",
        "genesis/ui/themes",
        "genesis/api",
        "genesis/utils",
        "genesis/risk",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "config",
        "scripts",
        "docker",
        "docs",
        "alembic",
        "alembic/versions",
        "requirements",
    ]

    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} does not exist"
        assert dir_path.is_dir(), f"{dir_name} is not a directory"


def test_python_packages_have_init_files():
    """Test that all Python packages have __init__.py files."""
    project_root = Path(__file__).parent.parent.parent

    python_packages = [
        "genesis",
        "genesis/core",
        "genesis/engine",
        "genesis/engine/executor",
        "genesis/tilt",
        "genesis/tilt/indicators",
        "genesis/strategies",
        "genesis/strategies/sniper",
        "genesis/strategies/hunter",
        "genesis/strategies/strategist",
        "genesis/exchange",
        "genesis/data",
        "genesis/analytics",
        "genesis/ui",
        "genesis/ui/widgets",
        "genesis/ui/themes",
        "genesis/api",
        "genesis/utils",
        "genesis/risk",
        "config",
        "tests",
    ]

    for package in python_packages:
        init_file = project_root / package / "__init__.py"
        assert init_file.exists(), f"Missing __init__.py in {package}"
        assert init_file.is_file(), f"__init__.py in {package} is not a file"


def test_configuration_files_exist():
    """Test that all configuration files exist."""
    project_root = Path(__file__).parent.parent.parent

    config_files = [
        ".gitignore",
        ".env.example",
        "requirements.txt",
        "requirements/base.txt",
        "requirements/sniper.txt",
        "requirements/hunter.txt",
        "requirements/strategist.txt",
        "requirements/dev.txt",
        "Makefile",
        "README.md",
        "pyproject.toml",
        ".pre-commit-config.yaml",
        ".secrets.baseline",
        "config/settings.py",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "docker/docker-compose.prod.yml",
        "docker/supervisord.conf",
    ]

    for file_name in config_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"Configuration file {file_name} does not exist"
        assert file_path.is_file(), f"{file_name} is not a file"


def test_gitignore_contains_critical_entries():
    """Test that .gitignore contains critical security entries."""
    project_root = Path(__file__).parent.parent.parent
    gitignore_path = project_root / ".gitignore"

    assert gitignore_path.exists(), ".gitignore file does not exist"

    gitignore_content = gitignore_path.read_text()

    critical_entries = [
        ".env",
        "*.key",
        "*.pem",
        "*.secret",
        "api_keys/",
        "secrets/",
        "credentials/",
        ".genesis/",
        "__pycache__/",
        "*.py[cod]",  # Covers *.pyc, *.pyo, *.pyd
        "venv/",
        ".coverage",
    ]

    for entry in critical_entries:
        assert entry in gitignore_content, f".gitignore missing critical entry: {entry}"


def test_env_example_contains_required_variables():
    """Test that .env.example contains all required configuration variables."""
    project_root = Path(__file__).parent.parent.parent
    env_example_path = project_root / ".env.example"

    assert env_example_path.exists(), ".env.example file does not exist"

    env_content = env_example_path.read_text()

    required_vars = [
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "BINANCE_TESTNET",
        "TRADING_TIER",
        "MAX_POSITION_SIZE_USDT",
        "MAX_DAILY_LOSS_USDT",
        "DATABASE_URL",
        "LOG_LEVEL",
        "API_SECRET_KEY",
        "DEPLOYMENT_ENV",
        "TILT_CLICK_SPEED_THRESHOLD",
        "TILT_CANCEL_RATE_THRESHOLD",
    ]

    for var in required_vars:
        assert var in env_content, f".env.example missing required variable: {var}"


def test_makefile_targets():
    """Test that Makefile contains all required targets."""
    project_root = Path(__file__).parent.parent.parent
    makefile_path = project_root / "Makefile"

    assert makefile_path.exists(), "Makefile does not exist"

    makefile_content = makefile_path.read_text()

    required_targets = [
        "help:",
        "setup:",
        "install:",
        "install-dev:",
        "test:",
        "test-unit:",
        "test-integration:",
        "test-coverage:",
        "format:",
        "lint:",
        "typecheck:",
        "run:",
        "run-docker:",
        "build-docker:",
        "deploy:",
        "backup:",
        "migrate:",
        "clean:",
        "pre-commit:",
    ]

    for target in required_targets:
        assert target in makefile_content, f"Makefile missing target: {target}"


def test_dockerfile_configuration():
    """Test that Dockerfile is properly configured."""
    project_root = Path(__file__).parent.parent.parent
    dockerfile_path = project_root / "docker" / "Dockerfile"

    assert dockerfile_path.exists(), "Dockerfile does not exist"

    dockerfile_content = dockerfile_path.read_text()

    # Check for critical configurations
    assert (
        "FROM python:3.11.8" in dockerfile_content
    ), "Dockerfile not using Python 3.11.8"
    assert "WORKDIR /app" in dockerfile_content, "Dockerfile missing WORKDIR"
    assert "genesis" in dockerfile_content, "Dockerfile missing genesis user creation"
    assert (
        "requirements.txt" in dockerfile_content
    ), "Dockerfile not installing requirements"
    assert "HEALTHCHECK" in dockerfile_content, "Dockerfile missing health check"


def test_docker_compose_files():
    """Test that docker-compose files are properly configured."""
    project_root = Path(__file__).parent.parent.parent

    # Check development docker-compose
    dev_compose = project_root / "docker" / "docker-compose.yml"
    assert dev_compose.exists(), "docker-compose.yml does not exist"

    dev_content = dev_compose.read_text()
    assert "genesis:" in dev_content, "docker-compose.yml missing genesis service"
    assert "volumes:" in dev_content, "docker-compose.yml missing volumes"
    assert "networks:" in dev_content, "docker-compose.yml missing networks"

    # Check production docker-compose
    prod_compose = project_root / "docker" / "docker-compose.prod.yml"
    assert prod_compose.exists(), "docker-compose.prod.yml does not exist"

    prod_content = prod_compose.read_text()
    assert "genesis:" in prod_content, "docker-compose.prod.yml missing genesis service"
    assert "redis:" in prod_content, "docker-compose.prod.yml missing redis service"
    assert (
        "postgres:" in prod_content
    ), "docker-compose.prod.yml missing postgres service"
    assert (
        "restart: always" in prod_content
    ), "Production compose missing restart policy"


def test_requirements_files_structure():
    """Test that requirements files are properly structured."""
    project_root = Path(__file__).parent.parent.parent

    # Check base.txt
    base_req = project_root / "requirements" / "base.txt"
    assert base_req.exists(), "requirements/base.txt does not exist"

    base_content = base_req.read_text()
    required_packages = [
        "ccxt==4.2.25",
        "rich==13.7.0",
        "textual==0.47.1",
        "aiohttp==3.9.3",
        "websockets==12.0",
        "pydantic==2.5.3",
        "structlog==24.1.0",
        "pandas==2.2.0",
        "numpy==1.26.3",
    ]

    for package in required_packages:
        assert package in base_content, f"base.txt missing required package: {package}"

    # Check dev.txt
    dev_req = project_root / "requirements" / "dev.txt"
    assert dev_req.exists(), "requirements/dev.txt does not exist"

    dev_content = dev_req.read_text()
    dev_packages = [
        "pytest==8.0.0",
        "black==24.1.1",
        "ruff==0.1.14",
        "mypy==1.8.0",
        "pre-commit==3.6.0",
    ]

    for package in dev_packages:
        assert package in dev_content, f"dev.txt missing required package: {package}"

    # Check tier requirements reference base.txt
    sniper_req = project_root / "requirements" / "sniper.txt"
    assert sniper_req.exists(), "requirements/sniper.txt does not exist"
    assert (
        "-r base.txt" in sniper_req.read_text()
    ), "sniper.txt not referencing base.txt"

    hunter_req = project_root / "requirements" / "hunter.txt"
    assert hunter_req.exists(), "requirements/hunter.txt does not exist"
    assert (
        "-r base.txt" in hunter_req.read_text()
    ), "hunter.txt not referencing base.txt"


def test_pyproject_toml_configuration():
    """Test that pyproject.toml is properly configured."""
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"

    assert pyproject_path.exists(), "pyproject.toml does not exist"

    pyproject_content = pyproject_path.read_text()

    # Check for tool configurations
    assert (
        "[tool.ruff]" in pyproject_content
    ), "pyproject.toml missing ruff configuration"
    assert (
        "[tool.black]" in pyproject_content
    ), "pyproject.toml missing black configuration"
    assert (
        "[tool.mypy]" in pyproject_content
    ), "pyproject.toml missing mypy configuration"
    assert (
        "[tool.pytest.ini_options]" in pyproject_content
    ), "pyproject.toml missing pytest configuration"
    assert (
        "[tool.coverage.run]" in pyproject_content
    ), "pyproject.toml missing coverage configuration"

    # Check specific settings
    assert (
        "line-length = 88" in pyproject_content
    ), "Incorrect line length configuration"
    assert (
        'target-version = "py311"' in pyproject_content
    ), "Incorrect Python version target"
    assert (
        'python_version = "3.11"' in pyproject_content
    ), "Incorrect mypy Python version"
