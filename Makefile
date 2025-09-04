# Project GENESIS Makefile
# Common commands for development and deployment

.PHONY: help install install-dev test test-unit test-integration test-coverage \
        format lint typecheck clean run run-docker deploy backup migrate \
        build-docker stop-docker logs pre-commit setup validate validate-quick \
        validate-security validate-performance smoke-test

# Default target
help:
	@echo "Project GENESIS - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Complete initial setup (venv, deps, pre-commit)"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run            - Run the application locally"
	@echo "  make run-docker     - Run the application in Docker"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-coverage  - Run tests with coverage report"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Run ruff linter"
	@echo "  make typecheck      - Run mypy type checking"
	@echo "  make pre-commit     - Run all pre-commit hooks"
	@echo ""
	@echo "Docker:"
	@echo "  make build-docker   - Build Docker image"
	@echo "  make stop-docker    - Stop Docker containers"
	@echo "  make logs           - View Docker logs"
	@echo ""
	@echo "Database:"
	@echo "  make migrate        - Run database migrations"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy         - Deploy to production"
	@echo "  make backup         - Run backup script"
	@echo ""
	@echo "Validation & Testing:"
	@echo "  make validate       - Run full production validation"
	@echo "  make validate-quick - Run quick smoke tests only"
	@echo "  make validate-security - Run security-focused validation"
	@echo "  make validate-performance - Run performance benchmarks"
	@echo "  make smoke-test     - Run smoke test suite"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove Python cache files"

# Setup & Installation
setup: install-dev pre-commit
	@echo "✓ Setup complete! Copy .env.example to .env and configure your settings."

install:
	@echo "Installing production dependencies..."
	@pip install -r requirements.txt
	@echo "✓ Production dependencies installed"

install-dev:
	@echo "Installing development dependencies..."
	@pip install -r requirements.txt
	@pip install -r requirements/dev.txt
	@echo "✓ Development dependencies installed"

# Testing
test:
	@echo "Running all tests..."
	@pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	@pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	@pytest tests/integration/ -v

test-coverage:
	@echo "Running tests with coverage..."
	@pytest tests/ --cov=genesis --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

# Code Quality
format:
	@echo "Formatting code with black..."
	@black genesis/ tests/ config/ scripts/ --line-length 88
	@echo "✓ Code formatted"

lint:
	@echo "Running ruff linter..."
	@ruff check genesis/ tests/ config/ scripts/
	@echo "✓ Linting complete"

typecheck:
	@echo "Running mypy type checker..."
	@mypy genesis/ config/ --ignore-missing-imports
	@echo "✓ Type checking complete"

# Pre-commit
pre-commit:
	@echo "Installing pre-commit hooks..."
	@pre-commit install
	@echo "Running pre-commit hooks..."
	@pre-commit run --all-files || true
	@echo "✓ Pre-commit hooks configured"

# Running the Application
doctor:
	@echo "Running Genesis Doctor health checks..."
	@python -m genesis.cli doctor

run:
	@echo "Starting Project GENESIS..."
	@python -m genesis

run-docker:
	@echo "Starting Project GENESIS in Docker..."
	@docker-compose -f docker/docker-compose.yml up

# Docker Commands
build-docker:
	@echo "Building Docker image..."
	@docker-compose -f docker/docker-compose.yml build
	@echo "✓ Docker image built"

stop-docker:
	@echo "Stopping Docker containers..."
	@docker-compose -f docker/docker-compose.yml down
	@echo "✓ Docker containers stopped"

logs:
	@docker-compose -f docker/docker-compose.yml logs -f

# Database
migrate:
	@echo "Running database migrations..."
	@alembic upgrade head
	@echo "✓ Migrations complete"

# Deployment
deploy:
	@echo "Deploying to production..."
	@bash scripts/deploy.sh
	@echo "✓ Deployment complete"

# Backup
backup:
	@echo "Running backup..."
	@bash scripts/backup.sh
	@echo "✓ Backup complete"

# Validation Commands
validate:
	@echo "Running full production validation..."
	@python scripts/validate_production.py --mode standard --format console
	@echo "✓ Validation complete"

validate-quick:
	@echo "Running quick validation (smoke tests)..."
	@python scripts/validate_production.py --mode quick --format console
	@echo "✓ Quick validation complete"

validate-security:
	@echo "Running security validation..."
	@python scripts/validate_production.py --mode security --format console
	@echo "✓ Security validation complete"

validate-performance:
	@echo "Running performance validation..."
	@python scripts/validate_production.py --mode performance --format console
	@echo "✓ Performance validation complete"

smoke-test:
	@echo "Running smoke test suite..."
	@python scripts/smoke_tests.py
	@echo "✓ Smoke tests complete"

# Maintenance
clean:
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cache files cleaned"