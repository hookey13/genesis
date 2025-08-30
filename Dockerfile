# Multi-stage Dockerfile for Genesis Trading System
# Optimized for minimal production image size (<500MB target)

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.11.8-slim as builder

# Set build arguments for tier selection
ARG TIER=sniper
ENV TIER=${TIER}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy dependency files first for better caching
COPY requirements/ requirements/
COPY pyproject.toml poetry.lock* ./

# Install pip-tools and create virtual environment
RUN python -m pip install --upgrade pip pip-tools && \
    python -m venv /opt/venv

# Activate virtual environment and install dependencies
ENV PATH="/opt/venv/bin:$PATH"

# Install tier-specific dependencies
RUN if [ -f requirements/${TIER}.txt ]; then \
        pip install --no-cache-dir -r requirements/${TIER}.txt; \
    else \
        pip install --no-cache-dir -r requirements/base.txt; \
    fi

# Install Poetry dependencies if lock file exists
RUN if [ -f poetry.lock ]; then \
        pip install poetry && \
        poetry config virtualenvs.create false && \
        poetry install --no-dev --no-interaction --no-ansi; \
    fi

# Generate Software Bill of Materials (SBOM)
RUN pip install pip-audit && \
    pip list --format=json > /build/sbom-packages.json && \
    pip-audit --format json --output /build/sbom-vulnerabilities.json || true

# ==============================================================================
# Stage 2: Production
# ==============================================================================
FROM python:3.11.8-slim as production

# Metadata
LABEL maintainer="Genesis Team"
LABEL version="1.0.0"
LABEL description="Genesis Trading System - Production Image"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    GENESIS_ENV=production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libffi8 \
    libxml2 \
    libxslt1.1 \
    libgomp1 \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user (UID 1000)
RUN groupadd -r genesis -g 1000 && \
    useradd -r -u 1000 -g genesis -m -d /home/genesis -s /bin/bash genesis

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=genesis:genesis /opt/venv /opt/venv

# Copy SBOM files from builder
COPY --from=builder --chown=genesis:genesis /build/sbom-*.json /app/

# Copy application code
COPY --chown=genesis:genesis genesis/ genesis/
COPY --chown=genesis:genesis config/ config/
COPY --chown=genesis:genesis scripts/emergency_close.py scripts/
COPY --chown=genesis:genesis alembic/ alembic/
COPY --chown=genesis:genesis alembic.ini .

# Create necessary directories
RUN mkdir -p /app/.genesis/data /app/.genesis/logs /app/.genesis/state && \
    chown -R genesis:genesis /app/.genesis

# Activate virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import genesis; print('healthy')" || exit 1

# Switch to non-root user
USER genesis

# Expose Prometheus metrics port
EXPOSE 9090

# Use tini as PID 1 to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["python", "-m", "genesis"]

# ==============================================================================
# Stage 3: Development (optional)
# ==============================================================================
FROM production as development

# Switch back to root for dev tools installation
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    htop \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install dev dependencies
COPY --chown=genesis:genesis requirements/dev.txt requirements/
RUN pip install --no-cache-dir -r requirements/dev.txt

# Install pre-commit hooks
COPY --chown=genesis:genesis .pre-commit-config.yaml .
RUN pre-commit install || true

# Copy test files
COPY --chown=genesis:genesis tests/ tests/
COPY --chown=genesis:genesis pytest.ini .

# Switch back to genesis user
USER genesis

# Development command with hot reload
CMD ["python", "-m", "genesis", "--dev-mode"]

# ==============================================================================
# Stage 4: Testing
# ==============================================================================
FROM development as testing

# Copy all source code for testing
COPY --chown=genesis:genesis . .

# Run tests
RUN python -m pytest tests/unit/ -v --tb=short

# Security scan
RUN pip install safety bandit && \
    safety check || true && \
    bandit -r genesis/ -ll || true

# Default to running all tests
CMD ["pytest", "tests/", "-v", "--cov=genesis", "--cov-report=term-missing"]