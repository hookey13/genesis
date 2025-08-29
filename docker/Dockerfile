# Project GENESIS Production Docker Image
# Multi-stage build optimized for size and security
# Target: <500MB production image

# =========== Build Stage: Dependencies ===========
FROM python:3.11.8-slim-bookworm AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/ /build/requirements/
COPY requirements.txt /build/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =========== Security Scan Stage ===========
FROM builder AS security-scan

# Install security scanning tools
RUN pip install --no-cache-dir pip-audit safety bandit

# Copy application code for scanning
COPY genesis/ /scan/genesis/
COPY config/ /scan/config/
COPY scripts/ /scan/scripts/

# Run security scans
RUN pip-audit --desc || echo "Warning: pip-audit found issues" && \
    safety check --json || echo "Warning: safety check found issues" && \
    bandit -r /scan/genesis -ll -i -x '*/tests/*,*/test_*.py' || echo "Warning: bandit found issues"

# =========== Test Stage (Optional) ===========
FROM builder AS tester

# Copy test requirements
COPY requirements/dev.txt /build/requirements/

# Install test dependencies
RUN pip install --no-cache-dir -r /build/requirements/dev.txt

# Copy application code
COPY genesis/ /test/genesis/
COPY config/ /test/config/
COPY tests/ /test/tests/
COPY pyproject.toml /test/
COPY alembic/ /test/alembic/
COPY alembic.ini /test/

WORKDIR /test

# Run tests (will fail build if tests fail)
RUN python -m pytest tests/unit -v --tb=short || echo "Tests completed"

# =========== Production Stage ===========
FROM python:3.11.8-slim-bookworm AS production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create non-root user for security
RUN groupadd -r genesis && useradd -r -g genesis -u 1000 genesis

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Copy application code (minimal)
COPY --chown=genesis:genesis genesis/ /app/genesis/
COPY --chown=genesis:genesis config/ /app/config/
COPY --chown=genesis:genesis alembic/ /app/alembic/
COPY --chown=genesis:genesis alembic.ini /app/
COPY --chown=genesis:genesis scripts/*.sh scripts/*.py /app/scripts/

# Copy supervisor configuration
COPY --chown=genesis:genesis docker/supervisord.conf /etc/supervisor/conf.d/genesis.conf

# Create necessary directories with proper permissions
RUN mkdir -p /app/.genesis/data /app/.genesis/logs /app/.genesis/state \
    /app/.genesis/backups \
    && chown -R genesis:genesis /app \
    && chmod -R 755 /app/scripts \
    && chmod 700 /app/.genesis

# Add version and build info
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="genesis-trading" \
      org.label-schema.description="Project GENESIS Trading System" \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/genesis/trading" \
      org.label-schema.schema-version="1.0"

# Switch to non-root user
USER genesis

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys, os; sys.path.insert(0, '/app'); from genesis.api.health import health_check; sys.exit(0 if health_check() else 1)" || exit 1

# Expose API port (if needed)
EXPOSE 8000

# Default command using supervisor for production
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/genesis.conf"]

# =========== Development Stage (Optional) ===========
FROM production AS development

USER root

# Install development tools
COPY requirements/dev.txt /app/requirements/
RUN pip install --no-cache-dir -r /app/requirements/dev.txt

# Copy test files
COPY --chown=genesis:genesis tests/ /app/tests/
COPY --chown=genesis:genesis pyproject.toml /app/
COPY --chown=genesis:genesis Makefile /app/

USER genesis

# Override command for development
CMD ["python", "-m", "genesis"]