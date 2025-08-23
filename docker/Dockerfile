# Project GENESIS Docker Image
# Based on Python 3.11.8 for consistency with development

FROM python:3.11.8-slim-bookworm

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r genesis && useradd -r -g genesis genesis

# Copy requirements first for better caching
COPY requirements/ /app/requirements/
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY genesis/ /app/genesis/
COPY config/ /app/config/
COPY alembic/ /app/alembic/
COPY scripts/ /app/scripts/

# Create necessary directories with proper permissions
RUN mkdir -p /app/.genesis/data /app/.genesis/logs /app/.genesis/state && \
    chown -R genesis:genesis /app

# Copy supervisor configuration for production
COPY docker/supervisord.conf /etc/supervisor/conf.d/genesis.conf

# Make scripts executable
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true
RUN chmod +x /app/scripts/*.py 2>/dev/null || true

# Switch to non-root user
USER genesis

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "genesis"]