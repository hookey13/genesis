# Docker Deployment Guide

## Overview

Genesis uses multi-stage Docker builds to create optimized, secure container images. Production images target <500MB size while maintaining full functionality.

## Table of Contents
1. [Docker Architecture](#docker-architecture)
2. [Building Images](#building-images)
3. [Running Containers](#running-containers)
4. [Docker Compose](#docker-compose)
5. [Security Best Practices](#security-best-practices)
6. [Troubleshooting](#troubleshooting)

## Docker Architecture

### Multi-Stage Build

Our Dockerfile uses four stages:

```dockerfile
# Stage 1: Builder - Compile dependencies
FROM python:3.11.8-slim as builder

# Stage 2: Production - Minimal runtime
FROM python:3.11.8-slim as production

# Stage 3: Development - With dev tools
FROM production as development

# Stage 4: Testing - Run tests
FROM development as testing
```

### Image Hierarchy

```
builder (compile)
    ↓
production (<500MB)
    ↓
development (with tools)
    ↓
testing (with test data)
```

## Building Images

### Production Build

```bash
# Build production image
docker build --target production -t genesis:prod .

# With specific tier
docker build --target production --build-arg TIER=hunter -t genesis:hunter .

# With build cache
docker build --target production \
  --cache-from genesis:cache \
  -t genesis:prod .
```

### Development Build

```bash
# Build development image
docker build --target development -t genesis:dev .

# With volume mounts for hot reload
docker build --target development -t genesis:dev . && \
docker run -v $(pwd)/genesis:/app/genesis genesis:dev
```

### Size Optimization

```bash
# Check image size
docker images genesis:prod --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Analyze layers
docker history genesis:prod --no-trunc

# Remove intermediate images
docker image prune -f
```

## Running Containers

### Basic Usage

```bash
# Run production container
docker run -d \
  --name genesis-trading \
  --env-file .env \
  -v genesis-data:/app/.genesis/data \
  -v genesis-logs:/app/.genesis/logs \
  -p 9090:9090 \
  genesis:prod

# Run with specific tier
docker run -d \
  --name genesis-hunter \
  -e TIER=hunter \
  --env-file .env \
  genesis:hunter

# Interactive mode
docker run -it --rm \
  --env-file .env \
  genesis:dev /bin/bash
```

### Environment Variables

Required environment variables:

```bash
# Create .env file
cat > .env << EOF
GENESIS_ENV=production
TIER=sniper
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=false
DATABASE_URL=postgresql://user:pass@db:5432/genesis
LOG_LEVEL=INFO
MAX_POSITION_SIZE_USDT=1000.0
MAX_DAILY_LOSS_USDT=100.0
EOF

# Run with env file
docker run --env-file .env genesis:prod
```

### Volume Management

```bash
# Create named volumes
docker volume create genesis-data
docker volume create genesis-logs
docker volume create genesis-state

# Backup volumes
docker run --rm \
  -v genesis-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/genesis-data.tar.gz /data

# Restore volumes
docker run --rm \
  -v genesis-data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar xzf /backup/genesis-data.tar.gz -C /
```

## Docker Compose

### Development Environment

```bash
# Start all services
docker-compose up -d

# Start specific tier
TIER=hunter docker-compose up -d genesis

# View logs
docker-compose logs -f genesis

# Stop services
docker-compose down

# Clean everything
docker-compose down -v --remove-orphans
```

### Production Deployment

```bash
# Production with PostgreSQL
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose --profile monitoring up -d

# Scale services
docker-compose up -d --scale genesis=3
```

### Service Profiles

Available profiles in docker-compose.yml:

| Profile | Services | Use Case |
|---------|----------|----------|
| default | genesis | Basic trading |
| dev | genesis-dev | Development |
| postgres | postgres | Production database |
| redis | redis | Caching layer |
| monitoring | prometheus, grafana | Metrics & dashboards |

```bash
# Run with specific profile
docker-compose --profile postgres up -d

# Multiple profiles
docker-compose --profile postgres --profile monitoring up -d
```

## Security Best Practices

### 1. Non-Root User

All containers run as non-root user (UID 1000):

```dockerfile
# Create genesis user
RUN useradd -r -u 1000 -g genesis genesis

# Switch to non-root
USER genesis
```

### 2. Read-Only Filesystem

```yaml
# docker-compose.yml
services:
  genesis:
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - genesis-data:/app/.genesis/data
```

### 3. Security Scanning

```bash
# Scan with Trivy
trivy image genesis:prod

# Scan with Snyk
snyk container test genesis:prod

# Scan with Docker Scout
docker scout cves genesis:prod
```

### 4. Secret Management

```bash
# Use Docker secrets (Swarm mode)
echo "my_api_key" | docker secret create binance_api_key -

# Reference in compose
services:
  genesis:
    secrets:
      - binance_api_key
    environment:
      BINANCE_API_KEY_FILE: /run/secrets/binance_api_key
```

### 5. Network Isolation

```yaml
# docker-compose.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  genesis:
    networks:
      - backend
  prometheus:
    networks:
      - frontend
      - backend
```

### 6. Resource Limits

```yaml
# docker-compose.yml
services:
  genesis:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

## Health Checks

### Container Health

```dockerfile
# In Dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import genesis; print('healthy')" || exit 1
```

### Monitoring Health

```bash
# Check health status
docker inspect --format='{{json .State.Health}}' genesis-trading

# Watch health
watch docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Custom Health Endpoint

```python
# genesis/health.py
async def health_check():
    checks = {
        "database": check_database(),
        "exchange": check_exchange_connection(),
        "memory": check_memory_usage(),
    }
    return all(checks.values())
```

## Troubleshooting

### Common Issues

#### 1. Container Exits Immediately

```bash
# Check logs
docker logs genesis-trading

# Run interactively to debug
docker run -it --rm --env-file .env genesis:prod /bin/bash

# Check exit code
docker inspect genesis-trading --format='{{.State.ExitCode}}'
```

#### 2. Permission Denied

```bash
# Fix volume permissions
docker run --rm \
  -v genesis-data:/data \
  alpine chown -R 1000:1000 /data

# Or in Dockerfile
COPY --chown=genesis:genesis . /app
```

#### 3. Out of Memory

```bash
# Check memory usage
docker stats genesis-trading

# Increase limits
docker run -m 4g genesis:prod

# Or in compose
services:
  genesis:
    mem_limit: 4g
```

#### 4. Network Issues

```bash
# Test connectivity
docker run --rm --network container:genesis-trading \
  nicolaka/netshoot ping -c 3 api.binance.com

# Check DNS
docker run --rm genesis:prod nslookup api.binance.com
```

#### 5. Build Cache Issues

```bash
# Build without cache
docker build --no-cache -t genesis:prod .

# Clear builder cache
docker builder prune -a

# Clear everything
docker system prune -a --volumes
```

### Debugging Commands

```bash
# Enter running container
docker exec -it genesis-trading /bin/bash

# Copy files from container
docker cp genesis-trading:/app/.genesis/logs/error.log ./

# View processes
docker top genesis-trading

# View filesystem
docker diff genesis-trading

# Export image for analysis
docker save genesis:prod | tar -tv | head -20
```

## Performance Optimization

### 1. Layer Caching

```dockerfile
# Copy requirements first (changes less often)
COPY requirements/ requirements/
RUN pip install -r requirements/base.txt

# Then copy code (changes frequently)
COPY genesis/ genesis/
```

### 2. Multi-Platform Builds

```bash
# Setup buildx
docker buildx create --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag genesis:multi \
  --push .
```

### 3. Build-Time Optimization

```bash
# Use BuildKit
DOCKER_BUILDKIT=1 docker build -t genesis:prod .

# Parallel builds
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from genesis:cache \
  -t genesis:prod .
```

## Production Checklist

- [ ] Image size <500MB
- [ ] Non-root user configured
- [ ] Health checks implemented
- [ ] Resource limits set
- [ ] Security scans passing
- [ ] Secrets externalized
- [ ] Volumes for persistent data
- [ ] Logging configured
- [ ] Restart policy set
- [ ] Network isolation configured

## Monitoring & Logging

### Prometheus Metrics

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9092:9090"
```

### Log Aggregation

```bash
# View logs
docker logs -f --tail 100 genesis-trading

# Export logs
docker logs genesis-trading > genesis.log 2>&1

# Log rotation
services:
  genesis:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "5"
```

## Deployment Strategies

### Rolling Update

```bash
# Build new image
docker build -t genesis:new .

# Start new container
docker run -d --name genesis-new genesis:new

# Verify health
docker exec genesis-new python -c "import genesis"

# Switch traffic
docker stop genesis-trading
docker rename genesis-new genesis-trading
```

### Blue-Green Deployment

```bash
# Run blue (current)
docker run -d --name genesis-blue -p 9090:9090 genesis:v1

# Deploy green (new)
docker run -d --name genesis-green -p 9091:9090 genesis:v2

# Test green
curl http://localhost:9091/health

# Switch
docker stop genesis-blue
docker run -d --name genesis-green -p 9090:9090 genesis:v2
```

## Next Steps

- [Container security](../security/container_security.md)
- [Kubernetes deployment](kubernetes.md)
- [CI/CD pipelines](cicd.md)
- [Monitoring setup](monitoring.md)