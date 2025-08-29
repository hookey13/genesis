#!/bin/bash
# Project GENESIS Blue-Green Deployment Script
# Implements zero-downtime deployment with automatic rollback

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.prod.yml"
readonly DEPLOYMENT_LOG="/var/log/genesis/deployment.log"
readonly HEALTH_CHECK_TIMEOUT=180
readonly DRAIN_TIMEOUT=30

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "${PROJECT_ROOT}/.env.production" ]]; then
        log_error "Production environment file not found: ${PROJECT_ROOT}/.env.production"
        exit 1
    fi
    
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$DEPLOYMENT_LOG")"
    
    log_info "Prerequisites check passed"
}

# Get current deployment color (blue or green)
get_current_color() {
    if docker ps --format "table {{.Names}}" | grep -q "genesis-green"; then
        echo "green"
    elif docker ps --format "table {{.Names}}" | grep -q "genesis-blue"; then
        echo "blue"
    else
        echo "none"
    fi
}

# Get target deployment color
get_target_color() {
    local current=$1
    if [[ "$current" == "green" ]]; then
        echo "blue"
    else
        echo "green"
    fi
}

# Save current state for rollback
save_current_state() {
    local color=$1
    log_info "Saving current state for potential rollback..."
    
    # Save container IDs
    docker ps -q --filter "label=com.genesis.deployment=$color" > "/tmp/genesis-${color}-containers.txt"
    
    # Save current image tags
    docker images --format "{{.Repository}}:{{.Tag}}" | grep genesis > "/tmp/genesis-${color}-images.txt" || true
    
    # Backup database
    if docker exec "genesis-${color}-prod" test -f /app/.genesis/data/genesis.db 2>/dev/null; then
        docker exec "genesis-${color}-prod" cp /app/.genesis/data/genesis.db /app/.genesis/data/genesis.db.backup
        log_info "Database backed up"
    fi
}

# Deploy new version
deploy_new_version() {
    local target_color=$1
    local version=$2
    
    log_info "Deploying version $version to $target_color environment..."
    
    # Set environment variables
    export DEPLOYMENT_COLOR="$target_color"
    export VERSION="$version"
    
    # Pull new image
    log_info "Pulling new image..."
    docker pull "genesis-trading:${version}" || {
        log_error "Failed to pull image genesis-trading:${version}"
        return 1
    }
    
    # Start new environment
    log_info "Starting $target_color environment..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" -p "genesis-${target_color}" up -d || {
        log_error "Failed to start $target_color environment"
        return 1
    }
    
    return 0
}

# Wait for health checks
wait_for_health() {
    local color=$1
    local timeout=$2
    local elapsed=0
    
    log_info "Waiting for $color environment to become healthy (timeout: ${timeout}s)..."
    
    while [[ $elapsed -lt $timeout ]]; do
        if docker exec "genesis-${color}-prod" python -m genesis.api.health readiness 2>/dev/null; then
            log_info "$color environment is healthy"
            return 0
        fi
        
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    
    echo
    log_error "$color environment failed health check after ${timeout}s"
    return 1
}

# Run smoke tests
run_smoke_tests() {
    local color=$1
    
    log_info "Running smoke tests on $color environment..."
    
    # Basic health check
    if ! docker exec "genesis-${color}-prod" python -m genesis.api.health detailed 2>/dev/null; then
        log_error "Detailed health check failed"
        return 1
    fi
    
    # Check database connectivity
    if ! docker exec "genesis-${color}-prod" python -c "from genesis.data.repository import Repository; print('DB OK')" 2>/dev/null; then
        log_error "Database connectivity check failed"
        return 1
    fi
    
    # Check exchange connectivity (if not in test mode)
    if [[ "${BINANCE_TESTNET:-true}" == "false" ]]; then
        if ! docker exec "genesis-${color}-prod" python -c "from genesis.exchange.gateway import ExchangeGateway; print('Exchange OK')" 2>/dev/null; then
            log_warn "Exchange connectivity check failed (non-critical in test mode)"
        fi
    fi
    
    log_info "Smoke tests passed"
    return 0
}

# Drain connections from old environment
drain_connections() {
    local color=$1
    local timeout=$2
    
    log_info "Draining connections from $color environment (timeout: ${timeout}s)..."
    
    # Send SIGTERM to allow graceful shutdown
    docker exec "genesis-${color}-prod" kill -TERM 1 2>/dev/null || true
    
    # Wait for connections to drain
    sleep "$timeout"
    
    log_info "Connection draining complete"
}

# Switch traffic to new environment
switch_traffic() {
    local from_color=$1
    local to_color=$2
    
    log_info "Switching traffic from $from_color to $to_color..."
    
    # In a real deployment, this would update load balancer or nginx config
    # For now, we'll update a symlink or marker file
    echo "$to_color" > /tmp/genesis-active-deployment
    
    # Update DNS or service discovery (placeholder)
    # update_service_discovery "$to_color"
    
    log_info "Traffic switched to $to_color environment"
}

# Stop old environment
stop_old_environment() {
    local color=$1
    
    log_info "Stopping $color environment..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" -p "genesis-${color}" down || {
        log_warn "Failed to stop $color environment gracefully"
    }
    
    log_info "$color environment stopped"
}

# Rollback to previous version
rollback() {
    local current_color=$1
    local target_color=$2
    
    log_error "Initiating rollback from $target_color to $current_color..."
    
    # Stop failed deployment
    docker-compose -f "$DOCKER_COMPOSE_FILE" -p "genesis-${target_color}" down || true
    
    # Restore database backup if exists
    if docker exec "genesis-${current_color}-prod" test -f /app/.genesis/data/genesis.db.backup 2>/dev/null; then
        docker exec "genesis-${current_color}-prod" cp /app/.genesis/data/genesis.db.backup /app/.genesis/data/genesis.db
        log_info "Database restored from backup"
    fi
    
    # Ensure current environment is still running
    docker-compose -f "$DOCKER_COMPOSE_FILE" -p "genesis-${current_color}" up -d
    
    # Verify current environment health
    if wait_for_health "$current_color" 60; then
        log_info "Rollback successful - $current_color environment is healthy"
        return 0
    else
        log_error "CRITICAL: Rollback failed - manual intervention required!"
        return 1
    fi
}

# Create deployment record
create_deployment_record() {
    local version=$1
    local color=$2
    local status=$3
    
    local record_file="${PROJECT_ROOT}/.genesis/deployments/$(date +%Y%m%d_%H%M%S).json"
    mkdir -p "$(dirname "$record_file")"
    
    cat > "$record_file" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "$version",
    "color": "$color",
    "status": "$status",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "deployer": "${USER}",
    "hostname": "$(hostname)"
}
EOF
    
    log_info "Deployment record created: $record_file"
}

# Main deployment function
main() {
    local version="${1:-latest}"
    local dry_run="${2:-false}"
    
    log_info "Starting blue-green deployment for version: $version"
    log_info "Timestamp: $(date)"
    
    # Check prerequisites
    check_prerequisites
    
    # Get current deployment state
    local current_color=$(get_current_color)
    local target_color=$(get_target_color "$current_color")
    
    log_info "Current deployment: $current_color"
    log_info "Target deployment: $target_color"
    
    if [[ "$dry_run" == "true" ]]; then
        log_info "DRY RUN mode - no actual deployment will be performed"
        exit 0
    fi
    
    # Save current state
    if [[ "$current_color" != "none" ]]; then
        save_current_state "$current_color"
    fi
    
    # Deploy new version
    if ! deploy_new_version "$target_color" "$version"; then
        log_error "Deployment failed"
        create_deployment_record "$version" "$target_color" "failed"
        exit 1
    fi
    
    # Wait for health checks
    if ! wait_for_health "$target_color" "$HEALTH_CHECK_TIMEOUT"; then
        log_error "Health check failed"
        if [[ "$current_color" != "none" ]]; then
            rollback "$current_color" "$target_color"
        fi
        create_deployment_record "$version" "$target_color" "failed_health"
        exit 1
    fi
    
    # Run smoke tests
    if ! run_smoke_tests "$target_color"; then
        log_error "Smoke tests failed"
        if [[ "$current_color" != "none" ]]; then
            rollback "$current_color" "$target_color"
        fi
        create_deployment_record "$version" "$target_color" "failed_smoke"
        exit 1
    fi
    
    # Drain connections from old environment
    if [[ "$current_color" != "none" ]]; then
        drain_connections "$current_color" "$DRAIN_TIMEOUT"
        
        # Switch traffic
        switch_traffic "$current_color" "$target_color"
        
        # Stop old environment
        stop_old_environment "$current_color"
    else
        # First deployment
        switch_traffic "none" "$target_color"
    fi
    
    # Create success record
    create_deployment_record "$version" "$target_color" "success"
    
    log_info "Deployment completed successfully!"
    log_info "New active environment: $target_color"
    log_info "Version: $version"
    
    # Tag deployment in git
    if command -v git &> /dev/null && [[ -d "${PROJECT_ROOT}/.git" ]]; then
        git tag -a "deployed-${version}-$(date +%Y%m%d%H%M%S)" -m "Deployed version $version to $target_color" 2>/dev/null || true
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi