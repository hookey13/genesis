#!/bin/bash
# Project GENESIS Rollback Script
# Emergency rollback to previous deployment

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly DOCKER_COMPOSE_FILE="${PROJECT_ROOT}/docker/docker-compose.prod.yml"
readonly DEPLOYMENT_LOG="/var/log/genesis/rollback.log"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

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

# Get current deployment color
get_current_color() {
    if [[ -f /tmp/genesis-active-deployment ]]; then
        cat /tmp/genesis-active-deployment
    else
        echo "unknown"
    fi
}

# Get previous deployment info
get_previous_deployment() {
    local deployment_dir="${PROJECT_ROOT}/.genesis/deployments"
    if [[ -d "$deployment_dir" ]]; then
        # Get second to last successful deployment
        find "$deployment_dir" -name "*.json" -exec grep -l '"status": "success"' {} \; | \
            sort -r | head -2 | tail -1
    else
        echo ""
    fi
}

# Perform emergency rollback
emergency_rollback() {
    log_error "Starting emergency rollback..."
    
    # Get current color
    local current_color=$(get_current_color)
    log_info "Current deployment: $current_color"
    
    # Determine target color for rollback
    local target_color
    if [[ "$current_color" == "blue" ]]; then
        target_color="green"
    elif [[ "$current_color" == "green" ]]; then
        target_color="blue"
    else
        log_error "Cannot determine deployment color for rollback"
        exit 1
    fi
    
    log_info "Rolling back to: $target_color"
    
    # Check if target environment exists
    if ! docker ps -a --format "table {{.Names}}" | grep -q "genesis-${target_color}"; then
        log_error "Target environment $target_color does not exist"
        
        # Try to restore from backup
        log_info "Attempting to restore from last known good deployment..."
        local previous_deployment=$(get_previous_deployment)
        
        if [[ -z "$previous_deployment" ]]; then
            log_error "No previous successful deployment found"
            exit 1
        fi
        
        # Extract version from deployment record
        local version=$(grep '"version"' "$previous_deployment" | cut -d'"' -f4)
        log_info "Restoring version: $version"
        
        # Start previous version
        VERSION="$version" docker-compose -f "$DOCKER_COMPOSE_FILE" -p "genesis-${target_color}" up -d
    else
        # Start existing target environment
        docker-compose -f "$DOCKER_COMPOSE_FILE" -p "genesis-${target_color}" up -d
    fi
    
    # Wait for health check
    local timeout=60
    local elapsed=0
    
    log_info "Waiting for $target_color environment to become healthy..."
    while [[ $elapsed -lt $timeout ]]; do
        if docker exec "genesis-${target_color}-prod" python -m genesis.api.health readiness 2>/dev/null; then
            log_info "$target_color environment is healthy"
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    
    if [[ $elapsed -ge $timeout ]]; then
        log_error "Rollback environment failed health check"
        exit 1
    fi
    
    # Update active deployment marker
    echo "$target_color" > /tmp/genesis-active-deployment
    
    # Stop failed environment
    log_info "Stopping failed $current_color environment..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" -p "genesis-${current_color}" down || true
    
    # Create rollback record
    local record_file="${PROJECT_ROOT}/.genesis/deployments/rollback_$(date +%Y%m%d_%H%M%S).json"
    mkdir -p "$(dirname "$record_file")"
    
    cat > "$record_file" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "type": "rollback",
    "from_color": "$current_color",
    "to_color": "$target_color",
    "reason": "${1:-manual}",
    "operator": "${USER}",
    "hostname": "$(hostname)"
}
EOF
    
    log_info "Rollback completed successfully!"
    log_info "Active environment: $target_color"
}

# Main function
main() {
    local reason="${1:-manual}"
    
    log_info "========================================="
    log_info "EMERGENCY ROLLBACK INITIATED"
    log_info "Reason: $reason"
    log_info "Timestamp: $(date)"
    log_info "========================================="
    
    # Create log directory
    mkdir -p "$(dirname "$DEPLOYMENT_LOG")"
    
    # Perform rollback
    emergency_rollback "$reason"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi