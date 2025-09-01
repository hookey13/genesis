#!/bin/bash
# Container Build Script for Genesis Trading System
# Builds multi-platform Docker images with proper tagging and caching

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKERFILE="${PROJECT_ROOT}/docker/Dockerfile"
IMAGE_NAME="${IMAGE_NAME:-genesis-trading}"
REGISTRY="${REGISTRY:-}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
BUILD_TARGET="${BUILD_TARGET:-production}"
MAX_IMAGE_SIZE_MB=500

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to get git information
get_git_info() {
    if [ -d "${PROJECT_ROOT}/.git" ]; then
        GIT_COMMIT=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown")
        GIT_BRANCH=$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        GIT_TAG=$(git -C "$PROJECT_ROOT" describe --tags --exact-match 2>/dev/null || echo "")
        GIT_DIRTY=$(git -C "$PROJECT_ROOT" diff --quiet || echo "-dirty")
    else
        GIT_COMMIT="unknown"
        GIT_BRANCH="unknown"
        GIT_TAG=""
        GIT_DIRTY=""
    fi
}

# Function to generate image tags
generate_tags() {
    local base_name="$1"
    local tags=""
    
    # Add version tag if git tag exists
    if [ -n "$GIT_TAG" ]; then
        tags="${tags} -t ${base_name}:${GIT_TAG}"
        tags="${tags} -t ${base_name}:latest"
    fi
    
    # Add commit-based tag
    tags="${tags} -t ${base_name}:${GIT_BRANCH}-${GIT_COMMIT}${GIT_DIRTY}"
    
    # Add branch tag
    if [ "$GIT_BRANCH" != "unknown" ]; then
        tags="${tags} -t ${base_name}:${GIT_BRANCH}"
    fi
    
    # Add timestamp tag for uniqueness
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    tags="${tags} -t ${base_name}:build-${TIMESTAMP}"
    
    # Default to latest if no other tags
    if [ -z "$tags" ]; then
        tags="-t ${base_name}:latest"
    fi
    
    echo "$tags"
}

# Function to check Docker buildx availability
check_buildx() {
    if ! docker buildx version &>/dev/null; then
        print_error "Docker buildx is not available. Please install Docker Desktop or enable buildx."
        exit 1
    fi
    
    # Create or use buildx builder
    BUILDER_NAME="genesis-builder"
    if ! docker buildx ls | grep -q "$BUILDER_NAME"; then
        print_status "Creating buildx builder: $BUILDER_NAME"
        docker buildx create --name "$BUILDER_NAME" --use --platform "$PLATFORMS"
    else
        print_status "Using existing buildx builder: $BUILDER_NAME"
        docker buildx use "$BUILDER_NAME"
    fi
    
    # Bootstrap builder
    docker buildx inspect --bootstrap
}

# Function to scan image for vulnerabilities
scan_image() {
    local image="$1"
    
    print_status "Scanning image for vulnerabilities..."
    
    # Try to use trivy if available
    if command -v trivy &>/dev/null; then
        trivy image --severity HIGH,CRITICAL "$image" || print_warning "Vulnerabilities found"
    # Try to use grype if available
    elif command -v grype &>/dev/null; then
        grype "$image" || print_warning "Vulnerabilities found"
    # Try Docker Scout if available
    elif docker scout version &>/dev/null; then
        docker scout cves "$image" || print_warning "Vulnerabilities found"
    else
        print_warning "No vulnerability scanner found. Install trivy, grype, or Docker Scout for security scanning."
    fi
}

# Function to check image size
check_image_size() {
    local image="$1"
    
    print_status "Checking image size..."
    
    # Get image size in MB
    SIZE_BYTES=$(docker inspect "$image" --format='{{.Size}}' 2>/dev/null || echo "0")
    SIZE_MB=$((SIZE_BYTES / 1024 / 1024))
    
    if [ "$SIZE_MB" -gt "$MAX_IMAGE_SIZE_MB" ]; then
        print_error "Image size (${SIZE_MB}MB) exceeds maximum allowed size (${MAX_IMAGE_SIZE_MB}MB)"
        return 1
    else
        print_status "Image size: ${SIZE_MB}MB (within ${MAX_IMAGE_SIZE_MB}MB limit)"
    fi
}

# Function to build the image
build_image() {
    local image_name="$1"
    local tags="$2"
    
    print_status "Building Docker image..."
    print_status "Target: $BUILD_TARGET"
    print_status "Platforms: $PLATFORMS"
    
    # Build arguments
    BUILD_ARGS=""
    BUILD_ARGS="${BUILD_ARGS} --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    BUILD_ARGS="${BUILD_ARGS} --build-arg VERSION=${GIT_TAG:-${GIT_COMMIT}}"
    BUILD_ARGS="${BUILD_ARGS} --build-arg VCS_REF=${GIT_COMMIT}"
    
    # Cache configuration
    CACHE_ARGS=""
    if [ -n "$BUILDKIT_CACHE_MOUNT" ]; then
        CACHE_ARGS="--cache-from type=local,src=/tmp/.buildx-cache"
        CACHE_ARGS="${CACHE_ARGS} --cache-to type=local,dest=/tmp/.buildx-cache-new,mode=max"
    fi
    
    # Build command
    BUILD_CMD="docker buildx build"
    BUILD_CMD="${BUILD_CMD} --platform ${PLATFORMS}"
    BUILD_CMD="${BUILD_CMD} --target ${BUILD_TARGET}"
    BUILD_CMD="${BUILD_CMD} ${BUILD_ARGS}"
    BUILD_CMD="${BUILD_CMD} ${CACHE_ARGS}"
    BUILD_CMD="${BUILD_CMD} ${tags}"
    BUILD_CMD="${BUILD_CMD} --file ${DOCKERFILE}"
    
    # Add load flag for single platform builds (for local testing)
    if [ "$PLATFORMS" = "linux/amd64" ] || [ "$PLATFORMS" = "linux/arm64" ]; then
        BUILD_CMD="${BUILD_CMD} --load"
    else
        BUILD_CMD="${BUILD_CMD} --push=false"
    fi
    
    BUILD_CMD="${BUILD_CMD} ${PROJECT_ROOT}"
    
    print_status "Executing: $BUILD_CMD"
    eval "$BUILD_CMD"
    
    # Move cache
    if [ -n "$BUILDKIT_CACHE_MOUNT" ] && [ -d "/tmp/.buildx-cache-new" ]; then
        rm -rf /tmp/.buildx-cache
        mv /tmp/.buildx-cache-new /tmp/.buildx-cache
    fi
}

# Function to run tests on built image
test_image() {
    local image="$1"
    
    print_status "Running container tests..."
    
    # Test 1: Container starts successfully
    print_status "Test 1: Container startup"
    if docker run --rm -d --name genesis-test "$image" sleep 10; then
        docker stop genesis-test 2>/dev/null || true
        print_status "✓ Container starts successfully"
    else
        print_error "✗ Container failed to start"
        return 1
    fi
    
    # Test 2: Health check command works
    print_status "Test 2: Health check"
    if docker run --rm "$image" python -m genesis.cli doctor --help &>/dev/null; then
        print_status "✓ Health check command available"
    else
        print_error "✗ Health check command failed"
        return 1
    fi
    
    # Test 3: Non-root user
    print_status "Test 3: Non-root user"
    USER_ID=$(docker run --rm "$image" id -u)
    if [ "$USER_ID" = "1000" ]; then
        print_status "✓ Running as non-root user (UID: $USER_ID)"
    else
        print_error "✗ Not running as expected user (UID: $USER_ID)"
        return 1
    fi
}

# Main execution
main() {
    print_status "Genesis Trading System - Container Build Script"
    print_status "================================================"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --platform)
                PLATFORMS="$2"
                shift 2
                ;;
            --target)
                BUILD_TARGET="$2"
                shift 2
                ;;
            --name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --no-scan)
                NO_SCAN=1
                shift
                ;;
            --no-test)
                NO_TEST=1
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --registry REGISTRY   Docker registry (e.g., docker.io/username)"
                echo "  --platform PLATFORMS  Build platforms (default: linux/amd64,linux/arm64)"
                echo "  --target TARGET       Build target (default: production)"
                echo "  --name NAME          Image name (default: genesis-trading)"
                echo "  --no-scan            Skip vulnerability scanning"
                echo "  --no-test            Skip image testing"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Get git information
    get_git_info
    
    # Prepare image name with registry
    if [ -n "$REGISTRY" ]; then
        FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
    else
        FULL_IMAGE_NAME="${IMAGE_NAME}"
    fi
    
    # Generate tags
    TAGS=$(generate_tags "$FULL_IMAGE_NAME")
    
    print_status "Image: $FULL_IMAGE_NAME"
    print_status "Git commit: $GIT_COMMIT"
    print_status "Git branch: $GIT_BRANCH"
    print_status "Git tag: ${GIT_TAG:-none}"
    
    # Check Docker buildx
    check_buildx
    
    # Build the image
    build_image "$FULL_IMAGE_NAME" "$TAGS"
    
    # For single-platform builds, run additional checks
    if [ "$PLATFORMS" = "linux/amd64" ] || [ "$PLATFORMS" = "linux/arm64" ]; then
        # Check image size
        check_image_size "${FULL_IMAGE_NAME}:latest"
        
        # Run tests unless disabled
        if [ -z "$NO_TEST" ]; then
            test_image "${FULL_IMAGE_NAME}:latest"
        fi
        
        # Scan for vulnerabilities unless disabled
        if [ -z "$NO_SCAN" ]; then
            scan_image "${FULL_IMAGE_NAME}:latest"
        fi
    fi
    
    print_status "================================================"
    print_status "Build completed successfully!"
    print_status "Image tags:"
    echo "$TAGS" | tr ' ' '\n' | grep '^-t' | sed 's/^-t /  - /'
}

# Run main function
main "$@"