#!/bin/bash
# Container Push Script for Genesis Trading System
# Pushes Docker images to registry with proper validation

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-genesis-trading}"
REGISTRY="${REGISTRY:-}"

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

# Function to check if logged into registry
check_registry_auth() {
    local registry="$1"
    
    print_status "Checking registry authentication..."
    
    if [ -z "$registry" ]; then
        print_warning "No registry specified, assuming local registry"
        return 0
    fi
    
    # Extract registry hostname
    REGISTRY_HOST=$(echo "$registry" | cut -d'/' -f1)
    
    # Check if we can access the registry
    if docker login "$REGISTRY_HOST" --username="${DOCKER_USERNAME}" --password-stdin <<< "${DOCKER_PASSWORD}" 2>/dev/null; then
        print_status "Successfully authenticated to $REGISTRY_HOST"
    else
        print_error "Failed to authenticate to $REGISTRY_HOST"
        print_error "Please run: docker login $REGISTRY_HOST"
        return 1
    fi
}

# Function to check if image exists locally
check_image_exists() {
    local image="$1"
    
    if docker image inspect "$image" &>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to tag image for registry
tag_for_registry() {
    local local_image="$1"
    local registry_image="$2"
    
    print_status "Tagging $local_image as $registry_image"
    docker tag "$local_image" "$registry_image"
}

# Function to push image
push_image() {
    local image="$1"
    
    print_status "Pushing $image..."
    
    if docker push "$image"; then
        print_status "Successfully pushed $image"
        return 0
    else
        print_error "Failed to push $image"
        return 1
    fi
}

# Function to verify pushed image
verify_pushed_image() {
    local image="$1"
    
    print_status "Verifying pushed image..."
    
    # Pull the image to verify it was pushed correctly
    docker pull "$image" &>/dev/null
    
    if [ $? -eq 0 ]; then
        print_status "Image verified successfully"
        
        # Get manifest digest
        DIGEST=$(docker inspect "$image" --format='{{.RepoDigests}}' | sed 's/.*@//' | sed 's/\].*//')
        print_status "Image digest: $DIGEST"
    else
        print_error "Failed to verify image"
        return 1
    fi
}

# Function to push multi-platform image
push_multiplatform() {
    local image_name="$1"
    local tags="$2"
    local platforms="${3:-linux/amd64,linux/arm64}"
    
    print_status "Building and pushing multi-platform image..."
    print_status "Platforms: $platforms"
    
    # Ensure buildx is available
    if ! docker buildx version &>/dev/null; then
        print_error "Docker buildx is required for multi-platform push"
        return 1
    fi
    
    # Use or create builder
    BUILDER_NAME="genesis-builder"
    docker buildx use "$BUILDER_NAME" 2>/dev/null || docker buildx create --name "$BUILDER_NAME" --use
    
    # Build and push in one step for multi-platform
    BUILD_CMD="docker buildx build"
    BUILD_CMD="${BUILD_CMD} --platform ${platforms}"
    BUILD_CMD="${BUILD_CMD} --target production"
    BUILD_CMD="${BUILD_CMD} --push"
    BUILD_CMD="${BUILD_CMD} ${tags}"
    BUILD_CMD="${BUILD_CMD} --file ${PROJECT_ROOT}/docker/Dockerfile"
    BUILD_CMD="${BUILD_CMD} ${PROJECT_ROOT}"
    
    print_status "Executing: $BUILD_CMD"
    eval "$BUILD_CMD"
}

# Function to create and push manifest list
create_manifest() {
    local base_image="$1"
    local tag="$2"
    
    print_status "Creating manifest for $base_image:$tag"
    
    # Create manifest list
    docker manifest create "${base_image}:${tag}" \
        "${base_image}:${tag}-amd64" \
        "${base_image}:${tag}-arm64"
    
    # Annotate architectures
    docker manifest annotate "${base_image}:${tag}" \
        "${base_image}:${tag}-amd64" --arch amd64
    
    docker manifest annotate "${base_image}:${tag}" \
        "${base_image}:${tag}-arm64" --arch arm64
    
    # Push manifest
    docker manifest push "${base_image}:${tag}"
}

# Main execution
main() {
    print_status "Genesis Trading System - Container Push Script"
    print_status "==============================================="
    
    # Parse command line arguments
    TAGS=""
    MULTI_PLATFORM=0
    PLATFORMS="linux/amd64,linux/arm64"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --tag)
                TAGS="${TAGS} $2"
                shift 2
                ;;
            --multi-platform)
                MULTI_PLATFORM=1
                shift
                ;;
            --platform)
                PLATFORMS="$2"
                shift 2
                ;;
            --name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --registry REGISTRY    Docker registry (required)"
                echo "  --tag TAG             Image tag to push (can be specified multiple times)"
                echo "  --multi-platform      Build and push multi-platform image"
                echo "  --platform PLATFORMS  Platforms for multi-platform build"
                echo "  --name NAME           Image name (default: genesis-trading)"
                echo "  --help                Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Validate registry
    if [ -z "$REGISTRY" ]; then
        print_error "Registry is required. Use --registry option."
        exit 1
    fi
    
    # Default tags if none specified
    if [ -z "$TAGS" ]; then
        TAGS="latest"
    fi
    
    # Prepare full image name
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
    
    print_status "Registry: $REGISTRY"
    print_status "Image: $FULL_IMAGE_NAME"
    print_status "Tags: $TAGS"
    
    # Check registry authentication
    check_registry_auth "$REGISTRY"
    
    if [ "$MULTI_PLATFORM" -eq 1 ]; then
        # Multi-platform build and push
        TAG_ARGS=""
        for tag in $TAGS; do
            TAG_ARGS="${TAG_ARGS} -t ${FULL_IMAGE_NAME}:${tag}"
        done
        
        push_multiplatform "$FULL_IMAGE_NAME" "$TAG_ARGS" "$PLATFORMS"
    else
        # Single platform push
        for tag in $TAGS; do
            LOCAL_IMAGE="${IMAGE_NAME}:${tag}"
            REGISTRY_IMAGE="${FULL_IMAGE_NAME}:${tag}"
            
            # Check if local image exists
            if ! check_image_exists "$LOCAL_IMAGE"; then
                print_error "Local image $LOCAL_IMAGE not found"
                print_status "Run build_container.sh first to build the image"
                exit 1
            fi
            
            # Tag for registry
            tag_for_registry "$LOCAL_IMAGE" "$REGISTRY_IMAGE"
            
            # Push image
            push_image "$REGISTRY_IMAGE"
            
            # Verify push
            verify_pushed_image "$REGISTRY_IMAGE"
        done
    fi
    
    print_status "==============================================="
    print_status "Push completed successfully!"
    print_status "Pushed images:"
    for tag in $TAGS; do
        echo "  - ${FULL_IMAGE_NAME}:${tag}"
    done
    
    # Show pull commands
    print_status ""
    print_status "Pull commands:"
    for tag in $TAGS; do
        echo "  docker pull ${FULL_IMAGE_NAME}:${tag}"
    done
}

# Run main function
main "$@"