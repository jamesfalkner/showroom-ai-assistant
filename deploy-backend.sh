#!/bin/bash

# Showroom AI Assistant - Simplified Deploy Script
# Deploy the AI assistant to Kubernetes/OpenShift with command line options

set -e  # Exit on any error

# Default configuration
NAMESPACE=""
CONFIG_ONLY="false"
CONTENT_SOURCE="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Show usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy the Showroom AI Assistant to Kubernetes/OpenShift"
    echo ""
    echo "Options:"
    echo "  --namespace NS      Specify target namespace (REQUIRED)"
    echo "  --config-only       Only update configuration, skip container build"
    echo "  --content-source    Path to showroom repo for content (default: current directory)"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --namespace showroom-user1"
    echo "  $0 --namespace production --config-only"
    echo "  $0 --content-source ../my-workshop --namespace showroom-user1"
    echo "  $0 --help"
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --config-only)
                CONFIG_ONLY="true"
                shift
                ;;
            --content-source)
                CONTENT_SOURCE="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use --help for usage information."
                ;;
        esac
    done
    
    # Validate required parameters
    if [ -z "$NAMESPACE" ]; then
        error "Namespace is required. Use --namespace to specify the target namespace."
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running on OpenShift or Kubernetes
    if command -v oc &> /dev/null && oc whoami &> /dev/null; then
        KUBE_CMD="oc"
        PLATFORM="OpenShift"
        log "Detected OpenShift platform"
    elif command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
        KUBE_CMD="kubectl"
        PLATFORM="Kubernetes"
        log "Detected Kubernetes platform"
    else
        error "Neither oc nor kubectl is available or cluster is not accessible"
    fi
    
    success "Prerequisites check completed"
}

# Validate content source
validate_content_source() {
    log "Validating content source: $CONTENT_SOURCE"
    
    # Convert to absolute path
    CONTENT_SOURCE=$(cd "$CONTENT_SOURCE" && pwd)
    
    if [ ! -d "$CONTENT_SOURCE" ]; then
        error "Content source directory does not exist: $CONTENT_SOURCE"
    fi
    
    if [ ! -d "$CONTENT_SOURCE/content" ]; then
        error "Content source does not contain a 'content' directory: $CONTENT_SOURCE"
    fi
    
    if [ ! -d "$CONTENT_SOURCE/content/modules/ROOT" ]; then
        error "Content source does not have the expected structure: $CONTENT_SOURCE/content/modules/ROOT"
    fi
    
    success "Content source validated: $CONTENT_SOURCE"
}

# Build the static site
build_site() {
    if [ "$CONFIG_ONLY" = "true" ]; then
        log "Skipping site build (config-only mode)"
        return 0
    fi
    
    log "Building workshop static site with Antora..."
    
    if command -v npx &> /dev/null; then
        log "Using npx antora..."
        npx antora default-site.yml
    elif command -v antora &> /dev/null; then
        log "Using antora directly..."
        antora default-site.yml
    else
        error "Cannot build site - neither npx nor antora command found"
    fi
    
    if [ -d "./www" ]; then
        success "Static site built successfully in ./www"
    else
        error "Site build failed - ./www directory not found"
    fi
}

# Build backend image using OpenShift BuildConfig
build_backend_image() {
    if [ "$CONFIG_ONLY" = "true" ]; then
        log "Skipping backend image build (config-only mode)"
        return 0
    fi
    
    log "Building backend image using OpenShift BuildConfig..."
    
    # Check if we have oc command and are logged in
    if ! command -v oc &> /dev/null; then
        warn "oc command not found, skipping OpenShift build"
        return 0
    fi
    
    if ! oc whoami &> /dev/null; then
        warn "Not logged into OpenShift, skipping build"
        return 0
    fi
    
    # Create namespace if it doesn't exist
    if ! $KUBE_CMD get namespace "$NAMESPACE" &> /dev/null; then
        log "Creating namespace: $NAMESPACE"
        $KUBE_CMD create namespace "$NAMESPACE"
    fi
    
    # Check if BuildConfig exists, create it if not
    if ! oc get buildconfig showroom-ai-assistant -n "$NAMESPACE" &> /dev/null; then
        log "BuildConfig not found, creating it..."
        if [ -f "k8s/buildconfig.yaml" ]; then
            oc apply -n "$NAMESPACE" -f "k8s/buildconfig.yaml"
            success "BuildConfig created"
        else
            error "BuildConfig file k8s/buildconfig.yaml not found"
        fi
    fi
    
    # Prepare build directory with content from source
    prepare_build_content
    
    # Start the build
    log "Starting OpenShift build..."
    if ! oc start-build showroom-ai-assistant --from-dir=. --follow --namespace="$NAMESPACE"; then
        error "Failed to build backend image using OpenShift BuildConfig"
    fi
    success "Backend image built successfully using OpenShift BuildConfig"
}

# Prepare build content from source repository
prepare_build_content() {
    if [ "$CONTENT_SOURCE" = "$(pwd)" ]; then
        log "Using content from current directory"
        return 0
    fi
    
    log "Preparing build content from: $CONTENT_SOURCE"
    
    # Backup existing content if it exists
    if [ -d "./content" ]; then
        log "Backing up existing content directory..."
        mv "./content" "./content.backup.$(date +%s)"
    fi
    
    # Copy content from source repository
    log "Copying content from source repository..."
    cp -r "$CONTENT_SOURCE/content" "./content"
    
    success "Build content prepared from source repository"
}

# Create or update secrets from .env.yaml
create_secrets() {
    if [ -f ".env.yaml" ]; then
        log "Reading API key from .env.yaml..."
        
        # Extract LLM API key from YAML file using simple grep/sed
        LLM_API_KEY=$(grep "llm_api_key:" .env.yaml | sed 's/.*llm_api_key: *"\([^"]*\)".*/\1/' | tr -d ' ')
        
        if [ -n "$LLM_API_KEY" ] && [ "$LLM_API_KEY" != "your-api-key-here" ]; then
            log "Creating secret with API key from .env.yaml..."
            $KUBE_CMD create secret generic ai-assistant-secrets \
                --from-literal=LLM_API_KEY="$LLM_API_KEY" \
                --namespace="$NAMESPACE" \
                --dry-run=client -o yaml | $KUBE_CMD apply -f -
            success "Secret created/updated with API key"
        else
            warn "No valid API key found in .env.yaml"
            warn "Please update .env.yaml with your actual LLM API key"
            # Create placeholder secret
            $KUBE_CMD create secret generic ai-assistant-secrets \
                --from-literal=LLM_API_KEY="REPLACE_WITH_YOUR_API_KEY" \
                --namespace="$NAMESPACE" \
                --dry-run=client -o yaml | $KUBE_CMD apply -f -
        fi
    else
        warn ".env.yaml file not found. Creating placeholder secret..."
        $KUBE_CMD create secret generic ai-assistant-secrets \
            --from-literal=LLM_API_KEY="REPLACE_WITH_YOUR_API_KEY" \
            --namespace="$NAMESPACE" \
            --dry-run=client -o yaml | $KUBE_CMD apply -f -
        warn "Please create .env.yaml file with your LLM API key"
    fi
}

# Deploy to Kubernetes/OpenShift
deploy_to_kubernetes() {
    log "Deploying to $PLATFORM cluster in namespace: $NAMESPACE"
    
    # Create namespace if it doesn't exist
    if ! $KUBE_CMD get namespace "$NAMESPACE" &> /dev/null; then
        log "Creating namespace: $NAMESPACE"
        $KUBE_CMD create namespace "$NAMESPACE"
    fi
    
    # Create or update secrets from .env.yaml
    create_secrets
    
    # Create configuration ConfigMap from config file
    log "Creating configuration ConfigMap..."
    if [ -f "config/assistant-config.yaml" ]; then
        log "Found config file, creating ConfigMap..."
        if $KUBE_CMD create configmap ai-assistant-config \
            --from-file=assistant-config.yaml=config/assistant-config.yaml \
            --namespace="$NAMESPACE" \
            --dry-run=client -o yaml | $KUBE_CMD apply -f -; then
            success "ConfigMap created/updated successfully"
        else
            error "Failed to create ConfigMap"
        fi
    else
        error "Configuration file config/assistant-config.yaml not found"
    fi
    
    # Apply Kubernetes manifests in order
    log "Applying Kubernetes manifests..."
    
    # First apply ServiceAccount and RBAC resources
    if [ -f "k8s/serviceaccount.yaml" ]; then
        log "Applying ServiceAccount and RBAC resources..."
        sed "s/NAMESPACE_PLACEHOLDER/$NAMESPACE/g" "k8s/serviceaccount.yaml" | $KUBE_CMD apply -n "$NAMESPACE" -f -
    fi
    
    # Then apply other manifests except serviceaccount (already applied)
    for manifest in k8s/*.yaml; do
        if [ -f "$manifest" ] && [[ "$manifest" != *"serviceaccount.yaml" ]]; then
            log "Applying $manifest..."
            # Replace namespace placeholder in deployment.yaml for image registry path
            if [[ "$manifest" == *"deployment.yaml" ]]; then
                sed "s/NAMESPACE_PLACEHOLDER/$NAMESPACE/g" "$manifest" | $KUBE_CMD apply -n "$NAMESPACE" -f -
            else
                $KUBE_CMD apply -n "$NAMESPACE" -f "$manifest"
            fi
        fi
    done
    
    # Wait for deployment to be ready (only if we built a new image)
    if [ "$CONFIG_ONLY" != "true" ]; then
        log "Waiting for deployment to be ready..."
        if ! $KUBE_CMD wait --for=condition=available deployment/ai-assistant-backend \
            --namespace="$NAMESPACE" --timeout=300s; then
            error "Deployment failed to become ready within timeout"
        fi
    fi
    
    # Restart deployment to ensure latest configuration is picked up
    log "Restarting deployment to ensure latest configuration..."
    if $KUBE_CMD rollout restart deployment/ai-assistant-backend --namespace="$NAMESPACE"; then
        success "Deployment restarted successfully"
        
        # Wait for the restart to complete
        log "Waiting for rollout restart to complete..."
        if ! $KUBE_CMD rollout status deployment/ai-assistant-backend \
            --namespace="$NAMESPACE" --timeout=300s; then
            warn "Rollout restart did not complete within timeout, but deployment may still be working"
        else
            success "Rollout restart completed successfully"
        fi
    else
        warn "Failed to restart deployment, but initial deployment was successful"
    fi
    
    # Get the route or service URL
    if [ "$PLATFORM" = "OpenShift" ]; then
        ROUTE_URL=$($KUBE_CMD get route ai-assistant-backend -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")
        if [ -n "$ROUTE_URL" ]; then
            success "Deployment completed! Backend available at: https://$ROUTE_URL"
        fi
    else
        SERVICE_IP=$($KUBE_CMD get service ai-assistant-backend -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        if [ -n "$SERVICE_IP" ]; then
            success "Deployment completed! Backend available at: http://$SERVICE_IP:8080"
        else
            success "Deployment completed! Use port-forward to access: $KUBE_CMD port-forward service/ai-assistant-backend 8080:8080 -n $NAMESPACE"
        fi
    fi
}

# Print final status and next steps
print_status() {
    log "Deployment completed! Configuration:"
    echo ""
    echo "  Namespace: $NAMESPACE"
    echo "  Platform: $PLATFORM"
    echo "  Config-only mode: $CONFIG_ONLY"
    echo ""
    
    if [ "$CONFIG_ONLY" = "true" ]; then
        log "Configuration updated. To rebuild the container image, run:"
        echo "  $0 --namespace $NAMESPACE"
    else
        log "Next steps:"
        echo ""
        echo "1. Access your workshop site:"
        if [ -d "www" ]; then
            echo "   Static site: file://$(pwd)/www/index.html"
        fi
        echo ""
        echo "2. Monitor the deployment:"
        echo "   $KUBE_CMD get pods -n $NAMESPACE"
        echo "   $KUBE_CMD logs -n $NAMESPACE deployment/ai-assistant-backend -f"
        echo ""
        echo "3. Update API key if needed:"
        echo "   Edit .env.yaml and re-run: $0 --namespace $NAMESPACE --config-only"
        echo ""
        echo "4. Troubleshooting:"
        echo "   $KUBE_CMD describe deployment ai-assistant-backend -n $NAMESPACE"
        echo "   $KUBE_CMD get events -n $NAMESPACE --sort-by='.lastTimestamp'"
    fi
}

# Main execution
main() {
    log "Starting Showroom AI Assistant deployment..."
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Execute deployment steps
    check_prerequisites
    validate_content_source
    build_site
    build_backend_image
    deploy_to_kubernetes
    print_status
    
    success "Deployment process completed!"
}

# Run main function with all arguments
main "$@"