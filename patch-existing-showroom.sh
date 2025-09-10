#!/bin/bash

# Patch Existing Showroom - Add AI Assistant Frontend
# This script patches an existing showroom repository with AI assistant frontend changes

set -e  # Exit on any error

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
    echo "Usage: $0 <target-repo-path>"
    echo ""
    echo "Patch an existing showroom repository with AI assistant frontend changes"
    echo ""
    echo "Arguments:"
    echo "  target-repo-path    Path to the existing showroom repository to patch"
    echo ""
    echo "Examples:"
    echo "  $0 ../my-existing-showroom"
    echo "  $0 /path/to/showroom-openshift-virt"
    echo ""
    echo "What this script does:"
    echo "  - Copies ai-chatbot.adoc with embedded AI assistant UI"
    echo "  - Enables AI assistant on all workshop pages (module-*.adoc, etc.)"
    echo "  - Copies backend files (backend/, config/, Dockerfile, .env.yaml.example, deploy-backend.sh)"
    echo "  - Creates techdocs directory structure"
    echo "  - Adds sample workshop files (SAMPLE_*)"
    echo "  - Preserves all existing workshop content"
}

# Validate arguments and target repository
validate_target() {
    local target_path="$1"
    
    if [ -z "$target_path" ]; then
        error "Target repository path is required. Use --help for usage information."
    fi
    
    if [ ! -d "$target_path" ]; then
        error "Target path does not exist or is not a directory: $target_path"
    fi
    
    # Check if it looks like a showroom repository
    if [ ! -f "$target_path/default-site.yml" ]; then
        error "Target does not appear to be a showroom repository (missing default-site.yml)"
    fi
    
    if [ ! -d "$target_path/content/modules/ROOT" ]; then
        error "Target does not have the expected content structure (missing content/modules/ROOT)"
    fi
    
    success "Target repository validated: $target_path"
}

# Validate Antora configuration
validate_antora_config() {
    local target_path="$1"
    
    log "Validating Antora configuration..."
    
    # Check if default-site.yml exists
    local site_config="$target_path/default-site.yml"
    if [ ! -f "$site_config" ]; then
        error "default-site.yml not found in target repository"
    fi
    
    # Check if antora.yml exists
    local antora_config="$target_path/content/antora.yml"
    if [ -f "$antora_config" ]; then
        log "antora.yml found - configuration looks good"
    else
        warn "antora.yml not found, this may cause issues"
    fi
    
    success "Antora configuration validated"
}


# Copy AI assistant supplemental UI
copy_supplemental_ui() {
    local target_path="$1"
    local source_path="$(pwd)"
    
    log "Copying AI assistant supplemental UI..."
    
    local source_ui_dir="$source_path/content/supplemental-ui"
    local target_ui_dir="$target_path/content/supplemental-ui"
    
    if [ ! -d "$source_ui_dir" ]; then
        error "Source supplemental-ui directory not found at: $source_ui_dir"
    fi
    
    # Create target directory if it doesn't exist
    mkdir -p "$target_ui_dir"
    
    # Copy all supplemental UI files
    log "Copying supplemental UI files from $source_ui_dir to $target_ui_dir..."
    cp -r "$source_ui_dir"/* "$target_ui_dir/"
    
    success "AI assistant supplemental UI copied successfully"
}

# Update Antora site configuration for supplemental UI
update_site_config() {
    local target_path="$1"
    
    log "Updating Antora site configuration for AI assistant..."
    
    # Check if yq is available
    if ! command -v yq &> /dev/null; then
        error "yq is required but not installed. Please install yq: https://github.com/mikefarah/yq"
    fi
    
    # Update default-site.yml to include supplemental UI
    local site_config="$target_path/default-site.yml"
    if [ -f "$site_config" ]; then
        log "Updating default-site.yml with supplemental UI configuration..."
        
        # Check if ui.supplemental_files already exists
        if yq eval '.ui.supplemental_files' "$site_config" | grep -q "null"; then
            # Create new supplemental_files array
            log "Adding supplemental_files configuration..."
            yq eval '.ui.supplemental_files = [{"path": "./content/supplemental-ui"}]' -i "$site_config"
            success "Added supplemental UI configuration to default-site.yml"
        else
            # Check if our path is already in the array
            local existing_path=$(yq eval '.ui.supplemental_files[] | select(.path == "./content/supplemental-ui") | .path' "$site_config")
            if [ -z "$existing_path" ]; then
                # Add our path to existing array
                log "Adding to existing supplemental_files configuration..."
                yq eval '.ui.supplemental_files += [{"path": "./content/supplemental-ui"}]' -i "$site_config"
                success "Added supplemental UI path to existing configuration"
            else
                log "Supplemental UI path already exists in configuration"
            fi
        fi
    else
        error "default-site.yml not found in target repository"
    fi
}

# Update antora.yml with AI assistant attributes
update_antora_yml() {
    local target_path="$1"
    
    log "Updating antora.yml with AI assistant attributes..."
    
    local antora_file="$target_path/content/antora.yml"
    if [ ! -f "$antora_file" ]; then
        error "antora.yml not found at: $antora_file"
    fi
    
    # Check if yq is available
    if ! command -v yq &> /dev/null; then
        error "yq is required but not installed. Please install yq: https://github.com/mikefarah/yq"
    fi
    
    log "Adding AI assistant attributes to antora.yml..."
    
    # Add AI assistant attributes if they don't already exist
    local attributes_to_add=(
        'assistant_name:"Workshop AI Assistant"'
        'workshop_title:"Your Workshop Title"'
        'welcome_message:"👋 Hi! I am your workshop assistant. I know all about the workshop content and can help you with questions, troubleshooting, and guidance."'
        'sample_question_1:"💬 What is the main objective of this workshop?"'
        'sample_question_2:"🔧 How do I troubleshoot common issues?"'
        'sample_question_3:"📖 Walk me through the first module"'
    )
    
    for attr in "${attributes_to_add[@]}"; do
        local key=$(echo "$attr" | cut -d':' -f1)
        local value=$(echo "$attr" | cut -d':' -f2-)
        
        # Check if attribute already exists
        if yq eval ".asciidoc.attributes.$key" "$antora_file" | grep -q "null"; then
            log "Adding attribute: $key"
            yq eval ".asciidoc.attributes.$key = $value" -i "$antora_file"
        else
            log "Attribute $key already exists, skipping"
        fi
    done
    
    success "Updated antora.yml with AI assistant attributes"
}

# Create techdocs directory structure
create_techdocs_structure() {
    local target_path="$1"
    
    log "Creating techdocs directory structure..."
    
    local techdocs_dir="$target_path/content/modules/ROOT/assets/techdocs"
    
    # Create the directory if it doesn't exist
    if [ ! -d "$techdocs_dir" ]; then
        mkdir -p "$techdocs_dir"
        log "Created directory: $techdocs_dir"
    else
        log "Directory already exists: $techdocs_dir"
    fi
    
    # Add .gitkeep file to ensure the directory is tracked in git
    local gitkeep_file="$techdocs_dir/.gitkeep"
    if [ ! -f "$gitkeep_file" ]; then
        cat > "$gitkeep_file" << 'EOF'
# This file ensures the techdocs directory is tracked in git
# Place your technical documentation PDF files in this directory
# for the AI assistant to include them in its knowledge base
EOF
        success "Created .gitkeep file in techdocs directory"
    else
        log ".gitkeep file already exists in techdocs directory"
    fi
}

# Copy backend files
copy_backend_files() {
    local target_path="$1"
    local source_path="$(pwd)"
    
    log "Copying backend files..."
    
    # List of backend files and directories to copy
    local backend_items=(
        "backend"
        "config"
        "Dockerfile"
        ".env.yaml.example"
        "deploy-backend.sh"
    )
    
    for item in "${backend_items[@]}"; do
        local source_item="$source_path/$item"
        local target_item="$target_path/$item"
        
        if [ -e "$source_item" ]; then
            if [ -e "$target_item" ]; then
                # Create backup if target already exists
                local backup_name="$target_item.backup.$(date +%s)"
                log "Backing up existing $item to $(basename "$backup_name")..."
                mv "$target_item" "$backup_name"
            fi
            
            if [ -d "$source_item" ]; then
                log "Copying directory: $item"
                cp -r "$source_item" "$target_item"
            else
                log "Copying file: $item"
                cp "$source_item" "$target_item"
            fi
            
            success "Copied $item successfully"
        else
            warn "Source $item not found: $source_item"
        fi
    done
    
    # Copy backend requirements.txt if it exists
    local source_req="$source_path/backend/requirements.txt"
    local target_req="$target_path/backend/requirements.txt"
    if [ -f "$source_req" ] && [ ! -f "$target_req" ]; then
        cp "$source_req" "$target_req"
        log "Copied backend/requirements.txt"
    fi
    
    success "Backend files copied successfully"
}

# Copy sample workshop files
copy_sample_files() {
    local target_path="$1"
    local source_path="$(pwd)"
    
    log "Copying sample workshop files..."
    
    local source_pages_dir="$source_path/content/modules/ROOT/pages"
    local target_pages_dir="$target_path/content/modules/ROOT/pages"
    
    if [ ! -d "$source_pages_dir" ]; then
        error "Source pages directory not found: $source_pages_dir"
    fi
    
    if [ ! -d "$target_pages_dir" ]; then
        error "Target pages directory not found: $target_pages_dir"
    fi
    
    # Copy only SAMPLE* files
    local sample_files_found=false
    for sample_file in "$source_pages_dir"/SAMPLE_*; do
        if [ -f "$sample_file" ]; then
            local filename=$(basename "$sample_file")
            local target_file="$target_pages_dir/$filename"
            
            if [ -f "$target_file" ]; then
                warn "Sample file already exists, skipping: $filename"
            else
                cp "$sample_file" "$target_file"
                log "Copied sample file: $filename"
                sample_files_found=true
            fi
        fi
    done
    
    if [ "$sample_files_found" = true ]; then
        success "Sample workshop files copied successfully"
    else
        warn "No new sample files to copy (they may already exist)"
    fi
}

# Enable AI assistant on all workshop pages
enable_ai_assistant_on_all_pages() {
    local target_path="$1"
    local source_path="$(pwd)"
    
    log "Enabling AI assistant on all workshop pages..."
    
    local source_ai_chatbot="$source_path/content/modules/ROOT/pages/ai-chatbot.adoc"
    local target_pages_dir="$target_path/content/modules/ROOT/pages"
    local target_ai_chatbot="$target_pages_dir/ai-chatbot.adoc"
    
    # First, copy the ai-chatbot.adoc file if it doesn't exist
    if [ ! -f "$target_ai_chatbot" ]; then
        if [ -f "$source_ai_chatbot" ]; then
            cp "$source_ai_chatbot" "$target_ai_chatbot"
            log "Copied ai-chatbot.adoc to target repository"
        else
            error "ai-chatbot.adoc not found in source: $source_ai_chatbot"
        fi
    else
        log "ai-chatbot.adoc already exists in target repository"
    fi
    
    # Find all .adoc files in the target pages directory (excluding ai-chatbot.adoc itself)
    local pages_updated=0
    for adoc_file in "$target_pages_dir"/*.adoc; do
        if [ -f "$adoc_file" ]; then
            local filename=$(basename "$adoc_file")
            
            # Skip ai-chatbot.adoc itself and any backup files
            if [ "$filename" = "ai-chatbot.adoc" ] || [[ "$filename" == *.backup.* ]]; then
                continue
            fi
            
            # Check if the include directive already exists
            if grep -q "include::ai-chatbot.adoc" "$adoc_file"; then
                log "AI assistant already enabled in: $filename"
                continue
            fi
            
            # Add the include directive at the end of the file
            log "Adding AI assistant to: $filename"
            echo "" >> "$adoc_file"
            echo "include::ai-chatbot.adoc[]" >> "$adoc_file"
            pages_updated=$((pages_updated + 1))
        fi
    done
    
    if [ $pages_updated -gt 0 ]; then
        success "AI assistant enabled on $pages_updated workshop pages"
    else
        log "No pages needed AI assistant enablement (may already be enabled)"
    fi
}


# Main execution
main() {
    local target_path="$1"
    
    # Check for help flag
    if [ "$target_path" = "--help" ] || [ "$target_path" = "-h" ]; then
        show_usage
        exit 0
    fi
    
    log "Starting showroom repository patch process..."
    
    # Validate target repository
    validate_target "$target_path"
    
    # Convert to absolute path for consistency
    target_path=$(cd "$target_path" && pwd)
    log "Target repository: $target_path"
    
    # Perform all patch operations
    validate_antora_config "$target_path"
    copy_supplemental_ui "$target_path"
    update_site_config "$target_path"
    update_antora_yml "$target_path"
    copy_backend_files "$target_path"
    create_techdocs_structure "$target_path"
    copy_sample_files "$target_path"
    enable_ai_assistant_on_all_pages "$target_path"
    
    # Final instructions
    log "Patch process completed! Next steps:"
    echo ""
    echo "1. Review the changes in your target repository:"
    echo "   cd $target_path"
    echo "   git status"
    echo ""
    echo "2. Customize the AI assistant UI in content/antora.yml:"
    echo "   # Update these attributes with your workshop-specific values:"
    echo "   #   assistant_name: \"Your Assistant Name\""
    echo "   #   workshop_title: \"Your Workshop Title\""
    echo "   #   welcome_message: \"Your custom welcome message\""
    echo "   #   sample_question_1: \"Your first sample question\""
    echo "   #   sample_question_2: \"Your second sample question\""
    echo "   #   sample_question_3: \"Your third sample question\""
    echo ""
    echo "3. Configure the AI assistant backend:"
    echo "   cp .env.yaml.example .env.yaml"
    echo "   # Edit .env.yaml with your LLM API configuration"
    echo "   # Edit config/assistant-config.yaml for workshop-specific AI behavior"
    echo ""
    echo "4. Add technical documentation PDFs to:"
    echo "   $target_path/content/modules/ROOT/assets/techdocs/"
    echo ""
    echo "5. Build and deploy your workshop with AI assistant:"
    echo "   # Deploy to your OpenShift/Kubernetes cluster"
    echo "   ./deploy-backend.sh --namespace your-namespace"
    echo ""
    echo "6. Test your workshop site with AI assistant:"
    echo "   # Frontend will be available at your workshop URL"
    echo "   # AI assistant will now appear on all workshop pages"
    echo "   # AI assistant backend will be available at /api/health"
    echo ""
    
    success "Showroom repository patching completed!"
}

# Run main function with all arguments
main "$@"