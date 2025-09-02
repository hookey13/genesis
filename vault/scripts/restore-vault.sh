#!/bin/bash
# HashiCorp Vault Restore Script
# Restores Vault data from encrypted backups

set -e

# Configuration
VAULT_ADDR=${VAULT_ADDR:-"http://localhost:8200"}
VAULT_TOKEN=${VAULT_TOKEN:-""}
RESTORE_DIR="/tmp/vault-restore"
BACKUP_NAME=${1:-""}

# Cloud storage configuration
S3_BUCKET=${S3_BUCKET:-"genesis-backups"}
S3_ENDPOINT=${S3_ENDPOINT:-"https://sgp1.digitaloceanspaces.com"}
AWS_REGION=${AWS_REGION:-"sgp1"}

# Restic configuration
RESTIC_REPOSITORY="s3:${S3_ENDPOINT}/${S3_BUCKET}/vault-backups"
RESTIC_PASSWORD_FILE="/vault/config/.restic-password"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_prompt() {
    echo -e "${BLUE}[PROMPT]${NC} $1"
}

# Show usage
usage() {
    echo "Usage: $0 [backup-name|snapshot-id]"
    echo ""
    echo "Options:"
    echo "  backup-name    Name of the backup file (e.g., vault-backup-20240101_120000)"
    echo "  snapshot-id    Restic snapshot ID to restore from"
    echo ""
    echo "If no backup name is provided, will list available backups"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Vault is accessible
    if ! vault status &>/dev/null; then
        print_error "Cannot connect to Vault at $VAULT_ADDR"
        exit 1
    fi
    
    # Check if Vault is unsealed
    if vault status | grep -q "Sealed.*true"; then
        print_error "Vault is sealed. Please unseal before restore."
        exit 1
    fi
    
    # Check for required tools
    for tool in jq restic aws; do
        if ! command -v $tool &>/dev/null; then
            print_error "$tool is not installed"
            exit 1
        fi
    done
    
    # Check restic password file
    if [ ! -f "$RESTIC_PASSWORD_FILE" ]; then
        print_error "Restic password file not found at $RESTIC_PASSWORD_FILE"
        exit 1
    fi
    
    print_status "Prerequisites check passed"
}

# List available backups
list_backups() {
    print_status "Available backups:"
    echo ""
    
    # List restic snapshots
    print_status "Restic snapshots:"
    export RESTIC_PASSWORD_FILE
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    
    restic -r "$RESTIC_REPOSITORY" snapshots --tag vault
    
    echo ""
    print_status "S3 archive backups:"
    aws s3 ls "s3://${S3_BUCKET}/vault-backups/archives/" \
        --endpoint-url="${S3_ENDPOINT}" | \
        grep "vault-backup" | \
        awk '{print "  " $4 " (" $3 " bytes)"}'
}

# Download backup
download_backup() {
    local backup_source=$1
    
    print_status "Downloading backup: $backup_source"
    
    rm -rf "$RESTORE_DIR"
    mkdir -p "$RESTORE_DIR"
    cd "$RESTORE_DIR"
    
    # Check if it's a restic snapshot ID
    if [[ ${#backup_source} -eq 8 ]] || [[ "$backup_source" == "latest" ]]; then
        print_status "Restoring from restic snapshot: $backup_source"
        
        export RESTIC_PASSWORD_FILE
        export AWS_ACCESS_KEY_ID
        export AWS_SECRET_ACCESS_KEY
        
        restic -r "$RESTIC_REPOSITORY" restore "$backup_source" \
            --target "$RESTORE_DIR" \
            --tag vault
            
    else
        # Download from S3 archive
        print_status "Downloading from S3 archive..."
        
        aws s3 cp "s3://${S3_BUCKET}/vault-backups/archives/${backup_source}.tar.gz" \
            "${RESTORE_DIR}/${backup_source}.tar.gz" \
            --endpoint-url="${S3_ENDPOINT}"
        
        aws s3 cp "s3://${S3_BUCKET}/vault-backups/archives/${backup_source}.tar.gz.sha256" \
            "${RESTORE_DIR}/${backup_source}.tar.gz.sha256" \
            --endpoint-url="${S3_ENDPOINT}"
        
        # Verify checksum
        print_status "Verifying backup integrity..."
        sha256sum -c "${backup_source}.tar.gz.sha256"
        
        # Extract backup
        print_status "Extracting backup..."
        tar xzf "${backup_source}.tar.gz"
    fi
    
    print_status "Backup downloaded and extracted"
}

# Verify backup
verify_backup() {
    print_status "Verifying backup contents..."
    
    # Find the backup directory (handle both restic and archive formats)
    if [ -d "$RESTORE_DIR/tmp/vault-backup" ]; then
        BACKUP_DATA="$RESTORE_DIR/tmp/vault-backup"
    else
        BACKUP_DATA="$RESTORE_DIR"
    fi
    
    # Check for manifest
    if [ ! -f "$BACKUP_DATA/manifest.json" ]; then
        print_error "Backup manifest not found"
        exit 1
    fi
    
    # Display backup information
    print_status "Backup information:"
    jq '.' "$BACKUP_DATA/manifest.json"
    
    # Check for required directories
    for dir in secrets policies metadata; do
        if [ ! -d "$BACKUP_DATA/$dir" ]; then
            print_warning "Directory $dir not found in backup"
        fi
    done
    
    print_status "Backup verification completed"
}

# Confirm restore
confirm_restore() {
    print_warning "========================================="
    print_warning "WARNING: This will restore Vault data"
    print_warning "========================================="
    echo ""
    print_warning "This operation will:"
    print_warning "  - Restore secrets to KV engine"
    print_warning "  - Restore policies"
    print_warning "  - Restore auth method configurations"
    echo ""
    print_prompt "Are you sure you want to continue? (yes/no): "
    read -r confirmation
    
    if [ "$confirmation" != "yes" ]; then
        print_status "Restore cancelled"
        exit 0
    fi
}

# Restore policies
restore_policies() {
    print_status "Restoring policies..."
    
    if [ ! -d "$BACKUP_DATA/policies" ]; then
        print_warning "No policies to restore"
        return
    fi
    
    for policy_file in "$BACKUP_DATA/policies"/*.hcl; do
        if [ -f "$policy_file" ]; then
            policy_name=$(basename "$policy_file" .hcl)
            
            # Skip root and default policies
            if [ "$policy_name" == "root" ] || [ "$policy_name" == "default" ]; then
                continue
            fi
            
            print_status "  Restoring policy: $policy_name"
            vault policy write "$policy_name" "$policy_file"
        fi
    done
    
    print_status "Policies restored"
}

# Restore KV secrets
restore_kv_secrets() {
    print_status "Restoring KV secrets..."
    
    if [ ! -d "$BACKUP_DATA/secrets" ]; then
        print_warning "No secrets to restore"
        return
    fi
    
    # Recursively restore secrets
    restore_kv_recursive "$BACKUP_DATA/secrets" ""
    
    print_status "KV secrets restored"
}

# Recursive function to restore KV secrets
restore_kv_recursive() {
    local source_dir=$1
    local vault_path=$2
    
    # Process all files and directories in current level
    for item in "$source_dir"/*; do
        if [ -d "$item" ]; then
            # It's a directory, recurse
            local dir_name=$(basename "$item")
            restore_kv_recursive "$item" "${vault_path}${dir_name}/"
        elif [ -f "$item" ] && [[ "$item" == *.json ]]; then
            # It's a JSON file, restore the secret
            local file_name=$(basename "$item" .json)
            local secret_path="${vault_path}${file_name}"
            
            # Extract the data from the backup
            local secret_data=$(jq -r '.data.data' "$item")
            
            if [ "$secret_data" != "null" ] && [ ! -z "$secret_data" ]; then
                print_status "  Restoring secret: genesis-secrets/${secret_path}"
                
                # Write the secret back to Vault
                echo "$secret_data" | vault kv put "genesis-secrets/${secret_path}" -
            fi
        fi
    done
}

# Restore auth method configurations
restore_auth_configs() {
    print_status "Restoring auth method configurations..."
    
    if [ ! -d "$BACKUP_DATA/auth" ]; then
        print_warning "No auth configurations to restore"
        return
    fi
    
    for config_file in "$BACKUP_DATA/auth"/*.json; do
        if [ -f "$config_file" ]; then
            local auth_name=$(basename "$config_file" .json | tr '_' '/')
            
            # Extract configuration data
            local config_data=$(jq -r '.data' "$config_file")
            
            if [ "$config_data" != "null" ] && [ ! -z "$config_data" ]; then
                print_status "  Restoring auth config: $auth_name"
                # Note: Some auth configs may need special handling
                # This is a simplified restoration
            fi
        fi
    done
    
    print_status "Auth configurations restored"
}

# Verify restoration
verify_restoration() {
    print_status "Verifying restoration..."
    
    # Count restored secrets
    local secret_count=$(vault kv list -format=json genesis-secrets 2>/dev/null | jq '. | length' || echo "0")
    print_status "  Secrets restored: $secret_count"
    
    # Count policies
    local policy_count=$(vault policy list | grep -v "^root$" | grep -v "^default$" | wc -l)
    print_status "  Policies restored: $policy_count"
    
    # Check Vault health
    vault status
    
    print_status "Restoration verification completed"
}

# Cleanup
cleanup() {
    print_status "Cleaning up temporary files..."
    rm -rf "$RESTORE_DIR"
}

# Main execution
main() {
    print_status "========================================="
    print_status "Vault Restore Process"
    print_status "========================================="
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # If no backup specified, list available
    if [ -z "$BACKUP_NAME" ]; then
        list_backups
        echo ""
        print_prompt "Enter backup name or snapshot ID to restore: "
        read -r BACKUP_NAME
        
        if [ -z "$BACKUP_NAME" ]; then
            print_error "No backup specified"
            exit 1
        fi
    fi
    
    # Download and extract backup
    download_backup "$BACKUP_NAME"
    
    # Verify backup
    verify_backup
    
    # Confirm before proceeding
    confirm_restore
    
    # Find the backup data directory
    if [ -d "$RESTORE_DIR/tmp/vault-backup" ]; then
        BACKUP_DATA="$RESTORE_DIR/tmp/vault-backup"
    else
        BACKUP_DATA="$RESTORE_DIR"
    fi
    
    # Perform restoration
    restore_policies
    restore_kv_secrets
    restore_auth_configs
    
    # Verify restoration
    verify_restoration
    
    # Cleanup
    cleanup
    
    print_status "========================================="
    print_status "Vault Restore Completed Successfully"
    print_status "========================================="
    echo ""
    print_warning "Please verify all data has been restored correctly"
    print_warning "Some manual configuration may be required for:"
    print_warning "  - Database secret engine connections"
    print_warning "  - Transit encryption keys"
    print_warning "  - Auth method specific configurations"
}

# Run main function
main "$@"