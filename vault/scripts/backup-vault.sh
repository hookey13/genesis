#!/bin/bash
# HashiCorp Vault Backup Script
# Creates encrypted backups of Vault data and uploads to cloud storage

set -e

# Configuration
VAULT_ADDR=${VAULT_ADDR:-"http://localhost:8200"}
VAULT_TOKEN=${VAULT_TOKEN:-""}
BACKUP_DIR="/tmp/vault-backup"
BACKUP_RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="vault-backup-${TIMESTAMP}"

# Cloud storage configuration (DigitalOcean Spaces/AWS S3)
S3_BUCKET=${S3_BUCKET:-"genesis-backups"}
S3_ENDPOINT=${S3_ENDPOINT:-"https://sgp1.digitaloceanspaces.com"}
AWS_REGION=${AWS_REGION:-"sgp1"}

# Encryption settings
ENCRYPTION_KEY_FILE="/vault/config/.backup-encryption-key"
RESTIC_REPOSITORY="s3:${S3_ENDPOINT}/${S3_BUCKET}/vault-backups"
RESTIC_PASSWORD_FILE="/vault/config/.restic-password"

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
        print_error "Vault is sealed. Please unseal before backup."
        exit 1
    fi
    
    # Check for required tools
    for tool in jq restic aws; do
        if ! command -v $tool &>/dev/null; then
            print_error "$tool is not installed"
            exit 1
        fi
    done
    
    # Check AWS/S3 credentials
    if ! aws s3 ls s3://${S3_BUCKET} --endpoint-url=${S3_ENDPOINT} &>/dev/null; then
        print_warning "Cannot access S3 bucket. Checking credentials..."
        aws configure list
    fi
    
    print_status "Prerequisites check passed"
}

# Initialize restic repository if needed
init_restic() {
    print_status "Initializing restic repository..."
    
    # Generate restic password if not exists
    if [ ! -f "$RESTIC_PASSWORD_FILE" ]; then
        openssl rand -base64 32 > "$RESTIC_PASSWORD_FILE"
        chmod 600 "$RESTIC_PASSWORD_FILE"
        print_warning "Generated new restic password at $RESTIC_PASSWORD_FILE"
    fi
    
    # Initialize repository if not exists
    export RESTIC_PASSWORD_FILE
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    
    if ! restic -r "$RESTIC_REPOSITORY" snapshots &>/dev/null; then
        print_status "Creating new restic repository..."
        restic -r "$RESTIC_REPOSITORY" init
    fi
}

# Create backup directory
create_backup_dir() {
    print_status "Creating backup directory..."
    rm -rf "$BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cd "$BACKUP_DIR"
}

# Backup Vault data
backup_vault_data() {
    print_status "Starting Vault backup..."
    
    # Create subdirectories
    mkdir -p "$BACKUP_DIR/secrets"
    mkdir -p "$BACKUP_DIR/policies"
    mkdir -p "$BACKUP_DIR/auth"
    mkdir -p "$BACKUP_DIR/audit"
    mkdir -p "$BACKUP_DIR/metadata"
    
    # Export Vault metadata
    print_status "Exporting Vault metadata..."
    vault status -format=json > "$BACKUP_DIR/metadata/status.json"
    vault read sys/health -format=json > "$BACKUP_DIR/metadata/health.json" 2>/dev/null || true
    
    # List all mounts
    vault secrets list -format=json > "$BACKUP_DIR/metadata/secrets-engines.json"
    vault auth list -format=json > "$BACKUP_DIR/metadata/auth-methods.json"
    vault audit list -format=json > "$BACKUP_DIR/metadata/audit-devices.json" 2>/dev/null || true
    
    # Backup KV secrets from genesis-secrets engine
    print_status "Backing up KV secrets..."
    backup_kv_recursive "genesis-secrets" "" "$BACKUP_DIR/secrets"
    
    # Backup policies
    print_status "Backing up policies..."
    for policy in $(vault policy list | grep -v "^root$" | grep -v "^default$"); do
        vault policy read "$policy" > "$BACKUP_DIR/policies/${policy}.hcl"
        print_status "  Backed up policy: $policy"
    done
    
    # Backup auth method configurations
    print_status "Backing up auth method configurations..."
    for auth in $(vault auth list -format=json | jq -r 'keys[]'); do
        safe_name=$(echo "$auth" | tr '/' '_')
        vault read -format=json "auth/${auth}config" > "$BACKUP_DIR/auth/${safe_name}config.json" 2>/dev/null || true
    done
    
    # Backup Transit keys metadata (not the actual keys)
    if vault secrets list -format=json | jq -r 'keys[]' | grep -q "genesis-transit"; then
        print_status "Backing up Transit engine metadata..."
        mkdir -p "$BACKUP_DIR/transit"
        
        # List all transit keys
        transit_keys=$(vault list -format=json genesis-transit/keys 2>/dev/null | jq -r '.[]' 2>/dev/null || echo "")
        
        for key in $transit_keys; do
            vault read -format=json "genesis-transit/keys/$key" > "$BACKUP_DIR/transit/${key}.json" 2>/dev/null || true
            print_status "  Backed up transit key metadata: $key"
        done
    fi
    
    # Create backup manifest
    print_status "Creating backup manifest..."
    cat > "$BACKUP_DIR/manifest.json" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "vault_version": $(vault version -format=json | jq -r '.version' 2>/dev/null || echo '"unknown"'),
    "backup_name": "${BACKUP_NAME}",
    "backup_tool": "vault-backup.sh",
    "backup_version": "1.0.0",
    "contents": {
        "secrets": true,
        "policies": true,
        "auth_methods": true,
        "audit_devices": true,
        "metadata": true
    }
}
EOF
    
    print_status "Vault data backup completed"
}

# Recursive function to backup KV secrets
backup_kv_recursive() {
    local mount_point=$1
    local path=$2
    local output_dir=$3
    
    # List secrets at current path
    local full_path="${mount_point}${path}"
    local secrets=$(vault kv list -format=json "$full_path" 2>/dev/null | jq -r '.[]' 2>/dev/null || echo "")
    
    for secret in $secrets; do
        if [[ "$secret" == */ ]]; then
            # It's a directory, recurse
            local subdir="${path}${secret}"
            mkdir -p "${output_dir}${subdir}"
            backup_kv_recursive "$mount_point" "$subdir" "$output_dir"
        else
            # It's a secret, back it up
            local secret_path="${path}${secret}"
            vault kv get -format=json "${mount_point}${secret_path}" > "${output_dir}${secret_path}.json" 2>/dev/null || true
            print_status "  Backed up secret: ${secret_path}"
        fi
    done
}

# Compress backup
compress_backup() {
    print_status "Compressing backup..."
    cd "$BACKUP_DIR"
    tar czf "${BACKUP_NAME}.tar.gz" ./*
    
    # Calculate checksum
    sha256sum "${BACKUP_NAME}.tar.gz" > "${BACKUP_NAME}.tar.gz.sha256"
    
    print_status "Backup compressed: ${BACKUP_NAME}.tar.gz"
}

# Encrypt backup with restic
encrypt_and_upload() {
    print_status "Encrypting and uploading backup..."
    
    export RESTIC_PASSWORD_FILE
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    
    # Backup with restic
    restic -r "$RESTIC_REPOSITORY" backup \
        --tag "vault" \
        --tag "timestamp:${TIMESTAMP}" \
        --tag "auto" \
        "$BACKUP_DIR" \
        --exclude="*.tar.gz*"
    
    # Also upload tar.gz to S3 directly for redundancy
    print_status "Uploading compressed backup to S3..."
    aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" \
        "s3://${S3_BUCKET}/vault-backups/archives/${BACKUP_NAME}.tar.gz" \
        --endpoint-url="${S3_ENDPOINT}" \
        --storage-class STANDARD_IA
    
    aws s3 cp "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz.sha256" \
        "s3://${S3_BUCKET}/vault-backups/archives/${BACKUP_NAME}.tar.gz.sha256" \
        --endpoint-url="${S3_ENDPOINT}"
    
    print_status "Backup uploaded successfully"
}

# Cleanup old backups
cleanup_old_backups() {
    print_status "Cleaning up old backups..."
    
    export RESTIC_PASSWORD_FILE
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    
    # Clean up old restic snapshots
    restic -r "$RESTIC_REPOSITORY" forget \
        --keep-daily 7 \
        --keep-weekly 4 \
        --keep-monthly 3 \
        --prune
    
    # Clean up old S3 archives
    cutoff_date=$(date -d "${BACKUP_RETENTION_DAYS} days ago" +%Y%m%d)
    
    aws s3 ls "s3://${S3_BUCKET}/vault-backups/archives/" \
        --endpoint-url="${S3_ENDPOINT}" | \
    while read -r line; do
        file=$(echo "$line" | awk '{print $4}')
        file_date=$(echo "$file" | grep -oE '[0-9]{8}' | head -1)
        
        if [ ! -z "$file_date" ] && [ "$file_date" -lt "$cutoff_date" ]; then
            print_status "  Deleting old backup: $file"
            aws s3 rm "s3://${S3_BUCKET}/vault-backups/archives/$file" \
                --endpoint-url="${S3_ENDPOINT}"
        fi
    done
    
    print_status "Cleanup completed"
}

# Cleanup temporary files
cleanup_temp() {
    print_status "Cleaning up temporary files..."
    rm -rf "$BACKUP_DIR"
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    # Send to monitoring system or email
    if [ ! -z "$ALERT_EMAIL" ]; then
        echo "$message" | mail -s "Vault Backup ${status}: ${BACKUP_NAME}" "$ALERT_EMAIL"
    fi
    
    # Log to syslog
    logger -t vault-backup "${status}: ${message}"
}

# Main execution
main() {
    print_status "========================================="
    print_status "Starting Vault Backup Process"
    print_status "========================================="
    print_status "Timestamp: ${TIMESTAMP}"
    print_status "Backup name: ${BACKUP_NAME}"
    echo ""
    
    # Set error trap
    trap 'handle_error $? $LINENO' ERR
    
    # Execute backup steps
    check_prerequisites
    init_restic
    create_backup_dir
    backup_vault_data
    compress_backup
    encrypt_and_upload
    cleanup_old_backups
    cleanup_temp
    
    print_status "========================================="
    print_status "Vault Backup Completed Successfully"
    print_status "========================================="
    
    send_notification "SUCCESS" "Vault backup ${BACKUP_NAME} completed successfully"
}

# Error handler
handle_error() {
    local exit_code=$1
    local line_number=$2
    
    print_error "Backup failed at line $line_number with exit code $exit_code"
    send_notification "FAILURE" "Vault backup ${BACKUP_NAME} failed at line $line_number"
    cleanup_temp
    exit $exit_code
}

# Run main function
main "$@"