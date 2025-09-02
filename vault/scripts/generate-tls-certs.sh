#!/bin/bash
# TLS Certificate Generation for HashiCorp Vault
# Creates self-signed certificates for development and CA-signed for production

set -e

# Configuration
CERT_DIR="vault/config/certs"
CERT_DAYS=365
COUNTRY="US"
STATE="California"
LOCALITY="San Francisco"
ORGANIZATION="Genesis Trading Platform"
ORGANIZATIONAL_UNIT="Security"
COMMON_NAME="vault.genesis.local"
EMAIL="admin@genesis.local"

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

# Check if openssl is installed
if ! command -v openssl &> /dev/null; then
    print_error "OpenSSL is not installed. Please install it first."
    exit 1
fi

# Create certificate directory
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

print_status "Generating TLS certificates for Vault..."

# Generate private key for CA
print_status "Generating CA private key..."
openssl genrsa -out ca-key.pem 4096

# Generate CA certificate
print_status "Generating CA certificate..."
cat > ca-csr.conf <<EOF
[req]
default_bits = 4096
prompt = no
default_md = sha256
distinguished_name = dn

[dn]
C=$COUNTRY
ST=$STATE
L=$LOCALITY
O=$ORGANIZATION CA
OU=$ORGANIZATIONAL_UNIT
CN=Genesis Vault CA
emailAddress=$EMAIL
EOF

openssl req -new -x509 -days 3650 -key ca-key.pem -out ca-cert.pem -config ca-csr.conf

# Generate server private key
print_status "Generating Vault server private key..."
openssl genrsa -out vault-key.pem 4096

# Generate server certificate signing request
print_status "Generating Vault server CSR..."
cat > vault-csr.conf <<EOF
[req]
default_bits = 4096
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = v3_req

[dn]
C=$COUNTRY
ST=$STATE
L=$LOCALITY
O=$ORGANIZATION
OU=$ORGANIZATIONAL_UNIT
CN=$COMMON_NAME
emailAddress=$EMAIL

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = vault
DNS.3 = vault.genesis.local
DNS.4 = *.vault.genesis.local
DNS.5 = genesis-vault
DNS.6 = genesis-vault-prod
IP.1 = 127.0.0.1
IP.2 = ::1
IP.3 = 0.0.0.0
EOF

openssl req -new -key vault-key.pem -out vault-csr.pem -config vault-csr.conf

# Sign the server certificate with CA
print_status "Signing Vault server certificate with CA..."
cat > vault-cert.conf <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names
extendedKeyUsage = serverAuth, clientAuth

[alt_names]
DNS.1 = localhost
DNS.2 = vault
DNS.3 = vault.genesis.local
DNS.4 = *.vault.genesis.local
DNS.5 = genesis-vault
DNS.6 = genesis-vault-prod
IP.1 = 127.0.0.1
IP.2 = ::1
IP.3 = 0.0.0.0
EOF

openssl x509 -req -in vault-csr.pem -CA ca-cert.pem -CAkey ca-key.pem \
    -CAcreateserial -out vault-cert.pem -days $CERT_DAYS \
    -extensions v3_req -extfile vault-cert.conf

# Create combined certificate chain
print_status "Creating certificate chain..."
cat vault-cert.pem ca-cert.pem > vault-fullchain.pem

# Generate client certificate for authentication
print_status "Generating client certificate for authentication..."
openssl genrsa -out client-key.pem 4096

cat > client-csr.conf <<EOF
[req]
default_bits = 4096
prompt = no
default_md = sha256
distinguished_name = dn

[dn]
C=$COUNTRY
ST=$STATE
L=$LOCALITY
O=$ORGANIZATION
OU=$ORGANIZATIONAL_UNIT Client
CN=Genesis Vault Client
emailAddress=$EMAIL
EOF

openssl req -new -key client-key.pem -out client-csr.pem -config client-csr.conf

# Sign client certificate
openssl x509 -req -in client-csr.pem -CA ca-cert.pem -CAkey ca-key.pem \
    -CAcreateserial -out client-cert.pem -days $CERT_DAYS

# Set proper permissions
chmod 600 *-key.pem
chmod 644 *-cert.pem *.pem

# Create symbolic links for easy reference
ln -sf vault-cert.pem vault.crt
ln -sf vault-key.pem vault.key
ln -sf ca-cert.pem ca.crt

# Verify certificates
print_status "Verifying certificates..."
openssl verify -CAfile ca-cert.pem vault-cert.pem
openssl verify -CAfile ca-cert.pem client-cert.pem

# Display certificate information
print_status "Certificate details:"
echo ""
openssl x509 -in vault-cert.pem -noout -subject -dates
echo ""

# Create certificate bundle for applications
print_status "Creating certificate bundle..."
cat > ../vault-tls-bundle.sh <<'EOF'
#!/bin/bash
# Export TLS certificates for application use

export VAULT_CACERT="/vault/config/certs/ca-cert.pem"
export VAULT_CLIENT_CERT="/vault/config/certs/client-cert.pem"
export VAULT_CLIENT_KEY="/vault/config/certs/client-key.pem"
export VAULT_ADDR="https://vault.genesis.local:8200"
export VAULT_SKIP_VERIFY=false

echo "Vault TLS environment variables set:"
echo "  VAULT_CACERT=$VAULT_CACERT"
echo "  VAULT_CLIENT_CERT=$VAULT_CLIENT_CERT"
echo "  VAULT_CLIENT_KEY=$VAULT_CLIENT_KEY"
echo "  VAULT_ADDR=$VAULT_ADDR"
EOF

chmod +x ../vault-tls-bundle.sh

# Create Python TLS configuration
print_status "Creating Python TLS configuration..."
cat > ../../../genesis/security/vault_tls.py <<'EOF'
"""
TLS Configuration for Vault Client
Provides TLS certificate paths and verification settings
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

class VaultTLSConfig:
    """TLS configuration for secure Vault communication"""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize TLS configuration
        
        Args:
            base_path: Base directory for certificates
        """
        self.base_path = Path(base_path or os.getenv('VAULT_CERT_PATH', 'vault/config/certs'))
        
        # Certificate paths
        self.ca_cert = self.base_path / 'ca-cert.pem'
        self.client_cert = self.base_path / 'client-cert.pem'
        self.client_key = self.base_path / 'client-key.pem'
        self.server_cert = self.base_path / 'vault-cert.pem'
        
        # Verify certificates exist
        self._verify_certificates()
    
    def _verify_certificates(self) -> None:
        """Verify that all required certificates exist"""
        required_files = [
            (self.ca_cert, "CA certificate"),
            (self.client_cert, "Client certificate"),
            (self.client_key, "Client private key")
        ]
        
        missing = []
        for file_path, description in required_files:
            if not file_path.exists():
                missing.append(f"{description} ({file_path})")
        
        if missing:
            raise FileNotFoundError(
                f"Missing TLS certificates: {', '.join(missing)}. "
                "Run vault/scripts/generate-tls-certs.sh to generate them."
            )
    
    def get_client_cert_tuple(self) -> tuple:
        """Get client certificate tuple for requests library
        
        Returns:
            Tuple of (cert_path, key_path)
        """
        return (str(self.client_cert), str(self.client_key))
    
    def get_verify_path(self) -> str:
        """Get CA certificate path for verification
        
        Returns:
            Path to CA certificate
        """
        return str(self.ca_cert)
    
    def get_hvac_config(self) -> Dict[str, Any]:
        """Get TLS configuration for hvac client
        
        Returns:
            Dictionary with TLS settings for hvac
        """
        return {
            'verify': self.get_verify_path(),
            'cert': self.get_client_cert_tuple()
        }
    
    def get_requests_config(self) -> Dict[str, Any]:
        """Get TLS configuration for requests library
        
        Returns:
            Dictionary with TLS settings for requests
        """
        return {
            'verify': self.get_verify_path(),
            'cert': self.get_client_cert_tuple()
        }
    
    @classmethod
    def from_environment(cls) -> 'VaultTLSConfig':
        """Create TLS config from environment variables
        
        Returns:
            VaultTLSConfig instance
        """
        base_path = os.getenv('VAULT_CERT_PATH')
        return cls(base_path)
    
    def to_environment(self) -> Dict[str, str]:
        """Export TLS paths as environment variables
        
        Returns:
            Dictionary of environment variables
        """
        return {
            'VAULT_CACERT': str(self.ca_cert),
            'VAULT_CLIENT_CERT': str(self.client_cert),
            'VAULT_CLIENT_KEY': str(self.client_key),
            'VAULT_SKIP_VERIFY': 'false'
        }
EOF

# Create nginx configuration for TLS termination (optional)
print_status "Creating nginx TLS termination config..."
cat > ../nginx-vault.conf <<'EOF'
# Nginx configuration for Vault TLS termination
upstream vault_backend {
    server vault:8200;
}

server {
    listen 443 ssl http2;
    server_name vault.genesis.local;

    # TLS configuration
    ssl_certificate /etc/nginx/certs/vault-fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/vault-key.pem;
    ssl_client_certificate /etc/nginx/certs/ca-cert.pem;
    ssl_verify_client optional;

    # Modern TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:50m;
    ssl_stapling on;
    ssl_stapling_verify on;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;

    location / {
        proxy_pass http://vault_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Client-Cert $ssl_client_escaped_cert;
        
        # WebSocket support for Vault UI
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name vault.genesis.local;
    return 301 https://$server_name$request_uri;
}
EOF

# Clean up temporary files
rm -f *-csr.pem *.conf *.srl

# Display summary
cd - > /dev/null
echo ""
print_status "========================================="
print_status "TLS Certificate Generation Complete"
print_status "========================================="
echo ""
print_status "Generated certificates in: $CERT_DIR"
print_status "  - CA Certificate: ca-cert.pem"
print_status "  - CA Private Key: ca-key.pem"
print_status "  - Server Certificate: vault-cert.pem"
print_status "  - Server Private Key: vault-key.pem"
print_status "  - Client Certificate: client-cert.pem"
print_status "  - Client Private Key: client-key.pem"
print_status "  - Full Chain: vault-fullchain.pem"
echo ""
print_status "Certificate validity: $CERT_DAYS days"
print_status "Common Name: $COMMON_NAME"
echo ""
print_warning "For production use:"
print_warning "  1. Replace self-signed certificates with CA-signed ones"
print_warning "  2. Use proper DNS names instead of vault.genesis.local"
print_warning "  3. Store private keys securely"
print_warning "  4. Rotate certificates before expiry"
echo ""
print_status "To use TLS with Vault:"
print_status "  1. Update vault.hcl with TLS listener configuration"
print_status "  2. Set tls_disable = 0 in listener configuration"
print_status "  3. Restart Vault service"
print_status "  4. Access Vault at https://vault.genesis.local:8200"