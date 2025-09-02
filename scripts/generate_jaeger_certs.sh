#!/bin/bash
# Generate TLS certificates for Jaeger production deployment

set -e

CERT_DIR="./certs"
DAYS_VALID=365
KEY_SIZE=4096

# Create certificate directory
mkdir -p $CERT_DIR

echo "Generating TLS certificates for Jaeger..."

# Generate CA private key
openssl genrsa -out $CERT_DIR/ca.key $KEY_SIZE

# Generate CA certificate
openssl req -new -x509 -days $DAYS_VALID -key $CERT_DIR/ca.key -out $CERT_DIR/ca.crt \
    -subj "/C=US/ST=State/L=City/O=Genesis Trading/OU=Infrastructure/CN=Genesis CA"

# Function to generate certificate for a service
generate_cert() {
    SERVICE=$1
    CN=$2
    
    # Generate private key
    openssl genrsa -out $CERT_DIR/${SERVICE}.key $KEY_SIZE
    
    # Generate certificate request
    openssl req -new -key $CERT_DIR/${SERVICE}.key -out $CERT_DIR/${SERVICE}.csr \
        -subj "/C=US/ST=State/L=City/O=Genesis Trading/OU=Tracing/CN=${CN}"
    
    # Sign certificate with CA
    openssl x509 -req -days $DAYS_VALID -in $CERT_DIR/${SERVICE}.csr \
        -CA $CERT_DIR/ca.crt -CAkey $CERT_DIR/ca.key -CAcreateserial \
        -out $CERT_DIR/${SERVICE}.crt
    
    # Clean up CSR
    rm $CERT_DIR/${SERVICE}.csr
    
    echo "Generated certificate for ${SERVICE}"
}

# Generate certificates for each service
generate_cert "collector" "jaeger-collector.genesis.internal"
generate_cert "query" "jaeger-query.genesis.internal"
generate_cert "agent" "jaeger-agent.genesis.internal"
generate_cert "otlp-grpc" "otlp-grpc.genesis.internal"
generate_cert "otlp-http" "otlp-http.genesis.internal"
generate_cert "elasticsearch" "elasticsearch.genesis.internal"
generate_cert "es-client" "es-client.genesis.internal"
generate_cert "curator" "curator.genesis.internal"

# Set appropriate permissions
chmod 600 $CERT_DIR/*.key
chmod 644 $CERT_DIR/*.crt

echo "TLS certificates generated successfully in $CERT_DIR/"
echo ""
echo "Certificate Summary:"
echo "==================="
echo "CA Certificate: $CERT_DIR/ca.crt"
echo "Service Certificates:"
ls -la $CERT_DIR/*.crt | grep -v ca.crt
echo ""
echo "To verify a certificate:"
echo "openssl x509 -in $CERT_DIR/collector.crt -text -noout"