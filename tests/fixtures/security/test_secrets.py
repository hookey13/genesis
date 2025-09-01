"""Test file with known secrets for testing purposes only.

WARNING: This file contains test secrets for validation testing.
Never use these in production!
"""

# Test secrets (DO NOT USE IN PRODUCTION)
TEST_API_KEY = "test_sk_live_1234567890abcdefghij"
TEST_PASSWORD = "test_password_123"
TEST_TOKEN = "test_bearer_token_abcdefghijklmnop"

# These should be flagged
HARDCODED_KEY = "AKIAIOSFODNN7EXAMPLE"  # AWS-like key
PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA0Z3VS... (truncated for testing)
-----END RSA PRIVATE KEY-----"""

# These should NOT be flagged (proper usage)
import os
PROPER_API_KEY = os.getenv("API_KEY")
PROPER_SECRET = os.environ.get("SECRET_KEY", "default")

# Placeholder values (should not be flagged)
EXAMPLE_KEY = "your-api-key-here"
DEMO_PASSWORD = "CHANGEME"
PLACEHOLDER_TOKEN = "${TOKEN}"