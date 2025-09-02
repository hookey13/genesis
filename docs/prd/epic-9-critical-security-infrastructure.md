# Epic 9: Critical Security & Infrastructure Hardening ($100k+ Production Grade)

**Goal:** Address all critical security vulnerabilities and infrastructure gaps identified in the production readiness assessment. Transform Genesis from a functional trading system to a bulletproof, production-grade platform capable of safely handling $100k+ capital with zero security compromises, proper database infrastructure, and enterprise-grade secret management.

## Story 9.1: Cryptographic Authentication System Overhaul
As a security architect,
I want to replace SHA256 password hashing with bcrypt and implement proper authentication,
So that user credentials are cryptographically secure and immune to rainbow table attacks.

**Acceptance Criteria:**
1. Replace all SHA256 hashing with bcrypt (cost factor 12)
2. Implement secure session management with JWT tokens
3. Add two-factor authentication (TOTP) for admin access
4. Password complexity requirements (min 12 chars, mixed case, numbers, symbols)
5. Account lockout after 5 failed attempts with exponential backoff
6. Secure password reset flow with time-limited tokens
7. API key generation with proper entropy (256 bits)
8. Session invalidation on password change
9. Audit logging for all authentication events
10. OWASP Authentication Cheat Sheet compliance

**Implementation Details:**
```python
# genesis/security/auth_manager.py
import bcrypt
import jwt
import pyotp
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from dataclasses import dataclass
import redis.asyncio as redis

@dataclass
class AuthConfig:
    bcrypt_rounds: int = 12
    jwt_secret: str = None  # Load from vault
    jwt_algorithm: str = "HS256"
    session_ttl: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    password_min_length: int = 12
    require_2fa_for_admin: bool = True
    api_key_length: int = 32  # 256 bits

class SecureAuthManager:
    def __init__(self, config: AuthConfig, redis_client: redis.Redis):
        self.config = config
        self.redis = redis_client
        self.lockout_tracker = {}
        
    async def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with proper salt rounds."""
        # Validate password complexity
        if not self._validate_password_complexity(password):
            raise ValueError("Password does not meet complexity requirements")
            
        # Generate salt and hash
        salt = bcrypt.gensalt(rounds=self.config.bcrypt_rounds)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # Store password history to prevent reuse
        await self._store_password_history(hashed)
        
        return hashed.decode('utf-8')
    
    async def verify_password(
        self, 
        username: str, 
        password: str, 
        hashed: str,
        require_2fa: bool = False
    ) -> bool:
        """Verify password with rate limiting and 2FA."""
        # Check lockout status
        if await self._is_locked_out(username):
            raise AuthenticationError("Account temporarily locked")
        
        # Verify password
        password_valid = bcrypt.checkpw(
            password.encode('utf-8'), 
            hashed.encode('utf-8')
        )
        
        if not password_valid:
            await self._record_failed_attempt(username)
            return False
        
        # Check 2FA if required
        if require_2fa:
            if not await self._verify_2fa(username):
                return False
        
        # Reset failed attempts on success
        await self._reset_failed_attempts(username)
        
        return True
    
    def generate_api_key(self) -> tuple[str, str]:
        """Generate cryptographically secure API key pair."""
        # Generate key ID and secret
        key_id = f"gk_{secrets.token_urlsafe(16)}"
        key_secret = secrets.token_urlsafe(self.config.api_key_length)
        
        # Hash secret for storage
        hashed_secret = self._hash_api_key(key_secret)
        
        return key_id, key_secret, hashed_secret
    
    def generate_jwt_token(
        self, 
        user_id: str, 
        role: str,
        additional_claims: Dict[str, Any] = None
    ) -> str:
        """Generate JWT token with proper claims."""
        now = datetime.now(timezone.utc)
        
        payload = {
            'sub': user_id,
            'role': role,
            'iat': now,
            'exp': now + timedelta(seconds=self.config.session_ttl),
            'nbf': now,
            'jti': secrets.token_urlsafe(16),  # Unique token ID
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm
        )
    
    def setup_2fa(self, username: str) -> tuple[str, str]:
        """Setup TOTP 2FA for user."""
        # Generate secret
        secret = pyotp.random_base32()
        
        # Generate provisioning URI for QR code
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=username,
            issuer_name='Genesis Trading System'
        )
        
        return secret, provisioning_uri
    
    def _validate_password_complexity(self, password: str) -> bool:
        """Validate password meets complexity requirements."""
        if len(password) < self.config.password_min_length:
            return False
            
        # Check for required character types
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        return all([has_upper, has_lower, has_digit, has_special])
```

## Story 9.2: PostgreSQL Migration & Database Infrastructure
As a database architect,
I want to migrate from SQLite to PostgreSQL with proper connection pooling,
So that the system can handle production trading loads with ACID compliance.

**Acceptance Criteria:**
1. PostgreSQL 15+ setup with proper configuration tuning
2. Connection pooling with PgBouncer (transaction mode)
3. Read replicas for analytics queries
4. Automated migration from SQLite with zero data loss
5. Partitioned tables for time-series data (orders, trades)
6. Proper indexes for all query patterns
7. Query performance monitoring with pg_stat_statements
8. Automated vacuum and analyze scheduling
9. Point-in-time recovery capability (PITR)
10. Database performance baseline (<5ms for critical queries)

**Implementation Details:**
```python
# genesis/database/postgres_manager.py
import asyncpg
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import logging
from dataclasses import dataclass

@dataclass
class PostgresConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "genesis_trading"
    user: str = "genesis"
    password: str = None  # Load from vault
    min_pool_size: int = 10
    max_pool_size: int = 50
    command_timeout: float = 10.0
    max_inactive_connection_lifetime: float = 300.0
    
    # Performance settings
    statement_cache_size: int = 1024
    max_cached_statement_lifetime: int = 3600
    
    # Read replica settings
    read_replica_host: Optional[str] = None
    read_replica_port: int = 5432

class PostgresManager:
    def __init__(self, config: PostgresConfig):
        self.config = config
        self.write_pool: Optional[asyncpg.Pool] = None
        self.read_pool: Optional[asyncpg.Pool] = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize connection pools with proper settings."""
        # Create write pool (primary)
        self.write_pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            command_timeout=self.config.command_timeout,
            max_inactive_connection_lifetime=self.config.max_inactive_connection_lifetime,
            setup=self._setup_connection,
            init=self._init_connection
        )
        
        # Create read pool (replica) if configured
        if self.config.read_replica_host:
            self.read_pool = await asyncpg.create_pool(
                host=self.config.read_replica_host,
                port=self.config.read_replica_port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                min_size=5,
                max_size=20,
                command_timeout=30.0,  # Longer timeout for analytics
                setup=self._setup_connection
            )
        else:
            self.read_pool = self.write_pool
    
    async def _setup_connection(self, connection):
        """Configure each connection with optimal settings."""
        # Set connection parameters
        await connection.execute("SET jit = 'off'")  # Disable JIT for consistent latency
        await connection.execute("SET statement_timeout = '10s'")
        await connection.execute("SET lock_timeout = '5s'")
        await connection.execute("SET idle_in_transaction_session_timeout = '30s'")
        
        # Prepare frequently used statements
        await self._prepare_statements(connection)
    
    async def _init_connection(self, connection):
        """Initialize connection with type codecs."""
        # Register custom types if needed
        await connection.set_type_codec(
            'json',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
    
    @asynccontextmanager
    async def transaction(self, isolation_level: str = 'read_committed'):
        """Execute operations in a transaction with proper isolation."""
        async with self.write_pool.acquire() as connection:
            async with connection.transaction(isolation=isolation_level):
                yield connection
    
    async def execute_write(self, query: str, *args, timeout: float = None):
        """Execute write operation on primary."""
        async with self.write_pool.acquire() as connection:
            return await connection.execute(query, *args, timeout=timeout)
    
    async def execute_read(self, query: str, *args, timeout: float = None):
        """Execute read operation on replica."""
        async with self.read_pool.acquire() as connection:
            return await connection.fetch(query, *args, timeout=timeout)
    
    async def migrate_from_sqlite(self, sqlite_path: str):
        """Migrate data from SQLite to PostgreSQL."""
        import sqlite3
        
        # Connect to SQLite
        sqlite_conn = sqlite3.connect(sqlite_path)
        sqlite_conn.row_factory = sqlite3.Row
        cursor = sqlite_conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table['name']
            if table_name.startswith('sqlite_'):
                continue
                
            self.logger.info(f"Migrating table: {table_name}")
            
            # Get table data
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            if not rows:
                continue
            
            # Prepare PostgreSQL insert
            columns = list(rows[0].keys())
            placeholders = [f'${i+1}' for i in range(len(columns))]
            
            insert_query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT DO NOTHING
            """
            
            # Batch insert with transaction
            async with self.transaction() as conn:
                # Prepare statement for efficiency
                prepared = await conn.prepare(insert_query)
                
                # Insert in batches
                batch_size = 1000
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i+batch_size]
                    values = [tuple(row[col] for col in columns) for row in batch]
                    await prepared.executemany(values)
                    
            self.logger.info(f"Migrated {len(rows)} rows for {table_name}")
        
        sqlite_conn.close()
        self.logger.info("Migration completed successfully")
    
    async def setup_partitions(self):
        """Setup table partitioning for time-series data."""
        partition_sql = """
        -- Create partitioned orders table
        CREATE TABLE IF NOT EXISTS orders (
            id BIGSERIAL,
            created_at TIMESTAMPTZ NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            side VARCHAR(4) NOT NULL,
            type VARCHAR(10) NOT NULL,
            quantity DECIMAL(20, 8) NOT NULL,
            price DECIMAL(20, 8),
            status VARCHAR(20) NOT NULL,
            exchange_order_id VARCHAR(100),
            metadata JSONB,
            PRIMARY KEY (id, created_at)
        ) PARTITION BY RANGE (created_at);
        
        -- Create monthly partitions
        CREATE TABLE IF NOT EXISTS orders_2024_01 
            PARTITION OF orders 
            FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
            
        CREATE TABLE IF NOT EXISTS orders_2024_02 
            PARTITION OF orders 
            FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
        
        -- Create indexes on partitions
        CREATE INDEX IF NOT EXISTS idx_orders_symbol_created 
            ON orders (symbol, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_orders_status 
            ON orders (status) WHERE status = 'pending';
        """
        
        await self.execute_write(partition_sql)
```

## Story 9.3: HashiCorp Vault Integration for Secrets Management
As a security engineer,
I want to integrate HashiCorp Vault for all secret management,
So that API keys and sensitive data are never exposed in code or environment variables.

**Acceptance Criteria:**
1. HashiCorp Vault deployment with auto-unseal
2. Dynamic secret generation for database credentials
3. API key encryption with envelope encryption
4. Secret rotation without service interruption
5. Audit logging for all secret access
6. Emergency break-glass procedures
7. Local development secrets with different vault
8. CI/CD integration for secret injection
9. Kubernetes secret operator integration ready
10. Compliance with CIS Vault Benchmark

**Implementation Details:**
```python
# genesis/security/vault_manager.py
import hvac
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from functools import lru_cache
import logging
from datetime import datetime, timedelta

@dataclass
class VaultConfig:
    address: str = "https://vault.genesis.internal:8200"
    token: Optional[str] = None  # Use token auth for simplicity
    namespace: str = "genesis"
    mount_point: str = "secret"
    transit_mount: str = "transit"
    database_mount: str = "database"
    
    # Auto-unseal config (AWS KMS example)
    auto_unseal_enabled: bool = True
    kms_key_id: Optional[str] = None
    
    # Caching
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 100

class VaultManager:
    def __init__(self, config: VaultConfig):
        self.config = config
        self.client = None
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        self._cache_timestamps = {}
        
    async def initialize(self):
        """Initialize Vault client with proper authentication."""
        self.client = hvac.Client(
            url=self.config.address,
            token=self.config.token,
            namespace=self.config.namespace
        )
        
        # Verify authentication
        if not self.client.is_authenticated():
            raise Exception("Vault authentication failed")
        
        # Initialize transit engine for encryption
        await self._init_transit_engine()
        
        # Setup database secret engine
        await self._init_database_engine()
    
    async def get_secret(self, path: str, field: Optional[str] = None) -> Any:
        """Retrieve secret from Vault with caching."""
        cache_key = f"{path}:{field}" if field else path
        
        # Check cache
        if self._is_cached(cache_key):
            return self._cache[cache_key]
        
        try:
            # Read from Vault
            response = self.client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.config.mount_point
            )
            
            data = response['data']['data']
            
            # Extract specific field if requested
            result = data.get(field) if field else data
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Audit log
            self.logger.info(
                "secret_accessed",
                path=path,
                field=field,
                timestamp=datetime.utcnow().isoformat()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret: {e}")
            raise
    
    async def store_secret(
        self, 
        path: str, 
        data: Dict[str, Any],
        cas: Optional[int] = None
    ):
        """Store secret in Vault with check-and-set."""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=data,
                mount_point=self.config.mount_point,
                cas=cas  # Prevents concurrent modifications
            )
            
            # Invalidate cache
            self._invalidate_cache(path)
            
            self.logger.info(f"Secret stored: {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to store secret: {e}")
            raise
    
    async def encrypt_data(self, plaintext: str, context: Optional[str] = None) -> str:
        """Encrypt data using Transit engine."""
        import base64
        
        # Encode plaintext
        encoded = base64.b64encode(plaintext.encode()).decode()
        
        # Encrypt using transit engine
        response = self.client.secrets.transit.encrypt_data(
            name='genesis-key',
            plaintext=encoded,
            context=context,
            mount_point=self.config.transit_mount
        )
        
        return response['data']['ciphertext']
    
    async def decrypt_data(self, ciphertext: str, context: Optional[str] = None) -> str:
        """Decrypt data using Transit engine."""
        import base64
        
        # Decrypt using transit engine
        response = self.client.secrets.transit.decrypt_data(
            name='genesis-key',
            ciphertext=ciphertext,
            context=context,
            mount_point=self.config.transit_mount
        )
        
        # Decode plaintext
        decoded = base64.b64decode(response['data']['plaintext']).decode()
        
        return decoded
    
    async def get_database_credentials(self, role: str = 'genesis-app') -> Dict[str, str]:
        """Get dynamic database credentials."""
        response = self.client.secrets.database.generate_credentials(
            name=role,
            mount_point=self.config.database_mount
        )
        
        return {
            'username': response['data']['username'],
            'password': response['data']['password'],
            'ttl': response['lease_duration']
        }
    
    async def rotate_encryption_key(self):
        """Rotate transit encryption key."""
        self.client.secrets.transit.rotate_encryption_key(
            name='genesis-key',
            mount_point=self.config.transit_mount
        )
        
        self.logger.info("Encryption key rotated")
    
    async def setup_api_key_encryption(self):
        """Setup encryption for API keys."""
        policies = {
            'policy': {
                'derived': True,  # Enable key derivation
                'convergent_encryption': True,  # Same plaintext = same ciphertext
                'min_decryption_version': 1,
                'min_encryption_version': 1,
                'deletion_allowed': False
            }
        }
        
        self.client.secrets.transit.create_key(
            name='api-keys',
            convergent_encryption=True,
            derived=True,
            mount_point=self.config.transit_mount
        )
    
    def _is_cached(self, key: str) -> bool:
        """Check if cache entry is valid."""
        if key not in self._cache:
            return False
            
        timestamp = self._cache_timestamps.get(key, 0)
        age = datetime.utcnow().timestamp() - timestamp
        
        return age < self.config.cache_ttl
    
    def _cache_result(self, key: str, value: Any):
        """Cache result with timestamp."""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.utcnow().timestamp()
        
        # Enforce cache size limit
        if len(self._cache) > self.config.max_cache_size:
            # Remove oldest entry
            oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get)
            del self._cache[oldest_key]
            del self._cache_timestamps[oldest_key]
```

## Story 9.4: Comprehensive Load Testing & Performance Validation
As a performance engineer,
I want comprehensive load testing that validates system behavior under extreme conditions,
So that we know exact breaking points and can handle 100x normal load.

**Acceptance Criteria:**
1. Load testing with Locust simulating 1000+ concurrent users
2. Order processing throughput >1000 orders/second
3. WebSocket connection stability with 10,000 connections
4. Database query performance <5ms p99 under load
5. Memory leak detection over 48-hour runs
6. CPU profiling identifying hot paths
7. Network latency simulation (100ms-500ms)
8. Chaos testing with random failures
9. Graceful degradation validation
10. Performance regression detection in CI/CD

**Implementation Details:**
```python
# tests/load/comprehensive_load_test.py
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import asyncio
import websockets
import json
import random
import time
from typing import List, Dict, Any
import psutil
import tracemalloc

class TradingSystemLoadTest(FastHttpUser):
    """Comprehensive load test simulating real trading patterns."""
    
    wait_time = between(0.1, 1.0)  # Aggressive trading
    
    def on_start(self):
        """Login and setup before testing."""
        # Authenticate
        response = self.client.post("/auth/login", json={
            "username": f"trader_{random.randint(1, 1000)}",
            "password": "test_password_123"
        })
        
        if response.status_code == 200:
            self.token = response.json()['token']
            self.headers = {'Authorization': f'Bearer {self.token}'}
        else:
            self.headers = {}
        
        # Initialize test data
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
        self.order_types = ['market', 'limit']
        self.sides = ['buy', 'sell']
    
    @task(10)
    def place_order(self):
        """Test order placement endpoint."""
        order_data = {
            'symbol': random.choice(self.symbols),
            'side': random.choice(self.sides),
            'type': random.choice(self.order_types),
            'quantity': round(random.uniform(0.001, 1.0), 4),
            'price': round(random.uniform(40000, 50000), 2) if random.random() > 0.5 else None
        }
        
        with self.client.post(
            "/trading/orders",
            json=order_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code != 201:
                response.failure(f"Failed to place order: {response.text}")
            elif response.elapsed.total_seconds() > 0.05:  # 50ms threshold
                response.failure(f"Order placement too slow: {response.elapsed.total_seconds()}")
    
    @task(5)
    def get_positions(self):
        """Test position retrieval endpoint."""
        with self.client.get(
            "/trading/positions",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed to get positions: {response.status_code}")
    
    @task(8)
    def get_orderbook(self):
        """Test orderbook endpoint."""
        symbol = random.choice(self.symbols)
        
        with self.client.get(
            f"/market/orderbook/{symbol}",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed to get orderbook: {response.status_code}")
    
    @task(2)
    def cancel_order(self):
        """Test order cancellation."""
        # First, get open orders
        response = self.client.get("/trading/orders/open", headers=self.headers)
        
        if response.status_code == 200 and response.json():
            orders = response.json()
            if orders:
                order_id = random.choice(orders)['id']
                
                with self.client.delete(
                    f"/trading/orders/{order_id}",
                    headers=self.headers,
                    catch_response=True
                ) as response:
                    if response.status_code not in [200, 204]:
                        response.failure(f"Failed to cancel order: {response.status_code}")

class WebSocketLoadTest:
    """WebSocket connection load testing."""
    
    def __init__(self, num_connections: int = 1000):
        self.num_connections = num_connections
        self.connections: List[websockets.WebSocketClientProtocol] = []
        self.metrics = {
            'connected': 0,
            'disconnected': 0,
            'messages_received': 0,
            'errors': 0
        }
    
    async def connect_client(self, client_id: int):
        """Establish WebSocket connection."""
        uri = f"ws://localhost:8000/ws/market?client_id={client_id}"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.metrics['connected'] += 1
                self.connections.append(websocket)
                
                # Subscribe to market data
                await websocket.send(json.dumps({
                    'action': 'subscribe',
                    'channels': ['ticker', 'orderbook', 'trades'],
                    'symbols': ['BTC/USDT', 'ETH/USDT']
                }))
                
                # Receive messages
                while True:
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=30.0
                        )
                        self.metrics['messages_received'] += 1
                        
                        # Occasionally send ping
                        if random.random() < 0.01:
                            await websocket.send(json.dumps({'action': 'ping'}))
                            
                    except asyncio.TimeoutError:
                        break
                        
        except Exception as e:
            self.metrics['errors'] += 1
            print(f"WebSocket error for client {client_id}: {e}")
        finally:
            self.metrics['disconnected'] += 1
    
    async def run_test(self):
        """Run WebSocket load test."""
        print(f"Starting WebSocket load test with {self.num_connections} connections")
        
        # Create connection tasks
        tasks = [
            self.connect_client(i) 
            for i in range(self.num_connections)
        ]
        
        # Run with controlled concurrency
        semaphore = asyncio.Semaphore(100)  # Max 100 concurrent connections
        
        async def limited_connect(client_id):
            async with semaphore:
                await self.connect_client(client_id)
        
        limited_tasks = [limited_connect(i) for i in range(self.num_connections)]
        
        # Execute
        await asyncio.gather(*limited_tasks, return_exceptions=True)
        
        # Report results
        print(f"WebSocket Load Test Results:")
        print(f"  Connections established: {self.metrics['connected']}")
        print(f"  Messages received: {self.metrics['messages_received']}")
        print(f"  Errors: {self.metrics['errors']}")

class MemoryLeakDetector:
    """Detect memory leaks during load testing."""
    
    def __init__(self):
        self.baseline_memory = None
        self.snapshots = []
        tracemalloc.start()
    
    def take_snapshot(self, label: str):
        """Take memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.snapshots.append({
            'label': label,
            'timestamp': time.time(),
            'memory_mb': memory_usage,
            'snapshot': snapshot
        })
        
        if self.baseline_memory is None:
            self.baseline_memory = memory_usage
        
        return memory_usage
    
    def analyze_growth(self) -> Dict[str, Any]:
        """Analyze memory growth patterns."""
        if len(self.snapshots) < 2:
            return {}
        
        # Compare first and last snapshot
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        growth_mb = last['memory_mb'] - first['memory_mb']
        duration_hours = (last['timestamp'] - first['timestamp']) / 3600
        growth_rate = growth_mb / duration_hours if duration_hours > 0 else 0
        
        # Get top memory consumers
        top_stats = last['snapshot'].statistics('lineno')[:10]
        
        return {
            'total_growth_mb': growth_mb,
            'growth_rate_mb_per_hour': growth_rate,
            'duration_hours': duration_hours,
            'potential_leak': growth_rate > 10,  # >10MB/hour growth
            'top_consumers': [
                {
                    'file': stat.traceback.format()[0],
                    'size_mb': stat.size / 1024 / 1024
                }
                for stat in top_stats
            ]
        }

# Performance monitoring during tests
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Initialize performance monitoring."""
    print("Starting performance monitoring...")
    environment.memory_detector = MemoryLeakDetector()
    environment.memory_detector.take_snapshot("test_start")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Analyze performance results."""
    print("Analyzing performance results...")
    
    # Memory analysis
    detector = environment.memory_detector
    detector.take_snapshot("test_end")
    analysis = detector.analyze_growth()
    
    print(f"\nMemory Analysis:")
    print(f"  Total growth: {analysis.get('total_growth_mb', 0):.2f} MB")
    print(f"  Growth rate: {analysis.get('growth_rate_mb_per_hour', 0):.2f} MB/hour")
    
    if analysis.get('potential_leak'):
        print("  ⚠️  POTENTIAL MEMORY LEAK DETECTED")
    
    # Performance summary
    stats = environment.stats
    print(f"\nPerformance Summary:")
    print(f"  Total requests: {stats.total.num_requests}")
    print(f"  Failure rate: {stats.total.fail_ratio:.2%}")
    print(f"  Avg response time: {stats.total.avg_response_time:.2f}ms")
    print(f"  P99 response time: {stats.total.get_response_time_percentile(0.99):.2f}ms")
```

## Story 9.5: Disaster Recovery & Business Continuity
As a reliability engineer,
I want comprehensive disaster recovery with automated failover,
So that the system can recover from any failure within 5 minutes with zero data loss.

**Acceptance Criteria:**
1. Automated database backups every 15 minutes
2. Point-in-time recovery to any second in last 7 days
3. Cross-region backup replication
4. Automated failover with <5 minute RTO
5. Zero data loss (RPO = 0) for committed transactions
6. Disaster recovery drills automated monthly
7. Backup integrity verification daily
8. Configuration backup and version control
9. Runbook automation for common failures
10. Post-incident analysis automation

**Implementation Details:**
```python
# genesis/reliability/disaster_recovery.py
import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import boto3
import asyncpg
from pathlib import Path
import hashlib
import json

@dataclass
class DRConfig:
    backup_interval_minutes: int = 15
    retention_days: int = 7
    cross_region_replication: bool = True
    backup_regions: List[str] = None
    s3_bucket: str = "genesis-backups"
    encryption_key_id: str = None
    max_parallel_uploads: int = 4
    backup_verification_enabled: bool = True

class DisasterRecoveryManager:
    def __init__(self, config: DRConfig, db_manager: PostgresManager):
        self.config = config
        self.db = db_manager
        self.s3 = boto3.client('s3')
        self.logger = logging.getLogger(__name__)
        self.backup_history = []
        
    async def start_automated_backups(self):
        """Start automated backup schedule."""
        while True:
            try:
                await self.perform_backup()
                await asyncio.sleep(self.config.backup_interval_minutes * 60)
            except Exception as e:
                self.logger.error(f"Backup failed: {e}")
                # Alert operations team
                await self.send_alert(
                    severity='critical',
                    message=f"Backup failed: {e}"
                )
                # Retry after shorter interval
                await asyncio.sleep(60)
    
    async def perform_backup(self) -> Dict[str, Any]:
        """Perform comprehensive system backup."""
        backup_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_metadata = {
            'id': backup_id,
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'automated',
            'components': []
        }
        
        try:
            # 1. Database backup with pg_dump
            db_backup = await self._backup_database(backup_id)
            backup_metadata['components'].append(db_backup)
            
            # 2. Configuration backup
            config_backup = await self._backup_configuration(backup_id)
            backup_metadata['components'].append(config_backup)
            
            # 3. Secrets backup (encrypted)
            secrets_backup = await self._backup_secrets(backup_id)
            backup_metadata['components'].append(secrets_backup)
            
            # 4. State backup (positions, orders)
            state_backup = await self._backup_trading_state(backup_id)
            backup_metadata['components'].append(state_backup)
            
            # 5. Upload to S3 with encryption
            await self._upload_to_s3(backup_id, backup_metadata)
            
            # 6. Cross-region replication
            if self.config.cross_region_replication:
                await self._replicate_backup(backup_id)
            
            # 7. Verify backup integrity
            if self.config.backup_verification_enabled:
                await self._verify_backup(backup_id)
            
            # 8. Clean old backups
            await self._cleanup_old_backups()
            
            self.backup_history.append(backup_metadata)
            self.logger.info(f"Backup completed: {backup_id}")
            
            return backup_metadata
            
        except Exception as e:
            self.logger.error(f"Backup failed for {backup_id}: {e}")
            raise
    
    async def _backup_database(self, backup_id: str) -> Dict[str, Any]:
        """Backup PostgreSQL database."""
        backup_file = f"/tmp/genesis_db_{backup_id}.sql"
        
        # Use pg_dump with custom format for faster restore
        cmd = [
            'pg_dump',
            '-h', self.db.config.host,
            '-p', str(self.db.config.port),
            '-U', self.db.config.user,
            '-d', self.db.config.database,
            '-F', 'custom',  # Custom format for parallel restore
            '-Z', '9',  # Maximum compression
            '-f', backup_file,
            '--no-owner',
            '--no-acl'
        ]
        
        # Set password via environment
        env = {'PGPASSWORD': self.db.config.password}
        
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Database backup failed: {result.stderr}")
        
        # Calculate checksum
        checksum = self._calculate_checksum(backup_file)
        
        return {
            'type': 'database',
            'file': backup_file,
            'size': Path(backup_file).stat().st_size,
            'checksum': checksum,
            'compression': 'gzip-9'
        }
    
    async def restore_from_backup(self, backup_id: str, target_time: Optional[datetime] = None):
        """Restore system from backup with point-in-time recovery."""
        self.logger.info(f"Starting restore from backup: {backup_id}")
        
        try:
            # 1. Download backup from S3
            backup_metadata = await self._download_backup(backup_id)
            
            # 2. Verify backup integrity
            if not await self._verify_backup_integrity(backup_metadata):
                raise Exception("Backup integrity check failed")
            
            # 3. Stop trading engine
            await self._stop_trading_engine()
            
            # 4. Restore database
            if target_time:
                # Point-in-time recovery using WAL
                await self._restore_database_pitr(backup_metadata, target_time)
            else:
                await self._restore_database(backup_metadata)
            
            # 5. Restore configuration
            await self._restore_configuration(backup_metadata)
            
            # 6. Restore secrets
            await self._restore_secrets(backup_metadata)
            
            # 7. Restore trading state
            await self._restore_trading_state(backup_metadata)
            
            # 8. Validate restoration
            if not await self._validate_restoration():
                raise Exception("Restoration validation failed")
            
            # 9. Restart trading engine
            await self._start_trading_engine()
            
            self.logger.info(f"Restore completed successfully from {backup_id}")
            
            return {
                'status': 'success',
                'backup_id': backup_id,
                'restore_time': datetime.utcnow().isoformat(),
                'target_time': target_time.isoformat() if target_time else None
            }
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            # Attempt to restore to previous known good state
            await self._emergency_recovery()
            raise
    
    async def perform_failover(self, target_region: str):
        """Perform automated failover to another region."""
        self.logger.info(f"Initiating failover to {target_region}")
        
        failover_steps = [
            ('Stop primary region services', self._stop_primary_services),
            ('Promote standby database', self._promote_standby_database),
            ('Update DNS records', self._update_dns_records),
            ('Start services in target region', self._start_target_services),
            ('Verify failover', self._verify_failover)
        ]
        
        results = []
        
        for step_name, step_func in failover_steps:
            try:
                self.logger.info(f"Executing: {step_name}")
                result = await step_func(target_region)
                results.append({
                    'step': step_name,
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                self.logger.error(f"Failover step failed: {step_name} - {e}")
                results.append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
                
                # Attempt rollback
                await self._rollback_failover(results)
                raise
        
        return {
            'status': 'completed',
            'target_region': target_region,
            'timestamp': datetime.utcnow().isoformat(),
            'steps': results
        }
    
    async def test_disaster_recovery(self) -> Dict[str, Any]:
        """Automated DR drill to test recovery procedures."""
        self.logger.info("Starting disaster recovery drill")
        
        test_results = {
            'start_time': datetime.utcnow().isoformat(),
            'tests': []
        }
        
        # Test 1: Backup creation
        backup_test = await self._test_backup_creation()
        test_results['tests'].append(backup_test)
        
        # Test 2: Restore to test environment
        restore_test = await self._test_restoration()
        test_results['tests'].append(restore_test)
        
        # Test 3: Failover simulation
        failover_test = await self._test_failover()
        test_results['tests'].append(failover_test)
        
        # Test 4: Data integrity verification
        integrity_test = await self._test_data_integrity()
        test_results['tests'].append(integrity_test)
        
        test_results['end_time'] = datetime.utcnow().isoformat()
        test_results['status'] = 'passed' if all(
            t['status'] == 'passed' for t in test_results['tests']
        ) else 'failed'
        
        # Generate report
        await self._generate_dr_report(test_results)
        
        return test_results
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
```

## Story 9.6: Production Monitoring & Observability
As a site reliability engineer,
I want comprehensive monitoring with full observability,
So that we can detect and resolve issues before they impact trading.

**Acceptance Criteria:**
1. Prometheus metrics for all critical operations
2. Grafana dashboards with drill-down capability
3. Distributed tracing with Jaeger/Zipkin
4. Centralized logging with ELK stack
5. Real-time alerting with PagerDuty integration
6. SLI/SLO tracking with error budgets
7. Custom metrics for trading performance
8. Anomaly detection with machine learning
9. Capacity planning projections
10. Executive dashboard with KPIs

**Implementation Details:**
```python
# genesis/monitoring/observability.py
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.aiohttp import AioHttpInstrumentor
import structlog
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta

# Metrics definitions
order_counter = Counter(
    'genesis_orders_total',
    'Total number of orders',
    ['exchange', 'symbol', 'side', 'type', 'status']
)

order_latency = Histogram(
    'genesis_order_latency_seconds',
    'Order execution latency',
    ['exchange', 'order_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

trading_pnl = Gauge(
    'genesis_trading_pnl_usdt',
    'Trading P&L in USDT',
    ['strategy', 'symbol']
)

active_positions = Gauge(
    'genesis_active_positions',
    'Number of active positions',
    ['symbol', 'side']
)

error_rate = Summary(
    'genesis_error_rate',
    'Error rate by component',
    ['component', 'error_type']
)

system_info = Info(
    'genesis_system',
    'System information'
)

class ObservabilityManager:
    def __init__(self):
        self.logger = structlog.get_logger()
        self._init_tracing()
        self._init_metrics()
        self._init_logging()
        self.slo_tracker = SLOTracker()
        
    def _init_tracing(self):
        """Initialize distributed tracing."""
        # Setup Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name='localhost',
            agent_port=6831,
            max_tag_value_length=1024
        )
        
        # Create tracer provider
        provider = TracerProvider()
        processor = BatchSpanProcessor(jaeger_exporter)
        provider.add_span_processor(processor)
        
        # Set global tracer provider
        trace.set_tracer_provider(provider)
        
        # Auto-instrument libraries
        AioHttpInstrumentor().instrument()
        
        self.tracer = trace.get_tracer(__name__)
    
    def _init_metrics(self):
        """Initialize metrics collection."""
        # System info
        system_info.info({
            'version': '1.0.0',
            'environment': 'production',
            'region': 'us-east-1'
        })
    
    def _init_logging(self):
        """Initialize structured logging."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    ]
                ),
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def track_order(
        self,
        order_id: str,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        status: str,
        latency: float
    ):
        """Track order metrics."""
        # Update counters
        order_counter.labels(
            exchange=exchange,
            symbol=symbol,
            side=side,
            type=order_type,
            status=status
        ).inc()
        
        # Record latency
        order_latency.labels(
            exchange=exchange,
            order_type=order_type
        ).observe(latency)
        
        # Log with structure
        self.logger.info(
            "order_executed",
            order_id=order_id,
            exchange=exchange,
            symbol=symbol,
            side=side,
            type=order_type,
            status=status,
            latency_ms=latency * 1000
        )
    
    def track_position(self, positions: List[Dict[str, Any]]):
        """Track position metrics."""
        # Reset gauges
        active_positions._metrics.clear()
        
        for position in positions:
            active_positions.labels(
                symbol=position['symbol'],
                side=position['side']
            ).set(position['quantity'])
    
    def track_pnl(self, strategy: str, symbol: str, pnl: float):
        """Track P&L metrics."""
        trading_pnl.labels(
            strategy=strategy,
            symbol=symbol
        ).set(pnl)
    
    async def trace_operation(self, operation_name: str, attributes: Dict[str, Any] = None):
        """Create trace span for operation."""
        with self.tracer.start_as_current_span(
            operation_name,
            attributes=attributes or {}
        ) as span:
            try:
                span.set_attribute("operation.start_time", datetime.utcnow().isoformat())
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                span.set_attribute("operation.end_time", datetime.utcnow().isoformat())

class SLOTracker:
    """Track Service Level Objectives and error budgets."""
    
    def __init__(self):
        self.slos = {
            'availability': {
                'target': 0.999,  # 99.9% uptime
                'window': timedelta(days=30),
                'budget_consumed': 0.0
            },
            'latency': {
                'target': 0.95,  # 95% of requests < 50ms
                'threshold_ms': 50,
                'window': timedelta(days=7),
                'budget_consumed': 0.0
            },
            'error_rate': {
                'target': 0.99,  # <1% error rate
                'window': timedelta(days=7),
                'budget_consumed': 0.0
            }
        }
        
        self.measurements = {
            'availability': [],
            'latency': [],
            'error_rate': []
        }
    
    def record_availability(self, is_available: bool):
        """Record availability measurement."""
        self.measurements['availability'].append({
            'timestamp': datetime.utcnow(),
            'value': is_available
        })
        
        self._update_error_budget('availability')
    
    def record_latency(self, latency_ms: float):
        """Record latency measurement."""
        self.measurements['latency'].append({
            'timestamp': datetime.utcnow(),
            'value': latency_ms,
            'meets_slo': latency_ms < self.slos['latency']['threshold_ms']
        })
        
        self._update_error_budget('latency')
    
    def record_error(self, is_error: bool):
        """Record error occurrence."""
        self.measurements['error_rate'].append({
            'timestamp': datetime.utcnow(),
            'value': is_error
        })
        
        self._update_error_budget('error_rate')
    
    def _update_error_budget(self, slo_type: str):
        """Update error budget consumption."""
        slo = self.slos[slo_type]
        window_start = datetime.utcnow() - slo['window']
        
        # Filter measurements within window
        recent = [
            m for m in self.measurements[slo_type]
            if m['timestamp'] > window_start
        ]
        
        if not recent:
            return
        
        if slo_type == 'availability':
            success_rate = sum(1 for m in recent if m['value']) / len(recent)
        elif slo_type == 'latency':
            success_rate = sum(1 for m in recent if m['meets_slo']) / len(recent)
        elif slo_type == 'error_rate':
            success_rate = sum(1 for m in recent if not m['value']) / len(recent)
        
        # Calculate budget consumption
        allowed_failures = 1 - slo['target']
        actual_failures = 1 - success_rate
        
        slo['budget_consumed'] = (actual_failures / allowed_failures) * 100 if allowed_failures > 0 else 0
        
        # Alert if budget exceeded
        if slo['budget_consumed'] > 100:
            self._alert_budget_exceeded(slo_type, slo['budget_consumed'])
    
    def get_slo_status(self) -> Dict[str, Any]:
        """Get current SLO status."""
        return {
            slo_type: {
                'target': slo['target'],
                'budget_consumed': slo['budget_consumed'],
                'budget_remaining': max(0, 100 - slo['budget_consumed']),
                'at_risk': slo['budget_consumed'] > 80
            }
            for slo_type, slo in self.slos.items()
        }
```

## Implementation Priority & Dependencies

```
Phase 1 (Hours 1-8): Critical Security Fixes
├── Story 9.1: Authentication System Overhaul (BLOCKING - SHA256 vulnerability)
├── Story 9.2: PostgreSQL Migration (BLOCKING - SQLite inadequate)
└── Story 9.3: Vault Integration (BLOCKING - Hardcoded credentials)

Phase 2 (Hours 9-16): Infrastructure Hardening
├── Story 9.4: Load Testing Suite
├── Story 9.5: Disaster Recovery
└── Story 9.6: Production Monitoring

Phase 3 (Hours 17-24): Validation & Testing
├── Integration testing of all components
├── Security penetration testing
└── 48-hour stability test

Phase 4 (Hours 25-32): Production Deployment
├── Staging environment validation
├── Production deployment with canary
└── Post-deployment monitoring
```

## Success Metrics

- **Security Score**: 0 critical vulnerabilities, 0 high-risk issues
- **Authentication**: Bcrypt with 2FA implemented, 0 plaintext passwords
- **Database Performance**: <5ms p99 latency, 1000+ TPS capability
- **Secret Management**: 100% secrets in Vault, 0 hardcoded credentials
- **Load Testing**: Handles 100x normal load without degradation
- **Disaster Recovery**: <5 minute RTO, 0 data loss RPO
- **Monitoring Coverage**: 100% critical path instrumentation
- **Error Budget**: <1% error rate maintained

## Risk Mitigation

1. **Migration Risk**: Parallel run old and new systems during transition
2. **Security Risk**: Immediate rotation of all credentials post-implementation
3. **Performance Risk**: Gradual rollout with canary deployments
4. **Data Loss Risk**: Triple backup strategy with cross-region replication
5. **Operational Risk**: Comprehensive runbooks and automated recovery

## Testing Requirements

Each story must include:
1. Unit tests with 100% coverage for security components
2. Integration tests for all external systems
3. Security scanning with OWASP ZAP
4. Load testing to validate performance requirements
5. Chaos engineering to test resilience
6. Penetration testing by security team

## Critical Path Items

**MUST be completed before ANY production deployment:**
1. Replace SHA256 with bcrypt (Story 9.1)
2. Remove hardcoded credentials (Story 9.3)
3. Migrate to PostgreSQL (Story 9.2)
4. Implement Vault for secrets (Story 9.3)
5. Complete load testing (Story 9.4)

## Notes

This epic addresses the **CRITICAL SECURITY VULNERABILITIES** that make the current system completely unsuitable for production. The use of SHA256 for password hashing is cryptographically broken and would fail any security audit. Hardcoded credentials in source code represent an immediate compromise risk. SQLite cannot handle production trading loads.

These are not optimizations or nice-to-haves - they are **MANDATORY** fixes that must be completed before the system touches any real money or production data. The current state would result in immediate compromise and potential total loss of funds.

The 32-hour estimate assumes focused development with no interruptions. In practice, with testing and validation, this could extend to 40-60 hours. However, given the critical nature of these vulnerabilities, this work should be the **ABSOLUTE TOP PRIORITY**.

**Remember**: "A trading system is only as strong as its weakest security link. One compromised API key means total account drainage."