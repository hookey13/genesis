# External APIs

Project GENESIS requires integration with Binance's API suite for market data and order execution. The connection resilience and retry logic are critical - a dropped connection during execution could mean the difference between profit and account destruction.

## Binance Spot Trading API
- **Purpose:** Execute spot trades, manage orders, query account balance
- **Documentation:** https://binance-docs.github.io/apidocs/spot/en/
- **Base URL(s):** 
  - REST: https://api.binance.com
  - WebSocket Streams: wss://stream.binance.com:9443
- **Authentication:** HMAC SHA256 signature with API key/secret
- **Rate Limits:** 
  - 1200 requests per minute (weight-based)
  - 50 orders per 10 seconds
  - 160,000 orders per day

**Key Endpoints Used:**
- `GET /api/v3/account` - Account balance and status
- `POST /api/v3/order` - Place new order
- `DELETE /api/v3/order` - Cancel order
- `GET /api/v3/order` - Query order status
- `GET /api/v3/depth` - Order book depth
- `GET /api/v3/klines` - Historical candlestick data
- `GET /api/v3/ticker/24hr` - 24hr price statistics

**Integration Notes:** 
- Use ccxt library's built-in rate limit handling
- Implement circuit breaker with exponential backoff (2^n seconds, max 60s)
- Maintain 3 WebSocket connections with automatic failover
- Track weight consumption, stay below 80% capacity
- Use `recvWindow` of 5000ms for time synchronization tolerance

## Binance WebSocket Streams API
- **Purpose:** Real-time market data feeds for price and order book updates
- **Documentation:** https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams
- **Base URL(s):** 
  - Public: wss://stream.binance.com:9443/ws
  - Combined: wss://stream.binance.com:9443/stream
- **Authentication:** None required for public market data
- **Rate Limits:** 
  - 5 connections per IP
  - 24 hour connection limit before forced disconnect

**Key Endpoints Used:**
- `/ws/<symbol>@trade` - Real-time trade executions
- `/ws/<symbol>@depth20@100ms` - Order book updates (20 levels, 100ms frequency)
- `/ws/<symbol>@kline_1m` - 1-minute candlestick updates
- `/ws/<symbol>@ticker` - 24hr rolling window ticker
- `/stream?streams=` - Combined streams for multiple symbols

**Integration Notes:**
- Implement heartbeat/pong every 30 seconds to keep connection alive
- Auto-reconnect with exponential backoff on disconnect
- Buffer missed messages during reconnection using REST API catchup
- Maintain separate connections for critical pairs vs monitoring pairs
- Use connection pooling: 1 for execution pairs, 1 for monitoring, 1 for backup

## Binance System Status API
- **Purpose:** Monitor exchange health and maintenance windows
- **Documentation:** https://binance-docs.github.io/apidocs/spot/en/#system-status-system
- **Base URL(s):** https://api.binance.com
- **Authentication:** None required
- **Rate Limits:** 1 request per second

**Key Endpoints Used:**
- `GET /sapi/v1/system/status` - System maintenance status
- `GET /api/v3/ping` - Test connectivity
- `GET /api/v3/time` - Server time for synchronization
- `GET /api/v3/exchangeInfo` - Trading rules and symbol information

**Integration Notes:**
- Check system status before market open
- Use server time to sync local clock (critical for signatures)
- Cache exchangeInfo for 24 hours (trading rules rarely change)
- If maintenance detected, enter MarketState.MAINTENANCE automatically
