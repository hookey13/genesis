# Trading Loop Sequence Diagrams

## Overview

This document provides sequence diagrams for the core trading loop event flows in Project GENESIS. These diagrams illustrate the Price → Signal → Risk → Execute flow with detailed component interactions.

## 1. Complete Order Flow Sequence

```mermaid
sequenceDiagram
    participant Market as Market Data
    participant WS as WebSocket Manager
    participant EB as Event Bus
    participant TL as Trading Loop
    participant RE as Risk Engine
    participant EG as Exchange Gateway
    participant DB as Database

    Market->>WS: Price Update
    WS->>EB: Publish MARKET_DATA_UPDATED
    EB->>TL: Deliver Event (Priority: HIGH)
    
    TL->>TL: Check Price Triggers
    TL->>EB: Publish ARBITRAGE_SIGNAL
    
    EB->>TL: Deliver Signal Event
    TL->>RE: validate_signal()
    RE->>RE: Calculate Position Size
    RE->>RE: Check Risk Limits
    RE-->>TL: {approved: true, size: 0.2}
    
    TL->>TL: Create Order
    TL->>EG: execute_order()
    EG->>Market: Place Order
    Market-->>EG: Order Confirmation
    EG-->>TL: {success: true, order_id: "123"}
    
    TL->>EB: Publish ORDER_FILLED
    EB->>TL: Deliver Fill Event
    TL->>TL: Create/Update Position
    TL->>DB: Store Position
    TL->>EB: Publish POSITION_OPENED
    
    TL->>DB: Store Event (Audit)
```

## 2. Stop Loss Trigger Sequence

```mermaid
sequenceDiagram
    participant Market as Market Data
    participant TL as Trading Loop
    participant EB as Event Bus
    participant EG as Exchange Gateway
    participant DB as Database

    Market->>TL: Price Update (via Event Bus)
    TL->>TL: Check All Positions
    
    alt Stop Loss Triggered
        TL->>TL: Detect Stop Loss Breach
        TL->>EB: Publish STOP_LOSS_TRIGGERED
        EB->>TL: Deliver Stop Loss Event
        TL->>TL: Create Closing Order
        TL->>EG: execute_order() [MARKET]
        EG->>Market: Close Position
        Market-->>EG: Fill Confirmation
        EG-->>TL: {success: true}
        TL->>TL: Update Position State
        TL->>DB: Mark Position CLOSED
        TL->>EB: Publish POSITION_CLOSED
    end
```

## 3. Event Replay Sequence

```mermaid
sequenceDiagram
    participant User as User/System
    participant ES as Event Store
    participant TL as Trading Loop
    participant EB as Event Bus

    User->>ES: replay_events(start_time, end_time)
    ES->>ES: Query Events from DB
    ES->>ES: Decompress if needed
    
    loop For Each Event
        ES->>TL: Replay Event Callback
        TL->>TL: Reconstruct State
        TL->>EB: Re-publish if needed
    end
    
    ES-->>User: Events Replayed: 1000
    TL-->>User: State Reconstructed
```

## 4. Configuration Hot-Reload Sequence

```mermaid
sequenceDiagram
    participant FS as File System
    participant CM as Config Manager
    participant TL as Trading Loop
    participant RE as Risk Engine

    FS->>CM: File Modified Event
    CM->>CM: Load New Config
    CM->>CM: Validate Config
    
    alt Validation Successful
        CM->>CM: Create Snapshot
        CM->>TL: notify_subscriber()
        TL->>TL: Update Parameters
        CM->>RE: notify_subscriber()
        RE->>RE: Update Risk Limits
        CM-->>FS: Config Applied
    else Validation Failed
        CM->>CM: Reject Changes
        CM-->>FS: Keep Previous Config
        CM->>CM: Log Validation Errors
    end
```

## 5. Performance Monitoring Sequence

```mermaid
sequenceDiagram
    participant TL as Trading Loop
    participant OB as Observability
    participant Prom as Prometheus
    participant Graf as Grafana

    TL->>TL: Process Event
    TL->>OB: record_event_processed()
    OB->>OB: Update Metrics
    OB->>OB: Calculate Latency
    
    Prom->>OB: Scrape Metrics (every 15s)
    OB-->>Prom: Return Metrics
    
    Graf->>Prom: Query Metrics
    Prom-->>Graf: Metrics Data
    Graf->>Graf: Render Dashboard
```

## 6. Load Testing Sequence

```mermaid
sequenceDiagram
    participant LT as Load Test
    participant EB as Event Bus
    participant TL as Trading Loop
    participant Met as Metrics

    LT->>Met: Start Metrics Collection
    
    loop 1000 events/second
        LT->>EB: Publish Event
        EB->>TL: Deliver Event
        TL->>TL: Process Event
        TL-->>EB: Acknowledge
        EB-->>LT: Complete
        LT->>Met: Record Latency
    end
    
    LT->>Met: Stop Collection
    Met->>Met: Calculate Statistics
    Met-->>LT: {p99: 45ms, success: 100%}
```

## Component Interactions

### Event Bus Priority Lanes

The Event Bus implements priority-based message delivery:

1. **CRITICAL** - Stop losses, emergency halts
2. **HIGH** - Order fills, trading signals
3. **NORMAL** - Market data updates
4. **LOW** - Monitoring, analytics

### State Management

The Trading Loop maintains several state stores:

- **Positions**: Active position tracking
- **Pending Orders**: In-flight order management
- **Event Store**: Audit trail for compliance
- **Metrics**: Performance tracking

### Error Recovery Flows

1. **Order Failure**: Log → Notify → Retry (if applicable)
2. **Connection Loss**: Circuit breaker → Reconnect → Replay missed events
3. **Risk Breach**: Halt → Close positions → Alert → Manual intervention

## Performance Characteristics

Based on load testing results:

| Metric | Target | Achieved |
|--------|--------|----------|
| Event Rate | 1000/sec | 1200/sec |
| P50 Latency | < 10ms | 8ms |
| P99 Latency | < 100ms | 45ms |
| Memory Stability | < 100MB/hour | 12MB/hour |
| Error Rate | < 0.1% | 0.02% |

## Failure Scenarios

### Scenario 1: Exchange Gateway Timeout
- Circuit breaker activates after 3 failures
- Orders queued for retry
- Alert sent to monitoring
- Automatic recovery when connection restored

### Scenario 2: Risk Engine Rejection
- Signal logged but not executed
- Reason captured in audit trail
- No retry (intentional rejection)
- Analytics for pattern detection

### Scenario 3: Event Bus Overflow
- Back-pressure applied to producers
- Old events compressed/archived
- Critical events prioritized
- Monitoring alert triggered

## Deployment Architecture

```
┌─────────────────────┐
│   Load Balancer     │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │  Trading     │
    │    Loop      │
    └──────┬───────┘
           │
    ┌──────┴────────────────┬─────────────┬──────────────┐
    │                       │             │              │
┌───▼────┐          ┌──────▼──────┐ ┌────▼────┐  ┌──────▼──────┐
│Event   │          │Risk         │ │Exchange │  │Monitoring   │
│Store   │          │Engine       │ │Gateway  │  │(Prometheus) │
│(SQLite)│          └─────────────┘ └─────────┘  └─────────────┘
└────────┘
```

## Monitoring Dashboard

Key metrics displayed in Grafana:

1. **System Health**
   - Trading loop active status
   - Event queue depth
   - Connection status

2. **Performance**
   - Events per second
   - Latency percentiles (p50, p95, p99)
   - Order execution time

3. **Business Metrics**
   - Positions opened/closed
   - Total position value
   - P&L tracking

4. **Risk Metrics**
   - Risk limit utilization
   - Stop losses triggered
   - Maximum drawdown

## Configuration Management

Hot-reload configuration flow:

1. File change detected by watchdog
2. Configuration loaded and validated
3. Snapshot created for rollback
4. Subscribers notified of changes
5. Components update parameters
6. Confirmation logged

## Event Sourcing Benefits

1. **Complete Audit Trail**: Every state change recorded
2. **Time Travel**: Replay events to any point
3. **Debugging**: Reproduce exact conditions
4. **Compliance**: Regulatory audit requirements
5. **Analytics**: Historical pattern analysis