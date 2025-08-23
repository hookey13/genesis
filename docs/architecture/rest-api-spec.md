# REST API Spec

While GENESIS is primarily a terminal-based system, we need internal REST APIs for monitoring, testing, and emergency control. These APIs are NOT exposed publicly - they're only accessible via SSH tunnel or VPN.

```yaml
openapi: 3.0.0
info:
  title: GENESIS Internal Control API
  version: 1.0.0
  description: Internal API for monitoring and emergency control. NOT PUBLIC - VPN access only.
servers:
  - url: http://localhost:8080
    description: Local access only (via SSH tunnel)
  - url: http://10.0.0.2:8080
    description: Private VPC access (Tailscale VPN required at $5k+)

security:
  - ApiKeyAuth: []

paths:
  /health:
    get:
      summary: System health check
      operationId: getHealth
      responses:
        '200':
          description: System healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthStatus'
        '503':
          description: System unhealthy

  /account/status:
    get:
      summary: Get current account status and tier
      operationId: getAccountStatus
      responses:
        '200':
          description: Account information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AccountStatus'

  /positions:
    get:
      summary: List all positions (open and closed)
      operationId: getPositions
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [open, closed, all]
          default: open
      responses:
        '200':
          description: Position list
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Position'

  /emergency/halt:
    post:
      summary: Emergency trading halt
      operationId: emergencyHalt
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - reason
                - confirmation
              properties:
                reason:
                  type: string
                  minLength: 10
                confirmation:
                  type: string
                  pattern: '^HALT TRADING NOW$'
      responses:
        '200':
          description: Trading halted
        '400':
          description: Invalid confirmation

  /emergency/close-all:
    post:
      summary: Close all positions immediately
      operationId: closeAllPositions
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - confirmation
              properties:
                confirmation:
                  type: string
                  pattern: '^CLOSE ALL POSITIONS$'
                max_slippage_percent:
                  type: number
                  default: 1.0
      responses:
        '200':
          description: Positions closed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/CloseAllResult'

  /tilt/status:
    get:
      summary: Get current tilt detection status
      operationId: getTiltStatus
      responses:
        '200':
          description: Tilt information
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TiltStatus'

  /tilt/override:
    post:
      summary: Override tilt intervention (dangerous)
      operationId: overrideTilt
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - journal_entry
                - confirmation
              properties:
                journal_entry:
                  type: string
                  minLength: 500
                confirmation:
                  type: string
                  pattern: '^I ACCEPT THE RISKS$'
      responses:
        '200':
          description: Tilt override accepted
        '423':
          description: Override locked (too many attempts)

  /config/tier:
    get:
      summary: Get tier configuration and gates
      operationId: getTierConfig
      responses:
        '200':
          description: Tier configuration
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/TierConfig'

  /analytics/performance:
    get:
      summary: Get performance metrics
      operationId: getPerformance
      parameters:
        - name: period
          in: query
          schema:
            type: string
            enum: [day, week, month, all]
          default: day
      responses:
        '200':
          description: Performance metrics
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/PerformanceMetrics'

  /system/events:
    get:
      summary: Get recent system events (audit trail)
      operationId: getEvents
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
            maximum: 1000
        - name: after
          in: query
          schema:
            type: string
            format: date-time
      responses:
        '200':
          description: Event list
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Event'

components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key

  schemas:
    HealthStatus:
      type: object
      required:
        - status
        - uptime_seconds
        - components
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        uptime_seconds:
          type: integer
        components:
          type: object
          properties:
            exchange_gateway:
              type: string
            event_bus:
              type: string
            database:
              type: string
            websocket_streams:
              type: string

    AccountStatus:
      type: object
      required:
        - account_id
        - balance
        - tier
        - tier_started_at
      properties:
        account_id:
          type: string
          format: uuid
        balance:
          type: string
          format: decimal
        tier:
          type: string
          enum: [SNIPER, HUNTER, STRATEGIST, ARCHITECT]
        tier_started_at:
          type: string
          format: date-time
        gates_passed:
          type: array
          items:
            type: string
        locked_features:
          type: array
          items:
            type: string

    Position:
      type: object
      required:
        - position_id
        - symbol
        - side
        - entry_price
        - quantity
        - pnl_dollars
      properties:
        position_id:
          type: string
          format: uuid
        symbol:
          type: string
        side:
          type: string
          enum: [LONG, SHORT]
        entry_price:
          type: string
          format: decimal
        quantity:
          type: string
          format: decimal
        pnl_dollars:
          type: string
          format: decimal
        pnl_percent:
          type: number
        opened_at:
          type: string
          format: date-time
        closed_at:
          type: string
          format: date-time

    TiltStatus:
      type: object
      required:
        - tilt_score
        - tilt_level
        - indicators_triggered
      properties:
        tilt_score:
          type: integer
          minimum: 0
          maximum: 100
        tilt_level:
          type: string
          enum: [NORMAL, CAUTION, WARNING, LOCKED]
        indicators_triggered:
          type: array
          items:
            type: string
        intervention_active:
          type: boolean
        journal_entries_required:
          type: integer

    CloseAllResult:
      type: object
      required:
        - positions_closed
        - total_pnl
      properties:
        positions_closed:
          type: integer
        total_pnl:
          type: string
          format: decimal
        errors:
          type: array
          items:
            type: string

    TierConfig:
      type: object
      required:
        - current_tier
        - next_tier_requirements
      properties:
        current_tier:
          type: string
        next_tier_requirements:
          type: array
          items:
            type: object
            properties:
              gate:
                type: string
              current_value:
                type: string
              required_value:
                type: string
              passed:
                type: boolean

    PerformanceMetrics:
      type: object
      properties:
        total_trades:
          type: integer
        winning_trades:
          type: integer
        losing_trades:
          type: integer
        win_rate:
          type: number
        profit_factor:
          type: number
        sharpe_ratio:
          type: number
        max_drawdown:
          type: string
          format: decimal
        total_pnl:
          type: string
          format: decimal

    Event:
      type: object
      required:
        - event_id
        - event_type
        - created_at
      properties:
        event_id:
          type: string
          format: uuid
        event_type:
          type: string
        aggregate_id:
          type: string
          format: uuid
        event_data:
          type: object
        created_at:
          type: string
          format: date-time
```
