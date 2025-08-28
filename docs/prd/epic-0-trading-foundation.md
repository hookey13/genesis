# Epic 0: Trading Foundation - Core Loop Implementation

## Epic Overview
Implement the minimal viable trading loop to establish Phase 0 success and enable all downstream epics.

## Business Value
- Establishes foundation for entire trading system
- Proves core architecture with real implementation
- Enables paper trading validation
- Unblocks all subsequent epics

## Success Criteria
- Execute 10 consecutive paper trades without system failure
- Maintain 24-hour autonomous operation
- All core components communicating via Event Bus
- Risk engine validating all trades
- Terminal UI displaying live positions

## Timeline
**Target**: 2 weeks (14 days)
- Week 1: Infrastructure & Core Loop
- Week 2: Integration & Validation

## Stories

### Story 0.1: Fix Infrastructure & Testing
**Priority**: CRITICAL
**Estimate**: 2 days

**Acceptance Criteria:**
- [ ] Alembic migrations working (alembic.ini restored)
- [ ] Test imports fixed (no import errors)
- [ ] Coverage reporting operational (>0% baseline)
- [ ] CI/CD pipeline green
- [ ] Makefile commands functional

**Technical Tasks:**
- Restore alembic.ini with correct database URL
- Fix test file imports in tests/unit/
- Ensure pytest configuration correct
- Verify GitHub Actions workflow

### Story 0.2: Complete Exchange Integration
**Priority**: CRITICAL
**Estimate**: 2 days

**Acceptance Criteria:**
- [ ] WebSocket connection stable for 1 hour
- [ ] Live price feed updating in real-time
- [ ] REST API placing test orders successfully
- [ ] Circuit breaker protecting API calls
- [ ] Rate limiter enforcing limits

**Technical Tasks:**
- Complete genesis/exchange/ws_manager.py implementation
- Integrate WebSocket with Event Bus
- Test order placement on testnet
- Verify circuit breaker v2 functionality

### Story 0.3: Implement Core Trading Loop
**Priority**: CRITICAL
**Estimate**: 1 day

**Acceptance Criteria:**
- [ ] Event Bus routing messages between components
- [ ] Price → Signal → Risk → Execute flow working
- [ ] Orders saved to database with proper state
- [ ] Position state transitions logged

**Technical Tasks:**
- Create genesis/engine/trading_loop.py
- Wire event handlers for order lifecycle
- Implement state machine for positions
- Add audit logging for all events

### Story 0.4: Risk Engine Integration
**Priority**: HIGH
**Estimate**: 2 days

**Acceptance Criteria:**
- [ ] Position sizing enforced (5% rule)
- [ ] Account balance checked before orders
- [ ] Maximum position limits working ($100 for Sniper)
- [ ] Stop-loss orders placed automatically

**Technical Tasks:**
- Connect RiskEngine to trading loop
- Implement pre-trade validation
- Add position size calculator
- Create stop-loss order logic

### Story 0.5: Paper Trading Validation
**Priority**: HIGH
**Estimate**: 3 days

**Acceptance Criteria:**
- [ ] 10 successful round-trip trades completed
- [ ] P&L calculation accurate to 2 decimal places
- [ ] 24-hour continuous operation achieved
- [ ] UI showing live positions and P&L
- [ ] No manual intervention required

**Technical Tasks:**
- Create paper trading test harness
- Implement P&L tracking
- Add UI components for position display
- Run 24-hour stability test

## Dependencies
- Definition of Done document (Created ✓)
- Test infrastructure functional
- Database migrations working
- Binance testnet API keys configured

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| WebSocket instability | HIGH | Implement reconnection logic with exponential backoff |
| Test data limitations | MEDIUM | Use historical data replay if needed |
| Hidden technical debt | MEDIUM | Fix as found, don't perfect |
| Clock drift issues | LOW | Already validated in initialization |

## Definition of Done
Per [Definition of Done document](/docs/definition-of-done.md):
- All story acceptance criteria met
- Tests passing with >70% coverage
- Integration verified end-to-end
- 24-hour paper trading test passed
- Documentation updated

## Handoff Criteria
Epic 0 complete when:
1. Paper trading operational for 24 hours
2. All stories meet Definition of Done
3. Core metrics dashboard showing live data
4. Team demo completed successfully
5. Ready for Sniper Tier implementation

---
*Epic Owner*: Development Team
*Created*: 2025-08-28
*Target Completion*: 2 weeks from start
*Status*: 🚧 IN PROGRESS