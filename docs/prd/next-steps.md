# Next Steps

## UX Expert Prompt

"Review the Project GENESIS PRD focusing on the terminal-based 'Digital Zen Garden' interface design. Create detailed wireframes for the three-panel layout (P&L, Positions, Commands) with special attention to the psychological aspects: color psychology for anti-tilt design, progressive disclosure by tier, and error message presentation. Consider how visual hierarchy and information density change between Sniper ($500), Hunter ($2k), and Strategist ($10k) tiers. Document the command syntax and autocomplete behavior. Pay particular attention to how tilt warning indicators manifest visually without triggering panic."

## Architect Prompt

"Design the technical architecture for Project GENESIS using the PRD as foundation. Focus on the evolutionary architecture that transforms from monolith (Python/SQLite) to service-oriented (Python/Rust/PostgreSQL) as capital grows. Address the critical <100ms execution requirement, tier-locked feature system implementation, and real-time tilt detection algorithms. Special attention needed for: state management across tiers, WebSocket connection resilience, order slicing algorithms, and the correlation calculation engine. Provide deployment architecture for DigitalOcean Singapore with failover planning. Document how the system enforces tier restrictions at the code level to prevent override attempts."

---

*End of Product Requirements Document v1.0*