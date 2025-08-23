# Introduction

This document outlines the overall project architecture for Project GENESIS, including backend systems, shared services, and non-UI specific concerns. Its primary goal is to serve as the guiding architectural blueprint for AI-driven development, ensuring consistency and adherence to chosen patterns and technologies.

**Relationship to Frontend Architecture:**
If the project includes a significant user interface, a separate Frontend Architecture Document will detail the frontend-specific design and MUST be used in conjunction with this document. Core technology stack choices documented herein (see "Tech Stack") are definitive for the entire project, including any frontend components.

## Starter Template or Existing Project

No existing trading frameworks (Freqtrade, Jesse, etc.) will be used - they would constrain the tier-locking mechanism and psychological safeguards that are core to GENESIS's survival strategy. Building from scratch with:
- ccxt for Binance API abstraction
- Rich/Textual for the terminal UI  
- asyncio as the core async framework
- Custom tier-locking state machine, tilt detection, and risk management

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-08-23 | 1.0 | Initial architecture document creation | Winston (Architect) |
