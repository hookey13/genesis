# Infrastructure and Deployment

## Infrastructure as Code

- **Tool:** Bash scripts (MVP), Terraform 1.7.0 ($2k+)
- **Location:** `/scripts/deploy.sh` (MVP), `/terraform/` ($2k+)
- **Approach:** Start simple with shell scripts, evolve to declarative IaC as complexity grows

## Deployment Strategy

- **Strategy:** Blue-green deployment with manual cutover
- **CI/CD Platform:** GitHub Actions (test only at MVP, full deploy at $2k+)
- **Pipeline Configuration:** `.github/workflows/test.yml`, `.github/workflows/deploy.yml`

## Environments

- **Development:** Local Docker container - Safe experimentation without risking capital
- **Paper Trading:** VPS with paper trading flag - Test strategies with fake money - Activated for 48 hours before any tier transition
- **Production:** Primary VPS (Singapore) - Live trading with real capital - DigitalOcean SGP1 region, closest to Binance
- **Disaster Recovery:** Backup VPS (San Francisco) - Activated at $5k+ - Cold standby, promoted if primary fails

## Environment Promotion Flow

```text
Local Development (Docker)
    ↓ [Code commit + tests pass]
Paper Trading VPS (48 hours minimum)
    ↓ [Profitable for 3 consecutive days]
Production VPS (Singapore)
    ↕ [Failover if needed]
DR VPS (San Francisco) - [$5k+ only]
```

## Rollback Strategy

- **Primary Method:** Git revert + redeploy previous version
- **Trigger Conditions:** 3 failed trades in 10 minutes, System crash loop, Tilt detection spike
- **Recovery Time Objective:** <5 minutes for critical issues

## Detailed Infrastructure Setup

See architecture document for complete deployment scripts, backup strategy, monitoring, and failover procedures.
