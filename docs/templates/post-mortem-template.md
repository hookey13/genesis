# Post-Mortem Report

## Incident Overview

**Incident ID:** [INC-YYYYMMDD-XXXX]  
**Date:** [YYYY-MM-DD]  
**Time:** [Start Time] - [End Time] (UTC)  
**Duration:** [X hours Y minutes]  
**Severity:** [Critical/High/Medium/Low]  
**Status:** [Resolved/Mitigated/Monitoring]

**Authors:** [List of people who contributed to this post-mortem]  
**Reviewed By:** [Manager/Director who reviewed]  
**Last Updated:** [YYYY-MM-DD HH:MM UTC]

## Executive Summary

[2-3 sentence summary of what happened, the impact, and current status. This should be understandable by non-technical stakeholders.]

## Impact

### Quantitative Impact
- **Trading Disruption:** [X minutes of downtime / degraded performance]
- **Financial Impact:** [$X lost revenue / opportunity cost]
- **Orders Affected:** [X orders failed/delayed]
- **Positions Affected:** [X positions, $Y value]
- **Users Affected:** [X% of users / specific tier impacts]
- **SLA Breach:** [Yes/No - details if yes]

### Qualitative Impact
- **Reputation:** [None/Minor/Moderate/Severe]
- **Regulatory:** [Any compliance implications]
- **Customer Trust:** [Impact on user confidence]

## Timeline

All times in UTC. Include relevant context about market conditions or concurrent events.

| Time | Event | Actor | Notes |
|------|-------|-------|-------|
| HH:MM | Initial detection of issue | [System/Person] | [How was it detected] |
| HH:MM | Alert triggered | Monitoring System | [Alert details] |
| HH:MM | Incident acknowledged | [On-call person] | [Initial assessment] |
| HH:MM | Investigation started | [Team members] | [Initial hypothesis] |
| HH:MM | Root cause identified | [Person] | [What was discovered] |
| HH:MM | Mitigation started | [Team] | [Actions taken] |
| HH:MM | Mitigation completed | [Team] | [Verification steps] |
| HH:MM | Incident resolved | [Person] | [Resolution confirmed] |
| HH:MM | Post-incident monitoring | [Team] | [Stability confirmed] |

## Root Cause Analysis

### What Happened
[Detailed technical explanation of what went wrong. Include system behavior, error messages, and technical details.]

### Why It Happened (5 Whys Analysis)
1. **Why did the incident occur?**
   - [Primary cause]

2. **Why did [primary cause] happen?**
   - [Secondary cause]

3. **Why did [secondary cause] happen?**
   - [Tertiary cause]

4. **Why did [tertiary cause] happen?**
   - [Quaternary cause]

5. **Why did [quaternary cause] happen?**
   - [Root cause]

### Contributing Factors
- [ ] **Technical Debt:** [Describe any technical debt that contributed]
- [ ] **Process Gap:** [Missing or inadequate processes]
- [ ] **Documentation:** [Missing or outdated documentation]
- [ ] **Monitoring Gap:** [What wasn't being monitored]
- [ ] **Human Factor:** [Training, fatigue, communication issues]
- [ ] **External Factor:** [Third-party service, market conditions]

## Detection & Response Analysis

### Detection
- **Time to Detection:** [X minutes from onset]
- **Detection Method:** [Automated alert / User report / Manual discovery]
- **Could it have been detected sooner?** [Yes/No - explain]

### Response
- **Time to Acknowledgment:** [X minutes]
- **Time to Resolution:** [X hours Y minutes]
- **Response Efficiency:** [Good/Adequate/Poor - explain]
- **Communication:** [How well did we communicate during the incident]

## Resolution

### Immediate Actions Taken
1. [Action 1 - who, what, when]
2. [Action 2 - who, what, when]
3. [Action 3 - who, what, when]

### Temporary Mitigations
- [Any temporary fixes or workarounds implemented]
- [Risks associated with temporary measures]
- [Timeline for permanent fixes]

### Permanent Resolution
- [Long-term fix implemented or planned]
- [Verification of fix effectiveness]

## What Went Well

[It's important to recognize what worked during the incident]

- ✅ [Positive aspect 1 - e.g., "Alert fired within 30 seconds of issue"]
- ✅ [Positive aspect 2 - e.g., "Team responded quickly despite being after hours"]
- ✅ [Positive aspect 3 - e.g., "Runbook procedure was accurate and helpful"]

## What Could Be Improved

[Be blameless - focus on systems and processes, not individuals]

- ⚠️ [Improvement area 1]
- ⚠️ [Improvement area 2]
- ⚠️ [Improvement area 3]

## Lessons Learned

### Technical Lessons
1. [Key technical learning]
2. [Architecture insight gained]
3. [System behavior discovered]

### Process Lessons
1. [Process improvement identified]
2. [Communication enhancement needed]
3. [Documentation gap found]

### Organizational Lessons
1. [Team structure insight]
2. [Training need identified]
3. [Resource allocation learning]

## Action Items

### Immediate (Within 24 hours)
| Action | Owner | Due Date | Status | Ticket |
|--------|-------|----------|--------|--------|
| [Critical fix] | [Person] | [Date] | [Not Started/In Progress/Complete] | [JIRA-XXX] |
| [Update monitoring] | [Person] | [Date] | [Status] | [JIRA-XXX] |

### Short-term (Within 1 week)
| Action | Owner | Due Date | Status | Ticket |
|--------|-------|----------|--------|--------|
| [Process update] | [Person] | [Date] | [Status] | [JIRA-XXX] |
| [Documentation update] | [Person] | [Date] | [Status] | [JIRA-XXX] |

### Long-term (Within 1 month)
| Action | Owner | Due Date | Status | Ticket |
|--------|-------|----------|--------|--------|
| [Architecture improvement] | [Team] | [Date] | [Status] | [JIRA-XXX] |
| [Automation implementation] | [Person] | [Date] | [Status] | [JIRA-XXX] |

## Supporting Documentation

### Logs and Evidence
- [Link to relevant logs]
- [Link to metrics dashboards]
- [Link to alert history]
- [Screenshots if applicable]

### Related Incidents
- [Previous similar incidents with links]
- [Pattern identification]

### References
- [Runbook used]
- [Documentation consulted]
- [External resources]

## Follow-up

### Review Meeting
- **Date:** [When post-mortem review will be held]
- **Attendees:** [Who should attend]
- **Agenda:** [Key discussion points]

### Success Criteria
[How will we know our action items were successful?]
- [ ] [Metric or test that proves fix works]
- [ ] [Monitoring that would catch this earlier]
- [ ] [Process change verification]

### Preventing Recurrence
[Specific steps to prevent this exact issue from happening again]
1. [Prevention measure 1]
2. [Prevention measure 2]
3. [Prevention measure 3]

## Appendices

### A. Technical Details
[Detailed technical information, stack traces, configuration dumps, etc.]

### B. Communication Log
[Slack/email/phone communications during incident]

### C. Cost Analysis
[Detailed breakdown of financial impact]

### D. Customer Communications
[Any customer-facing communications sent]

---

## Post-Mortem Meta

**Culture Check:**
- Was this post-mortem blameless? [Yes/No]
- Did everyone feel safe contributing? [Yes/No]
- Were all perspectives heard? [Yes/No]

**Process Check:**
- Was the post-mortem completed within 48 hours? [Yes/No]
- Were all stakeholders included? [Yes/No]
- Were action items clearly assigned? [Yes/No]

**Template Version:** 1.0  
**Template Updated:** 2025-08-30

---

*Remember: The goal of a post-mortem is to learn and improve, not to assign blame. Focus on making our systems and processes more resilient.*