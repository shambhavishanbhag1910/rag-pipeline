# Cloud Service Reliability Standard

Customer-facing Tier 1 services must have a monthly availability objective of 99.95 percent and a documented error budget. Tier 2 business services target 99.9 percent. Planned maintenance is excluded only when communicated at least five business days in advance and completed within the approved window.

Tier 1 services require multi-availability-zone deployment, automated backups, tested restoration, infrastructure as code, centralized logs, metrics, traces, and on-call coverage at all times. The recovery time objective is 60 minutes and the recovery point objective is 15 minutes. Disaster recovery exercises must be completed at least twice per year.

A deployment must be automatically halted or rolled back when the canary error rate is more than two percentage points above baseline for ten minutes, when p95 latency doubles for ten minutes, or when a critical health check fails. Emergency changes require incident commander approval and retrospective review within two business days.
