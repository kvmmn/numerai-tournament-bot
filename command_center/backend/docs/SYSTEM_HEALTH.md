# Native System Health

A loaded schedule is not proof that its work completed. The health supervisor
checks both sides:

```text
launchd service exists
        AND
latest durable evidence is recent
        AND
last report did not fail
        AND
last native exit is zero or has not run since a safe reload
             ↓
        job is healthy
```

## Daily sequence

```text
10:45–19:00  normal native jobs produce evidence
19:10        system-health checks all nine worker jobs
19:15        alert dispatcher reports any unhealthy worker
```

The supervisor checks:

| Worker | Durable evidence | Maximum age |
|---|---|---:|
| Platform monitor | `state/platform/latest.json` | 30 hours |
| Deadline guard | latest `portfolio-status` report | 30 hours |
| Portfolio readiness | latest `portfolio-prepare` report | 30 hours |
| Competition tracker | `state/competition/latest.json` | 30 hours |
| Outcome listener | latest `score-listen` report | 30 hours |
| Stake audit | latest `stake-status` report | 30 hours |
| Alert dispatcher | alert ledger | 30 hours |
| State backup | latest verified archive | 30 hours |
| Research review | latest research report | 192 hours |

The wide daily window tolerates normal scheduling jitter and maintenance while
still detecting a missed daily cycle before the next one ages out. The weekly
research window provides the same margin around its Sunday trigger.
When two files have the same filesystem timestamp, the timestamped filename is
the deterministic tie-breaker, so the newest named report is always selected.

## Result states

- `SYSTEM_HEALTHY`: every worker is loaded and has fresh successful evidence.
- `SYSTEM_HEALTH_ACTION_REQUIRED`: a service is absent, its last native exit
  failed, evidence is missing/stale/invalid, or the newest report has
  `ok=false`.

This monitor is read-only. It does not restart failed jobs automatically because
blind retry is unsafe for submission and stake workflows. It identifies the
failed boundary so the operator can use the relevant runbook.

Manual check:

```bash
python automation/daily_numerai_run.py --mode system-health --strict
```

Latest evidence:

```text
automation/state/system_health/latest.json
```
