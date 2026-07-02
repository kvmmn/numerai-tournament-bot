# Platform Compatibility Monitor

Numerai can change datasets, response fields, or client behavior. A green model
test does not prove that tomorrow's live workflow can still reach the correct
round, data, models, or stake state.

The platform monitor runs read-only before the daily deadline workflow:

```text
10:45 platform contract check
  ├─ API methods and authenticated account schema
  ├─ account models ↔ active portfolio assignments
  ├─ current round and real submission-window fields
  ├─ configured remote dataset version and required files
  ├─ local feature/parquet schema
  └─ read-only stake query for every account model
       ↓
  compatible → normal daily workflow
  newer data version → migration-review alert
  broken contract → fail closed + urgent alert
```

## Result states

| Status | Meaning | Operator action |
|---|---|---|
| `PLATFORM_COMPATIBLE` | Every required contract passed | No action |
| `PLATFORM_MIGRATION_REVIEW` | Current configuration still works, but a newer data version exists | Test the new version away from production |
| `PLATFORM_CONTRACT_BROKEN` | A required API, mapping, remote file, or local schema failed | Keep mutations gated and repair the failing contract |

The report contains only operational fields. It does not persist API tokens,
email, wallet address, or the raw account response.

## Evidence

The latest durable snapshot is:

```text
automation/state/platform/latest.json
```

Run it manually from the installed runtime:

```bash
python automation/daily_numerai_run.py --mode platform-status --strict
```

The native `com.numerai.platform` job runs at 10:45 local time. The 11:05 alert
dispatch includes platform failures and migration-review warnings.

## Migration rule

A newly listed data version is a review signal, not permission to switch live
models. Migration requires a separate branch, complete data integrity checks,
retraining or compatibility validation, temporal evaluation, and a governed
promotion decision. The active data version remains unchanged until that
evidence exists.
