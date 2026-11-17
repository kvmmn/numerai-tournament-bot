# Canonical Metrics Ledger

The metrics ledger is the single fast, auditable source for project-scale,
dataset, research, automation, and live Numerai statistics. It is read-only:
it never uploads predictions, changes a model, or changes stake.

## Outputs

Every run writes atomically under `automation/state/metrics/`:

| Artifact | Purpose |
|---|---|
| `latest.json` | Most recent attempt, including partial failures |
| `latest_successful.json` | Most recent snapshot in which every required source succeeded |
| `latest.md` | Human-readable metric table |
| `snapshots/<timestamp>.json` | Immutable timestamped history |
| `index.json` | Newest 400 snapshot references for fast history queries |

A failed source is recorded as an error in `latest.json`. Values from an older
run are not silently copied into the failed source, and
`latest_successful.json` is not overwritten. This prevents stale values from
looking current.

## Provenance contract

Each canonical metric contains:

- `value` and `unit`;
- `source` and UTC `measured_at` timestamp;
- `scope`;
- a plain-language `definition`;
- a `window` when the value is time- or round-bounded.

The ledger deliberately distinguishes:

- raw submission upload records (retries/replacements included);
- model-round performance records;
- resolved model-round scores;
- distinct tournament rounds;
- full-coverage portfolio rounds where every account model has a resolved
  CORR score;
- verified local submission-ledger records.

These values answer different questions and must not be substituted for one
another in reporting or a resume.

## Sources

Required sources are authenticated Numerai API data, local Parquet metadata,
durable automation artifacts, and a repository quality scan. The latest
research evaluation is optional. Dataset rows are read from Parquet footers,
not estimated from file size. Test functions are AST-counted and explicitly
labeled as definitions—not passing tests.

## Run and retrieve

```bash
python automation/daily_numerai_run.py --mode metrics-snapshot --strict
```

The native job runs daily at 19:05 local time, after backup and before the
19:10 cross-job health check. Fast API reads do not contact Numerai:

```text
GET /api/v1/metrics/latest
GET /api/v1/metrics/latest?successful=true
GET /api/v1/metrics/history?limit=30
```

`METRICS_START_ROUND` defaults to `1302`, `METRICS_PROJECT_START_UTC` to
`2026-07-02T00:00:00Z`, and `METRICS_RECENT_ROUNDS` to `20`. Numerai's
`submission_ids` response has no round number, so raw uploads use the explicit
project-start timestamp while performance history uses the start round.
Override these only when the reporting definition itself is deliberately
changed, and keep that change visible in the snapshot configuration.
