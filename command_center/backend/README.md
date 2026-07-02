# Numerai Competition Control Plane

This repository runs a governed operating system for the Numerai Tournament:
data refresh, model research, portfolio assignment, one-file/one-model
submission, outcome monitoring, postmortems, and controlled NMR staking.

## Operating loop

```text
new round
  → refresh and validate data once
  → prepare each approved portfolio slot
  → human approves each exact prediction file
  → upload and verify the returned submission ID
  → collect resolved CORR/MMC outcomes
  → open a postmortem when live performance regresses
  → research bounded challengers
  → separately govern promotion and staking
```

The default is fail-closed. There is no file-flag approval, multi-model
broadcast submission, automatic promotion, or automatic stake mutation.

## Start here

- [Operating system map](docs/OPERATING_SYSTEM.md)
- [Competition control matrix](docs/COMPETITION_CONTROL_MATRIX.md)
- [Competition and season tracking](docs/COMPETITION_TRACKING.md)
- [Platform and data-version compatibility](docs/PLATFORM_COMPATIBILITY.md)
- [Native system health](docs/SYSTEM_HEALTH.md)
- [Modeling and optimization](docs/MODELING_AND_OPTIMIZATION.md)
- [Operator runbook](docs/RUNBOOK.md)
- [Runtime deployment and recovery](docs/RUNTIME_RECOVERY.md)
- [Implementation status](docs/IMPLEMENTATION_STATUS.md)

## Canonical runtime

Scheduled work runs from:

```text
~/Library/Application Support/Numerai/
├── data/v5.2/
└── runtime/
    ├── .venv/
    └── backend/
```

This avoids macOS background-access restrictions on `Desktop` and `Documents`.
The runtime is the source of truth for active portfolio state, readiness
packets, approvals, submission records, reports, postmortems, and stake state.
GitHub is the source of truth for reviewed code and documentation.

## Governed commands

Run these from the runtime backend with its virtual environment:

```bash
# Read-only status
python automation/daily_numerai_run.py --mode portfolio-status --strict
python automation/daily_numerai_run.py --mode platform-status --strict
python automation/daily_numerai_run.py --mode system-health --strict
python automation/daily_numerai_run.py --mode score-listen --strict
python automation/daily_numerai_run.py --mode stake-status --strict

# Submission: three separate steps
python automation/daily_numerai_run.py --mode portfolio-prepare --strict
python automation/daily_numerai_run.py \
  --mode agent-approve --run-id RUN_ID \
  --challenge CHALLENGE --actor OPERATOR --strict
python automation/daily_numerai_run.py \
  --mode agent-submit --run-id RUN_ID --strict

# Stake mutation: separate proposal, approval, and exact confirmation
python automation/daily_numerai_run.py \
  --mode stake-propose --target-model MODEL \
  --stake-action decrease --amount-nmr AMOUNT \
  --rationale "REASON" --strict
python automation/daily_numerai_run.py \
  --mode stake-approve --stake-proposal-path PROPOSAL \
  --challenge CHALLENGE --actor OPERATOR --strict
python automation/daily_numerai_run.py \
  --mode stake-execute --stake-proposal-path PROPOSAL \
  --confirmation "EXACT CONFIRMATION" --strict
```

Stake increases additionally require a stake-eligible production assignment,
verified deployment artifact, at least 20 resolved live rounds, positive CORR
and MMC evidence, available NMR, and non-zero caps. Submissions never authorize
stake changes.

## Native schedule

| Local time | Native job | Mutation? |
|---|---|---|
| 10:45 daily | Platform/API/data compatibility | No |
| 11:00 daily | Deadline and portfolio coverage | No |
| 15:00 daily | Portfolio readiness preparation | No |
| 15:20 daily | Coverage, streak, rank, and season status | No |
| 18:00 daily | Outcome listener and postmortem trigger | No |
| 18:05 daily | Stake/portfolio policy reconciliation | No |
| 19:00 daily | Verified state backup with seven-copy retention | Local files only |
| 19:10 daily | Native job and evidence-freshness supervisor | No |
| 16:00 Sunday | Production robustness review | No |

Deduplicated native alerts run at 11:05, 15:25, 18:12, and 19:15 after the
corresponding read-only checks.

Codex watchdogs inspect these reports shortly afterward. Submissions,
promotions, portfolio activation, and stake changes are never scheduled.

## Safety boundaries

- One readiness packet targets exactly one Numerai model UUID.
- Prediction IDs, ranges, diversity, checksum, and current round are verified.
- A verified local ledger prevents repeat round/model uploads.
- Shadow models are permanently stake-ineligible.
- Stake approval binds the complete proposal file.
- Stake execution rechecks live balances, model mapping, current caps, and
  portfolio eligibility.
- An execution intent is written before a stake API call; an unresolved intent
  blocks retries and requires manual reconciliation.
- `full-auto`, `mcp-submit`, and `numerapi-submit` are disabled.

Numerai scoring and staking involve uncertainty and possible NMR burns. The
system improves process quality; it cannot guarantee profitability or rank.
See Numerai's official [submissions](https://docs.numer.ai/numerai-tournament/submissions),
[scoring](https://docs.numer.ai/numerai-tournament/scoring), and
[staking](https://docs.numer.ai/numerai-tournament/staking) documentation.
