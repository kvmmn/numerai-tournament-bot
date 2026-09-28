# Numerai Winning OS

A tested, governed Numerai Tournament control plane for data refresh, model
research, portfolio assignment, automatic submission, outcome monitoring,
postmortems, recovery, and separately approved NMR staking.

The production system lives in [`command_center/backend`](command_center/backend/README.md).
The root baseline scripts and `example-scripts` are retained as reference
material; they are not used by the scheduled production pipeline.

## Production workflow

```text
Numerai round opens
  → validate platform and dataset contracts
  → load the human-approved portfolio
  → refresh live data once
  → verify every approved model/evidence checksum
  → generate and validate one prediction file per model slot
  → apply robustness and zero-stake automation policy
  → approve, upload, and verify each eligible submission
  → record an idempotent round/model ledger entry
  → collect outcomes, health, alerts, and backups
```

Automatic submission is deliberately narrow. It is disabled by default,
requires `AUTO_SUBMIT_PORTFOLIO=true` in the installed runtime, accepts only the
already human-approved active portfolio, refuses stake-eligible assignments,
revalidates frozen checksums, and never authorizes promotion or stake changes.

## Repository map

- [`command_center/backend/app`](command_center/backend/app): API, agents, and domain control planes.
- [`command_center/backend/automation`](command_center/backend/automation): governed CLI, native schedules, reports, and deployment tooling.
- [`command_center/backend/tests`](command_center/backend/tests): unit and safety-contract tests.
- [`command_center/backend/docs`](command_center/backend/docs): architecture, operations, health, recovery, and competition policy.
- [`.github/workflows/agentic-winning-os.yml`](.github/workflows/agentic-winning-os.yml): Python 3.12 CI.

Runtime credentials, datasets, models, reports, logs, approval state, and
submission records are intentionally excluded from Git.

## Local verification

```bash
cd command_center/backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m compileall -q app automation tests
```

The complete production dependency set is in
`command_center/backend/requirements.txt`. Copy `.env.example` to `.env` only in
the installed runtime and never commit credentials.

## Runtime and operations

The reviewed code is stored in Git. Mutable production state runs from:

```text
~/Library/Application Support/Numerai/
├── data/v5.2/
├── backups/
└── runtime/
    ├── .venv/
    └── backend/
```

Start with the [operator runbook](command_center/backend/docs/RUNBOOK.md),
[operating map](command_center/backend/docs/OPERATING_SYSTEM.md), and
[runtime recovery guide](command_center/backend/docs/RUNTIME_RECOVERY.md).

This project improves process quality and operational safety; it cannot
guarantee tournament performance, rewards, or rank.
