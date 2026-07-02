# Runtime Deployment and Recovery

The GitHub checkout contains reviewed source. The Library runtime contains the
live operating state. Deployment must update code without replacing
credentials, readiness packets, approvals, submission records, model bundles,
reports, or stake records.

```text
reviewed Git source
  → run all source tests
  → stage managed code
  → checksum staged files
  → back up current runtime code
  → atomically replace managed files
  → verify source/runtime checksum parity

live mutable state
  → exclude .env
  → archive state + model registry + reports
  → write per-file manifest
  → checksum archive
  → reopen and verify every member
  → retain seven newest backups
```

## Read-only drift audit

```bash
python automation/runtime_ops.py --strict audit
```

`RUNTIME_IN_SYNC` means every managed source file has the same SHA-256 in the
runtime. Mutable directories are intentionally excluded.

## Deploy reviewed source

Run from the Git checkout after CI passes:

```bash
python automation/runtime_ops.py --strict deploy
```

The deploy command runs the full source test suite using the runtime Python,
creates a code backup, stages and verifies files, performs atomic per-file
replacement, prunes only previously managed stale files, and runs a final drift
audit.

It never copies:

- `.env`;
- `automation/state`;
- `automation/model_registry`;
- `automation/reports`;
- datasets, optimization outputs, or logs.

## State backup

```bash
python automation/runtime_ops.py --strict backup-state --keep 7
```

Backups are stored under:

```text
~/Library/Application Support/Numerai/backups/state/
```

Each archive has a checksum sidecar and an internal manifest containing every
member's size, mode, and SHA-256. Credential-like filenames, symlinks, hard
links, absolute paths, and path traversal are rejected.

The native `com.numerai.backup` job runs this at 19:00 local time. Local
backups protect against accidental code/state damage but not device loss.
Copying verified archives to encrypted off-device storage remains a separate
operator decision.

## Verify or stage a restore

```bash
python automation/runtime_ops.py --strict verify-backup --backup BACKUP

python automation/runtime_ops.py --strict restore-state \
  --backup BACKUP --destination /private/tmp/numerai-restore-check
```

Restore to a staging directory first. A live restore requires the exact
archive-bound confirmation printed by the verifier:

```text
RESTORE <first 12 characters of archive SHA-256>
```

Before a live restore, the manager creates another verified state backup.
Credentials are never restored or overwritten.

## Code rollback

Every deployment returns a code backup path. Rollback requires:

```text
ROLLBACK <first 12 characters of code archive SHA-256>
```

The manager backs up the current code again before applying the selected
rollback archive. After rollback, rerun the drift audit and all native
read-only checks before any submission approval.
