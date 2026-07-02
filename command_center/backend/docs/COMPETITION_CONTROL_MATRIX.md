# Competition Control Matrix

This page is the completion checklist for the full Numerai operating system.
“Implemented” means a tested code path exists. “Live verified” means the
deployed runtime has exercised that path against the current account.

| Competition concern | Control | Current evidence | State |
|---|---|---|---|
| Fresh data | Atomic download plus full parquet row-group validation | Daily preparation and tests | Live verified |
| Reproducible models | Frozen pickle, feature union, data and artifact checksums | Candidate/shadow bundles | Implemented |
| Overfitting control | Walk-forward folds, embargo, recent windows, one lockbox | Research packets and tests | Implemented |
| Portfolio diversity | Unique artifact checksums and correlation ceiling | Three-slot active portfolio | Live verified |
| Correct live predictions | Exact live IDs, finite range, diversity and checksum | Round-1302 readiness packets | Live verified |
| Correct target slot | Case-insensitive explicit model mapping; one packet per UUID | Three account models resolved | Live verified |
| Platform compatibility | API/account/round/data/mapping/stake-read contracts | Live `PLATFORM_COMPATIBLE`: 7/7 contracts passed | Live verified |
| Submission deadline | Round `closeTime`, native 11:00 inspection | Native report exit 0 | Live verified |
| Submission execution | Human challenge, expiring approval, idempotent ledger | `kvmmn_te` round-1302 submission | Live verified |
| Multi-slot coverage | Portfolio-wide prepare/status with submitted-slot skip | Production plus two shadows | Live verified |
| Outcome collection | Daily resolved CORR/MMC cursor | Native listener exit 0 | Live verified |
| Performance incident | Deduplicated durable postmortem artifact | Tested; waits for a new adverse outcome | Implemented |
| Research review | Weekly active-production robustness evaluation | Native `RESEARCH_PROMOTE` report | Live verified |
| Promotion | Immutable bundle plus separate approval and activation | Production bundle promoted | Live verified |
| Stake reconciliation | Daily live stake versus portfolio-policy audit | `kvmmn` shadow-stake violation detected | Live verified |
| Stake increase | Live evidence, verified deployed artifact, caps, approval | Disabled: caps and available NMR are zero | Correctly gated |
| Stake decrease | Live balance, per-change cap, approval, exact confirmation | Implemented; no decrease authorized | Correctly gated |
| Stake retry safety | Proposal hash, stale-balance check, execution intent | Unit tests | Implemented |
| Unattended execution | Privacy-safe Library runtime and `launchd` jobs | Nine jobs loaded | Live verified |
| Independent watchdogs | Codex jobs inspect runtime reports after native jobs | Five active watchdogs | Live verified |
| GitHub quality gate | Tests and compile check on draft PR | GitHub Actions | Live verified |
| Runtime deployment | Test, stage, checksum, code backup, atomic replacement, drift audit | Commit `b77c88c` deployed with 85/85 checksum parity | Live verified |
| State recovery | Credential-free manifest, archive checksum, member verification, guarded restore | 95-file staging restore; ledger/model hashes matched | Live verified |
| Local backup cadence | Daily 19:00 verified archive; seven-copy retention | Native backup exit 0; 19 MB archive | Live verified |
| Round participation | Per-slot verified coverage and local submission streak | Round-1302 live snapshot | Live verified |
| Season qualification | Participating rounds, total at-risk, 20-round progress | `0 / 20` qualified in live snapshot | Live verified |
| Reputation and rank | Model reputation age plus bounded account-rank scan | Account outside scanned top 1,000 | Live verified |
| Urgent local alerts | Deduplicated deadline/readiness/stake/postmortem notifications | 4 delivered; repeat dispatched 0 | Live verified |

## Current action queue

```text
1. Approve or reject the two round-1302 shadow submissions.
2. Decide whether the pre-existing 0.136245 NMR on shadow slot `kvmmn`
   should be reduced; no change occurs without a non-zero decrease cap and a
   separate exact approval.
3. Collect resolved live outcomes for the active production artifact.
4. Keep stake increases disabled until at least 20 artifact-matched resolved
   rounds satisfy the CORR/MMC policy.
5. Use adverse outcomes to open bounded research experiments, then require a
   fresh frozen bundle and promotion approval.
```

## Remaining engineering work

These items are useful but do not justify unsafe shortcuts in the live path:

- copy verified local backups to encrypted/off-host storage;
- add an optional off-device alert channel for machine/network outages.

The matrix should be updated whenever live evidence changes. Passing unit tests
alone does not move an item to “Live verified.”
