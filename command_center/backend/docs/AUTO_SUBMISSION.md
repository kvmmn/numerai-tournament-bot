# Automatic Submission Contract

The scheduled submission path is designed to automate execution without
automating portfolio selection, model promotion, or staking.

## Enablement

`AUTO_SUBMIT_PORTFOLIO` defaults to `false`. Set it to `true` only in the
installed runtime after the active portfolio has been proposed, challenge
approved by an operator, and activated. The launchd job runs:

```bash
python automation/daily_numerai_run.py --mode portfolio-auto-submit --strict
```

Setting the switch does not create or change portfolio approval.

## Per-round contract

For every active assignment the runner:

1. resolves exactly one configured Numerai model UUID;
2. skips a round/model pair already present as verified in the durable ledger;
3. refuses assignments marked `stake_eligible=true`;
4. re-hashes the approved model and shadow-evidence files;
5. refreshes live data once for all pending slots;
6. checks the round submission window and live model mapping;
7. generates predictions and validates IDs, schema, finiteness, range, and diversity;
8. evaluates the frozen robustness policy;
9. creates a challenge-bound readiness packet and an auditable automation approval;
10. uploads one file to one slot, verifies Numerai's submission ID, and records the ledger entry.

One failing slot produces a partial-failure report while independent slots may
complete. A rerun is safe because verified round/model pairs are skipped.

## Boundaries

Automatic submission cannot:

- create, approve, or activate a portfolio;
- submit an artifact whose approved checksum changed;
- submit a stake-eligible assignment;
- promote a candidate or change the champion pointer;
- increase, decrease, or otherwise mutate NMR stake;
- bypass an invalid prediction, closed round, broken platform contract, or failed robustness policy.

## Healthy outcomes

- `PORTFOLIO_SUBMITTED_VERIFIED`: every assigned slot was uploaded and verified.
- `PORTFOLIO_ALREADY_SUBMITTED`: every assigned slot was already verified.
- `PORTFOLIO_AUTO_SUBMIT_PARTIAL_COVERAGE`: configured Numerai slots are not all assigned.
- `PORTFOLIO_AUTO_SUBMIT_PARTIAL_FAILURE`: at least one assigned slot failed; inspect `results` and do not blindly retry an uncertain upload.

The latest report is written under
`automation/reports/*_portfolio-auto-submit.{json,md}` and is consumed by the
health supervisor and alert dispatcher.

## Disable or recover

Set `AUTO_SUBMIT_PORTFOLIO=false` to stop future automatic submissions. This
does not modify the approved portfolio or durable ledger. For an uncertain API
result, reconcile Numerai's submission history before any retry. For code or
state damage, follow `RUNTIME_RECOVERY.md` and verify the deployment manifest.
