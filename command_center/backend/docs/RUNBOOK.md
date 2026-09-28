# Operator Runbook

## Normal daily flow

1. Platform and deadline monitors confirm the round is safe to enter.
2. The 15:00 job loads the human-approved active portfolio.
3. It verifies model and evidence checksums, refreshes live data once, and runs
   each slot through prediction and robustness validation.
4. Eligible zero-stake assignments are approved, uploaded, and verified.
5. Confirm the report is `PORTFOLIO_SUBMITTED_VERIFIED` or the idempotent
   `PORTFOLIO_ALREADY_SUBMITTED`.

```bash
python automation/daily_numerai_run.py --mode portfolio-auto-submit --strict
```

Keep `AUTO_SUBMIT_PORTFOLIO=false` outside the installed scheduler. Disable it
immediately when portfolio identity, model checksums, platform contracts, live
IDs, or verification behavior are in doubt. See `AUTO_SUBMISSION.md`.

## Promoting a researched challenger

Promotion changes the local champion pointer; it does not submit to Numerai.

```bash
python automation/daily_numerai_run.py \
  --mode model-approve --manifest-path MANIFEST \
  --challenge CHALLENGE --actor OPERATOR --strict

python automation/daily_numerai_run.py \
  --mode model-promote --manifest-path MANIFEST --strict
```

The current frozen candidate is bundle `1449f3f4d590427740653ed6`
(`feature_small_serenity_neutral65`). Its promotion challenge is stored in the
local manifest and expires after 24 hours.

## Safe read-only checks

```bash
python automation/daily_numerai_run.py --mode numerapi-preflight --strict
python automation/daily_numerai_run.py --mode platform-status --strict
python automation/daily_numerai_run.py --mode system-health --strict
python automation/daily_numerai_run.py --mode score-listen --strict
python automation/daily_numerai_run.py --mode stake-status --strict
python automation/daily_numerai_run.py --mode competition-status --strict
python automation/daily_numerai_run.py --mode alert-dispatch --strict
python automation/daily_numerai_run.py --mode research-evaluate --strict
python automation/runtime_ops.py --strict audit
python automation/runtime_ops.py --strict backup-state --keep 7
python -m unittest discover -s tests -v
```

## When something fails

| Failure | Action |
|---|---|
| Platform contract broken | Keep mutations gated; inspect the failed contract in `automation/state/platform/latest.json` |
| New remote data version | Test migration on a branch; do not switch production automatically |
| Native job unhealthy | Inspect `automation/state/system_health/latest.json`, then rerun only the named read-only/preparation job after understanding its last exit |
| Dataset integrity | Keep old file, download to `.partial`, validate, then replace |
| Constant predictions | Reject model; retrain or fix features |
| Recent regime regression | Revoke packet; add research experiment |
| Upload returned but not verified | Do not retry automatically; reconcile ID |
| Auto-submit partial failure | Inspect the per-slot result; rely on the ledger to skip verified slots and rerun only after the cause is understood |
| Missed deadline | Record postmortem; never backdate approval |
| Poor resolved score | Durable postmortem opens; pause stake increases |
| Stake execution intent without result | Do not retry; reconcile with Numerai first |

## Staking

Stake increases remain disabled until all conditions hold:

- explicit non-zero portfolio, model, and per-change caps;
- deployment round recorded;
- at least 20 resolved rounds from the current deployed model;
- positive recent correlation and MMC;
- positive-era rate above policy floor;
- separate proposal, challenge approval, and exact confirmation.

Submitting a model never authorizes staking.

The live stake audit is read-only:

```bash
python automation/daily_numerai_run.py --mode stake-status --strict
```

If a decrease is deliberately authorized, first configure a non-zero
`MAX_STAKE_CHANGE_NMR`, then create a proposal. Zero total/model caps may remain
in place because they continue to block increases.

```bash
python automation/daily_numerai_run.py \
  --mode stake-propose --target-model MODEL \
  --stake-action decrease --amount-nmr AMOUNT \
  --rationale "REASON" --strict

python automation/daily_numerai_run.py \
  --mode stake-approve --stake-proposal-path PROPOSAL \
  --challenge CHALLENGE --actor OPERATOR --strict

python automation/daily_numerai_run.py \
  --mode stake-execute --stake-proposal-path PROPOSAL \
  --confirmation "EXACT CONFIRMATION FROM PROPOSAL" --strict
```

Never retry an execution when `execution_intent.json` exists without a matching
`execution.json`; the API request may already have reached Numerai.
