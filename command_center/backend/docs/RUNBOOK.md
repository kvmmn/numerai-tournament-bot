# Operator Runbook

## Normal daily flow

1. **Readiness automation** refreshes data and evaluates the frozen candidate.
2. If it passes, review the round, model, metrics, checksum, and expiry.
3. Approve with the packet's run ID and challenge.
4. Submit that run ID.
5. Confirm status is `SUBMITTED_VERIFIED`.

```bash
python automation/daily_numerai_run.py \
  --mode agent-approve --run-id RUN_ID \
  --challenge CHALLENGE --actor OPERATOR --strict

python automation/daily_numerai_run.py \
  --mode agent-submit --run-id RUN_ID --strict
```

Never approve a packet showing a revocation, failed recent window, changed
checksum, expired deadline, or unexpected model slot.

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
python automation/daily_numerai_run.py --mode score-listen --strict
python automation/daily_numerai_run.py --mode research-evaluate --strict
python -m unittest discover -s tests -v
```

## When something fails

| Failure | Action |
|---|---|
| Dataset integrity | Keep old file, download to `.partial`, validate, then replace |
| Constant predictions | Reject model; retrain or fix features |
| Recent regime regression | Revoke packet; add research experiment |
| Upload returned but not verified | Do not retry automatically; reconcile ID |
| Missed deadline | Record postmortem; never backdate approval |
| Poor resolved score | Trigger postmortem and pause stake increases |

## Staking

Stake increases remain disabled until all conditions hold:

- explicit non-zero portfolio, model, and per-change caps;
- deployment round recorded;
- at least 20 resolved rounds from the current deployed model;
- positive recent correlation and MMC;
- positive-era rate above policy floor;
- separate proposal, challenge approval, and exact confirmation.

Submitting a model never authorizes staking.
