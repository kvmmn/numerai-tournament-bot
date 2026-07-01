# Numerai Automation Kit (Codex App)

This folder contains the daily automation runner and ready-to-use prompt templates
for Codex Automations.

## Goals
- Run Numerai workflow daily without manual terminal work.
- Keep submissions governed with explicit checks.
- Emit monitorable artifacts for audit and review.

## Runner
- Script: `/Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/automation/daily_numerai_run.py`
- Reports output: `/Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/automation/reports`

## Governed agent modes
- `agent-prepare`: delegates round discovery, data sync, prediction generation,
  risk validation, and readiness packaging. It cannot submit.
- `agent-approve`: records a human operator's challenge-bound approval.
- `agent-submit`: verifies the approval, current round, target mapping, artifact
  hash, prediction schema, and idempotency ledger before one upload.

Example:
```bash
python automation/daily_numerai_run.py --mode agent-prepare \
  --target-model kvmmn_te --strict
python automation/daily_numerai_run.py --mode agent-approve \
  --run-id <run-id> --challenge <challenge> --actor <operator> --strict
python automation/daily_numerai_run.py --mode agent-submit \
  --run-id <run-id> --strict
```

## Legacy modes
- `mcp-preflight`: list/validate MCP tools and mapping.
- `mcp-dry-run`: run MCP workflow until evaluation, no submit.
- `mcp-submit`: disabled; direct MCP submission is not governed.
- `mcp-auto`: always dry-run. `ENABLE_SUBMIT` no longer grants approval.
- `numerapi-preflight`: validate NumerAPI credentials and model mapping.
- `numerapi-submit`: disabled; it bypassed durable human approval.
- `full-auto`: disabled because it bypassed human governance.

## Local smoke commands
```bash
cd /Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend
python automation/daily_numerai_run.py --mode mcp-preflight
python automation/daily_numerai_run.py --mode mcp-dry-run
python automation/daily_numerai_run.py --mode mcp-auto --strict
python automation/daily_numerai_run.py --mode mcp-submit --strict
```

## Codex automation strategy
1. Run `agent-prepare` daily.
2. Review the readiness packet and metrics in the inbox.
3. Record approval only after a human checks the round, model, artifact hash,
   and prediction validation.
4. Run `agent-submit` with the approved run id.
5. Monitor `automation/state/audit.jsonl` and `submission_ledger.json`.

## Suggested cadence
- Daily monitoring: every day at 09:00 local time.
- Daily submit (optional): every day at 11:30 local time after monitoring.

## Failure handling
- If `ok=false`, report stays in inbox for triage.
- If `ok=true`, review key metrics and submission summary.
- `--strict` should be enabled for submit automation so failures are explicit.

## Submission and staking gates
- A file flag is not an approval.
- Readiness approvals are bound to round, model UUID, and submission SHA-256.
- Stake increases are disabled while policy caps are zero.
- Stake execution requires a separate approval challenge and exact confirmation.
