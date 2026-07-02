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
- `portfolio-prepare`: loads the active distinct per-slot assignments, skips
  already submitted slots, and prepares independent packets for the rest.

Example:
```bash
python automation/daily_numerai_run.py --mode agent-prepare \
  --target-model kvmmn_te --strict
python automation/daily_numerai_run.py --mode agent-approve \
  --run-id <run-id> --challenge <challenge> --actor <operator> --strict
python automation/daily_numerai_run.py --mode agent-submit \
  --run-id <run-id> --strict
python automation/daily_numerai_run.py --mode portfolio-prepare --strict
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

## Automation strategy
1. Let macOS `launchd` run `portfolio-prepare` daily.
2. Let the Codex watchdog verify every assigned and unassigned slot.
3. Record approval only after a human checks the round, model, artifact hash,
   and prediction validation.
4. Run `agent-submit` with the approved run id.
5. Monitor `automation/state/audit.jsonl` and `submission_ledger.json`.

## Suggested cadence
- Native readiness preparation: every day at 15:00 local time.
- Codex readiness watchdog: every day at 15:30 local time.
- Deadline guard: every day at 11:00 local time before the 14:00 close.
- Submission is never scheduled; it requires a current, explicit approval.

## Failure handling
- If `ok=false`, report stays in inbox for triage.
- If `ok=true`, review key metrics and submission summary.
- `--strict` should be enabled for submit automation so failures are explicit.

## Submission and staking gates
- A file flag is not an approval.
- Readiness approvals are bound to round, model UUID, and submission SHA-256.
- Stake increases are disabled while policy caps are zero.
- Stake execution requires a separate approval challenge and exact confirmation.
