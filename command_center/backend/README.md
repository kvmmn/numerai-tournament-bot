# Numerai Command Center Backend

## Start here

- [Operating system map](docs/OPERATING_SYSTEM.md)
- [Modeling and optimization](docs/MODELING_AND_OPTIMIZATION.md)
- [Operator runbook](docs/RUNBOOK.md)
- [Implementation status](docs/IMPLEMENTATION_STATUS.md)

## What This Runs
- Syncs fresh Numerai datasets (`train`, `validation`, `live`)
- Trains challenger model
- Evaluates validation metrics
- Waits for approval (or auto-approves if enabled)
- Submits predictions to configured Numerai model(s)

## Safety-critical operating model

The daily path is `agent-prepare` → human review → `agent-approve` →
`agent-submit`. The control plane delegates work to named specialists and writes
durable readiness, approval, audit, and idempotency artifacts. It targets one
model slot per packet and verifies the returned submission ID.

`full-auto`, direct MCP submission, and file-flag approval are disabled. Staking
is implemented as a separate capped proposal/approval/execution workflow and is
disabled by default.

The original single-family candidate was rejected. A later `small + serenity`
feature-family challenger passed development and lockbox checks and is frozen in
an immutable bundle awaiting explicit model-promotion approval. It is not yet
the champion and has not been submitted.

## Setup
1. Create env file:
   - Copy `/Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/.env.example` to `/Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/.env`
2. Install dependencies:
   - `pip install -r /Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/requirements.txt`
3. Start API:
   - `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

## Required Env
- `NUMERAI_PUBLIC_ID`
- `NUMERAI_SECRET_KEY`

## Optional Env
- `NUMERAI_MODEL_NAMES=KVMMN,KVMMN_FN,KVMMN_TE`
  - If omitted, submissions are sent to all models returned by your Numerai account.
- `AUTO_APPROVE_SUBMISSION=false` (auto approval is ignored)
- `SUBMISSION_TARGET_MODEL=kvmmn_te`
- `MAX_TOTAL_STAKE_NMR=0` (set a deliberate cap before proposing increases)

## Preflight
- Run:
  - `GET /api/v1/numerai/preflight`
- This checks:
  - credentials
  - current round
  - model mapping that will be used for submission

## MCP-First Version (Alternative)
- This backend also supports an MCP-based flow against Numerai's MCP server.
- Configure in `.env`:
  - `NUMERAI_MCP_URL=https://api-tournament.numer.ai/mcp/sse`
  - `NUMERAI_MCP_AUTH=Token PUBLIC_KEY$PRIVATE_KEY`

### MCP Endpoints
- Discover tools:
  - `GET /api/v1/mcp/tools`
- Full MCP preflight + inferred tool mapping:
  - `GET /api/v1/mcp/preflight`
- Call one MCP tool directly:
  - `POST /api/v1/mcp/call`
  - body: `{"tool_name":"<name>","arguments":{...}}`
- Run MCP workflow stages:
  - `POST /api/v1/mcp/run`
  - body example:
    - `{"approve_submission":false,"args":{"train_model":{"model":"lgbm"}}}`
  - if `approve_submission=true`, submission stage is executed too.

## Cycle Endpoints
- Start cycle:
  - `POST /api/v1/os/start`
- Approve candidate:
  - `POST /api/v1/os/approve` with `{"decision":"APPROVED"}`
- Retry training:
  - `POST /api/v1/os/approve` with `{"decision":"RETRY_TRAINING"}`
- Reject candidate:
  - `POST /api/v1/os/approve` with `{"decision":"REJECTED"}`
- Read live state/logs:
  - `GET /api/v1/os/state`

## Automation Kit
- Full Codex automation runbook:
  - `/Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/automation/README.md`
- Ready prompts:
  - `/Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/automation/prompts/daily_monitor_mcp.md`
  - `/Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/automation/prompts/daily_submit_mcp.md`
- Runner:
  - `/Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend/automation/daily_numerai_run.py`
