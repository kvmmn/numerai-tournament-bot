Run the Numerai daily monitor in the current workspace.

Execution:
- `cd /Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend`
- `python automation/daily_numerai_run.py --mode mcp-dry-run`

Then:
- Read the latest report files in `automation/reports` (both `.json` and `.md`).
- Produce a short operator summary with:
  - `ok` and `status`
  - selected MCP tools used
  - key outputs from sync/train/evaluate stages
  - any warnings or missing tool mappings
- If there is no actionable issue, explicitly say so.
