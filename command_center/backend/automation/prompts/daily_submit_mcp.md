Run the Numerai daily MCP automation with strict failure mode and submit gate.

Execution:
- `cd /Users/kaveh/Desktop/base/_LAB/numerai/command_center/backend`
- `python automation/daily_numerai_run.py --mode mcp-auto --strict`

Then:
- Read the latest report files in `automation/reports` (both `.json` and `.md`).
- Provide a concise run report with:
  - `ok` and `status`
  - target models and per-model submission result
  - round or stage context if available
  - clear error block if any model failed
- If submit was skipped, explicitly mention whether `automation/ENABLE_SUBMIT` flag was missing.
