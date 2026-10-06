# Numerai supervisor

Read-only BDK agent for the Numerai Winning OS in this repository. It reads
the checkout, names one next model-comparison experiment, and refuses
submission and stake changes.

Defaults used because the interview tool was unavailable:

- Model: `grok-4.5` with `effort=high` and `fast=true`
- Channels: playground and HTTP
- MCP: none
- Tools: `project_snapshot` (read), `live_performance` (read, public Numerai scores), `record_direction` (write, confirm required)

```bash
cd numerai-supervisor
npx @cursor/bdk validate --dir .
npx @cursor/bdk call project_snapshot --dir . --input '{}'
npx @cursor/bdk dev
```

A model turn needs a Cursor credential (`CURSOR_API_KEY`, then
`CURSOR_API_KEY_FILE`, then `CURSOR_SERVICE_ACCOUNT_KEY`, then
`npx @cursor/bdk login`):

```bash
npx @cursor/bdk run --dir . --message "Which single model-comparison layer should we add next, and what must stay untouched?"
npx @cursor/bdk eval --dir .
```
