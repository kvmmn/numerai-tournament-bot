---
description: Use when choosing the next Numerai model comparison, research gap, or improvement to the Winning OS. Not for submitting, staking, or promoting.
---

# Steer research

1. Call `project_snapshot` with `{}` before recommending a change.
2. Treat `disabledModes` and `graphApprovalDisabled` as hard stops.
3. Recommend exactly one next experiment: a bake-off that runs one
   declared idea already represented in `modelSuite` or
   `optimizerEntrypoints` through walk-forward evaluation and the
   existing promotion gate. Prefer wiring `optimizerEntrypoints` into
   research when `optimizerImportedByDailyRunner` is false.
4. Do not invent a new uploader, re-enable a disabled mode, or move
   stake. Shadow slots stay zero-stake.
5. Reply with these sections, in the user's language:
   - Live now: champion evidence and active rows from the snapshot
   - Next experiment: one idea, the entrypoint to use, and why it is
     the strongest gap
   - Untouched: `full-auto`, `mcp-submit`, `stake`
   - Operator check: `research-evaluate`, then a human `model-approve`
     only if the snapshot shows those modes exist
6. Do not call `record_direction` in this procedure.
