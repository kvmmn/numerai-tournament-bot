---
description: Use when choosing the next Numerai model comparison, research gap, or improvement to the Winning OS. Not for submitting, staking, or promoting.
---

# Steer research

1. Call `project_snapshot` with `{}` before recommending a code or
   research change. If the question is about a score drop, a broken
   round, or which live slot is stronger, also call `live_performance`
   with `{}`.
2. Treat `disabledModes` and `graphApprovalDisabled` as hard stops.
   Rank live slots by `meanMmc`, `priorMeanMmc`, and `recentMeanMmc`
   from `live_performance`, not by the old promotion-candidate paragraph.
   A drop is `recentMean*` turning negative while `priorMean*` was positive.
3. Recommend exactly one next experiment. Rank slots by `eraMeanMmc`
   from `live_performance` and pass `era_mean_mmc` plus
   `recent_mean_mmc` through `choose_daily_step` (runner mode
   `daily-step`). That function names the bar slot, the weaker slot,
   and one existing optimizer entrypoint. It does not train or submit.
   Missing rounds stay an operations note, separate from that experiment.
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
