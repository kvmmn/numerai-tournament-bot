---
cron: "30 15 * * *"
---

Take exactly one Numerai research step, then stop.

Call `live_performance` first. Rank the slots by `eraMeanMmc`. The bar
is the highest era MMC, not the July promotion note. Pass those
`era_mean_mmc` and `recent_mean_mmc` values to the read-only
`daily-step` runner when you need the named experiment.

Do not submit, stake, promote, or enable `full-auto`, `mcp-submit`, or
`numerapi-submit`. If training data or credentials are absent, say which
blocker remains and do not invent a score.
