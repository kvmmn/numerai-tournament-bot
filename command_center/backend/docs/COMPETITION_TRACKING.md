# Competition Tracking

Operational health and competitive progress are different:

```text
job ran
  ≠ prediction submitted
  ≠ submission was on time
  ≠ round was stake-qualified
  ≠ score resolved
  ≠ reputation improved
  ≠ leaderboard rank improved
```

The tracker keeps these states separate.

## Daily status

For every account model it records:

- active portfolio tier;
- current-round verified local submission;
- consecutive locally verified submission streak;
- Numerai season participation rounds;
- latest available CORR, MMC, FNCv3, and TC reputation/rank;
- age of the rank observation.

At account level it records:

- current-round coverage across every model slot;
- season participating and qualified rounds;
- total at-risk NMR per round;
- progress toward the qualifying-round target;
- account rank when found within the configured scan range.

## Season qualification

The default policy mirrors the current official Grandmasters rule:

```text
20 distinct on-time rounds
AND
at least 1 NMR total at risk across the account in each round
```

The thresholds are explicit configuration:

```text
SEASON_QUALIFYING_ROUNDS=20
SEASON_MIN_AT_RISK_NMR=1.0
ACCOUNT_RANK_SCAN_LIMIT=1000
```

Review Numerai's official
[Grandmasters and Seasons](https://docs.numer.ai/numerai-tournament/scoring/grandmasters-and-seasons)
page before changing them.

## Evidence sources

| Question | Authoritative evidence |
|---|---|
| Did this control plane submit the current round? | Verified local submission ledger |
| Did Numerai record model participation? | `round_model_performances_v2` |
| Was the round season-qualified? | Sum of Numerai `atRisk` across account models |
| What is the latest model rank/reputation? | Numerai daily model performance |
| Is the account in the scanned leaderboard range? | Numerai account leaderboard |

The tracker never treats validation performance as live reputation. Numerai
states that leaderboard reputation is based on final live scores; see the
official [scoring documentation](https://docs.numer.ai/numerai-tournament/scoring).

## Current measured state

At the first live run on 2026-07-02:

- round `1302`: `1 / 3` model slots verified;
- missing: `kvmmn`, `kvmmn_fn`;
- production `kvmmn_te` local submission streak: `1`;
- 2026 participating rounds: `1302`;
- 2026 qualified rounds: `0 / 20`;
- round-1302 total at risk: `0 NMR`;
- account not found in the top 1,000 scanned account ranks.

These are measurements, not predictions about future rank.

After the two separately approved zero-stake shadow uploads on 2026-07-02:

- round `1302`: `3 / 3` model slots verified;
- missing slots: none;
- local submission streak: `1` for each slot;
- deadline state: `DEADLINE_GUARD_COMPLETE`;
- competition state: `COMPETITION_CURRENT_ROUND_COMPLETE`.

Submission coverage does not change the season qualification measurement:
round 1302 still has `0 NMR` at risk, and stake decisions remain separate.

## Alerts

The native alert dispatcher runs after deadline, readiness/competition, and
outcome/stake checks. It sends deduplicated macOS notifications for:

- broken platform contracts or a new data version requiring review;
- missing, stale, or failed native job evidence;
- incomplete current-round coverage;
- readiness waiting for review;
- deadline action;
- stake-policy violations;
- required live-performance postmortems.

Delivery state is durable. A successful alert is not repeated for the same
source report; failed notifications remain pending and are retried.
