import { defineTool } from "@cursor/bdk/tools";
import { z } from "zod";

const ENDPOINT = "https://api-tournament.numer.ai/graphql";
const DEFAULT_MODELS = ["kvmmn", "kvmmn_te", "kvmmn_fn"] as const;

type RoundPoint = {
  roundNumber: number;
  open: string | null;
  corr20: number | null;
  mmc: number | null;
  fncV3: number | null;
  stake: number | null;
  payout: number | null;
};

type StakeDrop = {
  fromRound: number;
  toRound: number;
  fromStake: number;
  toStake: number;
};

type ModelLive = {
  name: string;
  found: boolean;
  stakeNow: number | null;
  corrReputation: number | null;
  mmcReputation: number | null;
  fncReputation: number | null;
  corrRank: number | null;
  mmcRank: number | null;
  oneYearReturn: number | null;
  threeMonthReturn: number | null;
  scoredRounds: number;
  firstScoredRound: number | null;
  lastScoredRound: number | null;
  meanCorr20: number | null;
  meanMmc: number | null;
  negativeCorrRounds: number;
  negativeMmcRounds: number;
  priorMeanCorr20: number | null;
  priorMeanMmc: number | null;
  eraMeanCorr20: number | null;
  eraMeanMmc: number | null;
  eraScoredRounds: number;
  recentMeanCorr20: number | null;
  recentMeanMmc: number | null;
  recentNegativeCorrRounds: number;
  recent: RoundPoint[];
  missingRounds: number[];
  largestStakeDrop: StakeDrop | null;
};

type LivePerformance = {
  ok: boolean;
  source: string;
  latestRound: number | null;
  models: ModelLive[];
};

type RawRound = {
  roundNumber?: number;
  roundOpenTime?: string | null;
  corr20V2?: number | null;
  mmc?: number | null;
  fncV3?: number | null;
  selectedStakeValue?: string | null;
  payout?: string | null;
};

type RawProfile = {
  stakeValue?: string | null;
  latestReps?: { corr?: number | null; mmc?: number | null; fncV3?: number | null } | null;
  latestRanks?: { corr?: number | null; mmc?: number | null; fncV3?: number | null } | null;
  latestReturns?: { oneYear?: number | null; threeMonths?: number | null } | null;
  roundModelPerformances?: RawRound[] | null;
};

const QUERY = `
query ($name: String!) {
  v3UserProfile(modelName: $name) {
    stakeValue
    latestReps { corr mmc fncV3 }
    latestRanks { corr mmc fncV3 }
    latestReturns { oneYear threeMonths }
    roundModelPerformances {
      roundNumber
      roundOpenTime
      corr20V2
      mmc
      fncV3
      selectedStakeValue
      payout
    }
  }
}
`;

function round6(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return round6(values.reduce((sum, value) => sum + value, 0) / values.length);
}

function num(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined) return null;
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function emptyModel(name: string): ModelLive {
  return {
    name,
    found: false,
    stakeNow: null,
    corrReputation: null,
    mmcReputation: null,
    fncReputation: null,
    corrRank: null,
    mmcRank: null,
    oneYearReturn: null,
    threeMonthReturn: null,
    scoredRounds: 0,
    firstScoredRound: null,
    lastScoredRound: null,
    meanCorr20: null,
    meanMmc: null,
    negativeCorrRounds: 0,
    negativeMmcRounds: 0,
    priorMeanCorr20: null,
    priorMeanMmc: null,
    eraMeanCorr20: null,
    eraMeanMmc: null,
    eraScoredRounds: 0,
    recentMeanCorr20: null,
    recentMeanMmc: null,
    recentNegativeCorrRounds: 0,
    recent: [],
    missingRounds: [],
    largestStakeDrop: null,
  };
}

function lastLongPauseEnd(missing: number[]): number {
  let pauseEnd = -1;
  let runStart = 0;
  for (let i = 0; i < missing.length; i += 1) {
    const previous = missing[i - 1];
    if (i === 0 || (previous !== undefined && missing[i] !== previous + 1)) runStart = i;
    const runLength = i - runStart + 1;
    if (runLength >= 8) pauseEnd = missing[i] ?? pauseEnd;
  }
  return pauseEnd;
}

function stakeDrop(rounds: RawRound[]): StakeDrop | null {
  const staked = rounds
    .map((round) => ({
      roundNumber: round.roundNumber ?? 0,
      stake: num(round.selectedStakeValue),
    }))
    .filter((round) => round.stake !== null && round.stake > 0)
    .sort((a, b) => a.roundNumber - b.roundNumber);
  let largest: StakeDrop | null = null;
  for (let i = 1; i < staked.length; i += 1) {
    const previous = staked[i - 1];
    const current = staked[i];
    if (!previous || !current || previous.stake === null || current.stake === null) continue;
    const drop = previous.stake - current.stake;
    if (drop <= 0.01) continue;
    if (!largest || drop > largest.fromStake - largest.toStake) {
      largest = {
        fromRound: previous.roundNumber,
        toRound: current.roundNumber,
        fromStake: round6(previous.stake),
        toStake: round6(current.stake),
      };
    }
  }
  return largest;
}

function summarize(name: string, profile: RawProfile | null): ModelLive {
  if (!profile) return emptyModel(name);
  const rounds = [...(profile.roundModelPerformances ?? [])].sort(
    (a, b) => (a.roundNumber ?? 0) - (b.roundNumber ?? 0),
  );
  const scored = rounds.filter((round) => round.corr20V2 !== null && round.corr20V2 !== undefined);
  const first = scored[0]?.roundNumber ?? null;
  const last = scored[scored.length - 1]?.roundNumber ?? null;
  const scoredNumbers = new Set(scored.map((round) => round.roundNumber));
  const allMissing =
    first === null || last === null
      ? []
      : rounds
          .map((round) => round.roundNumber ?? 0)
          .filter((roundNumber) => roundNumber >= first && roundNumber < last && !scoredNumbers.has(roundNumber));
  const pauseEnd = lastLongPauseEnd(allMissing);
  const missingRounds = (pauseEnd < 0 ? allMissing : allMissing.filter((roundNumber) => roundNumber > pauseEnd)).slice(0, 40);
  const era = pauseEnd < 0 ? scored : scored.filter((round) => (round.roundNumber ?? 0) > pauseEnd);
  const recentScored = scored.slice(-12);
  const prior = scored.slice(Math.max(0, scored.length - 42), Math.max(0, scored.length - 12));
  const corr = scored.map((round) => round.corr20V2).filter((value): value is number => value !== null && value !== undefined);
  const mmc = scored.map((round) => round.mmc).filter((value): value is number => value !== null && value !== undefined);
  const priorCorr = prior.map((round) => round.corr20V2).filter((value): value is number => value != null);
  const priorMmc = prior.map((round) => round.mmc).filter((value): value is number => value != null);
  const eraCorr = era.map((round) => round.corr20V2).filter((value): value is number => value != null);
  const eraMmc = era.map((round) => round.mmc).filter((value): value is number => value != null);
  const recentCorr = recentScored.map((round) => round.corr20V2).filter((value): value is number => value != null);
  const recentMmc = recentScored.map((round) => round.mmc).filter((value): value is number => value != null);

  return {
    name,
    found: true,
    stakeNow: num(profile.stakeValue),
    corrReputation: profile.latestReps?.corr ?? null,
    mmcReputation: profile.latestReps?.mmc ?? null,
    fncReputation: profile.latestReps?.fncV3 ?? null,
    corrRank: profile.latestRanks?.corr ?? null,
    mmcRank: profile.latestRanks?.mmc ?? null,
    oneYearReturn: profile.latestReturns?.oneYear ?? null,
    threeMonthReturn: profile.latestReturns?.threeMonths ?? null,
    scoredRounds: scored.length,
    firstScoredRound: first,
    lastScoredRound: last,
    meanCorr20: mean(corr),
    meanMmc: mean(mmc),
    negativeCorrRounds: corr.filter((value) => value < 0).length,
    negativeMmcRounds: mmc.filter((value) => value < 0).length,
    priorMeanCorr20: mean(priorCorr),
    priorMeanMmc: mean(priorMmc),
    eraMeanCorr20: mean(eraCorr),
    eraMeanMmc: mean(eraMmc),
    eraScoredRounds: era.length,
    recentMeanCorr20: mean(recentCorr),
    recentMeanMmc: mean(recentMmc),
    recentNegativeCorrRounds: recentCorr.filter((value) => value < 0).length,
    recent: recentScored.map((round) => ({
      roundNumber: round.roundNumber ?? 0,
      open: round.roundOpenTime ?? null,
      corr20: round.corr20V2 ?? null,
      mmc: round.mmc ?? null,
      fncV3: round.fncV3 ?? null,
      stake: num(round.selectedStakeValue),
      payout: num(round.payout),
    })),
    missingRounds: missingRounds.slice(0, 40),
    largestStakeDrop: stakeDrop(rounds),
  };
}

async function loadProfile(name: string): Promise<RawProfile | null> {
  const response = await fetch(ENDPOINT, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "user-agent": "numerai-supervisor",
    },
    body: JSON.stringify({ query: QUERY, variables: { name } }),
    signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok) {
    throw new Error(`Numerai profile request failed for ${name}`);
  }
  const body = (await response.json()) as {
    data?: { v3UserProfile?: RawProfile | null };
    errors?: unknown[];
  };
  if (body.errors?.length) {
    throw new Error(`Numerai profile query failed for ${name}`);
  }
  return body.data?.v3UserProfile ?? null;
}

export default defineTool({
  description:
    "Read public Numerai round scores, reputation, ranks, and stake for the account models. Use this to explain a drop. Does not submit or stake.",
  effect: "read",
  inputSchema: z.object({
    models: z.array(z.string().regex(/^[A-Za-z0-9_-]{1,40}$/)).min(1).max(5).optional(),
  }),
  async execute({ models }): Promise<LivePerformance> {
    const names = models?.length ? models : [...DEFAULT_MODELS];
    const profiles = await Promise.all(names.map((name) => loadProfile(name)));
    const summarized = names.map((name, index) => summarize(name, profiles[index] ?? null));
    const latestRound = summarized.reduce<number | null>((max, model) => {
      if (model.lastScoredRound === null) return max;
      return max === null ? model.lastScoredRound : Math.max(max, model.lastScoredRound);
    }, null);
    return {
      ok: summarized.some((model) => model.found),
      source: ENDPOINT,
      latestRound,
      models: summarized,
    };
  },
});
