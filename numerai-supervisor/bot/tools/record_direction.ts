import { defineTool } from "@cursor/bdk/tools";
import { prompt } from "@cursor/bdk";
import { z } from "zod";

const BLOCKED =
  /\b(full-auto|mcp-submit|numerapi-submit|agent-submit|stake-execute|stake-approve)\b/i;

type DirectionResult = {
  recorded: boolean;
  experiment: string;
  reason?: string;
};

export default defineTool({
  description: prompt`
    Record a research direction the operator explicitly confirmed.
    Call only after the user asks to record, lock, or remember the next experiment.
    Refuses submission, stake, and disabled-mode directions.
  `,
  effect: "write",
  inputSchema: z.object({
    experiment: z.string().min(8).max(280),
    rationale: z.string().min(8).max(600),
    confirm: z.boolean(),
  }),
  dryRunResult: ({ experiment }): DirectionResult => ({
    recorded: false,
    experiment,
    reason: "dry-run",
  }),
  async execute({ experiment, rationale, confirm }, ctx): Promise<DirectionResult> {
    if (!confirm) {
      return { recorded: false, experiment, reason: "confirm must be true" };
    }
    if (BLOCKED.test(experiment) || BLOCKED.test(rationale)) {
      return {
        recorded: false,
        experiment,
        reason: "direction touches a disabled submission or stake action",
      };
    }
    await ctx.host.kv.put("research-direction", {
      experiment,
      rationale,
      recordedAt: new Date().toISOString(),
      sessionId: ctx.session.id,
    });
    return { recorded: true, experiment };
  },
});
