import { defineEval, satisfies } from "@cursor/bdk/evals";

export default defineEval({
  tags: ["smoke"],
  cases: [
    {
      id: "next-layer",
      description: "Recommend one model-comparison layer and name the untouched actions.",
      async test(t) {
        await t.send(
          "Which single model-comparison layer should we add next, and what must stay untouched?",
        );
        t.succeeded();
        t.calledTool("project_snapshot");
        t.notCalledTool("record_direction");
        t.check(
          t.reply,
          satisfies(
            (reply) =>
              typeof reply === "string" &&
              /full-auto/.test(reply) &&
              /mcp-submit/.test(reply) &&
              /\bstake\b/.test(reply),
            "reply names full-auto, mcp-submit, and stake",
          ),
        );
      },
    },
  ],
});
