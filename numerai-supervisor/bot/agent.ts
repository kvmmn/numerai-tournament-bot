import { defineAgent } from "@cursor/bdk";

export default defineAgent({
  name: "Numerai supervisor",
  description:
    "Read-only supervisor for the Numerai Winning OS. Steers the next model comparison and refuses submission or stake changes.",
  model: {
    id: "grok-4.5",
    params: [
      { id: "effort", value: "high" },
      { id: "fast", value: "true" },
    ],
  },
  tools: ["read", "grep", "glob", "ls"],
});
