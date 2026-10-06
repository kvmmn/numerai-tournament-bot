import { readFileSync, existsSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineTool } from "@cursor/bdk/tools";
import { z } from "zod";

const MAX_BYTES = 200_000;

type SuiteMember = {
  name: string;
  modelType: string;
  featureSet: string;
  engineered: boolean;
};

type StatusRow = {
  capability: string;
  status: string;
};

type ProjectSnapshot = {
  found: boolean;
  repo: string;
  modelSuite: SuiteMember[];
  optimizerEntrypoints: string[];
  optimizerImportedByDailyRunner: boolean;
  modes: string[];
  disabledModes: string[];
  graphApprovalDisabled: boolean;
  legacyGraphPresent: boolean;
  activeStatusRows: StatusRow[];
  experimentEvidence: string;
};

function findRepoRoot(): string | null {
  const starts = [
    path.dirname(fileURLToPath(import.meta.url)),
    process.cwd(),
  ];
  for (const start of starts) {
    let dir = start;
    for (let i = 0; i < 8; i += 1) {
      const marker = path.join(
        dir,
        "command_center",
        "backend",
        "app",
        "core",
        "models.py",
      );
      if (existsSync(marker)) return dir;
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  }
  return null;
}

function readText(root: string, relativePath: string): string {
  const full = path.resolve(root, relativePath);
  const relative = path.relative(root, full);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("path escapes the tournament checkout");
  }
  if (relativePath.includes(".env") || relativePath.endsWith(".pkl")) {
    throw new Error("refusing credential or model artifact");
  }
  const info = statSync(full);
  if (!info.isFile() || info.size > MAX_BYTES) {
    throw new Error(`unreadable or oversized file: ${relativePath}`);
  }
  return readFileSync(full, "utf8");
}

function repoSlug(root: string): string {
  const configPath = path.join(root, ".git", "config");
  if (!existsSync(configPath)) return "unknown";
  const config = readFileSync(configPath, "utf8");
  const match = config.match(/github\.com[:/]([^/\s]+)\/([^/\s."]+)/);
  if (!match) return "unknown";
  return `${match[1]}/${match[2]}`;
}

function modelSuite(source: string): SuiteMember[] {
  return source
    .split("ModelConfig(")
    .slice(1)
    .map((block) => {
      const name = block.match(/name="([^"]+)"/)?.[1];
      const modelType = block.match(/model_type="([^"]+)"/)?.[1];
      if (!name || !modelType) return null;
      return {
        name,
        modelType,
        featureSet: block.match(/feature_set="([^"]+)"/)?.[1] ?? "small",
        engineered: /use_engineered_features\s*=\s*True/.test(block),
      };
    })
    .filter((member): member is SuiteMember => member !== null);
}

function optimizerEntrypoints(source: string): string[] {
  return [...source.matchAll(/^def (run_[a-z0-9_]+|select_best_candidate)\(/gm)].map(
    (match) => match[1] ?? "",
  ).filter((name) => name.length > 0);
}

function modesFromRunner(source: string): { modes: string[]; disabledModes: string[] } {
  const choiceBlock = source.match(/choices=\[([\s\S]*?)\]/);
  const modes = choiceBlock
    ? [...choiceBlock[1].matchAll(/"([^"]+)"/g)].map((match) => match[1] ?? "")
    : [];
  const disabledModes = modes.filter((mode) => {
    const needle = `if mode == "${mode}"`;
    const index = source.indexOf(needle);
    if (index < 0) return false;
    const next = source.indexOf("if mode ==", index + needle.length);
    const slice = source.slice(index, next < 0 ? index + 800 : next);
    return slice.includes("unsafe_legacy_mode_disabled");
  });
  return { modes, disabledModes };
}

function activeStatusRows(markdown: string): StatusRow[] {
  return markdown
    .split("\n")
    .filter((line) => line.startsWith("|") && !line.includes("---"))
    .map((line) => line.split("|").slice(1, -1).map((cell) => cell.trim()))
    .filter((cells) => cells.length >= 2 && cells[0] !== "Capability")
    .filter((cells) => cells[1] === "Active" || cells[1].startsWith("Policy"))
    .map((cells) => ({ capability: cells[0] ?? "", status: cells[1] ?? "" }));
}

function experimentEvidence(markdown: string): string {
  const start = markdown.indexOf("## Experiment evidence");
  if (start < 0) return "";
  const end = markdown.indexOf("\n## ", start + 10);
  const section = markdown.slice(start, end < 0 ? undefined : end).trim();
  return section.slice(0, 2500);
}

function emptySnapshot(): ProjectSnapshot {
  return {
    found: false,
    repo: "unknown",
    modelSuite: [],
    optimizerEntrypoints: [],
    optimizerImportedByDailyRunner: false,
    modes: [],
    disabledModes: [],
    graphApprovalDisabled: false,
    legacyGraphPresent: false,
    activeStatusRows: [],
    experimentEvidence: "",
  };
}

export default defineTool({
  description:
    "Read the Numerai tournament checkout and return the model suite, optimizer entrypoints, disabled runner modes, and documented experiment evidence. Does not submit, stake, or train.",
  effect: "read",
  inputSchema: z.object({}),
  async execute(): Promise<ProjectSnapshot> {
    const root = findRepoRoot();
    if (!root) return emptySnapshot();

    const models = readText(root, "command_center/backend/app/core/models.py");
    const optimizer = readText(root, "command_center/backend/app/core/optimizer.py");
    const runner = readText(
      root,
      "command_center/backend/automation/daily_numerai_run.py",
    );
    const status = readText(
      root,
      "command_center/backend/docs/IMPLEMENTATION_STATUS.md",
    );
    const modeling = readText(
      root,
      "command_center/backend/docs/MODELING_AND_OPTIMIZATION.md",
    );
    const routes = readText(root, "command_center/backend/app/api/routes.py");
    const parsedModes = modesFromRunner(runner);

    return {
      found: true,
      repo: repoSlug(root),
      modelSuite: modelSuite(models),
      optimizerEntrypoints: optimizerEntrypoints(optimizer),
      optimizerImportedByDailyRunner:
        runner.includes("optimizer") || runner.includes("select_best_candidate"),
      modes: parsedModes.modes,
      disabledModes: parsedModes.disabledModes,
      graphApprovalDisabled: routes.includes("Legacy graph approval is disabled"),
      legacyGraphPresent: existsSync(
        path.join(root, "command_center", "backend", "app", "graph", "workflow.py"),
      ),
      activeStatusRows: activeStatusRows(status),
      experimentEvidence: experimentEvidence(modeling),
    };
  },
});
