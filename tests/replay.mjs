// Replay test: the browser implementation (web/src/steer.js) must reproduce the Python reference
// (src/steering.py) exactly on every exported demo case, and both decompositions must have zero residual.
// Run: node tests/replay.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { finalRanks, similarity, stageLog, shapleyControls, solveTargets } from "../web/src/steer.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const expected = JSON.parse(readFileSync(join(root, "tests", "replay_expected.json"), "utf8"));
let checks = 0, failures = 0, worstStage = 0, worstShapley = 0;
let solverChecks = 0, solverFailures = 0;
for (const [id, block] of Object.entries(expected)) {
  const c = JSON.parse(readFileSync(join(root, "web", "public", "data", "cases", `${id}.json`), "utf8"));
  const sim = similarity(c);
  for (const s of block.solver) {
    const res = solveTargets(c, s.model, s.targets);
    const ok = ["w_h", "alpha", "beta", "lam"].every((k) => res.theta[k] === s.theta[k]) &&
      res.top.every((v, i) => v === s.top[i]) && Object.keys(s.met).every((k) => res.met[k] === s.met[k]);
    solverChecks++;
    if (!ok) { solverFailures++; console.log(`SOLVER MISMATCH ${id} ${s.model} ${JSON.stringify(s.targets)}`); }
  }
  for (const run of block.thetas) {
    const ranks = finalRanks(c, run.theta, run.model, sim);
    const top20 = Array.from(ranks, (r, i) => [r, i]).sort((a, b) => a[0] - b[0]).slice(0, 20).map(([, i]) => i);
    const same = top20.every((v, i) => v === run.top20[i]);
    const log = stageLog(c, run.theta, run.model, sim);
    const sh = shapleyControls(c, run.theta, run.model, sim);
    const rs = Math.max(...log.residual.map(Math.abs));
    const rsh = Math.max(...sh.residual.map(Math.abs));
    worstStage = Math.max(worstStage, rs); worstShapley = Math.max(worstShapley, rsh);
    checks++;
    if (!same || rs !== 0 || rsh > 1e-9) {
      failures++;
      console.log(`MISMATCH ${id} ${run.model} ${JSON.stringify(run.theta)} top20 ${same ? "ok" : "differs"} residuals ${rs} ${rsh}`);
    }
  }
}
console.log(`${checks} replays over ${Object.keys(expected).length} cases: ${failures} failures; ` +
            `max fixed-order residual ${worstStage}; max Shapley residual ${worstShapley.toExponential(1)}`);
console.log(`${solverChecks} outcome-target solves: ${solverFailures} mismatches`);
process.exit(failures || solverFailures ? 1 : 0);
