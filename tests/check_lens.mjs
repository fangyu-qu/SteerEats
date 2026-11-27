// Model-lens check: the browser's overlap@10, Kendall tau-b and SHAP group shares (web/src/lens.js) must match
// src/22_model_lens.py on every demo case. Run: node tests/check_lens.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { THETA0, finalRanks, similarity } from "../web/src/steer.js";
import { groupShares, kendallTauB, overlapAt } from "../web/src/lens.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const expected = JSON.parse(readFileSync(join(root, "tests", "lens_expected.json"), "utf8"));
let failures = 0, worstTau = 0, worstShare = 0, pairs = 0;
for (const [id, e] of Object.entries(expected)) {
  const c = JSON.parse(readFileSync(join(root, "web", "public", "data", "cases", `${id}.json`), "utf8"));
  const sim = similarity(c);
  for (const [pair, want] of Object.entries(e.pairs)) {
    const [a, b] = pair.split("|");
    const ov = overlapAt(finalRanks(c, THETA0, a, sim), finalRanks(c, THETA0, b, sim));
    const tau = kendallTauB(c[a].raw, c[b].raw);
    worstTau = Math.max(worstTau, Math.abs(tau - want.tau));
    pairs++;
    if (ov !== want.overlap || Math.abs(tau - want.tau) > 1e-12) { failures++; console.log(`MISMATCH ${id} ${pair}: overlap ${ov} vs ${want.overlap}, tau ${tau} vs ${want.tau}`); }
  }
  for (const m of Object.keys(e.shares)) {
    const s = groupShares(c, m);
    for (const g of Object.keys(e.shares[m])) worstShare = Math.max(worstShare, Math.abs(s[g] - e.shares[m][g]));
  }
}
if (worstShare > 1e-12) failures++;
console.log(`${Object.keys(expected).length} cases, ${pairs} model pairs: ${failures} failures; max tau error ${worstTau.toExponential(1)}; ` +
            `max group-share error ${worstShare.toExponential(1)}`);
process.exit(failures ? 1 : 0);
