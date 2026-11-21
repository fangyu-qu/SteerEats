// Model-lens check: the browser's overlap@10, Kendall tau-b and SHAP group shares (web/src/lens.js) must match
// src/22_model_lens.py on every demo case. Run: node tests/check_lens.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { THETA0, finalRanks, similarity } from "../web/src/steer.js";
import { groupShares, kendallTauB, overlapAt } from "../web/src/lens.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const expected = JSON.parse(readFileSync(join(root, "tests", "lens_expected.json"), "utf8"));
let failures = 0, worstTau = 0, worstShare = 0;
for (const [id, e] of Object.entries(expected)) {
  const c = JSON.parse(readFileSync(join(root, "web", "public", "data", "cases", `${id}.json`), "utf8"));
  const sim = similarity(c);
  const ov = overlapAt(finalRanks(c, THETA0, "lr", sim), finalRanks(c, THETA0, "lgbm", sim));
  const tau = kendallTauB(c.lr.raw, c.lgbm.raw);
  worstTau = Math.max(worstTau, Math.abs(tau - e.tau));
  for (const m of ["lr", "lgbm"]) {
    const s = groupShares(c, m);
    for (const g of Object.keys(e.shares[m])) worstShare = Math.max(worstShare, Math.abs(s[g] - e.shares[m][g]));
  }
  if (ov !== e.overlap || Math.abs(tau - e.tau) > 1e-12) { failures++; console.log(`MISMATCH ${id}: overlap ${ov} vs ${e.overlap}, tau ${tau} vs ${e.tau}`); }
}
if (worstShare > 1e-12) failures++;
console.log(`${Object.keys(expected).length} cases: ${failures} failures; max tau error ${worstTau.toExponential(1)}; ` +
            `max group-share error ${worstShare.toExponential(1)}`);
process.exit(failures ? 1 : 0);
