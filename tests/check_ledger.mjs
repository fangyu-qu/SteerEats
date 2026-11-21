// Ledger arithmetic: on every demo case, for both models and several control settings, the ledger's
// typical-candidate start must equal the SHAP base value on the normalized scale (so bars are exact SHAP
// values, not a plug), and model score + penalties must equal the steered score used for ranking.
// Run: node tests/check_ledger.mjs
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { stages, similarity } from "../web/src/steer.js";
import { ledgerData } from "../web/src/views/ledger.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dir = join(root, "web", "public", "data", "cases");
const thetas = [{ w_h: 1, alpha: 0, beta: 0, lam: 0 }, { w_h: 0.4, alpha: 0.3, beta: 0.2, lam: 0.2 }, { w_h: 0, alpha: 1, beta: 0.5, lam: 0.5 }];
let worstBase = 0, worstScore = 0, checks = 0;
for (const f of readdirSync(dir).filter((x) => x.startsWith("demo_"))) {
  const c = JSON.parse(readFileSync(join(dir, f), "utf8"));
  const sim = similarity(c);
  for (const model of ["lr", "lgbm"]) for (const th of thetas) {
    const st = stages(c, th, model, sim);
    const ranks = { default: st.S0, model: st.S0, distance: st.S1, popularity: st.S2, boost: st.S2b, diversity: st.S3 };
    const m = c[model], n = m.raw.length;
    let zmin = Infinity, zmax = -Infinity;
    for (let k = 0; k < n; k++) { const z = m.noncollab[k] + th.w_h * m.collab[k]; zmin = Math.min(zmin, z); zmax = Math.max(zmax, z); }
    for (let i = 0; i < n; i++) {
      const L = ledgerData(c, model, th, ranks, i);
      worstBase = Math.max(worstBase, Math.abs(L.base - (m.base - zmin) / (zmax - zmin)));
      worstScore = Math.max(worstScore, Math.abs(L.s2 - st.score.S2[i]));
      checks++;
    }
  }
}
console.log(`${checks} ledgers: max |start - SHAP base| ${worstBase.toExponential(1)} (normalized units); max |ledger - steered score| ${worstScore.toExponential(1)}`);
process.exit(worstBase < 1e-4 && worstScore < 1e-9 ? 0 : 1);
