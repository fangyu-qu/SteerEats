// Trade-off check: the browser's per-user grid outcomes and predicted visits (web/src/tradeoff.js) must match
// src/25_export_tradeoff.py on every demo case, and its Pareto frontiers over the population grid must match
// outputs/rq1/frontiers.csv (src/21_rq1_frontier.py). Run: node tests/check_tradeoff.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { frontier, userSpace } from "../web/src/tradeoff.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (...p) => readFileSync(join(root, ...p), "utf8");
const pop = JSON.parse(read("web", "public", "data", "tradeoff.json"));
const expected = JSON.parse(read("tests", "tradeoff_expected.json"));
const NAME = { lr: "Logistic regression", lgbm: "LightGBM", lmr: "LambdaRank" };
let failures = 0, worst = 0, n = 0;
for (const [id, e] of Object.entries(expected)) {
  const c = JSON.parse(read("web", "public", "data", "cases", `${id}.json`));
  for (const m of Object.keys(NAME)) {
    userSpace(c, m, pop.platt).forEach((p, k) => {
      const [km, popu, cui, visits] = e[m][k];
      const err = Math.max(Math.abs(p.km - km), Math.abs(p.popularity - popu), Math.abs(p.visits - visits) / visits);
      worst = Math.max(worst, err);
      n++;
      if (p.cuisines !== cui || err > 1e-12) { failures++; if (failures < 6) console.log(`MISMATCH ${id} ${m} setting ${k}`, p, e[m][k]); }
    });
  }
}

// frontiers.csv: quantity, model_name, value, ndcg, w_h, alpha, beta, lam
const [head, ...lines] = read("outputs", "rq1", "frontiers.csv").trim().split("\n");
const cols = head.split(",");
const rows = lines.map((l) => Object.fromEntries(l.split(",").map((v, j) => [cols[j], j === 0 || j === 1 ? v : +v])));
let fronts = 0;
for (const [qty, low] of [["distance_km", true], ["popularity", true], ["cuisines", false], ["neighbourhoods", false], ["outside_home_borough", false]]) {
  for (const m of Object.keys(NAME)) {
    const want = rows.filter((r) => r.quantity === qty && r.model_name === NAME[m]).map((r) => `${r.w_h},${r.alpha},${r.beta},${r.lam}`);
    const got = frontier(pop.settings[m], qty, low, "ndcg").map((r) => `${r.w_h},${r.alpha},${r.beta},${r.lam}`);
    fronts++;
    if (want.join("|") !== got.join("|")) { failures++; console.log(`FRONTIER MISMATCH ${qty} ${m}\n  want ${want.join(" ")}\n  got  ${got.join(" ")}`); }
  }
}
console.log(`${n} case settings, max error ${worst.toExponential(1)}; ${fronts} population frontiers; ${failures} failures`);
process.exit(failures ? 1 : 0);
