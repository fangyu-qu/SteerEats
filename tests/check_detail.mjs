// Restaurant-detail check (V8). From the exported monthly star counts alone, the browser arithmetic
// (web/src/detail.js) must (1) reproduce what the rankers knew at the snapshot, as stored in every demo case:
// review counts exactly and the two smoothed ratings; and (2) match the rolling-window values computed by
// src/29_export_restaurant_detail.py. Run: node tests/check_detail.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expand, rolling, snapshot, toDate } from "../web/src/detail.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (...p) => JSON.parse(readFileSync(join(root, ...p), "utf8"));
const meta = read("web", "public", "data", "detail", "index.json");
const expected = read("tests", "detail_expected.json");
const close = (a, b, tol) => (a == null || b == null ? a == null && b == null : Math.abs(a - b) <= tol);

let failures = 0, candidates = 0, values = 0, worstRating = 0, worstRolling = 0;
if (JSON.stringify(meta) !== JSON.stringify(expected.index)) { failures++; console.log("MISMATCH: detail/index.json differs from the settings the expectations were computed with"); }
for (const id of meta.cases) {
  const c = read("web", "public", "data", "cases", `${id}.json`), d = read("web", "public", "data", "detail", `${id}.json`);
  const cand = c.candidates;
  if (d.restaurants.length !== cand.gmap_id.length || c.cutoff !== meta.cutoff) { failures++; console.log(`MISMATCH ${id}: candidates or snapshot differ`); continue; }
  const tables = d.restaurants.map((r) => expand(r, meta.months));
  d.restaurants.forEach((r, i) => {
    const s = snapshot(r, tables[i], meta);
    const n = s.all ? s.all.n : 0, nRecent = s.recent ? s.recent.n : 0;
    const e1 = Math.abs(s.smoothed - cand.rating[i]), e2 = Math.abs(s.smoothedRecent - cand.recent_rating[i]);
    worstRating = Math.max(worstRating, e1, e2);
    if (n !== cand.reviews[i] || nRecent !== cand.recent_reviews[i] || e1 > 1e-6 || e2 > 1e-6) {
      failures++; console.log(`MISMATCH ${id} #${i}: ${n}/${nRecent} reviews vs ${cand.reviews[i]}/${cand.recent_reviews[i]}, smoothed ${s.smoothed} vs ${cand.rating[i]}`);
    }
    candidates++;
  });
  const cache = new Map();
  for (const [i, w, m, n, mean, lo, hi, nAll, meanAll] of expected.cases[id]) {
    const key = `${i}|${w}`;
    if (!cache.has(key)) cache.set(key, rolling(tables[i], w, meta));
    const s = cache.get(key)[m], t = toDate(d.restaurants[i], tables[i], m);
    const got = s ?? { n: 0, mean: null, lo: null, hi: null };
    for (const [a, b] of [[got.mean, mean], [got.lo, lo], [got.hi, hi], [t.mean, meanAll]]) if (a != null && b != null) worstRolling = Math.max(worstRolling, Math.abs(a - b));
    if (got.n !== n || t.n !== nAll || !close(got.mean, mean, 1e-12) || !close(got.lo, lo, 1e-12) || !close(got.hi, hi, 1e-12) || !close(t.mean, meanAll, 1e-12)) {
      failures++; console.log(`MISMATCH ${id} #${i} window ${w} month ${m}`);
    }
    values++;
  }
}
console.log(`${candidates} candidates: ${failures} failures; max smoothed-rating error ${worstRating.toExponential(1)}; ` +
            `${values} rolling-window values, max error ${worstRolling.toExponential(1)}`);
process.exit(failures ? 1 : 0);
