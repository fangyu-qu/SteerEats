// The browser's one-sentence code, run from Python (tests/check_nl.py, src/40_nl_eval.py).
// Reads one JSON request on stdin, writes one JSON answer on stdout.
//   {"cmd":"constants"}                               the slot vocabularies
//   {"cmd":"normalize","raw":[...]}                   normalize() of each raw answer
//   {"cmd":"keyword","sentences":[...]}               keywordParse() of each sentence
//   {"cmd":"pool","jobs":[{case,filters,thetas,targets}]}   pool index, final ranks, top 10 and solved settings on the filtered pool
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as nl from "../web/src/nl.js";
import { finalRanks, solveTargets, topK } from "../web/src/steer.js";

const CASES = join(dirname(fileURLToPath(import.meta.url)), "..", "web", "public", "data", "cases");
const req = JSON.parse(readFileSync(0, "utf8"));
let out;
if (req.cmd === "constants")
  out = { cuisines: Object.keys(nl.CUISINES), needs: Object.keys(nl.NEEDS), boroughs: nl.BOROUGHS, popularity: Object.keys(nl.POPULARITY),
    variety: Object.keys(nl.VARIETY), history: Object.keys(nl.HISTORY), km_range: nl.KM_RANGE, km_step: nl.KM_STEP, max_sentence: nl.MAX_SENTENCE };
else if (req.cmd === "normalize") out = req.raw.map(nl.normalize);
else if (req.cmd === "keyword") out = req.sentences.map(nl.keywordParse);
else if (req.cmd === "pool") {
  const cache = new Map();
  const load = (f) => { if (!cache.has(f)) cache.set(f, JSON.parse(readFileSync(join(CASES, f), "utf8"))); return cache.get(f); };
  out = req.jobs.map((job) => {
    const full = load(job.case), m = nl.poolMask(full, nl.asFilters(job.filters));
    if (m.size === 0) return { index: [] };
    const c = nl.filterCase(full, m.keep, nl.filterKey(nl.asFilters(job.filters)));
    const res = { index: c.pool.index, runs: [] };
    for (const model of ["lr", "lgbm"]) {
      for (const theta of job.thetas) res.runs.push({ model, ranks: Array.from(finalRanks(c, theta, model)), top: Array.from(topK(c, theta, model)) });
      for (const targets of job.targets) { const s = solveTargets(c, model, targets); res.runs.push({ model, theta: s.theta, top: Array.from(s.top), met: s.met }); }
    }
    return res;
  });
} else throw new Error(`unknown cmd ${req.cmd}`);
process.stdout.write(JSON.stringify(out));
