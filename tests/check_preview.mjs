// Console-preview check: for every demo case, model, console state and control, the browser's sweep
// (web/src/preview.js) must match src/25_export_tradeoff.py (src/steering.py's solver): same values met,
// same outcomes, same predicted visits kept. Run: node tests/check_preview.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { previewSweep } from "../web/src/preview.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (...p) => readFileSync(join(root, ...p), "utf8");
const { platt } = JSON.parse(read("web", "public", "data", "tradeoff.json"));
const { states, cases } = JSON.parse(read("tests", "preview_expected.json"));
let failures = 0, worst = 0, points = 0;
for (const [id, sweeps] of Object.entries(cases)) {
  const c = JSON.parse(read("web", "public", "data", "cases", `${id}.json`));
  for (const [key, want] of Object.entries(sweeps)) {
    const [m, state, control] = key.split("|");
    const { w_h, targets } = states[state];
    previewSweep(c, m, platt, w_h, targets, control).forEach((d, k) => {
      const [kept, met, km, pop, cui] = want[k];
      const err = Math.max(Math.abs(d.kept - kept), Math.abs(d.outcomes.km - km), Math.abs(d.outcomes.popularity - pop));
      worst = Math.max(worst, err);
      points++;
      if (d.met !== met || d.outcomes.cuisines !== cui || err > 1e-12) {
        failures++;
        if (failures < 6) console.log(`MISMATCH ${id} ${key} #${k}`, { kept: d.kept, met: d.met, ...d.outcomes }, want[k]);
      }
    });
  }
}
console.log(`${points} preview points: ${failures} failures; max error ${worst.toExponential(1)}`);
process.exit(failures ? 1 : 0);
