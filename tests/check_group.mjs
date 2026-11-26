// Parity: web/src/group.js against src/group.py on the cases of tests/group_expected.json (tests/check_group.py --write).
import { readFileSync } from "node:fs";
import { STRATEGIES, groupScore, rankPool, satisfaction } from "../web/src/group.js";

const cases = JSON.parse(readFileSync(new URL("./group_expected.json", import.meta.url)));
let failures = 0, checks = 0, maxErr = 0;
for (const [n, c] of cases.entries()) {
  for (const how of ["minmax", "percentile"]) {
    const sat = satisfaction(c.scores, how);
    for (const st of STRATEGIES) {
      const want = c.expected[`${how}|${st}`];
      const { score, feasible } = groupScore(sat, st, c.tau, c.allowed);
      const order = rankPool(score, feasible, c.ids).slice(0, 10);
      const err = Math.max(...score.map((v, i) => Math.abs(v - want.score[i])));
      maxErr = Math.max(maxErr, err);
      const ok = err < 1e-12 && JSON.stringify(order) === JSON.stringify(want.order) && JSON.stringify(feasible) === JSON.stringify(want.feasible);
      checks++;
      if (!ok) { failures++; if (failures <= 5) console.error(`case ${n} ${how} ${st}: mismatch`); }
    }
  }
}
console.log(`${checks} group lists over ${cases.length} cases: ${failures} failures; max score error ${maxErr.toExponential(1)}`);
process.exit(failures ? 1 : 0);
