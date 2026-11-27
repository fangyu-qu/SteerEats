// Group aggregation in the browser. Mirrors src/group.py exactly (tests/check_group.mjs); change both or neither.
export const TAU = 0.4; // misery threshold of "average without misery", fixed in advance
export const STRATEGIES = ["average", "least_misery", "average_without_misery"];
export const STRATEGY_LABEL = { average: "Average", least_misery: "Least misery", average_without_misery: "Average without misery" };

// scores[m][i]: member m's (steered) score for pool item i. Returns satisfaction in [0, 1] per member and item.
export function satisfaction(scores, how = "minmax") {
  if (how === "minmax") {
    return scores.map((row) => {
      let lo = Infinity, hi = -Infinity;
      for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
      const span = hi - lo;
      return row.map((v) => (span > 0 ? (v - lo) / span : 1)); // an indifferent member never blocks a place
    });
  }
  if (how === "percentile") {
    return scores.map((row) => {
      const n = row.length;
      if (n === 1) return [1];
      const order = row.map((_, i) => i).sort((a, b) => row[a] - row[b] || a - b);
      const ranks = new Array(n);
      for (let s = 0; s < n;) {
        let e = s;
        while (e + 1 < n && row[order[e + 1]] === row[order[s]]) e++;
        const mean = (s + e) / 2; // ties share their mean rank
        for (let k = s; k <= e; k++) ranks[order[k]] = mean;
        s = e + 1;
      }
      return ranks.map((r) => r / (n - 1));
    });
  }
  throw new Error(`unknown satisfaction: ${how}`);
}

// allowed carries hard vetoes: an item any member vetoes is out.
export function groupScore(sat, strategy, tau = TAU, allowed = null) {
  const n = sat[0].length, m = sat.length;
  const feasible = allowed ? allowed.map(Boolean) : new Array(n).fill(true);
  const score = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0, worst = Infinity;
    for (let k = 0; k < m; k++) { sum += sat[k][i]; if (sat[k][i] < worst) worst = sat[k][i]; }
    if (strategy === "average") score[i] = sum / m;
    else if (strategy === "least_misery") score[i] = worst;
    else if (strategy === "average_without_misery") { score[i] = sum / m; if (!(worst >= tau)) feasible[i] = false; }
    else throw new Error(`unknown strategy: ${strategy}`);
  }
  return { score, feasible };
}

// Positions of the feasible items, best first; ties broken by the smaller id.
export function rankPool(score, feasible, ids) {
  const pos = [];
  for (let i = 0; i < score.length; i++) if (feasible[i]) pos.push(i);
  return pos.sort((a, b) => score[b] - score[a] || ids[a] - ids[b]);
}

export function groupList(scores, ids, strategy, { k = 10, tau = TAU, how = "minmax", allowed = null } = {}) {
  const sat = satisfaction(scores, how);
  const { score, feasible } = groupScore(sat, strategy, tau, allowed);
  return { order: rankPool(score, feasible, ids).slice(0, k), sat, score, feasible };
}
