// Trade-off space (V5): this user's outcomes on the RQ1 joint grid, predicted visits, and Pareto frontiers.
// Checked against src/25_export_tradeoff.py and outputs/rq1/frontiers.csv by tests/check_tradeoff.mjs.
import { LAM_MAX, THETA0, outcomes, topK } from "./steer.js";

// Step 20's joint grid, in the same order (w_h outermost, lam innermost).
export const TRADE_GRID = { w_h: [0, 0.5, 1], alpha: [0, 0.25, 0.5, 1], beta: [0, 0.25, 0.5, 1], lam: [0, 0.125, 0.25, 0.5] };
export const SETTINGS = [];
for (const w_h of TRADE_GRID.w_h) for (const alpha of TRADE_GRID.alpha) for (const beta of TRADE_GRID.beta) for (const lam of TRADE_GRID.lam)
  SETTINGS.push({ w_h, alpha, beta, lam });

// L1 distance from the model's own list, every control on its [0, 1] slider scale (as in step 20).
export const l1 = (t) => (1 - t.w_h) + t.alpha + t.beta + t.lam / LAM_MAX;
export const sliderDistance = (a, b) =>
  Math.abs(a.w_h - b.w_h) + Math.abs(a.alpha - b.alpha) + Math.abs(a.beta - b.beta) + Math.abs(a.lam - b.lam) / LAM_MAX;

// Platt-calibrated probability of a later visit for every candidate (model's own score, unsteered).
export function visitProbs(c, model, platt) {
  const [a, b] = platt[model];
  return c[model].raw.map((s) => 1 / (1 + Math.exp(-(a * s + b))));
}

export function predictedVisits(p, top) {
  let s = 0;
  for (const i of top) s += p[i];
  return s;
}

// One point per grid setting: the top 10's outcomes and the model's predicted visits.
export function userSpace(c, model, platt) {
  const p = visitProbs(c, model, platt);
  return SETTINGS.map((s, k) => {
    const top = topK(c, { ...THETA0, ...s }, model);
    const o = outcomes(c, top);
    return { k, ...s, top, km: o.km, popularity: o.popularity, cuisines: o.cuisines, visits: predictedVisits(p, top), l1: l1(s) };
  });
}

// Pareto frontier: walk from the best x to the worst, keep each point that beats every y seen so far.
// Ties in x put the higher y first, so a tied point is kept only if it is strictly better.
export function frontier(points, x, betterLow, y) {
  const s = [...points].sort((a, b) => (betterLow ? a[x] - b[x] : b[x] - a[x]) || b[y] - a[y]);
  let best = -Infinity;
  const keep = [];
  for (const p of s) if (p[y] > best) { keep.push(p); best = p[y]; }
  return keep;
}

// Grid settings that give this user the same top 10 collapse into one point (represented by the
// smallest change from the model's list).
export function groupByList(points) {
  const groups = new Map();
  for (const p of points) {
    const key = [...p.top].sort((a, b) => a - b).join(",");
    if (!groups.has(key)) groups.set(key, { ...p, members: [] });
    const g = groups.get(key);
    g.members.push(p);
    if (p.l1 < g.l1 || (p.l1 === g.l1 && p.k < g.k)) Object.assign(g, { ...p, members: g.members });
  }
  return [...groups.values()];
}
