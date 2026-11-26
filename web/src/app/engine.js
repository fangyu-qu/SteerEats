// Diner app engine: from a party (one or two sample taste profiles with start points) and its settings to the
// list of K places. Mirrors src/app_engine.py operation for operation (tests/check_app.py); change both or neither.
//
// Pool      union of the members' stage-1 candidates (as in the offline group study, src/33_rq2_groups.py),
//           minus the places a filter removes (hard vetoes).
// Member    z = (raw - collab) + w_h * collab, min-max over the pool, minus alpha * distance from that member's
//           start, minus beta * popularity percentile in the pool: the same steps as web/src/steer.js.
// Group     satisfaction = member score, min-max over the pool (web/src/group.js); combined by the strategy.
// Variety   one setting for the list: MMR over cuisine group and neighbourhood on the group score.
// Targets   "within X km", "less mainstream", "more variety" are outcomes of the list; the solver finds the
//           smallest alpha, beta and lam that reach them, like solveTargets in steer.js.
import { TAU, groupScore, rankPool, satisfaction } from "../group.js";
import { minmax } from "../steer.js";
import { poolMask } from "../nl.js";

export const K = 6;
export const SIM_CUISINE = 0.5, SIM_NTA = 0.5;
export const THETA0 = Object.freeze({ w_h: 1, alpha: 0, beta: 0, lam: 0 });
const grid = (n, d) => Array.from({ length: n }, (_, k) => +(k * d).toFixed(4));
export const SOLVER_GRIDS = { alpha: grid(51, 0.02), beta: grid(51, 0.02), lam: grid(51, 0.01) };
export const STRATEGY = { fair: "average_without_misery", floor: "least_misery", solo: "average" };
export const NO_TARGETS = Object.freeze({ km: null, popularity: null, cuisines: null });

const round6 = (v) => Math.round(v * 1e6) / 1e6;
// Same formula as ranker_core.haversine_km, rounded to a metre's fraction so both languages agree exactly.
export function km(lat0, lon0, lat, lon) {
  const p = Math.PI / 180;
  const a = Math.sin((lat - lat0) * p / 2) ** 2 + Math.cos(lat0 * p) * Math.cos(lat * p) * Math.sin((lon - lon0) * p / 2) ** 2;
  return round6(12742.0 * Math.asin(Math.sqrt(a)));
}

// The party's pool: place indices (ascending) that at least one member has as a candidate and that pass the filters.
// `filters` are those of web/src/nl.js plus min_rating.
export function buildPool(places, cuisineNames, members, filters) {
  const seen = new Set();
  for (const m of members) for (const i of m.profile.candidates) seen.add(i);
  const union = [...seen].sort((a, b) => a - b);
  // a case-shaped view of the union, so the filter code of the Explorer is used as it is
  const view = {
    cuisine: union.map((i) => places.cuisine[i]), cuisine_names: cuisineNames,
    candidates: { price: union.map((i) => places.price[i]), borough: union.map((i) => places.borough[i]), nta_name: union.map((i) => places.nta_name[i]) },
    feature_values: Object.fromEntries(Object.entries(places.needs).map(([a, col]) => [`attr_${a}`, union.map((i) => col[i])])),
  };
  const m = poolMask(view, filters);
  const ids = union.filter((i, k) => m.keep[k] && (filters.min_rating == null || places.rating[i] >= filters.min_rating));
  return { ids, union: union.length, areas: m.areas };
}

// Popularity percentile within the pool: 0 for the place with the fewest reviews in the past year, 1 for the most;
// ties keep pool order (the same rule as the demo cases).
export function popularity(places, ids) {
  const n = ids.length, order = ids.map((_, k) => k);
  order.sort((a, b) => places.recent_reviews[ids[a]] - places.recent_reviews[ids[b]] || a - b);
  const out = new Float64Array(n);
  order.forEach((k, rank) => { out[k] = n > 1 ? rank / (n - 1) : 0; });
  return out;
}

// Everything that does not depend on the controls: per member distance, per place popularity.
export function prepare(places, cap, members, ids) {
  const dist = members.map((m) => ids.map((i) => km(m.start.lat, m.start.lon, places.lat[i], places.lon[i])));
  return { ids, n: ids.length, dist, distN: dist.map((row) => row.map((d) => Math.min(d, cap) / cap)), pop: popularity(places, ids),
    cuisine: ids.map((i) => places.cuisine[i]), nta: ids.map((i) => places.nta[i]) };
}

// Each member's steered score over the pool, with the parts kept for the ledger.
export function memberScores(members, pre, theta) {
  return members.map((m, k) => {
    const p = m.profile, n = pre.n, z = new Float64Array(n);
    for (let j = 0; j < n; j++) { const i = pre.ids[j]; z[j] = (p.raw[i] - p.collab[i]) + theta.w_h * p.collab[i]; }
    const s0 = minmax(z), s = new Float64Array(n);
    for (let j = 0; j < n; j++) s[j] = s0[j] - theta.alpha * pre.distN[k][j] - theta.beta * pre.pop[j];
    return { s0, score: s };
  });
}

// The first k of the list: the strategy's order, re-ranked for variety when lam > 0.
export function listOf(pre, sat, strategy, lam, k = K) {
  const { score, feasible } = groupScore(sat, strategy, TAU, null);
  const order = rankPool(score, feasible, pre.ids);
  if (lam === 0 || order.length === 0) return { top: order.slice(0, k), score, feasible, count: order.length };
  const rel = minmax(order.map((j) => score[j]));          // over the feasible places, best first
  const taken = new Uint8Array(order.length), maxSim = new Float64Array(order.length), top = [];
  for (let step = 0; step < Math.min(k, order.length); step++) {
    let best = -1, bestVal = -Infinity;
    for (let a = 0; a < order.length; a++) {
      if (taken[a]) continue;
      const v = (1 - lam) * rel[a] - lam * maxSim[a];
      if (v > bestVal) { best = a; bestVal = v; }
    }
    taken[best] = 1; top.push(order[best]);
    const jb = order[best];
    for (let a = 0; a < order.length; a++) {
      const j = order[a];
      const s = SIM_CUISINE * (pre.cuisine[j] === pre.cuisine[jb] ? 1 : 0) + SIM_NTA * (pre.nta[j] === pre.nta[jb] ? 1 : 0);
      if (s > maxSim[a]) maxSim[a] = s;
    }
  }
  return { top, score, feasible, count: order.length };
}

// What the list is like: the farthest member's average distance, average popularity, number of cuisine groups.
export function outcomes(pre, top) {
  if (!top.length) return { km: 0, popularity: 0, cuisines: 0 };
  let worst = 0, pop = 0;
  for (const row of pre.dist) { let s = 0; for (const j of top) s += row[j]; if (s / top.length > worst) worst = s / top.length; }
  for (const j of top) pop += pre.pop[j];
  return { km: worst, popularity: pop / top.length, cuisines: new Set(top.map((j) => pre.cuisine[j])).size };
}

const meets = (o, t, key) => t[key] == null || (key === "cuisines" ? o.cuisines >= t[key] : o[key] <= t[key]);

export function run(members, pre, theta, strategy, k = K) {
  const ms = memberScores(members, pre, theta);
  const sat = satisfaction(ms.map((m) => Array.from(m.score)));
  const list = listOf(pre, sat, strategy, theta.lam, k);
  return { theta, ms, sat, ...list, outcomes: outcomes(pre, list.top) };
}

// targets: { km: farthest member's average distance at most, popularity: average percentile at most, cuisines: groups at least }
export function solve(members, pre, targets, w_h, strategy, k = K, passes = 2) {
  const theta = { ...THETA0, w_h };
  const knobs = [["alpha", "km"], ["beta", "popularity"], ["lam", "cuisines"]];
  for (let p = 0; p < passes; p++) {
    for (const [key, target] of knobs) {
      if (targets[target] == null) continue;
      const g = SOLVER_GRIDS[key];
      let chosen = g[g.length - 1];
      for (const v of g) {
        theta[key] = v;
        if (meets(run(members, pre, theta, strategy, k).outcomes, targets, target)) { chosen = v; break; }
      }
      theta[key] = chosen;
    }
  }
  const res = run(members, pre, theta, strategy, k);
  return { ...res, met: { km: meets(res.outcomes, targets, "km"), popularity: meets(res.outcomes, targets, "popularity"), cuisines: meets(res.outcomes, targets, "cuisines") } };
}

// The feature group with the largest positive SHAP value for a member and a place (null when none is positive).
export function mainReason(profile, groups, i) {
  let best = null, bestVal = 0;
  for (const g of groups) { const v = profile.groups[g][i]; if (v > bestVal) { best = g; bestVal = v; } }
  return best;
}
