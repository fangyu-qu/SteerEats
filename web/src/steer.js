// Client-side steering and change accounting. Mirrors src/steering.py operation for operation so the
// browser reproduces the Python rankings exactly (checked by tests/replay.mjs).
//
// theta = { w_h, alpha, beta, lam } in [0, 1]; THETA0 reproduces the model's own ranking.
// Stages: S0 model (history weight on the collaborative SHAP part, then min-max) -> S1 distance
// -> S2 popularity -> S3 MMR diversity over cuisine group and neighbourhood.

export const CONTROLS = ["w_h", "alpha", "beta", "lam"];
export const THETA0 = Object.freeze({ w_h: 1, alpha: 0, beta: 0, lam: 0, boost: 0, boost_nta: -1 });
export const BOOST_GRID = Array.from({ length: 101 }, (_, k) => +(k * 0.02).toFixed(4));

// Neighbourhood boost (map control): +boost for candidates in neighbourhood boost_nta. Mirrors boost_vector.
function boostAt(c, theta, i) {
  const b = theta.boost || 0, n = theta.boost_nta ?? -1;
  return b && n >= 0 && c.nta[i] === n ? b * 1 : 0;
}
const SIM_CUISINE = 0.5;
const SIM_NTA = 0.5;

export function minmax(v) {
  let lo = Infinity, hi = -Infinity;
  for (const x of v) { if (x < lo) lo = x; if (x > hi) hi = x; }
  const out = new Float64Array(v.length);
  if (hi - lo <= 0) return out;
  for (let i = 0; i < v.length; i++) out[i] = (v[i] - lo) / (hi - lo);
  return out;
}

// Indices best to worst; ties keep candidate order (same as numpy lexsort((index, -score))).
export function order(scores) {
  const idx = Array.from(scores, (_, i) => i);
  idx.sort((a, b) => (scores[b] > scores[a] ? 1 : scores[b] < scores[a] ? -1 : a - b));
  return idx;
}

export function ranksFromOrder(o) {
  const r = new Int32Array(o.length);
  o.forEach((i, pos) => { r[i] = pos + 1; });
  return r;
}

export function similarity(c) {
  const n = c.cuisine.length;
  const sim = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      sim[i][j] = SIM_CUISINE * (c.cuisine[i] === c.cuisine[j] ? 1 : 0) + SIM_NTA * (c.nta[i] === c.nta[j] ? 1 : 0);
  return sim;
}

export function mmrOrder(rel, sim, lam) {
  const n = rel.length;
  if (lam === 0) return order(rel);
  const chosen = [];
  let left = Array.from({ length: n }, (_, i) => i);
  const maxSim = new Float64Array(n);
  for (let step = 0; step < n; step++) {
    let best = -1, bestVal = -Infinity;
    for (const i of left) {
      const v = (1 - lam) * rel[i] - lam * maxSim[i];
      if (v > bestVal) { best = i; bestVal = v; }
    }
    chosen.push(best);
    left = left.filter((i) => i !== best);
    const row = sim[best];
    for (let i = 0; i < n; i++) if (row[i] > maxSim[i]) maxSim[i] = row[i];
  }
  return chosen;
}

export function stages(c, theta, model, sim = similarity(c)) {
  const m = c[model], n = m.raw.length;
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) z[i] = m.noncollab[i] + theta.w_h * m.collab[i];
  const s0 = minmax(z);
  const s1 = new Float64Array(n), s2 = new Float64Array(n);
  for (let i = 0; i < n; i++) s1[i] = s0[i] - theta.alpha * c.distance_n[i];
  for (let i = 0; i < n; i++) s2[i] = s1[i] - theta.beta * c.popularity_n[i];
  const s2b = new Float64Array(n);
  for (let i = 0; i < n; i++) s2b[i] = s2[i] + boostAt(c, theta, i);
  const o3 = mmrOrder(minmax(s2b), sim, theta.lam);
  return {
    S0: ranksFromOrder(order(s0)), S1: ranksFromOrder(order(s1)), S2: ranksFromOrder(order(s2)),
    S2b: ranksFromOrder(order(s2b)), S3: ranksFromOrder(o3), score: { S0: s0, S1: s1, S2: s2, S2b: s2b }, order: o3,
  };
}

export const finalRanks = (c, theta, model, sim) => stages(c, theta, model, sim).S3;

// Fixed-order decomposition of each candidate's change from THETA0's final rank to theta's.
export function stageLog(c, theta, model, sim = similarity(c)) {
  const before = finalRanks(c, THETA0, model, sim);
  const st = stages(c, theta, model, sim);
  const n = before.length;
  const steps = { model: new Int32Array(n), distance: new Int32Array(n), popularity: new Int32Array(n), boost: new Int32Array(n), diversity: new Int32Array(n) };
  const total = new Int32Array(n), residual = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    steps.model[i] = st.S0[i] - before[i];
    steps.distance[i] = st.S1[i] - st.S0[i];
    steps.popularity[i] = st.S2[i] - st.S1[i];
    steps.boost[i] = st.S2b[i] - st.S2[i];
    steps.diversity[i] = st.S3[i] - st.S2b[i];
    total[i] = st.S3[i] - before[i];
    residual[i] = total[i] - (steps.model[i] + steps.distance[i] + steps.popularity[i] + steps.boost[i] + steps.diversity[i]);
  }
  return { before, after: st.S3, steps, total, residual, order: st.order };
}

const FACT = [1, 1, 2, 6, 24, 120];

// Exact Shapley attribution of each candidate's final-rank change to the controls (16 rankings, or 32
// when the neighbourhood boost is on).
export function shapleyControls(c, theta, model, sim = similarity(c)) {
  const controls = theta.boost ? [...CONTROLS, "boost"] : CONTROLS;
  const cache = new Map();
  const rankWith = (subset) => {
    const key = [...subset].sort().join(",");
    if (!cache.has(key)) {
      const t = { ...THETA0 };
      for (const k of subset) { t[k] = theta[k]; if (k === "boost") t.boost_nta = theta.boost_nta; }
      cache.set(key, finalRanks(c, t, model, sim));
    }
    return cache.get(key);
  };
  const n = c.cuisine.length, N = controls.length;
  const phi = Object.fromEntries(controls.map((k) => [k, new Float64Array(n)]));
  for (const k of controls) {
    const others = controls.filter((o) => o !== k);
    for (let mask = 0; mask < 1 << others.length; mask++) {
      const s = others.filter((_, j) => mask & (1 << j));
      const w = (FACT[s.length] * FACT[N - s.length - 1]) / FACT[N];
      const withK = rankWith([...s, k]), without = rankWith(s);
      for (let i = 0; i < n; i++) phi[k][i] += w * (withK[i] - without[i]);
    }
  }
  const full = rankWith(controls), base = rankWith([]);
  const total = new Float64Array(n), residual = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    total[i] = full[i] - base[i];
    residual[i] = total[i] - controls.reduce((a, k) => a + phi[k][i], 0);
  }
  return { phi, total, residual };
}

// ---------------------------------------------------------------------------------------------
// Outcome targets (interface decision 2026-11-20, "Option B"). Mirrors solve_targets in
// src/steering.py: the user states the outcome, the controls are found per model.
// ---------------------------------------------------------------------------------------------
export const LAM_MAX = 0.5;
const grid = (n, d) => Array.from({ length: n }, (_, k) => +(k * d).toFixed(4));
export const SOLVER_GRIDS = { alpha: grid(51, 0.02), beta: grid(51, 0.02), lam: grid(51, 0.01) };

// First k picks of the final ranking, without the full MMR pass.
export function topK(c, theta, model, k = 10) {
  const m = c[model], n = m.raw.length;
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) z[i] = m.noncollab[i] + theta.w_h * m.collab[i];
  const s0 = minmax(z), s1 = new Float64Array(n), s2 = new Float64Array(n);
  for (let i = 0; i < n; i++) s1[i] = s0[i] - theta.alpha * c.distance_n[i];
  for (let i = 0; i < n; i++) s2[i] = s1[i] - theta.beta * c.popularity_n[i] + boostAt(c, theta, i);
  const rel = minmax(s2), lam = theta.lam;
  if (lam === 0) return order(rel).slice(0, k);
  const taken = new Uint8Array(n), maxSim = new Float64Array(n), out = [];
  for (let step = 0; step < k; step++) {
    let best = -1, bestVal = -Infinity;
    for (let i = 0; i < n; i++) {
      if (taken[i]) continue;
      const v = (1 - lam) * rel[i] - lam * maxSim[i];
      if (v > bestVal) { best = i; bestVal = v; }
    }
    out.push(best); taken[best] = 1;
    for (let i = 0; i < n; i++) {
      const s = SIM_CUISINE * (c.cuisine[i] === c.cuisine[best] ? 1 : 0) + SIM_NTA * (c.nta[i] === c.nta[best] ? 1 : 0);
      if (s > maxSim[i]) maxSim[i] = s;
    }
  }
  return out;
}

export function outcomes(c, top) {
  let km = 0, pop = 0;
  for (const i of top) { km += c.candidates.distance_km[i]; pop += c.popularity_n[i]; }
  return { km: km / top.length, popularity: pop / top.length, cuisines: new Set(top.map((i) => c.cuisine[i])).size };
}

const meets = (o, t, key) => t[key] == null || (key === "cuisines" ? o.cuisines >= t[key] : o[key] <= t[key]);

// targets: { km: max mean km | null, popularity: max mean percentile | null, cuisines: min groups | null }
export function solveTargets(c, model, targets, w_h = 1, passes = 2) {
  const theta = { ...THETA0, w_h };
  const knobs = [["alpha", "km"], ["beta", "popularity"], ["lam", "cuisines"]];
  for (let p = 0; p < passes; p++) {
    for (const [key, target] of knobs) {
      if (targets[target] == null) continue;
      const g = SOLVER_GRIDS[key];
      let chosen = g[g.length - 1];
      for (const v of g) {
        theta[key] = v;
        if (meets(outcomes(c, topK(c, theta, model)), targets, target)) { chosen = v; break; }
      }
      theta[key] = chosen;
    }
  }
  const top = topK(c, theta, model), o = outcomes(c, top);
  return { theta, top, outcomes: o, met: { km: meets(o, targets, "km"), popularity: meets(o, targets, "popularity"), cuisines: meets(o, targets, "cuisines") } };
}

// Smallest boost on BOOST_GRID that puts a restaurant from neighbourhood `nta` into the top 10.
export function solveBoost(c, model, theta, nta) {
  const t = { ...theta, boost_nta: nta };
  for (const b of BOOST_GRID) {
    t.boost = b;
    if (topK(c, t, model).some((i) => c.nta[i] === nta)) return { theta: t, reached: true };
  }
  t.boost = BOOST_GRID[BOOST_GRID.length - 1];
  return { theta: t, reached: false };
}

// Steering cost (proposal 4.2) for every neighbourhood among the candidates: the smallest L1 change of the
// four controls from the model's own list (lam on its slider scale) that brings one of its restaurants into
// the top 10, searched on the RQ1 grid. Infinity = not reachable with these controls.
export const COST_GRID = { w_h: [0, 0.5, 1], alpha: [0, 0.25, 0.5, 1], beta: [0, 0.25, 0.5, 1], lam: [0, 0.125, 0.25, 0.5] };
export function steeringCost(c, model) {
  const cost = new Map(c.nta.map((n) => [n, Infinity]));
  const how = new Map();
  for (const w_h of COST_GRID.w_h) for (const alpha of COST_GRID.alpha) for (const beta of COST_GRID.beta) for (const lam of COST_GRID.lam) {
    const l1 = (1 - w_h) + alpha + beta + lam / LAM_MAX;
    const t = { ...THETA0, w_h, alpha, beta, lam };
    for (const i of topK(c, t, model)) {
      const n = c.nta[i];
      if (l1 < cost.get(n)) { cost.set(n, l1); how.set(n, { w_h, alpha, beta, lam }); }
    }
  }
  return { cost, how };
}
