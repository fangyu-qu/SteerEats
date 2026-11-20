// Client-side steering and change accounting. Mirrors src/steering.py operation for operation so the
// browser reproduces the Python rankings exactly (checked by tests/replay.mjs).
//
// theta = { w_h, alpha, beta, lam } in [0, 1]; THETA0 reproduces the model's own ranking.
// Stages: S0 model (history weight on the collaborative SHAP part, then min-max) -> S1 distance
// -> S2 popularity -> S3 MMR diversity over cuisine group and neighbourhood.

export const CONTROLS = ["w_h", "alpha", "beta", "lam"];
export const THETA0 = Object.freeze({ w_h: 1, alpha: 0, beta: 0, lam: 0 });
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
  const o3 = mmrOrder(minmax(s2), sim, theta.lam);
  return {
    S0: ranksFromOrder(order(s0)), S1: ranksFromOrder(order(s1)),
    S2: ranksFromOrder(order(s2)), S3: ranksFromOrder(o3), score: { S0: s0, S1: s1, S2: s2 }, order: o3,
  };
}

export const finalRanks = (c, theta, model, sim) => stages(c, theta, model, sim).S3;

// Fixed-order decomposition of each candidate's change from THETA0's final rank to theta's.
export function stageLog(c, theta, model, sim = similarity(c)) {
  const before = finalRanks(c, THETA0, model, sim);
  const st = stages(c, theta, model, sim);
  const n = before.length;
  const steps = { model: new Int32Array(n), distance: new Int32Array(n), popularity: new Int32Array(n), diversity: new Int32Array(n) };
  const total = new Int32Array(n), residual = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    steps.model[i] = st.S0[i] - before[i];
    steps.distance[i] = st.S1[i] - st.S0[i];
    steps.popularity[i] = st.S2[i] - st.S1[i];
    steps.diversity[i] = st.S3[i] - st.S2[i];
    total[i] = st.S3[i] - before[i];
    residual[i] = total[i] - (steps.model[i] + steps.distance[i] + steps.popularity[i] + steps.diversity[i]);
  }
  return { before, after: st.S3, steps, total, residual, order: st.order };
}

const FACT = [1, 1, 2, 6, 24];

// Exact Shapley attribution of each candidate's final-rank change to the four controls (16 rankings).
export function shapleyControls(c, theta, model, sim = similarity(c)) {
  const cache = new Map();
  const rankWith = (subset) => {
    const key = [...subset].sort().join(",");
    if (!cache.has(key)) {
      const t = { ...THETA0 };
      for (const k of subset) t[k] = theta[k];
      cache.set(key, finalRanks(c, t, model, sim));
    }
    return cache.get(key);
  };
  const n = c.cuisine.length, N = CONTROLS.length;
  const phi = Object.fromEntries(CONTROLS.map((k) => [k, new Float64Array(n)]));
  for (const k of CONTROLS) {
    const others = CONTROLS.filter((o) => o !== k);
    for (let mask = 0; mask < 1 << others.length; mask++) {
      const s = others.filter((_, j) => mask & (1 << j));
      const w = (FACT[s.length] * FACT[N - s.length - 1]) / FACT[N];
      const withK = rankWith([...s, k]), without = rankWith(s);
      for (let i = 0; i < n; i++) phi[k][i] += w * (withK[i] - without[i]);
    }
  }
  const full = rankWith(CONTROLS), base = rankWith([]);
  const total = new Float64Array(n), residual = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    total[i] = full[i] - base[i];
    residual[i] = total[i] - CONTROLS.reduce((a, k) => a + phi[k][i], 0);
  }
  return { phi, total, residual };
}
