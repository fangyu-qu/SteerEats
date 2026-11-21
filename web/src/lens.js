// Model-lens arithmetic (view V6). Mirrors src/22_model_lens.py; checked by tests/check_lens.mjs.

// Kendall tau-b between two score vectors (ties handled as in scipy.stats.kendalltau).
export function kendallTauB(x, y) {
  const n = x.length;
  let s = 0, tx = 0, ty = 0, pairs = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const a = Math.sign(x[i] - x[j]), b = Math.sign(y[i] - y[j]);
      s += a * b; pairs++;
      if (a === 0) tx++;
      if (b === 0) ty++;
    }
  }
  const d = Math.sqrt((pairs - tx) * (pairs - ty));
  return d > 0 ? s / d : NaN;
}

// Restaurants two rank vectors both place in the top k.
export function overlapAt(ra, rb, k = 10) {
  let n = 0;
  for (let i = 0; i < ra.length; i++) if (ra[i] <= k && rb[i] <= k) n++;
  return n;
}

// Share of mean |SHAP| per feature group over this user's candidates (step 17's definition, summed by feature).
export function groupShares(c, model) {
  const imp = {};
  let tot = 0;
  for (const f of c.features) {
    const v = c[model].shap[f];
    let a = 0;
    for (const x of v) a += Math.abs(x);
    imp[f] = a / v.length; tot += imp[f];
  }
  return Object.fromEntries(Object.entries(c.feature_groups).map(([g, fs]) => [g, fs.reduce((a, f) => a + imp[f], 0) / tot]));
}

export function pearson(x, y) {
  const n = x.length;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : NaN;
}

export function sd(v) {
  let m = 0;
  for (const x of v) m += x;
  m /= v.length;
  let s = 0;
  for (const x of v) s += (x - m) ** 2;
  return Math.sqrt(s / v.length);
}

// Mid-rank percentile of each value among the vector (0 = lowest, 1 = highest; ties share a value).
export function percentiles(v) {
  const n = v.length, idx = Array.from(v, (_, i) => i).sort((a, b) => v[a] - v[b]);
  const out = new Float64Array(n);
  for (let s = 0; s < n;) {
    let e = s;
    while (e + 1 < n && v[idx[e + 1]] === v[idx[s]]) e++;
    const p = n > 1 ? (s + e) / 2 / (n - 1) : 0.5;
    for (let k = s; k <= e; k++) out[idx[k]] = p;
    s = e + 1;
  }
  return out;
}
