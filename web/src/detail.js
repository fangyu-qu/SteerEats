// Rating-history arithmetic for the restaurant detail view (V8). Mirrors src/29_export_restaurant_detail.py;
// checked by tests/check_detail.mjs. A restaurant arrives as { b: [n1..n5], m: [[month, n1..n5], ...] }: star
// counts before the first exported month, then the months that have a review. `meta` is detail/index.json.

const STARS = [1, 2, 3, 4, 5];
const zero = () => [0, 0, 0, 0, 0];

// Dense months x 5 table of star counts.
export function expand(rest, months) {
  const h = Array.from({ length: months }, zero);
  for (const [m, ...counts] of rest.m) h[m] = counts;
  return h;
}

// Star counts summed over months from..to (inclusive).
export function sumCounts(h, from, to) {
  const out = zero();
  for (let m = Math.max(0, from); m <= Math.min(to, h.length - 1); m++) for (let k = 0; k < 5; k++) out[k] += h[m][k];
  return out;
}

const total = (counts) => counts.reduce((a, v) => a + v, 0);
const starSum = (counts) => counts.reduce((a, v, k) => a + v * STARS[k], 0);
const plainMean = (counts) => { const n = total(counts); return n ? starSum(counts) / n : null; };

// Mean and 95% band of the ratings in one window. The band is mean ± z standard errors, clipped to 1–5 stars;
// the window's variance is pooled with the city-wide variance at the weight of `pool_weight` reviews, so a few
// identical ratings do not give a band of zero width.
export function windowStats(counts, meta) {
  const n = total(counts);
  if (n === 0) return null;
  const mean = plainMean(counts);
  const ss = counts.reduce((a, v, k) => a + v * (STARS[k] - mean) ** 2, 0);
  const se = Math.sqrt((ss + meta.pool_weight * meta.sd_all ** 2) / (n + meta.pool_weight - 1) / n);
  return { n, mean, lo: Math.max(1, mean - meta.z * se), hi: Math.min(5, mean + meta.z * se), counts };
}

// One entry per month: the window of `w` months ending in that month (null where it holds no review).
export function rolling(h, w, meta) {
  return h.map((_, m) => windowStats(sumCounts(h, m - w + 1, m), meta));
}

// All reviews up to and including a month.
export function toDate(rest, h, month) {
  const counts = sumCounts(h, 0, month).map((v, k) => v + rest.b[k]);
  return { n: total(counts), mean: plainMean(counts), counts };
}

// What the rankers knew at the snapshot, and what arrived later. `smoothed` and `smoothedRecent` reproduce the
// rankers' rating features (ranker_core.Snapshot): shrunk toward the city-wide mean, then toward the all-time value.
export function snapshot(rest, h, meta) {
  const cut = meta.cutoff_index, k = meta.prior_strength;
  const all = toDate(rest, h, cut - 1).counts, recent = sumCounts(h, cut - meta.recent_months, cut - 1);
  const smoothed = (starSum(all) + k * meta.prior_mean) / (total(all) + k);
  const smoothedRecent = (starSum(recent) + k * smoothed) / (total(recent) + k);
  return { all: windowStats(all, meta), recent: windowStats(recent, meta),
    later: windowStats(sumCounts(h, cut, h.length - 1), meta), smoothed, smoothedRecent };
}

// "2019-03" for a month index.
export function monthLabel(meta, m) {
  const [y, mo] = meta.first_month.split("-").map(Number);
  const t = (y * 12 + (mo - 1)) + m;
  return `${Math.floor(t / 12)}-${String((t % 12) + 1).padStart(2, "0")}`;
}
