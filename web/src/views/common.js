// Shared vocabulary for every view: steering steps, the controls behind them, and their effect hues.
export const STEPS = [
  { key: "default", label: "Model default", short: "Default", hue: null },
  { key: "model", label: "History weight", short: "History", hue: "model", control: "w_h" },
  { key: "distance", label: "Distance", hue: "control", control: "alpha" },
  { key: "popularity", label: "Popularity", hue: "control", control: "beta" },
  { key: "boost", label: "Neighborhood", short: "Boost", hue: "control", control: "boost" },
  { key: "diversity", label: "Variety", hue: "diversity", control: "lam" },
];
export const CONTROL_INFO = {
  w_h: { name: "history", hue: "model" },
  alpha: { name: "distance", hue: "control" },
  beta: { name: "popularity", hue: "control" },
  lam: { name: "variety", hue: "diversity" },
  boost: { name: "neighborhood", hue: "control" },
};
export const MODEL_NAMES = { lr: "Logistic regression", lgbm: "LightGBM" };
export const hueVar = (hue) => (hue ? `var(--${hue})` : "var(--track)");

export const STAR = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 .8l1.6 3.3 3.6.5-2.6 2.5.6 3.6L6 9l-3.2 1.7.6-3.6L.8 4.6l3.6-.5z"/></svg>';
export const CHECK = '<svg viewBox="0 0 12 12" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M2.5 6.5l2.3 2.3 4.7-5"/></svg>';
export const CROSS = '<svg viewBox="0 0 12 12" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 3l6 6M9 3l-6 6"/></svg>';

export const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
export const truncate = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
