// Shared vocabulary for every view: steering steps, the controls behind them, and their effect hues.
export const STEPS = [
  { key: "default", label: "Model default", short: "Default", hue: null },
  { key: "model", label: "History weight", short: "History", hue: "model", control: "w_h" },
  { key: "distance", label: "Distance", hue: "control", control: "alpha" },
  { key: "popularity", label: "Popularity", hue: "control", control: "beta" },
  { key: "boost", label: "Neighborhood", short: "Area", hue: "control", control: "boost" },
  { key: "diversity", label: "Variety", hue: "diversity", control: "lam" },
];
export const CONTROL_INFO = {
  w_h: { name: "history", hue: "model" },
  alpha: { name: "distance", hue: "control" },
  beta: { name: "popularity", hue: "control" },
  lam: { name: "variety", hue: "diversity" },
  boost: { name: "neighborhood", hue: "control" },
};
export const MODEL_NAMES = { lr: "Logistic regression", lgbm: "LightGBM", lmr: "LambdaRank" };
// One colour and one shape per ranker, the same in every chart (tokens --m-lr, --m-lgbm, --m-lmr).
export const MODEL_FILL = { lr: "var(--m-lr)", lgbm: "var(--m-lgbm)", lmr: "var(--m-lmr)" };
export const MODEL_SHAPE = { lr: "square", lgbm: "circle", lmr: "triangle" };
// The three baselines of the accuracy chart: their own colours, one shape (a diamond) for "baseline".
export const BASELINE_FILL = { Popularity: "var(--m-pop)", "Recent popularity": "var(--m-recent)", "Item-CF": "var(--m-cf)" };
// Text in a series colour: the colour mixed into ink, so it keeps a reading contrast on the cards.
export const inkOf = (fill) => `color-mix(in oklab, ${fill} 45%, var(--ink))`;
export const MODEL_KEY = { "Logistic regression": "lr", LightGBM: "lgbm", LambdaRank: "lmr", "LightGBM LambdaRank (supplementary)": "lmr" };
// The model each one is compared with unless the reader chooses another.
export const VERSUS = { lr: "lgbm", lgbm: "lr", lmr: "lgbm" };
// What a model's raw score measures. LambdaRank is trained to order a list; its score is not a probability.
export const SCORE_UNIT = { lr: "log-odds", lgbm: "log-odds", lmr: "score units" };
export const hueVar = (hue) => (hue ? `var(--${hue})` : "var(--track)");
// Lines use the darker line tokens plus a dash pattern, so an effect never rests on colour alone.
export const hueLine = (hue) => (hue ? `var(--${hue}-line)` : "var(--track)");
export const HUE_DASH = { model: null, control: "7 3", diversity: "2 3" };

export const STAR = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 .8l1.6 3.3 3.6.5-2.6 2.5.6 3.6L6 9l-3.2 1.7.6-3.6L.8 4.6l3.6-.5z"/></svg>';
export const CHECK = '<svg viewBox="0 0 12 12" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M2.5 6.5l2.3 2.3 4.7-5"/></svg>';
export const CROSS = '<svg viewBox="0 0 12 12" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 3l6 6M9 3l-6 6"/></svg>';

export const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
export const truncate = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
