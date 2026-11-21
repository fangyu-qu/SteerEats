// Console previews (V1): for one control, what the top 10 would look like at every value on its range, with the
// other controls held where they are. Each point re-solves the outcome targets (solveTargets), then reports the
// outcomes and the model's predicted visits kept, as a share of its own list (Platt-calibrated, as in V5).
// Checked against src/steering.py by tests/check_preview.mjs. The neighbourhood boost is not included.
import { THETA0, outcomes, solveTargets, topK } from "./steer.js";
import { predictedVisits, visitProbs } from "./tradeoff.js";

const steps = (lo, hi, d) => Array.from({ length: Math.round((hi - lo) / d) + 1 }, (_, k) => +(lo + k * d).toFixed(2));
// The values each control can take in the console (same ranges and steps as its input).
export const PREVIEW_VALUES = {
  w_h: steps(0, 1, 0.05),
  km: steps(0.5, 6, 0.25),
  popularity: [null, 0.5, 0.4, 0.3, 0.2],
  cuisines: [null, 5, 6, 7, 8],
};

export function previewSweep(c, model, platt, w_h, targets, key) {
  const p = visitProbs(c, model, platt);
  const base = predictedVisits(p, topK(c, THETA0, model));
  return PREVIEW_VALUES[key].map((v) => {
    const t = key === "w_h" ? targets : { ...targets, [key]: v };
    const s = solveTargets(c, model, t, key === "w_h" ? v : w_h);
    const visits = predictedVisits(p, s.top);
    return { v, kept: visits / base, visits, met: key === "w_h" ? true : s.met[key], outcomes: s.outcomes, theta: s.theta };
  });
}
