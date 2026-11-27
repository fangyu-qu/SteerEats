// "Why the model recommends this place" (V3): what the model's own score for one restaurant is made of.
// The cards above it say how the person's settings moved a rank, in places. This one explains the model's score
// before any setting, in the model's score units: SHAP values of the feature groups (interventional, a shared
// background of 200 candidate rows) from the typical candidate to this restaurant's score; base + the groups = the
// raw score (src/19_export_cases.py keeps the largest residual, about 1e-5).
// Two columns, one chart each: on the left the selected restaurant's whole score (start, every factor as a bar from
// one zero line, sum: Yila found a running-total waterfall unclear, 2026-11-26), on the right its factors
// against a second restaurant's, as differences. One sentence over each chart, written from the numbers. The
// person's settings are not here: what they do to a score is in the controls' "Model details" (`ledgerData`,
// checked by tests/check_ledger.mjs).
import { MODEL_NAMES, SCORE_UNIT, esc } from "./common.js";

export const GROUP_LABEL = {
  Collaborative: "Your history (similar places)", Popularity: "Popularity and recency", Rating: "Rating",
  Location: "Location", Taste: "Taste match", Attributes: "Amenities",
};
const yesNo = (v) => (v >= 0.5 ? "yes" : "no");
export const FEATURE = {
  cf_all: ["Similarity to places you reviewed", (v) => Math.expm1(v).toFixed(2)],
  cf_liked: ["Similarity to places you liked", (v) => Math.expm1(v).toFixed(2)],
  recent_cf_interaction: ["Liked-similarity × recent rating", (v) => v.toFixed(2)],
  log_popularity: ["Reviews, all time", (v) => Math.round(Math.expm1(v)).toLocaleString()],
  log_recent_popularity: ["Reviews, past 12 months", (v) => Math.round(Math.expm1(v)).toLocaleString()],
  log_popularity_3m: ["Reviews, past 3 months", (v) => Math.round(Math.expm1(v)).toLocaleString()],
  recent_share: ["Share of reviews in the past year", (v) => `${Math.round(v * 100)}%`],
  months_since_last_review: ["Months since the last review", (v) => v.toFixed(1)],
  log_item_age_months: ["Months since the first review", (v) => Math.round(Math.expm1(v))],
  shrunk_rating: ["Rating, all time (smoothed)", (v) => `${v.toFixed(2)} ★`],
  shrunk_recent_rating: ["Rating, past year (smoothed)", (v) => `${v.toFixed(2)} ★`],
  rating_change: ["Rating change, past year", (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2)],
  log_distance_km: ["Distance from your usual area", (v) => `${Math.expm1(v).toFixed(1)} km`],
  same_nta: ["In your usual neighborhood", yesNo],
  same_borough: ["In your usual borough", yesNo],
  cuisine_affinity: ["Your visits in this cuisine group", (v) => `${Math.round(v * 100)}%`],
  price_gap: ["Price gap from your usual", (v) => `${v.toFixed(1)} levels`],
  item_price_missing: ["Price not listed", yesNo],
  user_price_unknown: ["Your usual price unknown", yesNo],
  is_chain: ["Chain restaurant", yesNo],
  chain_affinity: ["Chain × your chain habit", (v) => `${Math.round(v * 100)}%`],
  log_user_history: ["Places you have reviewed", (v) => Math.round(Math.expm1(v))],
  attr_vegetarian_options: ["Vegetarian options", yesNo], attr_halal: ["Halal food", yesNo],
  attr_late_night: ["Late-night food", yesNo], attr_alcohol: ["Serves alcohol", yesNo],
  attr_reservations: ["Takes reservations", yesNo], attr_good_for_groups: ["Good for groups", yesNo],
  attr_good_for_kids: ["Good for kids", yesNo], attrs_missing: ["No amenity data", yesNo],
};

// Everything the ledger needs for one candidate under the current controls.
export function ledgerData(c, model, theta, ranks, i) {
  const m = c[model], n = m.raw.length;
  let zmin = Infinity, zmax = -Infinity;
  for (let k = 0; k < n; k++) { const z = m.noncollab[k] + theta.w_h * m.collab[k]; if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
  const R = zmax - zmin || 1;
  const z = m.noncollab[i] + theta.w_h * m.collab[i];
  const groups = Object.keys(c.feature_groups).map((g) => {
    const raw = m.shap_groups[g][i];
    const w = g === "Collaborative" ? theta.w_h : 1;
    return { key: g, label: GROUP_LABEL[g] ?? g, full: raw / R, value: (w * raw) / R, scaled: w !== 1 };
  });
  const s0 = (z - zmin) / R;
  const base = s0 - groups.reduce((a, g) => a + g.value, 0);   // typical candidate, on this user's scale
  const dist = -theta.alpha * c.distance_n[i], pop = -theta.beta * c.popularity_n[i];
  const boost = theta.boost && c.nta[i] === theta.boost_nta ? theta.boost : 0;
  return { i, R, base, groups, s0, dist, pop, boost, s2: s0 + dist + pop + boost,
    ranks: { default: ranks.default[i], s2: ranks.boost[i], final: ranks.diversity[i] } };
}

// ---- words and numbers ------------------------------------------------------------------------------------
const IN_SENTENCE = { Collaborative: "your history with similar places", Popularity: "popularity", Rating: "rating",
  Location: "location", Taste: "taste match", Attributes: "amenities" };
const NEAR_ZERO = 0.005;   // below what two decimals can show
const signed = (v) => { const a = Math.abs(v); return a === 0 ? "0" : `${v > 0 ? "+" : "−"}${a < NEAR_ZERO ? "<0.01" : a.toFixed(2)}`; };
const plain = (v) => v.toFixed(2).replace("-", "−");
const list = (items) => (items.length < 3 ? items.join(" and ") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
// the rows are shown to two decimals: when the rounded rows no longer add up to the rounded total, say so
const roundNote = (parts, total) => { const shown = parts.reduce((a, v) => a + +v.toFixed(2), 0);
  return Math.abs(shown - +total.toFixed(2)) > 0.005 ? ` Rounded to two decimals, the rows shown add up to ${plain(shown)}.` : ""; };

// The model's own explanation of one candidate, in the model's score units, before any setting.
export function modelFactors(c, model, i) {
  const m = c[model];
  const groups = Object.keys(c.feature_groups).map((g) => ({ key: g, label: GROUP_LABEL[g] ?? g, value: m.shap_groups?.[g]?.[i] }));
  const ok = groups.length > 0 && groups.every((g) => Number.isFinite(g.value)) && Number.isFinite(m.base) && Number.isFinite(m.raw?.[i]);
  return { ok, base: m.base, score: m.raw?.[i], groups, ranked: [...groups].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)) };
}

// One sentence from the largest factors: what raises the score most, what lowers it most, or that nothing does.
export function summarize(F, shown = 3) {
  if (!F.ok) return "No explanation data is available for this restaurant.";
  const top = F.ranked.slice(0, shown).filter((g) => Math.abs(g.value) >= NEAR_ZERO);
  const up = top.filter((g) => g.value > 0).map((g) => IN_SENTENCE[g.key] ?? g.label.toLowerCase());
  const down = top.filter((g) => g.value < 0).map((g) => IN_SENTENCE[g.key] ?? g.label.toLowerCase());
  const one = (n) => n === 1;
  if (!up.length && !down.length) return "No factor moves this restaurant’s model score much: every contribution is close to zero.";
  if (!down.length) return `${cap(list(up))} ${one(up.length) ? "is the strongest factor" : "are the strongest factors"} raising this restaurant’s model score.`;
  if (!up.length) return `${cap(list(down))} ${one(down.length) ? "is the strongest factor" : "are the strongest factors"} lowering this restaurant’s model score.`;
  return `${cap(list(up))} ${one(up.length) ? "raises" : "raise"} this restaurant’s model score the most; ${list(down)} ${one(down.length) ? "lowers" : "lower"} it the most.`;
}

// Two model scores are called similar when they are less than two usual steps apart at the top of this model's list
// for this user: twice the median gap between places ranked next to each other in its top 10 (so the rule follows
// each model's own score scale). With fewer than 11 candidates there is no such gap, and no pair is called similar.
export function similarGap(c, model) {
  const raw = [...c[model].raw].sort((a, b) => b - a);
  if (raw.length < 11) return null;
  const gaps = raw.slice(0, 10).map((v, k) => v - raw[k + 1]).sort((a, b) => a - b);
  return 2 * gaps[5];
}

// One sentence for the comparison: how the two scores stand, then the factors that differ most and whom they favour.
export function compareSentence(F, G, nameA, nameB, unit, near) {
  const gap = F.score - G.score;
  const diffs = F.groups.map((g, k) => ({ key: g.key, label: g.label, value: g.value - G.groups[k].value }))
    .sort((p, q) => Math.abs(q.value) - Math.abs(p.value)).slice(0, 3).filter((d) => Math.abs(d.value) >= NEAR_ZERO);
  const word = (d) => IN_SENTENCE[d.key] ?? d.label.toLowerCase();
  const forA = diffs.filter((d) => d.value > 0).map(word), forB = diffs.filter((d) => d.value < 0).map(word);
  const fav = (xs) => (xs.length === 1 ? "favours" : "favour");
  const head = near != null && Math.abs(gap) < near ? "The model scores these restaurants similarly."
    : `The model scores ${gap > 0 ? nameA : nameB} higher than ${gap > 0 ? nameB : nameA}, by ${Math.abs(gap) < NEAR_ZERO ? "less than 0.01" : Math.abs(gap).toFixed(2)} ${unit}.`;
  const tail = forA.length && forB.length ? `${cap(list(forA))} ${fav(forA)} ${nameA}, while ${list(forB)} ${fav(forB)} ${nameB}.`
    : forA.length ? `${cap(list(forA))} ${fav(forA)} ${nameA}.`
    : forB.length ? `${cap(list(forB))} ${fav(forB)} ${nameB}.`
    : "No factor differs by much.";
  return `${head} ${tail}`;
}

// ---- drawing: HTML rows, so names wrap and nothing sits on top of anything else -----------------------------
// bars from a zero line in the middle; what each side means is written above them
function diverging(rows, kind, ends) {
  const max = Math.max(...rows.filter((r) => !r.text).map((r) => Math.abs(r.value)), 1e-12);   // (a row with its own text has no bar)
  return `<table class="wf wf-div">
    <thead><tr><th scope="col"><span class="wf-hide">Factor</span></th><th scope="col" class="wf-ends"><span>${ends[0]}</span><span>${ends[1]}</span></th><th scope="col"><span class="wf-hide">Difference</span></th></tr></thead>
    <tbody>${rows.map((r) => `<tr class="${r.total ? "wf-total" : ""}${r.strong ? " wf-strong" : ""}"><th scope="row">${esc(r.label)}</th>
      <td class="wf-track${r.text ? " wf-plain" : ""}" aria-hidden="true">${r.text ? "" : r.value ? `<i class="${r.value > 0 ? "pos" : "neg"} ${kind}" style="width:${Math.max(1.5, (50 * Math.abs(r.value)) / max)}%"></i>` : ""}</td>
      <td class="wf-val">${r.text ?? signed(r.value)}</td></tr>`).join("")}</tbody></table>`;
}
function featureTable(c, model, i) {
  const feats = c.features.map((f) => ({ f, v: c[model].shap[f][i], x: c.feature_values[f][i] }))
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 8);
  const maxAbs = Math.max(...feats.map((d) => Math.abs(d.v)), 1e-9);
  const groupOf = Object.fromEntries(Object.entries(c.feature_groups).flatMap(([g, fs]) => fs.map((f) => [f, g])));
  return `<table class="feat"><thead><tr><th scope="col">Input</th><th scope="col">Value</th><th scope="col">Effect on the score</th></tr></thead><tbody>
    ${feats.map((d) => {
      const [label, fmt] = FEATURE[d.f] ?? [d.f, (v) => v.toFixed(2)];
      return `<tr><td>${esc(label)}<span class="grp">${esc(GROUP_LABEL[groupOf[d.f]] ?? "")}</span></td><td class="num">${esc(fmt(d.x))}</td>
        <td class="eff"><span class="mini ${d.v < 0 ? "neg" : "pos"}"><i style="width:${(46 * Math.abs(d.v)) / maxAbs}px"></i></span><span class="num">${signed(d.v)}</span></td></tr>`;
    }).join("")}</tbody></table>`;
}

// ctx: { c, model, i (the selected candidate or null), b (the restaurant compared with, or null),
//        options: [[index, text]] for the comparison }; handlers: { pick(index | null) }
export function renderWhy(root, ctx, handlers) {
  const { c, model, i } = ctx;
  if (!root.dataset.built) {
    root.dataset.built = "1";
    root.innerHTML = `
      <p class="why-empty">Select a restaurant to see what shaped its model score.</p>
      <div class="why-body" hidden>
        <p class="why-intro"></p>
        <div class="why-cols">
          <section class="why-col" aria-label="This restaurant's model score">
            <p class="why-name"></p>
            <p class="why-summary" id="why-one-sum"></p>
            <div class="why-chart" id="why-one"></div>
          </section>
          <section class="why-col" id="why-two-col" aria-label="Comparison with another restaurant">
            <div class="why-compare-bar"><label class="field inline" for="why-compare-select"><span>Compare with</span><select id="why-compare-select"></select></label>
              <button type="button" class="why-btn" id="why-compare-clear">Clear</button></div>
            <p class="why-summary" id="why-two-sum"></p>
            <div class="why-chart" id="why-two"></div>
          </section>
        </div>
        <p class="why-scope">These charts explain model scores before your adjustments, not the final ranking.</p>
        <details class="why-more why-tech"><summary>Single inputs with the largest effect on this score</summary><div class="why-tech-body"></div></details>
      </div>`;
    root.querySelector("#why-compare-clear").addEventListener("click", () => { handlers.pick(null); root.querySelector("#why-compare-select").focus({ preventScroll: true }); });
    root.querySelector("#why-compare-select").addEventListener("change", (e) => handlers.pick(e.target.value === "" ? null : +e.target.value));
  }
  const $ = (sel) => root.querySelector(sel);
  $(".why-empty").hidden = i != null;
  $(".why-body").hidden = i == null;
  if (i == null) return;

  const F = modelFactors(c, model, i), unit = SCORE_UNIT[model], name = c.candidates.name[i], modelName = MODEL_NAMES[model];
  $(".why-name").textContent = name;
  $("#why-one-sum").textContent = summarize(F);
  for (const sel of [".why-intro", "#why-one", "#why-two-col", ".why-scope", ".why-tech"]) $(sel).hidden = !F.ok;
  if (!F.ok) return;

  // Left: the whole model score as a sum that can be followed down the rows: the typical candidate's score, then
  // every factor group as a bar from one zero line (right adds, left takes away), then the result. The largest
  // factors are in bold.
  const lead = new Set(F.ranked.slice(0, 3).filter((g) => Math.abs(g.value) >= NEAR_ZERO).map((g) => g.key));
  $("#why-one").innerHTML = `<p class="why-unit">${esc(modelName)} score, in ${esc(unit)}</p>`
    + diverging([{ label: "Typical candidate (start)", value: F.base, text: plain(F.base), total: true },
      ...F.ranked.map((g) => ({ label: g.label, value: g.value, strong: lead.has(g.key) })),
      { label: "Model score (the sum)", value: F.score, text: `= ${plain(F.score)}`, total: true }], "model", ["← Lowers the score", "Raises the score →"])
    + `<p class="why-note why-callout">Read down: start from the typical candidate’s score, add each factor (a bar to the right adds, a bar to the left takes away), and the sum is this restaurant’s score.${roundNote([F.base, ...F.groups.map((g) => g.value)], F.score)}</p>`;

  // Right: the same factors against a second restaurant's, as differences; nothing is drawn until one is chosen
  const b = ctx.b != null && ctx.b !== i ? ctx.b : null, select = $("#why-compare-select");
  select.innerHTML = `<option value="">Choose a restaurant…</option>` + ctx.options.map(([k, text]) => `<option value="${k}">${esc(text)}</option>`).join("");
  select.value = b ?? "";
  $("#why-compare-clear").hidden = b == null;
  const G = b == null ? null : modelFactors(c, model, b), near = similarGap(c, model);   // near: what "similar" means here
  if (!G) { $("#why-two-sum").textContent = "Choose a second restaurant to see how the model’s scores differ."; $("#why-two-sum").classList.add("hint"); $("#why-two").innerHTML = ""; }
  else if (!G.ok) { $("#why-two-sum").textContent = "No explanation data is available for that restaurant."; $("#why-two-sum").classList.add("hint"); $("#why-two").innerHTML = ""; }
  else {
    const other = c.candidates.name[b], byKey = Object.fromEntries(G.groups.map((g) => [g.key, g.value]));
    $("#why-two-sum").classList.remove("hint");
    $("#why-two-sum").textContent = compareSentence(F, G, name, other, unit, near);
    // the rows keep the order of the chart on the left, so a factor sits on the same line in both
    const rows = F.ranked.map((g) => ({ label: g.label, value: g.value - byKey[g.key] }));
    $("#why-two").innerHTML = `<p class="why-unit">Difference in ${esc(modelName)} score, in ${esc(unit)}: ${esc(name)} minus ${esc(other)}</p>`
      + diverging([...rows, { label: "Model score difference", value: F.score - G.score, total: true }], "model", [`← Higher for ${esc(other)}`, `Higher for ${esc(name)} →`])
      + `<p class="why-note why-callout">${esc(name)} ${plain(F.score)}, ${esc(other)} ${plain(G.score)}. Both start from the same typical candidate, so the rows add up to the difference between the two scores.${roundNote(rows.map((r) => r.value), F.score - G.score)}</p>`;
  }

  // What a bar is, said once above both charts; the single inputs behind the groups open below them
  $(".why-intro").textContent = `Each bar is a SHAP value: the part of a restaurant’s ${modelName} score attributed to one group of the model’s inputs. `
    + `The model has ${c.features.length} inputs, added up into ${F.groups.length} groups. “Typical candidate” is the model’s average score over a fixed background sample of 200 candidates; `
    + `that starting point plus all the bars gives the restaurant’s score.`;
  $(".why-tech-body").innerHTML = `<p>The ${Math.min(8, c.features.length)} of the model’s ${c.features.length} inputs that move ${esc(name)}’s score most, in ${esc(unit)}.</p>${featureTable(c, model, i)}`;
}
