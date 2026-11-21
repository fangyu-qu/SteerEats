// Score ledger (V3): why one restaurant scores where it does, on the normalized scale the steering uses.
// Model part: SHAP feature groups (interventional, shared background) from the typical candidate to the model
// score, with the history weight shrinking the collaborative group. Steering part: distance and popularity
// penalties, then the variety step as a rank move. SHAP explains raw model scores; dividing by the per-user
// score range puts them on the same axis as the penalties, so the bars add up exactly to the steered score.
import * as d3 from "d3";
import { esc, hueVar, truncate } from "./common.js";

const GROUP_LABEL = {
  Collaborative: "Your history (similar places)", Popularity: "Popularity and recency", Rating: "Rating",
  Location: "Location", Taste: "Taste match", Attributes: "Amenities",
};
const yesNo = (v) => (v >= 0.5 ? "yes" : "no");
const FEATURE = {
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
  return { i, R, base, groups, s0, dist, pop, s2: s0 + dist + pop, ranks: { default: ranks.default[i], s2: ranks.popularity[i], final: ranks.diversity[i] } };
}

function steeredAt(c, model, theta, i) {
  const m = c[model], n = m.raw.length;
  let zmin = Infinity, zmax = -Infinity;
  for (let k = 0; k < n; k++) { const z = m.noncollab[k] + theta.w_h * m.collab[k]; if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
  const s0 = (m.noncollab[i] + theta.w_h * m.collab[i] - zmin) / (zmax - zmin || 1);
  return s0 - theta.alpha * c.distance_n[i] - theta.beta * c.popularity_n[i];
}

export function renderLedger(root, ctx) {
  const { c, model, theta, ranks, i } = ctx;
  const L = ledgerData(c, model, theta, ranks, i);
  const n = c.cuisine.length;
  const tenth = [...Array(n).keys()].find((k) => ranks.popularity[k] === 10);
  const cutoff = tenth != null ? steeredAt(c, model, theta, tenth) : null;

  const rows = [{ kind: "start", label: "Typical candidate", value: L.base }];
  [...L.groups].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).forEach((g) => rows.push({ kind: "model", ...g }));
  rows.push({ kind: "subtotal", label: "Model score", value: L.s0 });
  rows.push({ kind: "control", label: "Distance penalty", value: L.dist, off: theta.alpha === 0 });
  rows.push({ kind: "control", label: "Popularity penalty", value: L.pop, off: theta.beta === 0 });
  rows.push({ kind: "total", label: "Steered score", value: L.s2 });

  let run = 0;
  for (const r of rows) {
    if (r.kind === "start" || r.kind === "subtotal" || r.kind === "total") { r.x0 = 0; r.x1 = r.value; run = r.value; }
    else { r.x0 = run; r.x1 = run + r.value; run = r.x1; }
  }
  const lo = Math.min(0, ...rows.map((r) => Math.min(r.x0, r.x1)), cutoff ?? 0);
  const hi = Math.max(1, ...rows.map((r) => Math.max(r.x0, r.x1)), cutoff ?? 0);
  const width = Math.max(root.clientWidth, 420), labelW = 188, valueW = 62, rowH = 24, top = 8;
  const height = top + rows.length * rowH + 30;
  const x = d3.scaleLinear([lo, hi], [labelW, width - valueW]).nice();

  const svg = d3.select(root).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
    .attr("role", "img").attr("aria-label", `Score ledger for ${c.candidates.name[i]}`);
  svg.selectAll("g.axis").data([0]).join("g").attr("class", "axis").attr("transform", `translate(0,${top + rows.length * rowH + 4})`)
    .call(d3.axisBottom(x).ticks(5).tickSize(-rows.length * rowH - 4).tickFormat(d3.format(".1f")))
    .call((g) => g.select(".domain").remove());
  svg.selectAll("line.zero").data([0]).join("line").attr("class", "zero").attr("x1", x(0)).attr("x2", x(0))
    .attr("y1", top - 4).attr("y2", top + rows.length * rowH);
  const cut = svg.selectAll("g.cutline").data(cutoff == null ? [] : [cutoff]).join((e) => {
    const g = e.append("g").attr("class", "cutline"); g.append("line"); g.append("text"); return g; });
  cut.select("line").attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", top + (rows.length - 3) * rowH + 2).attr("y2", top + rows.length * rowH);
  cut.select("text").attr("x", (d) => x(d) - 6).attr("text-anchor", "end").attr("dy", "0.35em")
    .attr("y", top + (rows.length - 2) * rowH + rowH / 2).text("score of #10");

  const g = svg.selectAll("g.row").data(rows, (r) => r.label).join((e) => {
    const r = e.append("g").attr("class", "row");
    r.append("text").attr("class", "lab").attr("x", 0).attr("dy", "0.35em");
    r.append("rect").attr("class", "ghost");
    r.append("rect").attr("class", "bar");
    r.append("line").attr("class", "conn");
    r.append("text").attr("class", "val").attr("dy", "0.35em").attr("text-anchor", "end");
    return r;
  });
  g.attr("transform", (_, k) => `translate(0,${top + k * rowH + rowH / 2})`).attr("data-kind", (r) => r.kind);
  g.select(".lab").text((r) => r.label + (r.scaled ? ` × ${theta.w_h.toFixed(2)}` : ""));
  const color = (r) => (r.kind === "model" ? hueVar("model") : r.kind === "control" ? hueVar("control") : "var(--ink-2)");
  g.select(".bar").attr("x", (r) => x(Math.min(r.x0, r.x1))).attr("width", (r) => Math.max(1, Math.abs(x(r.x1) - x(r.x0))))
    .attr("y", -7).attr("height", 14).attr("rx", 2).attr("fill", color).attr("opacity", (r) => (r.off ? 0.25 : 1));
  g.select(".ghost").attr("x", (r) => (r.scaled ? x(Math.min(r.x0, r.x0 + r.full)) : 0))
    .attr("width", (r) => (r.scaled ? Math.abs(x(r.x0 + r.full) - x(r.x0)) : 0)).attr("y", -7).attr("height", 14).attr("rx", 2);
  g.select(".conn").attr("x1", (r) => x(r.x1)).attr("x2", (r) => x(r.x1)).attr("y1", 7).attr("y2", rowH - 7)
    .attr("visibility", (r, k) => (k < rows.length - 1 && r.kind !== "total" ? "visible" : "hidden"));
  g.select(".val").attr("x", width).text((r) => (r.kind === "model" || r.kind === "control" ? (r.value >= 0 ? "+" : "−") + Math.abs(r.value).toFixed(2) : r.value.toFixed(2)));

  // Variety and verdict lines (HTML under the chart)
  const name = esc(c.candidates.name[i]);
  const final = L.ranks.final, s2r = L.ranks.s2, def = L.ranks.default;
  const variety = theta.lam === 0 ? "Variety is off, so the steered score decides the order."
    : final === s2r ? `The variety step kept it at #${final}.`
    : `The variety step moved it from #${s2r} to #${final} (${final < s2r ? "up" : "down"} ${Math.abs(final - s2r)}), to spread cuisines and neighborhoods.`;
  const verdict = final <= 10 ? `${name} is #${final} in your list (the model alone put it at #${def}).`
    : `${name} is #${final}, outside your top 10 (the model alone put it at #${def}).`;
  ctx.notes.innerHTML = `<p><b>${verdict}</b> ${variety}</p>
    <p class="muted">Scale: 0 = this user's lowest-scoring candidate, 1 = highest, under the current history weight. Group bars are SHAP values from ${esc(c.models[model])} (raw score ${c[model].raw[i].toFixed(2)} log-odds) divided by the score range.${L.groups.some((x) => x.scaled) ? " The pale bar shows the history group at full weight." : ""}</p>`;

  // Feature detail: the largest individual SHAP values for this place
  const feats = c.features.map((f) => ({ f, v: c[model].shap[f][i], x: c.feature_values[f][i] }))
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 8);
  const maxAbs = Math.max(...feats.map((d) => Math.abs(d.v)), 1e-9);
  const groupOf = Object.fromEntries(Object.entries(c.feature_groups).flatMap(([g, fs]) => fs.map((f) => [f, g])));
  ctx.detail.innerHTML = `<table class="feat"><caption>Strongest single features (log-odds)</caption>
    <thead><tr><th scope="col">Feature</th><th scope="col">Value</th><th scope="col">Effect</th></tr></thead><tbody>
    ${feats.map((d) => {
      const [label, fmt] = FEATURE[d.f] ?? [d.f, (v) => v.toFixed(2)];
      const w = (46 * Math.abs(d.v)) / maxAbs;
      return `<tr><td>${esc(label)}<span class="grp">${esc(GROUP_LABEL[groupOf[d.f]] ?? "")}</span></td><td class="num">${esc(fmt(d.x))}</td>
        <td class="eff"><span class="mini ${d.v < 0 ? "neg" : "pos"}"><i style="width:${w}px"></i></span><span class="num">${d.v >= 0 ? "+" : "−"}${Math.abs(d.v).toFixed(2)}</span></td></tr>`;
    }).join("")}</tbody></table>`;
}

// Contrast (why A over B): the same ledger rows as differences A − B; bars right favour A.
export function renderContrast(root, ctx) {
  const { c, model, theta, ranks, a, b } = ctx;
  if (b == null || a === b) { d3.select(root).selectAll("*").remove(); ctx.caption.textContent = ""; return; }
  const A = ledgerData(c, model, theta, ranks, a), B = ledgerData(c, model, theta, ranks, b);
  const rows = A.groups.map((g, k) => ({ kind: "model", label: g.label, value: g.value - B.groups[k].value }))
    .sort((p, q) => Math.abs(q.value) - Math.abs(p.value));
  rows.push({ kind: "control", label: "Distance penalty", value: A.dist - B.dist });
  rows.push({ kind: "control", label: "Popularity penalty", value: A.pop - B.pop });
  rows.push({ kind: "total", label: "Steered score gap", value: A.s2 - B.s2 });
  const width = Math.max(root.clientWidth, 360), labelW = 188, valueW = 56, rowH = 24, top = 22;
  const height = top + rows.length * rowH + 8;
  const m = Math.max(0.05, ...rows.map((r) => Math.abs(r.value)));
  const x = d3.scaleLinear([-m, m], [labelW, width - valueW]);
  const svg = d3.select(root).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
    .attr("role", "img").attr("aria-label", `Why ${c.candidates.name[a]} scores differently from ${c.candidates.name[b]}`);
  svg.selectAll("text.side").data([["end", `◀ ${truncate(c.candidates.name[b], 18)}`, x(0) - 6], ["start", `${truncate(c.candidates.name[a], 18)} ▶`, x(0) + 6]])
    .join("text").attr("class", "side").attr("text-anchor", (d) => d[0]).attr("x", (d) => d[2]).attr("y", 10).text((d) => d[1]);
  svg.selectAll("line.zero").data([0]).join("line").attr("class", "zero").attr("x1", x(0)).attr("x2", x(0)).attr("y1", top - 4).attr("y2", height - 4);
  const g = svg.selectAll("g.row").data(rows, (r) => r.label).join((e) => {
    const r = e.append("g").attr("class", "row");
    r.append("text").attr("class", "lab").attr("x", 0).attr("dy", "0.35em");
    r.append("rect").attr("class", "bar");
    r.append("text").attr("class", "val").attr("dy", "0.35em").attr("text-anchor", "end");
    return r;
  });
  g.attr("transform", (_, k) => `translate(0,${top + k * rowH + rowH / 2})`).attr("data-kind", (r) => r.kind);
  g.select(".lab").text((r) => r.label);
  g.select(".bar").attr("x", (r) => x(Math.min(0, r.value))).attr("width", (r) => Math.max(1, Math.abs(x(r.value) - x(0))))
    .attr("y", -7).attr("height", 14).attr("rx", 2)
    .attr("fill", (r) => (r.kind === "model" ? hueVar("model") : r.kind === "control" ? hueVar("control") : "var(--ink-2)"));
  g.select(".val").attr("x", width).text((r) => (r.value >= 0 ? "+" : "−") + Math.abs(r.value).toFixed(2));
  const lead = [...rows.slice(0, -1)].sort((p, q) => Math.abs(q.value) - Math.abs(p.value))[0];
  const winner = lead.value >= 0 ? c.candidates.name[a] : c.candidates.name[b];
  ctx.caption.innerHTML = `Bars point toward the place they favour. ${esc(c.candidates.name[a])} (#${A.ranks.final}) vs ${esc(c.candidates.name[b])} (${B.ranks.final <= 10 ? "#" + B.ranks.final : "now #" + B.ranks.final}): the biggest difference is <b>${esc(lead.label.toLowerCase())}</b>, which favours ${esc(winner)}.`;
}
