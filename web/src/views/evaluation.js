// Calibration (V7): do predicted probabilities match actual visits? One reliability diagram per ranker, on the same
// log axes: the calibrated curve in the ranker's colour (solid line, filled points), the raw one in a neutral grey
// (thin line, hollow points), the diagonal of perfect agreement dashed, 95% Wilson intervals thin and light.
// Everything comes from evaluation.json (src/23_calibration.py through src/24_export_evaluation.py); nothing is
// computed here but layout, two counts for the headline, and what the reader points at.
// A bin with no visits has an observed rate of 0, which a log axis cannot hold: it is drawn in a row marked "0"
// under the axis, never moved to a small positive number. The axes start a decade below the smallest value drawn.
// The method, the numbers and the count of predictions per bin are in one "Method & data" disclosure.
import * as d3 from "d3";
import { MODEL_FILL, MODEL_KEY, esc } from "./common.js";

const NAMES = ["Logistic regression", "LightGBM", "LambdaRank"];
const pctTick = (v) => { const p = v * 100; return `${p >= 1 ? d3.format(",")(p) : +p.toPrecision(1)}%`; };
const pct2 = (v) => (v === 0 ? "0%" : `${+(v * 100).toPrecision(2)}%`);         // two significant figures at any size
const skill = (v) => (Math.abs(v) < 1e-9 ? "0" : Math.abs(v) < 1e-3 ? v.toExponential(1) : v.toFixed(4));
const WHEN = { before: "before calibration", after: "after calibration" };

export function createEvaluation(root, ev) {
  const cal = ev.calibration, D = cal.deployed, names = NAMES.filter((m) => D[m]);
  const base = cal.base_rate_predictor.base_rate;
  const series = (m) => [["before", D[m].reliability_before], ["after", D[m].reliability]].filter(([, s]) => s);
  // what the headline rests on: is the mean prediction closer to the base rate after calibration, and do some bins'
  // intervals still miss the diagonal?
  const closer = names.filter((m) => D[m].before).every((m) => Math.abs(D[m].after.mean_p - base) < Math.abs(D[m].before.mean_p - base));
  const off = Object.fromEntries(names.map((m) => { const bins = D[m].reliability.filter((b) => b.positives > 0);
    return [m, [bins.filter((b) => b.mean_p < b.ci[0] || b.mean_p > b.ci[1]).length, bins.length]]; }));
  const someOff = names.some((m) => off[m][0] > 0);
  // both axes start a decade below the smallest prediction or rate drawn (zero rates aside)
  const drawn = names.flatMap((m) => series(m).flatMap(([, s]) => s.flatMap((b) => (b.positives > 0 ? [b.mean_p, b.observed, b.ci[0]] : [b.mean_p]))));
  const lo = 10 ** Math.floor(Math.log10(d3.min(drawn.filter((v) => v > 0))));
  const decades = d3.range(Math.round(Math.log10(lo)), 1).map((k) => 10 ** k);

  root.innerHTML = `
    <p class="ev-finding">${closer && someOff ? "Calibration reduces the overall overprediction, though some probability ranges remain inaccurate."
      : closer ? "Calibration reduces the overall overprediction." : "After calibration the mean prediction is not closer to the observed rate."}</p>
    <ul class="ev-key" aria-label="Legend">
      <li><svg width="18" height="18" aria-hidden="true"><circle cx="9" cy="9" r="5.5" class="ev-pt before"/></svg>Before calibration</li>
      <li><svg width="18" height="18" aria-hidden="true"><circle cx="9" cy="9" r="5.5" class="ev-pt after key"/></svg>After calibration</li>
      <li><svg width="26" height="18" aria-hidden="true"><line x1="1" y1="9" x2="25" y2="9" class="ev-diag"/></svg>Perfect agreement</li>
    </ul>
    <div class="ev-grid" id="ev-cal"></div>
    <p class="ev-read" id="ev-read" aria-live="polite"></p>
    <p class="ev-foot">Both axes use a logarithmic scale.</p>
    <details class="why-more ev-method" id="ev-method"><summary>Method &amp; data</summary><div id="ev-method-body"></div></details>`;
  const $ = (id) => root.querySelector("#" + id);
  const HINT = `<span class="ev-hint">Point at a point or tap it for exact values; with a chart in focus, the arrow keys move along the curve.</span>`;
  let active = null;   // { m, when, k }: the bin in hand

  function describe(a) {
    if (!a) { $("ev-read").innerHTML = HINT; return; }
    const b = (a.when === "before" ? D[a.m].reliability_before : D[a.m].reliability)[a.k];
    $("ev-read").innerHTML = `<b>${esc(a.m)}, ${WHEN[a.when]}.</b> Predicted ${pct2(b.lo_edge)} to ${pct2(b.hi_edge)} (mean ${pct2(b.mean_p)}): `
      + `${b.positives.toLocaleString()} ${b.positives === 1 ? "visit" : "visits"} in ${b.n.toLocaleString()} predictions, an observed rate of ${pct2(b.observed)} `
      + `(95% interval ${pct2(b.ci[0])} to ${pct2(b.ci[1])}).${b.positives ? "" : " No visits in this range: drawn in the row marked 0."}`;
  }
  function setActive(a) {
    active = a;
    describe(a);
    for (const el of root.querySelectorAll(".ev-hit")) el.classList.toggle("on", !!a && el.dataset.m === a.m && el.dataset.when === a.when && +el.dataset.k === a.k);
  }

  // one chart: the same margins, scales and plot size for every ranker
  function chart(node, m, cellW, first, countsOnly) {
    const M = { l: 74, r: 18, t: 32, b: 56 }, ZERO = countsOnly ? 0 : 28;   // room for 14.5px tick labels
    const pw = cellW - M.l - M.r, ph = countsOnly ? 110 : Math.min(pw, 340), H = M.t + ph + ZERO + M.b;
    const x = d3.scaleLog([lo, 1], [M.l, M.l + pw]), y = d3.scaleLog([lo, 1], [M.t + ph, M.t]), y0 = M.t + ph + ZERO - 9;
    const every = pw / decades.length >= 58 ? 1 : 2, labelled = (v) => (decades.length - 1 - decades.indexOf(v)) % every === 0;
    const svg = d3.select(node).append("svg").attr("width", cellW).attr("height", H).attr("class", "ev-svg");
    svg.style("--mc", MODEL_FILL[MODEL_KEY[m]]);
    const grid = svg.append("g").attr("class", "ev-grid-lines");
    grid.selectAll("line.v").data(decades).join("line").attr("x1", x).attr("x2", x).attr("y1", M.t).attr("y2", M.t + ph);
    const xa = svg.append("g").attr("class", "ev-axis");
    xa.selectAll("text").data(decades.filter(labelled)).join("text").attr("x", x).attr("y", M.t + ph + ZERO + 20).attr("text-anchor", "middle").text(pctTick);
    svg.append("text").attr("class", "ev-title").attr("x", M.l + pw / 2).attr("y", H - 8).attr("text-anchor", "middle").text("Predicted visit probability");

    if (countsOnly) {   // how many predictions fall in each range, on the same horizontal axis as the diagram
      const bins = D[m].reliability, top = d3.max(bins, (b) => b.n), hs = d3.scaleLinear([0, top], [M.t + ph, M.t + 6]);
      svg.append("line").attr("class", "ev-base").attr("x1", M.l).attr("x2", M.l + pw).attr("y1", M.t + ph).attr("y2", M.t + ph);
      svg.append("g").selectAll("rect").data(bins).join("rect").attr("class", "ev-count")
        .attr("x", (b, k) => x(k === 0 ? Math.min(b.lo_edge, lo) : b.lo_edge) + 1).attr("width", (b, k) => Math.max(1, x(b.hi_edge) - x(k === 0 ? Math.min(b.lo_edge, lo) : b.lo_edge) - 2))
        .attr("y", (b) => hs(b.n)).attr("height", (b) => M.t + ph - hs(b.n)).append("title").text((b) => `${b.n.toLocaleString()} predictions from ${pct2(b.lo_edge)} to ${pct2(b.hi_edge)}`);
      const ya = svg.append("g").attr("class", "ev-axis");
      ya.append("text").attr("x", M.l - 8).attr("y", M.t + ph).attr("text-anchor", "end").attr("dy", "0.32em").text("0");
      ya.append("text").attr("x", M.l - 8).attr("y", hs(top)).attr("text-anchor", "end").attr("dy", "0.32em").text(d3.format(".2s")(top));
      svg.append("text").attr("class", "ev-name").attr("x", M.l).attr("y", 16).text(m);
      return;
    }

    svg.attr("tabindex", 0).attr("role", "group").attr("aria-label", `${m}: predicted probability against observed visit rate, ${series(m).map(([w]) => WHEN[w]).join(" and ")}. Arrow keys move along the curve.`);
    grid.selectAll("line.h").data(decades).join("line").attr("x1", M.l).attr("x2", M.l + pw).attr("y1", y).attr("y2", y);
    const ya = svg.append("g").attr("class", "ev-axis");
    ya.selectAll("text").data(decades.filter(labelled)).join("text").attr("x", M.l - 8).attr("y", y).attr("text-anchor", "end").attr("dy", "0.32em").text(pctTick);
    // the row for a rate of zero, under the axis, with a mark where the scale is broken
    ya.append("text").attr("x", M.l - 8).attr("y", y0).attr("text-anchor", "end").attr("dy", "0.32em").text("0");
    svg.append("line").attr("class", "ev-base").attr("x1", M.l).attr("x2", M.l + pw).attr("y1", y0).attr("y2", y0);
    svg.append("path").attr("class", "ev-break").attr("d", `M${M.l - 5},${M.t + ph + 4}l10,-3M${M.l - 5},${M.t + ph + 9}l10,-3`);
    if (first) svg.append("text").attr("class", "ev-title").attr("x", 2).attr("y", 14).text("Observed visit rate");
    svg.append("line").attr("class", "ev-diag").attr("x1", x(lo)).attr("y1", y(lo)).attr("x2", x(1)).attr("y2", y(1));

    for (const [when, bins] of series(m)) {
      const g = svg.append("g").attr("class", `ev-series ${when}`);
      const cy = (b) => (b.positives > 0 ? y(b.observed) : y0);
      g.append("path").attr("class", "ev-line").attr("d", d3.line().defined((b) => b.positives > 0).x((b) => x(b.mean_p)).y((b) => y(b.observed))(bins));
      // the interval: for a bin without visits it runs from the zero row up to its upper end
      g.selectAll("line.ev-ci").data(bins).join("line").attr("class", "ev-ci").attr("x1", (b) => x(b.mean_p)).attr("x2", (b) => x(b.mean_p))
        .attr("y1", (b) => (b.ci[0] > 0 ? y(Math.max(lo, b.ci[0])) : y0)).attr("y2", (b) => y(Math.max(lo, b.ci[1])));
      g.selectAll("circle.ev-pt").data(bins).join("circle").attr("class", `ev-pt ${when}`).attr("cx", (b) => x(b.mean_p)).attr("cy", cy).attr("r", when === "after" ? 4.2 : 3.8);
    }
    // what the pointer, a finger or the keyboard takes hold of: a wide target over each point, the calibrated ones on top
    const hits = svg.append("g");
    for (const [when, bins] of series(m)) hits.selectAll(null).data(bins).join("circle").attr("class", "ev-hit").attr("data-m", m).attr("data-when", when).attr("data-k", (_, k) => k)
      .attr("cx", (b) => x(b.mean_p)).attr("cy", (b) => (b.positives > 0 ? y(b.observed) : y0)).attr("r", 11)
      .on("pointerenter", (e, b) => { if (e.pointerType !== "touch") setActive({ m, when, k: bins.indexOf(b) }); })
      .on("click", (e, b) => setActive({ m, when, k: bins.indexOf(b) }));
    svg.on("pointerleave", (e) => { if (e.pointerType !== "touch" && document.activeElement !== svg.node()) setActive(null); })
      .on("focus", () => { if (!active || active.m !== m) setActive({ m, when: "after", k: 0 }); })
      .on("blur", () => setActive(null))
      .on("keydown", (e) => {
        if (!active || active.m !== m) return;
        const own = series(m), cur = own.find(([w]) => w === active.when)[1];
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); setActive({ ...active, k: Math.max(0, Math.min(cur.length - 1, active.k + (e.key === "ArrowRight" ? 1 : -1))) }); }
        else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && own.length > 1) { e.preventDefault(); const to = active.when === "after" ? "before" : "after"; setActive({ m, when: to, k: Math.min(active.k, own.find(([w]) => w === to)[1].length - 1) }); }
        else if (e.key === "Escape") setActive(null);
      });
  }

  function renderCal() {
    const node = $("ev-cal"), W = node.clientWidth;
    if (!W) return false;
    const GAP = 20, cols = W >= 860 ? names.length : 1, cellW = Math.floor(cols === 1 ? Math.min(W, 520) : (W - GAP * (cols - 1)) / cols);
    node.style.setProperty("--cols", cols);
    node.innerHTML = names.map((m) => `<figure class="ev-fig"><figcaption><svg width="30" height="16" aria-hidden="true" style="--mc:${MODEL_FILL[MODEL_KEY[m]]}"><line x1="1" y1="8" x2="29" y2="8" class="ev-line key"/><circle cx="15" cy="8" r="4.2" class="ev-pt after"/></svg><b>${m}</b>
      ${D[m].reliability_before ? "" : `<span>Raw ranking scores are not probabilities.</span>`}</figcaption><div class="ev-plot"></div></figure>`).join("");
    [...node.querySelectorAll(".ev-plot")].forEach((plot, k) => chart(plot, names[k], cellW, cols === 1 || k === 0, false));
    describe(active = null);
    return true;
  }

  function renderMethod() {
    const body = $("ev-method-body"), vc = cal.validation_check, pc = (v) => `${(v * 100).toFixed(2)}%`;
    const raw = names.filter((m) => D[m].before);
    const rowsT = [];
    for (const m of [...names, "Popularity"].filter((x) => D[x])) {
      if (D[m].before) rowsT.push([`${m}, before calibration`, D[m].before]);
      rowsT.push([m === "Popularity" ? "Popularity (log reviews), calibrated" : `${m}, after calibration`, D[m].after]);
    }
    rowsT.push(["Constant base rate", cal.base_rate_predictor]);
    const perDecade = Math.round((cal.edges.length - 1) / Math.log10(cal.edges[cal.edges.length - 1] / cal.edges[0]));
    body.innerHTML = `
      <p><b>Data.</b> Every pair of a test user and one of their candidates at the ${esc(cal.cutoff)} snapshot: ${cal.rows.toLocaleString()} predictions for ${cal.users.toLocaleString()} test users. A visit means the user reviewed the restaurant in the test period: ${cal.positives.toLocaleString()} visits, a base rate of ${pc(base)}.</p>
      <p><b>Before calibration.</b> The models were trained on one visit for every ten sampled candidates, so their raw probabilities describe that mix. Their mean prediction is ${raw.map((m) => `${pc(D[m].before.mean_p)} for ${m}`).join(" and ")}, ${raw.map((m) => `${Math.round(D[m].before.mean_p / base)}`).join(" and ")} times the base rate. LambdaRank’s raw score only orders places and is not a probability, so it has no curve before calibration.</p>
      <p><b>Calibration.</b> Platt scaling refits sigmoid(a·score + b); it never changes the order of a user’s list. The models shown were refit on the training and validation periods, so validation is not held out for them: the fit is made on the test users instead, split into ${cal.folds} folds, each fold calibrated by a fit on the other ${cal.folds - 1}. No prediction is calibrated by a fit that saw its own user.</p>
      <p><b>After calibration.</b> The mean prediction is ${names.map((m) => `${(D[m].after.mean_p * 100).toFixed(3)}% for ${m}`).join(", ")}, against the base rate of ${(base * 100).toFixed(3)}%. That is an average over all predictions; range by range, the 95% interval still misses the diagonal in ${names.map((m) => `${off[m][0]} of ${off[m][1]} bins for ${m}`).join(", ")} (bins with at least one visit).</p>
      <p><b>Bins and intervals.</b> Predictions are grouped into ranges of equal width on a log scale, ${perDecade} per factor of ten from ${pct2(cal.edges[0])} to ${pct2(cal.edges[cal.edges.length - 1])}; anything below ${pct2(cal.edges[0])} goes into the first range. A point sits at the range’s mean prediction and its observed visit rate, with a 95% Wilson interval.</p>
      <p><b>Zeros.</b> A range with no visits has an observed rate of 0, which a log axis cannot show. It is drawn in the row marked 0 under the axis, with its interval running up from there; nothing is replaced by a small positive number. Both axes start at ${pct2(lo)}, below the smallest value drawn.</p>
      <p><b>Measures.</b> Brier skill compares with always predicting the base rate (higher is better). ECE is the expected calibration error over 15 bins of equal size (lower is better). AUC measures ordering alone. A constant base-rate prediction is calibrated on average and has an AUC of 0.5, so ECE has to be read with AUC.</p>
      <div class="scroll-x"><table class="feat lens-acc ev-table">
        <thead><tr><th>Score</th><th>Brier skill</th><th>ECE</th><th>AUC</th><th>Mean predicted</th></tr></thead>
        <tbody>${rowsT.map(([l, s]) => `<tr><td>${esc(l)}</td><td class="mono">${skill(s.brier_skill)}</td><td class="mono">${s.ece_mass15.toExponential(1)}</td>
          <td class="mono">${s.auc.toFixed(3)}</td><td class="mono">${pc(s.mean_p)}</td></tr>`).join("")}</tbody></table></div>
      <p><b>A second check.</b> With models trained on the training period only and Platt fitted on validation (${vc.users.toLocaleString()} users), then scored on test, the fit under-predicts, because the models are older and the period has shifted:
        ${Object.entries(vc.models).map(([m, r]) => `${esc(m)} ECE ${r.after.ece_mass15.toExponential(1)}, Brier skill ${skill(r.after.brier_skill)}, mean predicted ${pc(r.after.mean_p)}`).join("; ")}.</p>
      <h4>Number of predictions <span>in each range, after calibration, on the diagrams’ horizontal axis</span></h4>
      <div class="ev-grid" id="ev-counts"></div>`;
  }
  function renderCounts() {
    const node = $("ev-counts"), W = node?.clientWidth;
    if (!W) return;
    const GAP = 20, cols = W >= 860 ? names.length : 1, cellW = Math.floor(cols === 1 ? Math.min(W, 520) : (W - GAP * (cols - 1)) / cols);
    node.style.setProperty("--cols", cols);
    node.innerHTML = names.map(() => `<div class="ev-plot"></div>`).join("");
    [...node.children].forEach((plot, k) => chart(plot, names[k], cellW, false, true));
  }

  let drawnW = null;
  function render(force = false) {
    const w = root.clientWidth;
    if (!w || (!force && w === drawnW)) return;
    if (renderCal()) drawnW = w;
    renderCounts();
  }
  renderMethod();
  $("ev-method").addEventListener("toggle", renderCounts);   // drawn when opened: a closed disclosure has no width
  render(true);
  return { render };
}
