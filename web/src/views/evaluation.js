// Evaluation panel (V7): is the learned ranker actually better, and are its probabilities usable?
// Accuracy with 95% user-bootstrap intervals, paired differences, stability over time and sampling seeds,
// and reliability diagrams before and after Platt scaling. Everything comes from evaluation.json, written by
// src/24_export_evaluation.py from the evaluation scripts' outputs; nothing is computed here but layout.
import * as d3 from "d3";
import { esc } from "./common.js";

const ORDER = ["LightGBM LambdaRank (supplementary)", "LightGBM", "Logistic regression", "Item-CF", "Recent popularity", "Popularity"];
const NAME = { "LightGBM LambdaRank (supplementary)": "LambdaRank", "Logistic regression": "Logistic regression",
  LightGBM: "LightGBM", "Item-CF": "Item-CF", "Recent popularity": "Recent popularity", Popularity: "Popularity" };
const SHORT = { "LightGBM LambdaRank (supplementary)": "LambdaRank", "Logistic regression": "LR", LightGBM: "LightGBM", "Item-CF": "Item-CF" };
const KIND = (m) => (m.startsWith("LightGBM LambdaRank") ? "supp" : m === "LightGBM" || m === "Logistic regression" ? "learned" : "base");
const METRIC = { ndcg: "NDCG@10", recall: "Recall@10" };
const f4 = d3.format(".4f");
const pctTick = (v) => { const p = v * 100; return `${p >= 1 ? d3.format(",")(p) : +p.toPrecision(1)}%`; };
const skill = (v) => (Math.abs(v) < 1e-9 ? "0" : Math.abs(v) < 1e-3 ? v.toExponential(1) : v.toFixed(4));
const pctf = (v) => `${(v * 100).toFixed(1)}%`;

export function createEvaluation(root, ev) {
  let metric = "ndcg";
  root.innerHTML = `
    <div class="eval-top">
      <p class="lens-headline" id="ev-headline"></p>
      <div class="segmented" role="radiogroup" aria-label="Accuracy metric">
        <button type="button" role="radio" data-metric="ndcg" aria-checked="true">NDCG@10</button>
        <button type="button" role="radio" data-metric="recall" aria-checked="false">Recall@10</button>
      </div>
    </div>
    <div class="lens-row">
      <figure class="lens-fig">
        <figcaption><b>Accuracy on the test period</b><span id="ev-acc-cap"></span></figcaption>
        <div id="ev-acc" class="lens-svg"></div>
      </figure>
      <figure class="lens-fig">
        <figcaption><b>Paired differences</b><span>Same users, model A minus model B, with the 95% bootstrap interval of the mean difference and the Wilcoxon signed-rank p. An interval that crosses zero (hollow dot) means no detectable difference.</span></figcaption>
        <div id="ev-diff" class="lens-svg"></div>
      </figure>
    </div>
    <div class="lens-row">
      <figure class="lens-fig">
        <figcaption><b>Over time</b><span>Validation windows use models trained before December 2019; test periods use the final models, all from the August 2020 snapshot. Accuracy falls after March 2020 for every ranker.</span></figcaption>
        <div id="ev-time" class="lens-svg"></div>
      </figure>
      <figure class="lens-fig">
        <figcaption><b>Across sampling seeds</b><span>Test NDCG@10 when the training negatives and the validation users for model selection are resampled. Filled: the run used everywhere else (seed 42).</span></figcaption>
        <div id="ev-seeds" class="lens-svg"></div>
      </figure>
    </div>
    <figure class="lens-fig">
      <figcaption><b>Are the probabilities right?</b><span id="ev-cal-cap"></span></figcaption>
      <ul class="key" id="ev-cal-key"></ul>
      <div id="ev-cal" class="lens-svg"></div>
      <table class="feat lens-acc" id="ev-cal-table"></table>
    </figure>
    <div class="lens-tip" id="ev-tip" role="tooltip" hidden></div>`;
  const $ = (id) => root.querySelector("#" + id);
  const tip = $("ev-tip");
  const showTip = (event, html) => {
    const box = root.getBoundingClientRect();
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, x = event.clientX - box.left, y = event.clientY - box.top;
    tip.style.left = `${Math.min(Math.max(4, x + 14), box.width - w - 4)}px`;
    tip.style.top = `${y + 16}px`;
  };
  const hideTip = () => { tip.hidden = true; };
  root.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => {
    metric = b.dataset.metric;
    root.querySelectorAll(".segmented button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    render(true);
  }));
  const cal = ev.calibration;

  function headline() {
    const p = (a, b) => ev.paired.find((r) => r.model_a === a && r.model_b === b && r.metric === "ndcg");
    const lrcf = p("Logistic regression", "Item-CF"), gbcf = p("LightGBM", "Item-CF"), gblr = p("LightGBM", "Logistic regression");
    const rklr = p("LightGBM LambdaRank (supplementary)", "Logistic regression");
    const d = cal.deployed;
    $("ev-headline").innerHTML = `Both learned rankers beat item-to-item collaborative filtering on NDCG@10 `
      + `(+${pctf(lrcf.relative)} and +${pctf(gbcf.relative)}, intervals above zero), but <b>logistic regression and LightGBM cannot be told apart</b>: `
      + `difference ${f4(gblr.mean_diff)}, 95% interval ${f4(gblr.lo)} to ${f4(gblr.hi)}. `
      + (rklr ? `LambdaRank, trained to order the list, is ahead of logistic regression by ${f4(rklr.mean_diff)} (${f4(rklr.lo)} to ${f4(rklr.hi)}). ` : "")
      + `Raw probabilities overstate the chance of a visit `
      + `${Math.round(d.LightGBM.before.mean_p / d.LightGBM.before.base_rate)}–${Math.round(d["Logistic regression"].before.mean_p / d["Logistic regression"].before.base_rate)}-fold; `
      + `after Platt scaling the mean prediction matches the observed rate and the ranking is unchanged.`;
  }

  // --- Accuracy dot plot ------------------------------------------------------------------------
  function renderAcc() {
    const node = $("ev-acc");
    // narrow (phone): short names, the value under the name instead of a value column
    const width = Math.max(node.clientWidth, 280), narrow = width < 520;
    const LABEL = narrow ? 118 : 178, VAL = narrow ? 12 : 178, ROW = narrow ? 42 : 30, TOP = 6, AX = 24;
    const models = ORDER.filter((m) => ev.rankers[m]?.[metric]);
    const R = (m) => ev.rankers[m][metric];
    const x = d3.scaleLinear([0, d3.max(models, (m) => R(m).hi) * 1.05], [LABEL, width - VAL]).nice();
    const height = TOP + models.length * ROW + AX;
    const ceil = ev.rankers["Stage 1 candidates"]?.recall;
    $("ev-acc-cap").textContent = `Mean over ${ev.users.toLocaleString()} test users with the 95% user-bootstrap interval. Baselines rank the whole catalog; the learned rankers re-rank 200 candidates`
      + (ceil ? `, which hold ${pctf(ceil.mean)} of the places users went on to review (the recall ceiling).` : ".");
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
      .attr("role", "img").attr("aria-label", `${METRIC[metric]} with 95% intervals for each ranker`);
    svg.selectAll("*").remove();
    svg.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${TOP + models.length * ROW})`)
      .call(d3.axisBottom(x).ticks(4).tickSize(3).tickFormat(d3.format(".2f")));
    models.forEach((m, k) => {
      const cy = TOP + (k + 0.5) * ROW - (narrow ? 6 : 0), r = R(m), kind = KIND(m), g = svg.append("g").attr("class", `acc ${kind}`);
      g.append("line").attr("class", "lens-guide").attr("x1", LABEL).attr("x2", width - VAL).attr("y1", cy).attr("y2", cy);
      g.append("text").attr("class", "lab").attr("x", 0).attr("y", cy).attr("dy", "0.35em").text(narrow ? SHORT[m] ?? NAME[m] : NAME[m]);
      g.append("line").attr("class", "ci").attr("x1", x(r.lo)).attr("x2", x(r.hi)).attr("y1", cy).attr("y2", cy);
      for (const v of [r.lo, r.hi]) g.append("line").attr("class", "ci").attr("x1", x(v)).attr("x2", x(v)).attr("y1", cy - 4).attr("y2", cy + 4);
      g.append("circle").attr("cx", x(r.mean)).attr("cy", cy).attr("r", 5).attr("class", "pt");
      g.append("text").attr("class", narrow ? "sub mono-sub" : "val").attr("x", narrow ? 0 : width - VAL + 10).attr("y", narrow ? cy + 15 : cy).attr("dy", "0.35em")
        .text(`${f4(r.mean)} [${f4(r.lo)}, ${f4(r.hi)}]`);
      g.append("rect").attr("class", "hit").attr("x", 0).attr("y", cy - ROW / 2).attr("width", width).attr("height", ROW)
        .on("pointermove", (e) => showTip(e, `<b>${esc(NAME[m])}</b><br>${METRIC[metric]} <span class="mono">${f4(r.mean)}</span><br>95% interval <span class="mono">${f4(r.lo)} to ${f4(r.hi)}</span>`
          + (kind === "supp" ? "<br>Listwise objective, added after the proposal's pointwise comparison. The diner app ranks with it." : "")))
        .on("pointerleave", hideTip);
    });
  }

  // --- Paired differences ------------------------------------------------------------------------
  function renderDiff() {
    const node = $("ev-diff");
    const width = Math.max(node.clientWidth, 280), narrow = width < 520;
    const LABEL = narrow ? 150 : 178, VAL = narrow ? 12 : 150, ROW = narrow ? 56 : 40, TOP = 6, AX = 24;
    const rows = ev.paired.filter((r) => r.metric === metric);
    const m = d3.max(rows, (r) => Math.max(Math.abs(r.lo), Math.abs(r.hi))) * 1.1;
    const x = d3.scaleLinear([-m, m], [LABEL, width - VAL]).nice();
    const height = TOP + rows.length * ROW + AX;
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
      .attr("role", "img").attr("aria-label", `Paired ${METRIC[metric]} differences`);
    svg.selectAll("*").remove();
    svg.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${TOP + rows.length * ROW})`)
      .call(d3.axisBottom(x).ticks(narrow ? 2 : 3).tickSize(3).tickFormat((v) => (v > 0 ? "+" : v < 0 ? "−" : "") + d3.format(".3f")(Math.abs(v))));
    svg.append("line").attr("class", "zero").attr("x1", x(0)).attr("x2", x(0)).attr("y1", TOP).attr("y2", TOP + rows.length * ROW);
    rows.forEach((r, k) => {
      const cy = TOP + (k + 0.5) * ROW - 4, g = svg.append("g").attr("class", "acc learned");
      const sig = r.lo > 0 || r.hi < 0;
      g.append("text").attr("class", "lab").attr("x", 0).attr("y", cy).attr("dy", "0.35em").text(`${SHORT[r.model_a] ?? r.model_a} − ${SHORT[r.model_b] ?? r.model_b}`);
      g.append("text").attr("class", "sub").attr("x", 0).attr("y", cy + 15).attr("dy", "0.35em")
        .text(`better for ${r.users_a_better.toLocaleString()} vs ${r.users_b_better.toLocaleString()} users`);
      g.append("line").attr("class", "ci").attr("x1", x(r.lo)).attr("x2", x(r.hi)).attr("y1", cy).attr("y2", cy);
      for (const v of [r.lo, r.hi]) g.append("line").attr("class", "ci").attr("x1", x(v)).attr("x2", x(v)).attr("y1", cy - 4).attr("y2", cy + 4);
      g.append("circle").attr("cx", x(r.mean_diff)).attr("cy", cy).attr("r", 5).attr("class", sig ? "pt" : "pt hollow");
      const stat = `${r.mean_diff >= 0 ? "+" : "−"}${pctf(Math.abs(r.relative))} · p ${r.wilcoxon_p < 0.001 ? "< 0.001" : r.wilcoxon_p.toFixed(3)}`;
      if (narrow) g.append("text").attr("class", "sub").attr("x", 0).attr("y", cy + 30).attr("dy", "0.35em").text(`${stat} · ${sig ? "excludes 0" : "no detectable difference"}`);
      else {
        g.append("text").attr("class", "val").attr("x", width - VAL + 10).attr("y", cy).attr("dy", "0.35em").text(stat);
        g.append("text").attr("class", "sub").attr("x", width - VAL + 10).attr("y", cy + 15).attr("dy", "0.35em").text(sig ? "interval excludes 0" : "no detectable difference");
      }
      g.append("rect").attr("class", "hit").attr("x", 0).attr("y", cy - ROW / 2 + 4).attr("width", width).attr("height", ROW)
        .on("pointermove", (e) => showTip(e, `<b>${esc(NAME[r.model_a] ?? r.model_a)} − ${esc(NAME[r.model_b] ?? r.model_b)}</b><br>`
          + `mean difference <span class="mono">${r.mean_diff >= 0 ? "+" : "−"}${Math.abs(r.mean_diff).toFixed(5)}</span><br>`
          + `95% interval <span class="mono">${r.lo.toFixed(5)} to ${r.hi.toFixed(5)}</span><br>Wilcoxon p <span class="mono">${r.wilcoxon_p.toPrecision(2)}</span>`))
        .on("pointerleave", hideTip);
    });
  }

  // --- Over time ---------------------------------------------------------------------------------
  function renderTime() {
    const node = $("ev-time");
    const width = Math.max(node.clientWidth, 280), narrow = width < 520, M = { l: 46, r: narrow ? 78 : 96, t: 26, b: 26 }, H = 230;
    const P = ev.periods, models = ["LightGBM LambdaRank (supplementary)", "LightGBM", "Logistic regression", "Item-CF"];
    const x = d3.scalePoint(P.map((p) => p.label), [M.l, width - M.r]).padding(0.3);
    const vals = P.flatMap((p) => models.map((m) => p[metric][m]).filter((v) => v != null));
    const y = d3.scaleLinear(d3.extent(vals), [H - M.b, M.t]).nice();
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", H)
      .attr("role", "img").attr("aria-label", `${METRIC[metric]} by period for each ranker`);
    svg.selectAll("*").remove();
    svg.append("g").attr("class", "tk-axis").attr("transform", `translate(${M.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(3).tickFormat(d3.format(".3f")));
    svg.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${H - M.b})`).call(d3.axisBottom(x).tickSize(3).tickFormat((v) => (narrow ? v.split("–")[0] : v)));
    const split = (x(P[1].label) + x(P[2].label)) / 2;
    svg.append("line").attr("class", "lens-guide").attr("x1", split).attr("x2", split).attr("y1", 8).attr("y2", H - M.b);
    svg.append("text").attr("class", "lens-note").attr("x", split - 6).attr("y", 12).attr("text-anchor", "end").text(narrow ? "validation" : "validation · train-only models");
    svg.append("text").attr("class", "lens-note").attr("x", split + 6).attr("y", 12).text(narrow ? "test" : "test · final models");
    const ends = [];
    for (const m of models) {
      const cls = KIND(m) === "supp" ? "supp" : KIND(m) === "learned" ? "learned" : "base";
      for (const part of ["validation", "test"]) {
        const pts = P.filter((p) => p.split === part && p[metric][m] != null).map((p) => [x(p.label), y(p[metric][m]), p]);
        if (!pts.length) continue;
        const g = svg.append("g").attr("class", `series ${cls}`);
        g.append("path").attr("d", d3.line()(pts.map((d) => [d[0], d[1]])));
        g.selectAll("g.m").data(pts).join("g").attr("class", "m").each(function (d) {
          const s = d3.select(this);
          if (m === "Logistic regression") s.append("rect").attr("x", d[0] - 4).attr("y", d[1] - 4).attr("width", 8).attr("height", 8);
          else s.append("circle").attr("cx", d[0]).attr("cy", d[1]).attr("r", 4.2);
        }).on("pointermove", (e, d) => showTip(e, `<b>${esc(NAME[m])}</b> · ${d[2].label} (${d[2].split})<br>${METRIC[metric]} <span class="mono">${f4(d[2][metric][m])}</span> · ${d[2].users.toLocaleString()} users`))
          .on("pointerleave", hideTip);
        if (part === "test") ends.push({ m, y: pts[pts.length - 1][1], x: pts[pts.length - 1][0] });
      }
    }
    ends.sort((a, b) => a.y - b.y);
    for (let k = 1; k < ends.length; k++) ends[k].ly = Math.max(ends[k].y, (ends[k - 1].ly ?? ends[k - 1].y) + 13);
    ends[0].ly = ends[0].y;
    for (const e of ends) svg.append("text").attr("class", "end-lab").attr("x", e.x + 10).attr("y", e.ly).attr("dy", "0.35em").text(SHORT[e.m]);
  }

  // --- Seeds -------------------------------------------------------------------------------------
  function renderSeeds() {
    const node = $("ev-seeds");
    const width = Math.max(node.clientWidth, 280), narrow = width < 520;
    const LABEL = narrow ? 84 : 178, VAL = narrow ? 8 : 120, ROW = narrow ? 42 : 32, TOP = 6, AX = 24;
    const models = ORDER.filter((m) => ev.seeds[m] && KIND(m) !== "base");
    const all = models.flatMap((m) => Object.values(ev.seeds[m]));
    const cf = ev.seeds["Item-CF"] ? Object.values(ev.seeds["Item-CF"])[0] : null;
    const x = d3.scaleLinear(d3.extent([...all, ...(cf ? [cf] : [])]), [LABEL + 8, width - VAL - 8]).nice();
    const height = TOP + models.length * ROW + AX + 8;
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
      .attr("role", "img").attr("aria-label", "Test NDCG@10 across sampling seeds");
    svg.selectAll("*").remove();
    svg.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${TOP + models.length * ROW + 8})`)
      .call(d3.axisBottom(x).ticks(narrow ? 3 : 4).tickSize(3).tickFormat(d3.format(".3f")));
    if (cf) {
      svg.append("line").attr("class", "diag").attr("x1", x(cf)).attr("x2", x(cf)).attr("y1", TOP).attr("y2", TOP + models.length * ROW + 8);
      svg.append("text").attr("class", "lens-note halo").attr("x", x(cf) + 4).attr("y", TOP + 2).attr("dy", "0.7em").text(`Item-CF ${f4(cf)}`);
    }
    models.forEach((m, k) => {
      const cy = TOP + (k + 0.5) * ROW - (narrow ? 6 : 0), v = ev.seeds[m], xs = Object.values(v), g = svg.append("g").attr("class", "acc learned");
      g.append("text").attr("class", "lab").attr("x", 0).attr("y", cy).attr("dy", "0.35em").text(narrow ? SHORT[m] ?? NAME[m] : NAME[m]);
      g.append("line").attr("class", "range").attr("x1", x(d3.min(xs))).attr("x2", x(d3.max(xs))).attr("y1", cy).attr("y2", cy);
      for (const [seed, val] of Object.entries(v)) {
        g.append("circle").attr("cx", x(val)).attr("cy", cy).attr("r", 4.5).attr("class", seed === "42" ? "pt" : "pt hollow")
          .on("pointermove", (e) => showTip(e, `<b>${esc(NAME[m])}</b> · seed ${seed}<br>NDCG@10 <span class="mono">${f4(val)}</span>`)).on("pointerleave", hideTip);
      }
      g.append("text").attr("class", narrow ? "sub" : "val").attr("x", narrow ? 0 : width - VAL + 10).attr("y", narrow ? cy + 15 : cy).attr("dy", "0.35em").text(`range ${f4(d3.max(xs) - d3.min(xs))}`);
    });
  }

  // --- Calibration -------------------------------------------------------------------------------
  function renderCal() {
    const node = $("ev-cal"), D = cal.deployed;
    const names = ["Logistic regression", "LightGBM", "LambdaRank", "Popularity"].filter((m) => D[m]);
    $("ev-cal-cap").textContent = `Every (user, candidate) pair in the test period: ${cal.rows.toLocaleString()} rows, ${cal.positives.toLocaleString()} visits `
      + `(base rate ${(cal.base_rate_predictor.base_rate * 100).toFixed(2)}%). Bins of predicted probability (log scale, four per decade) against the share that `
      + `were visited, with 95% Wilson intervals; on the dashed diagonal a prediction means what it says. The models were trained on one visit per ten sampled `
      + `candidates, so their raw probabilities are far too high. LambdaRank's score is not a probability, so it has no raw series. Platt scaling, cross-fitted over ${cal.folds} user folds, refits sigmoid(a·score + b); it never changes the order of a user's list.`;
    $("ev-cal-key").innerHTML = `<li><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="var(--panel)" stroke="var(--ink-3)" stroke-width="1.5"/></svg>Raw model probability</li>
      <li><svg width="12" height="12"><circle cx="6" cy="6" r="4.5" fill="var(--ink)"/></svg>After Platt scaling</li>
      <li><svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" stroke="var(--ink-3)" stroke-dasharray="3 3"/></svg>Perfect calibration</li>
      <li><i style="background:var(--track)"></i>Rows per bin (after Platt)</li>`;
    // three panels side by side; one per row on a phone
    const width = Math.max(node.clientWidth, 280), GAP = 22, COLS = width < 600 ? 1 : width < 1000 && names.length > 3 ? 2 : names.length, ROWS = Math.ceil(names.length / COLS);
    const cell = Math.min(330, (width - GAP * (COLS - 1)) / COLS), M = { l: 52, r: 12, t: 22, b: 26 }, HIST = 46;
    const ph = cell - 20, PH = ph + HIST + 30;
    const lo = 1e-5;
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", COLS * cell + GAP * (COLS - 1))
      .attr("height", ROWS * PH + (ROWS - 1) * GAP).attr("role", "img").attr("aria-label", "Reliability diagrams before and after Platt scaling");
    svg.selectAll("*").remove();
    names.forEach((m, k) => {
      const ox = (k % COLS) * (cell + GAP), oy = Math.floor(k / COLS) * (PH + GAP), g = svg.append("g").attr("transform", `translate(${ox},${oy})`);
      const x = d3.scaleLog([lo, 1], [M.l, cell - M.r]), y = d3.scaleLog([lo, 1], [ph - M.b, M.t]);
      const cl = (v) => Math.max(lo, v);
      g.append("text").attr("class", "lens-col").attr("x", M.l).attr("y", 12).text(m === "Popularity" ? "Popularity (reference)" : m);
      g.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${ph - M.b})`).call(d3.axisBottom(x).tickValues([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]).tickFormat(pctTick).tickSize(3));
      g.append("g").attr("class", "tk-axis").attr("transform", `translate(${M.l},0)`).call(d3.axisLeft(y).tickValues([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]).tickFormat(pctTick).tickSize(3));
      g.append("line").attr("class", "diag ref").attr("x1", x(lo)).attr("y1", y(lo)).attr("x2", x(1)).attr("y2", y(1));
      const ang = (Math.atan2(y(1) - y(lo), x(1) - x(lo)) * 180) / Math.PI;
      // the in-plot label would cross the models' bins; the key names the diagonal for them
      if (m === "Popularity") g.append("text").attr("class", "tk halo").attr("transform", `translate(${x(2e-4)},${y(2e-4) + 12}) rotate(${ang})`).attr("text-anchor", "middle").text("perfectly calibrated");
      if (k % COLS === 0) {
        g.append("text").attr("class", "tk").attr("transform", `translate(10,${(ph - M.b + M.t) / 2}) rotate(-90)`).attr("text-anchor", "middle").text("share visited");
      }
      g.append("text").attr("class", "tk").attr("x", cell - M.r).attr("y", ph - M.b - 5).attr("text-anchor", "end").text("predicted →");
      const series = [["before", D[m].reliability_before], ["after", D[m].reliability]].filter(([, s]) => s);
      for (const [when, bins] of series) {
        const sg = g.append("g").attr("class", `rel ${when}`);
        sg.append("path").attr("d", d3.line().defined((b) => b.positives > 0).x((b) => x(cl(b.mean_p))).y((b) => y(cl(b.observed)))(bins));
        bins.forEach((b) => {
          const cx = x(cl(b.mean_p));
          if (b.positives > 0) sg.append("line").attr("class", "ci").attr("x1", cx).attr("x2", cx).attr("y1", y(cl(b.ci[0]))).attr("y2", y(cl(b.ci[1])));
          sg.append("circle").attr("cx", cx).attr("cy", y(cl(b.observed))).attr("r", 3.6).attr("class", b.positives ? "pt" : "pt zero-bin")
            .on("pointermove", (e) => showTip(e, `<b>${esc(m)}</b>, ${when === "before" ? "raw" : "after Platt"}<br>predicted ${b.lo_edge.toExponential(0)} to ${b.hi_edge.toExponential(0)} (mean ${b.mean_p.toExponential(2)})<br>`
              + `visited ${b.positives.toLocaleString()} of ${b.n.toLocaleString()} = ${b.observed.toExponential(2)}`
              + (b.positives ? `<br>95% interval ${b.ci[0].toExponential(1)} to ${b.ci[1].toExponential(1)}` : "<br>no visits in this bin (drawn at the floor)")))
            .on("pointerleave", hideTip);
        });
      }
      const bins = D[m].reliability, hy = ph + 4;
      const hmax = d3.max(bins, (b) => b.n);
      const hs = d3.scaleLinear([0, hmax], [0, HIST - 8]);
      const hg = g.append("g").attr("class", "hist");
      bins.forEach((b) => {
        const x0 = x(b.lo_edge) + 1, x1 = x(b.hi_edge) - 1;
        hg.append("rect").attr("x", x0).attr("width", Math.max(1, x1 - x0)).attr("y", hy + HIST - 8 - hs(b.n)).attr("height", hs(b.n)).attr("rx", 1.5)
          .on("pointermove", (e) => showTip(e, `${b.n.toLocaleString()} rows predicted ${b.lo_edge.toExponential(0)} to ${b.hi_edge.toExponential(0)}`)).on("pointerleave", hideTip);
      });
      hg.append("line").attr("class", "zero").attr("x1", M.l).attr("x2", cell - M.r).attr("y1", hy + HIST - 8).attr("y2", hy + HIST - 8);
    });
    const rowsT = [];
    for (const m of names) {
      if (D[m].before) rowsT.push([`${m}, raw probability`, D[m].before]);
      rowsT.push([m === "Popularity" ? "Popularity (log reviews), Platt" : `${m}, after Platt`, D[m].after]);
    }
    rowsT.push(["Constant base rate", cal.base_rate_predictor]);
    const vc = cal.validation_check;
    const best = rowsT.reduce((a, b) => (b[1].ece_mass15 < a[1].ece_mass15 ? b : a));
    const constNote = best[0] === "Constant base rate"
      ? "The constant base rate has the lowest ECE of all and an AUC of 0.5: judged by ECE alone, the useless predictor would win."
      : "A constant base-rate predictor is perfectly calibrated in expectation and has an AUC of 0.5, so ECE must be read with AUC.";
    $("ev-cal-table").innerHTML = `<caption>Probability quality on the test period. Brier skill compares with always predicting the base rate (higher is better);
      ECE uses 15 equal-mass bins (lower is better) and AUC measures ranking alone. ${constNote}</caption>
      <thead><tr><th>Score</th><th>Brier skill</th><th>ECE</th><th>AUC</th><th>Mean predicted</th></tr></thead>
      <tbody>${rowsT.map(([l, s]) => `<tr><td>${esc(l)}</td><td class="mono">${skill(s.brier_skill)}</td><td class="mono">${s.ece_mass15.toExponential(1)}</td>
        <td class="mono">${s.auc.toFixed(3)}</td><td class="mono">${(s.mean_p * 100).toFixed(2)}%</td></tr>`).join("")}</tbody>
      <tfoot><tr><td colspan="5" class="note">Check with the plan's protocol: train-only models, Platt fitted on validation (${vc.users.toLocaleString()} users), evaluated on test. The models are older and the period has shifted, so the validation fit under-predicts:
        ${Object.entries(vc.models).map(([m, r]) => `${esc(m)} ECE ${r.after.ece_mass15.toExponential(1)}, Brier skill ${skill(r.after.brier_skill)}, mean predicted ${(r.after.mean_p * 100).toFixed(2)}%`).join("; ")}.</td></tr></tfoot>`;
  }

  let drawnW = null;
  function render(force = false) {
    const w = root.clientWidth;
    if (!force && w === drawnW) return;
    drawnW = w;
    renderAcc(); renderDiff(); renderTime(); renderSeeds();
    if (!force || !$("ev-cal").firstChild) renderCal();
  }
  headline();
  render(true);
  renderCal();
  return { render };
}
