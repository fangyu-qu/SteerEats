// Overview: three questions with one chart and one sentence each, then the other results as rows of one table
// with their chart one click away. Every number comes from a data file written by a pipeline script (named in
// each panel's footer).
import * as d3 from "d3";
import { BASELINE_FILL, MODEL_FILL, MODEL_KEY, MODEL_SHAPE, esc } from "./common.js";

const f4 = d3.format(".4f"), f3 = d3.format(".3f"), f2 = d3.format(".2f"), f1 = d3.format(".1f"), int = d3.format(",");
const sgn4 = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4);
const pct = (v) => `${Math.round(v * 100)}%`;
// Series marks: colour, shape and (for lines) dash together, so no series depends on colour alone.
const SERIES = [
  { fill: "var(--mist)", shape: d3.symbolCircle, dash: null },
  { fill: "var(--terracotta)", shape: d3.symbolSquare, dash: "7 3" },
  { fill: "var(--grape)", shape: d3.symbolDiamond, dash: "2 3" },
];
const sym = (k, size = 90) => d3.symbol(SERIES[k].shape, size)();
// the three rankers: their own colour and shape wherever they are compared
const SHAPES = { square: d3.symbolSquare, circle: d3.symbolCircle, triangle: d3.symbolTriangle };
const modelSym = (key, size = 95) => d3.symbol(SHAPES[MODEL_SHAPE[key]], size)();
const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function createOverview(root, D) {
  root.innerHTML = `<div class="ov-tip" hidden></div>`;
  const tip = root.querySelector(".ov-tip");
  const showTip = (event, html) => {
    tip.hidden = false; tip.innerHTML = html;
    const box = root.getBoundingClientRect(), w = tip.offsetWidth;
    tip.style.left = `${Math.min(Math.max(8, event.clientX - box.left + 12), box.width - w - 8)}px`;
    tip.style.top = `${event.clientY - box.top + 14}px`;
  };
  const hideTip = () => { tip.hidden = true; };
  const hover = (sel, html) => sel.on("pointermove", (e, d) => showTip(e, html(d))).on("pointerleave", hideTip);

  const panels = [];
  function panel({ id, wide, title, source, build }) {
    const sec = document.createElement("section");
    sec.className = `ov-panel${wide ? " wide" : ""}`;
    sec.setAttribute("aria-labelledby", `ov-${id}`);
    sec.innerHTML = `<h2 id="ov-${id}">${title}</h2><p class="ov-take"></p><div class="ov-controls"></div><div class="ov-chart"></div>
      <details class="ov-details"><summary>Numbers</summary><div class="ov-table"></div></details><p class="ov-source">${source}</p>`;
    grid.appendChild(sec);
    const parts = { take: sec.querySelector(".ov-take"), controls: sec.querySelector(".ov-controls"), chart: sec.querySelector(".ov-chart"), table: sec.querySelector(".ov-table") };
    panels.push(() => build(parts));
  }
  const missing = (p, what) => { p.take.textContent = `${what} has not been run yet.`; p.chart.innerHTML = ""; p.table.innerHTML = ""; };
  const table = (head, rows) => `<table><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r) => `<tr>${r.map((c, j) => (j ? `<td class="num">${c}</td>` : `<th scope="row">${c}</th>`)).join("")}</tr>`).join("")}</tbody></table>`;

  // Rows with a mean and an interval on one shared axis; `zero` draws a reference line at 0.
  // `band` draws the interval as a shaded bar in the row's colour (strongest in the middle, fading to both ends,
  // as in a newspaper poll chart) instead of a whisker. `zoom` starts the axis at the rows that can be compared:
  // a row far below them gets an arrow at the left edge, and the axis title states where the axis starts.
  let bands = 0;
  function intervals(node, rows, { fmt = f4, zero = false, labelW = 168, rowH = 30, domain, axisLabel, tipHtml, band = false, zoom = false }) {
    // on a phone the interval text gives way to the plot; the Numbers table still holds it
    const width = Math.max(node.clientWidth, 300), narrow = width < 480, top = 6;
    const svg = d3.select(node).html("").append("svg").attr("width", width).attr("role", "img");
    // The names, the values and the axis title take the room their text needs at the size the stylesheet
    // gives them, so a change of font size there cannot cut a word off.
    const valText = (r) => (r.lo != null && !narrow ? `${fmt(r.mean)}  [${fmt(r.lo)}, ${fmt(r.hi)}]` : fmt(r.mean));
    const widest = (cls, text) => d3.max(rows, (r) => { const tmp = svg.append("g").attr("class", `ov-row${r.sub ? " sub" : ""}`),
      w = tmp.append("text").attr("class", cls).text(text(r)).node().getComputedTextLength(); tmp.remove(); return w; });
    labelW = Math.min(Math.ceil(widest("ov-label", (r) => r.label)) + 14, Math.round(width * 0.42));
    const valW = Math.ceil(widest("ov-val num", valText)) + 6;
    const title = svg.append("text").attr("class", "ov-axis-title").attr("text-anchor", "middle");
    const bottom = 30 + (parseFloat(getComputedStyle(title.node()).fontSize) || 12);
    const h = top + rows.length * rowH + bottom;
    svg.attr("height", h);
    const top1 = d3.max(rows, (r) => r.mean), step = 0.005;
    rows.forEach((r) => { r.off = zoom && r.hi < top1 / 2; });
    const zoomed = rows.some((r) => r.off), shown = rows.filter((r) => !r.off);
    if (zoomed) domain = [Math.floor(d3.min(shown, (r) => r.lo) / step - 0.4) * step, Math.ceil(d3.max(shown, (r) => r.hi) / step) * step];
    const lo = domain ? domain[0] : Math.min(zero ? 0 : Infinity, d3.min(rows, (r) => r.lo ?? r.mean));
    const hi = domain ? domain[1] : Math.max(zero ? 0 : -Infinity, d3.max(rows, (r) => r.hi ?? r.mean));
    const pad = (hi - lo) * 0.08 || 0.001;
    const x = d3.scaleLinear().domain(domain ?? [zero ? lo - pad : Math.max(0, lo - pad), hi + pad]).range([labelW + (zoomed ? 34 : 0), width - valW - 18]);
    if (!zoomed) x.nice(5);
    const tv = zoomed ? d3.range(x.domain()[0], x.domain()[1] + step / 2, narrow ? step * 2 : step).map((v) => +v.toFixed(4)) : null;
    if (zoomed) axisLabel += ` (axis starts at ${d3.format(".3f")(x.domain()[0])})`;
    svg.attr("aria-label", axisLabel);
    svg.append("g").attr("class", "ov-grid").selectAll("line").data(tv ?? x.ticks(5)).join("line")
      .attr("x1", x).attr("x2", x).attr("y1", top).attr("y2", h - bottom);
    svg.append("g").attr("class", "ov-axis").attr("transform", `translate(0,${h - bottom})`).call(d3.axisBottom(x).ticks(narrow ? 3 : 5).tickValues(tv).tickSize(4).tickFormat(d3.format(zoomed ? ".3f" : "~g")));
    // the axis title: centred under the plot, moved in from an edge it would cross, shortened if it cannot fit
    const short = axisLabel.split(":")[0].split(",")[0];
    title.text(narrow ? short : axisLabel);
    if (title.node().getComputedTextLength() > width) title.text(short);
    const tw = title.node().getComputedTextLength(), mid = (x.range()[0] + x.range()[1]) / 2;
    title.attr("x", Math.max(tw / 2, Math.min(width - tw / 2, mid))).attr("y", h - 7);
    if (zero) svg.append("line").attr("class", "ov-zero").attr("x1", x(0)).attr("x2", x(0)).attr("y1", top).attr("y2", h - bottom);
    const g = svg.selectAll("g.ov-row").data(rows).join("g").attr("class", (r) => `ov-row${r.sub ? " sub" : ""}`)
      .attr("transform", (_, i) => `translate(0,${top + i * rowH + rowH / 2})`);
    g.append("rect").attr("class", "ov-hit").attr("x", 0).attr("y", -rowH / 2).attr("width", width).attr("height", rowH);
    g.append("text").attr("class", "ov-label").attr("x", labelW - 12).attr("text-anchor", "end").attr("dy", "0.35em").text((r) => r.label);
    const ci = g.filter((r) => r.lo != null && !r.off);
    if (band) {
      const id = `band-${bands++}`, defs = svg.append("defs");
      rows.forEach((r, k) => { const lg = defs.append("linearGradient").attr("id", `${id}-${k}`);
        [[0, 0.24], [0.5, 0.66], [1, 0.24]].forEach(([o, a]) => lg.append("stop").attr("offset", o).attr("stop-color", r.fill).attr("stop-opacity", a)); });
      ci.append("rect").attr("x", (r) => x(r.lo)).attr("width", (r) => Math.max(2, x(r.hi) - x(r.lo))).attr("y", -6).attr("height", 12).attr("rx", 4).attr("fill", (r) => `url(#${id}-${rows.indexOf(r)})`);
    } else {
      ci.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.lo)).attr("x2", (r) => x(r.hi)).style("stroke", (r) => r.fill ?? null).style("stroke-width", (r) => (r.fill ? 2.5 : null));
      ci.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.lo)).attr("x2", (r) => x(r.lo)).attr("y1", -4).attr("y2", 4);
      ci.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.hi)).attr("x2", (r) => x(r.hi)).attr("y1", -4).attr("y2", 4);
    }
    // far below the axis: an arrow at the left edge, the mark beside it, the value in its usual place on the right
    g.filter((r) => r.off).append("path").attr("class", "ov-offarrow").attr("d", `M${labelW + 12},0h-12m0,0l5,-4m-5,4l5,4`);
    g.append("path").attr("class", "ov-mark").attr("transform", (r) => `translate(${r.off ? labelW + 22 : x(r.mean)},0)`)
      .attr("d", (r) => (r.shape ? d3.symbol(r.shape, 95)() : r.model ? modelSym(r.model) : sym(r.series ?? 0, r.sub ? 45 : 95))).attr("fill", (r) => r.fill ?? (r.plain ? "var(--panel)" : SERIES[r.series ?? 0].fill));
    g.append("text").attr("class", "ov-val num").attr("x", width).attr("text-anchor", "end").attr("dy", "0.35em")
      .text(valText);
    if (tipHtml) hover(g, tipHtml);
  }

  root.insertAdjacentHTML("beforeend", `<header class="ov-head"><h1>What the evaluation shows</h1><dl class="ov-facts"></dl></header><div class="ov-grid-wrap"></div>`);
  const grid = root.querySelector(".ov-grid-wrap");
  const o = D.overview, ev = D.evaluation;
  if (o) {
    // the facts behind every chart, one small tile each: the number, what it counts, and how it is used
    root.querySelector(".ov-facts").innerHTML = [
      [int(ev.users), "test users", "liked an restaurant in the test period"],
      [int(o.sample_users), "sampled users", "training, validation and test"],
      [int(o.restaurants), "restaurants", `New York City, ${o.cuisine_groups} cuisine groups`],
      [`${d3.format(".1f")(o.reviews / 1e6)} M`, "reviews", `${int(o.reviews)} after cleaning`],
      [o.test_start, "snapshot", "nothing later is used to train"]]
      .map(([v, k, note]) => `<div class="ov-fact"><dt>${k}</dt><dd class="num">${v}</dd><dd class="note">${note}</dd></div>`).join("");
  }

  // 1. Rankers
  panel({
    id: "rank", title: "Which ranker finds the places people go next?", source: "src/15_train_two_stage.py, src/14_evaluate.py",
    build(p) {
      const R = ev.rankers, names = ["Popularity", "Recent popularity", "Item-CF", "Logistic regression", "LightGBM", "LightGBM LambdaRank (supplementary)"];
      const short = (n) => n.replace("LightGBM LambdaRank (supplementary)", "LambdaRank");
      const row = (n) => ({ label: short(n), ...R[n].ndcg, fill: MODEL_KEY[n] ? MODEL_FILL[MODEL_KEY[n]] : BASELINE_FILL[n],
        shape: d3.symbolCircle });   // every row is named on the left, so the mark shape carries nothing here
      const groups = [{ name: "Baselines", rows: names.filter((n) => R[n] && !MODEL_KEY[n]).map(row) }, { name: "Learned rankers", rows: names.filter((n) => R[n] && MODEL_KEY[n]).map(row) }];
      const rows = groups.flatMap((g) => g.rows);
      const best = rows.reduce((a, b) => (b.mean > a.mean ? b : a));
      const by = Object.fromEntries(rows.map((r) => [r.label, r.mean]));
      const stage1 = R["Stage 1 candidates"]?.recall.mean;
      p.take.innerHTML = `<b>${best.label}</b> ranks best (NDCG@10 <span class="num">${f4(best.mean)}</span>). Logistic regression (<span class="num">${f4(by["Logistic regression"])}</span>) and LightGBM (<span class="num">${f4(by.LightGBM)}</span>) are similar, and all three learned rankers beat item-CF (<span class="num">${f4(by["Item-CF"])}</span>).` +
        (stage1 ? ` Stage 1 brings <span class="num">${pct(stage1)}</span> of later visits into the 200 candidates, which caps every re-ranker.` : "");
      intervals(p.chart, rows, { axisLabel: "NDCG@10 on test, 95% interval", labelW: 150, band: true, zoom: true,
        tipHtml: (r) => `<b>${r.label}</b><br>NDCG@10 ${f4(r.mean)} [${f4(r.lo)}, ${f4(r.hi)}]` });
      const sw = (r) => `<li><svg viewBox="-9 -9 18 18" aria-hidden="true"><path d="${d3.symbol(r.shape, 110)()}" fill="${r.fill}" stroke="var(--ink)" stroke-width="1.2"/></svg>${esc(r.label)}</li>`;
      p.controls.innerHTML = groups.map((g) => `<ul class="ov-legend"><li class="ov-legend-h">${g.name}</li>${g.rows.map(sw).join("")}</ul>`).join("")
        + `<p class="ov-chart-note">Shaded bars span the 95% bootstrap interval over ${int(ev.users)} test users; the mark is the mean. </p>`;
      const nd = ev.paired.filter((q) => q.metric === "ndcg");
      p.table.innerHTML = table(["Ranker", "NDCG@10", "95% interval", "Recall@10"], names.filter((n) => R[n]).map((n) => [short(n), f4(R[n].ndcg.mean), `${f4(R[n].ndcg.lo)} to ${f4(R[n].ndcg.hi)}`, f4(R[n].recall.mean)])) +
        table(["Paired difference in NDCG@10", "Mean", "95% interval", "Wilcoxon p"], nd.map((q) => [`${short(q.model_a)} − ${short(q.model_b)}`, sgn4(q.mean_diff), `${sgn4(q.lo)} to ${sgn4(q.hi)}`, q.wilcoxon_p < 0.001 ? "< 0.001" : q.wilcoxon_p.toFixed(3)]));
    },
  });

  // 2. RQ1 trade-off
  panel({
    id: "rq1", title: "How much relevance do closer recommendations cost?", source: "src/20_rq1_sweep.py, src/25_export_tradeoff.py",
    build(p) {
      const mk = D.tradeoff?.settings?.lmr ? "lmr" : "lgbm", who = { lmr: "LambdaRank", lgbm: "LightGBM" }[mk];
      const S = D.tradeoff?.settings?.[mk];
      if (!S) return missing(p, "The steering sweep");
      const base = S.find((s) => s.w_h === 1 && !s.alpha && !s.beta && !s.lam);
      const sorted = [...S].sort((a, b) => a.distance_km - b.distance_km || b.ndcg - a.ndcg);
      // the frontier: going from the closest setting outwards, each setting that beats every closer one,
      // which makes it the best relevance at or below its distance
      const frontier = []; let bestN = -Infinity;
      for (const s of sorted) if (s.ndcg > bestN) { frontier.push(s); bestN = s.ndcg; }
      const half = frontier.filter((s) => s.distance_km <= base.distance_km / 2).pop();
      p.take.innerHTML = `Without steering, ${who}'s recommended restaurants are <span class="num">${f1(base.distance_km)} km</span> away on average.` +
        (half ? ` One setting brings that distance down to <span class="num">${f1(half.distance_km)} km</span> while retaining <span class="num">${pct(half.ndcg / base.ndcg)}</span> of the baseline NDCG@10.` : "");
      const width = Math.max(p.chart.clientWidth, 320), narrow = width < 480, h = 344, m = { l: 56, r: 16, t: 40, b: 50 };
      const x = d3.scaleLinear().domain([0, d3.max(S, (s) => s.distance_km)]).nice().range([m.l, width - m.r]);
      const y = d3.scaleLinear().domain([0, d3.max(S, (s) => s.ndcg)]).nice().range([h - m.b, m.t]);
      const svg = d3.select(p.chart).html("").append("svg").attr("width", width).attr("height", h).attr("role", "img")
        .attr("aria-label", `NDCG@10 against average distance for ${S.length} steering settings`);
      svg.append("g").attr("class", "ov-grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y).attr("y2", y);
      svg.append("g").attr("class", "ov-axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(6).tickSize(4));
      svg.append("g").attr("class", "ov-axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(4));
      svg.append("text").attr("class", "ov-axis-title").attr("x", (m.l + width - m.r) / 2).attr("y", h - 6).attr("text-anchor", "middle")
        .text(narrow ? "Average distance (km)" : "Average distance of recommended restaurants (km)");
      svg.append("path").attr("class", "ov-frontier").attr("d", d3.line().x((s) => x(s.distance_km)).y((s) => y(s.ndcg)).curve(d3.curveStepAfter)(frontier));
      const on = new Set(frontier);
      // shade: how many cuisine groups the top 10 holds, one hue from light to dark
      const seq = d3.interpolateLab(token("--seq-lo"), token("--seq-hi")), ext = d3.extent(S, (s) => s.cuisines), shade = d3.scaleSequential(seq).domain(ext);
      const dots = svg.selectAll("circle.ov-dot").data(S.filter((s) => s !== base)).join("circle").attr("class", (s) => `ov-dot${on.has(s) ? " on" : ""}`)
        .attr("cx", (s) => x(s.distance_km)).attr("cy", (s) => y(s.ndcg)).attr("r", (s) => (on.has(s) ? 5 : 3.6)).style("fill", (s) => shade(s.cuisines));
      const tip = (s) => `History ${s.w_h} · distance ${s.alpha} · popularity ${s.beta} · variety ${f2(s.lam / (D.tradeoff.lam_max || 1))}<br>NDCG@10 ${f4(s.ndcg)} · ${f1(s.distance_km)} km · ${f1(s.cuisines)} cuisines`;
      hover(dots, tip);
      // no steering: a solid red circle, larger than the settings and named beside it, so it is not told by colour alone
      const ref = svg.append("circle").datum(base).attr("class", "ov-ref").attr("cx", x(base.distance_km)).attr("cy", y(base.ndcg)).attr("r", 7.5);
      hover(ref, (s) => `<b>No steering</b><br>${tip(s)}`);
      // the two settings the sentence above compares, named in the plot: lines of text above the point, a short
      // leader down to it, kept inside the plot. On a phone only the names fit; the numbers are in the sentence.
      const callout = (s, lines, right = width - m.r) => {
        const px = x(s.distance_km), py = y(s.ndcg), r = s === base ? 7.5 : 5, n = lines.length;
        const t = svg.append("text").attr("class", "ov-callout").attr("text-anchor", "middle");
        lines.forEach((line, k) => t.append("tspan").attr("class", k ? null : "head").attr("x", 0).attr("y", py - r - 12 - (n - 1 - k) * 17).text(line));
        const half_w = d3.max(t.selectAll("tspan").nodes(), (node) => node.getComputedTextLength()) / 2;
        if (right - (m.l + 8) < 2 * half_w) return t.remove();   // no room without covering a point: the sentence above has it
        const cx = Math.max(m.l + 8 + half_w, Math.min(right - half_w, px));
        t.selectAll("tspan").attr("x", cx);
        svg.insert("line", "text.ov-callout").attr("class", "ov-leader").attr("x1", px).attr("x2", px).attr("y1", py - r - 2).attr("y2", py - r - 8);
      };
      callout(base, narrow ? ["No steering"] : ["No steering", `${f1(base.distance_km)} km · NDCG ${f3(base.ndcg)}`]);
      // its text stops short of the next rise of the line
      const next = frontier[frontier.indexOf(half) + 1];
      if (half) callout(half, narrow ? ["Closer option"] : ["Closer option", `${f1(half.distance_km)} km · NDCG ${f3(half.ndcg)}`, `${pct(half.ndcg / base.ndcg)} of baseline NDCG`], next ? x(next.distance_km) - 12 : undefined);
      p.controls.innerHTML = `<p class="ov-chart-note">Each dot is one of ${S.length} tested combinations of four controls (<b>history</b>, <b>distance</b>, <b>popularity</b>, <b>variety</b>), averaged across ${int(D.tradeoff.users)} test users. The neighborhood boost is not part of this test.</p>
        <ul class="ov-legend"><li class="ov-ramp-row">Average number of cuisines in each user's top 10 <span class="ov-ramp">${f1(ext[0])}<i style="background:linear-gradient(90deg, ${d3.range(0, 1.01, 0.1).map(seq).join(", ")})"></i>${f1(ext[1])}</span></li>
        <li><svg viewBox="-9 -9 18 18" aria-hidden="true"><circle r="6.5" class="ov-ref"/></svg>No steering</li>
        <li><svg viewBox="-9 -9 18 18" aria-hidden="true"><circle r="5.5" fill="none" stroke="var(--ink)" stroke-width="1.6"/></svg>Outlined dots: best relevance at or below this distance</li></ul>
        <p class="ov-chart-note">NDCG@10 measures how well the ranking matches restaurants users later liked; higher is better. The step line shows the best result among the tested settings for each distance limit.</p>`;
      p.table.innerHTML = table(["Frontier setting (history, distance, popularity, variety)", "km", "NDCG@10", "Change [95% interval]", "Cuisines"],
        frontier.map((s) => [`${s.w_h}, ${s.alpha}, ${s.beta}, ${f2(s.lam / (D.tradeoff.lam_max || 1))}`, f2(s.distance_km), f4(s.ndcg), s === base ? "reference" : `${sgn4(s.d_ndcg)} [${sgn4(s.lo)}, ${sgn4(s.hi)}]`, f1(s.cuisines)]));
    },
  });

  // 3. RQ2 groups
  const rq2 = { model: "LambdaRank" };
  panel({
    id: "rq2", wide: true, title: "Does protecting the least satisfied member reduce recommendation quality?", source: "src/group.py, src/33_rq2_groups.py",
    build(p) {
      const G = D.rq2;
      if (!G) return missing(p, "The group evaluation");
      const main = G.summary.filter((r) => r.model === rq2.model && r.how === "minmax" && r.tau === G.tau);
      const pick = (kind, st) => main.find((r) => r.kind === kind && r.strategy === st);
      const dif = (kind, st, m) => G.differences.find((r) => r.model === rq2.model && r.kind === kind && r.strategy === st && r.metric === m);
      // The sentence follows the model chosen below: random groups, Least misery against Average. The differences,
      // their intervals and the relative change stay in the Numbers table, not in the sentence (Yila, 2026-11-26).
      const avg = pick("random", "average"), lm = pick("random", "least_misery");
      const wSat = dif("random", "average_without_misery", "min_sat"), wN = dif("random", "average_without_misery", "ndcg_mean");
      const same = [wSat, wN].every((q) => q.diff === 0 && q.lo === 0 && q.hi === 0);
      p.take.innerHTML = `<span class="ov-para">Each row is one group rule. Each kind of group has ${int(G.groups_per_kind)} simulated groups of three test users; the list is the top 10 of the places any member could be shown. A point is the mean over the groups, and the shaded bar is the 95% bootstrap interval of that mean (groups resampled 2,000 times).</span><span class="ov-para">With ${esc(rq2.model)}, in random groups of three, ranking by the least satisfied member (Least misery) instead of the average raises that member's satisfaction from <span class="num">${f3(avg.min_sat)}</span> to <span class="num">${f3(lm.min_sat)}</span>, while member NDCG@10 falls from <span class="num">${f4(avg.ndcg_mean)}</span> to <span class="num">${f4(lm.ndcg_mean)}</span>.` +
        (same ? ` Average without misery gives the same results as Average here.` : ` Average without misery lands close to Average.`) + `</span>`;
      // How a group's list is ranked, as formulas, in a highlighted box in the card's top right corner (src/group.py:
      // min-max satisfaction per member over the pool, then one of three rules, then the 10 best).
      const formula = { average: "group score = <b>mean</b> of the members’ satisfaction", least_misery: "group score = the <b>lowest</b> member’s satisfaction",
        average_without_misery: `keep a place only if every member’s satisfaction ≥ <b>${G.tau.toFixed(2)}</b>; group score = <b>mean</b>` };
      let aside = p.take.parentNode.querySelector(".ov-aside");
      if (!aside) { aside = document.createElement("aside"); aside.className = "ov-aside"; aside.setAttribute("aria-label", "How a group's list is ranked"); p.take.after(aside); p.take.parentNode.classList.add("has-aside"); }
      aside.innerHTML = `<h3>How a group’s list is ranked</h3>
        <p><span class="ov-f">satisfaction = (score − lowest) ÷ (highest − lowest)</span> for each member, over the places any member could be shown: 0 for that member’s lowest-scoring place, 1 for the highest.</p>
        <ul>${G.strategies.map((st, k) => `<li><svg viewBox="-9 -9 18 18" aria-hidden="true"><circle r="6" fill="${SERIES[k].fill}" stroke="var(--ink)" stroke-width="1.5"/></svg><span><b>${esc(st.label)}</b><span class="ov-f">${formula[st.id] ?? ""}</span></span></li>`).join("")}</ul>
        <p>The list is the 10 places with the highest group score.</p>`;
      p.controls.innerHTML = `<div class="segmented" role="radiogroup" aria-label="Model">${G.models.map((m) =>
        `<button type="button" role="radio" data-m="${esc(m)}" aria-checked="${m === rq2.model}">${esc(m)}</button>`).join("")}</div>
        <ul class="ov-legend">${G.strategies.map((st, k) => `<li><svg viewBox="-9 -9 18 18" aria-hidden="true"><circle r="6" fill="${SERIES[k].fill}" stroke="var(--ink)" stroke-width="1.5"/></svg><span><b>${esc(st.label)}</b></span></li>`).join("")}</ul>`;
      p.controls.querySelectorAll("button[data-m]").forEach((btn) => btn.addEventListener("click", () => { rq2.model = btn.dataset.m; render(); }));
      // One row per rule, so no two rules can sit on top of each other; the two measures side by side on the same
      // rows. The names are written once, on the left, when the two plots are next to each other.
      const measures = [
        { key: "min_sat", fmt: f3, head: "Least satisfied member's satisfaction", axis: "Satisfaction, 0 to 1", zero: false },
        { key: "ndcg_mean", fmt: f4, head: "Member NDCG@10", axis: "NDCG@10", zero: true }];
      const side = p.chart.clientWidth >= 760;
      p.chart.innerHTML = `<div class="ov-pair${side ? " ov-pair-two" : ""}">${measures.map((mm, j) => `<figure><figcaption><b>${mm.head}</b></figcaption><div data-m="${j}"></div></figure>`).join("")}</div>`;
      const rows = G.kinds.flatMap((kind) => [{ head: `${kind[0].toUpperCase() + kind.slice(1)} tastes` }, ...G.strategies.map((st, k) => ({ ...pick(kind, st.id), k, name: st.label, kind }))]);
      measures.forEach((mm, j) => {
        const node = p.chart.querySelector(`[data-m="${j}"]`), width = Math.max(node.clientWidth, 260), named = j === 0 || !side;
        const svg = d3.select(node).append("svg").attr("width", width).attr("role", "img").attr("aria-label", `${mm.head} for three group rules in three kinds of group`);
        const widest = (cls, strings) => { const t = svg.append("text").attr("class", cls), w = d3.max(strings, (str) => t.text(str).node().getComputedTextLength()); t.remove(); return w; };
        const labelW = named ? Math.min(Math.ceil(widest("ov-label", G.strategies.map((st) => st.label))) + 30, Math.round(width * 0.5)) : 10;
        const valW = Math.ceil(widest("ov-val num", ["0.0000"])) + 8;
        const title = svg.append("text").attr("class", "ov-axis-title").attr("text-anchor", "middle");
        const rowH = 30, top = 4, bottom = 30 + (parseFloat(getComputedStyle(title.node()).fontSize) || 12);
        let yy = top;
        rows.forEach((r) => { if (r.head && yy > top) yy += 10; r.y = yy + rowH / 2; yy += rowH; });
        const h = yy + bottom, data = rows.filter((r) => !r.head);
        svg.attr("height", h);
        const x = d3.scaleLinear().domain([mm.zero ? 0 : d3.min(data, (r) => r[`${mm.key}_lo`]), d3.max(data, (r) => r[`${mm.key}_hi`])]).nice(4).range([labelW + 8, width - valW - 14]);
        const start = x.domain()[0];
        svg.append("g").attr("class", "ov-grid").selectAll("line").data(x.ticks(4)).join("line").attr("x1", x).attr("x2", x).attr("y1", top).attr("y2", yy);
        svg.append("g").attr("class", "ov-axis").attr("transform", `translate(0,${yy})`).call(d3.axisBottom(x).ticks(x.range()[1] - x.range()[0] < 160 ? 2 : 4).tickSize(4));
        title.text(mm.axis + (start ? ` (axis starts at ${d3.format(".2f")(start)})` : ""));
        if (title.node().getComputedTextLength() > width) title.text(mm.axis);
        const tw = title.node().getComputedTextLength(), mid = (x.range()[0] + x.range()[1]) / 2;
        title.attr("x", Math.max(tw / 2, Math.min(width - tw / 2, mid))).attr("y", h - 7);
        if (named) svg.selectAll("text.ov-group").data(rows.filter((r) => r.head)).join("text").attr("class", "ov-group").attr("x", 0).attr("y", (r) => r.y).attr("dy", "0.35em").text((r) => r.head);
        const id = `band-${bands++}`, defs = svg.append("defs");
        G.strategies.forEach((_, k) => { const lg = defs.append("linearGradient").attr("id", `${id}-${k}`);
          [[0, 0.24], [0.5, 0.66], [1, 0.24]].forEach(([o, al]) => lg.append("stop").attr("offset", o).attr("stop-color", SERIES[k].fill).attr("stop-opacity", al)); });
        const g = svg.selectAll("g.ov-row").data(data).join("g").attr("class", "ov-row").attr("transform", (r) => `translate(0,${r.y})`);
        g.append("rect").attr("class", "ov-hit").attr("x", 0).attr("y", -rowH / 2).attr("width", width).attr("height", rowH);
        if (named) g.append("text").attr("class", "ov-label").attr("x", labelW - 6).attr("text-anchor", "end").attr("dy", "0.35em").text((r) => r.name);
        g.append("rect").attr("x", (r) => x(r[`${mm.key}_lo`])).attr("width", (r) => Math.max(2, x(r[`${mm.key}_hi`]) - x(r[`${mm.key}_lo`]))).attr("y", -6).attr("height", 12).attr("rx", 4).attr("fill", (r) => `url(#${id}-${r.k})`);
        g.append("circle").attr("class", "ov-mark").attr("cx", (r) => x(r[mm.key])).attr("r", 5.5).attr("fill", (r) => SERIES[r.k].fill);
        g.append("text").attr("class", "ov-val num").attr("x", width).attr("text-anchor", "end").attr("dy", "0.35em").text((r) => mm.fmt(r[mm.key]));
        hover(g, (r) => `<b>${r.name}</b>, ${r.kind} tastes<br>Least satisfied member's satisfaction ${f3(r.min_sat)} [${f3(r.min_sat_lo)}, ${f3(r.min_sat_hi)}]<br>Member NDCG@10 ${f4(r.ndcg_mean)} [${f4(r.ndcg_mean_lo)}, ${f4(r.ndcg_mean_hi)}]<br>Longest trip ${f1(r.max_km)} km`);
      });
      p.table.innerHTML = table(["Tastes", "Rule", "Least satisfied member's satisfaction", "Satisfaction gap (most minus least satisfied)", "Longest trip (km)", "Member NDCG@10", "Own list NDCG@10"],
        G.kinds.flatMap((kind) => G.strategies.map((s) => { const r = pick(kind, s.id); return [kind, s.label, f3(r.min_sat), f3(r.sat_gap), f2(r.max_km), f4(r.ndcg_mean), f4(r.own_ndcg_mean)]; }))) +
        table(["Tastes", "Rule, against Average", "Change in least satisfied member's satisfaction [95% interval]", "Change in member NDCG@10 [95% interval]", "Relative change in NDCG@10"],
          G.kinds.flatMap((kind) => G.strategies.slice(1).map((st) => { const x = dif(kind, st.id, "min_sat"), n = dif(kind, st.id, "ndcg_mean"), r = pick(kind, st.id), r0 = pick(kind, "average");
            return [kind, st.label, `${d3.format("+.3f")(x.diff)} [${d3.format("+.3f")(x.lo)}, ${d3.format("+.3f")(x.hi)}]`, `${sgn4(n.diff)} [${sgn4(n.lo)}, ${sgn4(n.hi)}]`, d3.format("+.0%")(r.ndcg_mean / r0.ndcg_mean - 1)]; })));
    },
  });

  function render() { hideTip(); panels.forEach((b) => b()); }
  return { render };
}
