// Overview: one chart and one sentence per question, each with its numbers one click away.
// Every number comes from a data file written by a pipeline script (named in each panel's footer).
import * as d3 from "d3";
import { esc } from "./common.js";

const f4 = d3.format(".4f"), f3 = d3.format(".3f"), f2 = d3.format(".2f"), f1 = d3.format(".1f"), int = d3.format(",");
const sgn4 = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4);
const pct = (v) => `${Math.round(v * 100)}%`;
// Series marks: colour, shape and (for lines) dash together, so no series depends on colour alone.
const SERIES = [
  { fill: "var(--model)", shape: d3.symbolCircle, dash: null },
  { fill: "var(--control)", shape: d3.symbolSquare, dash: "7 3" },
  { fill: "var(--diversity)", shape: d3.symbolDiamond, dash: "2 3" },
];
const sym = (k, size = 90) => d3.symbol(SERIES[k].shape, size)();

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
  const legend = (items) => `<ul class="ov-legend">${items.map((t, k) =>
    `<li><svg viewBox="-9 -9 18 18" aria-hidden="true"><path d="${sym(k, 110)}" fill="${SERIES[k].fill}" stroke="var(--ink)" stroke-width="1.5"/></svg>${esc(t)}</li>`).join("")}</ul>`;

  // Rows with a mean and an interval on one shared axis; `zero` draws a reference line at 0.
  function intervals(node, rows, { fmt = f4, zero = false, labelW = 168, rowH = 30, domain, axisLabel, tipHtml }) {
    const width = Math.max(node.clientWidth, 320), valW = 150, top = 6, bottom = 34;
    const h = top + rows.length * rowH + bottom;
    const lo = domain ? domain[0] : Math.min(zero ? 0 : Infinity, d3.min(rows, (r) => r.lo ?? r.mean));
    const hi = domain ? domain[1] : Math.max(zero ? 0 : -Infinity, d3.max(rows, (r) => r.hi ?? r.mean));
    const pad = (hi - lo) * 0.08 || 0.001;
    const x = d3.scaleLinear().domain(domain ?? [zero ? lo - pad : Math.max(0, lo - pad), hi + pad]).range([labelW, width - valW - 18]).nice(5);
    const svg = d3.select(node).html("").append("svg").attr("width", width).attr("height", h).attr("role", "img").attr("aria-label", axisLabel);
    svg.append("g").attr("class", "ov-grid").selectAll("line").data(x.ticks(5)).join("line")
      .attr("x1", x).attr("x2", x).attr("y1", top).attr("y2", h - bottom);
    svg.append("g").attr("class", "ov-axis").attr("transform", `translate(0,${h - bottom})`).call(d3.axisBottom(x).ticks(5).tickSize(4).tickFormat(d3.format("~g")));
    svg.append("text").attr("class", "ov-axis-title").attr("x", width - valW).attr("y", h - 4).attr("text-anchor", "end").text(axisLabel);
    if (zero) svg.append("line").attr("class", "ov-zero").attr("x1", x(0)).attr("x2", x(0)).attr("y1", top).attr("y2", h - bottom);
    const g = svg.selectAll("g.ov-row").data(rows).join("g").attr("class", (r) => `ov-row${r.sub ? " sub" : ""}`)
      .attr("transform", (_, i) => `translate(0,${top + i * rowH + rowH / 2})`);
    g.append("rect").attr("class", "ov-hit").attr("x", 0).attr("y", -rowH / 2).attr("width", width).attr("height", rowH);
    g.append("text").attr("class", "ov-label").attr("x", labelW - 12).attr("text-anchor", "end").attr("dy", "0.35em").text((r) => r.label);
    const ci = g.filter((r) => r.lo != null);
    ci.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.lo)).attr("x2", (r) => x(r.hi));
    ci.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.lo)).attr("x2", (r) => x(r.lo)).attr("y1", -4).attr("y2", 4);
    ci.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.hi)).attr("x2", (r) => x(r.hi)).attr("y1", -4).attr("y2", 4);
    g.append("path").attr("class", "ov-mark").attr("transform", (r) => `translate(${x(r.mean)},0)`)
      .attr("d", (r) => sym(r.series ?? 0, r.sub ? 45 : 95)).attr("fill", (r) => (r.plain ? "var(--panel)" : SERIES[r.series ?? 0].fill));
    g.append("text").attr("class", "ov-val num").attr("x", width).attr("text-anchor", "end").attr("dy", "0.35em")
      .text((r) => (r.lo != null ? `${fmt(r.mean)}  [${fmt(r.lo)}, ${fmt(r.hi)}]` : fmt(r.mean)));
    if (tipHtml) hover(g, tipHtml);
  }

  root.insertAdjacentHTML("beforeend", `<header class="ov-head"><h1>What we built, and what the evaluation shows</h1><dl class="ov-facts"></dl></header><div class="ov-grid-wrap"></div>`);
  const grid = root.querySelector(".ov-grid-wrap");
  const o = D.overview, ev = D.evaluation;
  if (o) {
    root.querySelector(".ov-facts").innerHTML = [
      [int(o.restaurants), "New York restaurants"], [int(o.reviews), "reviews, cleaned"], [int(o.sample_users), "sampled users"],
      [int(ev.users), "test users evaluated"], [o.test_start, "snapshot: nothing later is used"]]
      .map(([v, k]) => `<div><dt>${k}</dt><dd class="num">${v}</dd></div>`).join("");
  }

  // 1. Rankers
  panel({
    id: "rank", title: "Which ranker finds the places people go next?", source: "src/15_train_two_stage.py, src/14_evaluate.py",
    build(p) {
      const R = ev.rankers, names = ["Popularity", "Recent popularity", "Item-CF", "Logistic regression", "LightGBM", "LightGBM LambdaRank (supplementary)"];
      const short = (n) => n.replace("LightGBM LambdaRank (supplementary)", "LambdaRank");
      const rows = names.filter((n) => R[n]).map((n) => ({ label: short(n), ...R[n].ndcg, series: n.startsWith("Light") || n.startsWith("Logistic") ? 1 : 0, plain: !(n.startsWith("Light") || n.startsWith("Logistic")) }));
      const best = rows.reduce((a, b) => (b.mean > a.mean ? b : a));
      const by = Object.fromEntries(rows.map((r) => [r.label, r.mean]));
      const stage1 = R["Stage 1 candidates"]?.recall.mean;
      p.take.innerHTML = `<b>${best.label}</b> ranks best (NDCG@10 <span class="num">${f4(best.mean)}</span>). Logistic regression (<span class="num">${f4(by["Logistic regression"])}</span>) and LightGBM (<span class="num">${f4(by.LightGBM)}</span>) are level, and all three learned rankers beat item-CF (<span class="num">${f4(by["Item-CF"])}</span>).` +
        (stage1 ? ` Stage 1 brings <span class="num">${pct(stage1)}</span> of later visits into the 200 candidates, which caps every re-ranker.` : "");
      intervals(p.chart, rows, { axisLabel: "NDCG@10 on test, 95% interval", labelW: 150,
        tipHtml: (r) => `<b>${r.label}</b><br>NDCG@10 ${f4(r.mean)} [${f4(r.lo)}, ${f4(r.hi)}]` });
      p.controls.innerHTML = `<ul class="ov-legend"><li><svg viewBox="-9 -9 18 18" aria-hidden="true"><path d="${sym(1, 110)}" fill="var(--control)" stroke="var(--ink)" stroke-width="1.5"/></svg>Learned ranker</li><li><svg viewBox="-9 -9 18 18" aria-hidden="true"><path d="${sym(0, 110)}" fill="var(--panel)" stroke="var(--ink)" stroke-width="1.5"/></svg>Baseline</li></ul>`;
      const nd = ev.paired.filter((q) => q.metric === "ndcg");
      p.table.innerHTML = table(["Ranker", "NDCG@10", "95% interval", "Recall@10"], names.filter((n) => R[n]).map((n) => [short(n), f4(R[n].ndcg.mean), `${f4(R[n].ndcg.lo)} to ${f4(R[n].ndcg.hi)}`, f4(R[n].recall.mean)])) +
        table(["Paired difference in NDCG@10", "Mean", "95% interval", "Wilcoxon p"], nd.map((q) => [`${short(q.model_a)} − ${short(q.model_b)}`, sgn4(q.mean_diff), `${sgn4(q.lo)} to ${sgn4(q.hi)}`, q.wilcoxon_p < 0.001 ? "< 0.001" : q.wilcoxon_p.toFixed(3)]));
    },
  });

  // 2. RQ1 trade-off
  panel({
    id: "rq1", title: "What does steering cost?", source: "src/20_rq1_sweep.py, src/25_export_tradeoff.py",
    build(p) {
      const S = D.tradeoff?.settings?.lgbm;
      if (!S) return missing(p, "The steering sweep");
      const base = S.find((s) => s.w_h === 1 && !s.alpha && !s.beta && !s.lam);
      const sorted = [...S].sort((a, b) => a.distance_km - b.distance_km || b.ndcg - a.ndcg);
      const frontier = []; let bestN = -Infinity;
      for (const s of sorted) if (s.ndcg > bestN) { frontier.push(s); bestN = s.ndcg; }
      const half = frontier.filter((s) => s.distance_km <= base.distance_km / 2).pop();
      p.take.innerHTML = `With no steering, LightGBM's top 10 is <span class="num">${f1(base.distance_km)} km</span> away on average (NDCG@10 <span class="num">${f4(base.ndcg)}</span>).` +
        (half ? ` The best setting that at least halves that distance (<span class="num">${f1(half.distance_km)} km</span>) keeps <span class="num">${pct(half.ndcg / base.ndcg)}</span> of the relevance. Settings below the line give up relevance without getting closer.` : "");
      const width = Math.max(p.chart.clientWidth, 320), h = 300, m = { l: 56, r: 16, t: 10, b: 44 };
      const x = d3.scaleLinear().domain([0, d3.max(S, (s) => s.distance_km)]).nice().range([m.l, width - m.r]);
      const y = d3.scaleLinear().domain([0, d3.max(S, (s) => s.ndcg)]).nice().range([h - m.b, m.t]);
      const svg = d3.select(p.chart).html("").append("svg").attr("width", width).attr("height", h).attr("role", "img")
        .attr("aria-label", "NDCG@10 against average distance for 192 steering settings");
      svg.append("g").attr("class", "ov-grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y).attr("y2", y);
      svg.append("g").attr("class", "ov-axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(6).tickSize(4));
      svg.append("g").attr("class", "ov-axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(4));
      svg.append("text").attr("class", "ov-axis-title").attr("x", width - m.r).attr("y", h - 6).attr("text-anchor", "end").text("Average distance of the top 10 (km)");
      svg.append("text").attr("class", "ov-axis-title").attr("x", m.l + 6).attr("y", m.t + 8).text("NDCG@10");
      svg.append("path").attr("class", "ov-line").attr("d", d3.line().x((s) => x(s.distance_km)).y((s) => y(s.ndcg)).curve(d3.curveStepAfter)(frontier));
      const on = new Set(frontier);
      const dots = svg.selectAll("circle.ov-dot").data(S).join("circle").attr("class", (s) => `ov-dot${on.has(s) ? " on" : ""}`)
        .attr("cx", (s) => x(s.distance_km)).attr("cy", (s) => y(s.ndcg)).attr("r", (s) => (on.has(s) ? 4.5 : 3));
      hover(dots, (s) => `History ${s.w_h} · distance ${s.alpha} · popularity ${s.beta} · variety ${f2(s.lam / (D.tradeoff.lam_max || 1))}<br>NDCG@10 ${f4(s.ndcg)} · ${f1(s.distance_km)} km · ${f1(s.cuisines)} cuisines`);
      svg.append("path").attr("class", "ov-mark").attr("transform", `translate(${x(base.distance_km)},${y(base.ndcg)})`).attr("d", sym(1, 130)).attr("fill", "var(--control)");
      svg.append("text").attr("class", "ov-note").attr("x", x(base.distance_km) + 10).attr("y", y(base.ndcg) - 8).text("No steering");
      p.controls.innerHTML = `<ul class="ov-legend"><li><svg viewBox="-9 -9 18 18" aria-hidden="true"><circle r="5" class="ov-dot on"/></svg>On the frontier</li><li><svg viewBox="-9 -9 18 18" aria-hidden="true"><circle r="3.5" class="ov-dot"/></svg>Other settings (${S.length} in all, ${int(D.tradeoff.users)} users)</li></ul>`;
      p.table.innerHTML = table(["Frontier setting (history, distance, popularity, variety)", "km", "NDCG@10", "Change [95% interval]", "Cuisines"],
        frontier.map((s) => [`${s.w_h}, ${s.alpha}, ${s.beta}, ${f2(s.lam / (D.tradeoff.lam_max || 1))}`, f2(s.distance_km), f4(s.ndcg), s === base ? "reference" : `${sgn4(s.d_ndcg)} [${sgn4(s.lo)}, ${sgn4(s.hi)}]`, f1(s.cuisines)]));
    },
  });

  // 3. RQ2 groups
  const rq2 = { model: "LambdaRank" };
  panel({
    id: "rq2", wide: true, title: "Eating as a group: who gives up what?", source: "src/group.py, src/33_rq2_groups.py",
    build(p) {
      const G = D.rq2;
      if (!G) return missing(p, "The group evaluation");
      const main = G.summary.filter((r) => r.model === rq2.model && r.how === "minmax" && r.tau === G.tau);
      const pick = (kind, st) => main.find((r) => r.kind === kind && r.strategy === st);
      const dif = (kind, st, m) => G.differences.find((r) => r.model === rq2.model && r.kind === kind && r.strategy === st && r.metric === m);
      const a = dif("random", "least_misery", "min_sat"), b = dif("random", "least_misery", "ndcg_mean"), gap = dif("random", "least_misery", "sat_gap");
      const awm = dif("random", "average_without_misery", "ndcg_mean");
      p.take.innerHTML = `In random groups of three, picking by the least happy member instead of the average raises that member's satisfaction by <span class="num">${f3(a.diff)}</span> and narrows the gap between members by <span class="num">${f3(-gap.diff)}</span>, but each member's own NDCG@10 falls by <span class="num">${f4(-b.diff)}</span>.` +
        ` Dropping places below ${G.tau.toFixed(2)} for anyone changes the list little (NDCG@10 ${sgn4(awm.diff)}): the top of an averaged list is rarely a place someone dislikes.`;
      p.controls.innerHTML = `<div class="segmented" role="radiogroup" aria-label="Model">${G.models.map((m) =>
        `<button type="button" role="radio" data-m="${esc(m)}" aria-checked="${m === rq2.model}">${esc(m)}</button>`).join("")}</div>` + legend(G.strategies.map((s) => s.label));
      p.controls.querySelectorAll("button[data-m]").forEach((btn) => btn.addEventListener("click", () => { rq2.model = btn.dataset.m; render(); }));
      p.chart.innerHTML = `<div class="ov-multi">${G.kinds.map((k) => `<figure><figcaption>${k[0].toUpperCase() + k.slice(1)} tastes</figcaption><div data-kind="${k}"></div></figure>`).join("")}</div>`;
      const xs = main.flatMap((r) => [r.min_sat_lo, r.min_sat_hi]), ys = main.flatMap((r) => [r.ndcg_mean_lo, r.ndcg_mean_hi]);
      G.kinds.forEach((kind) => {
        const node = p.chart.querySelector(`[data-kind="${kind}"]`), width = Math.max(node.clientWidth, 220), h = 230, m = { l: 52, r: 12, t: 10, b: 42 };
        const x = d3.scaleLinear().domain(d3.extent(xs)).nice(4).range([m.l, width - m.r]);
        const y = d3.scaleLinear().domain([0, d3.max(ys)]).nice(4).range([h - m.b, m.t]);
        const svg = d3.select(node).append("svg").attr("width", width).attr("height", h).attr("role", "img")
          .attr("aria-label", `${kind} groups: worst-off satisfaction against member NDCG@10 for three strategies`);
        svg.append("g").attr("class", "ov-grid").selectAll("line").data(y.ticks(4)).join("line").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y).attr("y2", y);
        svg.append("g").attr("class", "ov-axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(4).tickSize(4));
        svg.append("g").attr("class", "ov-axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4).tickSize(4));
        svg.append("text").attr("class", "ov-axis-title").attr("x", width - m.r).attr("y", h - 6).attr("text-anchor", "end").text("Worst-off member's satisfaction");
        svg.append("text").attr("class", "ov-axis-title").attr("x", m.l + 6).attr("y", m.t + 8).text("Member NDCG@10");
        // "average" and "average without misery" often coincide: the larger diamond is drawn first so both stay visible
        const pts = G.strategies.map((s, k) => ({ ...pick(kind, s.id), k, name: s.label })).reverse();
        const g = svg.selectAll("g.pt").data(pts).join("g").attr("class", "pt");
        g.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.min_sat_lo)).attr("x2", (r) => x(r.min_sat_hi)).attr("y1", (r) => y(r.ndcg_mean)).attr("y2", (r) => y(r.ndcg_mean));
        g.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.min_sat)).attr("x2", (r) => x(r.min_sat)).attr("y1", (r) => y(r.ndcg_mean_lo)).attr("y2", (r) => y(r.ndcg_mean_hi));
        g.append("path").attr("class", "ov-mark").attr("transform", (r) => `translate(${x(r.min_sat)},${y(r.ndcg_mean)})`).attr("d", (r) => sym(r.k, r.k === 2 ? 300 : 110)).attr("fill", (r) => SERIES[r.k].fill);
        hover(g, (r) => `<b>${r.name}</b>, ${kind} groups<br>Worst-off satisfaction ${f3(r.min_sat)} [${f3(r.min_sat_lo)}, ${f3(r.min_sat_hi)}]<br>Member NDCG@10 ${f4(r.ndcg_mean)} [${f4(r.ndcg_mean_lo)}, ${f4(r.ndcg_mean_hi)}]<br>Longest trip ${f1(r.max_km)} km`);
      });
      p.table.innerHTML = table(["Tastes", "Strategy", "Worst-off satisfaction", "Satisfaction gap", "Longest trip (km)", "Member NDCG@10", "Own list NDCG@10"],
        G.kinds.flatMap((kind) => G.strategies.map((s) => { const r = pick(kind, s.id); return [kind, s.label, f3(r.min_sat), f3(r.sat_gap), f2(r.max_km), f4(r.ndcg_mean), f4(r.own_ndcg_mean)]; })));
    },
  });

  // 4. RQ3 text features
  panel({
    id: "rq3", title: "Do features read from review text help?", source: "src/34_text_sample.py, src/36_llm_ablation.py",
    build(p) {
      const T = D.text;
      if (!T) return missing(p, "The text-feature comparison");
      // one row per model and text feature set: mean over seeds, with the range of the seeds as the bar
      const textSets = T.sets.filter((x) => x.id !== "base");
      const vsBase = T.differences.filter((r) => (r.against ?? "base") === "base");
      const rows = [];
      T.models.forEach((m) => textSets.forEach((x, k) => {
        const d = vsBase.filter((r) => r.model === m && r.set === x.id);
        rows.push({ label: `${m.replace("Logistic regression", "Logistic")} · ${x.id === "bow" ? "bag-of-words" : "LLM aspects"}`,
          mean: d3.mean(d, (r) => r.diff), lo: d3.min(d, (r) => r.diff), hi: d3.max(d, (r) => r.diff), series: x.id === "bow" ? 1 : 0, d, k });
      }));
      const count = (list) => { const c = list.filter((r) => r.lo > 0 || r.hi < 0), up = c.filter((r) => r.lo > 0).length; return { n: list.length, clear: c.length, up, down: c.length - up }; };
      const say = (c) => (c.clear === 0 ? `none of ${c.n} model-and-seed pairs differs from zero` : `${c.clear} of ${c.n} model-and-seed pairs differ from zero (${c.up} better, ${c.down} worse)`);
      const bow = count(vsBase.filter((r) => r.set === "bow")), llm = count(vsBase.filter((r) => r.set === "llm"));
      const head = count(T.differences.filter((r) => r.set === "llm" && r.against === "bow"));
      p.take.innerHTML = `Added to the 30 base features, bag-of-words features from pre-${T.text_cutoff.slice(0, 4)} review text: ${say(bow)}.` +
        (T.llm_pending ? ` <span class="ov-pending">The LLM aspect features are not run yet: they need an API key.</span>`
          : ` LLM aspect scores read from the same text: ${say(llm)}. LLM against bag-of-words directly: ${say(head)}.`);
      p.controls.innerHTML = textSets.length > 1 ? `<ul class="ov-legend">${textSets.map((x) => { const k = x.id === "bow" ? 1 : 0;
        return `<li><svg viewBox="-9 -9 18 18" aria-hidden="true"><path d="${sym(k, 110)}" fill="${SERIES[k].fill}" stroke="var(--ink)" stroke-width="1.5"/></svg>${esc(x.label)}</li>`; }).join("")}</ul>` : "";
      intervals(p.chart, rows, { zero: true, fmt: sgn4, labelW: 190, rowH: 28, axisLabel: `Change in NDCG@10 against base: mean of ${T.seeds.length} seeds, bar = range of the seeds`,
        tipHtml: (r) => `<b>${r.label}</b><br>` + r.d.map((q) => `seed ${q.seed}: ${sgn4(q.diff)} [${sgn4(q.lo)}, ${sgn4(q.hi)}]`).join("<br>") });
      const nd = (m, s) => d3.mean(T.ndcg.filter((r) => r.model === m && r.set === s), (r) => r.ndcg);
      p.table.innerHTML = table(["Model", ...T.sets.map((x) => `NDCG@10, ${x.label}`)], T.models.map((m) => [m, ...T.sets.map((x) => f4(nd(m, x.id)))])) +
        table(["Paired difference", "Seed", "Mean", "95% interval"], T.differences.map((r) =>
          [`${r.model}: ${r.set === "bow" ? "bag-of-words" : "LLM"} − ${(r.against ?? "base") === "base" ? "base" : "bag-of-words"}`, r.seed, sgn4(r.diff), `${sgn4(r.lo)} to ${sgn4(r.hi)}`])) +
        table(["Share of attribution on the text columns", ...textSets.map((x) => x.label)], T.models.map((m) =>
          [m, ...textSets.map((x) => { const t = T.text_share.find((r) => r.model === m && r.set === x.id); return t ? `${(100 * t.share).toFixed(1)}%` : "n/a"; })]));
    },
  });

  // 5. Cold start
  panel({
    id: "cold", title: "How many liked places does a new user need?", source: "src/32_cold_start.py",
    build(p) {
      const C = D.cold;
      if (!C) return missing(p, "The cold-start study");
      const of = (m) => C.rows.filter((r) => r.model === m);
      const lead = of("LambdaRank"), at = (k) => lead.find((r) => r.k === k);
      p.take.innerHTML = `For ${int(C.users)} test users with at least ${C.min_liked} liked places, a profile built from only their <b>5</b> most recent likes reaches <span class="num">${pct(at("5").share_of_full)}</span> of the full-history NDCG@10 (LambdaRank); <b>10</b> likes reach <span class="num">${pct(at("10").share_of_full)}</span>, and a single like <span class="num">${pct(at("1").share_of_full)}</span>.`;
      const width = Math.max(p.chart.clientWidth, 320), h = 280, m = { l: 56, r: 110, t: 10, b: 44 };
      const x = d3.scalePoint().domain(C.ks).range([m.l + 10, width - m.r]);
      const y = d3.scaleLinear().domain([0, d3.max(C.rows, (r) => r.ndcg_hi)]).nice().range([h - m.b, m.t]);
      const svg = d3.select(p.chart).html("").append("svg").attr("width", width).attr("height", h).attr("role", "img").attr("aria-label", "NDCG@10 by number of liked places used");
      svg.append("g").attr("class", "ov-grid").selectAll("line").data(y.ticks(5)).join("line").attr("x1", m.l).attr("x2", width - m.r).attr("y1", y).attr("y2", y);
      svg.append("g").attr("class", "ov-axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).tickSize(4).tickFormat((k) => (k === "full" ? "all" : k)));
      svg.append("g").attr("class", "ov-axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(4));
      svg.append("text").attr("class", "ov-axis-title").attr("x", width - m.r).attr("y", h - 6).attr("text-anchor", "end").text("Liked places in the profile");
      svg.append("text").attr("class", "ov-axis-title").attr("x", m.l + 6).attr("y", m.t + 8).text("NDCG@10");
      const ends = C.models.map((name, k) => ({ name, k, y: y(of(name).at(-1).ndcg) })).sort((a, b) => a.y - b.y);
      ends.forEach((e, i) => { if (i && e.y - ends[i - 1].y < 14) e.y = ends[i - 1].y + 14; });
      C.models.forEach((name, k) => {
        const rows = of(name), off = (k - 1) * 5;
        svg.append("path").attr("class", "ov-series").attr("stroke-dasharray", SERIES[k].dash).attr("d", d3.line().x((r) => x(r.k) + off).y((r) => y(r.ndcg))(rows));
        const g = svg.selectAll(`g.cs${k}`).data(rows).join("g").attr("class", `cs${k}`);
        g.append("line").attr("class", "ov-ci").attr("x1", (r) => x(r.k) + off).attr("x2", (r) => x(r.k) + off).attr("y1", (r) => y(r.ndcg_lo)).attr("y2", (r) => y(r.ndcg_hi));
        g.append("path").attr("class", "ov-mark").attr("transform", (r) => `translate(${x(r.k) + off},${y(r.ndcg)})`).attr("d", sym(k, 80)).attr("fill", SERIES[k].fill);
        hover(g, (r) => `<b>${name}</b>, ${r.k === "full" ? "full history" : `${r.k} liked`}<br>NDCG@10 ${f4(r.ndcg)} [${f4(r.ndcg_lo)}, ${f4(r.ndcg_hi)}]<br>${pct(r.share_of_full)} of full history`);
        svg.append("text").attr("class", "ov-note").attr("x", width - m.r + 24).attr("y", ends.find((e) => e.k === k).y).attr("dy", "0.35em").text(name.replace("Logistic regression", "Logistic"));
      });
      p.controls.innerHTML = legend(C.models);
      p.table.innerHTML = table(["Model", "Liked places", "NDCG@10", "95% interval", "Share of full", "Stage-1 recall"],
        C.rows.map((r) => [r.model, r.k, f4(r.ndcg), `${f4(r.ndcg_lo)} to ${f4(r.ndcg_hi)}`, pct(r.share_of_full), f3(r.stage1_recall)]));
    },
  });

  // 6. Retrievers
  panel({
    id: "retr", title: "Does stage 1 need four retrievers?", source: "src/37_retriever_ablation.py",
    build(p) {
      const R = D.retrievers;
      if (!R) return missing(p, "The retriever study");
      const worst = R.rows.reduce((a, b) => (b.change < a.change ? b : a));
      const gain = R.rows.filter((r) => r.change_lo > 0).length;
      const within = R.item_cf_200 >= R.recall_lo && R.item_cf_200 <= R.recall_hi;
      p.take.innerHTML = `Stage-1 recall is <span class="num">${f4(R.recall)}</span>. Removing <b>${worst.label.split(" (")[0]}</b> costs the most (<span class="num">${sgn4(worst.change)}</span>); ${gain} of the ${R.rows.length} retrievers can be swapped for more item-CF at a small gain.` +
        ` Item-CF alone with 200 candidates reaches <span class="num">${f4(R.item_cf_200)}</span>${within ? ", inside the interval of the four-retriever mix: on recall the mix is not better, it only spreads where candidates come from" : ""}.`;
      intervals(p.chart, R.rows.map((r) => ({ label: r.label.split(" (")[0], mean: r.change, lo: r.change_lo, hi: r.change_hi, series: r.change < 0 ? 1 : 0 })),
        { zero: true, fmt: sgn4, labelW: 150, axisLabel: "Change in stage-1 recall when the retriever is removed",
          tipHtml: (r) => `<b>${r.label}</b><br>${sgn4(r.mean)} [${sgn4(r.lo)}, ${sgn4(r.hi)}]` });
      p.table.innerHTML = table(["Retriever", "Quota", "Recall of its own list", "Visits credited", "Recall without it"],
        R.rows.map((r) => [r.label, r.quota, f4(r.recall_alone), `${int(r.credited_positives)} (${(100 * r.credited_share).toFixed(1)}%)`, f4(r.recall_without)]));
    },
  });

  // 7. One-sentence steering
  panel({
    id: "nl", title: "Can one sentence set the controls?", source: "src/40_nl_eval.py, tests/nl_cases.json, server/nl.py",
    build(p) {
      const N = D.nl;
      if (!N) return missing(p, "The one-sentence evaluation");
      const kw = N.readers.keyword, lm = N.readers.llm;
      if (!lm) { p.take.innerHTML = `The language model has not been run on the ${N.sentences} test sentences yet. Keyword rules read <span class="num">${kw.exact}</span> of them completely right.`; }
      else {
        const zh = N.kinds.find((k) => k.kind === "zh"), out = N.kinds.find((k) => k.kind === "out");
        const rest = N.kinds.filter((k) => k.kind !== "zh" && k.kind !== "out").reduce((a, k) => ({ n: a.n + k.n, llm: a.llm + k.llm, keyword: a.keyword + k.keyword }), { n: 0, llm: 0, keyword: 0 });
        p.take.innerHTML = `A language model that only fills in a form reads <b><span class="num">${lm.exact} of ${N.sentences}</span></b> test sentences completely right (<span class="num">${pct(lm.share)}</span>); keyword rules read <span class="num">${kw.exact}</span> (<span class="num">${pct(kw.share)}</span>). `
          + `The lead is in two kinds of sentence that rules cannot handle: Chinese (<span class="num">${zh.llm} against ${zh.keyword} of ${zh.n}</span>) and sentences that are not about eating (<span class="num">${out.llm} against ${out.keyword} of ${out.n}</span>). On the other <span class="num">${rest.n}</span> the two are close (<span class="num">${rest.llm} and ${rest.keyword}</span>). `
          + `The sentences and their expected answers were written by us, so this checks the instructions, not unseen language. One sentence costs about <span class="num">$${N.per_sentence_usd.toFixed(4)}</span>.`;
      }
      const rows = Object.entries(N.readers).map(([key, r]) => ({ label: key === "llm" ? "Language model" : "Keyword rules", mean: r.share, lo: r.lo, hi: r.hi, series: key === "llm" ? 1 : 0 }));
      intervals(p.chart, rows, { fmt: pct, domain: [0, 1], labelW: 150, axisLabel: "Share of sentences read completely right, 95% Wilson interval",
        tipHtml: (r) => `<b>${r.label}</b><br>${pct(r.mean)} [${pct(r.lo)}, ${pct(r.hi)}] of ${N.sentences} sentences` });
      const cols = Object.keys(N.readers), nm = { llm: "Language model", keyword: "Keyword rules" };
      p.table.innerHTML = table(["Kind of sentence", "Sentences", ...cols.map((c) => `${nm[c]}: right`)], N.kinds.map((k) => [k.label, k.n, ...cols.map((c) => k[c])]))
        + table(["Setting", "Should be set", ...cols.map((c) => `${nm[c]}: right`), ...cols.map((c) => `${nm[c]}: set by mistake`)],
          N.slots.map((s) => [s.slot, s[cols[0]].should, ...cols.map((c) => s[c].right), ...cols.map((c) => `${s[c].wrongly_set} of ${s[c].rest}`)]));
    },
  });

  function render() { hideTip(); panels.forEach((b) => b()); }
  return { render };
}
