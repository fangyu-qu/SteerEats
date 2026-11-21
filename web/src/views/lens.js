// Model lens (V6): do logistic regression and LightGBM rank and explain this user's candidates the same way?
// Four linked figures: the two top-10 lists side by side, where this user sits among all test users, the SHAP
// summary of both models on a shared axis, and per-group attribution agreement. Hover and click link to the
// other views through the candidate index.
import * as d3 from "d3";
import { MODEL_NAMES, esc, truncate } from "./common.js";
import { FEATURE, GROUP_LABEL } from "./ledger.js";
import { groupShares, kendallTauB, overlapAt, pearson, percentiles, sd } from "../lens.js";

const MODELS = ["lr", "lgbm"];
const SHORT = { lr: "LR", lgbm: "LightGBM" };
const SWARM_ROWS = 8;
const SHORT_GROUP = { Collaborative: "your history", Popularity: "popularity and recency", Rating: "rating", Location: "location",
  Taste: "taste match", Attributes: "amenities" };
const pct = (v) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;
// Axis half-width from the 99th percentile of |values|, so a few extreme candidates do not flatten the rest;
// values beyond it are pinned to the edge and drawn hollow.
function robustMax(arrays) {
  const all = [];
  for (const a of arrays) for (const v of a) all.push(Math.abs(v));
  all.sort((p, q) => p - q);
  return Math.max(d3.quantileSorted(all, 0.99) * 1.15, 1e-6);
}
const clampTo = (m) => (v) => Math.max(-m, Math.min(m, v));
const signed = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);

export function createLens(root, pop, handlers) {
  root.innerHTML = `
    <p class="lens-headline" id="lens-headline"></p>
    <div class="lens-row">
      <figure class="lens-fig">
        <figcaption><b>Two lists for this user</b><span id="lens-slope-cap"></span></figcaption>
        <div id="lens-slope" class="lens-svg"></div>
      </figure>
      <figure class="lens-fig">
        <figcaption><b>Against all ${pop.users.toLocaleString()} test users</b><span>Both models' own lists, no steering. The highlighted bar and the vertical line mark this user.</span></figcaption>
        <div id="lens-pop" class="lens-svg"></div>
        <table class="feat lens-acc" id="lens-acc"></table>
      </figure>
    </div>
    <figure class="lens-fig">
      <figcaption><b>What each model's scores rest on</b><span>One dot per candidate. Position is the feature's SHAP value (log-odds of a review); shade is the feature's value among this user's candidates. Rows are the features that spread this user's scores most (standard deviation), since a feature that shifts every candidate equally cannot change the order. Both panels share one axis.</span></figcaption>
      <div class="swarm-key" id="lens-swarm-key"></div>
      <div id="lens-swarm" class="lens-svg"></div>
    </figure>
    <div class="lens-row">
      <figure class="lens-fig">
        <figcaption><b>Which model leans on what</b><span>Share of mean |SHAP| by feature group. Filled marks: this user's candidates; hollow: ${pop.shap_groups.users} random test users.</span></figcaption>
        <ul class="key" id="lens-rel-key"></ul>
        <div id="lens-rel" class="lens-svg"></div>
      </figure>
      <figure class="lens-fig">
        <figcaption><b>Do they credit the same reasons?</b><span>Each dot is a candidate: LR's SHAP for the group across, LightGBM's up. On the dashed line the models agree; above it LightGBM gives the group more credit. Darker dots are in either model's top 10.</span></figcaption>
        <div id="lens-agree" class="lens-svg"></div>
      </figure>
    </div>
    <div class="lens-tip" id="lens-tip" role="tooltip" hidden></div>`;
  const $ = (id) => root.querySelector("#" + id);
  const tip = $("lens-tip");
  const showTip = (event, html) => {
    const box = root.getBoundingClientRect();
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, x = event.clientX - box.left, y = event.clientY - box.top;
    tip.style.left = `${Math.min(Math.max(4, x + 14), box.width - w - 4)}px`;
    tip.style.top = `${y + 16}px`;
  };
  const hideTip = () => { tip.hidden = true; };
  $("lens-rel-key").innerHTML = `<li><svg width="12" height="12"><rect x="1.5" y="1.5" width="9" height="9" fill="var(--ink)"/></svg>Logistic regression</li>
    <li><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--ink)"/></svg>LightGBM</li>`;

  let cache = null;      // per-case statistics
  let drawnKey = null;   // case + widths of the static figures
  const layers = {};     // ring layers for hover and selection
  let selected = null, hovered = null;

  function caseStats(c) {
    if (cache?.id === c.id) return cache;
    const shares = { lr: groupShares(c, "lr"), lgbm: groupShares(c, "lgbm") };
    const spread = c.features.map((f) => ({ f, s: Math.max(sd(c.lr.shap[f]), sd(c.lgbm.shap[f])) }))
      .sort((a, b) => b.s - a.s).slice(0, SWARM_ROWS).map((d) => d.f);
    const pctl = Object.fromEntries(spread.map((f) => [f, percentiles(c.feature_values[f])]));
    cache = { id: c.id, tau: kendallTauB(c.lr.raw, c.lgbm.raw), shares, spread, pctl };
    return cache;
  }

  function update(ctx) {
    const { c } = ctx;
    const st = caseStats(c);
    selected = ctx.selected;
    const ovDefault = overlapAt(ctx.defaults.lr, ctx.defaults.lgbm);
    const ovNow = overlapAt(ctx.finals.lr, ctx.finals.lgbm);
    headline(c, st, ovDefault);
    renderSlope(c, ctx, ovNow, ovDefault);
    const key = `${c.id}|${$("lens-pop").clientWidth}|${$("lens-swarm").clientWidth}|${$("lens-agree").clientWidth}|${$("lens-rel").clientWidth}`;
    if (key !== drawnKey) {
      drawnKey = key;
      renderPop(st, ovDefault);
      renderSwarm(c, st);
      renderReliance(st);
    }
    renderAgree(c, ctx, drawnKey + "|" + topKey(ctx));
    rings();
  }

  const topKey = (ctx) => MODELS.map((m) => Array.from(ctx.finals[m]).map((r, i) => (r <= 10 ? i : -1)).filter((i) => i >= 0).join(",")).join("|");

  function headline(c, st, ov) {
    const counts = pop.overlap.counts, n = pop.users;
    const more = counts.slice(ov + 1).reduce((a, b) => a + b, 0) / n;
    const diffs = Object.keys(st.shares.lr).map((g) => ({ g, d: st.shares.lgbm[g] - st.shares.lr[g] })).sort((a, b) => b.d - a.d);
    const hi = diffs[0], lo = diffs[diffs.length - 1];
    const lean = (x, m, o) => `<b>${SHORT_GROUP[x.g] ?? x.g}</b> (${pct(st.shares[m][x.g])} of its attribution, against ${pct(st.shares[o][x.g])})`;
    $("lens-headline").innerHTML = `With no steering the two models' top 10s share <b>${ov} of 10</b> places for this user; `
      + `${pct(more)} of test users have more in common (median ${pop.overlap.median}). `
      + `LightGBM leans more on ${lean(hi, "lgbm", "lr")}, logistic regression on ${lean(lo, "lr", "lgbm")}.`;
  }

  // --- Two lists ---------------------------------------------------------------------------------
  function renderSlope(c, ctx, ovNow, ovDefault) {
    const node = $("lens-slope");
    const width = Math.max(node.clientWidth, 400), ROW = 22, TOP = 34, BELOW = 17;
    const nameW = Math.min(200, (width - 150) / 2), xl = nameW + 16, xr = width - nameW - 16;
    const chars = Math.floor(nameW / 6.7);
    const R = ctx.finals, n = c.cuisine.length;
    const union = [...Array(n).keys()].filter((i) => R.lr[i] <= 10 || R.lgbm[i] <= 10);
    const below = { lr: union.filter((i) => R.lr[i] > 10).sort((a, b) => R.lr[a] - R.lr[b]),
      lgbm: union.filter((i) => R.lgbm[i] > 10).sort((a, b) => R.lgbm[a] - R.lgbm[b]) };
    const belowY = TOP + 10 * ROW + 18;
    const y = (m, i) => (R[m][i] <= 10 ? TOP + (R[m][i] - 0.5) * ROW : belowY + 8 + below[m].indexOf(i) * BELOW);
    const height = belowY + 14 + Math.max(below.lr.length, below.lgbm.length, 1) * BELOW;
    $("lens-slope-cap").textContent = ctx.steered
      ? `Each model solved for your current targets. ${ovNow} of 10 places in both lists (${ovDefault} with the models' own lists).`
      : `The models' own lists. ${ovNow} of 10 places in both. Dashed lines lead to places only one model ranks in its top 10.`;
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
      .attr("role", "img").attr("aria-label", `Slope chart comparing the two models' top 10: ${ovNow} shared`);
    svg.selectAll("*").remove();
    const head = (x, anchor, m) => svg.append("text").attr("class", "lens-col").attr("x", x).attr("y", 14).attr("text-anchor", anchor)
      .text(MODEL_NAMES[m] + (m === ctx.model ? " · shown above" : ""));
    head(xl + 4, "end", "lr"); head(xr - 4, "start", "lgbm");
    svg.append("line").attr("class", "lens-guide").attr("x1", 0).attr("x2", width).attr("y1", belowY - 4).attr("y2", belowY - 4);
    layers.slopeNote = svg.append("text").attr("class", "lens-note halo").attr("x", width / 2).attr("y", belowY - 8).attr("text-anchor", "middle").text("outside the top 10");
    const g = svg.selectAll("g.it").data(union, (i) => i).join("g").attr("class", "it").attr("data-id", (i) => i)
      .classed("selected", (i) => i === selected)
      .on("pointerenter", (_, i) => handlers.hover(i)).on("pointerleave", () => handlers.hover(null)).on("click", (_, i) => handlers.select(i));
    g.append("path").attr("class", (i) => (R.lr[i] <= 10 && R.lgbm[i] <= 10 ? "both" : "one"))
      .attr("d", (i) => { const a = y("lr", i), b = y("lgbm", i), m = (xl + xr) / 2; return `M${xl},${a} C${m},${a} ${m},${b} ${xr},${b}`; });
    for (const [m, x, anchor, dx] of [["lr", xl, "end", -10], ["lgbm", xr, "start", 10]]) {
      g.append("circle").attr("cx", x).attr("cy", (i) => y(m, i)).attr("r", (i) => (R[m][i] <= 10 ? 3.5 : 2.8))
        .attr("class", (i) => (R[m][i] <= 10 ? "in" : "out"));
      g.append("text").attr("x", x + dx).attr("y", (i) => y(m, i)).attr("dy", "0.35em").attr("text-anchor", anchor)
        .attr("class", (i) => (R[m][i] <= 10 ? "nm" : "rk"))
        .text((i) => (R[m][i] <= 10 ? `${R[m][i]}  ${truncate(c.candidates.name[i], chars)}` : `#${R[m][i]}`));
    }
    g.append("title").text((i) => `${c.candidates.name[i]}: LR #${R.lr[i]}, LightGBM #${R.lgbm[i]}`);
    layers.slopeNote.raise();
    layers.slope = svg;
  }

  // --- Population --------------------------------------------------------------------------------
  function renderPop(st, ov) {
    const node = $("lens-pop");
    const width = Math.max(node.clientWidth, 300), H = 118, M = { l: 10, r: 14, t: 26, b: 22 };
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", H * 2 + 18)
      .attr("role", "img").attr("aria-label", "Histograms of top-10 overlap and Kendall tau across test users");
    svg.selectAll("*").remove();
    const n = pop.users;
    // overlap
    const g1 = svg.append("g");
    const x1 = d3.scaleBand(d3.range(11), [M.l, width - M.r]).padding(0.18);
    const y1 = d3.scaleLinear([0, d3.max(pop.overlap.counts) / n], [H - M.b, M.t]);
    g1.append("text").attr("class", "lens-col").attr("x", M.l).attr("y", 12).text("Places both top 10s share");
    g1.selectAll("rect").data(pop.overlap.counts).join("rect")
      .attr("x", (_, k) => x1(k)).attr("width", x1.bandwidth()).attr("y", (d) => y1(d / n)).attr("height", (d) => y1(0) - y1(d / n))
      .attr("rx", 2).attr("class", (_, k) => (k === ov ? "bar me" : "bar"))
      .each(function (d, k) {
        d3.select(this).on("pointermove", (e) => showTip(e, `<b>${k} shared</b><br>${d.toLocaleString()} users (${pct(d / n)})${k === ov ? "<br>this user" : ""}`))
          .on("pointerleave", hideTip);
      });
    g1.selectAll("text.tk").data(d3.range(11)).join("text").attr("class", "tk").attr("x", (k) => x1(k) + x1.bandwidth() / 2)
      .attr("y", H - M.b + 14).attr("text-anchor", "middle").text((k) => k);
    g1.append("text").attr("class", "lens-me").attr("x", x1(ov) + x1.bandwidth() / 2).attr("y", y1(pop.overlap.counts[ov] / n) - 5)
      .attr("text-anchor", "middle").text("this user");
    // tau
    const g2 = svg.append("g").attr("transform", `translate(0,${H + 18})`);
    const e = pop.tau.edges, cnt = pop.tau.counts;
    const first = Math.max(0, cnt.findIndex((v) => v > 0) - 1), last = Math.min(cnt.length - 1, cnt.length - 1 - [...cnt].reverse().findIndex((v) => v > 0) + 1);
    const lo = Math.min(e[first], st.tau - 0.02), hi = Math.max(e[last + 1], st.tau + 0.02);
    const x2 = d3.scaleLinear([lo, hi], [M.l, width - M.r]);
    const y2 = d3.scaleLinear([0, d3.max(cnt) / n], [H - M.b, M.t]);
    g2.append("text").attr("class", "lens-col").attr("x", M.l).attr("y", 12)
      .text(`Kendall τ over all candidates · median ${pop.tau.median.toFixed(2)}`);
    const bins = cnt.map((v, k) => ({ v, a: e[k], b: e[k + 1] })).slice(first, last + 1);
    g2.selectAll("rect").data(bins).join("rect").attr("class", "bar").attr("rx", 2)
      .attr("x", (d) => x2(d.a) + 1).attr("width", (d) => Math.max(1, x2(d.b) - x2(d.a) - 2))
      .attr("y", (d) => y2(d.v / n)).attr("height", (d) => y2(0) - y2(d.v / n))
      .on("pointermove", (ev, d) => showTip(ev, `<b>τ ${d.a.toFixed(2)} to ${d.b.toFixed(2)}</b><br>${d.v.toLocaleString()} users (${pct(d.v / n)})`))
      .on("pointerleave", hideTip);
    g2.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${H - M.b})`)
      .call(d3.axisBottom(x2).ticks(6).tickSize(3).tickFormat(d3.format(".1f")));
    g2.append("line").attr("class", "lens-mark").attr("x1", x2(st.tau)).attr("x2", x2(st.tau)).attr("y1", M.t - 4).attr("y2", H - M.b);
    const right = x2(st.tau) > width - 90;
    g2.append("text").attr("class", "lens-me").attr("x", x2(st.tau) + (right ? -5 : 5)).attr("y", M.t + 4)
      .attr("text-anchor", right ? "end" : "start").text(`this user ${st.tau.toFixed(2)}`);
    // accuracy by agreement
    const rows = pop.by_overlap;
    $("lens-acc").innerHTML = `<caption>Does disagreement cost accuracy? Mean NDCG@10 by how much the lists agree</caption>
      <thead><tr><th>Lists share</th><th>Users</th><th>LR</th><th>LightGBM</th><th>Found only by</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td>${r.label.replace(" shared", "")}</td><td class="mono">${r.users.toLocaleString()}</td>
        <td class="mono">${r.ndcg_lr.toFixed(4)}</td><td class="mono">${r.ndcg_lgbm.toFixed(4)}</td>
        <td class="mono">LR ${r.only_lr} · LightGBM ${r.only_lgbm}</td></tr>`).join("")}</tbody>`;
  }

  // --- SHAP beeswarms ----------------------------------------------------------------------------
  function dodge(xs, r, maxOff) {
    const idx = Array.from(xs, (_, i) => i).sort((a, b) => xs[a] - xs[b]);
    const ys = new Float64Array(xs.length), placed = [];
    for (const i of idx) {
      const x = xs[i], near = [];
      for (let k = placed.length - 1; k >= 0 && x - xs[placed[k]] < 2 * r; k--) near.push(placed[k]);
      let y = null;
      for (let k = 0; k < 60; k++) {
        const cand = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * r * 0.75;
        if (Math.abs(cand) > maxOff) break;
        if (near.every((j) => (xs[j] - x) ** 2 + (ys[j] - cand) ** 2 >= (2 * r) ** 2 * 0.81)) { y = cand; break; }
      }
      ys[i] = y ?? (((i * 7919) % 101) / 100 * 2 - 1) * maxOff;
      placed.push(i);
    }
    return ys;
  }

  function renderSwarm(c, st) {
    const node = $("lens-swarm");
    const width = Math.max(node.clientWidth, 640), LABEL = 210, GAP = 24, ROW = 34, TOP = 22, AX = 26;
    const pw = (width - LABEL - GAP) / 2;
    const feats = st.spread;
    const height = TOP + feats.length * ROW + AX;
    const m0 = robustMax(feats.flatMap((f) => MODELS.map((mo) => c[mo].shap[f])));
    const panels = MODELS.map((mo, k) => ({ mo, x0: LABEL + k * (pw + GAP) }));
    const x = (p) => d3.scaleLinear([-m0, m0], [p.x0 + 6, p.x0 + pw - 6]).nice();
    const m = x(panels[0]).domain()[1], clip = clampTo(m);
    let clipped = 0;
    for (const f of feats) for (const mo of MODELS) for (const v of c[mo].shap[f]) if (Math.abs(v) > m) clipped++;
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink").trim(), panel = css.getPropertyValue("--panel").trim();
    const shade = d3.interpolateLab(d3.interpolateLab(panel, ink)(0.3), ink);
    $("lens-swarm-key").innerHTML = `<span>Feature value among this user's candidates</span><span class="ramp">low<i style="background:linear-gradient(90deg, ${shade(0)}, ${shade(1)})"></i>high</span>`
      + (clipped ? `<span class="ramp">${clipped} extreme ${clipped === 1 ? "value" : "values"} beyond ±${m} drawn hollow at the edge</span>` : "");
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
      .attr("role", "img").attr("aria-label", "SHAP beeswarm for both models on a shared axis");
    svg.selectAll("*").remove();
    const pos = {};
    for (const p of panels) {
      const sx = x(p), g = svg.append("g");
      g.append("text").attr("class", "lens-col").attr("x", p.x0 + 6).attr("y", 12).text(MODEL_NAMES[p.mo]);
      g.append("line").attr("class", "zero").attr("x1", sx(0)).attr("x2", sx(0)).attr("y1", TOP).attr("y2", TOP + feats.length * ROW);
      g.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${TOP + feats.length * ROW + 2})`)
        .call(d3.axisBottom(sx).ticks(5).tickSize(3).tickFormat((v) => (v > 0 ? "+" : "") + d3.format(".1f")(v)));
      pos[p.mo] = {};
      feats.forEach((f, r) => {
        const cy = TOP + (r + 0.5) * ROW;
        const xs = Float64Array.from(c[p.mo].shap[f], (v) => sx(clip(v)));
        const ys = dodge(xs, 2.3, ROW / 2 - 3);
        pos[p.mo][f] = { xs, ys: ys.map((v) => v + cy) };
        g.append("g").attr("class", "dots").selectAll("circle").data(Array.from(xs, (_, i) => i)).join("circle")
          .attr("cx", (i) => xs[i]).attr("cy", (i) => cy + ys[i]).attr("r", 2.3)
          .attr("fill", (i) => (Math.abs(c[p.mo].shap[f][i]) > m ? "var(--panel)" : shade(st.pctl[f][i])))
          .attr("stroke", (i) => (Math.abs(c[p.mo].shap[f][i]) > m ? shade(st.pctl[f][i]) : null));
      });
      const pts = [];
      feats.forEach((f) => pos[p.mo][f].xs.forEach((xv, i) => pts.push([xv, pos[p.mo][f].ys[i], i, f])));
      const del = d3.Delaunay.from(pts, (d) => d[0], (d) => d[1]);
      g.append("rect").attr("class", "hit").attr("x", p.x0).attr("y", TOP).attr("width", pw).attr("height", feats.length * ROW)
        .on("pointermove", (e) => {
          const [mx, my] = d3.pointer(e, svg.node()), k = del.find(mx, my), d = pts[k];
          if (Math.hypot(d[0] - mx, d[1] - my) > 14) { hideTip(); handlers.hover(null); return; }
          const [, , i, f] = d, fmt = FEATURE[f]?.[1] ?? ((v) => v.toFixed(2));
          showTip(e, `<b>${esc(c.candidates.name[i])}</b><br>${esc(FEATURE[f]?.[0] ?? f)}: ${fmt(c.feature_values[f][i])}<br>
            SHAP <span class="mono">LR ${signed(c.lr.shap[f][i])} · LightGBM ${signed(c.lgbm.shap[f][i])}</span>`);
          handlers.hover(i);
        })
        .on("pointerleave", () => { hideTip(); handlers.hover(null); })
        .on("click", (e) => { const [mx, my] = d3.pointer(e, svg.node()), d = pts[del.find(mx, my)]; if (Math.hypot(d[0] - mx, d[1] - my) <= 14) handlers.select(d[2]); });
    }
    feats.forEach((f, r) => {
      const cy = TOP + (r + 0.5) * ROW;
      svg.append("line").attr("class", "lens-guide").attr("x1", 0).attr("x2", width).attr("y1", cy + ROW / 2).attr("y2", cy + ROW / 2);
      svg.append("text").attr("class", "lab").attr("x", 0).attr("y", cy).attr("dy", "0.35em").text(truncate(FEATURE[f]?.[0] ?? f, 30))
        .append("title").text(`${f} (${Object.keys(c.feature_groups).find((g) => c.feature_groups[g].includes(f))})`);
    });
    layers.swarm = { svg: svg.append("g").attr("class", "rings"), pos, feats };
  }

  // --- Reliance dumbbell -------------------------------------------------------------------------
  function renderReliance(st) {
    const node = $("lens-rel");
    const width = Math.max(node.clientWidth, 320), LABEL = 172, ROW = 34, TOP = 8, AX = 24;
    const groups = Object.keys(st.shares.lr).sort((a, b) => Math.max(st.shares.lgbm[b], st.shares.lr[b]) - Math.max(st.shares.lgbm[a], st.shares.lr[a]));
    const P = pop.shap_groups.rows;
    const max = d3.max(groups, (g) => Math.max(st.shares.lr[g], st.shares.lgbm[g], P[g]?.lr ?? 0, P[g]?.lgbm ?? 0));
    const x = d3.scaleLinear([0, max], [LABEL, width - 12]).nice();
    const height = TOP + groups.length * ROW + AX;
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", height)
      .attr("role", "img").attr("aria-label", "Share of SHAP attribution by feature group for each model");
    svg.selectAll("*").remove();
    svg.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${TOP + groups.length * ROW})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(3).tickFormat(d3.format(".0%")));
    const mark = (g, m, cx, cy, filled, size) => (m === "lr"
      ? g.append("rect").attr("x", cx - size).attr("y", cy - size).attr("width", 2 * size).attr("height", 2 * size)
      : g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", size + 0.6)).attr("class", filled ? "mk fill" : "mk hollow");
    groups.forEach((gname, r) => {
      const cy = TOP + (r + 0.5) * ROW, g = svg.append("g").attr("class", "rel-row");
      g.append("text").attr("class", "lab").attr("x", 0).attr("y", cy - 4).attr("dy", "0.35em").text(GROUP_LABEL[gname] ?? gname);
      const u = st.shares, p = P[gname];
      if (p) {
        g.append("line").attr("class", "db pop").attr("x1", x(p.lr)).attr("x2", x(p.lgbm)).attr("y1", cy + 7).attr("y2", cy + 7);
        mark(g, "lr", x(p.lr), cy + 7, false, 3); mark(g, "lgbm", x(p.lgbm), cy + 7, false, 3);
      }
      g.append("line").attr("class", "db me").attr("x1", x(u.lr[gname])).attr("x2", x(u.lgbm[gname])).attr("y1", cy - 4).attr("y2", cy - 4);
      mark(g, "lr", x(u.lr[gname]), cy - 4, true, 4); mark(g, "lgbm", x(u.lgbm[gname]), cy - 4, true, 4);
      g.append("rect").attr("class", "hit").attr("x", 0).attr("y", cy - ROW / 2).attr("width", width).attr("height", ROW)
        .on("pointermove", (e) => showTip(e, `<b>${GROUP_LABEL[gname] ?? gname}</b><br>This user: LR ${pct(u.lr[gname])} · LightGBM ${pct(u.lgbm[gname])}`
          + (p ? `<br>${pop.shap_groups.users} test users: LR ${pct(p.lr)} · LightGBM ${pct(p.lgbm)}` : "")))
        .on("pointerleave", hideTip);
    });
  }

  // --- Attribution agreement ---------------------------------------------------------------------
  let agreeKey = null;
  function renderAgree(c, ctx, key) {
    if (key === agreeKey) return;
    agreeKey = key;
    const node = $("lens-agree");
    const width = Math.max(node.clientWidth, 330), COLS = 3, GAP = 14;
    const cell = Math.min(200, (width - GAP * (COLS - 1)) / COLS), M = { l: 6, t: 20, r: 6, b: 6 };
    const groups = Object.keys(c.feature_groups);
    const rowsN = Math.ceil(groups.length / COLS);
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", COLS * cell + GAP * (COLS - 1))
      .attr("height", rowsN * (cell + GAP) + 16).attr("role", "img").attr("aria-label", "Scatter of LR against LightGBM SHAP per feature group");
    svg.selectAll("*").remove();
    const inTop = (i) => ctx.finals.lr[i] <= 10 || ctx.finals.lgbm[i] <= 10;
    const n = c.cuisine.length, ids = [...Array(n).keys()].sort((a, b) => inTop(a) - inTop(b));
    const pos = {};
    groups.forEach((gname, k) => {
      const ox = (k % COLS) * (cell + GAP), oy = Math.floor(k / COLS) * (cell + GAP);
      const a = c.lr.shap_groups[gname], b = c.lgbm.shap_groups[gname];
      const sx = d3.scaleLinear([-robustMax([a, b]), robustMax([a, b])], [ox + M.l, ox + cell - M.r]).nice(), sy = d3.scaleLinear(sx.domain(), [oy + cell - M.b, oy + M.t]);
      const g = svg.append("g");
      g.append("rect").attr("class", "frame").attr("x", ox).attr("y", oy + M.t - 4).attr("width", cell).attr("height", cell - M.t + 4).attr("rx", 4);
      g.append("line").attr("class", "zero").attr("x1", sx(0)).attr("x2", sx(0)).attr("y1", sy.range()[0]).attr("y2", sy.range()[1]);
      g.append("line").attr("class", "zero").attr("y1", sy(0)).attr("y2", sy(0)).attr("x1", sx.range()[0]).attr("x2", sx.range()[1]);
      const [d0, d1] = sx.domain();
      g.append("line").attr("class", "diag").attr("x1", sx(d0)).attr("y1", sy(d0)).attr("x2", sx(d1)).attr("y2", sy(d1));
      const r = pearson(a, b);
      const title = (SHORT_GROUP[gname] ?? gname).replace(/^./, (ch) => ch.toUpperCase());
      g.append("text").attr("class", "lens-col").attr("x", ox).attr("y", oy + 10).text(truncate(title, Math.floor((cell - 46) / 6.8)));
      g.append("text").attr("class", "lens-r").attr("x", ox + cell).attr("y", oy + 10).attr("text-anchor", "end")
        .text(Number.isFinite(r) ? `r ${r.toFixed(2)}` : "r —");
      g.append("text").attr("class", "tk").attr("x", ox + cell - M.r - 2).attr("y", sy(0) - 4).attr("text-anchor", "end").text("LR →");
      g.append("text").attr("class", "tk").attr("x", sx(0) + 4).attr("y", oy + M.t + 8).text("↑ LightGBM");
      const lim = sx.domain()[1], cl = clampTo(lim), out = (i) => Math.abs(a[i]) > lim || Math.abs(b[i]) > lim;
      const pts = Array.from({ length: n }, (_, i) => [sx(cl(a[i])), sy(cl(b[i]))]);
      g.selectAll("circle").data(ids).join("circle").attr("cx", (i) => pts[i][0]).attr("cy", (i) => pts[i][1]).attr("r", 2.2)
        .attr("class", (i) => (inTop(i) ? "pt top" : "pt") + (out(i) ? " clip" : ""));
      pos[gname] = pts;
      const nOut = ids.filter(out).length;
      if (nOut) g.append("text").attr("class", "tk").attr("x", ox + 4).attr("y", oy + cell - 4).text(`${nOut} pinned to edge`);
      const del = d3.Delaunay.from(pts);
      const near = (e) => { const [mx, my] = d3.pointer(e, svg.node()), i = del.find(mx, my); return Math.hypot(pts[i][0] - mx, pts[i][1] - my) <= 12 ? i : null; };
      g.append("rect").attr("class", "hit").attr("x", ox).attr("y", oy + M.t - 4).attr("width", cell).attr("height", cell - M.t + 4)
        .on("pointermove", (e) => {
          const i = near(e);
          if (i == null) { hideTip(); handlers.hover(null); return; }
          showTip(e, `<b>${esc(c.candidates.name[i])}</b><br>${GROUP_LABEL[gname] ?? gname}: <span class="mono">LR ${signed(a[i])} · LightGBM ${signed(b[i])}</span><br>`
            + `rank LR #${ctx.finals.lr[i]} · LightGBM #${ctx.finals.lgbm[i]}`);
          handlers.hover(i);
        })
        .on("pointerleave", () => { hideTip(); handlers.hover(null); })
        .on("click", (e) => { const i = near(e); if (i != null) handlers.select(i); });
    });
    layers.agree = { svg: svg.append("g").attr("class", "rings"), pos };
  }

  // Rings for the selected and hovered restaurant in the dot figures; emphasis in the slope chart.
  function rings() {
    const marks = [[selected, "sel"], [hovered, "hov"]].filter(([i]) => i != null);
    if (layers.swarm) {
      const { svg, pos, feats } = layers.swarm;
      svg.selectAll("circle").data(marks.flatMap(([i, cls]) => MODELS.flatMap((m) => feats.map((f) => ({ x: pos[m][f].xs[i], y: pos[m][f].ys[i], cls })))))
        .join("circle").attr("cx", (d) => d.x).attr("cy", (d) => d.y).attr("r", 4.5).attr("class", (d) => `ring ${d.cls}`);
    }
    if (layers.agree) {
      const { svg, pos } = layers.agree;
      svg.selectAll("circle").data(marks.flatMap(([i, cls]) => Object.values(pos).map((pts) => ({ x: pts[i][0], y: pts[i][1], cls }))))
        .join("circle").attr("cx", (d) => d.x).attr("cy", (d) => d.y).attr("r", 4.5).attr("class", (d) => `ring ${d.cls}`);
    }
    const sl = $("lens-slope");
    sl.classList.toggle("has-hover", hovered != null);
    sl.querySelectorAll("g.it").forEach((g) => { g.classList.toggle("hover", +g.dataset.id === hovered); g.classList.toggle("selected", +g.dataset.id === selected); });
  }

  function highlight(i) { hovered = i; rings(); }
  return { update, highlight };
}
