// Trade-off space (V5): what does steering cost? Two linked scatters over the same 192 control settings
// (step 20's joint grid). Left: this user, relevance as the model's predicted visits in the top 10
// (Platt-calibrated), computed in the browser. Right: 2,000 test users, measured NDCG@10 with paired 95%
// intervals. Both: the quantity being steered across, relevance kept (% of the model's own list) up, shade
// = cuisine groups, staircase = Pareto frontier. Hover links the panels by setting; click loads a setting.
import * as d3 from "d3";
import { MODEL_NAMES, esc } from "./common.js";
import { LAM_MAX, THETA0, outcomes, topK } from "../steer.js";
import { SETTINGS, frontier, groupByList, l1, predictedVisits, sliderDistance, userSpace, visitProbs } from "../tradeoff.js";

const ordinal = (n) => `${n}${[11, 12, 13].includes(n % 100) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" })[n % 10] ?? "th"}`;
const QUANT = {
  km: { pop: "distance_km", label: "Average distance of the top 10 (km)", short: "Mean distance (km)", noun: "average distance", low: true, better: "closer", most: "closest",
    asGood: "at least as close", fmt: (v) => `${v.toFixed(1)} km`, tick: (v) => d3.format("~g")(v) },
  popularity: { pop: "popularity", label: "Average popularity percentile of the top 10", short: "Mean popularity pct", noun: "average popularity", low: true, phrase: (v) => `the ${ordinal(Math.round(v * 100))} percentile`,
    better: "less mainstream", most: "least mainstream", asGood: "at least as far from the mainstream", fmt: (v) => `${ordinal(Math.round(v * 100))} percentile`, tick: (v) => `${Math.round(v * 100)}` },
  cuisines: { pop: "cuisines", label: "Cuisine groups in the top 10", short: "Cuisine groups", noun: "number of cuisine groups", low: false, better: "more varied", most: "most varied",
    asGood: "at least as varied", fmt: (v) => (Number.isInteger(v) ? `${v}` : v.toFixed(1)), tick: (v) => d3.format("~g")(v) },
};
const H = 300, M = { l: 44, r: 18, t: 22, b: 42 };
// never round a value below 100% up to "100%"
const pct = (v) => (v < 100 && Math.round(v) >= 100 ? `${Math.floor(v * 10) / 10}%` : `${Math.round(v)}%`);
const say = (Q, v) => (Q.phrase ?? Q.fmt)(v);
const thetaText = (t) => {
  const parts = [];
  if (t.w_h !== 1) parts.push(`history ${t.w_h.toFixed(2)}`);
  if (t.alpha) parts.push(`α ${t.alpha.toFixed(2)}`);
  if (t.beta) parts.push(`β ${t.beta.toFixed(2)}`);
  if (t.lam) parts.push(`λ ${(t.lam / LAM_MAX).toFixed(2)}`);
  return parts.length ? parts.join(" · ") : "no steering";
};

export function createTradeoff(root, pop, handlers) {
  root.innerHTML = `
    <div class="eval-top">
      <p class="lens-headline" id="to-headline"></p>
      <div class="segmented" role="radiogroup" aria-label="Quantity to steer">
        <button type="button" role="radio" data-q="km" aria-checked="true">Closer</button>
        <button type="button" role="radio" data-q="popularity" aria-checked="false">Less mainstream</button>
        <button type="button" role="radio" data-q="cuisines" aria-checked="false">More variety</button>
      </div>
    </div>
    <div class="swarm-key to-key" id="to-key"></div>
    <div class="lens-row">
      <figure class="lens-fig">
        <figcaption><b>This user: predicted visits kept</b><span id="to-user-cap"></span></figcaption>
        <div id="to-user" class="lens-svg"></div>
      </figure>
      <figure class="lens-fig">
        <figcaption><b>${pop.users.toLocaleString()} test users: NDCG@10 kept</b><span id="to-pop-cap"></span></figcaption>
        <div id="to-pop" class="lens-svg"></div>
      </figure>
    </div>
    <div class="lens-tip" id="to-tip" role="tooltip" hidden></div>`;
  const $ = (id) => root.querySelector("#" + id);
  const tip = $("to-tip");
  const showTip = (event, html) => {
    const box = root.getBoundingClientRect();
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, x = event.clientX - box.left, y = event.clientY - box.top;
    tip.style.left = `${Math.min(Math.max(4, x + 14), box.width - w - 4)}px`;
    tip.style.top = `${y + 16}px`;
  };
  const hideTip = () => { tip.hidden = true; };

  let q = "km", ctx = null, hoverK = null;
  const cache = new Map();
  const layers = {};
  root.querySelectorAll(".segmented button").forEach((b) => b.addEventListener("click", () => {
    q = b.dataset.q;
    root.querySelectorAll(".segmented button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    render();
  }));

  // Population rows with relevance kept and the interval on it, per model.
  const popRows = {};
  for (const m of Object.keys(pop.settings)) {
    const rows = pop.settings[m];
    const base = rows.find((r) => r.w_h === 1 && !r.alpha && !r.beta && !r.lam);
    popRows[m] = rows.map((r, k) => ({ ...r, k, kept: (r.ndcg / base.ndcg) * 100, keptLo: ((base.ndcg + r.lo) / base.ndcg) * 100,
      keptHi: ((base.ndcg + r.hi) / base.ndcg) * 100, base: k === rows.indexOf(base), free: r.hi >= 0 && k !== rows.indexOf(base) }));
  }

  function userData(c, model) {
    const key = `${c.id}|${model}`;
    if (!cache.has(key)) {
      const pts = userSpace(c, model, pop.platt);
      const base = pts.find((p) => p.l1 === 0);
      for (const p of pts) p.kept = (p.visits / base.visits) * 100;
      const groups = groupByList(pts);
      for (const g of groups) g.kept = (g.visits / base.visits) * 100;
      cache.set(key, { pts, base, groups, probs: visitProbs(c, model, pop.platt), groupOf: new Map(groups.flatMap((g) => g.members.map((p) => [p.k, g]))) });
    }
    return cache.get(key);
  }

  function update(next) {
    ctx = next;
    render();
  }

  function ramp() {
    const css = getComputedStyle(document.documentElement);
    const v = (n) => css.getPropertyValue(n).trim();
    const panel = v("--panel"), div = v("--diversity"), ink = v("--ink");
    const lo = d3.interpolateLab(panel, div)(0.28), hi = d3.interpolateLab(div, ink)(0.4);
    return (t) => (t < 0.5 ? d3.interpolateLab(lo, div)(t * 2) : d3.interpolateLab(div, hi)((t - 0.5) * 2));
  }

  function render() {
    if (!ctx) return;
    const { c, model, theta } = ctx;
    const Q = QUANT[q];
    const u = userData(c, model);
    // This user's current list (exact, including a neighbourhood boost)
    const top = topK(c, theta, model), o = outcomes(c, top);
    const visits = predictedVisits(u.probs, top);
    const cur = { km: o.km, popularity: o.popularity, cuisines: o.cuisines, visits, kept: (visits / u.base.visits) * 100, theta };
    const steered = theta.w_h !== 1 || theta.alpha || theta.beta || theta.lam || theta.boost;
    const gridTheta = { w_h: theta.w_h, alpha: theta.alpha, beta: theta.beta, lam: theta.lam };
    // nearest grid setting in control space; when steering is lighter than any grid step, the nearest steered one
    const dist = (s) => sliderDistance(s, gridTheta) + l1(s) * 1e-9;
    const larger = steered && l1(SETTINGS[d3.minIndex(SETTINGS, dist)]) === 0;
    const nearK = d3.minIndex(SETTINGS, (s) => (larger && l1(s) === 0 ? Infinity : dist(s)));
    const exact = sliderDistance(SETTINGS[nearK], gridTheta) < 1e-9 && !theta.boost;
    const P = popRows[model], other = popRows[ctx.versus];
    const fU = frontier(u.groups, q, Q.low, "visits");
    const fP = frontier(P, Q.pop, Q.low, "ndcg"), fO = frontier(other, Q.pop, Q.low, "ndcg");
    const onU = new Set(fU.map((g) => g.k)), onP = new Set(fP.map((r) => r.k));

    // shade = cuisine groups, one domain for both panels
    const cuis = [...u.groups.map((g) => g.cuisines), ...P.map((r) => r.cuisines), cur.cuisines];
    const cd = [Math.floor(d3.min(cuis)), Math.ceil(d3.max(cuis))];
    const shade = ramp(), color = (v) => shade((v - cd[0]) / Math.max(1e-9, cd[1] - cd[0]));
    const maxCount = d3.max(u.groups, (g) => g.members.length);
    $("to-key").innerHTML = `<span class="ramp">Cuisine groups in the top 10 <b class="mono">${cd[0]}</b><i style="background:linear-gradient(90deg, ${d3.range(0, 1.01, 0.25).map(shade).join(", ")})"></i><b class="mono">${cd[1]}</b></span>
      <span class="ramp"><svg width="44" height="18" aria-hidden="true"><path d="M2 15 H14 V9 H28 V3 H42" class="to-front"/></svg>Frontier: most relevance at each level</span>
      <span class="ramp"><svg width="30" height="18" aria-hidden="true"><path d="M2 12 H28" class="to-front other"/></svg>${MODEL_NAMES[ctx.versus]} frontier</span>
      <span class="ramp"><svg width="18" height="18" aria-hidden="true"><circle cx="9" cy="9" r="6" class="to-ring me"/></svg>Your setting</span>
      <span class="ramp"><svg width="18" height="18" aria-hidden="true"><circle cx="9" cy="9" r="4" style="fill:var(--panel);stroke:var(--ink);stroke-width:1.8"/></svg>Change from the model's list within noise</span>`;
    $("to-user-cap").textContent = `${MODEL_NAMES[model]}. Each dot is a top 10 reached by one or more of the 192 settings (${u.groups.length} distinct lists; larger dots are reached by more settings, up to ${maxCount}). `
      + `Up is the model's own estimate of later visits in the list (Platt-calibrated, step 23), not a measured outcome.`;
    $("to-pop-cap").textContent = `${MODEL_NAMES[model]} on the same 192 settings, mean over ${pop.users.toLocaleString()} test users (RQ1 sweep). `
      + `Up is measured NDCG@10 as a share of the model's own list; whiskers on the frontier are paired 95% bootstrap intervals.`;

    // ---- user panel
    drawPanel("to-user", {
      pts: u.groups, x: (g) => g[q], y: (g) => g.kept, r: d3.scaleSqrt([1, maxCount], [3.5, 8.5]), count: (g) => g.members.length,
      front: fU, on: onU, color: (g) => color(g.cuisines), xExtra: [cur[q]], yExtra: [cur.kept],
      base: u.base, me: steered ? cur : null, meLabel: theta.boost ? "Your setting (with boost)" : "Your setting",
      keyOf: (g) => g.k, members: (g) => g.members.map((p) => p.k), aria: "This user's trade-off between the steered quantity and predicted visits",
      tipHtml: (g) => userTip(g, onU.has(g.k), u.groups, Q),
      click: (g) => handlers.load?.({ w_h: g.w_h, alpha: g.alpha, beta: g.beta, lam: g.lam }),
    });
    // ---- population panel
    drawPanel("to-pop", {
      pts: P, x: (r) => r[Q.pop], y: (r) => r.kept, r: () => 4, count: () => 1, front: fP, on: onP, color: (r) => color(r.cuisines),
      ci: true, other: fO.map((r) => ({ x: r[Q.pop], y: r.kept })), otherName: MODEL_NAMES[ctx.versus],
      xExtra: [], yExtra: other.map((r) => r.kept), base: P.find((r) => r.base),
      me: steered ? P[nearK] : null, meLabel: exact ? "Your setting" : larger ? "Nearest steered grid setting" : "Nearest grid setting", keyOf: (r) => r.k, members: (r) => [r.k],
      aria: "Trade-off over 2,000 test users", tipHtml: (r) => popTip(r, onP.has(r.k), Q),
      click: (r) => handlers.load?.({ w_h: r.w_h, alpha: r.alpha, beta: r.beta, lam: r.lam }),
    });
    headline(Q, u, cur, steered, fU, P[nearK], fP, exact, larger);
    rings();
  }

  function drawPanel(id, d) {
    const node = $(id);
    const width = Math.max(node.clientWidth, 300);
    const xs = [...d.pts.map(d.x), ...d.xExtra], ys = [...d.pts.map(d.y), ...d.yExtra, ...(d.ci ? d.front.map((r) => r.keptLo) : []), 100];
    const x = d3.scaleLinear(d3.extent(xs), [M.l + 10, width - M.r - 10]).nice();
    const y = d3.scaleLinear([Math.min(d3.min(ys), 100), 100], [H - M.b - 8, M.t + 10]).nice();
    const svg = d3.select(node).selectAll("svg").data([0]).join("svg").attr("width", width).attr("height", H)
      .attr("role", "img").attr("aria-label", d.aria);
    svg.selectAll("*").remove();
    const Q = QUANT[q];
    svg.append("g").attr("class", "tk-axis").attr("transform", `translate(0,${H - M.b})`)
      .call(d3.axisBottom(x).ticks(Math.max(3, Math.floor(width / 90))).tickFormat(Q.tick).tickSizeOuter(0));
    svg.append("g").attr("class", "tk-axis").attr("transform", `translate(${M.l},0)`)
      .call(d3.axisLeft(y).ticks(5).tickFormat((v) => `${v}%`).tickSizeOuter(0));
    svg.append("g").attr("class", "to-grid").selectAll("line").data(y.ticks(5)).join("line")
      .attr("x1", M.l).attr("x2", width - M.r).attr("y1", y).attr("y2", y);
    // axis title at the opposite end from the direction hint, so they cannot collide on a phone
    svg.append("text").attr("class", "lens-note").attr("x", Q.low ? width - M.r : M.l).attr("y", H - 6)
      .attr("text-anchor", Q.low ? "end" : "start").text(width < 480 ? Q.short : Q.label);
    svg.append("text").attr("class", "lens-note to-dir").attr("x", Q.low ? M.l : width - M.r).attr("y", H - 6)
      .attr("text-anchor", Q.low ? "start" : "end").text(Q.low ? `← ${Q.better}` : `${Q.better} →`);
    svg.append("text").attr("class", "lens-note").attr("x", M.l).attr("y", 11).text("Relevance kept");
    // other model's frontier (population only), then this model's
    const step = Q.low ? d3.curveStepAfter : d3.curveStepBefore;
    const line = d3.line().x((p) => x(p.x)).y((p) => y(p.y)).curve(step);
    if (d.other) {
      const o = [...d.other].sort((a, b) => a.x - b.x);
      svg.append("path").attr("class", "to-front other").attr("d", line(o));
    }
    const fr = d.front.map((p) => ({ x: d.x(p), y: d.y(p) })).sort((a, b) => a.x - b.x);
    svg.append("path").attr("class", "to-front").attr("d", line(fr));
    if (d.ci) {
      svg.append("g").attr("class", "to-ci").selectAll("line").data(d.front).join("line")
        .attr("x1", (r) => x(d.x(r))).attr("x2", (r) => x(d.x(r))).attr("y1", (r) => y(r.keptLo)).attr("y2", (r) => y(Math.min(r.keptHi, y.domain()[1])));
    }
    // points: dominated first, frontier on top
    const pts = [...d.pts].sort((a, b) => d.on.has(d.keyOf(a)) - d.on.has(d.keyOf(b)) || d.count(b) - d.count(a));
    const pos = new Map();
    svg.append("g").selectAll("circle").data(pts).join("circle")
      .attr("class", (p) => `to-pt${d.on.has(d.keyOf(p)) ? " on" : ""}${p.free ? " free" : ""}`)
      .attr("cx", (p) => x(d.x(p))).attr("cy", (p) => y(d.y(p))).attr("r", (p) => d.r(d.count(p)))
      .style("fill", (p) => (p.free ? "var(--panel)" : d.color(p)))
      .style("stroke", (p) => (p.free ? "var(--ink)" : null))
      .each((p) => { for (const k of d.members(p)) pos.set(k, { x: x(d.x(p)), y: y(d.y(p)), r: d.r(d.count(p)) }); });
    // model's list and the current setting
    const bx = x(d.x(d.base)), by = y(d.y(d.base));
    const label = (px, py, text, cls) => {
      const right = px < width - 150;
      svg.append("text").attr("class", `lens-me halo ${cls ?? ""}`).attr("x", px + (right ? 10 : -10)).attr("y", py - 9)
        .attr("text-anchor", right ? "start" : "end").text(text);
    };
    label(bx, by, "Model's list");
    if (d.me) {
      const mx = x(d.me[q] ?? d.x(d.me)), my = y(d.me.kept);
      if (Math.hypot(mx - bx, my - by) > 14) {
        svg.append("path").attr("class", "to-move").attr("d", `M${bx},${by} L${mx},${my}`).attr("marker-end", "url(#to-arrow-" + id + ")");
        svg.append("defs").append("marker").attr("id", "to-arrow-" + id).attr("viewBox", "0 0 8 8").attr("refX", 13).attr("refY", 4)
          .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
          .append("path").attr("d", "M0,0 L8,4 L0,8 z").attr("class", "to-arrow");
      }
      svg.append("circle").attr("class", "to-ring me").attr("cx", mx).attr("cy", my).attr("r", 8.5);
      label(mx, my + 26, d.meLabel);
    }
    // hit targets: Voronoi over point centres
    const delaunay = d3.Delaunay.from(pts, (p) => x(d.x(p)), (p) => y(d.y(p)));
    svg.append("rect").attr("class", "hit").attr("x", M.l).attr("y", M.t).attr("width", width - M.l - M.r).attr("height", H - M.t - M.b)
      .on("pointermove", (e) => {
        const [px, py] = d3.pointer(e);
        const p = pts[delaunay.find(px, py)];
        if (Math.hypot(x(d.x(p)) - px, y(d.y(p)) - py) > 24) { setHover(null); hideTip(); return; }
        setHover(d.members(p));
        showTip(e, d.tipHtml(p));
      })
      .on("pointerleave", () => { setHover(null); hideTip(); })
      .on("click", (e) => {
        const [px, py] = d3.pointer(e);
        const p = pts[delaunay.find(px, py)];
        if (Math.hypot(x(d.x(p)) - px, y(d.y(p)) - py) <= 24) { hideTip(); d.click(p); }
      });
    layers[id] = { g: svg.append("g").attr("class", "rings"), pos };
  }

  function userTip(g, on, groups, Q) {
    const n = g.members.length;
    let verdict = on ? `On the frontier: no other list is ${Q.asGood} and keeps more.` : "";
    if (!on) {
      const better = groups.filter((h) => (Q.low ? h[q] <= g[q] : h[q] >= g[q]) && h.visits > g.visits + 1e-12);
      const best = d3.greatest(better, (h) => h.visits);
      verdict = `Dominated: another list is ${Q.asGood} and keeps ${pct(best.kept)}.`;
    }
    return `<b>${n === 1 ? "1 setting gives" : `${n} settings give`} this list</b><br>
      <span class="mono">${thetaText(g)}</span>${n > 1 ? " (smallest change)" : ""}<br>
      ${QUANT.km.fmt(g.km)} · ${QUANT.popularity.fmt(g.popularity)} popularity · ${g.cuisines} cuisine groups<br>
      Predicted visits <span class="mono">${g.visits.toFixed(3)}</span> (${pct(g.kept)} of the model's list)<br>
      <span class="muted">${verdict} Click to load.</span>`;
  }

  function popTip(r, on, Q) {
    const sign = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4);
    return `<b class="mono">${thetaText(r)}</b><br>
      NDCG@10 <span class="mono">${r.ndcg.toFixed(4)}</span> (${pct(r.kept)} of the model's list)<br>
      ${r.base ? "" : `Change <span class="mono">${sign(r.d_ndcg)} [${sign(r.lo)}, ${sign(r.hi)}]</span>${r.free ? ", not distinguishable from zero" : ""}<br>`}
      ${QUANT.km.fmt(r.distance_km)} · ${QUANT.popularity.fmt(r.popularity)} popularity · ${r.cuisines.toFixed(1)} cuisine groups<br>
      <span class="muted">${on ? "On the frontier. " : ""}Click to load this setting for this user.</span>`;
  }

  function headline(Q, u, cur, steered, fU, near, fP, exact, larger) {
    // frontiers run from the most-steered point to the model's own list
    const end = fU[0], endP = fP[0];
    let s;
    if (!steered) {
      s = `This is the model's own list (${Q.noun}: ${Q.fmt(u.base[q])}). For this user the ${Q.most} list on the grid (${say(Q, end[q])}) keeps `
        + `<b>${pct(end.kept)}</b> of the visits the model predicts; across ${pop.users.toLocaleString()} users the ${Q.most} setting (${say(Q, endP[QUANT[q].pop])}) keeps `
        + `<b>${pct(endP.kept)}</b> of NDCG@10. Click any dot to try it.`;
    } else {
      s = `Your setting moves the ${Q.noun} from ${say(Q, u.base[q])} to ${say(Q, cur[q])} and keeps <b>${pct(cur.kept)}</b> of the visits the model predicts for this user.`;
      const better = u.groups.filter((h) => (Q.low ? h[q] <= cur[q] + 1e-9 : h[q] >= cur[q] - 1e-9) && h.visits > cur.visits + 1e-9);
      const best = d3.greatest(better, (h) => h.visits);
      s += best ? ` A grid setting is ${Q.asGood} and keeps ${pct(best.kept)} (<span class="mono">${thetaText(best)}</span>).`
        : ` No grid setting is ${Q.asGood} and keeps more.`;
      s += ` Across ${pop.users.toLocaleString()} users ${exact ? "the same setting" : `the nearest ${larger ? "steered " : ""}grid setting (<span class="mono">${thetaText(near)}</span>${larger ? ", a larger step than yours" : ""})`} keeps <b>${pct(near.kept)}</b> of NDCG@10`
        + (near.base ? "." : ` (95% CI ${pct(near.keptLo)}–${pct(near.keptHi)}).`);
    }
    $("to-headline").innerHTML = s;
  }

  function setHover(ks) {
    const key = ks ? ks.join(",") : null;
    if (key === hoverK) return;
    hoverK = key;
    rings();
  }

  function rings() {
    const ks = hoverK ? hoverK.split(",").map(Number) : [];
    for (const { g, pos } of Object.values(layers)) {
      const seen = new Set();
      const marks = ks.map((k) => pos.get(k)).filter((p) => p && !seen.has(`${p.x},${p.y}`) && seen.add(`${p.x},${p.y}`));
      g.selectAll("circle").data(marks).join("circle").attr("class", "to-ring hov")
        .attr("cx", (p) => p.x).attr("cy", (p) => p.y).attr("r", (p) => p.r + 3.5);
    }
  }

  return { update, render };
}
